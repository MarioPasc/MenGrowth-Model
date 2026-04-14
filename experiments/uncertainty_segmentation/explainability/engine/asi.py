"""Attention Selectivity Index (ASI) from WindowAttention weights.

For each window-attention block, the post-softmax attention tensor has
shape ``[n_windows * B, num_heads, N, N]`` with ``N = wd * wh * ww``
tokens per window.  ASI measures, *per head and per window*, the ratio
of attention that tumor-token queries direct to tumor-token keys vs to
non-tumor-token keys::

    ASI_h = mean_q∈T mean_k∈T attn_h[q, k] / mean_q∈T mean_k∈B attn_h[q, k]

A value of ``1.0`` indicates uniform attention (attention is
independent of the tumor mask).  Values ``> 1`` indicate that the head
preferentially routes attention from tumor queries to tumor keys.

Token-grid alignment
--------------------
``WindowAttention.forward`` receives a flattened ``[n_windows*B, N, C]``
tensor and the spatial layout is recoverable only from
``n_windows`` and ``self.window_size``.  The MONAI implementation pads
the feature map up to a multiple of ``window_size`` with
``np.ceil(D / ws) * ws`` (see ``BasicLayer.forward`` and
``SwinTransformerBlock.forward_part1`` in MONAI 1.5's
``swin_unetr.py``).  This module reproduces that padding and the
``torch.roll`` for shifted blocks so that a downsampled ground-truth
mask can be partitioned into the same windows the attention used.

Memory note
-----------
Stage 1 has ``~2700`` windows per scan at 192³.  The per-window ASI
returns ``[H]`` floats which are immediately summarised, so peak
memory is dominated by the captured attention tensor itself
(handled streaming-style by the ``hooks.AttentionCapture`` callback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Default minimum tokens of each class for a window to be "boundary".
DEFAULT_MIN_TUMOR = 5
DEFAULT_MIN_NONTUMOR = 5

# Numerical floor for the denominator of the ratio.
ASI_EPS = 1e-12


# -----------------------------------------------------------------------------
# Mask downsampling and window partitioning
# -----------------------------------------------------------------------------


def downsample_mask_to_stage(
    mask: torch.Tensor,
    stage: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Downsample a tumor mask to the spatial resolution of stage ``s`` attention.

    SwinUNETR with ``patch_size=2`` reduces the input by a factor of 2
    at the patch embedding and a further factor of 2 per ``PatchMerging``.
    The ``WindowAttention`` modules inside ``layers{s}`` therefore
    operate on a feature map of size ``D / 2**s``.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask of shape ``[D, H, W]`` (float).
    stage : int
        SwinViT stage index in ``{1, 2, 3, 4}``.
    threshold : float
        Pooling threshold; a downsampled voxel is tumor if more than
        ``threshold`` of the source block is tumor.

    Returns
    -------
    torch.Tensor
        Binary mask of shape ``[D/2**s, H/2**s, W/2**s]`` in float.
    """
    if stage not in (1, 2, 3, 4):
        raise ValueError(f"stage must be in {{1,2,3,4}}, got {stage}")
    if mask.dim() != 3:
        raise ValueError(f"mask must be [D,H,W], got shape {tuple(mask.shape)}")
    factor = 2 ** stage
    pooled = F.avg_pool3d(
        mask.unsqueeze(0).unsqueeze(0).float(),
        kernel_size=factor,
        stride=factor,
    ).squeeze(0).squeeze(0)
    return (pooled > threshold).float()


def _pad_to_window_multiple(
    mask: torch.Tensor, window_size: tuple[int, int, int]
) -> torch.Tensor:
    """Pad ``mask`` (D, H, W) up to the next multiple of ``window_size`` per dim.

    Mirrors the ``F.pad`` call inside MONAI's
    ``SwinTransformerBlock.forward_part1`` (zero-pads the trailing edge
    of each spatial dimension).  Padded voxels are treated as
    background (``0``).
    """
    d, h, w = mask.shape
    pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
    pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
    pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
    if pad_d1 == 0 and pad_b == 0 and pad_r == 0:
        return mask
    return F.pad(mask, (0, pad_r, 0, pad_b, 0, pad_d1), value=0.0)


def partition_mask_windows(
    mask: torch.Tensor,
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> torch.Tensor:
    """Partition a stage-``s`` mask into the same windows the attention used.

    Replicates the MONAI partitioning sequence:
    ``pad → roll(-shift) → window_partition``.  Window partition
    reshape is identical to MONAI's ``window_partition`` with
    ``len(x_shape) == 5`` (with ``C = 1``).

    Parameters
    ----------
    mask : torch.Tensor
        Float mask of shape ``[D, H, W]`` at the stage's feature resolution.
    window_size : tuple[int, int, int]
        Effective (clamped) window size used by ``WindowAttention``.
    shift_size : tuple[int, int, int]
        Per-dim shift (``(0,0,0)`` for the non-shifted block).

    Returns
    -------
    torch.Tensor
        ``[n_windows, N]`` float mask (``N = prod(window_size)``).
    """
    if mask.dim() != 3:
        raise ValueError(f"mask must be [D,H,W], got {tuple(mask.shape)}")
    padded = _pad_to_window_multiple(mask, window_size)
    if any(s > 0 for s in shift_size):
        padded = torch.roll(
            padded,
            shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
            dims=(0, 1, 2),
        )
    dp, hp, wp = padded.shape
    wd, wh, ww = window_size
    if dp % wd or hp % wh or wp % ww:
        raise ValueError(
            "Padded mask shape not divisible by window_size: "
            f"{(dp, hp, wp)} vs {window_size}"
        )
    # Match MONAI's window_partition: [n_windows, prod(ws)].
    x = padded.view(dp // wd, wd, hp // wh, wh, wp // ww, ww)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(-1, wd * wh * ww)


# -----------------------------------------------------------------------------
# Boundary window selection and per-window ASI
# -----------------------------------------------------------------------------


def select_boundary_windows(
    mask_windows: torch.Tensor,
    min_tumor: int = DEFAULT_MIN_TUMOR,
    min_nontumor: int = DEFAULT_MIN_NONTUMOR,
) -> torch.Tensor:
    """Indices of windows containing both tumor and non-tumor tokens.

    A boundary window must satisfy ``n_tumor >= min_tumor`` and
    ``n_nontumor >= min_nontumor``.  Pure-tumor and pure-background
    windows are excluded because the ratio is degenerate.

    Parameters
    ----------
    mask_windows : torch.Tensor
        ``[n_windows, N]`` binary mask windows.
    min_tumor, min_nontumor : int
        Minimum token counts.

    Returns
    -------
    torch.Tensor
        1-D ``LongTensor`` of selected window indices.
    """
    n_tumor = mask_windows.sum(dim=1)
    n_nontumor = mask_windows.shape[1] - n_tumor
    valid = (n_tumor >= min_tumor) & (n_nontumor >= min_nontumor)
    return torch.nonzero(valid, as_tuple=False).squeeze(-1)


def compute_asi_per_window(
    attn_w: torch.Tensor,
    mask_w: torch.Tensor,
    eps: float = ASI_EPS,
) -> torch.Tensor:
    """Per-head ASI for a single window.

    Parameters
    ----------
    attn_w : torch.Tensor
        Post-softmax attention for one window, shape ``[H, N, N]``.
        Row ``i`` is the attention distribution of query token ``i``;
        ``attn_w[h, i, :].sum() == 1`` per row.
    mask_w : torch.Tensor
        Binary token mask for the same window, shape ``[N]`` (float).
    eps : float
        Floor for the denominator.

    Returns
    -------
    torch.Tensor
        ``[H]`` ASI per head.  Returns ``NaN`` per head if either
        tumor or non-tumor token set is empty.
    """
    if attn_w.dim() != 3:
        raise ValueError(f"attn_w must be [H,N,N], got {tuple(attn_w.shape)}")
    if mask_w.dim() != 1 or mask_w.shape[0] != attn_w.shape[-1]:
        raise ValueError(
            f"mask_w shape {tuple(mask_w.shape)} incompatible with attn_w "
            f"{tuple(attn_w.shape)}"
        )
    tumor_idx = torch.nonzero(mask_w > 0.5, as_tuple=False).squeeze(-1)
    bg_idx = torch.nonzero(mask_w <= 0.5, as_tuple=False).squeeze(-1)
    n_t, n_b = tumor_idx.numel(), bg_idx.numel()
    h = attn_w.shape[0]
    if n_t == 0 or n_b == 0:
        return torch.full((h,), float("nan"), device=attn_w.device, dtype=attn_w.dtype)
    # Tumor-query rows: [H, n_t, N]
    rows = attn_w.index_select(dim=1, index=tumor_idx)
    # Mean over tumor-key columns: [H, n_t]
    mu_tt = rows.index_select(dim=2, index=tumor_idx).mean(dim=2)
    # Mean over background-key columns: [H, n_t]
    mu_tb = rows.index_select(dim=2, index=bg_idx).mean(dim=2)
    # Average over the n_t tumor queries → per-head scalar.
    return mu_tt.mean(dim=1) / mu_tb.mean(dim=1).clamp(min=eps)


# -----------------------------------------------------------------------------
# Per-scan accumulator: pre-computes mask windows for every (stage, block)
# -----------------------------------------------------------------------------


@dataclass
class ASIScanResult:
    """ASI summary for a single scan.

    Attributes
    ----------
    per_block_per_head : dict[str, np.ndarray]
        Mapping ``"stage_{s}_block_{b}" -> [n_boundary_windows, num_heads]``
        ASI values.  Empty arrays when no boundary windows exist.
    n_windows_total : dict[str, int]
        Total windows discovered per (stage, block).
    n_windows_boundary : dict[str, int]
        Subset that passed the ``min_tumor / min_nontumor`` filter.
    """

    per_block_per_head: dict[str, np.ndarray] = field(default_factory=dict)
    n_windows_total: dict[str, int] = field(default_factory=dict)
    n_windows_boundary: dict[str, int] = field(default_factory=dict)


class ASIScanAccumulator:
    """Streaming ASI computation that plugs into :class:`AttentionCapture`.

    Usage
    -----
    >>> acc = ASIScanAccumulator(gt_mask=gt_mask, target_stages={1, 2, 3, 4})
    >>> with AttentionCapture(model, mode="callback", process_fn=acc) as cap:
    ...     _ = model(image)
    >>> result = acc.result()

    The accumulator pre-computes the downsampled mask once per stage
    and partitions it into windows on the first call to each block;
    subsequent scans reuse the same accumulator instance only if you
    call ``reset(new_mask)``.
    """

    def __init__(
        self,
        gt_mask: torch.Tensor,
        target_stages: set[int] | None = None,
        min_tumor: int = DEFAULT_MIN_TUMOR,
        min_nontumor: int = DEFAULT_MIN_NONTUMOR,
    ) -> None:
        if gt_mask.dim() != 3:
            raise ValueError(
                f"gt_mask must be [D,H,W], got {tuple(gt_mask.shape)}"
            )
        self._gt_mask = gt_mask.float().cpu()
        self._target_stages = target_stages or {1, 2, 3, 4}
        self._min_tumor = int(min_tumor)
        self._min_nontumor = int(min_nontumor)
        # Cached downsampled masks per stage.
        self._stage_mask: dict[int, torch.Tensor] = {}
        # Per (stage, block) accumulators of [n_boundary_window, H] arrays.
        self._buf: dict[str, list[torch.Tensor]] = {}
        self._n_total: dict[str, int] = {}
        self._n_boundary: dict[str, int] = {}

    def reset(self, gt_mask: torch.Tensor) -> None:
        """Reuse the accumulator for a new scan."""
        self.__init__(
            gt_mask,
            target_stages=self._target_stages,
            min_tumor=self._min_tumor,
            min_nontumor=self._min_nontumor,
        )

    # ------------------------------------------------------------------
    # Hook-callback interface
    # ------------------------------------------------------------------
    def __call__(
        self,
        key: str,
        attn: torch.Tensor,
        mask: torch.Tensor | None,  # noqa: ARG002 - unused but part of signature
    ) -> None:
        """``process_fn`` for :class:`AttentionCapture` (callback mode).

        ``key`` is ``"stage_{s}_block_{b}"``.  ``attn`` has shape
        ``[B*n_windows, H, N, N]``.  ``mask`` (the shifted-window
        attention mask) is ignored — its purpose is to zero out
        cross-region attention in the model itself, which is already
        baked into ``attn``.
        """
        parts = key.split("_")
        stage = int(parts[1])
        block = int(parts[3])
        if stage not in self._target_stages:
            return

        n_windows_b, num_heads, n_tokens, n_tokens2 = attn.shape
        if n_tokens != n_tokens2:
            raise RuntimeError(
                f"Non-square attention at {key}: shape {tuple(attn.shape)}"
            )

        mask_windows = self._mask_windows_for(stage, block, n_windows_b, n_tokens)
        # mask_windows is on CPU; move to attn device for indexing.
        mask_windows = mask_windows.to(attn.device)

        n_w = mask_windows.shape[0]
        if n_windows_b % n_w != 0:
            raise RuntimeError(
                f"{key}: attn windows {n_windows_b} not divisible by mask "
                f"windows {n_w}"
            )
        bsz = n_windows_b // n_w

        boundary_idx = select_boundary_windows(
            mask_windows,
            min_tumor=self._min_tumor,
            min_nontumor=self._min_nontumor,
        )

        self._n_total[key] = self._n_total.get(key, 0) + n_w * bsz
        if boundary_idx.numel() == 0:
            self._n_boundary[key] = self._n_boundary.get(key, 0)
            return

        per_w_per_head: list[torch.Tensor] = []
        for b_idx in range(bsz):
            for w_idx in boundary_idx.tolist():
                attn_w = attn[b_idx * n_w + w_idx]  # [H, N, N]
                mask_w = mask_windows[w_idx]
                asi = compute_asi_per_window(attn_w, mask_w)
                per_w_per_head.append(asi)
        if per_w_per_head:
            stacked = torch.stack(per_w_per_head, dim=0).detach().cpu()
            self._buf.setdefault(key, []).append(stacked)
            self._n_boundary[key] = self._n_boundary.get(key, 0) + len(per_w_per_head)

    # ------------------------------------------------------------------
    def _mask_windows_for(
        self, stage: int, block: int, n_windows_b: int, n_tokens: int
    ) -> torch.Tensor:
        """Token-grid mask windows for ``(stage, block)``; cached lazily.

        Recovers the effective window size from ``n_tokens`` (assumes
        cubic windows) and the spatial layout from the assumption that
        the feature map is cubic at this stage (true for BraTS inputs).
        """
        if stage not in self._stage_mask:
            self._stage_mask[stage] = downsample_mask_to_stage(self._gt_mask, stage)
        ds_mask = self._stage_mask[stage]

        # Recover window_size assuming cubic windows.
        wd = round(n_tokens ** (1.0 / 3.0))
        if wd ** 3 != n_tokens:
            raise NotImplementedError(
                f"Non-cubic window not supported (n_tokens={n_tokens})."
            )
        window_size = (wd, wd, wd)

        # MONAI clamps the declared window_size *down* to the spatial
        # extent at runtime.  At small spatial sizes (e.g. 4³ feature
        # map) the actual window may be 4 even if declared was 7.
        # Either way, the partition logic below uses ``window_size``
        # consistently.
        d, h, w = ds_mask.shape
        # The non-shifted block (block 0) has shift=0; the shifted
        # block (block 1) has shift_size = ws // 2 per dim, but only
        # when the spatial dim is strictly larger than the window.
        # MONAI's ``get_window_size`` zeros the shift in the clamped
        # direction; we replicate that behaviour.
        if block == 0:
            shift_size = (0, 0, 0)
        else:
            shift_size = tuple(
                wd // 2 if dim_sz > wd else 0
                for dim_sz in (d, h, w)
            )

        return partition_mask_windows(ds_mask, window_size, shift_size)

    # ------------------------------------------------------------------
    def result(self) -> ASIScanResult:
        """Concatenate buffered per-window ASI arrays and return summary."""
        out: dict[str, np.ndarray] = {}
        for key, chunks in self._buf.items():
            cat = torch.cat(chunks, dim=0).numpy()
            out[key] = cat
        # Also report stages/blocks with no boundary windows.
        for key in self._n_total:
            out.setdefault(key, np.zeros((0, 0), dtype=np.float32))
        return ASIScanResult(
            per_block_per_head=out,
            n_windows_total=dict(self._n_total),
            n_windows_boundary=dict(self._n_boundary),
        )


# -----------------------------------------------------------------------------
# Aggregation across scans
# -----------------------------------------------------------------------------


def aggregate_asi_across_scans(
    per_scan_results: list[ASIScanResult],
) -> dict[str, np.ndarray]:
    """Concatenate per-scan ASI arrays into one ``[N_total, num_heads]`` array per (stage, block).

    Parameters
    ----------
    per_scan_results : list[ASIScanResult]

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``"stage_{s}_block_{b}"`` → concatenated boundary-window
        ASI values across all scans.
    """
    by_key: Mapping[str, list[np.ndarray]] = {}
    for r in per_scan_results:
        for k, arr in r.per_block_per_head.items():
            if arr.size == 0:
                continue
            by_key.setdefault(k, []).append(arr)  # type: ignore[union-attr]
    return {k: np.concatenate(v, axis=0) for k, v in by_key.items()}
