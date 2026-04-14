"""Brain-masked Tumor Selectivity Index (TSI).

For each encoder stage and each channel:

    TSI_c = mean(|h_c| inside tumor) / mean(|h_c| inside brain - tumor)

The denominator is restricted to **brain** voxels (excluding background air
and skull-stripped zeros). This corrects the bias of the original
``tsi_analysis.py`` formulation, where the denominator included background
voxels and therefore inflated the selectivity of shallow stages whose
activations are mostly anatomy-driven.

When ``brain_mask_ds`` is ``None`` the function reduces to the legacy
formula ``mean_T / mean_(1-T)`` so existing tests continue to pass.

References
----------
- Bau et al. (CVPR 2017). Network Dissection.
- Spec: ``docs/growth-related/lora/EXPLAINABILITY_ANALYSIS_SPEC.md`` §3.1.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

if TYPE_CHECKING:
    from growth.models.segmentation.original_decoder import LoRAOriginalDecoderModel  # noqa: F401

logger = logging.getLogger(__name__)


# Stage metadata for feature_size=48 (BSF-Tiny).
# Stage 0 is the patch embedding (no WindowAttention); LoRA targets are
# resolved from the run config via ``build_stage_meta``.
_STAGE_SHAPE = {
    0: {"channels": 48, "downsample": 2, "name": "patch_embed"},
    1: {"channels": 96, "downsample": 4, "name": "layers1"},
    2: {"channels": 192, "downsample": 8, "name": "layers2"},
    3: {"channels": 384, "downsample": 16, "name": "layers3"},
    4: {"channels": 768, "downsample": 32, "name": "layers4"},
}


def build_stage_meta(target_stages: list[int] | tuple[int, ...]) -> dict[int, dict]:
    """Return per-stage metadata with ``has_lora`` resolved from the run config.

    Parameters
    ----------
    target_stages : list[int] | tuple[int, ...]
        LoRA target stages from ``config.lora.target_stages``. Must be a
        subset of {1, 2, 3, 4}.

    Returns
    -------
    dict[int, dict]
        Mapping from stage index (0-4) to a dict containing
        ``channels``, ``downsample``, ``name`` and ``has_lora``.

    Raises
    ------
    ValueError
        If ``target_stages`` contains values outside {1, 2, 3, 4}.
    """
    target_set = {int(s) for s in target_stages}
    invalid = target_set - {1, 2, 3, 4}
    if invalid:
        raise ValueError(
            f"target_stages must be a subset of {{1,2,3,4}}, got: {sorted(invalid)}"
        )
    return {s: {**_STAGE_SHAPE[s], "has_lora": s in target_set} for s in range(5)}


# Backward-compatibility alias used by legacy tests.
STAGE_META = build_stage_meta([3, 4])


@dataclasses.dataclass
class TSIResult:
    """TSI for one stage of one scan.

    Attributes
    ----------
    stage : int
        Stage index (0-4).
    n_channels : int
        Number of feature channels at this stage.
    resolution : tuple[int, int, int]
        Spatial dimensions of the hidden state ``(D_s, H_s, W_s)``.
    tsi_per_channel : np.ndarray
        Per-channel TSI values, shape ``[C_s]``.
    mean_tsi, std_tsi : float
        Summary statistics across channels.
    frac_above : dict[float, float]
        Fraction of channels with TSI above each threshold.
    wilcoxon_p : float
        One-sided Wilcoxon signed-rank ``p`` for ``H_0: median(TSI)=1``.
    top_k_channels : list[int]
        Indices of the most tumor-selective channels (descending).
    mean_activation_map : np.ndarray | None
        Optional channel-mean ``|h|`` map at stage resolution.
    top_channels_map : np.ndarray | None
        Optional mean ``|h|`` map over the top-K channels.
    """

    stage: int
    n_channels: int
    resolution: tuple[int, int, int]
    tsi_per_channel: np.ndarray
    mean_tsi: float
    std_tsi: float
    frac_above: dict[float, float]
    wilcoxon_p: float
    top_k_channels: list[int]
    mean_activation_map: np.ndarray | None = None
    top_channels_map: np.ndarray | None = None


@dataclasses.dataclass
class ScanTSIResult:
    """TSI for all 5 stages of one scan under one condition."""

    scan_id: str
    condition: str
    stages: list[TSIResult]


def downsample_mask(
    mask: torch.Tensor,
    target_size: tuple[int, int, int],
) -> torch.Tensor:
    """Nearest-neighbour downsample of a binary mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask ``[D, H, W]``.
    target_size : tuple[int, int, int]
        Target spatial dimensions.

    Returns
    -------
    torch.Tensor
        Downsampled binary mask ``[D_s, H_s, W_s]``.
    """
    mask_5d = mask.float().unsqueeze(0).unsqueeze(0)
    ds = F.interpolate(mask_5d, size=target_size, mode="nearest")
    return ds.squeeze(0).squeeze(0)


def compute_tsi_single_stage(
    hidden_state: torch.Tensor,
    mask_ds: torch.Tensor,
    thresholds: list[float],
    top_k: int = 3,
    return_maps: bool = False,
    epsilon: float = 1e-8,
    brain_mask_ds: torch.Tensor | None = None,
) -> TSIResult:
    """Compute per-channel TSI at one encoder stage.

    Parameters
    ----------
    hidden_state : torch.Tensor
        Activation tensor ``[1, C_s, D_s, H_s, W_s]`` on CPU.
    mask_ds : torch.Tensor
        Downsampled binary tumor (WT) mask ``[D_s, H_s, W_s]``.
    thresholds : list[float]
        TSI thresholds for ``Frac(TSI > tau)`` reporting.
    top_k : int
        Number of top tumor-selective channels to record.
    return_maps : bool
        If True, also store a channel-mean ``|h|`` map and a top-K mean map.
    epsilon : float
        Numerical stability constant.
    brain_mask_ds : torch.Tensor | None
        Optional binary brain mask ``[D_s, H_s, W_s]``. When provided, the
        denominator is restricted to ``brain ∩ ¬tumor`` (corrected TSI).
        When ``None`` the legacy ``mean_T / mean_(1-T)`` formula is used so
        existing test fixtures and saved CSVs remain comparable.

    Returns
    -------
    TSIResult
        Result with ``stage = -1`` (the caller fills this in).
    """
    h = hidden_state.squeeze(0).float()
    C_s = h.shape[0]
    spatial = tuple(h.shape[1:])
    abs_h = h.abs()

    if brain_mask_ds is None:
        denom_mask = 1.0 - mask_ds
    else:
        # Brain-masked denominator: restrict non-tumor to brain voxels only.
        denom_mask = brain_mask_ds * (1.0 - mask_ds)

    tumor_count = mask_ds.sum() + epsilon
    denom_count = denom_mask.sum() + epsilon

    if mask_ds.sum() < 1 or denom_mask.sum() < 1:
        logger.warning(
            "Stage spatial=%s: tumor=%.0f, denom=%.0f -> TSI undefined",
            spatial, mask_ds.sum().item(), denom_mask.sum().item(),
        )
        tsi_arr = np.full(C_s, np.nan)
        return TSIResult(
            stage=-1,
            n_channels=C_s,
            resolution=spatial,
            tsi_per_channel=tsi_arr,
            mean_tsi=float("nan"),
            std_tsi=float("nan"),
            frac_above={t: float("nan") for t in thresholds},
            wilcoxon_p=float("nan"),
            top_k_channels=[],
            mean_activation_map=None,
            top_channels_map=None,
        )

    mu_tumor = (abs_h * mask_ds).sum(dim=(1, 2, 3)) / tumor_count
    mu_denom = (abs_h * denom_mask).sum(dim=(1, 2, 3)) / denom_count
    tsi = mu_tumor / (mu_denom + epsilon)
    tsi_np = tsi.numpy()

    mean_tsi = float(np.nanmean(tsi_np))
    std_tsi = float(np.nanstd(tsi_np))
    frac_above = {tau: float(np.nanmean(tsi_np > tau)) for tau in thresholds}

    try:
        wilcoxon_p = float(
            scipy.stats.wilcoxon(
                tsi_np - 1.0, alternative="greater", nan_policy="omit"
            ).pvalue
        )
    except ValueError:
        # Triggered when all (tsi - 1) are zero / identical.
        wilcoxon_p = 1.0

    valid_mask = ~np.isnan(tsi_np)
    if valid_mask.sum() >= top_k:
        top_indices = np.argsort(tsi_np)[-top_k:][::-1].tolist()
    else:
        top_indices = np.where(valid_mask)[0].tolist()

    mean_act_map = None
    top_ch_map = None
    if return_maps:
        mean_act_map = abs_h.mean(dim=0).numpy()
        if top_indices:
            top_ch_map = abs_h[top_indices].mean(dim=0).numpy()

    return TSIResult(
        stage=-1,
        n_channels=C_s,
        resolution=spatial,
        tsi_per_channel=tsi_np,
        mean_tsi=mean_tsi,
        std_tsi=std_tsi,
        frac_above=frac_above,
        wilcoxon_p=wilcoxon_p,
        top_k_channels=top_indices,
        mean_activation_map=mean_act_map,
        top_channels_map=top_ch_map,
    )


def extract_hidden_states(
    model: nn.Module,
    images: torch.Tensor,
) -> list[torch.Tensor]:
    """Return the 5 SwinViT hidden states for both frozen and adapted models.

    Parameters
    ----------
    model : nn.Module
        Frozen ``SwinUNETR`` or LoRA-adapted ``LoRAOriginalDecoderModel``.
    images : torch.Tensor
        Input ``[1, 4, D, H, W]`` on the model's device.

    Returns
    -------
    list[torch.Tensor]
        Five CPU float32 tensors (one per stage), in the order returned by
        MONAI's ``SwinTransformer.forward``.
    """
    from growth.models.segmentation.original_decoder import LoRAOriginalDecoderModel

    with torch.no_grad():
        if isinstance(model, LoRAOriginalDecoderModel):
            hidden_states = model.lora_encoder.get_hidden_states(images)
        elif isinstance(model, SwinUNETR):
            hidden_states = model.swinViT(images, model.normalize)
        else:
            raise TypeError(
                f"Unsupported model type: {type(model)}. "
                "Expected SwinUNETR or LoRAOriginalDecoderModel."
            )

    return [h.detach().cpu().float() for h in hidden_states]


def compute_tsi_single_scan(
    hidden_states: list[torch.Tensor],
    gt_mask: torch.Tensor,
    scan_id: str,
    condition: str,
    thresholds: list[float] = (1.5, 2.0),
    top_k: int = 3,
    return_maps: bool = False,
    epsilon: float = 1e-8,
    brain_mask: torch.Tensor | None = None,
) -> ScanTSIResult:
    """Compute TSI for all 5 stages on one scan.

    Parameters
    ----------
    hidden_states : list[torch.Tensor]
        Output of :func:`extract_hidden_states`, on CPU.
    gt_mask : torch.Tensor
        Binary tumor mask ``[D, H, W]`` at full resolution.
    scan_id : str
        Identifier of the scan.
    condition : str
        Free-form label, e.g. ``"frozen"`` or ``"adapted_r8_m0"``.
    thresholds : list[float]
        TSI thresholds for ``Frac`` reporting.
    top_k : int
        Number of top channels.
    return_maps : bool
        If True, store activation maps for visualisation.
    epsilon : float
        Numerical stability constant.
    brain_mask : torch.Tensor | None
        Optional binary brain mask ``[D, H, W]``. Downsampled per stage and
        passed to :func:`compute_tsi_single_stage` to enable the corrected
        formulation.

    Returns
    -------
    ScanTSIResult
        Per-stage results aggregated for the scan.
    """
    assert len(hidden_states) == 5, f"Expected 5 hidden states, got {len(hidden_states)}"

    stage_results: list[TSIResult] = []
    for s in range(5):
        h = hidden_states[s]
        spatial = tuple(h.shape[2:])
        mask_ds = downsample_mask(gt_mask, spatial)
        brain_ds = downsample_mask(brain_mask, spatial) if brain_mask is not None else None

        result = compute_tsi_single_stage(
            hidden_state=h,
            mask_ds=mask_ds,
            thresholds=list(thresholds),
            top_k=top_k,
            return_maps=return_maps,
            epsilon=epsilon,
            brain_mask_ds=brain_ds,
        )
        result.stage = s
        stage_results.append(result)

    return ScanTSIResult(scan_id=scan_id, condition=condition, stages=stage_results)
