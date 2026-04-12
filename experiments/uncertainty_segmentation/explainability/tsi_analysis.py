"""Core TSI (Tumor Selectivity Index) computation.

For each encoder stage, for each channel:
    TSI_c = mean(|activation_c| inside tumor) / mean(|activation_c| outside tumor)

TSI >> 1: channel is tumor-selective.
TSI ≈ 1: channel is anatomically general.
TSI < 1: channel encodes "absence of normal tissue".

Reference: Network Dissection (Bau et al., CVPR 2017) — TSI is the continuous,
medical-domain analogue of unit-level IoU interpretability.
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
    from growth.models.segmentation.original_decoder import LoRAOriginalDecoderModel

logger = logging.getLogger(__name__)

# Stage metadata: (channels, downsample_factor) for feature_size=48.
# has_lora is derived per-run from config.lora.target_stages via build_stage_meta().
# Stage 0 is the patch embedding (before swinViT.layers*) and is never LoRA-adapted.
_STAGE_SHAPE = {
    0: {"channels": 48, "downsample": 2, "name": "patch_embed"},
    1: {"channels": 96, "downsample": 4, "name": "layers1"},
    2: {"channels": 192, "downsample": 8, "name": "layers2"},
    3: {"channels": 384, "downsample": 16, "name": "layers3"},
    4: {"channels": 768, "downsample": 32, "name": "layers4"},
}


def build_stage_meta(target_stages: list[int] | tuple[int, ...]) -> dict[int, dict]:
    """Build per-stage metadata with has_lora derived from the run's target_stages.

    Args:
        target_stages: The LoRA target_stages list from config.lora.target_stages.
            Values must be in {1, 2, 3, 4} (stage 0 is the patch embedding and
            cannot host LoRA).

    Returns:
        Dict mapping stage index (0-4) to metadata dict with keys
        "channels", "downsample", "name", and "has_lora".
    """
    target_set = {int(s) for s in target_stages}
    invalid = target_set - {1, 2, 3, 4}
    if invalid:
        raise ValueError(
            f"target_stages must be a subset of {{1,2,3,4}}, got invalid values: {sorted(invalid)}"
        )
    return {
        s: {**_STAGE_SHAPE[s], "has_lora": s in target_set}
        for s in range(5)
    }


# Backward-compat alias: historical default was target_stages=[3, 4].
# New code should call build_stage_meta(target_stages) instead.
STAGE_META = build_stage_meta([3, 4])


@dataclasses.dataclass
class TSIResult:
    """TSI computation result for one scan at one stage."""

    stage: int
    n_channels: int
    resolution: tuple[int, int, int]
    tsi_per_channel: np.ndarray  # [C_s]
    mean_tsi: float
    std_tsi: float
    frac_above: dict[float, float]  # {threshold: fraction}
    wilcoxon_p: float  # H0: median TSI = 1 (one-sided greater)
    top_k_channels: list[int]  # Indices of top-K TSI channels

    # Visualization data (optional, only for first scan)
    mean_activation_map: np.ndarray | None = None  # [D_s, H_s, W_s]
    top_channels_map: np.ndarray | None = None  # [D_s, H_s, W_s]


@dataclasses.dataclass
class ScanTSIResult:
    """TSI results for all 5 stages of one scan under one condition."""

    scan_id: str
    condition: str  # "frozen" or "adapted"
    stages: list[TSIResult]  # len=5


def downsample_mask(
    mask: torch.Tensor,
    target_size: tuple[int, int, int],
) -> torch.Tensor:
    """Downsample a binary mask to a target spatial resolution.

    Uses nearest-neighbor interpolation to preserve binary values.

    Args:
        mask: Binary mask [D, H, W] with values in {0, 1}.
        target_size: Target spatial dimensions (D_s, H_s, W_s).

    Returns:
        Downsampled binary mask [D_s, H_s, W_s].
    """
    # F.interpolate expects [B, C, D, H, W]
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
) -> TSIResult:
    """Compute TSI for all channels at one encoder stage.

    Args:
        hidden_state: Activation tensor [1, C_s, D_s, H_s, W_s] (float32, CPU).
        mask_ds: Downsampled binary WT mask [D_s, H_s, W_s].
        thresholds: TSI thresholds for Frac computation.
        top_k: Number of top tumor-selective channels to track.
        return_maps: If True, compute and store activation maps for visualization.
        epsilon: Numerical stability constant.

    Returns:
        TSIResult with per-channel TSI values and summary statistics.
    """
    # Remove batch dim: [C_s, D_s, H_s, W_s]
    h = hidden_state.squeeze(0).float()
    C_s = h.shape[0]
    spatial = tuple(h.shape[1:])

    # Absolute activation
    abs_h = h.abs()

    # Mask volumes
    tumor_count = mask_ds.sum() + epsilon
    non_tumor_count = (1.0 - mask_ds).sum() + epsilon

    # Guard: if mask is empty in either region, TSI is undefined
    if mask_ds.sum() < 1 or (1.0 - mask_ds).sum() < 1:
        logger.warning(
            f"Stage spatial={spatial}: tumor_voxels={mask_ds.sum().item():.0f}, "
            f"non_tumor_voxels={(1-mask_ds).sum().item():.0f} — TSI undefined"
        )
        tsi_arr = np.full(C_s, np.nan)
        return TSIResult(
            stage=-1,  # filled by caller
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

    # Per-channel TSI: vectorized over channels
    # abs_h: [C, D, H, W], mask_ds: [D, H, W]
    mu_tumor = (abs_h * mask_ds).sum(dim=(1, 2, 3)) / tumor_count  # [C]
    mu_non_tumor = (abs_h * (1.0 - mask_ds)).sum(dim=(1, 2, 3)) / non_tumor_count
    tsi = mu_tumor / (mu_non_tumor + epsilon)  # [C]

    tsi_np = tsi.numpy()

    # Summary statistics
    mean_tsi = float(np.nanmean(tsi_np))
    std_tsi = float(np.nanstd(tsi_np))

    # Frac above thresholds
    frac_above = {}
    for tau in thresholds:
        frac_above[tau] = float(np.nanmean(tsi_np > tau))

    # Wilcoxon signed-rank test: H0: median(TSI) = 1
    try:
        stat_result = scipy.stats.wilcoxon(
            tsi_np - 1.0, alternative="greater", nan_policy="omit"
        )
        wilcoxon_p = float(stat_result.pvalue)
    except ValueError:
        # All values identical (e.g., all TSI = 1.0) — no evidence against H0
        wilcoxon_p = 1.0

    # Top-K channels by TSI
    valid_mask = ~np.isnan(tsi_np)
    if valid_mask.sum() >= top_k:
        top_indices = np.argsort(tsi_np)[-top_k:][::-1].tolist()
    else:
        top_indices = np.where(valid_mask)[0].tolist()

    # Visualization maps (optional)
    mean_act_map = None
    top_ch_map = None
    if return_maps:
        mean_act_map = abs_h.mean(dim=0).numpy()  # [D_s, H_s, W_s]
        if top_indices:
            top_ch_map = abs_h[top_indices].mean(dim=0).numpy()

    return TSIResult(
        stage=-1,  # filled by caller
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
    """Extract encoder hidden states from either frozen or LoRA-adapted model.

    Handles both model types transparently:
    - SwinUNETR (frozen): model.swinViT(x, model.normalize)
    - LoRAOriginalDecoderModel (adapted): model.lora_encoder.get_hidden_states(x)

    All hidden states are returned on CPU in float32 to free GPU memory.

    Args:
        model: Frozen SwinUNETR or LoRA-adapted LoRAOriginalDecoderModel.
        images: Input tensor [1, 4, D, H, W] on the model's device.

    Returns:
        List of 5 tensors (one per stage), each on CPU in float32.
    """
    # Import here to avoid circular dependency at module level
    from growth.models.segmentation.original_decoder import LoRAOriginalDecoderModel

    with torch.no_grad():
        if isinstance(model, LoRAOriginalDecoderModel):
            hidden_states = model.lora_encoder.get_hidden_states(images)
        elif isinstance(model, SwinUNETR):
            hidden_states = model.swinViT(images, model.normalize)
        else:
            raise TypeError(
                f"Unsupported model type: {type(model)}. "
                f"Expected SwinUNETR or LoRAOriginalDecoderModel."
            )

    # Move to CPU and cast to float32 (hidden states may be bf16 under AMP)
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
) -> ScanTSIResult:
    """Compute TSI for all 5 stages on one scan.

    Args:
        hidden_states: List of 5 tensors from extract_hidden_states(), on CPU.
        gt_mask: Binary WT mask [D, H, W] at full resolution (192³).
        scan_id: Scan identifier string.
        condition: "frozen" or "adapted".
        thresholds: TSI thresholds for Frac computation.
        top_k: Number of top channels to track.
        return_maps: If True, store activation maps for the first scan.
        epsilon: Numerical stability constant.

    Returns:
        ScanTSIResult with TSI for all 5 stages.
    """
    assert len(hidden_states) == 5, f"Expected 5 hidden states, got {len(hidden_states)}"

    stage_results = []
    for s in range(5):
        h = hidden_states[s]
        spatial = tuple(h.shape[2:])  # (D_s, H_s, W_s)

        # Downsample GT mask to this stage's resolution
        mask_ds = downsample_mask(gt_mask, spatial)

        result = compute_tsi_single_stage(
            hidden_state=h,
            mask_ds=mask_ds,
            thresholds=list(thresholds),
            top_k=top_k,
            return_maps=return_maps,
            epsilon=epsilon,
        )
        result.stage = s

        stage_results.append(result)

    return ScanTSIResult(
        scan_id=scan_id,
        condition=condition,
        stages=stage_results,
    )
