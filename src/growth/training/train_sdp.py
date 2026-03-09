# src/growth/training/train_sdp.py
"""Phase 2 training entry point: SDP training.

Loads precomputed encoder features and trains the SDP projection network.
Utility functions for loading features from HDF5 files, building the
SDPLitModule, and generating quality reports.
"""

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from growth.losses.dcor import distance_correlation
from growth.training.lit_modules.sdp_module import SDPLitModule

logger = logging.getLogger(__name__)


def load_precomputed_features(
    features_path: str,
    feature_level: str = "encoder10",
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Load precomputed features and targets from HDF5 file.

    Methodology Revision R1: only loads volume target (index 0 = log V_WT).

    Args:
        features_path: Path to .h5 file containing features and targets.
        feature_level: Feature level to load (e.g., "encoder10").
        **kwargs: Ignored (backward compat for shape_indices).

    Returns:
        Tuple of (features [N, 768], targets dict with "vol").
    """
    features_path = Path(features_path)

    with h5py.File(features_path, "r") as f:
        h = torch.from_numpy(np.array(f[f"features/{feature_level}"]))

        targets = {}
        volume_all = torch.from_numpy(np.array(f["targets/volume"]))
        targets["vol"] = volume_all[:, 0:1]  # [N, 1] — log(V_WT + 1)

    logger.info(
        f"Loaded features from {features_path}: h={h.shape}, "
        f"vol={targets['vol'].shape}"
    )
    return h, targets


def load_and_combine_splits(
    features_dir: str,
    split_names: list[str],
    feature_level: str = "encoder10",
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Load and concatenate features from multiple splits.

    Args:
        features_dir: Directory containing per-split .h5 files.
        split_names: List of split names to combine.
        feature_level: Feature level to load.
        **kwargs: Ignored (backward compat for shape_indices).

    Returns:
        Combined (features, targets) tensors.
    """
    all_h = []
    all_targets: dict[str, list[torch.Tensor]] = {"vol": []}

    for split in split_names:
        h5_path = Path(features_dir) / f"{split}.h5"
        h, targets = load_precomputed_features(str(h5_path), feature_level)
        all_h.append(h)
        for key in all_targets:
            all_targets[key].append(targets[key])

    combined_h = torch.cat(all_h, dim=0)
    combined_targets = {key: torch.cat(vals, dim=0) for key, vals in all_targets.items()}

    logger.info(f"Combined {len(split_names)} splits: {combined_h.shape[0]} total samples")
    return combined_h, combined_targets


def build_sdp_module(config: DictConfig) -> SDPLitModule:
    """Build SDPLitModule from configuration.

    Args:
        config: Full SDP configuration.

    Returns:
        Configured SDPLitModule.
    """
    return SDPLitModule(config)


@torch.no_grad()
def generate_quality_report(
    module: SDPLitModule,
    h_val: torch.Tensor,
    targets_val: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Generate quality report on validation data.

    Computes R^2 per partition, dCor between partitions,
    cross-partition correlations, and per-dimension variance stats.

    Args:
        module: Trained SDPLitModule.
        h_val: Validation encoder features (unnormalized).
        targets_val: Validation targets (unnormalized).

    Returns:
        Quality report dict.
    """
    module.eval()
    device = next(module.parameters()).device

    # Normalize using train stats
    h_norm = module._normalize_features(h_val.to(device))
    targets_norm = module._normalize_targets({k: v.to(device) for k, v in targets_val.items()})

    # Forward pass
    z, partitions, predictions = module.model(h_norm)

    report: dict[str, Any] = {}

    # R^2 per partition (volume only after R1)
    for key in ["vol"]:
        pred = predictions[key]
        target = targets_norm[key]
        ss_res = ((pred - target) ** 2).sum()
        ss_tot = ((target - target.mean(dim=0)) ** 2).sum()
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
        report[f"r2_{key}"] = r2

    # Distance correlation between vol and residual
    dcor = float(distance_correlation(partitions["vol"], partitions["residual"]))
    report["dcor_vol_residual"] = dcor

    # Max cross-partition Pearson correlation (vol vs residual)
    zi = partitions["vol"]
    zj = partitions["residual"]
    corr_matrix = torch.corrcoef(torch.cat([zi.T, zj.T], dim=0))
    di = zi.shape[1]
    cross_block = corr_matrix[:di, di:]
    report["max_cross_partition_corr"] = float(cross_block.abs().max())

    # Per-dimension variance stats
    z_std = z.std(dim=0)
    report["pct_dims_variance_gt_03"] = float((z_std > 0.3).float().mean())
    report["pct_dims_variance_gt_05"] = float((z_std > 0.5).float().mean())
    report["mean_dim_std"] = float(z_std.mean())
    report["min_dim_std"] = float(z_std.min())

    return report
