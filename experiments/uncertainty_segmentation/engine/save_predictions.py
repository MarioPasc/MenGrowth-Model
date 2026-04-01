# experiments/uncertainty_segmentation/engine/save_predictions.py
"""NIfTI saving of ensemble predictions for thesis figures and analysis.

Two saving modes:
    1. All scans: ensemble WT hard mask (int8, ~1-7 MB compressed)
    2. Sample scans: full per-member predictions, ensemble probs, uncertainty maps

NIfTI convention:
    - 3D masks [D, H, W] → save directly (nibabel interprets as spatial axes)
    - 4D data [C, D, H, W] → permute to [D, H, W, C] (nibabel 4th dim = channel)
    - Affine: np.eye(4) for 1mm isotropic RAS (BraTS preprocessing standard)
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import torch

if TYPE_CHECKING:
    from .ensemble_inference import EnsemblePrediction

logger = logging.getLogger(__name__)

# BraTS-preprocessed data: 1mm isotropic, RAS orientation
_IDENTITY_AFFINE = np.eye(4, dtype=np.float64)


def save_ensemble_mask(
    mask: torch.Tensor,
    output_dir: Path,
    scan_id: str,
) -> Path:
    """Save ensemble WT hard mask as NIfTI.

    Args:
        mask: Binary WT mask [D, H, W] (bool or float).
        output_dir: Predictions base directory.
        scan_id: Scan identifier (used as subdirectory name).

    Returns:
        Path to saved file.
    """
    scan_dir = output_dir / scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)

    data = mask.numpy().astype(np.int8)
    img = nib.Nifti1Image(data, _IDENTITY_AFFINE)
    out_path = scan_dir / "ensemble_mask.nii.gz"
    nib.save(img, str(out_path))

    logger.debug(f"Saved ensemble mask: {out_path} ({data.sum()} voxels)")
    return out_path


def save_sample_predictions(
    result: "EnsemblePrediction",
    output_dir: Path,
    scan_id: str,
) -> None:
    """Save full spatial predictions for a sample scan.

    Saves ensemble-level maps and per-member predictions for thesis figures
    showing expert disagreement at tumor boundaries.

    Args:
        result: EnsemblePrediction with per_member_probs/masks populated.
        output_dir: Predictions base directory.
        scan_id: Scan identifier.
    """
    scan_dir = output_dir / scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)

    # Ensemble mask (always)
    _save_3d(result.ensemble_mask, scan_dir / "ensemble_mask.nii.gz", dtype=np.int8)

    # Ensemble mean probabilities [C, D, H, W] → [D, H, W, C]
    _save_4d(result.mean_probs, scan_dir / "ensemble_probs.nii.gz")

    # Uncertainty maps
    _save_4d(result.predictive_entropy, scan_dir / "entropy.nii.gz")
    _save_4d(result.mutual_information, scan_dir / "mutual_information.nii.gz")

    # Per-member predictions
    if result.per_member_probs is not None:
        for m, (probs_m, mask_m) in enumerate(
            zip(result.per_member_probs, result.per_member_masks or [])
        ):
            _save_4d(probs_m, scan_dir / f"member_{m}_probs.nii.gz")
            if mask_m is not None:
                _save_3d(mask_m, scan_dir / f"member_{m}_mask.nii.gz", dtype=np.int8)

    logger.info(f"Saved sample predictions: {scan_dir}")


def save_multilabel_mask(
    probs: torch.Tensor,
    output_dir: Path,
    scan_id: str,
    filename: str = "segmentation.nii.gz",
) -> Path:
    """Save a multi-label segmentation mask in BraTS convention.

    Converts 3-channel sigmoid probabilities (TC/WT/ET) to integer labels:
        0 = Background
        1 = NCR/NET (in TC but not ET)
        2 = ED (in WT but not TC)
        3 = ET (enhancing tumor)

    This is the inverse of the TC/WT/ET → integer label mapping used during
    training. Matches BraTS submission format for clinical review.

    Args:
        probs: Sigmoid probabilities [3, D, H, W] (TC=ch0, WT=ch1, ET=ch2).
        output_dir: Predictions base directory.
        scan_id: Scan identifier.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    scan_dir = output_dir / scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)

    # Threshold to binary
    tc = probs[0] > 0.5  # Tumor Core
    wt = probs[1] > 0.5  # Whole Tumor
    et = probs[2] > 0.5  # Enhancing Tumor

    # Convert to BraTS integer labels
    seg = torch.zeros_like(probs[0], dtype=torch.int8)
    seg[wt & ~tc] = 2         # ED: in WT but not TC
    seg[tc & ~et] = 1         # NCR/NET: in TC but not ET
    seg[et] = 3               # ET: enhancing

    out_path = scan_dir / filename
    nib.save(
        nib.Nifti1Image(seg.numpy().astype(np.int8), _IDENTITY_AFFINE),
        str(out_path),
    )
    logger.debug(f"Saved multi-label mask: {out_path}")
    return out_path


def save_per_member_masks_all(
    result: "EnsemblePrediction",
    output_dir: Path,
    scan_id: str,
) -> None:
    """Save per-member WT masks for a scan (lightweight, for all scans).

    Only saves the binary WT masks (int8), not full probability maps.
    Each mask is ~1-7MB compressed. For M=5, total ~5-35MB per scan.

    Args:
        result: EnsemblePrediction (per_member_masks must be populated).
        output_dir: Predictions base directory.
        scan_id: Scan identifier.
    """
    if result.per_member_masks is None:
        return

    scan_dir = output_dir / scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)

    for m, mask_m in enumerate(result.per_member_masks):
        _save_3d(mask_m, scan_dir / f"member_{m}_mask.nii.gz", dtype=np.int8)


def select_sample_indices(
    n_total: int,
    n_samples: int,
    strategy: str = "spread",
) -> list[int]:
    """Select indices for sample scans that get full prediction saving.

    Args:
        n_total: Total number of scans.
        n_samples: Number of sample scans to select.
        strategy: "spread" (evenly spaced) or "first" (first N).

    Returns:
        List of selected indices.
    """
    if n_samples >= n_total:
        return list(range(n_total))

    if strategy == "first":
        return list(range(n_samples))
    elif strategy == "spread":
        return [int(i) for i in np.linspace(0, n_total - 1, n_samples)]
    else:
        raise ValueError(f"Unknown sample strategy: {strategy}. Use 'spread' or 'first'.")


# =============================================================================
# Internal helpers
# =============================================================================


def _save_3d(
    tensor: torch.Tensor,
    path: Path,
    dtype: type = np.float32,
) -> None:
    """Save a 3D tensor as NIfTI."""
    data = tensor.numpy().astype(dtype)
    nib.save(nib.Nifti1Image(data, _IDENTITY_AFFINE), str(path))


def _save_4d(
    tensor: torch.Tensor,
    path: Path,
    dtype: type = np.float32,
) -> None:
    """Save a 4D [C, D, H, W] tensor as NIfTI [D, H, W, C]."""
    data = tensor.permute(1, 2, 3, 0).numpy().astype(dtype)
    nib.save(nib.Nifti1Image(data, _IDENTITY_AFFINE), str(path))
