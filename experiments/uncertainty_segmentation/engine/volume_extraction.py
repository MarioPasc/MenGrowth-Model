# experiments/uncertainty_segmentation/engine/volume_extraction.py
"""Volume extraction with uncertainty from ensemble predictions.

Iterates over all scans in an HDF5 dataset, runs ensemble inference, and
collects per-scan volume statistics into a structured DataFrame.

Output CSV format:
    scan_id, patient_id, timepoint_idx, vol_mean, vol_std, logvol_mean,
    logvol_std, vol_m0, vol_m1, ..., mean_entropy
"""

import logging
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from growth.data.bratsmendata import BraTSDatasetH5
from growth.data.transforms import get_h5_val_transforms

from .ensemble_inference import EnsemblePredictor
from .save_predictions import (
    save_ensemble_mask,
    save_multilabel_mask,
    save_per_member_masks_all,
    save_sample_predictions,
    select_sample_indices,
)

logger = logging.getLogger(__name__)


def extract_ensemble_volumes(
    predictor: EnsemblePredictor,
    h5_path: Path | str,
    config: DictConfig,
    split: str | None = None,
    predictions_dir: Path | None = None,
) -> pd.DataFrame:
    """Extract volumes with uncertainty for all scans in an HDF5 file.

    Args:
        predictor: Initialized EnsemblePredictor.
        h5_path: Path to HDF5 file (BraTS-MEN or MenGrowth).
        config: Full experiment configuration.
        split: Optional split name to restrict to (e.g., "test"). If None,
            processes all scans.

    Returns:
        DataFrame with one row per scan, columns:
            scan_id, patient_id, timepoint_idx,
            vol_mean, vol_std, logvol_mean, logvol_std,
            vol_m0, ..., vol_m{M-1},
            mean_entropy, mean_mi
    """
    h5_path = Path(h5_path)
    logger.info(f"Extracting volumes from {h5_path} (split={split})")

    # Read metadata from H5
    with h5py.File(h5_path, "r") as f:
        scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
        patient_ids = [s.decode() if isinstance(s, bytes) else s for s in f["patient_ids"][:]]
        timepoint_idx = (
            f["timepoint_idx"][:] if "timepoint_idx" in f else np.zeros(len(scan_ids), dtype=int)
        )
        n_total = len(scan_ids)

    # Determine indices to process
    if split is not None:
        with h5py.File(h5_path, "r") as f:
            if f"splits/{split}" in f:
                indices = f[f"splits/{split}"][:]
            else:
                logger.warning(f"Split '{split}' not found, using all scans")
                indices = np.arange(n_total)
    else:
        indices = np.arange(n_total)

    # Create dataset with validation transforms (no augmentation)
    roi_size = tuple(config.data.get("inference_roi_size", config.data.val_roi_size))
    transform = get_h5_val_transforms(roi_size=roi_size)
    dataset = BraTSDatasetH5(
        h5_path=h5_path,
        indices=indices,
        transform=transform,
        compute_semantic=False,
    )

    logger.info(f"Processing {len(dataset)} scans with {predictor.n_members} members")

    # Determine which scans get prediction saving
    save_masks = config.inference.get("save_ensemble_masks", False)
    save_samples = config.inference.get("save_sample_predictions", False)
    save_per_member_all = config.inference.get("save_per_member_masks_all", False)
    n_sample_scans = config.inference.get("n_sample_scans", 5)
    sample_strategy = config.inference.get("sample_scan_strategy", "spread")
    sample_indices = (
        set(select_sample_indices(len(dataset), n_sample_scans, sample_strategy))
        if save_samples
        else set()
    )

    # Collect results
    rows: list[dict] = []
    M = len(predictor.available_members)

    for i in range(len(dataset)):
        sample = dataset[i]
        scan_idx = indices[i]
        sid = scan_ids[scan_idx]
        pid = patient_ids[scan_idx]
        tp = int(timepoint_idx[scan_idx])

        logger.info(f"Scan {i + 1}/{len(dataset)}: {sid} (patient={pid}, tp={tp})")

        # Run ensemble prediction
        is_sample = i in sample_indices
        # Save per-member data if: sample scan (full probs+masks) OR all-scan masks
        need_per_member = is_sample or (save_per_member_all and predictions_dir is not None)
        images = sample["image"].unsqueeze(0)  # [1, 4, D, H, W]
        result = predictor.predict_scan(images, save_per_member=need_per_member)

        # Build row
        row: dict = {
            "scan_id": sid,
            "patient_id": pid,
            "timepoint_idx": tp,
            "vol_mean": result.volume_mean,
            "vol_std": result.volume_std,
            "logvol_mean": result.log_volume_mean,
            "logvol_std": result.log_volume_std,
            "vol_median": result.volume_median,
            "vol_mad": result.volume_mad,
            "logvol_median": result.log_volume_median,
            "logvol_mad": result.log_volume_mad,
            "logvol_mad_scaled": 1.4826 * result.log_volume_mad,
            "vol_ensemble_mask": float(result.ensemble_mask.sum().item()),
            "logvol_ensemble_mask": math.log(float(result.ensemble_mask.sum().item()) + 1),
        }

        # Per-member volumes and log-volumes
        for m_idx, vol in enumerate(result.per_member_volumes):
            row[f"vol_m{m_idx}"] = vol
            row[f"logvol_m{m_idx}"] = math.log(vol + 1)

        # Timing
        row["inference_time_sec"] = round(result.inference_time_sec, 1)

        # Mean uncertainty statistics
        row["mean_entropy"] = float(result.predictive_entropy.mean().item())
        row["mean_mi"] = float(result.mutual_information.mean().item())
        row["mean_var"] = float(result.var_probs.mean().item())

        # Meningioma-mass-specific uncertainty (from BSF ch0 = BraTS-TC
        # = labels 1|3). This is the clinical volume label.
        men_entropy = result.predictive_entropy[0]  # BSF ch0 (meningioma)
        men_mi = result.mutual_information[0]
        row["men_mean_entropy"] = float(men_entropy.mean().item())
        row["men_mean_mi"] = float(men_mi.mean().item())

        # Boundary uncertainty at the predicted meningioma boundary
        ensemble_mask = result.ensemble_mask
        if ensemble_mask.any():
            # Dilate mask to find boundary region (1-voxel)
            boundary = _find_boundary(ensemble_mask)
            if boundary.any():
                row["men_boundary_entropy"] = float(men_entropy[boundary].mean().item())
                row["men_boundary_mi"] = float(men_mi[boundary].mean().item())
            else:
                row["men_boundary_entropy"] = 0.0
                row["men_boundary_mi"] = 0.0
        else:
            row["men_boundary_entropy"] = 0.0
            row["men_boundary_mi"] = 0.0

        # Save predictions to NIfTI if configured
        if predictions_dir is not None:
            if save_masks or is_sample:
                save_ensemble_mask(result.ensemble_mask, predictions_dir, sid)
                # Multi-label segmentation (BraTS convention: 0/1/2/3)
                save_multilabel_mask(result.mean_probs, predictions_dir, sid)
            if save_per_member_all and result.per_member_masks is not None:
                save_per_member_masks_all(result, predictions_dir, sid)
            if is_sample and result.per_member_probs is not None:
                save_sample_predictions(result, predictions_dir, sid)

        rows.append(row)

    # Create DataFrame and sort by patient, timepoint
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["patient_id", "timepoint_idx"]).reset_index(drop=True)

    logger.info(
        f"Extraction complete: {len(df)} scans, "
        f"mean vol={df['vol_mean'].mean():.0f} ± {df['vol_std'].mean():.0f} mm³"
    )

    return df


def _find_boundary(mask: torch.Tensor) -> torch.Tensor:
    """Find boundary voxels of a binary mask via morphological gradient.

    Boundary = dilated(mask) XOR mask. Uses 6-connectivity (3D cross kernel).

    Args:
        mask: Binary mask [D, H, W].

    Returns:
        Binary boundary mask [D, H, W].
    """
    # Pad, then check 6-connected neighbors
    padded = torch.nn.functional.pad(
        mask.float().unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1, 1, 1),
        mode="constant",
        value=0.0,
    )

    # 3D max pool with kernel 3 = dilation
    dilated = torch.nn.functional.max_pool3d(padded, kernel_size=3, stride=1, padding=0)
    dilated = dilated.squeeze(0).squeeze(0) > 0.5

    # Boundary = dilated XOR original (must be bool for XOR)
    return dilated ^ mask.bool()
