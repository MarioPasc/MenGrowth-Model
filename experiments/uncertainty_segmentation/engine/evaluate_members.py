# experiments/uncertainty_segmentation/engine/evaluate_members.py
"""Per-member test-set evaluation for LoRA ensemble.

Evaluates each ensemble member independently on the BraTS-MEN test set,
producing per-member × per-subject Dice scores. This data enables:
    - Bootstrap 95% CIs on per-member Dice
    - Paired Wilcoxon signed-rank test: ensemble vs. each member
    - Cohen's d: effect size of ensembling
    - ICC across members
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from growth.data.bratsmendata import BraTSDatasetH5
from growth.data.transforms import get_h5_val_transforms
from growth.inference.sliding_window import sliding_window_segment
from growth.losses.segmentation import DiceMetric3Ch

from .ensemble_inference import EnsemblePredictor
from .paths import get_run_dir
from .save_predictions import save_multilabel_mask, _save_3d

logger = logging.getLogger(__name__)


def _convert_seg_to_binary(seg: torch.Tensor, domain: str = "MEN") -> torch.Tensor:
    """Convert integer seg labels to 3-channel binary masks (TC/WT/ET).

    Args:
        seg: Integer labels [1, D, H, W] with values {0, 1, 2, 3}.
        domain: "MEN" or "GLI".

    Returns:
        Binary masks [3, D, H, W].
    """
    seg = seg.squeeze(0).long()
    if domain == "MEN":
        tc = ((seg == 1) | (seg == 2)).float()
        wt = ((seg == 1) | (seg == 2) | (seg == 3)).float()
        et = (seg == 1).float()
    else:
        tc = ((seg == 1) | (seg == 3)).float()
        wt = (seg > 0).float()
        et = (seg == 3).float()
    return torch.stack([tc, wt, et], dim=0)


def _compute_dice_per_channel(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5
) -> torch.Tensor:
    """Compute per-channel Dice.

    Args:
        pred: Binary [C, D, H, W].
        target: Binary [C, D, H, W].
        smooth: Smoothing constant.

    Returns:
        Dice [C].
    """
    C = pred.shape[0]
    dice = torch.zeros(C)
    for c in range(C):
        p, t = pred[c].flatten(), target[c].flatten()
        intersection = (p * t).sum()
        dice[c] = (2 * intersection + smooth) / (p.sum() + t.sum() + smooth)
    return dice


def evaluate_per_member(
    config: DictConfig,
    device: str = "cuda",
    run_dir: str | Path | None = None,
    predictions_dir: Path | None = None,
) -> pd.DataFrame:
    """Evaluate each ensemble member independently on the test set.

    For each member m and each test scan:
        1. Load member m's model
        2. Run sliding_window_segment
        3. Compute Dice (TC, WT, ET) against ground truth
        4. Compute predicted WT volume
        5. Optionally save predicted masks (NIfTI)

    Args:
        config: Full experiment configuration.
        device: Inference device.
        run_dir: Override run directory.
        predictions_dir: If provided, save per-member and ensemble masks
            for BraTS-MEN test scans (for thesis figures).

    Returns:
        DataFrame with columns: member_id, scan_id, dice_tc, dice_wt,
        dice_et, dice_mean, volume_pred. Has M × N_test rows.
    """
    predictor = EnsemblePredictor(config, device=device, run_dir=run_dir)
    M = len(predictor.available_members)

    # Load test dataset
    h5_path = config.paths.men_h5_file
    roi_size = tuple(config.data.get("inference_roi_size", config.data.val_roi_size))
    transform = get_h5_val_transforms(roi_size=roi_size)
    dataset = BraTSDatasetH5(
        h5_path=h5_path,
        split=config.data.test_split,
        transform=transform,
        compute_semantic=False,
    )

    # Load scan IDs from H5 metadata (not from dataset samples — BUG-2 fix)
    with h5py.File(h5_path, "r") as f:
        all_scan_ids = [
            s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]
        ]
    splits = BraTSDatasetH5.load_splits_from_h5(h5_path)
    test_indices = splits.get(config.data.test_split, np.arange(len(all_scan_ids)))

    logger.info(f"Evaluating {M} members on {len(dataset)} test scans")

    rows: list[dict] = []

    for member_id in predictor.available_members:
        logger.info(f"Loading member {member_id}...")
        model = predictor._load_member_model(member_id)

        for i in range(len(dataset)):
            sample = dataset[i]
            images = sample["image"].unsqueeze(0).to(device)
            seg_gt = sample["seg"]
            sid = all_scan_ids[test_indices[i]]

            # Forward pass
            with torch.no_grad():
                logits = sliding_window_segment(
                    model, images,
                    roi_size=predictor.sw_roi_size,
                    sw_batch_size=predictor.sw_batch_size,
                    overlap=predictor.sw_overlap,
                    mode=predictor.sw_mode,
                )
            probs = torch.sigmoid(logits).float().squeeze(0).cpu()  # [3, D, H, W]
            pred_binary = (probs > 0.5).float()

            # Ground truth
            gt_binary = _convert_seg_to_binary(seg_gt, domain="MEN")

            # Dice
            dice = _compute_dice_per_channel(pred_binary, gt_binary)

            # Voxel count == mm³ (H5 pre-resampled to 1mm isotropic)
            vol_pred = float(pred_binary[1].sum().item())

            # Save per-member mask for BraTS-MEN test (thesis figures)
            if predictions_dir is not None:
                scan_dir = predictions_dir / sid
                scan_dir.mkdir(parents=True, exist_ok=True)
                _save_3d(
                    pred_binary[1],
                    scan_dir / f"member_{member_id}_mask.nii.gz",
                    dtype=np.int8,
                )

            rows.append({
                "member_id": member_id,
                "scan_id": sid,
                "dice_tc": float(dice[0]),
                "dice_wt": float(dice[1]),
                "dice_et": float(dice[2]),
                "dice_mean": float(dice.mean()),
                "volume_pred": vol_pred,
            })

        # Free GPU
        del model
        torch.cuda.empty_cache()
        logger.info(f"Member {member_id}: done ({len(dataset)} scans)")

    df = pd.DataFrame(rows)
    logger.info(
        f"Per-member evaluation: {len(df)} rows "
        f"({M} members × {len(dataset)} scans)"
    )
    return df


def evaluate_ensemble_per_subject(
    config: DictConfig,
    device: str = "cuda",
    run_dir: str | Path | None = None,
    collect_calibration: bool = True,
    predictions_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    """Evaluate the ensemble (mean probs) on the test set, per subject.

    Args:
        config: Full experiment configuration.
        device: Inference device.
        run_dir: Override run directory.
        collect_calibration: If True, collect subsampled probs/labels for
            calibration metrics (ECE, Brier, reliability).
        predictions_dir: If provided, save ensemble masks and multi-label
            segmentations for BraTS-MEN test scans.

    Returns:
        Tuple of (DataFrame, calibration_data_or_None).
        DataFrame columns: scan_id, dice_tc, dice_wt, dice_et,
        dice_mean, volume_ensemble, volume_gt. One row per scan.
        calibration_data: dict with 'probs' and 'labels' numpy arrays,
        or None if collect_calibration=False.
    """
    predictor = EnsemblePredictor(config, device=device, run_dir=run_dir)

    h5_path = config.paths.men_h5_file
    roi_size = tuple(config.data.get("inference_roi_size", config.data.val_roi_size))
    transform = get_h5_val_transforms(roi_size=roi_size)
    dataset = BraTSDatasetH5(
        h5_path=h5_path,
        split=config.data.test_split,
        transform=transform,
        compute_semantic=False,
    )

    # Load scan IDs from H5 metadata (BUG-2 fix)
    with h5py.File(h5_path, "r") as f:
        all_scan_ids = [
            s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]
        ]
    splits = BraTSDatasetH5.load_splits_from_h5(h5_path)
    test_indices = splits.get(config.data.test_split, np.arange(len(all_scan_ids)))

    logger.info(f"Evaluating ensemble on {len(dataset)} test scans")

    rows: list[dict] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    step = config.evaluation.get("subsample_voxels_step", 8)

    for i in range(len(dataset)):
        sample = dataset[i]
        images = sample["image"].unsqueeze(0)
        seg_gt = sample["seg"]
        sid = all_scan_ids[test_indices[i]]

        logger.info(f"Scan {i + 1}/{len(dataset)}: {sid}")

        result = predictor.predict_scan(images)

        # Ensemble Dice
        ensemble_pred = (result.mean_probs > 0.5).float()
        gt_binary = _convert_seg_to_binary(seg_gt, domain="MEN")
        dice = _compute_dice_per_channel(ensemble_pred, gt_binary)

        # Volumes
        vol_ensemble = float(ensemble_pred[1].sum().item())
        vol_gt = float(gt_binary[1].sum().item())

        rows.append({
            "scan_id": sid,
            "dice_tc": float(dice[0]),
            "dice_wt": float(dice[1]),
            "dice_et": float(dice[2]),
            "dice_mean": float(dice.mean()),
            "volume_ensemble": vol_ensemble,
            "volume_gt": vol_gt,
        })

        # Save ensemble mask + multi-label for BraTS-MEN test (thesis figures)
        if predictions_dir is not None:
            from .save_predictions import save_ensemble_mask
            save_ensemble_mask(result.ensemble_mask, predictions_dir, sid)
            save_multilabel_mask(result.mean_probs, predictions_dir, sid)

        # Collect calibration data (subsampled)
        if collect_calibration:
            probs_np = result.mean_probs.numpy()[:, ::step, ::step, ::step]
            labels_np = gt_binary.numpy()[:, ::step, ::step, ::step]
            all_probs.append(probs_np.reshape(probs_np.shape[0], -1).T)
            all_labels.append(labels_np.reshape(labels_np.shape[0], -1).T)

    calibration_data = None
    if collect_calibration and all_probs:
        calibration_data = {
            "probs": np.concatenate(all_probs, axis=0),
            "labels": np.concatenate(all_labels, axis=0),
        }

    return pd.DataFrame(rows), calibration_data
