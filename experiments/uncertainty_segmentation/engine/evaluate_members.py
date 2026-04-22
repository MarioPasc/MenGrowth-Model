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
from growth.inference.postprocess import remove_small_components
from growth.inference.sliding_window import sliding_window_segment

from .ensemble_convergence import (
    compute_ensemble_k_dice,
    compute_threshold_sensitivity,
)
from .ensemble_inference import EnsemblePredictor
from .save_predictions import (
    _save_3d,
    save_multilabel_mask,
    save_per_member_probs_all,
)

logger = logging.getLogger(__name__)


def _convert_seg_to_binary(seg: torch.Tensor, domain: str = "MEN") -> torch.Tensor:
    """Convert integer seg labels to 3-channel binary masks [TC, WT, ED].

    Thin wrapper around :func:`growth.losses.segmentation._convert_single_domain`
    so the MEN/GLI label mapping lives in a single source of truth.

    Args:
        seg: Integer labels [1, D, H, W] with values {0, 1, 2, 3}.
        domain: "MEN" or "GLI".

    Returns:
        Binary masks [3, D, H, W].
    """
    from growth.losses.segmentation import _convert_single_domain

    seg = seg.squeeze(0).long().unsqueeze(0)  # [1, D, H, W] for the shared helper
    masks = _convert_single_domain(seg, domain)  # [1, 3, D, H, W]
    return masks.squeeze(0)  # [3, D, H, W]


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
        dice_et, dice_mean, volume_pred. Has M × N_test rows. Dice
        columns are BraTS-hierarchical (TC=1|3, WT=seg>0, ET=3) —
        matching the training targets. volume_pred is the meningioma
        mass voxel count (BSF ch0 = BraTS-TC) after CC cleanup.
    """
    predictor = EnsemblePredictor(config, device=device, run_dir=run_dir)
    M = len(predictor.available_members)
    min_component_voxels = int(config.inference.get("min_component_voxels", 64))

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
        all_scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
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
                    model,
                    images,
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

            # Meningioma-mass volume from BSF ch0 (BraTS-TC = labels
            # 1|3). CC cleanup before counting voxels.
            pred_men = remove_small_components(
                pred_binary[0].bool(), min_voxels=min_component_voxels
            )
            vol_pred = float(pred_men.sum().item())

            # Save per-member meningioma-mass mask (thesis figures)
            if predictions_dir is not None:
                scan_dir = predictions_dir / sid
                scan_dir.mkdir(parents=True, exist_ok=True)
                _save_3d(
                    pred_men,
                    scan_dir / f"member_{member_id}_mask.nii.gz",
                    dtype=np.int8,
                )

            rows.append(
                {
                    "member_id": member_id,
                    "scan_id": sid,
                    "dice_tc": float(dice[0]),
                    "dice_wt": float(dice[1]),
                    "dice_et": float(dice[2]),
                    "dice_mean": float(dice[1:].mean()),
                    "volume_pred": vol_pred,
                }
            )

        # Free GPU
        del model
        torch.cuda.empty_cache()
        logger.info(f"Member {member_id}: done ({len(dataset)} scans)")

    df = pd.DataFrame(rows)
    logger.info(f"Per-member evaluation: {len(df)} rows ({M} members × {len(dataset)} scans)")
    return df


def evaluate_ensemble_per_subject(
    config: DictConfig,
    device: str = "cuda",
    run_dir: str | Path | None = None,
    collect_calibration: bool = True,
    predictions_dir: Path | None = None,
    eval_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    """Evaluate the ensemble (mean probs) on the test set, per subject.

    In addition to producing one Dice row per scan, this function
    optionally runs two additional analyses when
    ``config.inference.save_per_member_probs_all`` is true:

    * **Ensemble-of-k Dice** per channel, per scan (``convergence_ensemble_dice_{wt,tc,et}.csv``).
    * **Threshold-sensitivity** Dice per channel, per member + ensemble
      (``threshold_sensitivity.csv``).

    Both analyses are computed **streaming** from the in-memory
    ``result.per_member_probs`` returned by :class:`EnsemblePredictor` —
    they do not re-read the NIfTIs from disk, so enabling them adds
    only a few extra tensor ops per scan. The soft-prob NIfTIs are
    still written to disk so future post-hoc analyses (e.g., new
    threshold grids) can run without re-inference.

    Args:
        config: Full experiment configuration.
        device: Inference device.
        run_dir: Override run directory.
        collect_calibration: If True, collect subsampled probs/labels for
            calibration metrics (ECE, Brier, reliability).
        predictions_dir: If provided, save ensemble masks, multi-label
            segmentations, and (when enabled) per-member soft probs for
            BraTS-MEN test scans.
        eval_dir: Destination for the ensemble-k / threshold CSVs. If
            omitted, they are not written (caller owns the path).

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
        all_scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
    splits = BraTSDatasetH5.load_splits_from_h5(h5_path)
    test_indices = splits.get(config.data.test_split, np.arange(len(all_scan_ids)))

    logger.info(f"Evaluating ensemble on {len(dataset)} test scans")

    # Feature flags (evaluated once outside the per-scan loop).
    save_probs_all = (
        bool(config.inference.get("save_per_member_probs_all", False))
        and predictions_dir is not None
    )
    eval_cfg = config.get("evaluation", {})
    do_ensemble_k = bool(eval_cfg.get("compute_ensemble_k_dice", True))
    do_threshold = bool(eval_cfg.get("compute_threshold_sensitivity", True))
    threshold_grid = [
        float(t)
        for t in eval_cfg.get(
            "threshold_grid",
            [
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
            ],
        )
    ]
    # Streaming analyses need per-member probs in memory. Enable them
    # whenever at least one of {save-to-disk, ensemble-k, threshold}
    # is requested.
    need_per_member_probs = save_probs_all or do_ensemble_k or do_threshold

    rows: list[dict] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    step = eval_cfg.get("subsample_voxels_step", 8)

    # Accumulators for streaming analyses. Lists of per-scan DataFrames.
    ensemble_k_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        images = sample["image"].unsqueeze(0)
        seg_gt = sample["seg"]
        sid = all_scan_ids[test_indices[i]]

        logger.info(f"Scan {i + 1}/{len(dataset)}: {sid}")

        result = predictor.predict_scan(images, save_per_member=need_per_member_probs)

        # Ensemble Dice
        ensemble_pred = (result.mean_probs > 0.5).float()
        gt_binary = _convert_seg_to_binary(seg_gt, domain="MEN")
        dice = _compute_dice_per_channel(ensemble_pred, gt_binary)

        # ET volumes (ch2) = meningioma mass — the growth target
        vol_ensemble = float(ensemble_pred[2].sum().item())
        vol_gt = float(gt_binary[2].sum().item())

        rows.append(
            {
                "scan_id": sid,
                "dice_tc": float(dice[0]),
                "dice_wt": float(dice[1]),
                "dice_et": float(dice[2]),
                "dice_mean": float(dice[1:].mean()),
                "volume_ensemble": vol_ensemble,
                "volume_gt": vol_gt,
            }
        )

        # Save ensemble mask + multi-label + per-member probs for thesis figures.
        if predictions_dir is not None:
            from .save_predictions import save_ensemble_mask

            save_ensemble_mask(result.ensemble_mask, predictions_dir, sid)
            save_multilabel_mask(result.mean_probs, predictions_dir, sid)
            if save_probs_all and result.per_member_probs is not None:
                save_per_member_probs_all(result, predictions_dir, sid)

        # Streaming analyses over the in-memory per-member probs.
        if result.per_member_probs is not None:
            if do_ensemble_k:
                ek = compute_ensemble_k_dice(
                    result.per_member_probs,
                    gt_binary,
                )
                ek.insert(0, "scan_id", sid)
                ensemble_k_frames.append(ek)
            if do_threshold:
                ts = compute_threshold_sensitivity(
                    result.per_member_probs,
                    gt_binary,
                    threshold_grid,
                )
                ts.insert(0, "scan_id", sid)
                threshold_frames.append(ts)

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

    # Write the streaming-analysis CSVs if we have both an eval dir and content.
    if eval_dir is not None:
        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        if ensemble_k_frames:
            long_df = pd.concat(ensemble_k_frames, ignore_index=True)
            # Per-channel files for symmetry with convergence_dice_*.csv.
            for ch in ("wt", "tc", "et"):
                col = f"dice_{ch}"
                sub = long_df[["scan_id", "k", col]].rename(columns={col: "ensemble_dice"})
                sub.to_csv(eval_dir / f"convergence_ensemble_dice_{ch}.csv", index=False)
            logger.info(
                "Ensemble-of-k Dice written to %s/convergence_ensemble_dice_{wt,tc,et}.csv"
                " (%d scans × up to k=%d)",
                eval_dir,
                long_df["scan_id"].nunique(),
                int(long_df["k"].max()),
            )
        if threshold_frames:
            tdf = pd.concat(threshold_frames, ignore_index=True)
            tdf.to_csv(eval_dir / "threshold_sensitivity.csv", index=False)
            logger.info(
                "Threshold-sensitivity written to %s/threshold_sensitivity.csv"
                " (%d rows across %d scans, %d thresholds)",
                eval_dir,
                len(tdf),
                tdf["scan_id"].nunique(),
                tdf["threshold"].nunique(),
            )

    return pd.DataFrame(rows), calibration_data
