# experiments/uncertainty_segmentation/engine/evaluate_baseline.py
"""Frozen BrainSegFounder baseline evaluation on BraTS-MEN test set.

Evaluates the original BrainSegFounder without any LoRA adaptation or
decoder training. This provides the reference Dice scores for paired
statistical tests (ensemble vs. baseline).
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
from growth.models.encoder.swin_loader import load_full_swinunetr

from .evaluate_members import _compute_dice_per_channel, _convert_seg_to_binary
from .paths import get_run_dir

logger = logging.getLogger(__name__)


def evaluate_baseline(
    config: DictConfig,
    device: str = "cuda",
    run_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Evaluate frozen BrainSegFounder on BraTS-MEN test set.

    Loads the model with freeze_encoder=True, freeze_decoder=True (no training
    at all), then runs sliding-window inference on every test scan.

    This only needs to be computed once regardless of (rank, M, seed). If the
    output CSV already exists, returns the cached result unless force=True.

    Args:
        config: Full experiment configuration.
        device: Inference device.
        run_dir: Override run directory.
        force: Recompute even if CSV exists.

    Returns:
        DataFrame with columns: scan_id, dice_tc, dice_wt, dice_et,
        dice_mean, volume_ensemble, volume_gt. One row per scan.
    """
    resolved_run_dir = get_run_dir(config, override=run_dir)
    output_path = resolved_run_dir / "evaluation" / "baseline_test_dice.csv"

    if output_path.exists() and not force:
        logger.info(f"Baseline CSV already exists: {output_path}. Use --force to recompute.")
        return pd.read_csv(output_path)

    # Load frozen model
    checkpoint_path = str(
        Path(config.paths.checkpoint_dir) / config.paths.checkpoint_filename
    )
    out_channels = config.training.get("out_channels", 3)

    logger.info("Loading frozen BrainSegFounder (no adaptation)...")
    model = load_full_swinunetr(
        checkpoint_path,
        freeze_encoder=True,
        freeze_decoder=True,
        out_channels=out_channels,
        device=device,
    )
    model.eval()

    # Sliding window config
    sw_roi_size = tuple(config.inference.sw_roi_size)
    sw_batch_size = config.inference.sw_batch_size
    sw_overlap = config.inference.sw_overlap
    sw_mode = config.inference.sw_mode

    # Test dataset
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

    logger.info(f"Evaluating baseline on {len(dataset)} test scans")

    rows: list[dict] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        images = sample["image"].unsqueeze(0).to(device)
        seg_gt = sample["seg"]
        sid = all_scan_ids[test_indices[i]]

        logger.info(f"Baseline scan {i + 1}/{len(dataset)}: {sid}")

        with torch.no_grad():
            logits = sliding_window_segment(
                model, images,
                roi_size=sw_roi_size,
                sw_batch_size=sw_batch_size,
                overlap=sw_overlap,
                mode=sw_mode,
            )
        probs = torch.sigmoid(logits).float().squeeze(0).cpu()
        pred_binary = (probs > 0.5).float()

        gt_binary = _convert_seg_to_binary(seg_gt, domain="MEN")
        dice = _compute_dice_per_channel(pred_binary, gt_binary)

        vol_pred = float(pred_binary[1].sum().item())
        vol_gt = float(gt_binary[1].sum().item())

        rows.append({
            "scan_id": sid,
            "dice_tc": float(dice[0]),
            "dice_wt": float(dice[1]),
            "dice_et": float(dice[2]),
            "dice_mean": float(dice.mean()),
            "volume_pred": vol_pred,
            "volume_gt": vol_gt,
        })

    df = pd.DataFrame(rows)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        f"Baseline: mean Dice WT={df['dice_wt'].mean():.4f}, "
        f"saved to {output_path}"
    )

    return df
