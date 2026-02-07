#!/usr/bin/env python
# experiments/lora_ablation/diagnose_frozen_gli.py
"""Diagnose frozen BrainSegFounder Dice on BraTS-GLI data.

BraTS-GLI uses labels {0, 2, 4} (no label 1/NCR, ET=4), but _convert_target()
expects {0, 1, 2, 3}. This causes TC and ET channels to be empty for GLI data,
fully explaining the near-zero Dice observed in previous evaluations.

This script tests 4 conditions to confirm the root cause:
  1. (center_crop, original_labels)  — current pipeline (expected ~0.02 Dice)
  2. (center_crop, remapped_labels)  — remap {4→3} before _convert_target()
  3. (sliding_window, original_labels) — MONAI sliding window inference
  4. (sliding_window, remapped_labels) — combined fix

Usage:
    ~/.conda/envs/growth/bin/python experiments/lora_ablation/diagnose_frozen_gli.py
"""

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.inferers import sliding_window_inference

from growth.losses.segmentation import DiceMetric3Ch
from growth.models.encoder.swin_loader import load_full_swinunetr

# Reuse BraTSGLIDataset from existing code
from experiments.lora_ablation.extract_domain_features import BraTSGLIDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Configuration ----
GLI_DATA_ROOT = (
    "/media/mpascual/PortableSSD/BraTS_GLI/source/"
    "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2"
)
CHECKPOINT_PATH = (
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/checkpoints/"
    "BrainSegFounder_finetuned_BraTS/finetuned_model_fold_0.pt"
)
OUTPUT_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "diagnose_frozen_gli"
)
MAX_SUBJECTS = 200
BATCH_SIZE = 2
NUM_WORKERS = 4
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 4
SW_OVERLAP = 0.5


def remap_gli_labels(seg: torch.Tensor) -> torch.Tensor:
    """Remap BraTS-GLI labels {0, 2, 4} → {0, 2, 3}.

    GLI uses: 0=background, 2=ED, 4=ET.
    BrainSegFounder expects: 0=background, 1=NCR, 2=ED, 3=ET.
    GLI has no NCR (label 1), so we only need to remap 4→3.
    """
    seg = seg.clone()
    seg[seg == 4] = 3
    return seg


def log_label_statistics(
    dataloader: DataLoader,
    n_subjects: int = 5,
) -> None:
    """Log per-sample label statistics for the first few subjects."""
    logger.info(f"\nLabel statistics for first {n_subjects} subjects:")
    logger.info("-" * 60)

    for i, batch in enumerate(dataloader):
        if i >= n_subjects:
            break
        seg = batch["seg"]  # [B, 1, D, H, W]
        subject_id = batch["subject_id"]

        for b in range(seg.shape[0]):
            s = seg[b]
            unique_vals = torch.unique(s).tolist()
            total_voxels = s.numel()
            fractions = {
                int(v): float((s == v).sum()) / total_voxels
                for v in unique_vals
            }
            sid = subject_id[b] if isinstance(subject_id, list) else subject_id
            logger.info(
                f"  {sid}: unique={unique_vals}, "
                f"fractions={{{', '.join(f'{k}: {v:.4f}' for k, v in sorted(fractions.items()))}}}"
            )


@torch.no_grad()
def evaluate_center_crop(
    model: nn.Module,
    dataloader: DataLoader,
    dice_metric: DiceMetric3Ch,
    remap: bool,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Evaluate with center-crop (standard pipeline).

    Returns:
        Tuple of (per_sample_dice [N, 3], summary dict).
    """
    model.eval()
    all_dice = []

    for batch in tqdm(dataloader, desc=f"Center-crop (remap={remap})", leave=False):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        if remap:
            segs = remap_gli_labels(segs)

        pred = model(images)
        dice = dice_metric(pred, segs)  # [B, 3]
        all_dice.append(dice.cpu())

    dice_all = torch.cat(all_dice, dim=0)  # [N, 3]
    return dice_all, _summarize(dice_all)


@torch.no_grad()
def evaluate_sliding_window(
    model: nn.Module,
    dataloader: DataLoader,
    dice_metric: DiceMetric3Ch,
    remap: bool,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Evaluate with MONAI sliding window inference.

    Returns:
        Tuple of (per_sample_dice [N, 3], summary dict).
    """
    model.eval()
    all_dice = []

    for batch in tqdm(dataloader, desc=f"Sliding-window (remap={remap})", leave=False):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        if remap:
            segs = remap_gli_labels(segs)

        pred = sliding_window_inference(
            inputs=images,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=SW_OVERLAP,
        )

        dice = dice_metric(pred, segs)  # [B, 3]
        all_dice.append(dice.cpu())

    dice_all = torch.cat(all_dice, dim=0)  # [N, 3]
    return dice_all, _summarize(dice_all)


def _summarize(dice_all: torch.Tensor) -> Dict[str, float]:
    """Summarize per-sample Dice tensor [N, 3] → dict."""
    return {
        "dice_mean": float(dice_all.mean()),
        "dice_TC": float(dice_all[:, 0].mean()),
        "dice_WT": float(dice_all[:, 1].mean()),
        "dice_ET": float(dice_all[:, 2].mean()),
        "dice_TC_std": float(dice_all[:, 0].std()),
        "dice_WT_std": float(dice_all[:, 1].std()),
        "dice_ET_std": float(dice_all[:, 2].std()),
        "n_samples": int(dice_all.shape[0]),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ---- Output directory ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    logger.info("Loading frozen BrainSegFounder model...")
    model = load_full_swinunetr(
        ckpt_path=CHECKPOINT_PATH,
        freeze_encoder=True,
        freeze_decoder=True,
        out_channels=3,
        device=device,
    )
    model.eval()

    # ---- Load GLI dataset ----
    logger.info(f"Loading BraTS-GLI from {GLI_DATA_ROOT}")
    gli_path = Path(GLI_DATA_ROOT)

    # Filter to subjects with "-100" in folder name (per plan)
    all_subjects = sorted([
        d.name for d in gli_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    subjects_100 = [s for s in all_subjects if "-100" in s]

    if len(subjects_100) == 0:
        logger.warning("No subjects with '-100' in name. Using all subjects.")
        subjects_100 = all_subjects

    subjects = subjects_100[:MAX_SUBJECTS]
    logger.info(f"Selected {len(subjects)} subjects (from {len(all_subjects)} total, {len(subjects_100)} with '-100')")

    dataset = BraTSGLIDataset(data_root=GLI_DATA_ROOT, subject_ids=subjects)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ---- Log label statistics ----
    log_label_statistics(dataloader, n_subjects=5)

    # ---- Evaluate all 4 conditions ----
    dice_metric = DiceMetric3Ch()
    results = {}
    conditions = [
        ("center_crop", "original_labels", False, False),
        ("center_crop", "remapped_labels", False, True),
        ("sliding_window", "original_labels", True, False),
        ("sliding_window", "remapped_labels", True, True),
    ]

    for inference_mode, label_mode, use_sw, use_remap in conditions:
        key = f"{inference_mode}__{label_mode}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition: {key}")
        logger.info(f"  Inference: {inference_mode}, Labels: {label_mode}")
        logger.info(f"{'='*60}")

        t0 = time.time()
        if use_sw:
            dice_all, summary = evaluate_sliding_window(
                model, dataloader, dice_metric, use_remap, device
            )
        else:
            dice_all, summary = evaluate_center_crop(
                model, dataloader, dice_metric, use_remap, device
            )
        elapsed = time.time() - t0

        summary["time_seconds"] = round(elapsed, 1)
        results[key] = summary

        logger.info(
            f"  Dice mean={summary['dice_mean']:.4f}  "
            f"TC={summary['dice_TC']:.4f} (+/-{summary['dice_TC_std']:.4f})  "
            f"WT={summary['dice_WT']:.4f} (+/-{summary['dice_WT_std']:.4f})  "
            f"ET={summary['dice_ET']:.4f} (+/-{summary['dice_ET_std']:.4f})  "
            f"[{elapsed:.1f}s]"
        )

        # Save per-sample dice for this condition
        torch.save(dice_all, OUTPUT_DIR / f"dice_per_sample__{key}.pt")

    # ---- Save results CSV ----
    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "dice_mean", "dice_TC", "dice_WT", "dice_ET",
            "dice_TC_std", "dice_WT_std", "dice_ET_std", "n_samples", "time_seconds",
        ])
        for key, s in results.items():
            writer.writerow([
                key, f"{s['dice_mean']:.4f}",
                f"{s['dice_TC']:.4f}", f"{s['dice_WT']:.4f}", f"{s['dice_ET']:.4f}",
                f"{s['dice_TC_std']:.4f}", f"{s['dice_WT_std']:.4f}", f"{s['dice_ET_std']:.4f}",
                s["n_samples"], s["time_seconds"],
            ])
    logger.info(f"\nSaved results CSV to {csv_path}")

    # ---- Generate analysis text ----
    analysis_lines = [
        "=" * 70,
        "DIAGNOSIS: Frozen BrainSegFounder on BraTS-GLI",
        "=" * 70,
        "",
        "HYPOTHESIS: BraTS-GLI labels use {0, 2, 4} while _convert_target()",
        "expects {0, 1, 2, 3}. This makes TC=(label==1|label==3) and",
        "ET=(label==3) empty for GLI, causing near-zero Dice.",
        "",
        "RESULTS:",
        f"  {'Condition':<40} {'Mean':>6} {'TC':>6} {'WT':>6} {'ET':>6}",
        "  " + "-" * 64,
    ]

    for key, s in results.items():
        analysis_lines.append(
            f"  {key:<40} {s['dice_mean']:>6.4f} {s['dice_TC']:>6.4f} "
            f"{s['dice_WT']:>6.4f} {s['dice_ET']:>6.4f}"
        )

    # Compare original vs remapped for center_crop
    cc_orig = results.get("center_crop__original_labels", {})
    cc_remap = results.get("center_crop__remapped_labels", {})
    if cc_orig and cc_remap:
        delta = cc_remap.get("dice_mean", 0) - cc_orig.get("dice_mean", 0)
        analysis_lines.extend([
            "",
            f"IMPROVEMENT from label remapping (center_crop): +{delta:.4f}",
            "",
            "CONCLUSION: " + (
                "Label remapping dramatically improves Dice, confirming that the "
                "near-zero frozen-GLI Dice was caused by the label encoding mismatch, "
                "NOT by poor model generalization."
                if delta > 0.1
                else "Label remapping had limited effect. Investigate further."
            ),
        ])

    analysis_text = "\n".join(analysis_lines)
    print("\n" + analysis_text)

    analysis_path = OUTPUT_DIR / "analysis.txt"
    with open(analysis_path, "w") as f:
        f.write(analysis_text + "\n")
    logger.info(f"Saved analysis to {analysis_path}")


if __name__ == "__main__":
    main()
