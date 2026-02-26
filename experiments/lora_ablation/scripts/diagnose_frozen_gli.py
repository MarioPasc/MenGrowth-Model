#!/usr/bin/env python
# experiments/lora_ablation/diagnose_frozen_gli.py
"""Diagnose frozen BrainSegFounder segmentation on BraTS-GLI and BraTS-MEN.

Runs identical diagnostics on both datasets to identify whether poor GLI Dice
is caused by label encoding issues, domain gap, or pipeline problems.

Diagnostics per dataset:
  1. Label statistics (unique values, volume fractions, bounding boxes)
  2. Channel target coverage after _convert_target() conversion
  3. Model output activations (mean sigmoid, positive fraction)
  4. Spatial analysis (tumor retention after center-crop)
  5. Dice evaluation: center-crop (192^3) + sliding window (full resolution)
     GLI also tests remapped labels.

Reads paths (checkpoint, glioma_root, data_root) from any server config YAML.

Usage:
    python experiments/lora_ablation/diagnose_frozen_gli.py \
        --config experiments/lora_ablation/config/server/LoRA_semantic_heads_icai.yaml \
        --output-dir /path/to/results/diagnose_frozen \
        --dataset both
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.lora_ablation.pipeline.data_splits import load_splits
from experiments.lora_ablation.pipeline.extract_domain_features import BraTSGLIDataset
from growth.data.bratsmendata import BraTSMENDataset
from growth.data.transforms import (
    DEFAULT_ROI_SIZE,
    FEATURE_ROI_SIZE,
    get_sliding_window_transforms,
    get_val_transforms,
)
from growth.inference.sliding_window import sliding_window_segment
from growth.losses.segmentation import DiceMetric3Ch
from growth.models.encoder.swin_loader import load_full_swinunetr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Defaults (overridable from CLI) ----
DEFAULT_MAX_SUBJECTS = 200
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 4


# =========================================================================
# Label Remapping
# =========================================================================


def remap_gli_labels(seg: torch.Tensor) -> torch.Tensor:
    """Remap BraTS-GLI labels {0, 2, 4} -> {0, 2, 3}.

    GLI uses: 0=background, 2=ED, 4=ET.
    BrainSegFounder expects: 0=background, 1=NCR, 2=ED, 3=ET.
    GLI has no NCR (label 1), so we only need to remap 4->3.
    """
    seg = seg.clone()
    seg[seg == 4] = 3
    return seg


# =========================================================================
# Diagnostics
# =========================================================================


def collect_label_statistics(
    dataloader: DataLoader,
    n_subjects: int = 10,
) -> list[dict]:
    """Collect per-sample label statistics for the first n subjects."""
    stats = []
    count = 0

    for batch in dataloader:
        seg = batch["seg"]  # [B, 1, D, H, W]
        subject_ids = batch["subject_id"]

        for b in range(seg.shape[0]):
            if count >= n_subjects:
                return stats

            s = seg[b].squeeze(0)  # [D, H, W]
            unique_vals = sorted(torch.unique(s).tolist())
            total_voxels = s.numel()
            fractions = {int(v): float((s == v).sum()) / total_voxels for v in unique_vals}
            sid = subject_ids[b] if isinstance(subject_ids, list) else subject_ids

            # Bounding box of non-background
            nonzero = (s > 0).nonzero(as_tuple=False)
            if len(nonzero) > 0:
                bbox_min = nonzero.min(dim=0).values.tolist()
                bbox_max = nonzero.max(dim=0).values.tolist()
                bbox_size = [int(mx - mn + 1) for mn, mx in zip(bbox_min, bbox_max)]
            else:
                bbox_size = [0, 0, 0]

            entry = {
                "subject_id": sid,
                "unique_labels": unique_vals,
                "volume_fractions": fractions,
                "spatial_size": list(s.shape),
                "tumor_bbox_size": bbox_size,
                "tumor_voxels": int((s > 0).sum()),
            }
            stats.append(entry)
            count += 1

    return stats


def log_label_statistics(stats: list[dict], dataset_name: str) -> None:
    """Log collected label statistics."""
    logger.info(f"\n  Label statistics ({len(stats)} {dataset_name} subjects):")
    logger.info("  " + "-" * 78)
    for entry in stats:
        frac_str = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(entry["volume_fractions"].items()))
        logger.info(
            f"    {entry['subject_id']}: "
            f"labels={entry['unique_labels']}, "
            f"spatial={entry['spatial_size']}, "
            f"bbox={entry['tumor_bbox_size']}, "
            f"tumor={entry['tumor_voxels']}"
        )
        logger.info(f"      fractions={{{frac_str}}}")


@torch.no_grad()
def analyze_channel_targets(
    dataloader: DataLoader,
    remap: bool = False,
) -> dict[str, float]:
    """Analyze what _convert_target produces for data."""
    metric = DiceMetric3Ch()

    ch_names = ["TC", "WT", "ET"]
    nonempty_counts = [0, 0, 0]
    vol_fraction_sums = [0.0, 0.0, 0.0]
    total_samples = 0

    for batch in dataloader:
        seg = batch["seg"]
        if remap:
            seg = remap_gli_labels(seg)

        target_3ch = metric._convert_target(seg)  # [B, 3, D, H, W]

        for b in range(target_3ch.shape[0]):
            total_voxels = target_3ch.shape[2] * target_3ch.shape[3] * target_3ch.shape[4]
            for c in range(3):
                ch_sum = target_3ch[b, c].sum().item()
                if ch_sum > 0:
                    nonempty_counts[c] += 1
                vol_fraction_sums[c] += ch_sum / total_voxels
            total_samples += 1

    result = {"total_samples": total_samples}
    for c, name in enumerate(ch_names):
        result[f"{name}_nonempty_frac"] = nonempty_counts[c] / max(total_samples, 1)
        result[f"{name}_mean_vol_frac"] = vol_fraction_sums[c] / max(total_samples, 1)

    return result


@torch.no_grad()
def analyze_model_activations(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_batches: int = 10,
) -> dict[str, float]:
    """Analyze model output activations."""
    model.eval()
    ch_names = ["TC", "WT", "ET"]
    activation_sums = [0.0, 0.0, 0.0]
    positive_frac_sums = [0.0, 0.0, 0.0]
    total_samples = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        images = batch["image"].to(device)
        pred = model(images)
        pred_prob = torch.sigmoid(pred)  # [B, 3, D, H, W]

        for b in range(pred_prob.shape[0]):
            total_voxels = pred_prob.shape[2] * pred_prob.shape[3] * pred_prob.shape[4]
            for c in range(3):
                activation_sums[c] += pred_prob[b, c].mean().item()
                positive_frac_sums[c] += (pred_prob[b, c] > 0.5).float().sum().item() / total_voxels
            total_samples += 1

    result = {}
    for c, name in enumerate(ch_names):
        result[f"{name}_mean_activation"] = activation_sums[c] / max(total_samples, 1)
        result[f"{name}_positive_frac"] = positive_frac_sums[c] / max(total_samples, 1)

    return result


def analyze_spatial_coverage(
    dataloader: DataLoader,
) -> dict[str, float]:
    """Analyze tumor bounding boxes and edge-touching fraction."""
    bbox_sizes_all = []
    exceeds_roi = 0
    total_samples = 0
    tumor_voxel_fracs = []

    for batch in dataloader:
        seg = batch["seg"]  # [B, 1, D, H, W]
        for b in range(seg.shape[0]):
            s = seg[b].squeeze(0)  # [D, H, W]
            nonzero = (s > 0).nonzero(as_tuple=False)

            if len(nonzero) > 0:
                bbox_min = nonzero.min(dim=0).values
                bbox_max = nonzero.max(dim=0).values
                bbox_size = (bbox_max - bbox_min + 1).tolist()

                touches_edge = False
                for dim_idx in range(3):
                    if bbox_min[dim_idx] == 0 or bbox_max[dim_idx] == s.shape[dim_idx] - 1:
                        touches_edge = True
                        break

                if touches_edge:
                    exceeds_roi += 1

                bbox_sizes_all.append(bbox_size)
                tumor_voxel_fracs.append(float((s > 0).sum()) / s.numel())
            else:
                bbox_sizes_all.append([0, 0, 0])
                tumor_voxel_fracs.append(0.0)

            total_samples += 1

    bbox_arr = np.array(bbox_sizes_all)

    return {
        "total_samples": total_samples,
        "mean_bbox_D": float(bbox_arr[:, 0].mean()),
        "mean_bbox_H": float(bbox_arr[:, 1].mean()),
        "mean_bbox_W": float(bbox_arr[:, 2].mean()),
        "max_bbox_D": int(bbox_arr[:, 0].max()),
        "max_bbox_H": int(bbox_arr[:, 1].max()),
        "max_bbox_W": int(bbox_arr[:, 2].max()),
        "bbox_touches_edge_frac": exceeds_roi / max(total_samples, 1),
        "mean_tumor_vol_frac": float(np.mean(tumor_voxel_fracs)),
        "median_tumor_vol_frac": float(np.median(tumor_voxel_fracs)),
    }


# =========================================================================
# Evaluation
# =========================================================================


@torch.no_grad()
def evaluate_dice(
    model: nn.Module,
    dataloader: DataLoader,
    dice_metric: DiceMetric3Ch,
    device: str,
    remap: bool = False,
    desc: str = "Evaluating",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate Dice with center-crop patches."""
    model.eval()
    all_dice = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
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
def evaluate_dice_sliding_window(
    model: nn.Module,
    dataloader: DataLoader,
    dice_metric: DiceMetric3Ch,
    device: str,
    roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    remap: bool = False,
    desc: str = "SW Eval",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate Dice using sliding window on full-resolution volumes.

    Requires batch_size=1 dataloader with get_sliding_window_transforms.
    """
    model.eval()
    all_dice = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        images = batch["image"].to(device)  # [1, 4, D, H, W]
        segs = batch["seg"].to(device)  # [1, 1, D, H, W]

        if remap:
            segs = remap_gli_labels(segs)

        pred = sliding_window_segment(
            model,
            images,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
        )
        dice = dice_metric(pred, segs)  # [1, 3]
        all_dice.append(dice.cpu())

    dice_all = torch.cat(all_dice, dim=0)  # [N, 3]
    return dice_all, _summarize(dice_all)


def _summarize(dice_all: torch.Tensor) -> dict[str, float]:
    """Summarize per-sample Dice tensor [N, 3] -> dict."""
    return {
        "dice_mean": float(dice_all.mean()),
        "dice_TC": float(dice_all[:, 0].mean()),
        "dice_WT": float(dice_all[:, 1].mean()),
        "dice_ET": float(dice_all[:, 2].mean()),
        "dice_TC_std": float(dice_all[:, 0].std()),
        "dice_WT_std": float(dice_all[:, 1].std()),
        "dice_ET_std": float(dice_all[:, 2].std()),
        "dice_TC_median": float(dice_all[:, 0].median()),
        "dice_WT_median": float(dice_all[:, 1].median()),
        "dice_ET_median": float(dice_all[:, 2].median()),
        "n_samples": int(dice_all.shape[0]),
    }


# =========================================================================
# Per-dataset diagnostic runner
# =========================================================================


def run_diagnostics(
    model: nn.Module,
    dataloader: DataLoader,
    sw_dataloader: DataLoader | None,
    dataset_name: str,
    out_dir: Path,
    device: str,
    is_gli: bool,
    roi_size: tuple[int, int, int] = (128, 128, 128),
) -> dict:
    """Run all diagnostics for a single dataset.

    Args:
        model: Frozen BrainSegFounder model.
        dataloader: DataLoader with center-crop transforms.
        sw_dataloader: DataLoader with sliding window transforms (batch=1).
        dataset_name: Display name (e.g. "GLI", "MEN").
        out_dir: Output subdirectory for this dataset.
        device: Device string.
        is_gli: True for GLI (enables label remapping conditions).
        roi_size: ROI size for sliding window patches.

    Returns:
        Dict with all diagnostic results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ==================================================================
    # Diagnostic 1: Label Statistics
    # ==================================================================
    logger.info(f"\n  {'=' * 66}")
    logger.info(f"  DIAGNOSTIC 1: Label Statistics ({dataset_name})")
    logger.info(f"  {'=' * 66}")

    label_stats = collect_label_statistics(dataloader, n_subjects=10)
    log_label_statistics(label_stats, dataset_name)

    with open(out_dir / "label_statistics.json", "w") as f:
        json.dump(label_stats, f, indent=2, default=str)

    all_labels_seen = set()
    for entry in label_stats:
        all_labels_seen.update(int(v) for v in entry["unique_labels"])

    logger.info(f"\n    All unique labels seen: {sorted(all_labels_seen)}")
    all_results["labels_seen"] = sorted(all_labels_seen)

    # ==================================================================
    # Diagnostic 2: Channel Target Coverage
    # ==================================================================
    logger.info(f"\n  {'=' * 66}")
    logger.info(f"  DIAGNOSTIC 2: Channel Target Coverage ({dataset_name})")
    logger.info(f"  {'=' * 66}")

    coverage_modes = [("standard", False)]
    if is_gli:
        coverage_modes.append(("remapped_4to3", True))

    all_results["channel_coverage"] = {}
    for label, remap in coverage_modes:
        coverage = analyze_channel_targets(dataloader, remap=remap)
        all_results["channel_coverage"][label] = coverage

        logger.info(f"\n    Labels mode: {label}")
        for ch in ["TC", "WT", "ET"]:
            logger.info(
                f"      {ch}: "
                f"non-empty in {coverage[f'{ch}_nonempty_frac'] * 100:.1f}% of samples, "
                f"mean vol fraction={coverage[f'{ch}_mean_vol_frac']:.6f}"
            )

        with open(out_dir / f"channel_coverage_{label}.json", "w") as f:
            json.dump(coverage, f, indent=2)

    # ==================================================================
    # Diagnostic 3: Model Activations
    # ==================================================================
    logger.info(f"\n  {'=' * 66}")
    logger.info(f"  DIAGNOSTIC 3: Model Activations ({dataset_name}, first 10 batches)")
    logger.info(f"  {'=' * 66}")

    activations = analyze_model_activations(model, dataloader, device, max_batches=10)
    all_results["activations"] = activations

    for ch in ["TC", "WT", "ET"]:
        logger.info(
            f"    {ch}: mean sigmoid={activations[f'{ch}_mean_activation']:.4f}, "
            f"positive (>0.5) frac={activations[f'{ch}_positive_frac']:.6f}"
        )

    with open(out_dir / "model_activations.json", "w") as f:
        json.dump(activations, f, indent=2)

    # ==================================================================
    # Diagnostic 4: Spatial / Tumor Retention Analysis
    # ==================================================================
    logger.info(f"\n  {'=' * 66}")
    logger.info(f"  DIAGNOSTIC 4: Spatial Analysis ({dataset_name})")
    logger.info(f"  {'=' * 66}")

    spatial = analyze_spatial_coverage(dataloader)
    all_results["spatial"] = spatial

    logger.info(
        f"    Mean tumor bbox (D,H,W): "
        f"({spatial['mean_bbox_D']:.1f}, {spatial['mean_bbox_H']:.1f}, {spatial['mean_bbox_W']:.1f})"
    )
    logger.info(
        f"    Max tumor bbox (D,H,W): "
        f"({spatial['max_bbox_D']}, {spatial['max_bbox_H']}, {spatial['max_bbox_W']})"
    )
    logger.info(
        f"    Tumor touches edge: {spatial['bbox_touches_edge_frac'] * 100:.1f}% of samples"
    )
    logger.info(f"    Mean tumor vol fraction: {spatial['mean_tumor_vol_frac']:.6f}")
    logger.info(f"    Median tumor vol fraction: {spatial['median_tumor_vol_frac']:.6f}")

    with open(out_dir / "spatial_analysis.json", "w") as f:
        json.dump(spatial, f, indent=2)

    # ==================================================================
    # Diagnostic 5: Dice Evaluation
    # ==================================================================
    logger.info(f"\n  {'=' * 66}")
    logger.info(f"  DIAGNOSTIC 5: Dice Evaluation ({dataset_name})")
    logger.info(f"  {'=' * 66}")

    dice_metric = DiceMetric3Ch()
    dice_results = {}

    # --- Center-crop conditions ---
    conditions = [("center_crop", False)]
    if is_gli:
        conditions.append(("center_crop_remapped", True))

    for label, remap in conditions:
        t0 = time.time()
        dice_all, summary = evaluate_dice(
            model,
            dataloader,
            dice_metric,
            device,
            remap=remap,
            desc=f"{dataset_name} Dice ({label})",
        )
        elapsed = time.time() - t0
        summary["time_seconds"] = round(elapsed, 1)
        dice_results[label] = summary

        logger.info(
            f"    [{label}] "
            f"Dice mean={summary['dice_mean']:.4f}  "
            f"TC={summary['dice_TC']:.4f}+/-{summary['dice_TC_std']:.4f}  "
            f"WT={summary['dice_WT']:.4f}+/-{summary['dice_WT_std']:.4f}  "
            f"ET={summary['dice_ET']:.4f}+/-{summary['dice_ET_std']:.4f}  "
            f"[{elapsed:.1f}s]"
        )

        torch.save(dice_all, out_dir / f"dice_per_sample__{label}.pt")

    # --- Sliding window conditions ---
    if sw_dataloader is not None:
        sw_conditions = [("sliding_window", False)]
        if is_gli:
            sw_conditions.append(("sliding_window_remapped", True))

        for label, remap in sw_conditions:
            t0 = time.time()
            dice_all, summary = evaluate_dice_sliding_window(
                model,
                sw_dataloader,
                dice_metric,
                device,
                roi_size=roi_size,
                remap=remap,
                desc=f"{dataset_name} SW Dice ({label})",
            )
            elapsed = time.time() - t0
            summary["time_seconds"] = round(elapsed, 1)
            dice_results[label] = summary

            logger.info(
                f"    [{label}] "
                f"Dice mean={summary['dice_mean']:.4f}  "
                f"TC={summary['dice_TC']:.4f}+/-{summary['dice_TC_std']:.4f}  "
                f"WT={summary['dice_WT']:.4f}+/-{summary['dice_WT_std']:.4f}  "
                f"ET={summary['dice_ET']:.4f}+/-{summary['dice_ET_std']:.4f}  "
                f"[{elapsed:.1f}s]"
            )

            torch.save(dice_all, out_dir / f"dice_per_sample__{label}.pt")

    all_results["dice"] = dice_results

    # Save combined results
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# =========================================================================
# Comparative Analysis
# =========================================================================


def generate_comparison(
    gli_results: dict | None,
    men_results: dict | None,
    output_dir: Path,
    config_path: str,
    checkpoint_path: str,
) -> str:
    """Generate comparative analysis text and CSV."""

    lines = [
        "=" * 72,
        "DIAGNOSIS: Frozen BrainSegFounder — GLI vs MEN Comparison",
        "=" * 72,
        "",
        f"Config: {config_path}",
        f"Checkpoint: {checkpoint_path}",
        "",
    ]

    # ---- Labels ----
    lines.append("1. LABEL ENCODING")
    lines.append("-" * 50)
    if gli_results:
        lines.append(f"   GLI labels seen: {gli_results.get('labels_seen', '?')}")
    if men_results:
        lines.append(f"   MEN labels seen: {men_results.get('labels_seen', '?')}")
    lines.append("")

    # ---- Channel targets ----
    lines.append("2. CHANNEL TARGET COVERAGE (after _convert_target)")
    lines.append("-" * 50)
    header = f"   {'Dataset':<8} {'Mode':<18} {'TC %':>7} {'WT %':>7} {'ET %':>7}   {'TC vol':>10} {'WT vol':>10} {'ET vol':>10}"
    lines.append(header)

    for name, results in [("GLI", gli_results), ("MEN", men_results)]:
        if results is None:
            continue
        for mode, cov in results.get("channel_coverage", {}).items():
            lines.append(
                f"   {name:<8} {mode:<18} "
                f"{cov['TC_nonempty_frac'] * 100:>6.1f}% "
                f"{cov['WT_nonempty_frac'] * 100:>6.1f}% "
                f"{cov['ET_nonempty_frac'] * 100:>6.1f}%   "
                f"{cov['TC_mean_vol_frac']:>10.6f} "
                f"{cov['WT_mean_vol_frac']:>10.6f} "
                f"{cov['ET_mean_vol_frac']:>10.6f}"
            )
    lines.append("")

    # ---- Activations ----
    lines.append("3. MODEL OUTPUT ACTIVATIONS")
    lines.append("-" * 50)
    lines.append(
        f"   {'Dataset':<8} {'TC sig':>8} {'WT sig':>8} {'ET sig':>8}   {'TC pos%':>8} {'WT pos%':>8} {'ET pos%':>8}"
    )

    for name, results in [("GLI", gli_results), ("MEN", men_results)]:
        if results is None:
            continue
        act = results.get("activations", {})
        lines.append(
            f"   {name:<8} "
            f"{act.get('TC_mean_activation', 0):>8.4f} "
            f"{act.get('WT_mean_activation', 0):>8.4f} "
            f"{act.get('ET_mean_activation', 0):>8.4f}   "
            f"{act.get('TC_positive_frac', 0) * 100:>7.4f}% "
            f"{act.get('WT_positive_frac', 0) * 100:>7.4f}% "
            f"{act.get('ET_positive_frac', 0) * 100:>7.4f}%"
        )
    lines.append("")

    # ---- Spatial ----
    lines.append("4. SPATIAL ANALYSIS")
    lines.append("-" * 50)
    lines.append(
        f"   {'Dataset':<8} {'Mean bbox':>20} {'Max bbox':>20} {'Edge %':>8} {'Tumor vol%':>10}"
    )

    for name, results in [("GLI", gli_results), ("MEN", men_results)]:
        if results is None:
            continue
        sp = results.get("spatial", {})
        mean_bbox = f"({sp.get('mean_bbox_D', 0):.0f},{sp.get('mean_bbox_H', 0):.0f},{sp.get('mean_bbox_W', 0):.0f})"
        max_bbox = (
            f"({sp.get('max_bbox_D', 0)},{sp.get('max_bbox_H', 0)},{sp.get('max_bbox_W', 0)})"
        )
        lines.append(
            f"   {name:<8} {mean_bbox:>20} {max_bbox:>20} "
            f"{sp.get('bbox_touches_edge_frac', 0) * 100:>7.1f}% "
            f"{sp.get('mean_tumor_vol_frac', 0) * 100:>9.4f}%"
        )
    lines.append("")

    # ---- Dice ----
    lines.append("5. DICE SCORES")
    lines.append("-" * 50)
    lines.append(f"   {'Dataset':<8} {'Mode':<22} {'Mean':>7} {'TC':>7} {'WT':>7} {'ET':>7}")

    csv_rows = []
    for name, results in [("GLI", gli_results), ("MEN", men_results)]:
        if results is None:
            continue
        for mode, d in results.get("dice", {}).items():
            lines.append(
                f"   {name:<8} {mode:<22} "
                f"{d['dice_mean']:>7.4f} "
                f"{d['dice_TC']:>7.4f} "
                f"{d['dice_WT']:>7.4f} "
                f"{d['dice_ET']:>7.4f}"
            )
            csv_rows.append(
                {
                    "dataset": name,
                    "label_mode": mode,
                    **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in d.items()},
                }
            )
    lines.append("")

    # ---- Interpretation ----
    lines.append("6. INTERPRETATION")
    lines.append("-" * 50)

    if gli_results and men_results:
        # Compare sliding window vs center crop
        gli_sw = gli_results.get("dice", {}).get("sliding_window", {})
        gli_cc = gli_results.get("dice", {}).get("center_crop", {})
        men_sw = men_results.get("dice", {}).get("sliding_window", {})
        men_cc = men_results.get("dice", {}).get("center_crop", {})

        if gli_sw and gli_cc:
            sw_delta = gli_sw.get("dice_mean", 0) - gli_cc.get("dice_mean", 0)
            lines.append(f"   GLI sliding window vs center crop: {sw_delta:+.4f} Dice")
        if men_sw and men_cc:
            sw_delta = men_sw.get("dice_mean", 0) - men_cc.get("dice_mean", 0)
            lines.append(f"   MEN sliding window vs center crop: {sw_delta:+.4f} Dice")

        # Domain gap assessment
        gli_best = gli_sw.get("dice_mean", 0) if gli_sw else gli_cc.get("dice_mean", 0)
        men_best = men_sw.get("dice_mean", 0) if men_sw else men_cc.get("dice_mean", 0)

        if gli_best > 0.5 and men_best < gli_best:
            gap = gli_best - men_best
            lines.append(f"\n   Domain gap (GLI -> MEN): {gap:.4f} Dice drop")
            lines.append("   -> LoRA adaptation targets this gap.")

        # Label remapping
        gli_std = gli_results.get("dice", {}).get("sliding_window", {}).get("dice_mean", 0)
        gli_remap = (
            gli_results.get("dice", {}).get("sliding_window_remapped", {}).get("dice_mean", 0)
        )
        if gli_std and gli_remap:
            remap_delta = gli_remap - gli_std
            if abs(remap_delta) > 0.01:
                lines.append(f"\n   GLI label remapping effect (SW): {remap_delta:+.4f}")

    lines.append("")

    analysis_text = "\n".join(lines)

    # Save analysis
    analysis_path = output_dir / "analysis.txt"
    with open(analysis_path, "w") as f:
        f.write(analysis_text + "\n")

    # Save comparison CSV
    if csv_rows:
        csv_path = output_dir / "dice_comparison.csv"
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    return analysis_text


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose frozen BrainSegFounder segmentation on GLI and/or MEN"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a server config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save diagnosis results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["gli", "men", "both"],
        help="Which dataset(s) to diagnose (default: both)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=DEFAULT_MAX_SUBJECTS,
        help=f"Max subjects per dataset (default: {DEFAULT_MAX_SUBJECTS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for center-crop eval (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"DataLoader workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="Skip sliding window evaluation (faster, center-crop only)",
    )
    parser.add_argument(
        "--sw-max-subjects",
        type=int,
        default=50,
        help="Max subjects for sliding window eval (default: 50, slower than center-crop)",
    )
    args = parser.parse_args()

    # ---- Load config ----
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_path = config["paths"]["checkpoint"]
    data_root = config["paths"]["data_root"]
    glioma_root = config["paths"].get("glioma_root")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    run_gli = args.dataset in ("gli", "both")
    run_men = args.dataset in ("men", "both")
    run_sw = not args.no_sliding_window

    if run_gli and glioma_root is None:
        raise ValueError(
            f"Config {config_path} has no paths.glioma_root but --dataset includes 'gli'"
        )

    roi_size = tuple(DEFAULT_ROI_SIZE)

    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Datasets: {args.dataset}")
    logger.info(f"ROI size: {roi_size}")
    logger.info(f"Sliding window: {'yes' if run_sw else 'no'}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")

    # Save resolved config for reproducibility
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(
            {
                "source_config": str(config_path),
                "checkpoint": checkpoint_path,
                "data_root": data_root,
                "glioma_root": glioma_root,
                "dataset": args.dataset,
                "roi_size": list(roi_size),
                "sliding_window": run_sw,
                "sw_max_subjects": args.sw_max_subjects,
                "max_subjects": args.max_subjects,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "device": device,
            },
            f,
            default_flow_style=False,
        )

    # ---- Load model (once, shared) ----
    logger.info("\nLoading frozen BrainSegFounder model...")
    model = load_full_swinunetr(
        ckpt_path=checkpoint_path,
        freeze_encoder=True,
        freeze_decoder=True,
        out_channels=3,
        device=device,
    )
    model.eval()

    # ---- GLI diagnostics ----
    gli_results = None
    if run_gli:
        logger.info("\n" + "#" * 72)
        logger.info("# BraTS-GLI DIAGNOSTICS")
        logger.info("#" * 72)

        gli_path = Path(glioma_root)
        if not gli_path.exists():
            raise FileNotFoundError(f"GLI data not found: {gli_path}")

        all_gli = sorted(
            [d.name for d in gli_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )
        gli_subjects = all_gli[: args.max_subjects]
        logger.info(f"GLI subjects: {len(gli_subjects)} (from {len(all_gli)} available)")

        # Center-crop dataloader (192³ for full tumor containment)
        gli_dataset = BraTSGLIDataset(
            data_root=str(glioma_root),
            subject_ids=gli_subjects,
            transform=get_val_transforms(roi_size=FEATURE_ROI_SIZE),
        )
        gli_loader = DataLoader(
            gli_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Sliding window dataloader (subset, batch_size=1)
        sw_gli_loader = None
        if run_sw:
            sw_gli_subjects = gli_subjects[: args.sw_max_subjects]
            sw_gli_dataset = BraTSGLIDataset(
                data_root=str(glioma_root),
                subject_ids=sw_gli_subjects,
                transform=get_sliding_window_transforms(),
            )
            sw_gli_loader = DataLoader(
                sw_gli_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

        gli_results = run_diagnostics(
            model,
            gli_loader,
            sw_gli_loader,
            "GLI",
            output_dir / "gli",
            device,
            is_gli=True,
            roi_size=roi_size,
        )

    # ---- MEN diagnostics ----
    men_results = None
    if run_men:
        logger.info("\n" + "#" * 72)
        logger.info("# BraTS-MEN DIAGNOSTICS")
        logger.info("#" * 72)

        # Load test split subjects
        try:
            splits = load_splits(str(config_path))
            men_subjects = splits["test"][: args.max_subjects]
            logger.info(f"MEN subjects: {len(men_subjects)} (from test split)")
        except (FileNotFoundError, KeyError):
            logger.warning("No data_splits.json found, discovering subjects from data_root")
            men_path = Path(data_root)
            all_men = sorted(
                [d.name for d in men_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
            )
            men_subjects = all_men[: args.max_subjects]
            logger.info(f"MEN subjects: {len(men_subjects)} (discovered from {data_root})")

        # Center-crop dataloader (192³ for full tumor containment)
        h5_path = config.get("paths", {}).get("h5_file")
        if h5_path:
            from growth.data.bratsmendata import BraTSMENDatasetH5
            from growth.data.transforms import get_h5_val_transforms

            men_dataset = BraTSMENDatasetH5(
                h5_path=h5_path,
                split="test",
                transform=get_h5_val_transforms(roi_size=FEATURE_ROI_SIZE),
                compute_semantic=False,
            )
        else:
            men_dataset = BraTSMENDataset(
                data_root=data_root,
                subject_ids=men_subjects,
                compute_semantic=False,
                transform=get_val_transforms(roi_size=FEATURE_ROI_SIZE),
            )
        men_loader = DataLoader(
            men_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Sliding window dataloader (subset, batch_size=1)
        sw_men_loader = None
        if run_sw:
            sw_men_subjects = men_subjects[: args.sw_max_subjects]
            sw_men_dataset = BraTSMENDataset(
                data_root=data_root,
                subject_ids=sw_men_subjects,
                compute_semantic=False,
                transform=get_sliding_window_transforms(),
            )
            sw_men_loader = DataLoader(
                sw_men_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

        men_results = run_diagnostics(
            model,
            men_loader,
            sw_men_loader,
            "MEN",
            output_dir / "men",
            device,
            is_gli=False,
            roi_size=roi_size,
        )

    # ---- Comparative analysis ----
    logger.info("\n" + "#" * 72)
    logger.info("# COMPARATIVE ANALYSIS")
    logger.info("#" * 72)

    analysis = generate_comparison(
        gli_results,
        men_results,
        output_dir,
        str(config_path),
        checkpoint_path,
    )
    print("\n" + analysis)

    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
