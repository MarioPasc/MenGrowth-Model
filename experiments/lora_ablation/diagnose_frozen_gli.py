#!/usr/bin/env python
# experiments/lora_ablation/diagnose_frozen_gli.py
"""Diagnose frozen BrainSegFounder Dice on BraTS-GLI data.

BraTS-GLI uses labels {0, 2, 4} (no label 1/NCR, ET=4), but _convert_target()
expects {0, 1, 2, 3}. This causes TC and ET channels to be empty for GLI data,
fully explaining the near-zero Dice observed in previous evaluations.

This script tests 4 inference×label conditions to confirm the root cause:
  1. (center_crop, original_labels)  — current pipeline (expected ~0.02 Dice)
  2. (center_crop, remapped_labels)  — remap {4→3} before _convert_target()
  3. (sliding_window, original_labels) — MONAI sliding window inference
  4. (sliding_window, remapped_labels) — combined fix

It also runs additional diagnostics:
  - Per-sample label statistics (unique values, volume fractions)
  - Channel activation analysis (what the model actually predicts)
  - Per-channel target coverage (fraction of non-zero targets after conversion)

Reads paths (checkpoint, glioma_root) from any server config YAML in
experiments/lora_ablation/config/server/.

Usage:
    python experiments/lora_ablation/diagnose_frozen_gli.py \
        --config experiments/lora_ablation/config/server/LoRA_semantic_heads_icai.yaml \
        --output-dir /path/to/results/diagnose_frozen_gli

    # Override defaults:
    python experiments/lora_ablation/diagnose_frozen_gli.py \
        --config experiments/lora_ablation/config/server/DoRA_semantic_heads_icai.yaml \
        --output-dir /tmp/gli_diag \
        --max-subjects 50 --batch-size 1 --num-workers 2
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

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

# ---- Defaults (overridable from CLI) ----
DEFAULT_MAX_SUBJECTS = 200
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 4
DEFAULT_ROI_SIZE = (96, 96, 96)
DEFAULT_SW_BATCH_SIZE = 4
DEFAULT_SW_OVERLAP = 0.5


# =========================================================================
# Label Remapping
# =========================================================================

def remap_gli_labels(seg: torch.Tensor) -> torch.Tensor:
    """Remap BraTS-GLI labels {0, 2, 4} → {0, 2, 3}.

    GLI uses: 0=background, 2=ED, 4=ET.
    BrainSegFounder expects: 0=background, 1=NCR, 2=ED, 3=ET.
    GLI has no NCR (label 1), so we only need to remap 4→3.
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
) -> List[Dict]:
    """Collect per-sample label statistics for the first n subjects.

    Returns list of dicts with subject_id, unique_labels, volume_fractions,
    and spatial extent (bounding box size of non-background voxels).
    """
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
            fractions = {
                int(v): float((s == v).sum()) / total_voxels
                for v in unique_vals
            }
            sid = subject_ids[b] if isinstance(subject_ids, list) else subject_ids

            # Bounding box of non-background
            nonzero = (s > 0).nonzero(as_tuple=False)
            if len(nonzero) > 0:
                bbox_min = nonzero.min(dim=0).values.tolist()
                bbox_max = nonzero.max(dim=0).values.tolist()
                bbox_size = [mx - mn + 1 for mn, mx in zip(bbox_min, bbox_max)]
            else:
                bbox_size = [0, 0, 0]

            entry = {
                "subject_id": sid,
                "unique_labels": unique_vals,
                "volume_fractions": fractions,
                "tumor_bbox_size": bbox_size,
                "tumor_voxels": int((s > 0).sum()),
            }
            stats.append(entry)
            count += 1

    return stats


def log_label_statistics(stats: List[Dict]) -> None:
    """Log collected label statistics."""
    logger.info(f"\nLabel statistics ({len(stats)} subjects):")
    logger.info("-" * 80)
    for entry in stats:
        frac_str = ", ".join(
            f"{k}: {v:.4f}" for k, v in sorted(entry["volume_fractions"].items())
        )
        logger.info(
            f"  {entry['subject_id']}: "
            f"labels={entry['unique_labels']}, "
            f"fractions={{{frac_str}}}, "
            f"bbox={entry['tumor_bbox_size']}, "
            f"tumor_voxels={entry['tumor_voxels']}"
        )


@torch.no_grad()
def analyze_channel_targets(
    dataloader: DataLoader,
    remap: bool,
) -> Dict[str, float]:
    """Analyze what _convert_target produces for GLI data.

    Reports per-channel statistics: what fraction of samples have non-empty
    targets for TC/WT/ET, and mean target volume fraction.
    """
    from growth.losses.segmentation import DiceMetric3Ch
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
) -> Dict[str, float]:
    """Analyze model output activations on GLI data.

    Reports per-channel mean sigmoid activation and prediction coverage
    (fraction of voxels predicted as positive).
    """
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


# =========================================================================
# Evaluation
# =========================================================================

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
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
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
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=sw_overlap,
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
        "dice_TC_median": float(dice_all[:, 0].median()),
        "dice_WT_median": float(dice_all[:, 1].median()),
        "dice_ET_median": float(dice_all[:, 2].median()),
        "n_samples": int(dice_all.shape[0]),
    }


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose frozen BrainSegFounder Dice on BraTS-GLI data"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to a server config YAML (e.g. experiments/lora_ablation/config/server/LoRA_semantic_heads_icai.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save diagnosis results",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=DEFAULT_MAX_SUBJECTS,
        help=f"Maximum GLI subjects to evaluate (default: {DEFAULT_MAX_SUBJECTS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
        help=f"DataLoader workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--roi-size", type=int, nargs=3, default=list(DEFAULT_ROI_SIZE),
        help=f"ROI size for sliding window (default: {list(DEFAULT_ROI_SIZE)})",
    )
    parser.add_argument(
        "--sw-batch-size", type=int, default=DEFAULT_SW_BATCH_SIZE,
        help=f"Sliding window batch size (default: {DEFAULT_SW_BATCH_SIZE})",
    )
    parser.add_argument(
        "--sw-overlap", type=float, default=DEFAULT_SW_OVERLAP,
        help=f"Sliding window overlap (default: {DEFAULT_SW_OVERLAP})",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)",
    )
    args = parser.parse_args()

    # ---- Load config ----
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_path = config["paths"]["checkpoint"]
    glioma_root = config["paths"].get("glioma_root")
    if glioma_root is None:
        raise ValueError(
            f"Config {config_path} does not have paths.glioma_root. "
            "Add it to your config or use a config that has it."
        )

    output_dir = Path(args.output_dir)
    roi_size = tuple(args.roi_size)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"GLI data root: {glioma_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.dump({
            "source_config": str(config_path),
            "checkpoint": checkpoint_path,
            "glioma_root": glioma_root,
            "max_subjects": args.max_subjects,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "roi_size": list(roi_size),
            "sw_batch_size": args.sw_batch_size,
            "sw_overlap": args.sw_overlap,
            "device": device,
        }, f, default_flow_style=False)

    # ---- Load model ----
    logger.info("Loading frozen BrainSegFounder model...")
    model = load_full_swinunetr(
        ckpt_path=checkpoint_path,
        freeze_encoder=True,
        freeze_decoder=True,
        out_channels=3,
        device=device,
    )
    model.eval()

    # ---- Discover GLI subjects ----
    gli_path = Path(glioma_root)
    if not gli_path.exists():
        raise FileNotFoundError(f"GLI data root not found: {gli_path}")

    all_subjects = sorted([
        d.name for d in gli_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    subjects = all_subjects[:args.max_subjects]
    logger.info(
        f"Selected {len(subjects)} subjects "
        f"(from {len(all_subjects)} available, capped at {args.max_subjects})"
    )

    dataset = BraTSGLIDataset(data_root=str(glioma_root), subject_ids=subjects)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ==================================================================
    # Diagnostic 1: Label statistics
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC 1: Label Statistics")
    logger.info("=" * 70)
    label_stats = collect_label_statistics(dataloader, n_subjects=10)
    log_label_statistics(label_stats)

    # Save label stats
    with open(output_dir / "label_statistics.json", "w") as f:
        json.dump(label_stats, f, indent=2, default=str)

    # Aggregate: what label values exist across all sampled subjects?
    all_labels_seen = set()
    for entry in label_stats:
        all_labels_seen.update(int(v) for v in entry["unique_labels"])
    logger.info(f"\n  All unique labels seen: {sorted(all_labels_seen)}")
    has_label_4 = 4 in all_labels_seen
    has_label_3 = 3 in all_labels_seen
    has_label_1 = 1 in all_labels_seen
    logger.info(
        f"  Has label 4 (GLI ET): {has_label_4}, "
        f"Has label 3 (expected ET): {has_label_3}, "
        f"Has label 1 (NCR): {has_label_1}"
    )

    # ==================================================================
    # Diagnostic 2: Channel target coverage (original vs remapped)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC 2: Channel Target Coverage (_convert_target output)")
    logger.info("=" * 70)

    for remap, label in [(False, "original"), (True, "remapped")]:
        coverage = analyze_channel_targets(dataloader, remap=remap)
        logger.info(f"\n  Labels: {label}")
        for ch in ["TC", "WT", "ET"]:
            logger.info(
                f"    {ch}: "
                f"non-empty in {coverage[f'{ch}_nonempty_frac']*100:.1f}% of samples, "
                f"mean vol fraction={coverage[f'{ch}_mean_vol_frac']:.6f}"
            )

        with open(output_dir / f"channel_coverage_{label}.json", "w") as f:
            json.dump(coverage, f, indent=2)

    # ==================================================================
    # Diagnostic 3: Model activation analysis
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC 3: Model Output Activations (first 10 batches)")
    logger.info("=" * 70)

    activations = analyze_model_activations(model, dataloader, device, max_batches=10)
    for ch in ["TC", "WT", "ET"]:
        logger.info(
            f"  {ch}: mean sigmoid={activations[f'{ch}_mean_activation']:.4f}, "
            f"positive (>0.5) frac={activations[f'{ch}_positive_frac']:.6f}"
        )

    with open(output_dir / "model_activations.json", "w") as f:
        json.dump(activations, f, indent=2)

    # ==================================================================
    # Diagnostic 4: Dice evaluation under 4 conditions
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC 4: Dice Evaluation (4 conditions)")
    logger.info("=" * 70)

    dice_metric = DiceMetric3Ch()
    results = {}
    conditions = [
        ("center_crop",     "original_labels", False, False),
        ("center_crop",     "remapped_labels", False, True),
        ("sliding_window",  "original_labels", True,  False),
        ("sliding_window",  "remapped_labels", True,  True),
    ]

    for inference_mode, label_mode, use_sw, use_remap in conditions:
        key = f"{inference_mode}__{label_mode}"
        logger.info(f"\n  Condition: {key}")

        t0 = time.time()
        if use_sw:
            dice_all, summary = evaluate_sliding_window(
                model, dataloader, dice_metric, use_remap, device,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                sw_overlap=args.sw_overlap,
            )
        else:
            dice_all, summary = evaluate_center_crop(
                model, dataloader, dice_metric, use_remap, device,
            )
        elapsed = time.time() - t0

        summary["time_seconds"] = round(elapsed, 1)
        results[key] = summary

        logger.info(
            f"    Dice mean={summary['dice_mean']:.4f}  "
            f"TC={summary['dice_TC']:.4f}+/-{summary['dice_TC_std']:.4f}  "
            f"WT={summary['dice_WT']:.4f}+/-{summary['dice_WT_std']:.4f}  "
            f"ET={summary['dice_ET']:.4f}+/-{summary['dice_ET_std']:.4f}  "
            f"[{elapsed:.1f}s]"
        )

        # Save per-sample Dice
        torch.save(dice_all, output_dir / f"dice_per_sample__{key}.pt")

    # ---- Save results CSV ----
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "dice_mean",
            "dice_TC", "dice_WT", "dice_ET",
            "dice_TC_std", "dice_WT_std", "dice_ET_std",
            "dice_TC_median", "dice_WT_median", "dice_ET_median",
            "n_samples", "time_seconds",
        ])
        for key, s in results.items():
            writer.writerow([
                key, f"{s['dice_mean']:.4f}",
                f"{s['dice_TC']:.4f}", f"{s['dice_WT']:.4f}", f"{s['dice_ET']:.4f}",
                f"{s['dice_TC_std']:.4f}", f"{s['dice_WT_std']:.4f}", f"{s['dice_ET_std']:.4f}",
                f"{s['dice_TC_median']:.4f}", f"{s['dice_WT_median']:.4f}", f"{s['dice_ET_median']:.4f}",
                s["n_samples"], s["time_seconds"],
            ])
    logger.info(f"\nSaved results CSV to {csv_path}")

    # Save full results JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Generate analysis text ----
    analysis_lines = [
        "=" * 70,
        "DIAGNOSIS: Frozen BrainSegFounder on BraTS-GLI",
        "=" * 70,
        "",
        f"Config: {config_path}",
        f"Checkpoint: {checkpoint_path}",
        f"GLI root: {glioma_root}",
        f"Subjects: {len(subjects)}",
        "",
        "HYPOTHESIS:",
        "  BraTS-GLI labels use {0, 2, 4} while _convert_target() expects",
        "  {0, 1, 2, 3}. This makes TC=(label==1|label==3) and ET=(label==3)",
        "  empty for GLI, causing near-zero Dice on TC and ET channels.",
        "",
        "LABEL EVIDENCE:",
        f"  Labels observed in data: {sorted(all_labels_seen)}",
        f"  Has label 4 (GLI ET): {has_label_4}",
        f"  Has label 3 (expected ET): {has_label_3}",
        f"  Has label 1 (NCR): {has_label_1}",
        "",
        "DICE RESULTS:",
        f"  {'Condition':<40} {'Mean':>6} {'TC':>6} {'WT':>6} {'ET':>6}",
        "  " + "-" * 64,
    ]

    for key, s in results.items():
        analysis_lines.append(
            f"  {key:<40} {s['dice_mean']:>6.4f} {s['dice_TC']:>6.4f} "
            f"{s['dice_WT']:>6.4f} {s['dice_ET']:>6.4f}"
        )

    # Deltas
    analysis_lines.append("")
    analysis_lines.append("COMPARISONS:")

    cc_orig = results.get("center_crop__original_labels", {})
    cc_remap = results.get("center_crop__remapped_labels", {})
    sw_orig = results.get("sliding_window__original_labels", {})
    sw_remap = results.get("sliding_window__remapped_labels", {})

    if cc_orig and cc_remap:
        delta = cc_remap.get("dice_mean", 0) - cc_orig.get("dice_mean", 0)
        analysis_lines.append(
            f"  Label remap effect (center_crop): "
            f"+{delta:.4f} mean Dice"
        )
        for ch in ["TC", "WT", "ET"]:
            d = cc_remap.get(f"dice_{ch}", 0) - cc_orig.get(f"dice_{ch}", 0)
            analysis_lines.append(f"    {ch}: +{d:.4f}")

    if cc_remap and sw_remap:
        delta = sw_remap.get("dice_mean", 0) - cc_remap.get("dice_mean", 0)
        analysis_lines.append(
            f"  Sliding window effect (remapped): "
            f"+{delta:.4f} mean Dice"
        )

    # Conclusion
    if cc_orig and cc_remap:
        remap_delta = cc_remap.get("dice_mean", 0) - cc_orig.get("dice_mean", 0)
        analysis_lines.extend([
            "",
            "CONCLUSION:",
        ])
        if remap_delta > 0.1:
            analysis_lines.append(
                "  Label remapping dramatically improves Dice (+{:.4f}), confirming ".format(remap_delta)
                + "that the\n"
                "  near-zero frozen-GLI Dice was caused by the label encoding "
                "mismatch ({0,2,4}\n"
                "  vs expected {0,1,2,3}), NOT by poor model generalization."
            )
        else:
            analysis_lines.append(
                "  Label remapping had limited effect (+{:.4f}). ".format(remap_delta)
                + "Investigate further."
            )

    analysis_text = "\n".join(analysis_lines)
    print("\n" + analysis_text)

    analysis_path = output_dir / "analysis.txt"
    with open(analysis_path, "w") as f:
        f.write(analysis_text + "\n")
    logger.info(f"Saved analysis to {analysis_path}")


if __name__ == "__main__":
    main()
