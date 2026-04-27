#!/usr/bin/env python3
"""Evaluate benchmark model predictions against BraTS-MEN ground truth.

Computes per-patient Dice scores for WT/TC/ET regions, generates per-model
summaries, cross-model comparison tables, and paired statistical tests.

Usage:
    python evaluate_benchmark.py --output-dir /path/to/benchmark_segmentation
    python evaluate_benchmark.py --output-dir /path/to/benchmark_segmentation --models BraTS25_1 BraTS23_1
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

LABEL_MAPS = {
    "brats25": {"wt": [1, 2], "tc": [2], "et": [2]},
    "brats23": {"wt": [1, 2, 3], "tc": [1, 3], "et": [3]},
    "ground_truth": {"wt": [1, 2, 3], "tc": [1, 3], "et": [3]},
}

MODEL_LABEL_SCHEME = {
    "BraTS25_1": "brats25",
    "BraTS25_2": "brats25",
    "BraTS23_1": "brats23",
    "BraTS23_2": "brats23",
    "BraTS23_3": "brats23",
}

REGIONS = ["wt", "tc", "et"]


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice similarity coefficient between two binary masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        Binary prediction mask.
    gt_mask : np.ndarray
        Binary ground-truth mask.

    Returns
    -------
    float
        Dice score in [0, 1]. Returns 1.0 if both masks are empty.
    """
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def labels_to_mask(seg: np.ndarray, label_values: list[int]) -> np.ndarray:
    """Convert integer segmentation to binary mask for specified labels.

    Parameters
    ----------
    seg : np.ndarray
        Integer-labeled segmentation volume.
    label_values : list[int]
        Label integers to include in the mask.

    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    mask = np.zeros_like(seg, dtype=bool)
    for lv in label_values:
        mask |= seg == lv
    return mask


def find_prediction_file(pred_dir: Path, scan_id: str) -> Path | None:
    """Locate prediction NIfTI for a scan, handling different naming conventions.

    Parameters
    ----------
    pred_dir : Path
        Model predictions directory.
    scan_id : str
        BraTS scan ID, e.g. ``BraTS-MEN-00032-000``.

    Returns
    -------
    Path | None
        Path to the prediction NIfTI, or None if not found.
    """
    candidates = [
        pred_dir / f"{scan_id}.nii.gz",
        pred_dir / scan_id / "seg.nii.gz",
        pred_dir / scan_id / f"{scan_id}.nii.gz",
    ]
    identifier = "-".join(scan_id.split("-")[-2:])
    candidates.append(pred_dir / f"{identifier}.nii.gz")
    candidates.append(pred_dir / scan_id / f"{identifier}.nii.gz")

    for c in candidates:
        if c.exists():
            return c

    niftis = list(pred_dir.rglob(f"*{identifier}*.nii.gz"))
    if len(niftis) == 1:
        return niftis[0]

    return None


def evaluate_model(
    model_id: str,
    output_dir: Path,
    gt_dir: Path,
    scan_ids: list[str],
) -> pd.DataFrame:
    """Evaluate a single model's predictions against ground truth.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g. ``BraTS25_1``).
    output_dir : Path
        Root benchmark output directory.
    gt_dir : Path
        Ground-truth segmentation directory.
    scan_ids : list[str]
        List of scan IDs to evaluate.

    Returns
    -------
    pd.DataFrame
        Per-patient Dice scores with columns: scan_id, dice_wt, dice_tc, dice_et.
    """
    model_dir = output_dir / "models" / model_id
    pred_dir = model_dir / "predictions"

    if not pred_dir.exists():
        logger.warning("Predictions dir not found for %s: %s", model_id, pred_dir)
        return pd.DataFrame()

    label_scheme = MODEL_LABEL_SCHEME.get(model_id, "brats23")
    pred_label_map = LABEL_MAPS[label_scheme]
    gt_label_map = LABEL_MAPS["ground_truth"]

    records = []
    n_missing = 0

    for scan_id in scan_ids:
        gt_path = gt_dir / scan_id / "seg.nii.gz"
        if not gt_path.exists():
            logger.warning("Ground truth not found: %s", gt_path)
            continue

        pred_path = find_prediction_file(pred_dir, scan_id)
        if pred_path is None:
            logger.warning("Prediction not found for %s in %s", scan_id, pred_dir)
            n_missing += 1
            continue

        gt_seg = np.asarray(nib.load(gt_path).dataobj, dtype=np.int8)
        pred_seg = np.asarray(nib.load(pred_path).dataobj, dtype=np.int8)

        if gt_seg.shape != pred_seg.shape:
            logger.warning(
                "%s shape mismatch: gt=%s, pred=%s — cropping to min",
                scan_id,
                gt_seg.shape,
                pred_seg.shape,
            )
            min_shape = tuple(min(g, p) for g, p in zip(gt_seg.shape, pred_seg.shape))
            gt_seg = gt_seg[: min_shape[0], : min_shape[1], : min_shape[2]]
            pred_seg = pred_seg[: min_shape[0], : min_shape[1], : min_shape[2]]

        row = {"scan_id": scan_id}
        for region in REGIONS:
            gt_mask = labels_to_mask(gt_seg, gt_label_map[region])
            pred_mask = labels_to_mask(pred_seg, pred_label_map[region])
            row[f"dice_{region}"] = dice_score(pred_mask, gt_mask)

        records.append(row)

    if n_missing > 0:
        logger.warning("%s: %d/%d predictions missing", model_id, n_missing, len(scan_ids))

    df = pd.DataFrame(records)
    if not df.empty:
        metrics_dir = model_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(metrics_dir / "per_patient_dice.csv", index=False)

        summary = {}
        for region in REGIONS:
            col = f"dice_{region}"
            vals = df[col].values
            summary[region] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "ci_95_lower": float(np.percentile(vals, 2.5)),
                "ci_95_upper": float(np.percentile(vals, 97.5)),
                "n_evaluated": int(len(vals)),
            }
        summary["n_missing"] = n_missing

        with open(metrics_dir / "summary.json", "w") as fp:
            json.dump(summary, fp, indent=2)

        logger.info(
            "%s: WT=%.3f±%.3f  TC=%.3f±%.3f  ET=%.3f±%.3f  (n=%d)",
            model_id,
            summary["wt"]["mean"],
            summary["wt"]["std"],
            summary["tc"]["mean"],
            summary["tc"]["std"],
            summary["et"]["mean"],
            summary["et"]["std"],
            summary["wt"]["n_evaluated"],
        )

    return df


def generate_comparison(
    all_results: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate cross-model comparison tables and paired statistical tests.

    Parameters
    ----------
    all_results : dict[str, pd.DataFrame]
        Mapping from model_id to per-patient Dice DataFrame.
    output_dir : Path
        Root benchmark output directory.
    """
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_id, df in all_results.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            for region in REGIONS:
                rows.append(
                    {
                        "scan_id": row["scan_id"],
                        "model": model_id,
                        "region": region,
                        "dice": row[f"dice_{region}"],
                    }
                )

    if not rows:
        logger.warning("No results to compare")
        return

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(comp_dir / "model_comparison.csv", index=False)

    summary_rows = []
    for model_id, df in all_results.items():
        if df.empty:
            continue
        row = {"model": model_id}
        for region in REGIONS:
            col = f"dice_{region}"
            vals = df[col].values
            row[f"{region}_mean"] = float(np.mean(vals))
            row[f"{region}_std"] = float(np.std(vals))
            row[f"{region}_median"] = float(np.median(vals))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("et_mean", ascending=False)
    summary_df.to_csv(comp_dir / "model_comparison_summary.csv", index=False)

    with open(comp_dir / "model_comparison_summary.json", "w") as fp:
        json.dump(summary_rows, fp, indent=2)

    logger.info("Model ranking by ET Dice (mean):")
    for _, row in summary_df.iterrows():
        logger.info(
            "  %s: WT=%.3f  TC=%.3f  ET=%.3f",
            row["model"],
            row["wt_mean"],
            row["tc_mean"],
            row["et_mean"],
        )

    model_ids = [m for m in all_results if not all_results[m].empty]
    paired_tests = {}
    for region in REGIONS:
        paired_tests[region] = {}
        for m1, m2 in combinations(model_ids, 2):
            df1 = all_results[m1].set_index("scan_id")
            df2 = all_results[m2].set_index("scan_id")
            common = df1.index.intersection(df2.index)
            if len(common) < 5:
                continue
            v1 = df1.loc[common, f"dice_{region}"].values
            v2 = df2.loc[common, f"dice_{region}"].values
            stat, pval = stats.wilcoxon(v1, v2, alternative="two-sided")
            paired_tests[region][f"{m1}_vs_{m2}"] = {
                "statistic": float(stat),
                "p_value": float(pval),
                "n_common": int(len(common)),
                "mean_diff": float(np.mean(v1 - v2)),
            }

    with open(comp_dir / "paired_tests.json", "w") as fp:
        json.dump(paired_tests, fp, indent=2)

    logger.info("Paired Wilcoxon tests saved to %s", comp_dir / "paired_tests.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate benchmark segmentation predictions")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root benchmark output directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to evaluate (default: all found)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    gt_dir = output_dir / "extraction" / "ground_truth"
    manifest_path = output_dir / "extraction" / "manifest.json"

    if not manifest_path.exists():
        logger.error("Manifest not found: %s. Run extract_h5_to_nifti.py first.", manifest_path)
        return

    with open(manifest_path) as fp:
        manifest = json.load(fp)

    scan_ids = sorted(manifest["scans"].keys())
    logger.info("Manifest: %d patients", len(scan_ids))

    if args.models:
        model_ids = args.models
    else:
        models_dir = output_dir / "models"
        if not models_dir.exists():
            logger.error("Models dir not found: %s", models_dir)
            return
        model_ids = sorted(
            d.name for d in models_dir.iterdir() if d.is_dir() and (d / "predictions").exists()
        )

    if not model_ids:
        logger.error("No models found to evaluate")
        return

    logger.info("Evaluating models: %s", model_ids)

    all_results: dict[str, pd.DataFrame] = {}
    for model_id in model_ids:
        df = evaluate_model(model_id, output_dir, gt_dir, scan_ids)
        all_results[model_id] = df

    generate_comparison(all_results, output_dir)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
