#!/usr/bin/env python
# experiments/uncertainty_segmentation/run_evaluate.py
"""CLI entry point: full evaluation of LoRA ensemble on BraTS-MEN test split.

Orchestrates:
    1. Per-member Dice evaluation → per_member_test_dice.csv
    2. Ensemble Dice evaluation → ensemble_test_dice.csv
    3. Baseline (frozen BSF) Dice → baseline_test_dice.csv
    4. Calibration metrics → calibration.json
    5. Statistical summary → statistical_summary.json

Usage:
    python -m experiments.uncertainty_segmentation.run_evaluate \
        --config experiments/uncertainty_segmentation/config.yaml \
        --device cuda:0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from .engine.ensemble_inference import EnsemblePredictor
from .engine.evaluate_baseline import evaluate_baseline
from .engine.evaluate_members import evaluate_ensemble_per_subject, evaluate_per_member
from .engine.paths import get_run_dir
from .engine.statistical_analysis import compute_statistical_summary
from .engine.uncertainty_metrics import (
    compute_brier_score,
    compute_ece,
    compute_reliability_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_full_evaluation(
    config: DictConfig,
    device: str = "cuda",
    run_dir: str | None = None,
    force_baseline: bool = False,
    skip_per_member: bool = False,
    skip_baseline: bool = False,
    skip_stats: bool = False,
) -> dict:
    """Run the full evaluation pipeline.

    Args:
        config: Full experiment configuration.
        device: Inference device.
        run_dir: Override run directory.
        force_baseline: Recompute baseline even if CSV exists.
        skip_per_member: Skip per-member evaluation (expensive).
        skip_baseline: Skip baseline evaluation.
        skip_stats: Skip statistical summary computation.

    Returns:
        Dict with summary of all evaluation results.
    """
    resolved_run_dir = get_run_dir(config, override=run_dir)
    eval_dir = resolved_run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Predictions dir for test-set mask saving
    test_predictions_dir = resolved_run_dir / "predictions" / "brats_men_test"
    test_predictions_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}

    # --- Step 1: Per-member evaluation (R2, R6) ---
    per_member_path = eval_dir / "per_member_test_dice.csv"
    if not skip_per_member:
        logger.info("=" * 60)
        logger.info("STEP 1: Per-member test-set evaluation")
        logger.info("=" * 60)
        per_member_df = evaluate_per_member(
            config, device=device, run_dir=run_dir,
            predictions_dir=test_predictions_dir,
        )
        per_member_df.to_csv(per_member_path, index=False)
        logger.info(f"Saved: {per_member_path} ({len(per_member_df)} rows)")
        summary["per_member_rows"] = len(per_member_df)
    elif per_member_path.exists():
        per_member_df = pd.read_csv(per_member_path)
        logger.info(f"Loaded cached: {per_member_path}")
    else:
        per_member_df = None
        logger.warning("Per-member evaluation skipped and no cache found")

    # --- Step 2: Ensemble evaluation (per-subject) ---
    ensemble_path = eval_dir / "ensemble_test_dice.csv"
    logger.info("=" * 60)
    logger.info("STEP 2: Ensemble evaluation (per-subject)")
    logger.info("=" * 60)
    ensemble_df, calibration_data = evaluate_ensemble_per_subject(
        config, device=device, run_dir=run_dir, collect_calibration=True,
        predictions_dir=test_predictions_dir,
    )
    ensemble_df.to_csv(ensemble_path, index=False)
    logger.info(f"Saved: {ensemble_path} ({len(ensemble_df)} rows)")

    summary["ensemble_dice_wt_mean"] = float(ensemble_df["dice_wt"].mean())
    summary["ensemble_dice_wt_std"] = float(ensemble_df["dice_wt"].std())

    # --- Step 3: Baseline evaluation (R3) ---
    if not skip_baseline:
        logger.info("=" * 60)
        logger.info("STEP 3: Baseline evaluation (frozen BrainSegFounder)")
        logger.info("=" * 60)
        baseline_df = evaluate_baseline(
            config, device=device, run_dir=run_dir, force=force_baseline
        )
        summary["baseline_dice_wt_mean"] = float(baseline_df["dice_wt"].mean())
    elif (eval_dir / "baseline_test_dice.csv").exists():
        baseline_df = pd.read_csv(eval_dir / "baseline_test_dice.csv")
        logger.info(f"Loaded cached baseline: {eval_dir / 'baseline_test_dice.csv'}")
    else:
        baseline_df = None
        logger.warning("Baseline evaluation skipped and no cache found")

    # --- Step 4: Statistical summary (R5, R9) ---
    if not skip_stats and per_member_df is not None and baseline_df is not None:
        logger.info("=" * 60)
        logger.info("STEP 4: Statistical summary")
        logger.info("=" * 60)
        n_bootstrap = config.evaluation.get("n_bootstrap", 10_000)
        alpha = config.evaluation.get("alpha", 0.05)

        stats_summary = compute_statistical_summary(
            per_member_dice=per_member_df,
            ensemble_dice=ensemble_df,
            baseline_dice=baseline_df,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )

        # Calibration from Step 2 data (BUG-5 fix)
        if calibration_data is not None:
            cal_probs = calibration_data["probs"]
            cal_labels = calibration_data["labels"]
            n_bins = config.evaluation.get("ece_n_bins", 15)

            cal_results: dict = {}
            if config.evaluation.get("compute_ece", True):
                cal_results["ece"] = compute_ece(cal_probs, cal_labels, n_bins=n_bins)
                logger.info(f"ECE: {cal_results['ece']:.4f}")
            if config.evaluation.get("compute_brier", True):
                cal_results["brier_score"] = compute_brier_score(cal_probs, cal_labels)
                logger.info(f"Brier: {cal_results['brier_score']:.4f}")
            if config.evaluation.get("compute_reliability", True):
                rel = compute_reliability_data(cal_probs, cal_labels, n_bins=n_bins)
                cal_results["reliability"] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in rel.items()
                }

            # Save calibration separately
            cal_path = eval_dir / "calibration.json"
            with open(cal_path, "w") as f:
                json.dump(cal_results, f, indent=2)
            logger.info(f"Saved: {cal_path}")

            stats_summary["calibration"] = cal_results
        else:
            stats_summary["calibration"] = {}

        stats_path = eval_dir / "statistical_summary.json"
        with open(stats_path, "w") as f:
            json.dump(stats_summary, f, indent=2, default=str)
        logger.info(f"Saved: {stats_path}")

        # Log key results
        if "ensemble_vs_baseline" in stats_summary:
            evb = stats_summary["ensemble_vs_baseline"]
            if "wt" in evb:
                wt = evb["wt"]
                logger.info(
                    f"Ensemble vs Baseline (WT): "
                    f"Δ={wt['delta']:.4f} "
                    f"[{wt['ci_95_lower']:.4f}, {wt['ci_95_upper']:.4f}], "
                    f"p={wt['p_value_wilcoxon']:.4f}, "
                    f"d={wt['cohens_d']:.3f}"
                )

        summary["statistical_summary"] = stats_summary
    else:
        logger.info("Statistical summary skipped (missing per-member or baseline data)")

    # --- Step 5: Convergence analysis ---
    logger.info("=" * 60)
    logger.info("STEP 5: Convergence analysis")
    logger.info("=" * 60)

    from .engine.convergence_analysis import compute_convergence_curve, compute_convergence_summary

    # Volume convergence from existing CSVs
    volume_dir = resolved_run_dir / "volumes"
    for csv_name in ["mengrowth_ensemble_volumes.csv", "men_ensemble_volumes.csv"]:
        vol_csv = volume_dir / csv_name
        if vol_csv.exists():
            convergence_df = compute_convergence_summary(vol_csv)
            if not convergence_df.empty:
                out_path = eval_dir / f"convergence_{csv_name}"
                convergence_df.to_csv(out_path, index=False)
                logger.info(f"Volume convergence saved: {out_path}")

    # Dice convergence from per-member evaluation
    if per_member_df is not None:
        pivot = per_member_df.pivot_table(
            index="scan_id", columns="member_id", values="dice_wt"
        ).dropna()
        if pivot.shape[1] >= 2:
            all_curves = []
            for scan_id, row_data in pivot.iterrows():
                curve = compute_convergence_curve(row_data.tolist())
                curve["scan_id"] = scan_id
                all_curves.append(curve)
            dice_convergence = pd.concat(all_curves, ignore_index=True)
            out_path = eval_dir / "convergence_dice_wt.csv"
            dice_convergence.to_csv(out_path, index=False)
            logger.info(f"Dice convergence saved: {out_path}")

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Full evaluation of LoRA ensemble on BraTS-MEN test split."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config-override", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Override run directory (from SLURM).")
    parser.add_argument("--force-baseline", action="store_true",
                        help="Recompute baseline even if cached.")
    parser.add_argument("--skip-per-member", action="store_true",
                        help="Skip per-member evaluation (use cached if available).")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline evaluation.")
    parser.add_argument("--skip-stats", action="store_true",
                        help="Skip statistical summary.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.config_override:
        config = OmegaConf.merge(config, OmegaConf.load(args.config_override))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    run_full_evaluation(
        config,
        device=device,
        run_dir=args.run_dir,
        force_baseline=args.force_baseline,
        skip_per_member=args.skip_per_member,
        skip_baseline=args.skip_baseline,
        skip_stats=args.skip_stats,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
