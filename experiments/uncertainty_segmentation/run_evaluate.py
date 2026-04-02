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
import time
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

    eval_start = time.time()
    summary: dict = {}
    n_members = config.ensemble.n_members

    logger.info(
        f"\n{'=' * 60}\n"
        f"LORA-ENSEMBLE EVALUATION PIPELINE\n"
        f"  Run dir:  {resolved_run_dir}\n"
        f"  Members:  {n_members}\n"
        f"  Device:   {device}\n"
        f"  Steps:    1→Per-member  2→Ensemble  3→Baseline  "
        f"4→Stats  5→Convergence  6→Curves\n"
        f"{'=' * 60}"
    )

    # --- Step 1: Per-member evaluation (R2, R6) ---
    per_member_path = eval_dir / "per_member_test_dice.csv"
    if not skip_per_member:
        step_start = time.time()
        logger.info("=" * 60)
        logger.info("STEP 1/6: Per-member test-set evaluation")
        logger.info("=" * 60)
        per_member_df = evaluate_per_member(
            config, device=device, run_dir=run_dir,
            predictions_dir=test_predictions_dir,
        )
        per_member_df.to_csv(per_member_path, index=False)
        step_time = time.time() - step_start
        logger.info(f"Saved: {per_member_path} ({len(per_member_df)} rows) [{step_time/60:.1f}min]")
        summary["per_member_rows"] = len(per_member_df)
    elif per_member_path.exists():
        per_member_df = pd.read_csv(per_member_path)
        logger.info(f"Loaded cached: {per_member_path}")
    else:
        per_member_df = None
        logger.warning("Per-member evaluation skipped and no cache found")

    # --- Step 2: Ensemble evaluation (per-subject) ---
    ensemble_path = eval_dir / "ensemble_test_dice.csv"
    step_start = time.time()
    logger.info("=" * 60)
    logger.info("STEP 2/6: Ensemble evaluation (per-subject)")
    logger.info("=" * 60)
    ensemble_df, calibration_data = evaluate_ensemble_per_subject(
        config, device=device, run_dir=run_dir, collect_calibration=True,
        predictions_dir=test_predictions_dir,
    )
    ensemble_df.to_csv(ensemble_path, index=False)
    step_time = time.time() - step_start
    logger.info(f"Saved: {ensemble_path} ({len(ensemble_df)} rows) [{step_time/60:.1f}min]")

    summary["ensemble_dice_wt_mean"] = float(ensemble_df["dice_wt"].mean())
    summary["ensemble_dice_wt_std"] = float(ensemble_df["dice_wt"].std())

    # --- Step 3: Baseline evaluation (R3) ---
    if not skip_baseline:
        step_start = time.time()
        logger.info("=" * 60)
        logger.info("STEP 3/6: Baseline evaluation (frozen BrainSegFounder)")
        logger.info("=" * 60)
        baseline_df = evaluate_baseline(
            config, device=device, run_dir=run_dir, force=force_baseline
        )
        step_time = time.time() - step_start
        logger.info(f"  Baseline WT Dice: {baseline_df['dice_wt'].mean():.4f} [{step_time/60:.1f}min]")
        summary["baseline_dice_wt_mean"] = float(baseline_df["dice_wt"].mean())
    elif (eval_dir / "baseline_test_dice.csv").exists():
        baseline_df = pd.read_csv(eval_dir / "baseline_test_dice.csv")
        logger.info(f"Loaded cached baseline: {eval_dir / 'baseline_test_dice.csv'}")
    else:
        baseline_df = None
        logger.warning("Baseline evaluation skipped and no cache found")

    # --- Step 4: Statistical summary (R5, R9) ---
    if not skip_stats and per_member_df is not None and baseline_df is not None:
        step_start = time.time()
        logger.info("=" * 60)
        logger.info("STEP 4/6: Statistical summary + calibration")
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

        # Paired differences CSV (for thesis figures)
        if baseline_df is not None:
            common_scans = set(ensemble_df["scan_id"]) & set(baseline_df["scan_id"])
            if common_scans:
                ens_aligned = ensemble_df[ensemble_df["scan_id"].isin(common_scans)].sort_values("scan_id")
                bas_aligned = baseline_df[baseline_df["scan_id"].isin(common_scans)].sort_values("scan_id")
                paired = pd.DataFrame({
                    "scan_id": ens_aligned["scan_id"].values,
                    "dice_tc_delta": ens_aligned["dice_tc"].values - bas_aligned["dice_tc"].values,
                    "dice_wt_delta": ens_aligned["dice_wt"].values - bas_aligned["dice_wt"].values,
                    "dice_et_delta": ens_aligned["dice_et"].values - bas_aligned["dice_et"].values,
                })
                paired_path = eval_dir / "paired_differences.csv"
                paired.to_csv(paired_path, index=False)
                logger.info(f"Saved: {paired_path} ({len(paired)} scan deltas)")

        summary["statistical_summary"] = stats_summary
    else:
        logger.info("Statistical summary skipped (missing per-member or baseline data)")

    # --- Step 5: Convergence analysis ---
    step_start = time.time()
    logger.info("=" * 60)
    logger.info("STEP 5/6: Convergence analysis")
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

    # Dice convergence from per-member evaluation (all channels)
    if per_member_df is not None:
        for ch in ["dice_wt", "dice_tc", "dice_et"]:
            pivot = per_member_df.pivot_table(
                index="scan_id", columns="member_id", values=ch
            ).dropna()
            if pivot.shape[1] >= 2:
                all_curves = []
                for scan_id, row_data in pivot.iterrows():
                    curve = compute_convergence_curve(row_data.tolist())
                    curve["scan_id"] = scan_id
                    all_curves.append(curve)
                dice_convergence = pd.concat(all_curves, ignore_index=True)
                out_path = eval_dir / f"convergence_{ch}.csv"
                dice_convergence.to_csv(out_path, index=False)
                logger.info(f"Dice convergence ({ch}) saved: {out_path}")

    # --- Step 6: Aggregated training curves ---
    logger.info("=" * 60)
    logger.info("STEP 6/6: Aggregated training curves")
    logger.info("=" * 60)

    adapters_dir = resolved_run_dir / "adapters"
    training_logs = sorted(adapters_dir.glob("member_*/training_log.csv"))
    if training_logs:
        all_logs = []
        for log_csv in training_logs:
            member_id = int(log_csv.parent.name.split("_")[1])
            member_log = pd.read_csv(log_csv)
            member_log["member_id"] = member_id
            all_logs.append(member_log)
        combined = pd.concat(all_logs, ignore_index=True)
        # Aggregate by epoch: mean and std across members
        agg_cols = [c for c in combined.columns if c not in ("epoch", "member_id")]
        agg = combined.groupby("epoch")[agg_cols].agg(["mean", "std"]).reset_index()
        agg.columns = ["epoch"] + [f"{col}_{stat}" for col, stat in agg.columns[1:]]
        agg_path = eval_dir / "aggregated_training_curves.csv"
        agg.to_csv(agg_path, index=False)
        logger.info(f"Aggregated {len(training_logs)} training logs: {agg_path}")
    else:
        logger.info("No training logs found for aggregation")

    total_eval_time = time.time() - eval_start
    logger.info(
        f"\n{'=' * 60}\n"
        f"EVALUATION COMPLETE\n"
        f"  Total time: {total_eval_time / 60:.1f} min ({total_eval_time / 3600:.1f} h)\n"
        f"  Ensemble WT Dice: {summary.get('ensemble_dice_wt_mean', 'N/A')}\n"
        f"  Baseline WT Dice: {summary.get('baseline_dice_wt_mean', 'N/A')}\n"
        f"  Output: {eval_dir}\n"
        f"{'=' * 60}"
    )

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
