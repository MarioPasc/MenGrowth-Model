# experiments/segment_based_approach/run_baseline.py
"""Ablation A0: Segment-Based Baseline — main orchestrator.

Usage:
    python -m experiments.segment_based_approach.run_baseline \
        --config experiments/segment_based_approach/config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from growth.evaluation.lopo_evaluator import LOPOEvaluator
from growth.models.growth.scalar_gp import ScalarGP

from .segment import ScanVolumes, SegmentationVolumeExtractor

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_results(
    results_manual: object,
    results_predicted: object,
    volumes: list[ScanVolumes],
    output_dir: Path,
) -> None:
    """Save all results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "lopo_results_manual.json", "w") as f:
        json.dump(results_manual.to_dict(), f, indent=2)

    with open(output_dir / "lopo_results_predicted.json", "w") as f:
        json.dump(results_predicted.to_dict(), f, indent=2)

    # Summary of volumes
    vol_summary = {
        "n_scans": len(volumes),
        "n_empty_manual": sum(1 for v in volumes if v.is_empty_manual),
        "n_empty_predicted": sum(1 for v in volumes if v.is_empty_predicted),
        "mean_dice": float(np.mean([v.wt_dice for v in volumes])),
        "median_dice": float(np.median([v.wt_dice for v in volumes])),
    }
    with open(output_dir / "volume_summary.json", "w") as f:
        json.dump(vol_summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def generate_figures(
    results_manual: object,
    results_predicted: object,
    volumes: list[ScanVolumes],
    output_dir: Path,
) -> None:
    """Generate all baseline figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _fig_volume_scatter(volumes, fig_dir)
    _fig_lopo_scatter(results_manual, results_predicted, fig_dir)
    _fig_calibration(results_manual, results_predicted, fig_dir)
    _fig_dice_distribution(volumes, fig_dir)

    logger.info(f"Figures saved to {fig_dir}")


def _fig_volume_scatter(volumes: list[ScanVolumes], fig_dir: Path) -> None:
    """Figure 1: Manual vs predicted volume scatter."""
    manual = np.array([v.manual_vol_mm3 for v in volumes if not v.is_empty_manual])
    predicted = np.array([v.predicted_vol_mm3 for v in volumes if not v.is_empty_manual])
    dices = np.array([v.wt_dice for v in volumes if not v.is_empty_manual])

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(manual, predicted, c=dices, cmap="viridis", alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax, label="WT Dice")

    lim = max(manual.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Identity")
    ax.set_xlabel("Manual WT Volume (mm$^3$)")
    ax.set_ylabel("Predicted WT Volume (mm$^3$)")
    ax.set_title("Manual vs BSF-Predicted WT Volume")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    fig.tight_layout()
    fig.savefig(fig_dir / "volume_scatter.pdf", dpi=300)
    fig.savefig(fig_dir / "volume_scatter.png", dpi=150)
    plt.close(fig)


def _fig_lopo_scatter(results_manual: object, results_predicted: object, fig_dir: Path) -> None:
    """Figure 3: LOPO prediction scatter for both volume sources."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, title in [
        (axes[0], results_manual, "Manual Volumes"),
        (axes[1], results_predicted, "BSF-Predicted Volumes"),
    ]:
        preds, actuals = [], []
        for fr in results.fold_results:
            if "last_from_rest" in fr.predictions:
                for p in fr.predictions["last_from_rest"]:
                    preds.append(p["pred_mean"])
                    actuals.append(p["actual"])

        preds = np.array(preds)
        actuals = np.array(actuals)

        ax.scatter(actuals, preds, alpha=0.6, s=40)
        lim_lo = min(actuals.min(), preds.min()) - 0.5
        lim_hi = max(actuals.max(), preds.max()) + 0.5
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.5)
        ax.set_xlabel("Actual log(V+1)")
        ax.set_ylabel("Predicted log(V+1)")
        ax.set_title(f"LOPO: {title}")

        r2_key = "last_from_rest/r2_log"
        if r2_key in results.aggregate_metrics:
            r2 = results.aggregate_metrics[r2_key]
            ax.text(
                0.05,
                0.95,
                f"R$^2$ = {r2:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
            )

    fig.tight_layout()
    fig.savefig(fig_dir / "lopo_scatter.pdf", dpi=300)
    fig.savefig(fig_dir / "lopo_scatter.png", dpi=150)
    plt.close(fig)


def _fig_calibration(results_manual: object, results_predicted: object, fig_dir: Path) -> None:
    """Figure 4: Calibration plot."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for results, label in [
        (results_manual, "Manual"),
        (results_predicted, "Predicted"),
    ]:
        # Collect all z-scores from last_from_rest
        z_scores = []
        for fr in results.fold_results:
            if "last_from_rest" not in fr.predictions:
                continue
            for p in fr.predictions["last_from_rest"]:
                std = np.sqrt(max(p["pred_var"], 1e-15))
                z = abs(p["actual"] - p["pred_mean"]) / std
                z_scores.append(z)

        if not z_scores:
            continue

        z_arr = np.array(z_scores)
        # Compute empirical coverage at various nominal levels
        from scipy.stats import norm

        nominal = np.linspace(0.5, 0.99, 20)
        empirical = []
        for q in nominal:
            z_thresh = norm.ppf((1 + q) / 2)
            empirical.append(float(np.mean(z_arr <= z_thresh)))

        ax.plot(nominal, empirical, "o-", label=label, markersize=4)

    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5, label="Ideal")
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("GP Calibration (last_from_rest)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "calibration.pdf", dpi=300)
    fig.savefig(fig_dir / "calibration.png", dpi=150)
    plt.close(fig)


def _fig_dice_distribution(volumes: list[ScanVolumes], fig_dir: Path) -> None:
    """Figure: WT Dice distribution across scans."""
    dices = [v.wt_dice for v in volumes if not v.is_empty_manual]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dices, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(dices), color="red", linestyle="--", label=f"Mean={np.mean(dices):.3f}")
    ax.axvline(
        np.median(dices), color="blue", linestyle=":", label=f"Median={np.median(dices):.3f}"
    )
    ax.set_xlabel("WT Dice")
    ax.set_ylabel("Count")
    ax.set_title("BSF WT Segmentation Quality on MenGrowth")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "dice_distribution.pdf", dpi=300)
    fig.savefig(fig_dir / "dice_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="A0: Segment-Based Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/segment_based_approach/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])

    # Phase 1: Segmentation + Volume Extraction
    logger.info("=== Phase 1: Volume Extraction ===")
    extractor = SegmentationVolumeExtractor(cfg)
    volumes = extractor.extract_all(force_recompute=args.force_recompute)

    # Summary stats
    non_empty = [v for v in volumes if not v.is_empty_manual]
    logger.info(
        f"Extracted {len(volumes)} scans, {len(non_empty)} non-empty. "
        f"Mean Dice={np.mean([v.wt_dice for v in non_empty]):.3f}"
    )

    # Phase 2: Build trajectories for manual and predicted
    logger.info("=== Phase 2: Build Trajectories ===")
    traj_manual = extractor.build_trajectories(volumes, "manual")
    traj_predicted = extractor.build_trajectories(volumes, "predicted")
    logger.info(f"Manual: {len(traj_manual)} patients, Predicted: {len(traj_predicted)} patients")

    # Phase 3: LOPO-CV with ScalarGP
    logger.info("=== Phase 3: LOPO-CV ===")
    gp_cfg = cfg["gp"]
    gp_kwargs = {
        "kernel_type": gp_cfg["kernel"],
        "mean_function": gp_cfg["mean_function"],
        "n_restarts": gp_cfg["n_restarts"],
        "max_iter": gp_cfg["max_iter"],
        "lengthscale_bounds": tuple(gp_cfg["lengthscale_bounds"]),
        "signal_var_bounds": tuple(gp_cfg["signal_var_bounds"]),
        "noise_var_bounds": tuple(gp_cfg["noise_var_bounds"]),
        "seed": cfg["experiment"]["seed"],
    }

    evaluator = LOPOEvaluator()

    logger.info("Running LOPO-CV on manual volumes...")
    results_manual = evaluator.evaluate(ScalarGP, traj_manual, **gp_kwargs)

    logger.info("Running LOPO-CV on predicted volumes...")
    results_predicted = evaluator.evaluate(ScalarGP, traj_predicted, **gp_kwargs)

    # Phase 4: Save results
    logger.info("=== Phase 4: Save Results ===")
    save_results(results_manual, results_predicted, volumes, output_dir)

    # Phase 5: Generate figures
    logger.info("=== Phase 5: Generate Figures ===")
    generate_figures(results_manual, results_predicted, volumes, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION A0: SEGMENT-BASED BASELINE — RESULTS")
    print("=" * 60)
    print("\nManual volumes LOPO metrics:")
    for k, v in sorted(results_manual.aggregate_metrics.items()):
        print(f"  {k}: {v:.4f}")
    print("\nPredicted volumes LOPO metrics:")
    for k, v in sorted(results_predicted.aggregate_metrics.items()):
        print(f"  {k}: {v:.4f}")
    print(f"\nFailed folds (manual): {results_manual.failed_folds}")
    print(f"Failed folds (predicted): {results_predicted.failed_folds}")
    print("=" * 60)


if __name__ == "__main__":
    main()
