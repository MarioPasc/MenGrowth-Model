# experiments/segment_based_approach/run_baseline.py
"""Ablation A0: Segment-Based Baseline — main orchestrator.

Runs frozen BSF segmentation on MenGrowth scans, then evaluates three growth
models (ScalarGP, LME, H-GP) via LOPO-CV on both manual and predicted WT
volumes.  Saves comprehensive results, segmentation comparison, and figures.

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
from scipy.stats import norm

from growth.evaluation.lopo_evaluator import LOPOEvaluator, LOPOResults
from growth.models.growth.hgp_model import HierarchicalGPModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.models.growth.scalar_gp import ScalarGP

from .segment import ScanVolumes, SegmentationVolumeExtractor, generate_segmentation_report

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_model_configs(cfg: dict) -> dict[str, tuple[type, dict]]:
    """Build model class + kwargs from config.

    Returns:
        Dict mapping model display name to (model_class, kwargs).
    """
    gp_cfg = cfg["gp"]
    seed = cfg["experiment"]["seed"]
    models_cfg = cfg.get("models", {})

    shared_gp_kwargs = {
        "kernel_type": gp_cfg["kernel"],
        "n_restarts": gp_cfg["n_restarts"],
        "max_iter": gp_cfg["max_iter"],
        "lengthscale_bounds": tuple(gp_cfg["lengthscale_bounds"]),
        "signal_var_bounds": tuple(gp_cfg["signal_var_bounds"]),
        "noise_var_bounds": tuple(gp_cfg["noise_var_bounds"]),
        "seed": seed,
    }

    models: dict[str, tuple[type, dict]] = {}

    if models_cfg.get("scalar_gp", True):
        models["ScalarGP"] = (
            ScalarGP,
            {**shared_gp_kwargs, "mean_function": gp_cfg.get("mean_function", "linear")},
        )

    if models_cfg.get("lme", True):
        lme_cfg = cfg.get("lme", {})
        models["LME"] = (
            LMEGrowthModel,
            {"method": lme_cfg.get("method", "reml")},
        )

    if models_cfg.get("hgp", True):
        models["HGP"] = (
            HierarchicalGPModel,
            shared_gp_kwargs,
        )

    return models


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------


def save_all_results(
    lopo_results: dict[str, LOPOResults],
    volumes: list[ScanVolumes],
    seg_report: dict,
    output_dir: Path,
) -> None:
    """Save all results to JSON files.

    Args:
        lopo_results: Mapping ``"{model}_{source}"`` -> LOPOResults.
        volumes: Raw volume extraction data.
        seg_report: Segmentation comparison report.
        output_dir: Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-model LOPO results
    for key, results in lopo_results.items():
        with open(output_dir / f"lopo_results_{key}.json", "w") as f:
            json.dump(results.to_dict(), f, indent=2)

    # Segmentation comparison
    with open(output_dir / "segmentation_comparison.json", "w") as f:
        json.dump(seg_report, f, indent=2, default=str)

    # Volume summary
    non_empty = [v for v in volumes if not v.is_empty_manual]
    vol_summary = {
        "n_scans": len(volumes),
        "n_empty_manual": sum(1 for v in volumes if v.is_empty_manual),
        "n_empty_predicted": sum(1 for v in volumes if v.is_empty_predicted),
        "mean_wt_dice": float(np.mean([v.wt_dice for v in non_empty])),
        "mean_tc_dice": float(np.mean([v.tc_dice for v in non_empty])),
        "mean_et_dice": float(np.mean([v.et_dice for v in non_empty])),
        "median_wt_dice": float(np.median([v.wt_dice for v in non_empty])),
    }
    with open(output_dir / "volume_summary.json", "w") as f:
        json.dump(vol_summary, f, indent=2)

    # Model comparison summary
    comparison = _build_model_comparison(lopo_results)
    with open(output_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"All results saved to {output_dir}")


def _build_model_comparison(lopo_results: dict[str, LOPOResults]) -> dict:
    """Build head-to-head model comparison table."""
    comparison: dict = {"models": {}}

    for key, results in lopo_results.items():
        entry: dict = {
            "model_name": results.model_name,
            "n_folds_succeeded": len(results.fold_results),
            "n_folds_failed": len(results.failed_folds),
        }
        # Copy all aggregate metrics
        for metric_name, metric_val in sorted(results.aggregate_metrics.items()):
            entry[metric_name] = metric_val

        # Per-fold timing
        fit_times = [fr.fit_time_s for fr in results.fold_results]
        if fit_times:
            entry["mean_fit_time_s"] = float(np.mean(fit_times))
            entry["total_fit_time_s"] = float(np.sum(fit_times))

        # Per-fold hyperparameters summary (save all for reproducibility)
        hyper_summaries: dict[str, list[float]] = {}
        for fr in results.fold_results:
            for hp_name, hp_val in fr.fit_result.hyperparameters.items():
                hyper_summaries.setdefault(hp_name, []).append(hp_val)
        entry["hyperparameter_summary"] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in hyper_summaries.items()
        }

        comparison["models"][key] = entry

    # Rank models by last_from_rest R^2 on manual volumes
    r2_ranking: list[tuple[str, float]] = []
    for key, entry in comparison["models"].items():
        r2 = entry.get("last_from_rest/r2_log", float("-inf"))
        r2_ranking.append((key, r2))
    r2_ranking.sort(key=lambda x: x[1], reverse=True)
    comparison["ranking_by_r2_log"] = [{"key": k, "r2_log": v} for k, v in r2_ranking]

    return comparison


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

_FIG_KWARGS = {"dpi": 300}
_COLORS = {"ScalarGP": "#1f77b4", "LME": "#ff7f0e", "HGP": "#2ca02c"}


def generate_all_figures(
    lopo_results: dict[str, LOPOResults],
    volumes: list[ScanVolumes],
    output_dir: Path,
) -> None:
    """Generate all figures for the baseline experiment."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _fig_seg_quality_by_region(volumes, fig_dir)
    _fig_volume_scatter(volumes, fig_dir)
    _fig_volume_bland_altman(volumes, fig_dir)
    _fig_lopo_scatter_multimodel(lopo_results, fig_dir)
    _fig_calibration_multimodel(lopo_results, fig_dir)
    _fig_model_comparison_bar(lopo_results, fig_dir)
    _fig_dice_distribution(volumes, fig_dir)

    logger.info(f"Figures saved to {fig_dir}")


def _fig_seg_quality_by_region(volumes: list[ScanVolumes], fig_dir: Path) -> None:
    """Box plot of Dice scores by region (TC, WT, ET)."""
    non_empty = [v for v in volumes if not v.is_empty_manual]

    data = {
        "TC": [v.tc_dice for v in non_empty],
        "WT": [v.wt_dice for v in non_empty],
        "ET": [v.et_dice for v in non_empty],
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [data["TC"], data["WT"], data["ET"]],
        labels=["TC", "WT", "ET"],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
    )

    colors = ["#66c2a5", "#fc8d62", "#8da0cb"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Annotate means
    for i, region in enumerate(["TC", "WT", "ET"]):
        mean_val = np.mean(data[region])
        ax.annotate(
            f"{mean_val:.3f}",
            xy=(i + 1, mean_val),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_ylabel("Dice Score")
    ax.set_title("BSF Segmentation Quality on MenGrowth (per region)")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(fig_dir / "seg_quality_by_region.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "seg_quality_by_region.png", dpi=150)
    plt.close(fig)


def _fig_volume_scatter(volumes: list[ScanVolumes], fig_dir: Path) -> None:
    """Manual vs predicted WT volume scatter with identity line."""
    non_empty = [v for v in volumes if not v.is_empty_manual]
    manual = np.array([v.manual_wt_vol_mm3 for v in non_empty])
    predicted = np.array([v.predicted_wt_vol_mm3 for v in non_empty])
    dices = np.array([v.wt_dice for v in non_empty])

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
    fig.savefig(fig_dir / "volume_scatter.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "volume_scatter.png", dpi=150)
    plt.close(fig)


def _fig_volume_bland_altman(volumes: list[ScanVolumes], fig_dir: Path) -> None:
    """Bland-Altman plot for WT volume comparison."""
    non_empty = [v for v in volumes if not v.is_empty_manual]
    manual = np.array([v.manual_wt_vol_mm3 for v in non_empty])
    predicted = np.array([v.predicted_wt_vol_mm3 for v in non_empty])

    mean_vol = (manual + predicted) / 2
    diff = predicted - manual
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_vol, diff, alpha=0.6, s=30, c="#1f77b4")
    ax.axhline(mean_diff, color="red", linestyle="-", label=f"Mean bias: {mean_diff:.0f} mm$^3$")
    ax.axhline(
        mean_diff + 1.96 * std_diff,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"+1.96 SD: {mean_diff + 1.96 * std_diff:.0f}",
    )
    ax.axhline(
        mean_diff - 1.96 * std_diff,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"-1.96 SD: {mean_diff - 1.96 * std_diff:.0f}",
    )
    ax.set_xlabel("Mean WT Volume (mm$^3$)")
    ax.set_ylabel("Difference: Predicted $-$ Manual (mm$^3$)")
    ax.set_title("Bland-Altman: WT Volume (BSF vs Manual)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "volume_bland_altman.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "volume_bland_altman.png", dpi=150)
    plt.close(fig)


def _fig_lopo_scatter_multimodel(
    lopo_results: dict[str, LOPOResults],
    fig_dir: Path,
) -> None:
    """Multi-model LOPO prediction scatter (predicted vs actual log(V+1))."""
    # Separate by source
    sources = ["manual", "predicted"]
    model_names = sorted(
        {k.rsplit("_", 1)[0] for k in lopo_results},
        key=lambda x: list(_COLORS.keys()).index(x) if x in _COLORS else 99,
    )

    n_models = len(model_names)
    fig, axes = plt.subplots(
        len(sources), n_models, figsize=(4.5 * n_models, 4.5 * len(sources)), squeeze=False
    )

    for row, source in enumerate(sources):
        for col, model_name in enumerate(model_names):
            key = f"{model_name}_{source}"
            ax = axes[row, col]

            if key not in lopo_results:
                ax.set_visible(False)
                continue

            results = lopo_results[key]
            preds, actuals = [], []
            for fr in results.fold_results:
                if "last_from_rest" in fr.predictions:
                    for p in fr.predictions["last_from_rest"]:
                        preds.append(p["pred_mean"])
                        actuals.append(p["actual"])

            if not preds:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            pred_arr = np.array(preds)
            actual_arr = np.array(actuals)
            color = _COLORS.get(model_name, "#333333")

            ax.scatter(actual_arr, pred_arr, alpha=0.6, s=40, c=color)
            lim_lo = min(actual_arr.min(), pred_arr.min()) - 0.3
            lim_hi = max(actual_arr.max(), pred_arr.max()) + 0.3
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.5)

            r2_key = "last_from_rest/r2_log"
            cal_key = "last_from_rest/calibration_95"
            r2 = results.aggregate_metrics.get(r2_key, float("nan"))
            cal = results.aggregate_metrics.get(cal_key, float("nan"))
            ax.text(
                0.05,
                0.95,
                f"R$^2$ = {r2:.3f}\nCal$_{{95}}$ = {cal:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )

            ax.set_xlabel("Actual log(V+1)")
            ax.set_ylabel("Predicted log(V+1)")
            ax.set_title(f"{model_name} ({source})")

    fig.tight_layout()
    fig.savefig(fig_dir / "lopo_scatter_comparison.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "lopo_scatter_comparison.png", dpi=150)
    plt.close(fig)


def _fig_calibration_multimodel(
    lopo_results: dict[str, LOPOResults],
    fig_dir: Path,
) -> None:
    """Calibration curves for all models, separated by volume source."""
    sources = ["manual", "predicted"]
    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 5))
    if len(sources) == 1:
        axes = [axes]

    model_names = sorted(
        {k.rsplit("_", 1)[0] for k in lopo_results},
        key=lambda x: list(_COLORS.keys()).index(x) if x in _COLORS else 99,
    )

    nominal = np.linspace(0.5, 0.99, 20)

    for ax, source in zip(axes, sources):
        for model_name in model_names:
            key = f"{model_name}_{source}"
            if key not in lopo_results:
                continue

            results = lopo_results[key]
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
            empirical = [float(np.mean(z_arr <= norm.ppf((1 + q) / 2))) for q in nominal]

            color = _COLORS.get(model_name, "#333333")
            ax.plot(nominal, empirical, "o-", label=model_name, markersize=3, color=color)

        ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5, label="Ideal")
        ax.set_xlabel("Nominal Coverage")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title(f"Calibration ({source} volumes)")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_comparison.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "calibration_comparison.png", dpi=150)
    plt.close(fig)


def _fig_model_comparison_bar(
    lopo_results: dict[str, LOPOResults],
    fig_dir: Path,
) -> None:
    """Bar chart comparing R^2, MAE, and calibration across models."""
    # Focus on manual volumes for the primary comparison
    model_names = sorted(
        {k.rsplit("_", 1)[0] for k in lopo_results},
        key=lambda x: list(_COLORS.keys()).index(x) if x in _COLORS else 99,
    )

    metrics_to_plot = [
        ("last_from_rest/r2_log", "R$^2$ (log-space)", True),
        ("last_from_rest/mae_log", "MAE (log-space)", False),
        ("last_from_rest/calibration_95", "Calibration (95% CI)", True),
        ("last_from_rest/mean_ci_width_log", "Mean CI Width (log)", False),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5))

    for ax, (metric_key, metric_label, higher_better) in zip(axes, metrics_to_plot):
        vals_manual = []
        vals_predicted = []
        labels = []
        colors = []

        for model_name in model_names:
            key_m = f"{model_name}_manual"
            key_p = f"{model_name}_predicted"
            if key_m not in lopo_results:
                continue

            val_m = lopo_results[key_m].aggregate_metrics.get(metric_key, float("nan"))
            val_p = lopo_results[key_p].aggregate_metrics.get(metric_key, float("nan"))
            vals_manual.append(val_m)
            vals_predicted.append(val_p)
            labels.append(model_name)
            colors.append(_COLORS.get(model_name, "#333333"))

        x = np.arange(len(labels))
        width = 0.35

        bars_m = ax.bar(x - width / 2, vals_manual, width, label="Manual", color=colors, alpha=0.9)
        bars_p = ax.bar(
            x + width / 2, vals_predicted, width, label="Predicted", color=colors, alpha=0.5
        )

        # Annotate values
        for bar in list(bars_m) + list(bars_p):
            h = bar.get_height()
            if np.isfinite(h):
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel(metric_label)
        ax.legend(fontsize=8)

    fig.suptitle("Model Comparison: Manual vs Predicted Volumes (last_from_rest)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison_bar.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "model_comparison_bar.png", dpi=150)
    plt.close(fig)


def _fig_dice_distribution(volumes: list[ScanVolumes], fig_dir: Path) -> None:
    """WT Dice distribution across scans."""
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
    fig.savefig(fig_dir / "dice_distribution.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "dice_distribution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    # =========================================================================
    # Phase 1: Segmentation + Volume Extraction
    # =========================================================================
    logger.info("=== Phase 1: Volume Extraction ===")
    extractor = SegmentationVolumeExtractor(cfg)
    volumes = extractor.extract_all(force_recompute=args.force_recompute)

    non_empty = [v for v in volumes if not v.is_empty_manual]
    logger.info(
        f"Extracted {len(volumes)} scans, {len(non_empty)} non-empty. "
        f"Mean WT Dice={np.mean([v.wt_dice for v in non_empty]):.3f}, "
        f"Mean TC Dice={np.mean([v.tc_dice for v in non_empty]):.3f}, "
        f"Mean ET Dice={np.mean([v.et_dice for v in non_empty]):.3f}"
    )

    # =========================================================================
    # Phase 2: Segmentation Comparison Report
    # =========================================================================
    logger.info("=== Phase 2: Segmentation Comparison ===")
    seg_report = generate_segmentation_report(volumes)
    logger.info(
        f"Segmentation comparison: "
        f"WT Dice={seg_report['per_region']['wt']['dice_mean']:.3f} "
        f"(IQR {seg_report['per_region']['wt']['dice_q25']:.3f}-"
        f"{seg_report['per_region']['wt']['dice_q75']:.3f})"
    )

    # =========================================================================
    # Phase 3: Build Trajectories
    # =========================================================================
    logger.info("=== Phase 3: Build Trajectories ===")
    traj_manual = extractor.build_trajectories(volumes, "manual")
    traj_predicted = extractor.build_trajectories(volumes, "predicted")
    logger.info(f"Manual: {len(traj_manual)} patients, Predicted: {len(traj_predicted)} patients")

    # =========================================================================
    # Phase 4: Multi-Model LOPO-CV
    # =========================================================================
    logger.info("=== Phase 4: Multi-Model LOPO-CV ===")
    model_configs = _build_model_configs(cfg)
    evaluator = LOPOEvaluator()

    lopo_results: dict[str, LOPOResults] = {}

    sources = {"manual": traj_manual, "predicted": traj_predicted}

    for model_name, (model_cls, kwargs) in model_configs.items():
        for source_name, trajectories in sources.items():
            key = f"{model_name}_{source_name}"
            logger.info(f"--- LOPO-CV: {model_name} on {source_name} volumes ---")
            try:
                results = evaluator.evaluate(model_cls, trajectories, **kwargs)
                lopo_results[key] = results

                r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("nan"))
                cal = results.aggregate_metrics.get("last_from_rest/calibration_95", float("nan"))
                logger.info(
                    f"  {key}: R2_log={r2:.4f}, Cal_95={cal:.3f}, "
                    f"folds={len(results.fold_results)}/{len(results.fold_results) + len(results.failed_folds)}"
                )
            except Exception as e:
                logger.error(f"  {key} FAILED: {e}")

    # =========================================================================
    # Phase 5: Save Results
    # =========================================================================
    logger.info("=== Phase 5: Save Results ===")
    save_all_results(lopo_results, volumes, seg_report, output_dir)

    # =========================================================================
    # Phase 6: Generate Figures
    # =========================================================================
    logger.info("=== Phase 6: Generate Figures ===")
    generate_all_figures(lopo_results, volumes, output_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 72)
    print("ABLATION A0: SEGMENT-BASED BASELINE — RESULTS")
    print("=" * 72)

    # Segmentation quality
    print("\n--- Segmentation Quality (BSF vs Manual) ---")
    for region in ["wt", "tc", "et"]:
        stats = seg_report["per_region"][region]
        print(
            f"  {region.upper():>2}: Dice = {stats['dice_mean']:.3f} +/- {stats['dice_std']:.3f} "
            f"(median {stats['dice_median']:.3f}, range [{stats['dice_min']:.3f}, {stats['dice_max']:.3f}])"
        )
        print(
            f"       Vol corr r={stats['volume_pearson_r']:.3f}, "
            f"MAE={stats['volume_mae_mm3']:.0f}mm3, bias={stats['volume_mean_bias_mm3']:.0f}mm3"
        )

    # LOPO results per model
    for source_name in ["manual", "predicted"]:
        print(f"\n--- LOPO-CV Results ({source_name} volumes) ---")
        print(
            f"  {'Model':<12} {'R2_log':>8} {'MAE_log':>8} {'RMSE_log':>9} "
            f"{'Cal_95':>7} {'CI_width':>8} {'R2_orig':>8}"
        )
        print("  " + "-" * 65)

        for model_name in model_configs:
            key = f"{model_name}_{source_name}"
            if key not in lopo_results:
                continue
            m = lopo_results[key].aggregate_metrics
            print(
                f"  {model_name:<12} "
                f"{m.get('last_from_rest/r2_log', float('nan')):>8.4f} "
                f"{m.get('last_from_rest/mae_log', float('nan')):>8.4f} "
                f"{m.get('last_from_rest/rmse_log', float('nan')):>9.4f} "
                f"{m.get('last_from_rest/calibration_95', float('nan')):>7.3f} "
                f"{m.get('last_from_rest/mean_ci_width_log', float('nan')):>8.4f} "
                f"{m.get('last_from_rest/r2_original', float('nan')):>8.4f}"
            )

    # Failed folds
    any_failed = False
    for key, results in lopo_results.items():
        if results.failed_folds:
            any_failed = True
            print(f"\n  Failed folds ({key}): {results.failed_folds}")
    if not any_failed:
        print("\n  No failed folds.")

    print("\n" + "=" * 72)
    print(f"Output directory: {output_dir}")
    print(f"Figures: {output_dir}/figures/")
    print("=" * 72)


if __name__ == "__main__":
    main()
