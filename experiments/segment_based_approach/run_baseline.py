# experiments/segment_based_approach/run_baseline.py
"""Ablation A0: Segment-Based Baseline — main orchestrator.

Runs one or more frozen segmentation models on MenGrowth scans, then evaluates
three growth models (ScalarGP, LME, H-GP) via LOPO-CV on manual and predicted
WT volumes.  Saves comprehensive results, segmentation comparison, and figures.

Supports **multi-model** configs: each enabled segmentation model produces its
own set of LOPO-CV results and figures, plus cross-source comparison.

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

from .segment import (
    ScanVolumes,
    SegmentationVolumeExtractor,
    generate_segmentation_report,
    parse_seg_config,
)

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

    # Covariate settings (shared across all models)
    cov_cfg = cfg.get("covariates", {})
    cov_enabled = cov_cfg.get("enabled", False)
    cov_features = cov_cfg.get("features", [])
    cov_missing = cov_cfg.get("missing_strategy", "skip")

    cov_kwargs: dict = {}
    if cov_enabled:
        cov_kwargs = {
            "use_covariates": True,
            "covariate_names": cov_features,
            "missing_strategy": cov_missing,
        }

    shared_gp_kwargs = {
        "kernel_type": gp_cfg["kernel"],
        "n_restarts": gp_cfg["n_restarts"],
        "max_iter": gp_cfg["max_iter"],
        "lengthscale_bounds": tuple(gp_cfg["lengthscale_bounds"]),
        "signal_var_bounds": tuple(gp_cfg["signal_var_bounds"]),
        "noise_var_bounds": tuple(gp_cfg["noise_var_bounds"]),
        "seed": seed,
        **cov_kwargs,
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
            {"method": lme_cfg.get("method", "reml"), **cov_kwargs},
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
    sources: dict[str, list],
) -> None:
    """Save all results in organized directory structure.

    Directory layout::

        segmentation/
            comparison.json
            volume_summary.json
        growth_prediction/
            {source}/
                lopo_results_{gp_model}.json
                model_comparison.json
            cross_source_comparison.json

    Args:
        lopo_results: Mapping ``"{gp_model}_{source}"`` -> LOPOResults.
        volumes: Raw volume extraction data.
        seg_report: Segmentation comparison report.
        output_dir: Output directory.
        sources: Mapping ``{source_name: [PatientTrajectory]}``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Segmentation results
    seg_dir = output_dir / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    with open(seg_dir / "comparison.json", "w") as f:
        json.dump(seg_report, f, indent=2, default=str)

    # Volume summary
    non_empty = [v for v in volumes if not v.is_empty_manual]
    vol_summary: dict = {
        "n_scans": len(volumes),
        "n_empty_manual": sum(1 for v in volumes if v.is_empty_manual),
    }
    # Per-model empty counts + mean Dice
    for model_name in seg_report.get("per_model", {}):
        model_stats = seg_report["per_model"][model_name]
        vol_summary[f"n_empty_{model_name}"] = model_stats.get("n_empty_predicted", 0)
        wt_stats = model_stats.get("per_region", {}).get("wt", {})
        vol_summary[f"mean_wt_dice_{model_name}"] = wt_stats.get("dice_mean", 0.0)
        vol_summary[f"median_wt_dice_{model_name}"] = wt_stats.get("dice_median", 0.0)
    with open(seg_dir / "volume_summary.json", "w") as f:
        json.dump(vol_summary, f, indent=2)

    # Growth prediction results — organized by source
    gp_dir = output_dir / "growth_prediction"
    gp_dir.mkdir(parents=True, exist_ok=True)

    for source_name in sources:
        source_dir = gp_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        # Per-GP-model results for this source
        source_results = {}
        for key, results in lopo_results.items():
            if key.endswith(f"_{source_name}"):
                gp_name = key[: -(len(source_name) + 1)]
                with open(source_dir / f"lopo_results_{gp_name}.json", "w") as f:
                    json.dump(results.to_dict(), f, indent=2)
                source_results[key] = results

        # Model comparison for this source
        if source_results:
            comparison = _build_model_comparison(source_results)
            with open(source_dir / "model_comparison.json", "w") as f:
                json.dump(comparison, f, indent=2)

    # Cross-source comparison
    cross_comparison = _build_cross_source_comparison(lopo_results, sources)
    with open(gp_dir / "cross_source_comparison.json", "w") as f:
        json.dump(cross_comparison, f, indent=2)

    # Backward compat: also save flat model_comparison.json at root
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

        # Per-fold hyperparameters summary
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


def _build_cross_source_comparison(
    lopo_results: dict[str, LOPOResults],
    sources: dict[str, list],
) -> dict:
    """Build cross-source comparison: best GP model per source, ranking.

    Args:
        lopo_results: All ``"{gp_model}_{source}"`` results.
        sources: Source name -> trajectories.

    Returns:
        Dict with per-source best model and overall ranking.
    """
    comparison: dict = {"sources": {}}

    for source_name in sources:
        best_key = None
        best_r2 = float("-inf")
        source_models: dict[str, dict] = {}

        for key, results in lopo_results.items():
            if not key.endswith(f"_{source_name}"):
                continue
            gp_name = key[: -(len(source_name) + 1)]
            r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("nan"))
            cal = results.aggregate_metrics.get("last_from_rest/calibration_95", float("nan"))
            mae = results.aggregate_metrics.get("last_from_rest/mae_log", float("nan"))

            source_models[gp_name] = {"r2_log": r2, "calibration_95": cal, "mae_log": mae}

            if np.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                best_key = gp_name

        comparison["sources"][source_name] = {
            "best_gp_model": best_key,
            "best_r2_log": best_r2 if np.isfinite(best_r2) else None,
            "models": source_models,
        }

    # Rank sources by best R^2
    source_ranking = sorted(
        comparison["sources"].items(),
        key=lambda x: x[1].get("best_r2_log") or float("-inf"),
        reverse=True,
    )
    comparison["source_ranking"] = [
        {"source": name, "best_r2": data.get("best_r2_log")} for name, data in source_ranking
    ]

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
    sources: dict[str, list],
    model_configs: dict[str, tuple],
    h5_path: str | None = None,
    seg_model_names: list[str] | None = None,
) -> None:
    """Generate all figures for the baseline experiment.

    Args:
        lopo_results: Mapping ``"{model}_{source}"`` -> LOPOResults.
        volumes: Raw volume extraction data.
        output_dir: Output directory.
        sources: Dict ``{source_name: [PatientTrajectory]}``.
        model_configs: Dict ``{gp_model_name: (model_class, kwargs)}``.
        h5_path: Path to MenGrowth.h5. Required for segmentation overlay.
        seg_model_names: List of segmentation model names (for multi-model figs).
    """
    if seg_model_names is None:
        seg_model_names = [mn for mn in _discover_model_names(volumes) if mn != "manual"]

    # --- Segmentation figures ---
    seg_fig_dir = output_dir / "figures" / "segmentation"
    seg_fig_dir.mkdir(parents=True, exist_ok=True)

    _fig_seg_quality_by_region(volumes, seg_fig_dir, seg_model_names)
    for model_name in seg_model_names:
        _fig_volume_scatter(volumes, seg_fig_dir, model_name=model_name)
        _fig_volume_bland_altman(volumes, seg_fig_dir, model_name=model_name)
    _fig_dice_distribution(volumes, seg_fig_dir, seg_model_names)
    if len(seg_model_names) > 1:
        _fig_dice_comparison_boxplot(volumes, seg_fig_dir, seg_model_names)

    if h5_path is not None:
        generate_segmentation_overlay(h5_path, seg_fig_dir)

    # --- Growth prediction figures (per source) ---
    for source_name in sources:
        gp_fig_dir = output_dir / "figures" / "growth_prediction" / source_name
        gp_fig_dir.mkdir(parents=True, exist_ok=True)

        # Filter LOPO results for this source
        source_lopo = {k: v for k, v in lopo_results.items() if k.endswith(f"_{source_name}")}

        _fig_lopo_scatter_multimodel(source_lopo, gp_fig_dir, source_name)
        _fig_calibration_multimodel(source_lopo, gp_fig_dir, source_name)
        _fig_model_comparison_bar(source_lopo, gp_fig_dir, source_name)

        trajectories = sources.get(source_name, [])
        if trajectories and model_configs:
            _fig_gp_sausage_plots(
                {source_name: trajectories},
                model_configs,
                gp_fig_dir,
                source=source_name,
            )

    # --- Cross-source summary ---
    if len(sources) > 1:
        summary_fig_dir = output_dir / "figures"
        summary_fig_dir.mkdir(parents=True, exist_ok=True)
        _fig_cross_source_summary(lopo_results, sources, summary_fig_dir)

    logger.info(f"Figures saved to {output_dir / 'figures'}")


def _discover_model_names(volumes: list[ScanVolumes]) -> list[str]:
    """Discover all model names present in volumes."""
    names: list[str] = []
    for v in volumes:
        for mn in v.model_results:
            if mn not in names:
                names.append(mn)
    return names


def _fig_seg_quality_by_region(
    volumes: list[ScanVolumes],
    fig_dir: Path,
    seg_model_names: list[str],
) -> None:
    """Grouped box plot of Dice scores by region, one group per seg model."""
    non_empty = [v for v in volumes if not v.is_empty_manual]

    if len(seg_model_names) == 1:
        # Single-model: simple box plot (backward compat layout)
        mn = seg_model_names[0]
        data = {
            "TC": [v.model_results[mn].tc_dice for v in non_empty if mn in v.model_results],
            "WT": [v.model_results[mn].wt_dice for v in non_empty if mn in v.model_results],
            "ET": [v.model_results[mn].et_dice for v in non_empty if mn in v.model_results],
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
        ax.set_title(f"Segmentation Quality ({mn})")
        ax.set_ylim(0, 1.05)
    else:
        # Multi-model: grouped boxplots
        regions = ["TC", "WT", "ET"]
        n_models = len(seg_model_names)
        n_regions = len(regions)

        fig, ax = plt.subplots(figsize=(3 * n_regions + 1, 5))
        width = 0.7 / n_models
        model_colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))

        for m_idx, mn in enumerate(seg_model_names):
            positions = []
            data_per_region = []
            for r_idx, region in enumerate(regions):
                attr = f"{region.lower()}_dice"
                vals = [
                    getattr(v.model_results[mn], attr) for v in non_empty if mn in v.model_results
                ]
                data_per_region.append(vals)
                positions.append(r_idx + 1 + (m_idx - n_models / 2 + 0.5) * width)

            bp = ax.boxplot(
                data_per_region,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 4},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(model_colors[m_idx])
                patch.set_alpha(0.7)

        ax.set_xticks(range(1, n_regions + 1))
        ax.set_xticklabels(regions)
        ax.set_ylabel("Dice Score")
        ax.set_title("Segmentation Quality by Region and Model")
        ax.set_ylim(0, 1.05)

        # Legend
        from matplotlib.patches import Patch

        legend_patches = [
            Patch(facecolor=model_colors[i], alpha=0.7, label=mn)
            for i, mn in enumerate(seg_model_names)
        ]
        ax.legend(handles=legend_patches, fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(fig_dir / "seg_quality_by_region.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "seg_quality_by_region.png", dpi=150)
    plt.close(fig)


def _fig_volume_scatter(
    volumes: list[ScanVolumes],
    fig_dir: Path,
    model_name: str = "brainsegfounder",
) -> None:
    """Manual vs predicted WT volume scatter with identity line."""
    non_empty = [v for v in volumes if not v.is_empty_manual and model_name in v.model_results]
    if not non_empty:
        logger.warning(f"No data for volume scatter ({model_name})")
        return

    manual = np.array([v.manual_wt_vol_mm3 for v in non_empty])
    predicted = np.array([v.model_results[model_name].wt_vol_mm3 for v in non_empty])
    dices = np.array([v.model_results[model_name].wt_dice for v in non_empty])

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(manual, predicted, c=dices, cmap="viridis", alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax, label="WT Dice")

    lim = max(manual.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Identity")
    ax.set_xlabel("Manual WT Volume (mm$^3$)")
    ax.set_ylabel("Predicted WT Volume (mm$^3$)")
    ax.set_title(f"Manual vs Predicted WT Volume ({model_name})")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    fig.tight_layout()
    fig.savefig(fig_dir / f"volume_scatter_{model_name}.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / f"volume_scatter_{model_name}.png", dpi=150)
    plt.close(fig)


def _fig_volume_bland_altman(
    volumes: list[ScanVolumes],
    fig_dir: Path,
    model_name: str = "brainsegfounder",
) -> None:
    """Bland-Altman plot for WT volume comparison."""
    non_empty = [v for v in volumes if not v.is_empty_manual and model_name in v.model_results]
    if not non_empty:
        logger.warning(f"No data for Bland-Altman ({model_name})")
        return

    manual = np.array([v.manual_wt_vol_mm3 for v in non_empty])
    predicted = np.array([v.model_results[model_name].wt_vol_mm3 for v in non_empty])

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
    ax.set_title(f"Bland-Altman: WT Volume ({model_name} vs Manual)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / f"volume_bland_altman_{model_name}.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / f"volume_bland_altman_{model_name}.png", dpi=150)
    plt.close(fig)


def _fig_lopo_scatter_multimodel(
    lopo_results: dict[str, LOPOResults],
    fig_dir: Path,
    source: str = "manual",
) -> None:
    """Multi-model LOPO prediction scatter (predicted vs actual log(V+1))."""
    suffix = f"_{source}"
    model_names = sorted(
        {k[: -len(suffix)] for k in lopo_results if k.endswith(suffix)},
        key=lambda x: list(_COLORS.keys()).index(x) if x in _COLORS else 99,
    )

    n_models = len(model_names)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.5), squeeze=False)

    for col, model_name in enumerate(model_names):
        key = f"{model_name}_{source}"
        ax = axes[0, col]

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

        r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("nan"))
        cal = results.aggregate_metrics.get("last_from_rest/calibration_95", float("nan"))
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
    fig.savefig(fig_dir / "lopo_scatter.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "lopo_scatter.png", dpi=150)
    plt.close(fig)


def _fig_calibration_multimodel(
    lopo_results: dict[str, LOPOResults],
    fig_dir: Path,
    source: str = "manual",
) -> None:
    """Calibration curves for all GP models on a given source."""
    suffix = f"_{source}"
    model_names = sorted(
        {k[: -len(suffix)] for k in lopo_results if k.endswith(suffix)},
        key=lambda x: list(_COLORS.keys()).index(x) if x in _COLORS else 99,
    )

    nominal = np.linspace(0.5, 0.99, 20)

    fig, ax = plt.subplots(figsize=(6, 5))
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
    fig.savefig(fig_dir / "calibration.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "calibration.png", dpi=150)
    plt.close(fig)


def _fig_model_comparison_bar(
    lopo_results: dict[str, LOPOResults],
    fig_dir: Path,
    source: str = "manual",
) -> None:
    """Bar chart comparing R^2, MAE, and calibration across GP models."""
    suffix = f"_{source}"
    model_names = sorted(
        {k[: -len(suffix)] for k in lopo_results if k.endswith(suffix)},
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
        vals = []
        labels = []
        colors = []

        for model_name in model_names:
            key = f"{model_name}_{source}"
            if key not in lopo_results:
                continue
            val = lopo_results[key].aggregate_metrics.get(metric_key, float("nan"))
            vals.append(val)
            labels.append(model_name)
            colors.append(_COLORS.get(model_name, "#333333"))

        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=colors, alpha=0.85)

        for bar in bars:
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

    fig.suptitle(f"Model Comparison: {source} volumes (last_from_rest)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison_bar.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "model_comparison_bar.png", dpi=150)
    plt.close(fig)


def _fig_dice_distribution(
    volumes: list[ScanVolumes],
    fig_dir: Path,
    seg_model_names: list[str],
) -> None:
    """WT Dice distribution across scans (overlay for all models)."""
    non_empty = [v for v in volumes if not v.is_empty_manual]

    fig, ax = plt.subplots(figsize=(7, 4))
    model_colors = plt.cm.Set1(np.linspace(0, 1, max(len(seg_model_names), 3)))

    for m_idx, mn in enumerate(seg_model_names):
        dices = [v.model_results[mn].wt_dice for v in non_empty if mn in v.model_results]
        if not dices:
            continue
        ax.hist(
            dices,
            bins=20,
            edgecolor="black",
            alpha=0.5,
            color=model_colors[m_idx],
            label=f"{mn} (mean={np.mean(dices):.3f})",
        )

    ax.set_xlabel("WT Dice")
    ax.set_ylabel("Count")
    ax.set_title("WT Segmentation Quality on MenGrowth")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "dice_distribution.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "dice_distribution.png", dpi=150)
    plt.close(fig)


def _fig_dice_comparison_boxplot(
    volumes: list[ScanVolumes],
    fig_dir: Path,
    seg_model_names: list[str],
) -> None:
    """Side-by-side boxplots of WT Dice per segmentation model."""
    non_empty = [v for v in volumes if not v.is_empty_manual]

    data = []
    labels = []
    for mn in seg_model_names:
        dices = [v.model_results[mn].wt_dice for v in non_empty if mn in v.model_results]
        if dices:
            data.append(dices)
            labels.append(mn)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(4, 2 * len(labels)), 5))
    model_colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(model_colors[i])
        patch.set_alpha(0.7)

    # Annotate means
    for i, vals in enumerate(data):
        mean_val = np.mean(vals)
        ax.annotate(
            f"{mean_val:.3f}",
            xy=(i + 1, mean_val),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_ylabel("WT Dice Score")
    ax.set_title("WT Dice Comparison Across Segmentation Models")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(fig_dir / "dice_comparison_boxplot.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "dice_comparison_boxplot.png", dpi=150)
    plt.close(fig)


def _fig_cross_source_summary(
    lopo_results: dict[str, LOPOResults],
    sources: dict[str, list],
    fig_dir: Path,
) -> None:
    """Bar chart: best GP model's R^2 across all seg sources."""
    source_best: dict[str, tuple[str, float]] = {}

    for source_name in sources:
        best_name = ""
        best_r2 = float("-inf")
        for key, results in lopo_results.items():
            if not key.endswith(f"_{source_name}"):
                continue
            r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("nan"))
            if np.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                best_name = key[: -len(f"_{source_name}")]
        if np.isfinite(best_r2):
            source_best[source_name] = (best_name, best_r2)

    if not source_best:
        return

    labels = list(source_best.keys())
    r2_vals = [source_best[s][1] for s in labels]
    best_models = [source_best[s][0] for s in labels]

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(labels)), 5))
    colors = plt.cm.Paired(np.linspace(0, 1, max(len(labels), 3)))
    bars = ax.bar(range(len(labels)), r2_vals, color=colors, alpha=0.85)

    for i, (bar, gp_name) in enumerate(zip(bars, best_models)):
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}\n({gp_name})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Best R$^2$ (log-space, LOPO)")
    ax.set_title("Growth Prediction Quality by Segmentation Source")

    fig.tight_layout()
    fig.savefig(fig_dir / "cross_source_summary.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / "cross_source_summary.png", dpi=150)
    plt.close(fig)


def _fig_gp_sausage_plots(
    trajectories_by_source: dict[str, list],
    model_configs: dict[str, tuple],
    fig_dir: Path,
    source: str = "manual",
    max_patients: int = 9,
    min_timepoints: int = 3,
) -> None:
    """Generate GP posterior predictive ('sausage') plots for individual patients.

    For each selected patient, performs LOPO: fits each model on all other
    patients, conditions on the held-out patient's first observation, and
    plots the posterior predictive distribution over a fine time grid.

    Args:
        trajectories_by_source: Dict mapping source name to list of
            PatientTrajectory objects.
        model_configs: Dict mapping model name to (model_class, kwargs).
        fig_dir: Output directory for figures.
        source: Which volume source to use.
        max_patients: Maximum number of patients to plot.
        min_timepoints: Minimum timepoints required for a patient to be selected.
    """
    trajectories = trajectories_by_source.get(source, [])
    if not trajectories:
        logger.warning(f"No trajectories for source '{source}', skipping sausage plots")
        return

    # Select patients with enough timepoints, sorted by n_timepoints desc
    candidates = [p for p in trajectories if p.n_timepoints >= min_timepoints]
    candidates.sort(key=lambda p: (-p.n_timepoints, p.patient_id))
    selected = candidates[:max_patients]

    if not selected:
        logger.warning(f"No patients with >= {min_timepoints} timepoints, skipping sausage plots")
        return

    n_models = len(model_configs)
    n_patients = len(selected)
    n_cols = min(3, n_patients)
    n_rows_per_model = (n_patients + n_cols - 1) // n_cols

    for model_name, (model_cls, kwargs) in model_configs.items():
        fig, axes = plt.subplots(
            n_rows_per_model,
            n_cols,
            figsize=(5.5 * n_cols, 4.0 * n_rows_per_model),
            squeeze=False,
        )

        color = _COLORS.get(model_name, "#1f77b4")

        for idx, patient in enumerate(selected):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # LOPO: train on everyone except this patient
            train_patients = [p for p in trajectories if p.patient_id != patient.patient_id]

            try:
                model = model_cls(**kwargs)
                model.fit(train_patients)

                # Fine time grid for smooth posterior
                t_min = patient.times[0] - 0.3
                t_max = patient.times[-1] + 0.5
                t_grid = np.linspace(max(t_min, -0.5), t_max, 200)

                pred = model.predict(patient, t_grid, n_condition=1)

                mean = pred.mean[:, 0]
                std = np.sqrt(np.clip(pred.variance[:, 0], 0, None))

                ax.fill_between(
                    t_grid,
                    mean - 2 * std,
                    mean + 2 * std,
                    alpha=0.15,
                    color=color,
                    label="$\\pm 2\\sigma$ (95.4%)",
                )
                ax.fill_between(
                    t_grid,
                    mean - std,
                    mean + std,
                    alpha=0.3,
                    color=color,
                    label="$\\pm 1\\sigma$ (68.3%)",
                )
                ax.plot(t_grid, mean, "-", color=color, linewidth=1.5, label="GP mean")

            except Exception as e:
                logger.warning(f"Sausage plot failed for {patient.patient_id}/{model_name}: {e}")
                ax.text(
                    0.5,
                    0.5,
                    f"Fit failed:\n{e!s:.40}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

            # Plot all observations
            obs = patient.observations[:, 0]
            ax.plot(
                patient.times[0],
                obs[0],
                "o",
                color="black",
                markersize=8,
                zorder=5,
                label="Conditioning obs.",
            )
            if len(patient.times) > 1:
                ax.plot(
                    patient.times[1:],
                    obs[1:],
                    "D",
                    color="red",
                    markersize=7,
                    zorder=5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    label="Future obs. (held out)",
                )

            ax.set_title(
                f"{patient.patient_id} ({patient.n_timepoints} tp)",
                fontsize=10,
            )
            ax.set_xlabel("Timepoint (ordinal)")
            ax.set_ylabel("log(V+1)")

            if idx == 0:
                ax.legend(fontsize=7, loc="upper left")

        # Hide unused axes
        for idx in range(n_patients, n_rows_per_model * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        fig.suptitle(
            f"GP Posterior Predictive: {model_name} ({source} volumes)\n"
            f"Conditioned on 1st observation, predicting forward",
            fontsize=12,
        )
        fig.tight_layout()

        fname = f"sausage_{model_name}_{source}"
        fig.savefig(fig_dir / f"{fname}.pdf", **_FIG_KWARGS)
        fig.savefig(fig_dir / f"{fname}.png", dpi=150)
        plt.close(fig)
        logger.info(f"Sausage plot saved: {fname}")

    # Illustrative figure
    _fig_gp_illustrative(
        trajectories, selected[: min(4, n_patients)], model_configs, fig_dir, source
    )

    # Combined figure: all models for a subset of patients
    top_patients = selected[: min(4, n_patients)]
    fig, axes = plt.subplots(
        len(top_patients),
        n_models,
        figsize=(5.0 * n_models, 3.8 * len(top_patients)),
        squeeze=False,
    )

    for col, (model_name, (model_cls, kwargs)) in enumerate(model_configs.items()):
        color = _COLORS.get(model_name, "#1f77b4")

        for row, patient in enumerate(top_patients):
            ax = axes[row, col]
            train_patients = [p for p in trajectories if p.patient_id != patient.patient_id]

            try:
                model = model_cls(**kwargs)
                model.fit(train_patients)

                t_min = patient.times[0] - 0.3
                t_max = patient.times[-1] + 0.5
                t_grid = np.linspace(max(t_min, -0.5), t_max, 200)

                pred = model.predict(patient, t_grid, n_condition=1)
                mean = pred.mean[:, 0]
                std = np.sqrt(np.clip(pred.variance[:, 0], 0, None))

                ax.fill_between(t_grid, mean - 2 * std, mean + 2 * std, alpha=0.15, color=color)
                ax.fill_between(t_grid, mean - std, mean + std, alpha=0.3, color=color)
                ax.plot(t_grid, mean, "-", color=color, linewidth=1.5)

            except Exception as e:
                logger.warning(f"Combined sausage failed {patient.patient_id}/{model_name}: {e}")

            obs = patient.observations[:, 0]
            ax.plot(patient.times[0], obs[0], "o", color="black", markersize=8, zorder=5)
            if len(patient.times) > 1:
                ax.plot(
                    patient.times[1:],
                    obs[1:],
                    "D",
                    color="red",
                    markersize=7,
                    zorder=5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )

            if row == 0:
                ax.set_title(model_name, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{patient.patient_id}\nlog(V+1)", fontsize=9)
            else:
                ax.set_ylabel("")
            if row == len(top_patients) - 1:
                ax.set_xlabel("Timepoint (ordinal)")

    fig.suptitle(
        f"Model Comparison: GP Posterior Predictives ({source} volumes)\n"
        r"Black $\bullet$ = conditioning obs., Red $\diamondsuit$ = held-out obs.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"sausage_comparison_{source}.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / f"sausage_comparison_{source}.png", dpi=150)
    plt.close(fig)
    logger.info("Combined sausage comparison plot saved")


def _fig_gp_illustrative(
    all_trajectories: list,
    selected_patients: list,
    model_configs: dict[str, tuple],
    fig_dir: Path,
    source: str,
) -> None:
    """Illustrative GP posterior conditioned on ALL patient observations.

    Unlike the LOPO sausage plots, this fits on ALL patients and conditions on
    ALL of a patient's data. Shows the posterior *between* observed points.
    This figure is **not for evaluation**.
    """
    n_patients = len(selected_patients)
    n_models = len(model_configs)

    fig, axes = plt.subplots(
        n_patients,
        n_models,
        figsize=(5.0 * n_models, 3.5 * n_patients),
        squeeze=False,
    )

    for col, (model_name, (model_cls, kwargs)) in enumerate(model_configs.items()):
        color = _COLORS.get(model_name, "#1f77b4")

        try:
            model = model_cls(**kwargs)
            model.fit(all_trajectories)
        except Exception as e:
            logger.warning(f"Illustrative fit failed for {model_name}: {e}")
            for row in range(n_patients):
                axes[row, col].text(
                    0.5,
                    0.5,
                    "Fit failed",
                    transform=axes[row, col].transAxes,
                    ha="center",
                )
            continue

        for row, patient in enumerate(selected_patients):
            ax = axes[row, col]

            t_min = patient.times[0] - 0.5
            t_max = patient.times[-1] + 0.5
            t_grid = np.linspace(max(t_min, -0.5), t_max, 200)

            try:
                pred = model.predict(patient, t_grid, n_condition=None)
                mean = pred.mean[:, 0]
                std = np.sqrt(np.clip(pred.variance[:, 0], 0, None))

                ax.fill_between(
                    t_grid,
                    mean - 2 * std,
                    mean + 2 * std,
                    alpha=0.15,
                    color=color,
                )
                ax.fill_between(
                    t_grid,
                    mean - std,
                    mean + std,
                    alpha=0.3,
                    color=color,
                )
                ax.plot(t_grid, mean, "-", color=color, linewidth=1.5)

            except Exception as e:
                logger.warning(
                    f"Illustrative predict failed {patient.patient_id}/{model_name}: {e}"
                )

            obs = patient.observations[:, 0]
            ax.plot(
                patient.times,
                obs,
                "o",
                color="black",
                markersize=8,
                zorder=5,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

            if row == 0:
                ax.set_title(model_name, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{patient.patient_id}\nlog(V+1)", fontsize=9)
            else:
                ax.set_ylabel("")
            if row == n_patients - 1:
                ax.set_xlabel("Timepoint (ordinal)")

    fig.suptitle(
        f"GP Interpolation: Posterior Conditioned on All Observations ({source})\n"
        "Illustrative only (not for evaluation)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"sausage_illustrative_{source}.pdf", **_FIG_KWARGS)
    fig.savefig(fig_dir / f"sausage_illustrative_{source}.png", dpi=150)
    plt.close(fig)
    logger.info("Illustrative GP interpolation plot saved")


def _find_tumor_center(seg: np.ndarray) -> tuple[int, int, int]:
    """Find the center of mass of the whole tumor (any label > 0).

    Args:
        seg: Integer label volume, shape ``[D, H, W]``.

    Returns:
        (d, h, w) voxel coordinates of the tumor centroid.
        Falls back to the volume center if no tumor is present.
    """
    tumor = seg > 0
    if not tumor.any():
        return tuple(s // 2 for s in seg.shape)  # type: ignore[return-value]
    coords = np.argwhere(tumor)
    return tuple(coords.mean(axis=0).astype(int))  # type: ignore[return-value]


def _render_seg_overlay_single_channel(
    f: "h5py.File",
    patient_id: str,
    scan_indices: list[int],
    scan_ids: list[str],
    timepoint_idx: np.ndarray,
    seg_sources: dict[str, str],
    mri_channel: int,
    output_dir: Path,
) -> Path:
    """Render one segmentation overlay figure for a single MRI channel."""
    from matplotlib.lines import Line2D

    channel_names = ["FLAIR", "T1ce", "T1", "T2"]
    bg_name = channel_names[mri_channel]
    n_timepoints = len(scan_indices)
    n_sources = len(seg_sources)

    source_colors = [
        "#FF0000",  # Red — Manual (ground truth)
        "#00FF00",  # Green — first predicted model
        "#00BFFF",  # DeepSkyBlue — second predicted model
        "#FFD700",  # Gold — third
        "#FF69B4",  # Pink — fourth
        "#00FFFF",  # Cyan — fifth
    ]

    view_labels = ["Sagittal", "Coronal", "Axial"]

    fig, axes = plt.subplots(
        3,
        n_timepoints,
        figsize=(3.5 * n_timepoints, 3.5 * 3),
        squeeze=False,
    )

    for col, scan_idx in enumerate(scan_indices):
        tp = timepoint_idx[scan_idx]
        sid = scan_ids[scan_idx]

        mri = f["images"][scan_idx, mri_channel]  # [D, H, W]

        # Load WT masks from every source
        wt_masks: dict[str, np.ndarray] = {}
        for src_name, src_key in seg_sources.items():
            if src_key == "segs":
                seg = f["segs"][scan_idx, 0]  # [D, H, W] int8
            else:
                seg = f[src_key][scan_idx, 0]  # [D, H, W] int8
            wt_masks[src_name] = (seg > 0).astype(np.uint8)

        # Tumor center from first non-empty mask
        center = None
        for mask in wt_masks.values():
            if mask.any():
                center = _find_tumor_center(mask.astype(np.int8))
                break
        if center is None:
            center = tuple(s // 2 for s in mri.shape)

        d_c, h_c, w_c = center

        slices_data = [
            mri[:, h_c, :],  # Sagittal
            mri[:, :, w_c],  # Coronal
            mri[d_c, :, :],  # Axial
        ]
        slices_wt: dict[str, list[np.ndarray]] = {}
        for src_name, mask in wt_masks.items():
            slices_wt[src_name] = [
                mask[:, h_c, :],
                mask[:, :, w_c],
                mask[d_c, :, :],
            ]

        for row, (view_name, mri_slice) in enumerate(zip(view_labels, slices_data)):
            ax = axes[row, col]
            ax.imshow(mri_slice.T, cmap="gray", origin="lower", aspect="equal")

            for src_idx, (src_name, wt_slices) in enumerate(slices_wt.items()):
                wt_slice = wt_slices[row]
                if wt_slice.any():
                    color = source_colors[src_idx % len(source_colors)]
                    ax.contour(
                        wt_slice.T,
                        levels=[0.5],
                        colors=[color],
                        linewidths=1.5,
                        origin="lower",
                    )

            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={tp}\n{sid}", fontsize=9)
            if col == 0:
                ax.set_ylabel(view_name, fontsize=11, fontweight="bold")

    # Legend
    legend_handles = [
        Line2D([0], [0], color=source_colors[i % len(source_colors)], linewidth=2, label=name)
        for i, name in enumerate(seg_sources)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=n_sources,
        fontsize=10,
        frameon=True,
    )
    fig.suptitle(
        f"Longitudinal Segmentation: {patient_id} "
        f"({n_timepoints} timepoints, {bg_name} background)\n"
        f"Whole-tumor contours from {n_sources} source(s)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])

    out_path = output_dir / f"seg_overlay_{patient_id}_{bg_name}.png"
    fig.savefig(out_path, dpi=200)
    fig.savefig(output_dir / f"seg_overlay_{patient_id}_{bg_name}.pdf", **_FIG_KWARGS)
    plt.close(fig)
    return out_path


def generate_segmentation_overlay(
    h5_path: str,
    output_dir: str | Path,
    patient_id: str | None = None,
    min_timepoints: int = 4,
    mri_channels: list[int] | None = None,
) -> list[Path]:
    """Generate multi-timepoint segmentation overlay figures.

    Produces one figure per MRI channel (default: all four).  Auto-discovers
    all segmentation models stored in the H5 file.

    Args:
        h5_path: Path to MenGrowth.h5.
        output_dir: Directory for saved figures.
        patient_id: Patient to plot.  If ``None``, auto-selects.
        min_timepoints: Minimum timepoints for automatic selection.
        mri_channels: Which MRI channels to render.

    Returns:
        List of paths to saved PNG files.
    """
    import h5py

    if mri_channels is None:
        mri_channels = [0, 1, 2, 3]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_names = ["FLAIR", "T1ce", "T1", "T2"]
    saved: list[Path] = []

    with h5py.File(h5_path, "r") as f:
        scan_ids = [
            sid.decode() if isinstance(sid, bytes) else str(sid) for sid in f["scan_ids"][:]
        ]
        patient_ids_list = [
            pid.decode() if isinstance(pid, bytes) else str(pid) for pid in f["patient_ids"][:]
        ]
        timepoint_idx = f["timepoint_idx"][:].astype(int)

        # Build patient -> scan index mapping
        patient_scans: dict[str, list[int]] = {}
        for i, pid in enumerate(patient_ids_list):
            patient_scans.setdefault(pid, []).append(i)

        # Select patient
        if patient_id is None:
            candidates = {
                pid: idxs for pid, idxs in patient_scans.items() if len(idxs) >= min_timepoints
            }
            if not candidates:
                logger.warning(f"No patient with >= {min_timepoints} timepoints")
                return []
            patient_id = max(candidates, key=lambda p: len(candidates[p]))

        if patient_id not in patient_scans:
            logger.error(f"Patient {patient_id} not found in H5")
            return []

        scan_indices = sorted(patient_scans[patient_id], key=lambda i: timepoint_idx[i])

        # Discover segmentation sources
        seg_sources: dict[str, str] = {"Manual": "segs"}
        if "predicted_segs" in f:
            for model_name in f["predicted_segs"]:
                label_key = f"predicted_segs/{model_name}/labels"
                if (
                    label_key.rsplit("/", 1)[0] in f
                    and "labels" in f[f"predicted_segs/{model_name}"]
                ):
                    seg_sources[model_name] = label_key

        logger.info(
            f"Segmentation overlay: {patient_id} ({len(scan_indices)} tp), "
            f"{len(seg_sources)} source(s): {list(seg_sources.keys())}, "
            f"channels: {[channel_names[c] for c in mri_channels]}"
        )

        for ch in mri_channels:
            out_path = _render_seg_overlay_single_channel(
                f,
                patient_id,
                scan_indices,
                scan_ids,
                timepoint_idx,
                seg_sources,
                mri_channel=ch,
                output_dir=output_dir,
            )
            saved.append(out_path)
            logger.info(f"Segmentation overlay saved: {out_path}")

    return saved


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

    # Parse segmentation config
    seg_models, use_manual = parse_seg_config(cfg)
    seg_model_names = [mc.model_name for mc in seg_models]
    logger.info(f"Segmentation models: {seg_model_names}, use_manual={use_manual}")

    # =========================================================================
    # Phase 1: Multi-Model Segmentation + Volume Extraction
    # =========================================================================
    logger.info("=== Phase 1: Volume Extraction ===")
    extractor = SegmentationVolumeExtractor(cfg)
    volumes = extractor.extract_all(force_recompute=args.force_recompute)

    non_empty = [v for v in volumes if not v.is_empty_manual]
    for mn in seg_model_names:
        model_vols = [v for v in non_empty if mn in v.model_results]
        if model_vols:
            mean_wt = np.mean([v.model_results[mn].wt_dice for v in model_vols])
            mean_tc = np.mean([v.model_results[mn].tc_dice for v in model_vols])
            mean_et = np.mean([v.model_results[mn].et_dice for v in model_vols])
            logger.info(
                f"  {mn}: WT Dice={mean_wt:.3f}, TC Dice={mean_tc:.3f}, "
                f"ET Dice={mean_et:.3f} (n={len(model_vols)})"
            )

    # =========================================================================
    # Phase 2: Segmentation Comparison Report
    # =========================================================================
    logger.info("=== Phase 2: Segmentation Comparison ===")
    seg_report = generate_segmentation_report(volumes)
    for mn in seg_model_names:
        pm = seg_report.get("per_model", {}).get(mn, {})
        wt_stats = pm.get("per_region", {}).get("wt", {})
        if wt_stats:
            logger.info(
                f"  {mn}: WT Dice={wt_stats['dice_mean']:.3f} "
                f"(IQR {wt_stats.get('dice_q25', 0):.3f}-"
                f"{wt_stats.get('dice_q75', 0):.3f})"
            )

    # =========================================================================
    # Phase 3: Build Trajectories for Each Source
    # =========================================================================
    pred_target = cfg.get("prediction", {}).get("target", "absolute")
    logger.info(f"=== Phase 3: Build Trajectories (target={pred_target}) ===")
    sources: dict[str, list] = {}

    # Select trajectory builder based on prediction target
    if pred_target == "delta_v":
        build_fn = extractor.build_delta_trajectories
    else:
        build_fn = extractor.build_trajectories

    if use_manual:
        traj_manual = build_fn(volumes, "manual")
        sources["manual"] = traj_manual
        logger.info(f"  manual: {len(traj_manual)} patients")

    for mc in seg_models:
        traj = build_fn(volumes, mc.model_name)
        sources[mc.model_name] = traj
        logger.info(f"  {mc.model_name}: {len(traj)} patients")

    # =========================================================================
    # Phase 4: Multi-Model LOPO-CV (GP model x source)
    # =========================================================================
    logger.info("=== Phase 4: Multi-Model LOPO-CV ===")
    model_configs = _build_model_configs(cfg)
    evaluator = LOPOEvaluator()

    lopo_results: dict[str, LOPOResults] = {}

    for gp_name, (model_cls, kwargs) in model_configs.items():
        for source_name, trajectories in sources.items():
            key = f"{gp_name}_{source_name}"
            logger.info(f"--- LOPO-CV: {gp_name} on {source_name} volumes ---")
            try:
                results = evaluator.evaluate(model_cls, trajectories, **kwargs)
                lopo_results[key] = results

                r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("nan"))
                cal = results.aggregate_metrics.get("last_from_rest/calibration_95", float("nan"))
                logger.info(
                    f"  {key}: R2_log={r2:.4f}, Cal_95={cal:.3f}, "
                    f"folds={len(results.fold_results)}/"
                    f"{len(results.fold_results) + len(results.failed_folds)}"
                )
            except Exception as e:
                logger.error(f"  {key} FAILED: {e}")

    # =========================================================================
    # Phase 5: Save Results
    # =========================================================================
    logger.info("=== Phase 5: Save Results ===")
    save_all_results(lopo_results, volumes, seg_report, output_dir, sources)

    # =========================================================================
    # Phase 6: Generate Figures
    # =========================================================================
    logger.info("=== Phase 6: Generate Figures ===")
    generate_all_figures(
        lopo_results,
        volumes,
        output_dir,
        sources=sources,
        model_configs=model_configs,
        h5_path=cfg["paths"]["mengrowth_h5"],
        seg_model_names=seg_model_names,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 72)
    print("ABLATION A0: SEGMENT-BASED BASELINE — RESULTS")
    print("=" * 72)

    # Segmentation quality per model
    print("\n--- Segmentation Quality ---")
    for mn in seg_model_names:
        pm = seg_report.get("per_model", {}).get(mn, {})
        print(f"\n  Model: {mn}")
        for region in ["wt", "tc", "et"]:
            stats = pm.get("per_region", {}).get(region, {})
            if not stats:
                continue
            print(
                f"    {region.upper():>2}: Dice = {stats['dice_mean']:.3f} "
                f"+/- {stats['dice_std']:.3f} "
                f"(median {stats['dice_median']:.3f}, "
                f"range [{stats['dice_min']:.3f}, {stats['dice_max']:.3f}])"
            )
            print(
                f"         Vol corr r={stats['volume_pearson_r']:.3f}, "
                f"MAE={stats['volume_mae_mm3']:.0f}mm3, "
                f"bias={stats['volume_mean_bias_mm3']:.0f}mm3"
            )

    # LOPO results per source
    for source_name in sources:
        print(f"\n--- LOPO-CV Results ({source_name} volumes) ---")
        print(
            f"  {'Model':<12} {'R2_log':>8} {'MAE_log':>8} {'RMSE_log':>9} "
            f"{'Cal_95':>7} {'CI_width':>8} {'R2_orig':>8}"
        )
        print("  " + "-" * 65)

        for gp_name in model_configs:
            key = f"{gp_name}_{source_name}"
            if key not in lopo_results:
                continue
            m = lopo_results[key].aggregate_metrics
            print(
                f"  {gp_name:<12} "
                f"{m.get('last_from_rest/r2_log', float('nan')):>8.4f} "
                f"{m.get('last_from_rest/mae_log', float('nan')):>8.4f} "
                f"{m.get('last_from_rest/rmse_log', float('nan')):>9.4f} "
                f"{m.get('last_from_rest/calibration_95', float('nan')):>7.3f} "
                f"{m.get('last_from_rest/mean_ci_width_log', float('nan')):>8.4f} "
                f"{m.get('last_from_rest/r2_original', float('nan')):>8.4f}"
            )


if __name__ == "__main__":
    main()
