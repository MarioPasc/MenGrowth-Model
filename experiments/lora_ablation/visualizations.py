#!/usr/bin/env python
# experiments/lora_ablation/visualizations.py
"""Publication-quality visualizations for LoRA ablation study.

Generates IEEE-compliant figures for thesis:
1. R² comparison with confidence intervals
2. Dice score comparison across conditions
3. Latent space UMAP (Glioma vs Meningioma domain shift)
4. Training curves comparison
5. Effect size forest plot

Usage:
    python -m experiments.lora_ablation.visualizations \
        --config experiments/lora_ablation/config/ablation.yaml \
        --all
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import yaml

from .settings import (
    apply_ieee_style,
    get_figure_size,
    get_significance_stars,
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_MARKERS,
    SEMANTIC_COLORS,
    SEMANTIC_LABELS_SHORT,
    DICE_COLORS,
    DOMAIN_COLORS,
    DOMAIN_MARKERS,
    DOMAIN_LABELS,
    PLOT_SETTINGS,
)
from .statistical_analysis import (
    bootstrap_ci,
    run_statistical_analysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def setup_style():
    """Apply IEEE style settings."""
    apply_ieee_style()


def save_figure(fig, output_dir: Path, name: str, formats: List[str] = ["pdf", "png"]):
    """Save figure in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
        logger.info(f"Saved {path}")


def add_significance_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
):
    """Add significance bracket with stars above bars."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=PLOT_SETTINGS["significance_bracket_linewidth"], c="0.3")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=PLOT_SETTINGS["significance_text_fontsize"])


# =============================================================================
# Figure 1: R² Comparison Bar Plot
# =============================================================================

def plot_r2_comparison(
    config: dict,
    output_dir: Path,
    show_ci: bool = True,
) -> plt.Figure:
    """Create R² comparison bar plot with confidence intervals.

    This is the PRIMARY figure showing linear probe R² for each condition.
    """
    setup_style()

    # Load metrics for all conditions
    conditions = [c["name"] for c in config["conditions"]]
    metrics = {}

    for cond in conditions:
        metrics_path = output_dir / "conditions" / cond / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics[cond] = json.load(f)

    if not metrics:
        logger.warning("No metrics found for R² plot")
        return None

    # Prepare data
    semantic_features = ["volume", "location", "shape"]
    n_conditions = len(metrics)
    n_features = len(semantic_features)

    fig, ax = plt.subplots(figsize=get_figure_size("double", 0.5))

    # Bar positions
    x = np.arange(n_features)
    bar_width = PLOT_SETTINGS["bar_width"]
    offsets = np.linspace(-(n_conditions - 1) / 2, (n_conditions - 1) / 2, n_conditions) * bar_width * 1.2

    # Plot bars for each condition
    for i, cond in enumerate(conditions):
        if cond not in metrics:
            continue

        values = [metrics[cond].get(f"r2_{feat}", 0) for feat in semantic_features]

        # Get per-dimension values for error bars if available
        errors = None
        if show_ci:
            per_dim = [metrics[cond].get(f"r2_{feat}_per_dim", None) for feat in semantic_features]
            if all(p is not None for p in per_dim):
                # Use std of per-dimension R² as error estimate
                errors = [np.std(p) for p in per_dim]

        bars = ax.bar(
            x + offsets[i],
            values,
            bar_width,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLORS.get(cond, f"C{i}"),
            alpha=PLOT_SETTINGS["bar_alpha"],
            edgecolor="white",
            linewidth=0.5,
            yerr=errors if errors else None,
            capsize=PLOT_SETTINGS["errorbar_capsize"] if errors else 0,
        )

    # Customize axes
    ax.set_xlabel("Semantic Feature", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_ylabel(r"Linear Probe $R^2$", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_xticks(x)
    ax.set_xticklabels([SEMANTIC_LABELS_SHORT[f] for f in semantic_features])
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.9, color="0.7", linestyle="--", linewidth=0.5, label="Target (0.9)")

    # Legend
    ax.legend(
        loc="upper right",
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=False,
        ncol=2,
    )

    # Add mean R² as text annotation
    for i, cond in enumerate(metrics.keys()):
        mean_r2 = metrics[cond].get("r2_mean", 0)
        ax.annotate(
            f"Mean: {mean_r2:.3f}",
            xy=(0.02, 0.98 - i * 0.06),
            xycoords="axes fraction",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            color=CONDITION_COLORS.get(cond, f"C{i}"),
            va="top",
        )

    plt.tight_layout()
    save_figure(fig, output_dir / "figures", "r2_comparison")
    return fig


# =============================================================================
# Figure 2: Dice Score Comparison
# =============================================================================

def plot_dice_comparison(
    config: dict,
    output_dir: Path,
) -> plt.Figure:
    """Create Dice score comparison plot."""
    setup_style()

    conditions = [c["name"] for c in config["conditions"]]
    dice_data = {}

    for cond in conditions:
        summary_path = output_dir / "conditions" / cond / "training_summary.yaml"
        log_path = output_dir / "conditions" / cond / "training_log.csv"

        if summary_path.exists():
            with open(summary_path) as f:
                summary = yaml.safe_load(f)
            dice_data[cond] = {"best_dice": summary.get("best_val_dice", 0)}

            # Load per-class dice if available
            if log_path.exists():
                df = pd.read_csv(log_path)
                if "val_dice_0" in df.columns:
                    best_idx = df["val_dice_mean"].idxmax()
                    dice_data[cond]["dice_per_class"] = [
                        df.loc[best_idx, "val_dice_0"],
                        df.loc[best_idx, "val_dice_1"],
                        df.loc[best_idx, "val_dice_2"],
                    ]

    if not dice_data:
        logger.warning("No Dice data found")
        return None

    fig, ax = plt.subplots(figsize=get_figure_size("single", 0.8))

    # Simple bar plot of best dice
    conds = list(dice_data.keys())
    values = [dice_data[c]["best_dice"] for c in conds]
    colors = [CONDITION_COLORS.get(c, "gray") for c in conds]

    bars = ax.bar(
        range(len(conds)),
        values,
        color=colors,
        alpha=PLOT_SETTINGS["bar_alpha"],
        edgecolor="white",
    )

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
        )

    ax.set_xlabel("Condition", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_ylabel("Best Validation Dice", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conds], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    save_figure(fig, output_dir / "figures", "dice_comparison")
    return fig


# =============================================================================
# Figure 3: Training Curves
# =============================================================================

def plot_training_curves(
    config: dict,
    output_dir: Path,
) -> plt.Figure:
    """Plot training curves for all conditions."""
    setup_style()

    conditions = [c["name"] for c in config["conditions"]]
    fig, axes = plt.subplots(1, 2, figsize=get_figure_size("double", 0.45))

    for cond in conditions:
        log_path = output_dir / "conditions" / cond / "training_log.csv"
        if not log_path.exists():
            continue

        df = pd.read_csv(log_path)
        color = CONDITION_COLORS.get(cond, "gray")
        label = CONDITION_LABELS.get(cond, cond)
        linestyle = "-" if cond != "baseline" else "--"

        # Training loss
        axes[0].plot(
            df["epoch"],
            df["train_loss"],
            color=color,
            linestyle=linestyle,
            linewidth=PLOT_SETTINGS["line_width"],
            label=label,
        )

        # Validation Dice
        axes[1].plot(
            df["epoch"],
            df["val_dice_mean"],
            color=color,
            linestyle=linestyle,
            linewidth=PLOT_SETTINGS["line_width"],
            label=label,
        )

    # Customize axes
    axes[0].set_xlabel("Epoch", fontsize=PLOT_SETTINGS["axes_labelsize"])
    axes[0].set_ylabel("Training Loss", fontsize=PLOT_SETTINGS["axes_labelsize"])
    axes[0].set_title("(a) Training Loss", fontsize=PLOT_SETTINGS["axes_titlesize"])

    axes[1].set_xlabel("Epoch", fontsize=PLOT_SETTINGS["axes_labelsize"])
    axes[1].set_ylabel("Validation Dice", fontsize=PLOT_SETTINGS["axes_labelsize"])
    axes[1].set_title("(b) Validation Dice", fontsize=PLOT_SETTINGS["axes_titlesize"])
    axes[1].legend(
        loc="lower right",
        fontsize=PLOT_SETTINGS["legend_fontsize"] - 1,
        frameon=False,
    )

    plt.tight_layout()
    save_figure(fig, output_dir / "figures", "training_curves")
    return fig


# =============================================================================
# Figure 4: Latent Space UMAP (Domain Shift Visualization)
# =============================================================================

def plot_latent_space_umap(
    config: dict,
    output_dir: Path,
    glioma_features_path: Optional[Path] = None,
    n_samples: int = 500,
) -> plt.Figure:
    """Plot UMAP of latent space showing Glioma vs Meningioma separation.

    This visualization shows how LoRA adaptation affects the feature space
    relative to the source domain (glioma).

    Args:
        config: Experiment configuration.
        output_dir: Output directory.
        glioma_features_path: Optional path to pre-extracted glioma features.
            If None, only meningioma features are shown.
        n_samples: Number of samples per condition for visualization.
    """
    setup_style()

    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed. Skipping UMAP visualization.")
        logger.warning("Install with: pip install umap-learn")
        return None

    import torch

    conditions = [c["name"] for c in config["conditions"]]

    # Load meningioma features for each condition
    all_features = []
    all_labels = []
    all_conditions = []

    for cond in conditions:
        feat_path = output_dir / "conditions" / cond / "features_test.pt"
        if not feat_path.exists():
            continue

        features = torch.load(feat_path).numpy()

        # Subsample if needed
        if len(features) > n_samples:
            indices = np.random.choice(len(features), n_samples, replace=False)
            features = features[indices]

        all_features.append(features)
        all_labels.extend(["meningioma"] * len(features))
        all_conditions.extend([cond] * len(features))

    if not all_features:
        logger.warning("No features found for UMAP")
        return None

    # Optionally add glioma features
    if glioma_features_path and glioma_features_path.exists():
        glioma_features = torch.load(glioma_features_path).numpy()
        if len(glioma_features) > n_samples:
            indices = np.random.choice(len(glioma_features), n_samples, replace=False)
            glioma_features = glioma_features[indices]
        all_features.append(glioma_features)
        all_labels.extend(["glioma"] * len(glioma_features))
        all_conditions.extend(["glioma"] * len(glioma_features))

    # Concatenate and fit UMAP
    features_concat = np.vstack(all_features)
    labels = np.array(all_labels)
    conditions_arr = np.array(all_conditions)

    logger.info(f"Fitting UMAP on {len(features_concat)} samples...")
    reducer = UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(features_concat)

    # Create figure
    n_cols = min(len(conditions), 4)
    n_rows = (len(conditions) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(PLOT_SETTINGS["figure_width_double"],
                 PLOT_SETTINGS["figure_width_double"] * 0.3 * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    # Plot each condition separately
    for idx, cond in enumerate(conditions):
        ax = axes[idx]

        # Plot glioma points (if available) as background
        if "glioma" in labels:
            glioma_mask = labels == "glioma"
            ax.scatter(
                embedding[glioma_mask, 0],
                embedding[glioma_mask, 1],
                c=DOMAIN_COLORS["glioma"],
                marker=DOMAIN_MARKERS["glioma"],
                s=PLOT_SETTINGS["scatter_size"],
                alpha=PLOT_SETTINGS["scatter_alpha"] * 0.5,
                label="Glioma (Source)",
                edgecolors="none",
            )

        # Plot meningioma points for this condition
        cond_mask = conditions_arr == cond
        ax.scatter(
            embedding[cond_mask, 0],
            embedding[cond_mask, 1],
            c=CONDITION_COLORS.get(cond, "blue"),
            marker=DOMAIN_MARKERS["meningioma"],
            s=PLOT_SETTINGS["scatter_size"],
            alpha=PLOT_SETTINGS["scatter_alpha"],
            label="Meningioma",
            edgecolors="white",
            linewidths=PLOT_SETTINGS["scatter_edgewidth"],
        )

        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=PLOT_SETTINGS["axes_titlesize"])
        ax.set_xlabel("UMAP 1", fontsize=PLOT_SETTINGS["tick_labelsize"])
        ax.set_ylabel("UMAP 2", fontsize=PLOT_SETTINGS["tick_labelsize"])
        ax.tick_params(labelsize=PLOT_SETTINGS["tick_labelsize"] - 1)

        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(len(conditions), len(axes)):
        axes[idx].set_visible(False)

    # Add legend to last visible axis
    if "glioma" in labels:
        legend_elements = [
            Line2D([0], [0], marker=DOMAIN_MARKERS["glioma"], color="w",
                   markerfacecolor=DOMAIN_COLORS["glioma"], markersize=8, label="Glioma (Source)"),
            Line2D([0], [0], marker=DOMAIN_MARKERS["meningioma"], color="w",
                   markerfacecolor=DOMAIN_COLORS["meningioma"], markersize=8, label="Meningioma (Target)"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            fontsize=PLOT_SETTINGS["legend_fontsize"],
            frameon=False,
        )

    plt.tight_layout()
    save_figure(fig, output_dir / "figures", "latent_umap")
    return fig


# =============================================================================
# Figure 5: Effect Size Forest Plot
# =============================================================================

def plot_effect_size_forest(
    config: dict,
    output_dir: Path,
) -> plt.Figure:
    """Create forest plot of effect sizes with confidence intervals.

    This plot clearly shows the magnitude and uncertainty of improvements.
    """
    setup_style()

    # Load statistical analysis results
    stats_path = output_dir / "statistical_comparisons.json"
    if not stats_path.exists():
        logger.warning("Statistical comparisons not found. Run statistical_analysis.py first.")
        return None

    with open(stats_path) as f:
        comparisons = json.load(f)

    if not comparisons:
        return None

    # Prepare data for forest plot
    conditions = list(comparisons.keys())
    metrics = ["volume_neg_mse", "location_neg_mse", "shape_neg_mse"]
    metric_labels = [r"$R^2_\mathrm{vol}$", r"$R^2_\mathrm{loc}$", r"$R^2_\mathrm{shape}$"]

    fig, ax = plt.subplots(figsize=get_figure_size("single", 1.2))

    y_positions = []
    y_labels = []
    y_pos = 0

    for cond in conditions:
        if cond not in comparisons:
            continue

        for metric, label in zip(metrics, metric_labels):
            if metric not in comparisons[cond]:
                continue

            comp = comparisons[cond][metric]

            # Extract effect size and CI
            effect = comp.get("effect_size", 0)
            delta_ci = comp.get("delta_ci", [0, 0])

            # Plot point and error bar
            ax.errorbar(
                effect,
                y_pos,
                xerr=[[effect - delta_ci[0]], [delta_ci[1] - effect]],
                fmt="o",
                color=CONDITION_COLORS.get(cond, "gray"),
                markersize=PLOT_SETTINGS["marker_size"],
                capsize=PLOT_SETTINGS["errorbar_capsize"],
                capthick=PLOT_SETTINGS["errorbar_capthick"],
                linewidth=PLOT_SETTINGS["errorbar_linewidth"],
            )

            # Add significance marker
            p_value = comp.get("p_corrected", comp.get("p_value", 1))
            if p_value < 0.05:
                marker = "*" * (3 if p_value < 0.001 else 2 if p_value < 0.01 else 1)
                ax.text(
                    effect + 0.05,
                    y_pos,
                    marker,
                    va="center",
                    fontsize=PLOT_SETTINGS["significance_text_fontsize"],
                )

            y_labels.append(f"{CONDITION_LABELS.get(cond, cond)}\n{label}")
            y_positions.append(y_pos)
            y_pos += 1

        # Add spacing between conditions
        y_pos += 0.5

    # Reference line at 0
    ax.axvline(x=0, color="0.5", linestyle="--", linewidth=0.8)

    # Shade regions
    ax.axvspan(-0.2, 0.2, alpha=0.1, color="gray", label="Negligible")
    ax.axvspan(0.2, 0.5, alpha=0.1, color="green", label="Small")
    ax.axvspan(0.5, 0.8, alpha=0.15, color="green", label="Medium")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=PLOT_SETTINGS["tick_labelsize"] - 1)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_title("Effect Size Forest Plot", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # Invert y-axis so first condition is at top
    ax.invert_yaxis()

    plt.tight_layout()
    save_figure(fig, output_dir / "figures", "effect_size_forest")
    return fig


# =============================================================================
# Figure 6: Delta R² Summary Plot
# =============================================================================

def plot_delta_r2_summary(
    config: dict,
    output_dir: Path,
) -> plt.Figure:
    """Create summary plot showing improvement over baseline."""
    setup_style()

    conditions = [c["name"] for c in config["conditions"] if c["name"] != "baseline"]

    # Load metrics
    baseline_metrics_path = output_dir / "conditions" / "baseline" / "metrics.json"
    if not baseline_metrics_path.exists():
        return None

    with open(baseline_metrics_path) as f:
        baseline_metrics = json.load(f)

    baseline_r2 = baseline_metrics.get("r2_mean", 0)

    deltas = {}
    for cond in conditions:
        metrics_path = output_dir / "conditions" / cond / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            deltas[cond] = {
                "r2_vol": metrics.get("r2_volume", 0) - baseline_metrics.get("r2_volume", 0),
                "r2_loc": metrics.get("r2_location", 0) - baseline_metrics.get("r2_location", 0),
                "r2_shape": metrics.get("r2_shape", 0) - baseline_metrics.get("r2_shape", 0),
                "r2_mean": metrics.get("r2_mean", 0) - baseline_r2,
            }

    if not deltas:
        return None

    fig, ax = plt.subplots(figsize=get_figure_size("single", 0.8))

    # Plot delta R² mean for each condition
    x = np.arange(len(deltas))
    conds = list(deltas.keys())
    values = [deltas[c]["r2_mean"] * 100 for c in conds]  # Convert to percentage
    colors = [CONDITION_COLORS.get(c, "gray") for c in conds]

    bars = ax.bar(x, values, color=colors, alpha=PLOT_SETTINGS["bar_alpha"])

    # Color bars by improvement direction
    for bar, val in zip(bars, values):
        if val < 0:
            bar.set_color("#EE6677")  # Red for worse
        bar.set_edgecolor("white")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        sign = "+" if val >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.2 if val >= 0 else -0.5),
            f"{sign}{val:.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
        )

    ax.axhline(y=0, color="0.3", linewidth=0.8)
    ax.set_xlabel("LoRA Condition", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_ylabel(r"$\Delta R^2_\mathrm{mean}$ (%)", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conds])

    # Add baseline reference annotation
    ax.annotate(
        f"Baseline: {baseline_r2:.3f}",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"),
    )

    plt.tight_layout()
    save_figure(fig, output_dir / "figures", "delta_r2_summary")
    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def generate_all_figures(config_path: str, glioma_features_path: Optional[str] = None):
    """Generate all publication figures."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating publication figures...")

    # Figure 1: R² Comparison
    logger.info("Figure 1: R² Comparison")
    plot_r2_comparison(config, output_dir)

    # Figure 2: Dice Comparison
    logger.info("Figure 2: Dice Comparison")
    plot_dice_comparison(config, output_dir)

    # Figure 3: Training Curves
    logger.info("Figure 3: Training Curves")
    plot_training_curves(config, output_dir)

    # Figure 4: Latent Space UMAP
    logger.info("Figure 4: Latent Space UMAP")
    glioma_path = Path(glioma_features_path) if glioma_features_path else None
    plot_latent_space_umap(config, output_dir, glioma_path)

    # Figure 5: Effect Size Forest Plot
    logger.info("Figure 5: Effect Size Forest Plot")
    plot_effect_size_forest(config, output_dir)

    # Figure 6: Delta R² Summary
    logger.info("Figure 6: Delta R² Summary")
    plot_delta_r2_summary(config, output_dir)

    logger.info(f"All figures saved to {figures_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate publication figures for LoRA ablation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )
    parser.add_argument(
        "--glioma-features",
        type=str,
        default=None,
        help="Optional path to pre-extracted glioma features for UMAP",
    )
    parser.add_argument(
        "--figure",
        type=str,
        choices=["r2", "dice", "curves", "umap", "forest", "delta", "all"],
        default="all",
        help="Which figure to generate",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    output_dir = Path(config["experiment"]["output_dir"])

    if args.figure == "all":
        generate_all_figures(args.config, args.glioma_features)
    elif args.figure == "r2":
        plot_r2_comparison(config, output_dir)
    elif args.figure == "dice":
        plot_dice_comparison(config, output_dir)
    elif args.figure == "curves":
        plot_training_curves(config, output_dir)
    elif args.figure == "umap":
        glioma_path = Path(args.glioma_features) if args.glioma_features else None
        plot_latent_space_umap(config, output_dir, glioma_path)
    elif args.figure == "forest":
        plot_effect_size_forest(config, output_dir)
    elif args.figure == "delta":
        plot_delta_r2_summary(config, output_dir)


if __name__ == "__main__":
    main()
