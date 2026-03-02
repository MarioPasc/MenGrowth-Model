"""Generate domain gap figure from saved results (CPU-only).

Produces a 1x3 panel figure:
  (a) PCA feature space with KDE density
  (b) Grouped Dice bar chart (GLI vs MEN)
  (c) Per-dimension KDE for top discriminative features

Usage:
    python -m experiments.domain_gap.plot_domain_gap --output-dir <path> [--config <path>]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy import stats
from sklearn.decomposition import PCA

from experiments.utils.settings import (
    DOMAIN_COLORS,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
    get_significance_stars,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Load Saved Results
# ============================================================================


def load_results(output_dir: Path) -> dict:
    """Load all saved results from the output directory.

    Args:
        output_dir: Root output directory from run_domain_gap.py.

    Returns:
        Dict with gli_features, men_features, gli_dice, men_dice,
        domain_metrics, ks_statistics, ks_pvalues.
    """
    gli_feat_data = np.load(output_dir / "features" / "gli_features.npz", allow_pickle=True)
    men_feat_data = np.load(output_dir / "features" / "men_features.npz", allow_pickle=True)

    with open(output_dir / "dice" / "gli_dice.json") as f:
        gli_dice = json.load(f)
    with open(output_dir / "dice" / "men_dice.json") as f:
        men_dice = json.load(f)

    with open(output_dir / "metrics" / "domain_metrics.json") as f:
        domain_metrics = json.load(f)

    ks_data = np.load(output_dir / "metrics" / "ks_stats.npz")

    return {
        "gli_features": gli_feat_data["features"],
        "men_features": men_feat_data["features"],
        "gli_ids": gli_feat_data["subject_ids"],
        "men_ids": men_feat_data["subject_ids"],
        "gli_dice": gli_dice,
        "men_dice": men_dice,
        "domain_metrics": domain_metrics,
        "ks_statistics": ks_data["ks_statistics"],
        "ks_pvalues": ks_data["ks_pvalues"],
    }


# ============================================================================
# Panel (a): PCA Feature Space with KDE Density
# ============================================================================


def plot_pca_with_density(
    ax: plt.Axes,
    gli_feat: np.ndarray,
    men_feat: np.ndarray,
    mmd_sq: float,
    mmd_pvalue: float,
) -> None:
    """Plot PCA projection with KDE density overlays.

    Args:
        ax: Matplotlib axes.
        gli_feat: GLI features [N1, D].
        men_feat: MEN features [N2, D].
        mmd_sq: MMD² value for annotation.
        mmd_pvalue: MMD p-value for annotation.
    """
    # Fit PCA on combined data
    combined = np.vstack([gli_feat, men_feat])
    pca = PCA(n_components=2)
    pca.fit(combined)

    gli_2d = pca.transform(gli_feat)
    men_2d = pca.transform(men_feat)

    var_explained = pca.explained_variance_ratio_ * 100

    # Create grid for KDE evaluation
    x_all = np.concatenate([gli_2d[:, 0], men_2d[:, 0]])
    y_all = np.concatenate([gli_2d[:, 1], men_2d[:, 1]])
    x_margin = (x_all.max() - x_all.min()) * 0.15
    y_margin = (y_all.max() - y_all.min()) * 0.15

    xx, yy = np.mgrid[
        x_all.min() - x_margin : x_all.max() + x_margin : 100j,
        y_all.min() - y_margin : y_all.max() + y_margin : 100j,
    ]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # MEN: filled KDE colormap (background)
    kde_men = stats.gaussian_kde(men_2d.T)
    z_men = np.reshape(kde_men(positions), xx.shape)
    ax.contourf(xx, yy, z_men, levels=5, cmap="Blues", alpha=0.4)

    # GLI: contour lines (overlay)
    kde_gli = stats.gaussian_kde(gli_2d.T)
    z_gli = np.reshape(kde_gli(positions), xx.shape)
    ax.contour(
        xx,
        yy,
        z_gli,
        levels=5,
        colors=DOMAIN_COLORS["glioma"],
        linewidths=1.0,
    )

    # Scatter points
    n_gli = len(gli_2d)
    n_men = len(men_2d)
    ax.scatter(
        gli_2d[:, 0],
        gli_2d[:, 1],
        c=DOMAIN_COLORS["glioma"],
        alpha=0.3,
        s=8,
        zorder=3,
    )
    ax.scatter(
        men_2d[:, 0],
        men_2d[:, 1],
        c=DOMAIN_COLORS["meningioma"],
        alpha=0.3,
        s=8,
        zorder=3,
    )

    # Axes labels with variance explained
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")

    # MMD annotation with N
    n_total = n_gli + n_men
    p_str = f"p = {mmd_pvalue:.3f}" if mmd_pvalue >= 0.001 else "p < 0.001"
    ax.annotate(
        f"MMD$^2$ = {mmd_sq:.2f}, N={n_total}\n({p_str})",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.8),
    )


# ============================================================================
# Panel (b): Segmentation Dice Comparison
# ============================================================================


def plot_dice_comparison(
    ax: plt.Axes,
    gli_dice: dict,
    men_dice: dict,
) -> None:
    """Plot grouped bar chart of Dice scores (GLI vs MEN).

    Args:
        ax: Matplotlib axes.
        gli_dice: GLI Dice data with "per_subject" and "summary" keys.
        men_dice: MEN Dice data with same structure.
    """
    classes = ["TC", "WT", "ET", "Mean"]
    summary_keys = ["dice_TC", "dice_WT", "dice_ET", "dice_mean"]

    gli_means = [gli_dice["summary"][f"{k}_mean"] for k in summary_keys]
    gli_stds = [gli_dice["summary"][f"{k}_std"] for k in summary_keys]
    men_means = [men_dice["summary"][f"{k}_mean"] for k in summary_keys]
    men_stds = [men_dice["summary"][f"{k}_std"] for k in summary_keys]

    x = np.arange(len(classes))
    bar_width = PLOT_SETTINGS["bar_width"]

    ax.bar(
        x - bar_width / 2,
        gli_means,
        bar_width,
        yerr=gli_stds,
        color=DOMAIN_COLORS["glioma"],
        alpha=PLOT_SETTINGS["bar_alpha"],
        label="BraTS-GLI",
        capsize=PLOT_SETTINGS["errorbar_capsize"],
        error_kw={"linewidth": PLOT_SETTINGS["errorbar_linewidth"]},
    )
    ax.bar(
        x + bar_width / 2,
        men_means,
        bar_width,
        yerr=men_stds,
        color=DOMAIN_COLORS["meningioma"],
        alpha=PLOT_SETTINGS["bar_alpha"],
        label="BraTS-MEN",
        capsize=PLOT_SETTINGS["errorbar_capsize"],
        error_kw={"linewidth": PLOT_SETTINGS["errorbar_linewidth"]},
    )

    # Significance stars from Welch t-test
    gli_subjects = gli_dice["per_subject"]
    men_subjects = men_dice["per_subject"]

    for i, cls in enumerate(["dice_TC", "dice_WT", "dice_ET"]):
        gli_vals = [s[cls] for s in gli_subjects]
        men_vals = [s[cls] for s in men_subjects]
        _, p_val = stats.ttest_ind(gli_vals, men_vals, equal_var=False)
        stars = get_significance_stars(p_val)

        max_val = max(gli_means[i] + gli_stds[i], men_means[i] + men_stds[i])
        ax.text(
            x[i],
            max_val + 0.03,
            stars,
            ha="center",
            va="bottom",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
        )

    # Mean significance
    gli_mean_vals = [(s["dice_TC"] + s["dice_WT"] + s["dice_ET"]) / 3 for s in gli_subjects]
    men_mean_vals = [(s["dice_TC"] + s["dice_WT"] + s["dice_ET"]) / 3 for s in men_subjects]
    _, p_val = stats.ttest_ind(gli_mean_vals, men_mean_vals, equal_var=False)
    stars = get_significance_stars(p_val)
    max_val = max(gli_means[3] + gli_stds[3], men_means[3] + men_stds[3])
    ax.text(
        x[3],
        max_val + 0.03,
        stars,
        ha="center",
        va="bottom",
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


# ============================================================================
# Panel (c): Per-Dimension Feature KDE
# ============================================================================


def plot_top_ks_kde(
    ax: plt.Axes,
    gli_feat: np.ndarray,
    men_feat: np.ndarray,
    ks_statistics: np.ndarray,
    n_dims: int = 4,
) -> None:
    """Plot KDE curves for top discriminative feature dimensions.

    Args:
        ax: Matplotlib axes (will be subdivided).
        gli_feat: GLI features [N1, D].
        men_feat: MEN features [N2, D].
        ks_statistics: Per-dimension KS statistics [D].
        n_dims: Number of top dimensions to plot.
    """
    # Turn off the parent axes frame
    ax.set_axis_off()

    # Get position of the parent axes
    pos = ax.get_position()

    # Create sub-axes within the parent axes area
    fig = ax.figure
    inner_gs = GridSpecFromSubplotSpec(
        n_dims,
        1,
        subplot_spec=ax.get_subplotspec(),
        hspace=0.4,
    )

    # Top n_dims by KS statistic
    top_dims = np.argsort(ks_statistics)[::-1][:n_dims]

    for k, dim_idx in enumerate(top_dims):
        sub_ax = fig.add_subplot(inner_gs[k])

        gli_vals = gli_feat[:, dim_idx]
        men_vals = men_feat[:, dim_idx]

        # KDE curves
        x_min = min(gli_vals.min(), men_vals.min())
        x_max = max(gli_vals.max(), men_vals.max())
        margin = (x_max - x_min) * 0.1
        x_range = np.linspace(x_min - margin, x_max + margin, 200)

        kde_gli = stats.gaussian_kde(gli_vals)
        kde_men = stats.gaussian_kde(men_vals)

        sub_ax.fill_between(
            x_range,
            kde_gli(x_range),
            color=DOMAIN_COLORS["glioma"],
            alpha=0.3,
        )
        sub_ax.plot(
            x_range,
            kde_gli(x_range),
            color=DOMAIN_COLORS["glioma"],
            linewidth=1.0,
        )

        sub_ax.fill_between(
            x_range,
            kde_men(x_range),
            color=DOMAIN_COLORS["meningioma"],
            alpha=0.3,
        )
        sub_ax.plot(
            x_range,
            kde_men(x_range),
            color=DOMAIN_COLORS["meningioma"],
            linewidth=1.0,
        )

        # Annotation
        ks_val = ks_statistics[dim_idx]
        sub_ax.annotate(
            f"d{dim_idx} (D={ks_val:.2f})",
            xy=(0.97, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=PLOT_SETTINGS["annotation_fontsize"] - 1,
        )

        sub_ax.set_yticks([])
        if k < n_dims - 1:
            sub_ax.set_xticklabels([])
        else:
            sub_ax.set_xlabel("Feature value")

        # Minimal spines
        sub_ax.spines["top"].set_visible(False)
        sub_ax.spines["right"].set_visible(False)
        sub_ax.spines["left"].set_visible(False)


# ============================================================================
# Full Figure
# ============================================================================


def generate_figure(
    output_dir: Path,
    top_ks_dims: int = 4,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Generate the complete 1x3 domain gap figure.

    Args:
        output_dir: Root output directory with saved results.
        top_ks_dims: Number of top KS dimensions for panel (c).
        dpi: Output DPI.
        formats: Output formats (default: ["pdf", "png"]).
    """
    if formats is None:
        formats = ["pdf", "png"]

    apply_ieee_style()

    data = load_results(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Full-width figure, 1x3 panels
    fig_w, _ = get_figure_size("double")
    fig_h = fig_w * 0.33  # Wide format for 3 panels

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        1, 3, figure=fig, wspace=0.35, left=0.06, right=0.98, top=0.92, bottom=0.15
    )

    # Panel (a): PCA + KDE
    ax_a = fig.add_subplot(gs[0])
    plot_pca_with_density(
        ax_a,
        data["gli_features"],
        data["men_features"],
        data["domain_metrics"]["mmd_sq"],
        data["domain_metrics"]["mmd_pvalue"],
    )

    # Panel (b): Dice comparison
    ax_b = fig.add_subplot(gs[1])
    plot_dice_comparison(ax_b, data["gli_dice"], data["men_dice"])

    # Panel (c): Top KS dimensions
    ax_c = fig.add_subplot(gs[2])
    plot_top_ks_kde(
        ax_c,
        data["gli_features"],
        data["men_features"],
        data["ks_statistics"],
        n_dims=top_ks_dims,
    )

    # Panel labels
    panel_fs = PLOT_SETTINGS["panel_label_fontsize"]
    ax_a.set_title("(a)", fontsize=panel_fs, fontweight="bold", loc="left")
    ax_b.set_title("(b)", fontsize=panel_fs, fontweight="bold", loc="left")
    # Panel (c) label on figure directly since axes is off
    fig.text(
        gs[2].get_position(fig).x0,
        0.95,
        "(c)",
        fontsize=panel_fs,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # Shared legend at bottom center
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=DOMAIN_COLORS["glioma"], alpha=PLOT_SETTINGS["bar_alpha"], label="BraTS-GLI"),
        Patch(facecolor=DOMAIN_COLORS["meningioma"], alpha=PLOT_SETTINGS["bar_alpha"], label="BraTS-MEN"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=False,
        bbox_to_anchor=(0.5, -0.1),
    )

    # Save
    for fmt in formats:
        out_path = fig_dir / f"domain_gap.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved: {out_path}")

    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """Generate domain gap figure from saved results."""
    parser = argparse.ArgumentParser(description="Domain gap figure (CPU)")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory from run_domain_gap.py",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument("--top-ks-dims", type=int, default=4, help="Top KS dims for panel (c)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    generate_figure(
        output_dir=Path(args.output_dir),
        top_ks_dims=args.top_ks_dims,
        dpi=args.dpi,
    )
    logger.info("Figure generation complete.")


if __name__ == "__main__":
    main()
