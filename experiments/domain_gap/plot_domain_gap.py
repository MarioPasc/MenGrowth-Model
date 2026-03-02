"""Generate domain gap figure from saved results (CPU-only).

Supports 2-domain (GLI + MEN) and 3-domain (GLI + MEN + MenGrowth) modes.

2-domain mode (default, ``--datasets gli men``):
  (a) PCA feature space with KDE density
  (b) Grouped Dice bar chart (GLI vs MEN)
  (c) Per-dimension KDE for top discriminative features

3-domain mode (``--datasets gli men mengrowth``):
  (a) PCA feature space with 3 domains
  (b) Grouped Dice bar chart with Kruskal-Wallis + post-hoc
  (c) Pairwise metrics summary (MMD², PAD, Clf. Acc.)

Usage:
    python -m experiments.domain_gap.plot_domain_gap --output-dir <path> [--datasets gli men mengrowth]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch
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

# Canonical domain ordering and short display names
_DOMAIN_ORDER = ["gli", "men", "mengrowth"]
_DOMAIN_COLOR_KEYS = {"gli": "glioma", "men": "meningioma", "mengrowth": "mengrowth"}
_DOMAIN_DISPLAY = {"gli": "BraTS-GLI", "men": "BraTS-MEN", "mengrowth": "MenGrowth"}

# Pair display names (for 3-domain metric panel)
_PAIR_DISPLAY = {
    "gli_men": "GLI\u2194MEN",
    "gli_mengrowth": "GLI\u2194MG",
    "men_mengrowth": "MEN\u2194MG",
}
_PAIR_COLORS = {
    "gli_men": "#882255",  # wine
    "gli_mengrowth": "#CC6677",  # rose
    "men_mengrowth": "#44AA99",  # teal
}


# ============================================================================
# Load Saved Results
# ============================================================================


def load_results(output_dir: Path, datasets: list[str]) -> dict:
    """Load saved results for the requested datasets.

    Supports both new (pairwise_metrics.json) and legacy (domain_metrics.json)
    output formats for backward compatibility.

    Args:
        output_dir: Root output directory from run_domain_gap.py.
        datasets: List of dataset keys to load (e.g. ["gli", "men"]).

    Returns:
        Dict with features, dice, pairwise_metrics, per_dataset_metrics, ks data.
    """
    result: dict = {"features": {}, "ids": {}, "dice": {}}

    for ds in datasets:
        feat_path = output_dir / "features" / f"{ds}_features.npz"
        if feat_path.exists():
            feat_data = np.load(feat_path, allow_pickle=True)
            result["features"][ds] = feat_data["features"]
            result["ids"][ds] = feat_data["subject_ids"]

        dice_path = output_dir / "dice" / f"{ds}_dice.json"
        if dice_path.exists():
            with open(dice_path) as f:
                result["dice"][ds] = json.load(f)

    # Pairwise metrics: try new format first, fall back to legacy
    pw_path = output_dir / "metrics" / "pairwise_metrics.json"
    legacy_path = output_dir / "metrics" / "domain_metrics.json"

    if pw_path.exists():
        with open(pw_path) as f:
            result["pairwise_metrics"] = json.load(f)
    elif legacy_path.exists():
        with open(legacy_path) as f:
            legacy = json.load(f)
        result["pairwise_metrics"] = {"gli_men": legacy}
    else:
        result["pairwise_metrics"] = {}

    # Per-dataset metrics
    pd_path = output_dir / "metrics" / "per_dataset_metrics.json"
    if pd_path.exists():
        with open(pd_path) as f:
            result["per_dataset_metrics"] = json.load(f)
    elif legacy_path.exists():
        with open(legacy_path) as f:
            legacy = json.load(f)
        result["per_dataset_metrics"] = {
            "gli": {"effective_rank": legacy.get("effective_rank_gli", 0)},
            "men": {"effective_rank": legacy.get("effective_rank_men", 0)},
        }
    else:
        result["per_dataset_metrics"] = {}

    # KS stats: try pair-specific files first, then legacy
    result["ks"] = {}
    for a, b in itertools.combinations(sorted(datasets), 2):
        pair_key = f"{a}_{b}"
        pair_path = output_dir / "metrics" / f"ks_stats_{pair_key}.npz"
        if pair_path.exists():
            ks_data = np.load(pair_path)
            result["ks"][pair_key] = (ks_data["ks_statistics"], ks_data["ks_pvalues"])

    # Legacy fallback for gli_men
    if "gli_men" not in result["ks"]:
        legacy_ks = output_dir / "metrics" / "ks_stats.npz"
        if legacy_ks.exists():
            ks_data = np.load(legacy_ks)
            result["ks"]["gli_men"] = (ks_data["ks_statistics"], ks_data["ks_pvalues"])

    return result


# ============================================================================
# Panel (a): PCA Feature Space — 2 domains
# ============================================================================


def plot_pca_with_density(
    ax: plt.Axes,
    gli_feat: np.ndarray,
    men_feat: np.ndarray,
    mmd_sq: float,
    mmd_pvalue: float,
) -> None:
    """Plot PCA projection with KDE density overlays (2 domains).

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
# Panel (a): PCA Feature Space — 3 domains
# ============================================================================


def plot_pca_3domain(
    ax: plt.Axes,
    feat_dict: dict[str, np.ndarray],
    pairwise_metrics: dict[str, dict],
) -> None:
    """Plot PCA projection with 3 domains.

    MEN: filled KDE contours (Blues). GLI: solid contour lines (rose).
    MenGrowth: dashed contour lines (green). All have scatter points.

    Args:
        ax: Matplotlib axes.
        feat_dict: Mapping dataset key -> features [N, D].
        pairwise_metrics: Pairwise metric dicts.
    """
    # Fit PCA on combined features
    all_feat = [feat_dict[d] for d in _DOMAIN_ORDER if d in feat_dict]
    combined = np.vstack(all_feat)
    pca = PCA(n_components=2)
    pca.fit(combined)

    projected = {d: pca.transform(feat_dict[d]) for d in feat_dict}
    var_explained = pca.explained_variance_ratio_ * 100

    # KDE grid
    all_x = np.concatenate([p[:, 0] for p in projected.values()])
    all_y = np.concatenate([p[:, 1] for p in projected.values()])
    x_margin = (all_x.max() - all_x.min()) * 0.15
    y_margin = (all_y.max() - all_y.min()) * 0.15

    xx, yy = np.mgrid[
        all_x.min() - x_margin : all_x.max() + x_margin : 100j,
        all_y.min() - y_margin : all_y.max() + y_margin : 100j,
    ]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # MEN: filled KDE (background)
    if "men" in projected:
        kde_men = stats.gaussian_kde(projected["men"].T)
        z_men = np.reshape(kde_men(positions), xx.shape)
        ax.contourf(xx, yy, z_men, levels=5, cmap="Blues", alpha=0.4)

    # GLI: solid contour lines
    if "gli" in projected:
        kde_gli = stats.gaussian_kde(projected["gli"].T)
        z_gli = np.reshape(kde_gli(positions), xx.shape)
        ax.contour(
            xx,
            yy,
            z_gli,
            levels=5,
            colors=DOMAIN_COLORS["glioma"],
            linewidths=1.0,
        )

    # MenGrowth: dashed contour lines
    if "mengrowth" in projected:
        kde_mg = stats.gaussian_kde(projected["mengrowth"].T)
        z_mg = np.reshape(kde_mg(positions), xx.shape)
        ax.contour(
            xx,
            yy,
            z_mg,
            levels=5,
            colors=DOMAIN_COLORS["mengrowth"],
            linewidths=1.0,
            linestyles="dashed",
        )

    # Scatter for all domains
    for ds in _DOMAIN_ORDER:
        if ds not in projected:
            continue
        pts = projected[ds]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=DOMAIN_COLORS[_DOMAIN_COLOR_KEYS[ds]],
            alpha=0.25,
            s=6,
            zorder=3,
        )

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")

    # Mini-table of pairwise MMD²
    mmd_lines = []
    for pair_key, label in _PAIR_DISPLAY.items():
        if pair_key in pairwise_metrics:
            mmd_val = pairwise_metrics[pair_key]["mmd_sq"]
            mmd_lines.append(f"{label}: {mmd_val:.2f}")
    if mmd_lines:
        mmd_text = "MMD$^2$\n" + "\n".join(mmd_lines)
        ax.annotate(
            mmd_text,
            xy=(0.03, 0.05),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=PLOT_SETTINGS["annotation_fontsize"] - 1,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.8),
            family="monospace",
        )


# ============================================================================
# Panel (b): Dice Comparison — 2 domains
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
# Panel (b): Dice Comparison — 3 domains
# ============================================================================


def _subject_avg_dice(dice_data: dict) -> list[dict]:
    """Average per-scan Dice by subject_id.

    If entries have no subject_id, returns per_subject as-is.

    Args:
        dice_data: Dice data dict with "per_subject" key.

    Returns:
        List of per-subject averaged Dice dicts.
    """
    entries = dice_data["per_subject"]
    if not entries or "subject_id" not in entries[0]:
        return entries

    from collections import defaultdict

    grouped: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        grouped[e["subject_id"]].append(e)

    averaged = []
    for sid, scans in sorted(grouped.items()):
        avg = {"subject_id": sid}
        for key in ["dice_TC", "dice_WT", "dice_ET"]:
            avg[key] = float(np.mean([s[key] for s in scans]))
        averaged.append(avg)

    return averaged


def plot_dice_3domain(
    ax: plt.Axes,
    dice_dict: dict[str, dict],
    datasets: list[str],
) -> None:
    """Plot grouped bar chart for 3 domains with Kruskal-Wallis + post-hoc.

    Args:
        ax: Matplotlib axes.
        dice_dict: Mapping dataset key -> dice data dict.
        datasets: Ordered list of dataset keys.
    """
    classes = ["TC", "WT", "ET", "Mean"]
    summary_keys = ["dice_TC", "dice_WT", "dice_ET", "dice_mean"]
    n_datasets = len(datasets)
    bar_width = 0.12 if n_datasets == 3 else PLOT_SETTINGS["bar_width"]

    x = np.arange(len(classes))

    # Draw bars
    offsets = np.linspace(
        -(n_datasets - 1) * bar_width / 2,
        (n_datasets - 1) * bar_width / 2,
        n_datasets,
    )

    for j, ds in enumerate(datasets):
        summary = dice_dict[ds]["summary"]
        means = [summary[f"{k}_mean"] for k in summary_keys]
        stds = [summary[f"{k}_std"] for k in summary_keys]
        color = DOMAIN_COLORS[_DOMAIN_COLOR_KEYS[ds]]

        ax.bar(
            x + offsets[j],
            means,
            bar_width,
            yerr=stds,
            color=color,
            alpha=PLOT_SETTINGS["bar_alpha"],
            label=_DOMAIN_DISPLAY[ds],
            capsize=PLOT_SETTINGS["errorbar_capsize"],
            error_kw={"linewidth": PLOT_SETTINGS["errorbar_linewidth"]},
        )

    # Kruskal-Wallis per class + post-hoc
    per_subj = {ds: _subject_avg_dice(dice_dict[ds]) for ds in datasets}

    for i, cls in enumerate(["dice_TC", "dice_WT", "dice_ET"]):
        groups = [[s[cls] for s in per_subj[ds]] for ds in datasets]
        h_stat, kw_p = stats.kruskal(*groups)
        stars = get_significance_stars(kw_p)

        # Position above tallest bar
        max_val = max(
            dice_dict[ds]["summary"][f"{cls}_mean"] + dice_dict[ds]["summary"][f"{cls}_std"]
            for ds in datasets
        )
        ax.text(
            x[i],
            max_val + 0.03,
            stars,
            ha="center",
            va="bottom",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
        )

    # Mean Dice Kruskal-Wallis
    mean_groups = [
        [(s["dice_TC"] + s["dice_WT"] + s["dice_ET"]) / 3 for s in per_subj[ds]] for ds in datasets
    ]
    _, kw_p = stats.kruskal(*mean_groups)
    stars = get_significance_stars(kw_p)
    max_val = max(
        dice_dict[ds]["summary"]["dice_mean_mean"] + dice_dict[ds]["summary"]["dice_mean_std"]
        for ds in datasets
    )
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
# Panel (c): Per-Dimension KDE — 2 domains
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
# Panel (c): Pairwise Metrics Summary — 3 domains
# ============================================================================


def plot_pairwise_summary(
    ax: plt.Axes,
    pairwise_metrics: dict[str, dict],
) -> None:
    """Grouped horizontal bar chart of pairwise metrics.

    Shows MMD², PAD, and Clf. Acc. for each domain pair.

    Args:
        ax: Matplotlib axes.
        pairwise_metrics: Mapping pair_key -> metrics dict.
    """
    metric_keys = [
        ("mmd_sq", "MMD$^2$"),
        ("proxy_a_distance", "PAD"),
        ("classifier_accuracy", "Clf. Acc."),
    ]

    # Collect pairs in canonical order
    pairs = [pk for pk in ["gli_men", "gli_mengrowth", "men_mengrowth"] if pk in pairwise_metrics]
    n_pairs = len(pairs)
    n_metrics = len(metric_keys)

    y = np.arange(n_metrics)
    bar_height = 0.2

    offsets = np.linspace(
        -(n_pairs - 1) * bar_height / 2,
        (n_pairs - 1) * bar_height / 2,
        n_pairs,
    )

    for j, pair_key in enumerate(pairs):
        vals = [pairwise_metrics[pair_key].get(mk, 0.0) for mk, _ in metric_keys]
        color = _PAIR_COLORS.get(pair_key, "#888888")
        label = _PAIR_DISPLAY.get(pair_key, pair_key)

        ax.barh(
            y + offsets[j],
            vals,
            bar_height,
            color=color,
            alpha=PLOT_SETTINGS["bar_alpha"],
            label=label,
        )

        # Value labels
        for i, v in enumerate(vals):
            fmt = f"{v:.2f}" if v < 1.0 else f"{v:.1f}"
            ax.text(
                v + 0.01,
                y[i] + offsets[j],
                fmt,
                va="center",
                ha="left",
                fontsize=PLOT_SETTINGS["annotation_fontsize"] - 1,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in metric_keys])
    ax.set_xlabel("Value")
    ax.legend(
        fontsize=PLOT_SETTINGS["legend_fontsize"] - 1,
        loc="lower right",
        frameon=False,
    )
    ax.invert_yaxis()


# ============================================================================
# Full Figure
# ============================================================================


def generate_figure(
    output_dir: Path,
    datasets: list[str],
    top_ks_dims: int = 4,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Generate the domain gap figure.

    Dispatches to 2-domain or 3-domain panel functions based on
    the number of datasets.

    Args:
        output_dir: Root output directory with saved results.
        datasets: Dataset keys to include.
        top_ks_dims: Number of top KS dimensions for 2-domain panel (c).
        dpi: Output DPI.
        formats: Output formats (default: ["pdf", "png"]).
    """
    if formats is None:
        formats = ["pdf", "png"]

    apply_ieee_style()

    data = load_results(output_dir, datasets)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    is_3domain = len(datasets) >= 3

    # Full-width figure, 1x3 panels
    fig_w, _ = get_figure_size("double")
    fig_h = fig_w * 0.33

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        1, 3, figure=fig, wspace=0.35, left=0.06, right=0.98, top=0.92, bottom=0.15
    )

    panel_fs = PLOT_SETTINGS["panel_label_fontsize"]

    if is_3domain:
        # Panel (a): 3-domain PCA
        ax_a = fig.add_subplot(gs[0])
        plot_pca_3domain(ax_a, data["features"], data["pairwise_metrics"])
        ax_a.set_title("(a)", fontsize=panel_fs, fontweight="bold", loc="left")

        # Panel (b): 3-domain Dice
        ax_b = fig.add_subplot(gs[1])
        plot_dice_3domain(ax_b, data["dice"], datasets)
        ax_b.set_title("(b)", fontsize=panel_fs, fontweight="bold", loc="left")

        # Panel (c): Pairwise metrics summary
        ax_c = fig.add_subplot(gs[2])
        plot_pairwise_summary(ax_c, data["pairwise_metrics"])
        ax_c.set_title("(c)", fontsize=panel_fs, fontweight="bold", loc="left")

        # Shared legend
        legend_handles = [
            Patch(
                facecolor=DOMAIN_COLORS[_DOMAIN_COLOR_KEYS[ds]],
                alpha=PLOT_SETTINGS["bar_alpha"],
                label=_DOMAIN_DISPLAY[ds],
            )
            for ds in datasets
        ]
    else:
        # Panel (a): 2-domain PCA
        ax_a = fig.add_subplot(gs[0])
        gli_men_metrics = data["pairwise_metrics"].get("gli_men", {})
        plot_pca_with_density(
            ax_a,
            data["features"]["gli"],
            data["features"]["men"],
            gli_men_metrics.get("mmd_sq", 0),
            gli_men_metrics.get("mmd_pvalue", 1),
        )
        ax_a.set_title("(a)", fontsize=panel_fs, fontweight="bold", loc="left")

        # Panel (b): 2-domain Dice
        ax_b = fig.add_subplot(gs[1])
        plot_dice_comparison(ax_b, data["dice"]["gli"], data["dice"]["men"])
        ax_b.set_title("(b)", fontsize=panel_fs, fontweight="bold", loc="left")

        # Panel (c): Top KS dims
        ax_c = fig.add_subplot(gs[2])
        ks_data = data["ks"].get("gli_men", (np.zeros(768), np.ones(768)))
        plot_top_ks_kde(
            ax_c,
            data["features"]["gli"],
            data["features"]["men"],
            ks_data[0],
            n_dims=top_ks_dims,
        )
        fig.text(
            gs[2].get_position(fig).x0,
            0.95,
            "(c)",
            fontsize=panel_fs,
            fontweight="bold",
            ha="left",
            va="top",
        )

        legend_handles = [
            Patch(
                facecolor=DOMAIN_COLORS["glioma"],
                alpha=PLOT_SETTINGS["bar_alpha"],
                label="BraTS-GLI",
            ),
            Patch(
                facecolor=DOMAIN_COLORS["meningioma"],
                alpha=PLOT_SETTINGS["bar_alpha"],
                label="BraTS-MEN",
            ),
        ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
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
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gli", "men"],
        choices=["gli", "men", "mengrowth"],
        help="Datasets to include in the figure",
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
        datasets=args.datasets,
        top_ks_dims=args.top_ks_dims,
        dpi=args.dpi,
    )
    logger.info("Figure generation complete.")


if __name__ == "__main__":
    main()
