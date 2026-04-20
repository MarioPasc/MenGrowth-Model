"""Fig: Per-region Dice violin plots with pairwise statistical comparisons.

Single-panel figure with violin plots grouped by model type (Frozen BSF,
Individual members, Ensemble). Each group shows two violins colored by
region (WT, ET). Statistical brackets annotate pairwise comparisons
with p-values and Cohen's d.

Bracket ordering (bottom to top):
  (a) Frozen BSF vs Individual  — WT, ET
  (b) Individual vs Ensemble    — WT, ET
  (c) Frozen BSF vs Ensemble    — WT, ET
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import (
    significance_label,
)

C_WT = "#0072B2"
C_ET = "#D55E00"


def _paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    std = diff.std(ddof=1)
    if std < 1e-10:
        return 0.0
    return float(diff.mean() / std)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the per-region violin plot figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Optional pre-created axes.

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [6, 4.5])
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Data extraction — per scan_id for paired tests
    baseline = data.baseline_dice.set_index("scan_id")
    ensemble = data.ensemble_dice.set_index("scan_id")
    individual = (
        data.per_member_dice.groupby("scan_id")[["dice_wt", "dice_et"]]
        .mean()
    )

    # Align on common scan_ids
    common = baseline.index.intersection(ensemble.index).intersection(individual.index)
    bl = baseline.loc[common]
    ind = individual.loc[common]
    ens = ensemble.loc[common]
    n_scans = len(common)

    regions = ["WT", "ET"]
    region_colors = [C_WT, C_ET]
    group_keys = ["Frozen BSF", "Individual", "Ensemble"]
    group_labels = [
        f"Frozen BSF\n(n={n_scans})",
        f"Individual\n(n={n_scans})",
        f"Ensemble\n(n={n_scans})",
    ]
    data_dict = {
        "Frozen BSF": {"WT": bl["dice_wt"].values, "ET": bl["dice_et"].values},
        "Individual": {"WT": ind["dice_wt"].values, "ET": ind["dice_et"].values},
        "Ensemble": {"WT": ens["dice_wt"].values, "ET": ens["dice_et"].values},
    }

    # Layout: 3 groups × 2 regions = 6 violins
    group_centers = [0, 3, 6]
    width = 0.85
    offset = 0.5

    for group, center in zip(group_keys, group_centers):
        for r_idx, (region, color) in enumerate(zip(regions, region_colors)):
            pos = center + (r_idx - 0.5) * (offset * 2)
            vals = data_dict[group][region]
            vals_clean = vals[np.isfinite(vals)]
            if len(vals_clean) < 3:
                continue

            parts = ax.violinplot(
                vals_clean, positions=[pos], widths=width,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
                pc.set_edgecolor("none")

            q25, med, q75 = np.percentile(vals_clean, [25, 50, 75])
            ax.scatter(
                [pos], [med], color=color, s=20, zorder=5,
                edgecolors="k", linewidths=0.5,
            )
            ax.vlines(pos, q25, q75, color=color, lw=2.0, zorder=4)

    # X-axis
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Dice Score")

    # --- Statistical brackets ---
    # Order: short spans first (adjacent groups), widest last (top).
    # Within each comparison: WT bracket then ET bracket.
    comparisons = [
        ("Frozen BSF", "Individual", 0, 3),
        ("Individual", "Ensemble", 3, 6),
        ("Frozen BSF", "Ensemble", 0, 6),
    ]

    bracket_y_start = 1.03
    intra_pair_gap = 0.048
    inter_group_gap = 0.028
    bracket_h = 0.01

    bracket_idx = 0
    for comp_idx, (name_a, name_b, center_a, center_b) in enumerate(comparisons):
        for r_idx, (region, color) in enumerate(zip(regions, region_colors)):
            a_vals = data_dict[name_a][region]
            b_vals = data_dict[name_b][region]

            try:
                stat_result = stats.wilcoxon(
                    a_vals, b_vals, alternative="two-sided",
                )
                p = stat_result.pvalue
            except (ValueError, ZeroDivisionError):
                p = 1.0
            d = _paired_cohens_d(b_vals, a_vals)

            x1 = center_a + (r_idx - 0.5) * offset * 2
            x2 = center_b + (r_idx - 0.5) * offset * 2
            y = (
                bracket_y_start
                + comp_idx * (2 * intra_pair_gap + inter_group_gap)
                + r_idx * intra_pair_gap
            )

            sig = significance_label(p)
            label = f"{sig} d={d:.2f}"

            ax.plot(
                [x1, x1, x2, x2],
                [y, y + bracket_h, y + bracket_h, y],
                lw=0.7, color=color,
            )
            ax.text(
                (x1 + x2) / 2, y + bracket_h + 0.005,
                label, ha="center", va="bottom", fontsize=6.5, color=color,
            )
            bracket_idx += 1

    # Keep Dice-axis ticks clean (0–1) but extend ylim for brackets
    top_y = (
        bracket_y_start
        + 2 * (2 * intra_pair_gap + inter_group_gap)
        + intra_pair_gap
        + bracket_h + 0.05
    )
    ax.set_ylim(-0.05, top_y)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    # Legend — outside the plot, at the bottom, full label names
    patches = [
        mpatches.Patch(
            facecolor=C_WT, alpha=0.55, label="Whole Tumor (WT)",
        ),
        mpatches.Patch(
            facecolor=C_ET, alpha=0.55, label="Enhancing Tumor (ET)",
        ),
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=2,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    return fig
