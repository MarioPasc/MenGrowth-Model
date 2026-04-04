"""Fig 2: Segmentation Performance Comparison.

Box plots: Baseline vs Individual Members (pooled) vs Ensemble, for WT Dice.
Annotated with statistical significance brackets.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import (
    C_BASELINE,
    C_ENSEMBLE,
    C_MEMBERS,
    add_stat_bracket,
    significance_label,
)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the performance comparison figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Optional pre-created axes.

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [4.5, 3.5])
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    baseline_wt = data.baseline_dice["dice_wt"].values
    member_wt = data.per_member_dice["dice_wt"].values
    ensemble_wt = data.ensemble_dice["dice_wt"].values

    box_data = [baseline_wt, member_wt, ensemble_wt]
    positions = [0, 1, 2]
    colors = [C_BASELINE, C_MEMBERS, C_ENSEMBLE]
    labels = [
        f"Frozen BSF\n(n={len(baseline_wt)})",
        f"Individual\nmembers\n(n={len(member_wt)})",
        f"Ensemble\n(n={len(ensemble_wt)})",
    ]

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.5, patch_artist=True,
        showfliers=False, medianprops=dict(color="k", lw=1.2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Overlay individual points (jittered)
    if config.get("show_individual_points", True):
        rng = np.random.RandomState(42)
        pt_size = config.get("point_size", 6)
        pt_alpha = config.get("point_alpha", 0.3)
        for pos, d, c in zip(positions, box_data, colors):
            jitter = rng.uniform(-0.15, 0.15, size=len(d))
            ax.scatter(pos + jitter, d, s=pt_size, alpha=pt_alpha,
                       color=c, edgecolors="none", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Dice (Whole Tumor)")
    ax.set_ylim(-0.05, 1.05)

    # Statistical brackets
    evb = data.statistical_summary["ensemble_vs_baseline"]["wt"]
    p_val = evb["p_value_wilcoxon"]
    d_val = evb["cohens_d"]
    sig = significance_label(p_val)

    y_top = max(
        np.percentile(ensemble_wt, 95),
        np.percentile(baseline_wt, 95),
    ) + 0.02
    add_stat_bracket(ax, 0, 2, y_top, 0.03, f"{sig}  d={d_val:.2f}")

    ax.set_title("Whole Tumor Dice: Baseline vs Ensemble",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig
