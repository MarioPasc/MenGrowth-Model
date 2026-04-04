"""Fig 9: Inter-Member Agreement Heatmap.

Pairwise Pearson correlation heatmap between ensemble members on WT Dice,
annotated with mean pairwise r and ICC(3,1).
"""

from __future__ import annotations

import logging

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)

logger = logging.getLogger(__name__)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the inter-member agreement heatmap.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Optional pre-created axes.

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [5, 4.5])
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Pivot per_member_dice to [scan_id x member_id] on dice_wt
    pivot = data.per_member_dice.pivot(
        index="scan_id", columns="member_id", values="dice_wt",
    )
    M = pivot.shape[1]

    # Compute M x M pairwise Pearson correlation
    corr = pivot.corr()

    # Plot heatmap
    sns.heatmap(
        corr,
        ax=ax,
        vmin=0.5,
        vmax=1.0,
        cmap="RdYlBu_r",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        annot_kws={"fontsize": 7},
    )

    # Labels
    member_labels = [f"M{i}" for i in range(M)]
    ax.set_xticklabels(member_labels, fontsize=8)
    ax.set_yticklabels(member_labels, fontsize=8, rotation=0)

    # Title annotation with statistics
    agreement = data.statistical_summary.get("inter_member_agreement", {})
    mean_r = agreement.get("mean_pairwise_correlation_wt", float("nan"))
    icc = agreement.get("icc_wt", float("nan"))

    ax.set_title(
        f"Inter-member agreement (WT Dice)\n"
        f"Mean pairwise r = {mean_r:.3f}    ICC(3,1) = {icc:.3f}",
        fontweight="bold", fontsize=9,
    )

    fig.tight_layout()
    return fig
