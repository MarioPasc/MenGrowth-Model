"""Fig 10: Volume Bland-Altman (Ensemble vs Ground Truth).

Assesses systematic bias and proportional error in the ensemble's volume
estimates on the BraTS-MEN test set.
"""

from __future__ import annotations

import logging

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import (
    C_DELTA_NEG,
    C_DELTA_POS,
    C_ENSEMBLE,
)

logger = logging.getLogger(__name__)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the Bland-Altman plot.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Optional pre-created axes.

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [4.5, 4])
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ens = data.ensemble_dice
    v_ens = ens["volume_ensemble"].values.astype(float)
    v_gt = ens["volume_gt"].values.astype(float)

    # Bland-Altman quantities
    v_mean = (v_ens + v_gt) / 2.0
    v_diff = v_ens - v_gt

    bias = float(np.mean(v_diff))
    sd = float(np.std(v_diff, ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # Color by sign of difference
    colors = np.where(v_diff >= 0, C_DELTA_POS, C_DELTA_NEG)
    ax.scatter(v_mean, v_diff, c=colors, s=14, alpha=0.6,
               edgecolors="none", zorder=3)

    # Reference and limit lines
    ax.axhline(0, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.axhline(bias, color=C_ENSEMBLE, lw=1.2, ls="-",
               label=f"Bias = {bias:.0f} mm\u00b3")
    ax.axhline(loa_upper, color=C_ENSEMBLE, lw=0.8, ls="--",
               label=f"+1.96 SD = {loa_upper:.0f}")
    ax.axhline(loa_lower, color=C_ENSEMBLE, lw=0.8, ls="--",
               label=f"\u22121.96 SD = {loa_lower:.0f}")

    # Log x-scale (volumes span orders of magnitude)
    ax.set_xscale("log")
    ax.set_xlabel("Mean volume (mm\u00b3)")
    ax.set_ylabel("Difference: Ensemble \u2212 GT (mm\u00b3)")

    # Text box with summary
    ax.text(
        0.03, 0.97,
        f"Bias = {bias:.0f} mm\u00b3\n"
        f"95% LoA: [{loa_lower:.0f}, {loa_upper:.0f}]\n"
        f"n = {len(v_diff)}",
        transform=ax.transAxes, fontsize=7,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Bland-Altman: Ensemble vs GT volume", fontweight="bold")

    fig.tight_layout()
    return fig
