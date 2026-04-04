"""Fig 6: Calibration Reliability Diagram.

Reliability diagram with gap shading and ECE/Brier annotation.
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
    C_BEST,
    C_DELTA_NEG,
    C_ENSEMBLE,
)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the calibration reliability diagram.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Optional pre-created axes.

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [3.5, 3.5])
    min_bin_count = config.get("min_bin_count", 50)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    calibration = data.calibration
    rel = calibration["reliability"]
    bin_edges = np.array(rel["bin_edges"])
    bin_acc = np.array(rel["bin_accuracy"])
    bin_conf = np.array(rel["bin_confidence"])
    bin_count = np.array(rel["bin_count"])

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5, label="Perfect")

    # Bar chart
    width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers, bin_acc, width=width * 0.85, alpha=0.5,
           color=C_ENSEMBLE, edgecolor="white", lw=0.3, label="Observed")

    # Gap fill (miscalibration)
    for i in range(len(bin_acc)):
        if bin_count[i] > min_bin_count:
            color = C_DELTA_NEG if bin_acc[i] < bin_conf[i] else C_BEST
            ax.fill_between(
                [bin_centers[i] - width / 2, bin_centers[i] + width / 2],
                bin_acc[i], bin_conf[i],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ece = calibration["ece"]
    brier = calibration["brier_score"]
    ax.text(
        0.05, 0.92,
        f"ECE = {ece:.4f}\nBrier = {brier:.4f}",
        transform=ax.transAxes, fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Calibration (reliability diagram)", fontweight="bold")

    fig.tight_layout()
    return fig
