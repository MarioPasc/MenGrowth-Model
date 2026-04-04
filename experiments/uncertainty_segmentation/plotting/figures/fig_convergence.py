"""Fig 5: LLN Convergence.

Three-panel figure showing the running mean +/- SE as a function of ensemble
size k for WT, TC, and ET Dice, with theoretical 1/sqrt(k) overlay.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import (
    C_BASELINE,
    C_ENSEMBLE,
)


def _plot_single_convergence(
    conv_df: pd.DataFrame,
    metric_name: str,
    show_theoretical: bool,
    ax: matplotlib.axes.Axes,
) -> None:
    """Plot convergence for a single metric on the given axes."""
    grouped = conv_df[conv_df["k"] >= 2].groupby("k").agg(
        mean_of_means=("running_mean", "mean"),
        mean_se=("running_se", "mean"),
    ).reset_index()

    k = grouped["k"].values
    y = grouped["mean_of_means"].values
    se = grouped["mean_se"].values

    ax.fill_between(k, y - 1.96 * se, y + 1.96 * se,
                    alpha=0.2, color=C_ENSEMBLE)
    ax.plot(k, y, "o-", color=C_ENSEMBLE, ms=4, lw=1.2,
            label="Running mean +/- 1.96 SE")

    # Theoretical 1/sqrt(k) curve
    if show_theoretical and len(se) > 0:
        se_at_2 = se[0]
        k_theory = np.arange(2, k.max() + 1)
        se_theory = se_at_2 * np.sqrt(2) / np.sqrt(k_theory)
        ax.plot(k_theory, y[0] + 1.96 * se_theory, ":", color=C_BASELINE,
                lw=0.8)
        ax.plot(k_theory, y[0] - 1.96 * se_theory, ":", color=C_BASELINE,
                lw=0.8, label=r"Theoretical $\propto 1/\sqrt{k}$")

    ax.set_xlabel("Ensemble size (k)")
    ax.set_ylabel(metric_name)
    ax.set_xticks(range(2, int(k.max()) + 1))
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"Convergence of {metric_name}", fontweight="bold")


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the three-panel convergence figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Ignored (three-panel figure creates its own axes).

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [9, 2.8])
    show_theoretical = config.get("show_theoretical", True)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax_i, conv_data, name in zip(
        axes,
        [data.convergence_wt, data.convergence_tc, data.convergence_et],
        ["Dice (WT)", "Dice (TC)", "Dice (ET)"],
    ):
        _plot_single_convergence(conv_data, name, show_theoretical, ax_i)

    fig.tight_layout()
    return fig
