"""Fig 8: Dice by Sub-compartment.

Grouped bar chart: Baseline vs Ensemble for TC, WT, ET with CI error bars
and significance brackets.
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
    add_stat_bracket,
    significance_label,
)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the dice-by-compartment figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Optional pre-created axes.

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [4.5, 3])
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    stats = data.statistical_summary
    # TC is omitted: for MEN, the TC target is always empty (BSF has no
    # tumor-core concept in the 2-label meningioma space), so TC Dice is
    # trivially ~1.0 and carries no information.
    compartments = ["wt", "et"]
    labels = ["WT", "ET"]
    x = np.arange(len(compartments))
    width = 0.3

    bas_means: list[float] = []
    ens_means: list[float] = []
    bas_cis: list[list[float]] = []
    ens_cis: list[list[float]] = []

    for comp in compartments:
        evb = stats["ensemble_vs_baseline"][comp]
        bas_means.append(evb["baseline_mean"])
        ens_means.append(evb["ensemble_mean"])
        bas_cis.append([
            evb["baseline_mean"] - evb["baseline_ci95"][0],
            evb["baseline_ci95"][1] - evb["baseline_mean"],
        ])
        ens_cis.append([
            evb["ensemble_mean"] - evb["ensemble_ci95"][0],
            evb["ensemble_ci95"][1] - evb["ensemble_mean"],
        ])

    bas_err = np.array(bas_cis).T
    ens_err = np.array(ens_cis).T

    ax.bar(x - width / 2, bas_means, width, yerr=bas_err, color=C_BASELINE,
           alpha=0.6, label="Baseline", capsize=3, error_kw=dict(lw=0.8))
    ax.bar(x + width / 2, ens_means, width, yerr=ens_err, color=C_ENSEMBLE,
           alpha=0.6, label="Ensemble", capsize=3, error_kw=dict(lw=0.8))

    # Significance annotations
    for i, comp in enumerate(compartments):
        evb = stats["ensemble_vs_baseline"][comp]
        p = evb["p_value_wilcoxon"]
        d = evb["cohens_d"]
        sig = significance_label(p)
        y_max = (max(ens_means[i], bas_means[i])
                 + max(ens_err[1][i], bas_err[1][i]) + 0.03)
        add_stat_bracket(ax, i - width / 2, i + width / 2, y_max, 0.02,
                         f"{sig} d={d:.2f}", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Dice by sub-compartment (mean +/- 95% CI)",
                 fontweight="bold")

    fig.tight_layout()
    return fig
