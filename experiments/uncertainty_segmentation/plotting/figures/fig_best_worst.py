"""Fig 7: Best / Worst Case Analysis.

Horizontal bar chart showing the N best and N worst improvement cases
(ensemble vs baseline), with delta annotations.
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
    C_BEST,
    C_DELTA_NEG,
    C_DELTA_POS,
    C_ENSEMBLE,
)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the best/worst case analysis figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Optional pre-created axes.

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [5, 4.5])
    n_cases = config.get("n_cases", 5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    merged = data.baseline_dice.merge(
        data.ensemble_dice, on="scan_id", suffixes=("_bas", "_ens"),
    )
    merged["delta_wt"] = merged["dice_wt_ens"] - merged["dice_wt_bas"]
    merged = merged.sort_values("delta_wt")

    worst = merged.head(n_cases)
    best = merged.tail(n_cases).iloc[::-1]
    cases = pd.concat([best, worst])

    y_pos = np.arange(len(cases))
    h = 0.35

    ax.barh(y_pos - h / 2, cases["dice_wt_bas"].values, h,
            color=C_BASELINE, alpha=0.6, label="Baseline")
    ax.barh(y_pos + h / 2, cases["dice_wt_ens"].values, h,
            color=C_ENSEMBLE, alpha=0.6, label="Ensemble")

    # Delta annotations
    for i, (_, row) in enumerate(cases.iterrows()):
        delta = row["delta_wt"]
        color = C_DELTA_POS if delta > 0 else C_DELTA_NEG
        sign = "+" if delta > 0 else ""
        x_pos = max(row["dice_wt_bas"], row["dice_wt_ens"]) + 0.02
        ax.text(x_pos, i, f"{sign}{delta:.3f}", va="center",
                fontsize=7, color=color)

    # Separator line between best and worst
    ax.axhline(n_cases - 0.5, color="k", ls=":", lw=0.5, alpha=0.5)
    ax.text(-0.02, n_cases / 2 - 0.5, "Best \u0394", ha="right",
            va="center", fontsize=7, color=C_BEST, fontweight="bold",
            transform=ax.get_yaxis_transform())
    ax.text(-0.02, n_cases + n_cases / 2 - 0.5, "Worst \u0394", ha="right",
            va="center", fontsize=7, color=C_DELTA_NEG, fontweight="bold",
            transform=ax.get_yaxis_transform())

    # Truncate scan IDs for readability
    short_ids = [s.replace("BraTS-MEN-", "") for s in cases["scan_id"].values]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_ids, fontsize=7)
    ax.set_xlabel("Dice (WT)")
    ax.set_xlim(0, 1.15)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.invert_yaxis()
    ax.set_title(f"Top-{n_cases} best and worst improvements",
                 fontweight="bold")

    fig.tight_layout()
    return fig
