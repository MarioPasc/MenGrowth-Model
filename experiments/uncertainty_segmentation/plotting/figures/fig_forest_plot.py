"""Fig 4: Per-Member Forest Plot.

Per-member WT Dice mean +/- 95% CI, with ensemble and baseline reference
lines.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import (
    C_BASELINE,
    C_ENSEMBLE,
    C_MEMBERS,
)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the forest plot.

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

    stats = data.statistical_summary
    members = stats["per_member_summary"]
    M = len(members)

    y_positions = list(range(M))
    means = [m["dice_wt_mean"] for m in members]
    ci_lo = [m["dice_wt_ci95"][0] for m in members]
    ci_hi = [m["dice_wt_ci95"][1] for m in members]
    labels = [f"Member {m['member_id']}" for m in members]

    # Per-member CIs
    for i, (mean, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
        ax.plot([lo, hi], [i, i], color=C_MEMBERS, lw=1.5,
                solid_capstyle="round")
        ax.plot(mean, i, "o", color=C_MEMBERS, ms=5, zorder=4)

    # Ensemble reference band
    evb = stats["ensemble_vs_baseline"]["wt"]
    ens_mean = evb["ensemble_mean"]
    ens_ci = evb.get("ensemble_ci95", [ens_mean, ens_mean])
    ax.axvspan(ens_ci[0], ens_ci[1], alpha=0.12, color=C_ENSEMBLE, zorder=1)
    ax.axvline(ens_mean, color=C_ENSEMBLE, lw=1.2, ls="-",
               label="Ensemble", zorder=2)

    # Baseline reference
    bas_mean = evb["baseline_mean"]
    ax.axvline(bas_mean, color=C_BASELINE, lw=1.2, ls="--",
               label="Baseline", zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Dice (WT)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # ICC annotation
    icc = stats["inter_member_agreement"]["icc_wt"]
    ax.text(
        0.02, 0.02,
        f"ICC(3,1) = {icc:.3f}",
        transform=ax.transAxes, fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )
    ax.set_title("Per-member WT Dice (mean +/- 95% CI)", fontweight="bold")

    fig.tight_layout()
    return fig
