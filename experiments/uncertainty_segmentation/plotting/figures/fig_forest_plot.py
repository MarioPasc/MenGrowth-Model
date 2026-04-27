"""Fig 4: Per-Member Forest Plot.

Per-member WT Dice mean +/- 95% CI, with ensemble and baseline reference
lines.
"""

from __future__ import annotations

import logging

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import (
    C_BASELINE,
    C_ENSEMBLE,
    C_MEMBERS,
    REGION_DISPLAY_SHORT,
)

_ = REGION_DISPLAY_SHORT  # used in plot()


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

    region = config.get("region", "wt")
    region_short = REGION_DISPLAY_SHORT[region]

    stats = data.statistical_summary
    members = stats["per_member_summary"]
    M = len(members)

    mean_key = f"dice_{region}_mean"
    ci_key = f"dice_{region}_ci95"
    if mean_key not in members[0]:
        logger.warning(
            "Per-member summary lacks %s — skipping forest plot for %s", mean_key, region
        )
        plt.close(fig)
        return None

    y_positions = list(range(M))
    means = [m[mean_key] for m in members]
    ci_lo = [m[ci_key][0] for m in members]
    ci_hi = [m[ci_key][1] for m in members]
    labels = [f"Member {m['member_id']}" for m in members]

    # Per-member CIs
    for i, (mean, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
        ax.plot([lo, hi], [i, i], color=C_MEMBERS, lw=1.5, solid_capstyle="round")
        ax.plot(mean, i, "o", color=C_MEMBERS, ms=5, zorder=4)

    # Ensemble reference band
    evb = stats["ensemble_vs_baseline"][region]
    ens_mean = evb["ensemble_mean"]
    ens_ci = evb.get("ensemble_ci95", [ens_mean, ens_mean])
    ax.axvspan(ens_ci[0], ens_ci[1], alpha=0.12, color=C_ENSEMBLE, zorder=1)
    ax.axvline(ens_mean, color=C_ENSEMBLE, lw=1.2, ls="-", label="Ensemble", zorder=2)

    # Baseline reference
    bas_mean = evb["baseline_mean"]
    ax.axvline(bas_mean, color=C_BASELINE, lw=1.2, ls="--", label="Baseline", zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Dice ({region_short})")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # ICC annotation
    icc = stats["inter_member_agreement"][f"icc_{region}"]
    ax.text(
        0.02,
        0.02,
        f"ICC(3,1) = {icc:.3f}",
        transform=ax.transAxes,
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )
    ax.set_title(f"Per-member {region_short} Dice (mean +/- 95% CI)", fontweight="bold")

    fig.tight_layout()
    return fig
