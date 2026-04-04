"""Fig 3: Paired Comparison -- Ensemble vs Baseline.

Panel A: Scatter plot (baseline Dice vs ensemble Dice).
Panel B: Histogram of paired Dice differences.
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
    C_DELTA_NEG,
    C_DELTA_POS,
    C_ENSEMBLE,
    C_FILL,
    significance_label,
)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the paired comparison figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Ignored (two-panel figure creates its own axes).

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [7, 3.2])
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=figsize)

    # Merge on scan_id
    merged = data.baseline_dice.merge(
        data.ensemble_dice, on="scan_id", suffixes=("_bas", "_ens"),
    )
    x = merged["dice_wt_bas"].values
    y = merged["dice_wt_ens"].values

    # --- Panel A: Scatter ---
    colors_sc = np.where(y > x, C_DELTA_POS, C_DELTA_NEG)
    ax_scatter.scatter(x, y, s=14, c=colors_sc, alpha=0.6,
                       edgecolors="none", zorder=3)
    lims = [-0.05, 1.05]
    ax_scatter.plot(lims, lims, "k--", lw=0.7, alpha=0.5, label="Identity")
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel("Baseline Dice (WT)")
    ax_scatter.set_ylabel("Ensemble Dice (WT)")
    ax_scatter.set_aspect("equal")

    n_better = int((y > x).sum())
    n_worse = int((y < x).sum())
    ax_scatter.text(
        0.05, 0.92,
        f"Improved: {n_better}/{len(x)}\nWorsened: {n_worse}/{len(x)}",
        transform=ax_scatter.transAxes, fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax_scatter.set_title("a) Per-subject comparison", loc="left",
                         fontweight="bold")

    # --- Panel B: Paired differences histogram ---
    deltas = data.paired_differences["dice_wt_delta"].values

    evb = data.statistical_summary["ensemble_vs_baseline"]["wt"]
    delta_mean = evb["delta"]
    ci_lo = evb["ci_95_lower"]
    ci_hi = evb["ci_95_upper"]
    p_val = evb["p_value_wilcoxon"]
    d_val = evb["cohens_d"]

    n_bins = config.get("n_bins_histogram", 30)
    ax_hist.hist(deltas, bins=n_bins, color=C_FILL, alpha=0.5,
                 edgecolor="white", lw=0.5)
    ax_hist.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
    ax_hist.axvline(delta_mean, color=C_ENSEMBLE, lw=1.5,
                    label=f"Mean \u0394 = {delta_mean:.3f}")
    ax_hist.axvspan(ci_lo, ci_hi, alpha=0.15, color=C_ENSEMBLE,
                    label="95% CI")

    ax_hist.set_xlabel("\u0394Dice (Ensemble \u2212 Baseline)")
    ax_hist.set_ylabel("Count")

    sig = significance_label(p_val)
    ax_hist.text(
        0.95, 0.92,
        f"p = {p_val:.1e} {sig}\nd = {d_val:.2f}",
        transform=ax_hist.transAxes, fontsize=7,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax_hist.legend(frameon=False, fontsize=7, loc="upper left")
    ax_hist.set_title("b) Paired differences (WT Dice)", loc="left",
                      fontweight="bold")

    fig.tight_layout()
    return fig
