"""Fig 12: Volume Uncertainty vs Volume Size.

Two-panel heteroscedasticity check:
  Panel A: log-log scatter of vol_mean vs vol_std with OLS regression.
  Panel B: logvol_mean vs logvol_std (linear) + logvol_mad_scaled overlay.

Justifies the log-volume transform for the GP growth model.
"""

from __future__ import annotations

import logging

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import (
    C_ENSEMBLE,
    C_MEDIAN,
)

logger = logging.getLogger(__name__)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | None:
    """Generate the volume uncertainty vs size figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Ignored (two-panel figure creates its own axes).

    Returns:
        The Figure object, or None if volume data is unavailable.
    """
    if not data.has_volumes:
        logger.warning("MenGrowth volumes not available — skipping Fig 12")
        return None

    vol = data.mengrowth_volumes
    figsize = config.get("figsize", [4.5, 3.5])
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # ---- Panel A: Raw volume space (log-log) ----
    v_mean = vol["vol_mean"].values.astype(float)
    v_std = vol["vol_std"].values.astype(float)

    # Color by wt_mean_entropy (use fallback if column has NaN/empty)
    entropy_col = "wt_mean_entropy"
    if entropy_col in vol.columns:
        entropy = vol[entropy_col].values.astype(float)
        valid_mask = np.isfinite(entropy) & (v_mean > 0) & (v_std > 0)
    else:
        entropy = np.ones(len(v_mean))
        valid_mask = (v_mean > 0) & (v_std > 0)

    v_m = v_mean[valid_mask]
    v_s = v_std[valid_mask]
    ent = entropy[valid_mask]

    sc = ax_a.scatter(v_m, v_s, c=ent, s=14, alpha=0.7,
                      cmap="YlOrRd", edgecolors="none", zorder=3)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")

    # OLS on log-log
    log_m = np.log10(v_m)
    log_s = np.log10(v_s)
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(log_m, log_s)
    x_fit = np.linspace(log_m.min(), log_m.max(), 100)
    y_fit = slope * x_fit + intercept
    ax_a.plot(10**x_fit, 10**y_fit, "k--", lw=1.0, alpha=0.7,
              label=f"slope={slope:.2f}, r={r_val:.2f}")

    cbar = fig.colorbar(sc, ax=ax_a, shrink=0.8, pad=0.02)
    cbar.set_label("WT entropy", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax_a.set_xlabel("Mean volume (mm\u00b3)")
    ax_a.set_ylabel("\u03c3 volume (mm\u00b3)")
    ax_a.legend(frameon=False, fontsize=7, loc="upper left")
    ax_a.set_title("a) Raw volume (log-log)", loc="left", fontweight="bold")

    # ---- Panel B: Log-volume space (linear) ----
    lv_mean = vol["logvol_mean"].values.astype(float)
    lv_std = vol["logvol_std"].values.astype(float)
    lv_mad = vol["logvol_mad_scaled"].values.astype(float)

    valid_b = np.isfinite(lv_mean) & np.isfinite(lv_std)
    ax_b.scatter(lv_mean[valid_b], lv_std[valid_b], s=14, alpha=0.6,
                 color=C_ENSEMBLE, edgecolors="none", zorder=3,
                 label="log-vol std")

    valid_mad = np.isfinite(lv_mean) & np.isfinite(lv_mad)
    ax_b.scatter(lv_mean[valid_mad], lv_mad[valid_mad], s=14, alpha=0.4,
                 color=C_MEDIAN, marker="^", edgecolors="none", zorder=3,
                 label="log-vol MAD (scaled)")

    ax_b.set_xlabel("log(V+1) mean")
    ax_b.set_ylabel("log(V+1) uncertainty")
    ax_b.legend(frameon=False, fontsize=7, loc="upper left")
    ax_b.set_title("b) Log-volume space (linear)", loc="left",
                   fontweight="bold")

    fig.suptitle("Heteroscedasticity: uncertainty scales with tumour size",
                 fontweight="bold", fontsize=10, y=1.02)
    fig.tight_layout()
    return fig
