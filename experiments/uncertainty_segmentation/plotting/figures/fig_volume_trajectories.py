"""Fig 11: All-patient ET volume trajectories colored by ensemble uncertainty.

Single-panel figure showing all 58 MenGrowth patient trajectories overlaid
on shared axes. Each trajectory is the median ET volume (mm^3) with a
+/- std ribbon from the ensemble. Trajectories are colored by the mean
inter-member volume standard deviation (normalized), providing a visual
indicator of which tumors have high segmentation uncertainty.

The volume tracked is **ET** (enhancing tumor = meningioma mass, ch2),
not WT, following the BraTS-MEN label convention fix.
"""

from __future__ import annotations

import logging

import matplotlib.axes
import matplotlib.cm as cm
import matplotlib.colorbar
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)

logger = logging.getLogger(__name__)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | None:
    """Generate the all-patient volume trajectory figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Optional pre-created axes.

    Returns:
        The Figure object, or None if volume data is unavailable.
    """
    if not data.has_volumes:
        logger.warning("MenGrowth volumes not available — skipping Fig 11")
        return None

    import pandas as pd

    vol = data.mengrowth_volumes
    figsize = config.get("figsize", [7, 4])
    color_metric = config.get("color_metric", "vol_std")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    patients = sorted(vol["patient_id"].unique())
    n_patients = len(patients)

    # Compute per-patient uncertainty for coloring
    per_patient_unc = vol.groupby("patient_id")[color_metric].mean()

    # Normalize to [0, 1] for colormap
    unc_vals = per_patient_unc.values
    if unc_vals.max() - unc_vals.min() > 1e-8:
        unc_norm = (unc_vals - unc_vals.min()) / (unc_vals.max() - unc_vals.min())
    else:
        unc_norm = np.zeros_like(unc_vals)
    unc_map = dict(zip(per_patient_unc.index, unc_norm))

    cmap = cm.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=unc_vals.min(), vmax=unc_vals.max())

    for patient_id in patients:
        pdata = vol[vol["patient_id"] == patient_id].sort_values("timepoint_idx")
        t = pdata["timepoint_idx"].values
        v_median = pdata["vol_median"].values
        v_std = pdata["vol_std"].values

        color = cmap(unc_map[patient_id])

        # Ribbon: median +/- std
        ax.fill_between(
            t, v_median - v_std, v_median + v_std,
            alpha=0.12, color=color, linewidth=0,
        )
        ax.plot(t, v_median, "o-", color=color, ms=3, lw=1.0, alpha=0.8)

    ax.set_xlabel("Timepoint")
    ax.set_ylabel("ET Volume (mm\u00b3)")

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar_label = {
        "vol_std": "Mean inter-member \u03c3 (mm\u00b3)",
        "mean_var": "Mean voxel-wise variance",
        "mean_entropy": "Predictive entropy",
        "mean_mi": "Mutual information",
    }.get(color_metric, color_metric)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.text(
        0.02, 0.98, f"N = {n_patients} patients",
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
    )

    fig.tight_layout()
    return fig
