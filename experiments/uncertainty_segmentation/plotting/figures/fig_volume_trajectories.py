"""Fig 11: Volume Trajectories with Uncertainty.

Per-patient longitudinal tumour volume trajectories with mean +/- std and
median +/- MAD ribbons from the ensemble.  Connects segmentation uncertainty
to growth prediction.
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
    C_ENSEMBLE,
    C_MEDIAN,
)

logger = logging.getLogger(__name__)


def _select_diverse_patients(
    df: "pd.DataFrame",
    n_patients: int,
) -> list[str]:
    """Select patients at evenly-spaced uncertainty percentiles.

    Args:
        df: MenGrowth volumes DataFrame.
        n_patients: Number of patients to select.

    Returns:
        List of patient IDs.
    """
    import pandas as pd

    per_patient_unc = (
        df.groupby("patient_id")["vol_std"]
        .mean()
        .sort_values()
    )
    n = len(per_patient_unc)
    if n <= n_patients:
        return list(per_patient_unc.index)

    # Evenly-spaced percentile indices
    indices = np.linspace(0, n - 1, n_patients).astype(int)
    return [per_patient_unc.index[i] for i in indices]


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | None:
    """Generate the volume trajectories figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Ignored (multi-panel figure creates its own axes).

    Returns:
        The Figure object, or None if volume data is unavailable.
    """
    if not data.has_volumes:
        logger.warning("MenGrowth volumes not available — skipping Fig 11")
        return None

    import pandas as pd

    vol = data.mengrowth_volumes
    n_patients = config.get("n_patients", 6)
    selection = config.get("patient_selection", "diverse")
    figsize = config.get("figsize", [7, 5])

    # Patient selection
    if isinstance(selection, list):
        patients = selection
    elif selection == "first":
        patients = sorted(vol["patient_id"].unique())[:n_patients]
    else:  # "diverse"
        patients = _select_diverse_patients(vol, n_patients)

    # Determine grid layout
    ncols = min(3, len(patients))
    nrows = int(np.ceil(len(patients) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if not hasattr(axes, "__iter__"):
        axes = np.array([axes])
    axes_flat = axes.flatten()

    # Identify per-member volume columns
    member_cols = [c for c in vol.columns if c.startswith("vol_m")]

    for idx, patient_id in enumerate(patients):
        ax_i = axes_flat[idx]
        pdata = vol[vol["patient_id"] == patient_id].sort_values("timepoint_idx")
        t = pdata["timepoint_idx"].values

        # Mean +/- std ribbon
        v_mean = pdata["vol_mean"].values
        v_std = pdata["vol_std"].values
        ax_i.fill_between(t, v_mean - v_std, v_mean + v_std,
                          alpha=0.2, color=C_ENSEMBLE)
        ax_i.plot(t, v_mean, "o-", color=C_ENSEMBLE, ms=5, lw=1.2,
                  label="Mean +/- std")

        # Median +/- 1.4826*MAD ribbon
        v_median = pdata["vol_median"].values
        v_mad = pdata["vol_mad"].values
        mad_scaled = 1.4826 * v_mad
        ax_i.fill_between(t, v_median - mad_scaled, v_median + mad_scaled,
                          alpha=0.12, color=C_MEDIAN)
        ax_i.plot(t, v_median, "s--", color=C_MEDIAN, ms=4, lw=0.8,
                  label="Median +/- MAD")

        # Per-member volumes as faint grey dots
        for mc in member_cols:
            vals = pdata[mc].values
            jitter = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(t))
            ax_i.scatter(t + jitter, vals, s=4, c="grey", alpha=0.25,
                         edgecolors="none", zorder=2)

        mean_unc = float(v_std.mean())
        ax_i.set_title(f"{patient_id}\n(\u03c3\u0305={mean_unc:.0f} mm\u00b3)",
                       fontsize=8, fontweight="bold")
        ax_i.set_xlabel("Timepoint")
        if idx % ncols == 0:
            ax_i.set_ylabel("Volume (mm\u00b3)")

    # Hide unused subplots
    for idx in range(len(patients), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend from first subplot
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               frameon=False, fontsize=7)

    fig.suptitle("Per-patient volume trajectories with ensemble uncertainty",
                 fontweight="bold", fontsize=10, y=1.01)
    fig.tight_layout()
    return fig
