"""Fig: Threshold-sensitivity of Dice per channel.

Two-panel figure (WT, ET). TC omitted (trivially ~1.0 for MEN). For each channel, shows:

* **Per-member Dice vs. threshold**: one light-grey line per ensemble
  member, computed across scans (mean Dice at each threshold). These
  span the member-level variability.
* **Ensemble Dice vs. threshold**: the heavy line for the M-member
  ensemble (mean soft probs binarized at each τ, averaged across scans).
* **Per-member optimal threshold** distribution: scatter at the top of
  each panel marking each member's argmax-Dice τ. Answers "is 0.5
  actually the best threshold?"

Data source: ``threshold_sensitivity.csv`` (written by
``evaluate_ensemble_per_subject`` when
``inference.save_per_member_probs_all=True``). Columns:
``scan_id, source, threshold, dice_tc, dice_wt, dice_et``.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import (
    C_BASELINE,
    C_ENSEMBLE,
    C_MEMBERS,
)


def _aggregate(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Mean Dice across scans for each (source, threshold)."""
    col = f"dice_{channel}"
    if col not in df.columns:
        return pd.DataFrame(columns=["source", "threshold", "dice_mean"])
    return (
        df.groupby(["source", "threshold"])[col]
        .mean()
        .reset_index()
        .rename(columns={col: "dice_mean"})
    )


def _plot_one_channel(
    ax: matplotlib.axes.Axes,
    df_agg: pd.DataFrame,
    channel_label: str,
) -> None:
    if df_agg.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="grey",
        )
        ax.set_title(channel_label, fontweight="bold")
        return

    member_df = df_agg[df_agg["source"].str.startswith("member_")]
    ens_df = df_agg[df_agg["source"] == "ensemble"].sort_values("threshold")

    # Per-member thin lines
    member_optima: list[tuple[float, float]] = []
    for src, sub in member_df.groupby("source"):
        sub = sub.sort_values("threshold")
        ax.plot(
            sub["threshold"].values,
            sub["dice_mean"].values,
            "-",
            color=C_MEMBERS,
            alpha=0.35,
            lw=0.7,
        )
        if len(sub) > 0:
            imax = int(sub["dice_mean"].values.argmax())
            member_optima.append(
                (
                    float(sub["threshold"].values[imax]),
                    float(sub["dice_mean"].values[imax]),
                )
            )

    # Ensemble (heavy)
    if not ens_df.empty:
        ax.plot(
            ens_df["threshold"].values,
            ens_df["dice_mean"].values,
            "-",
            color=C_ENSEMBLE,
            lw=1.8,
            label=r"Ensemble ($M$ members)",
        )
        imax = int(ens_df["dice_mean"].values.argmax())
        tau_star = float(ens_df["threshold"].values[imax])
        d_star = float(ens_df["dice_mean"].values[imax])
        ax.axvline(
            tau_star,
            ls="--",
            color=C_ENSEMBLE,
            lw=0.8,
            alpha=0.6,
            label=rf"Ensemble $\tau^\ast={tau_star:.2f}$ (Dice={d_star:.3f})",
        )

    # Member optima marginal
    if member_optima:
        taus = np.array([o[0] for o in member_optima])
        y_top = ax.get_ylim()[1] if ax.has_data() else 1.0
        # Place markers just below the top so they aren't clipped.
        ax.scatter(
            taus,
            np.full_like(taus, y_top - 0.01 * max(1.0, abs(y_top))),
            marker="|",
            color=C_BASELINE,
            s=30,
            alpha=0.6,
            label=rf"Per-member $\tau^\ast$ (n={len(taus)})",
        )

    # 0.5 reference
    ax.axvline(0.5, ls=":", color="grey", lw=0.7)

    ax.set_xlabel(r"Binarization threshold $\tau$")
    ax.set_ylabel(f"Dice ({channel_label})")


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | None:
    """Merged into fig_convergence (combined 1×2 figure). Returns None."""
    return None
