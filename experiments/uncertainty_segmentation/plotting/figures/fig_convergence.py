"""Fig 5: Ensemble convergence.

Three-panel figure (one per channel: WT, TC, ET) contrasting two distinct
convergence quantities that are frequently conflated:

* **Sample-mean of per-member Dice** (existing, from
  ``convergence_dice_{wt,tc,et}.csv``): the running mean
  :math:`\\hat\\mu_k = (1/k)\\sum_{m=1}^k d_m` where :math:`d_m` is the
  scalar Dice of individual member m. For an i.i.d. ensemble this
  converges to :math:`\\mathbb{E}[d_m]`; its trend in k is a
  sample-ordering artifact, not an ensemble-performance curve. The
  value of this diagnostic is the shrinking error band
  (:math:`\\propto 1/\\sqrt{k}`) that verifies the LLN regime.

* **Dice of the ensemble-of-k prediction** (from
  ``convergence_ensemble_dice_{wt,tc,et}.csv``): compute
  :math:`\\bar p_k = (1/k)\\sum_{m=1}^k p_m`, binarize at 0.5, and
  Dice against GT. Typically **rises** with k because averaging soft
  probabilities reduces per-member noise, then plateaus. This is the
  curve a reader usually expects from a "convergence of Dice" figure.

When the second CSV is unavailable (older runs without saved per-member
soft probs), the figure falls back to the sample-mean panel only.
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
    C_MEMBERS,
)


def _aggregate_sample_mean(conv_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the sample-mean running-mean curve across scans."""
    return conv_df[conv_df["k"] >= 2].groupby("k").agg(
        y=("running_mean", "mean"),
        se=("running_se", "mean"),
    ).reset_index()


def _aggregate_ensemble_k(ek_df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Aggregate the Dice(ensemble_k) curve across scans."""
    col = f"dice_{channel}" if f"dice_{channel}" in ek_df.columns else "ensemble_dice"
    grouped = ek_df.groupby("k").agg(
        y=(col, "mean"),
        sd=(col, "std"),
        n=(col, "count"),
    ).reset_index()
    grouped["se"] = grouped["sd"] / np.sqrt(grouped["n"].clip(lower=1))
    return grouped


def _plot_single_convergence(
    sample_mean_df: pd.DataFrame,
    ensemble_k_df: pd.DataFrame | None,
    channel: str,
    metric_label: str,
    show_theoretical: bool,
    ax: matplotlib.axes.Axes,
) -> None:
    """Plot sample-mean and ensemble-of-k curves on one axis."""
    # --- Curve A: sample mean of per-member Dice (existing) ---
    agg_sm = _aggregate_sample_mean(sample_mean_df)
    k_sm = agg_sm["k"].values
    y_sm = agg_sm["y"].values
    se_sm = agg_sm["se"].values
    ax.fill_between(
        k_sm, y_sm - 1.96 * se_sm, y_sm + 1.96 * se_sm,
        alpha=0.15, color=C_MEMBERS,
    )
    ax.plot(
        k_sm, y_sm, "o-", color=C_MEMBERS, ms=3.5, lw=1.0,
        label=r"Sample mean of per-member Dice ($\hat\mu_k$)",
    )

    # Theoretical 1/sqrt(k) scaling of the sample-mean SE band.
    if show_theoretical and len(se_sm) > 0:
        se_at_2 = se_sm[0]
        k_theory = np.arange(2, int(k_sm.max()) + 1)
        se_theory = se_at_2 * np.sqrt(2) / np.sqrt(k_theory)
        ax.plot(k_theory, y_sm[0] + 1.96 * se_theory, ":", color=C_BASELINE, lw=0.7)
        ax.plot(
            k_theory, y_sm[0] - 1.96 * se_theory, ":", color=C_BASELINE, lw=0.7,
            label=r"Theoretical $\hat\mu_k$ band $\propto 1/\sqrt{k}$",
        )

    # --- Curve B: Dice of ensemble-of-k prediction (new) ---
    if ensemble_k_df is not None and not ensemble_k_df.empty:
        agg_ek = _aggregate_ensemble_k(ensemble_k_df, channel)
        if not agg_ek.empty:
            k_ek = agg_ek["k"].values
            y_ek = agg_ek["y"].values
            se_ek = agg_ek["se"].values
            ax.fill_between(
                k_ek, y_ek - 1.96 * se_ek, y_ek + 1.96 * se_ek,
                alpha=0.18, color=C_ENSEMBLE,
            )
            ax.plot(
                k_ek, y_ek, "s-", color=C_ENSEMBLE, ms=3.5, lw=1.2,
                label=r"Dice of ensemble-of-$k$ prediction ($D_k$)",
            )

    ax.set_xlabel("Ensemble size (k)")
    ax.set_ylabel(metric_label)
    ax.set_xticks(range(2, int(k_sm.max()) + 1, max(1, int(k_sm.max()) // 10)))
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_title(f"Convergence of {metric_label}", fontweight="bold")


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the three-panel convergence figure.

    Args:
        data: All loaded experiment data. If
            ``data.ensemble_k_convergence`` is ``None``, the figure
            falls back to the sample-mean curve only.
        config: Figure-specific config from config.yaml.
        ax: Ignored (three-panel figure creates its own axes).

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [9, 2.8])
    show_theoretical = config.get("show_theoretical", True)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    sample_mean_sources = [
        ("wt", data.convergence_wt, "Dice (WT)"),
        ("tc", data.convergence_tc, "Dice (TC)"),
        ("et", data.convergence_et, "Dice (ET)"),
    ]
    ek = getattr(data, "ensemble_k_convergence", None)
    ek_per_channel = ek if isinstance(ek, dict) else {}

    for ax_i, (ch, sm_df, label) in zip(axes, sample_mean_sources):
        _plot_single_convergence(
            sample_mean_df=sm_df,
            ensemble_k_df=ek_per_channel.get(ch),
            channel=ch,
            metric_label=label,
            show_theoretical=show_theoretical,
            ax=ax_i,
        )

    fig.tight_layout()
    return fig
