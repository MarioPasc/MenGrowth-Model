"""Fig 5: Ensemble convergence + threshold sensitivity (combined).

Left panel — Dice convergence with ensemble size k:
  * Dice of ensemble-of-k prediction (squares, solid)
  * Sample mean of per-member Dice (circles, dashed)
  * Optional theoretical 1/sqrt(k) SE band (dotted, grey)

Right panel — Dice vs binarization threshold tau:
  * Ensemble Dice vs threshold (squares, solid)
  * Mean per-member Dice vs threshold (circles, dashed)

Both panels show WT and ET as separate colors.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import C_BASELINE

C_WT = "#0072B2"
C_ET = "#D55E00"
REGION_COLORS = {"wt": C_WT, "et": C_ET}


# ── convergence helpers ──────────────────────────────────────────────


def _agg_sample_mean(conv_df: pd.DataFrame) -> pd.DataFrame:
    return (
        conv_df[conv_df["k"] >= 2]
        .groupby("k")
        .agg(y=("running_mean", "mean"), se=("running_se", "mean"))
        .reset_index()
    )


def _agg_ensemble_k(ek_df: pd.DataFrame, ch: str) -> pd.DataFrame:
    col = f"dice_{ch}" if f"dice_{ch}" in ek_df.columns else "ensemble_dice"
    g = ek_df.groupby("k").agg(y=(col, "mean"), sd=(col, "std"), n=(col, "count")).reset_index()
    g["se"] = g["sd"] / np.sqrt(g["n"].clip(lower=1))
    return g


def _plot_convergence(
    ax: matplotlib.axes.Axes,
    data: EnsembleResultsData,
    show_theoretical: bool,
) -> None:
    sm_sources = {"wt": data.convergence_wt, "et": data.convergence_et}
    ek = getattr(data, "ensemble_k_convergence", None)
    ek_dict: dict[str, pd.DataFrame] = ek if isinstance(ek, dict) else {}

    for ch, color in REGION_COLORS.items():
        # Sample mean (circles + dashed)
        sm = sm_sources[ch]
        agg_sm = _agg_sample_mean(sm)
        k_sm = agg_sm["k"].values
        y_sm = agg_sm["y"].values
        se_sm = agg_sm["se"].values

        ax.fill_between(
            k_sm,
            y_sm - 1.96 * se_sm,
            y_sm + 1.96 * se_sm,
            alpha=0.10,
            color=color,
        )
        ax.plot(
            k_sm,
            y_sm,
            "o--",
            color=color,
            ms=3.0,
            lw=0.9,
            alpha=0.85,
        )

        # Ensemble-of-k (squares + solid)
        ek_df = ek_dict.get(ch)
        if ek_df is not None and not ek_df.empty:
            agg_ek = _agg_ensemble_k(ek_df, ch)
            if not agg_ek.empty:
                k_ek = agg_ek["k"].values
                y_ek = agg_ek["y"].values
                se_ek = agg_ek["se"].values
                ax.fill_between(
                    k_ek,
                    y_ek - 1.96 * se_ek,
                    y_ek + 1.96 * se_ek,
                    alpha=0.12,
                    color=color,
                )
                ax.plot(
                    k_ek,
                    y_ek,
                    "s-",
                    color=color,
                    ms=3.5,
                    lw=1.2,
                )

    if show_theoretical:
        for ch, color in REGION_COLORS.items():
            agg_sm_ch = _agg_sample_mean(sm_sources[ch])
            se0 = agg_sm_ch["se"].values
            if len(se0) > 0:
                k_th = np.arange(2, int(agg_sm_ch["k"].max()) + 1)
                se_th = se0[0] * np.sqrt(2) / np.sqrt(k_th)
                y0 = agg_sm_ch["y"].values[0]
                ax.plot(k_th, y0 + 1.96 * se_th, ":", color=C_BASELINE, lw=0.7)
                ax.plot(k_th, y0 - 1.96 * se_th, ":", color=C_BASELINE, lw=0.7)

    k_max = int(agg_sm["k"].max())
    ax.set_xlabel("Ensemble size ($k$)")
    ax.set_ylabel("Dice")
    ax.set_xticks(range(2, k_max + 1, max(1, k_max // 10)))


# ── threshold-sensitivity helpers ────────────────────────────────────


def _plot_threshold(
    ax: matplotlib.axes.Axes,
    data: EnsembleResultsData,
) -> dict[str, float]:
    """Plot threshold sensitivity and return ensemble τ* per channel."""
    tau_stars: dict[str, float] = {}
    ts = data.threshold_sensitivity
    if ts is None or ts.empty:
        ax.text(
            0.5,
            0.5,
            "threshold_sensitivity.csv\nnot found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="grey",
        )
        return tau_stars

    members = ts[ts["source"].str.startswith("member_")]
    ens = ts[ts["source"] == "ensemble"]

    for ch, color in REGION_COLORS.items():
        col = f"dice_{ch}"
        if col not in ts.columns:
            continue

        # Individual per-member thin lines
        for _src, sub in members.groupby("source"):
            sub_agg = sub.groupby("threshold")[col].mean().reset_index()
            sub_agg = sub_agg.sort_values("threshold")
            ax.plot(
                sub_agg["threshold"].values,
                sub_agg[col].values,
                "-",
                color=color,
                alpha=0.15,
                lw=0.5,
            )

        # Mean per-member (circles + dashed)
        mem_agg = members.groupby("threshold")[col].mean().reset_index()
        mem_agg = mem_agg.sort_values("threshold")
        ax.plot(
            mem_agg["threshold"].values,
            mem_agg[col].values,
            "o--",
            color=color,
            ms=3.0,
            lw=0.9,
            alpha=0.85,
        )

        # Ensemble (squares + solid)
        ens_agg = ens.groupby("threshold")[col].mean().reset_index()
        ens_agg = ens_agg.sort_values("threshold")
        ax.plot(
            ens_agg["threshold"].values,
            ens_agg[col].values,
            "s-",
            color=color,
            ms=3.5,
            lw=1.2,
        )

        # Ensemble optimal τ*
        if not ens_agg.empty:
            imax = int(ens_agg[col].values.argmax())
            tau_star = float(ens_agg["threshold"].values[imax])
            tau_stars[ch] = tau_star
            ax.axvline(tau_star, ls="--", color=color, lw=0.8, alpha=0.5)

        # Per-member optimal τ* scatter at top
        member_optima: list[float] = []
        for _src, sub in members.groupby("source"):
            sub_agg = sub.groupby("threshold")[col].mean().reset_index()
            if not sub_agg.empty:
                imax = int(sub_agg[col].values.argmax())
                member_optima.append(float(sub_agg["threshold"].values[imax]))
        if member_optima:
            taus = np.array(member_optima)
            y_top = ax.get_ylim()[1] if ax.has_data() else 1.0
            ax.scatter(
                taus,
                np.full_like(taus, y_top - 0.003 * max(1.0, abs(y_top))),
                marker="|",
                color=color,
                s=25,
                alpha=0.5,
            )

    ax.axvline(0.5, ls=":", color="grey", lw=0.7)
    ax.set_xlabel(r"Binarization threshold ($\tau$)")
    return tau_stars


# ── main entry point ─────────────────────────────────────────────────


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the combined convergence + threshold sensitivity figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Ignored (creates its own axes).

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [9, 3.5])
    show_theoretical = config.get("show_theoretical", True)

    fig, (ax_conv, ax_thresh) = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=True,
    )

    _plot_convergence(ax_conv, data, show_theoretical)
    tau_stars = _plot_threshold(ax_thresh, data)

    # Right panel inherits y-label from shared axis; remove duplicate
    ax_thresh.set_ylabel("")

    # ── Region colour legend (top centre, outside plots) ──
    region_handles = [
        mpatches.Patch(facecolor=C_WT, alpha=0.7, label="Whole Tumor (WT)"),
        mpatches.Patch(facecolor=C_ET, alpha=0.7, label="Enhancing Tumor (ET)"),
    ]
    fig.legend(
        handles=region_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.06),
    )

    # ── Convergence legend (below left subplot, 1 col × 3 rows) ──
    conv_handles = [
        mlines.Line2D(
            [],
            [],
            color="grey",
            marker="s",
            ls="-",
            ms=4,
            lw=1.2,
            label=r"Dice of ensemble-of-$k$ ($D_k$)",
        ),
        mlines.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            ls="--",
            ms=3.5,
            lw=0.9,
            label=r"Sample mean of per-member Dice ($\hat\mu_k$)",
        ),
        mlines.Line2D(
            [],
            [],
            color=C_BASELINE,
            ls=":",
            lw=0.7,
            label=r"Theoretical $\hat\mu_k$ band $\propto 1/\sqrt{k}$",
        ),
    ]
    ax_conv.legend(
        handles=conv_handles,
        loc="upper center",
        ncol=1,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.18),
    )

    # ── Threshold legend (below right subplot, 1 col × 3 rows) ──
    thresh_handles = [
        mlines.Line2D(
            [],
            [],
            color="grey",
            ls="-",
            lw=0.5,
            alpha=0.4,
            label="Individual member curves",
        ),
        mlines.Line2D(
            [],
            [],
            color="grey",
            ls="--",
            lw=0.8,
            label=r"Ensemble $\tau^\ast$ (optimal threshold)",
        ),
        mlines.Line2D(
            [],
            [],
            color="grey",
            marker="|",
            ls="none",
            ms=8,
            label=r"Per-member $\tau^\ast$",
        ),
    ]
    ax_thresh.legend(
        handles=thresh_handles,
        loc="upper center",
        ncol=1,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.30)
    return fig
