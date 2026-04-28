"""Quant3: Multi-LoRA convergence and threshold sensitivity comparison.

3 rows (TC, WT, ET) × 2 columns (ensemble size, binarization threshold)
with a shared horizontal colorbar on top mapping LoRA rank via viridis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colorbar import ColorbarBase

from experiments.uncertainty_segmentation.plotting.inter_lora.io_layer import (
    InterLoraData,
)
from experiments.uncertainty_segmentation.plotting.inter_lora.style import (
    DOUBLE_COL_MM,
    LABEL_COLORS,
    MM_TO_INCH,
    REGION_DISPLAY,
    REGION_KEYS,
    rank_colormap,
)

logger = logging.getLogger(__name__)

_HEIGHT_MM: float = 170.0


# ── data loading (convergence CSVs live on disk, not in RankRun) ────


def _load_convergence_csv(run_dir: Path, label: str) -> pd.DataFrame | None:
    path = run_dir / "evaluation" / f"convergence_dice_{label}.csv"
    if not path.exists():
        logger.warning("Missing %s", path)
        return None
    return pd.read_csv(path)


def _load_ensemble_convergence_csv(run_dir: Path, label: str) -> pd.DataFrame | None:
    path = run_dir / "evaluation" / f"convergence_ensemble_dice_{label}.csv"
    if not path.exists():
        logger.warning("Missing %s", path)
        return None
    return pd.read_csv(path)


def _load_threshold_csv(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "evaluation" / "threshold_sensitivity.csv"
    if not path.exists():
        logger.warning("Missing %s", path)
        return None
    return pd.read_csv(path)


# ── aggregation helpers (adapted from fig_convergence.py) ───────────


def _agg_sample_mean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["k"] >= 2]
        .groupby("k")
        .agg(y=("running_mean", "mean"), se=("running_se", "mean"))
        .reset_index()
    )


def _agg_ensemble_k(df: pd.DataFrame, label: str) -> pd.DataFrame:
    col = f"dice_{label}" if f"dice_{label}" in df.columns else "ensemble_dice"
    g = df.groupby("k").agg(y=(col, "mean"), sd=(col, "std"), n=(col, "count")).reset_index()
    g["se"] = g["sd"] / np.sqrt(g["n"].clip(lower=1))
    return g


def _agg_threshold(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    col = f"dice_{label}"
    if col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    ens = (
        df[df["source"] == "ensemble"]
        .groupby("threshold")[col]
        .mean()
        .reset_index()
        .sort_values("threshold")
    )
    mem = (
        df[df["source"].str.startswith("member_")]
        .groupby("threshold")[col]
        .mean()
        .reset_index()
        .sort_values("threshold")
    )
    return ens, mem


# ── data export ─────────────────────────────────────────────────────


def _export_data(
    out_root: Path,
    all_data: dict[int, dict[str, Any]],
) -> None:
    """Persist aggregated convergence data for reproducibility."""
    path = out_root / "data" / "quant3_convergence_data.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    serialisable: dict[str, Any] = {}
    for rank, labels in all_data.items():
        rank_key = str(rank)
        serialisable[rank_key] = {}
        for label, contents in labels.items():
            entry: dict[str, Any] = {}
            for section_name, section_df in contents.items():
                if isinstance(section_df, pd.DataFrame) and not section_df.empty:
                    entry[section_name] = section_df.to_dict(orient="list")
                else:
                    entry[section_name] = {}
            serialisable[rank_key][label] = entry

    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info("Convergence data exported to %s", path)


# ── main entry point ────────────────────────────────────────────────


def plot(
    data: InterLoraData,
    config: dict,
) -> matplotlib.figure.Figure | None:
    """Generate the multi-LoRA convergence comparison figure.

    Args:
        data: Aggregated inter-LoRA data.
        config: Figure-specific config (unused currently).

    Returns:
        Figure or None if insufficient data.
    """
    ranks_list = data.rank_values
    if len(ranks_list) < 2:
        logger.warning("Need >= 2 ranks for convergence comparison, got %d", len(ranks_list))
        return None

    norm, cmap = rank_colormap(ranks_list)

    # ── Pre-load and aggregate all data ─────────────────────────────
    all_agg: dict[int, dict[str, Any]] = {}

    for rr in data.ranks:
        if rr.rank == 0:
            continue
        rank_data: dict[str, Any] = {}
        ts_df = _load_threshold_csv(rr.run_dir)

        for label in REGION_KEYS:
            label_data: dict[str, pd.DataFrame] = {}

            sm_df = _load_convergence_csv(rr.run_dir, label)
            ek_df = _load_ensemble_convergence_csv(rr.run_dir, label)

            if sm_df is not None and not sm_df.empty:
                label_data["sample_mean"] = _agg_sample_mean(sm_df)
            else:
                label_data["sample_mean"] = pd.DataFrame()

            if ek_df is not None and not ek_df.empty:
                label_data["ensemble_k"] = _agg_ensemble_k(ek_df, label)
            else:
                label_data["ensemble_k"] = pd.DataFrame()

            if ts_df is not None and not ts_df.empty:
                ens_th, mem_th = _agg_threshold(ts_df, label)
                label_data["threshold_ensemble"] = ens_th
                label_data["threshold_member"] = mem_th
            else:
                label_data["threshold_ensemble"] = pd.DataFrame()
                label_data["threshold_member"] = pd.DataFrame()

            rank_data[label] = label_data
        all_agg[rr.rank] = rank_data

    if not all_agg:
        logger.warning("No convergence data loaded for any rank")
        return None

    _export_data(data.out_root, all_agg)

    # ── Figure layout ───────────────────────────────────────────────
    fig_w = DOUBLE_COL_MM * MM_TO_INCH
    fig_h = _HEIGHT_MM * MM_TO_INCH

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[0.06, 1, 1, 1],
        hspace=0.35,
        wspace=0.18,
        left=0.10,
        right=0.95,
        top=0.93,
        bottom=0.10,
    )

    # ── Colorbar axis (top, spanning both columns) ──────────────────
    ax_cbar = fig.add_subplot(gs[0, :])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_ticks([np.log2(r) for r in ranks_list])
    cb.set_ticklabels([str(r) for r in ranks_list])
    cb.ax.tick_params(labelsize=8)
    cb.ax.set_title(r"LoRA rank $r$", fontsize=9, pad=4)

    # ── Subplot grid (3 rows × 2 cols, shared y per column) ──────────
    axes_conv: list[plt.Axes] = []
    axes_thresh: list[plt.Axes] = []

    # First row establishes the two column y-axis references
    ax_conv_0 = fig.add_subplot(gs[1, 0])
    ax_thresh_0 = fig.add_subplot(gs[1, 1])
    axes_conv.append(ax_conv_0)
    axes_thresh.append(ax_thresh_0)

    # Remaining rows share y within their column
    for row_i in range(1, len(REGION_KEYS)):
        ax_c = fig.add_subplot(gs[row_i + 1, 0], sharey=ax_conv_0)
        ax_t = fig.add_subplot(gs[row_i + 1, 1], sharey=ax_thresh_0)
        axes_conv.append(ax_c)
        axes_thresh.append(ax_t)

    for row_i, label in enumerate(REGION_KEYS):
        ax_conv = axes_conv[row_i]
        ax_thresh = axes_thresh[row_i]

        label_color = LABEL_COLORS[label]

        for rank in sorted(all_agg.keys()):
            color = cmap(norm(np.log2(rank)))
            ld = all_agg[rank][label]

            # Left panel: ensemble convergence
            ek = ld.get("ensemble_k")
            if isinstance(ek, pd.DataFrame) and not ek.empty:
                k_vals = ek["k"].values
                y_vals = ek["y"].values
                se_vals = ek["se"].values
                ax_conv.fill_between(
                    k_vals,
                    y_vals - 1.96 * se_vals,
                    y_vals + 1.96 * se_vals,
                    alpha=0.08,
                    color=color,
                )
                ax_conv.plot(
                    k_vals,
                    y_vals,
                    "s-",
                    color=color,
                    ms=2.5,
                    lw=1.0,
                )

            sm_agg = ld.get("sample_mean")
            if isinstance(sm_agg, pd.DataFrame) and not sm_agg.empty:
                k_sm = sm_agg["k"].values
                y_sm = sm_agg["y"].values
                se_sm = sm_agg["se"].values
                ax_conv.fill_between(
                    k_sm,
                    y_sm - 1.96 * se_sm,
                    y_sm + 1.96 * se_sm,
                    alpha=0.06,
                    color=color,
                )
                ax_conv.plot(
                    k_sm,
                    y_sm,
                    "o--",
                    color=color,
                    ms=2.0,
                    lw=0.7,
                    alpha=0.75,
                )

            # Right panel: threshold sensitivity
            ens_th = ld.get("threshold_ensemble")
            if isinstance(ens_th, pd.DataFrame) and not ens_th.empty:
                col_name = f"dice_{label}"
                ax_thresh.plot(
                    ens_th["threshold"].values,
                    ens_th[col_name].values,
                    "s-",
                    color=color,
                    ms=2.5,
                    lw=1.0,
                )

            mem_th = ld.get("threshold_member")
            if isinstance(mem_th, pd.DataFrame) and not mem_th.empty:
                col_name = f"dice_{label}"
                ax_thresh.plot(
                    mem_th["threshold"].values,
                    mem_th[col_name].values,
                    "o--",
                    color=color,
                    ms=2.0,
                    lw=0.7,
                    alpha=0.75,
                )

        ax_thresh.axvline(0.5, ls=":", color="grey", lw=0.6, alpha=0.5)

        # Row label on left y-axis
        ax_conv.set_ylabel(f"{REGION_DISPLAY[label]}  Dice", fontsize=8)

        for spine in ["top", "right"]:
            ax_conv.spines[spine].set_visible(False)
            ax_thresh.spines[spine].set_visible(False)

        # X-axis labels only on bottom row
        if row_i < len(REGION_KEYS) - 1:
            ax_conv.tick_params(labelbottom=False)
            ax_thresh.tick_params(labelbottom=False)
        else:
            ax_conv.set_xlabel(r"Ensemble size ($k$)", fontsize=8)
            ax_thresh.set_xlabel(r"Binarization threshold ($\tau$)", fontsize=8)

        # Right column: show y-tick labels (different scale from left)
        ax_thresh.set_ylabel("")

    # ── Column titles ───────────────────────────────────────────────
    axes_conv[0].set_title("Ensemble-of-$k$ Dice", fontsize=9, pad=4)
    axes_thresh[0].set_title(r"Dice vs binarization $\tau$", fontsize=9, pad=4)

    # ── Legend (below figure) ───────────────────────────────────────
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="grey",
            marker="s",
            ls="-",
            ms=4,
            lw=1.0,
            label=r"Ensemble prediction ($D_k$ / $D_\tau$)",
        ),
        mlines.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            ls="--",
            ms=3.5,
            lw=0.7,
            alpha=0.75,
            label=r"Mean per-member Dice ($\hat\mu_k$ / $\hat\mu_\tau$)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 0.005),
    )

    return fig
