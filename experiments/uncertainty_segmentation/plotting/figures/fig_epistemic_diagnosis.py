"""Fig 15: Bias–variance landscape across LoRA ranks.

Single figure that merges the previous per-rank diagnostics (bias vs
procedural std) with the cross-rank comparison. All sibling LoRA ranks
are overlaid in one scatter panel so the bias-variance trade-off is
visible qualitatively: as the rank grows, the per-rank cloud migrates
toward larger procedural std and smaller absolute bias, following the
Jiménez et al. (2026) decomposition of second-order epistemic
uncertainty.

Layout::

    ┌────────────────────────────┬──────────────────┬────────┐
    │  Bias–variance landscape   │  Calibration CI  │ legend │
    │  (|bias| vs procedural std │  (empirical vs   │ (out   │
    │  for all ranks)            │   nominal)       │  of    │
    │                            │                  │  plot) │
    └────────────────────────────┴──────────────────┴────────┘

The decision boundary ``|bias| = std`` partitions the plot into a
procedural-dominated region (below) and a bias-contaminated region
(above, shaded); per-rank medians are highlighted with large diamond
markers connected by an arrow so the reader can follow the trade-off
direction at a glance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.epistemic_metrics import (
    RUN_DIR_RE,
)

logger = logging.getLogger(__name__)


def _load_rank_diagnostics(parent_dir: Path) -> dict[int, dict[str, pd.DataFrame]]:
    """Collect per-rank bias_diagnostics + calibration_coverage CSVs.

    Args:
        parent_dir: Directory containing sibling ``r{R}_M{M}_s{S}`` runs.

    Returns:
        Mapping ``rank -> {"bias": DataFrame, "calib": DataFrame}``.
        Ranks missing either cached CSV are dropped with a warning.
    """
    ranks: dict[int, dict[str, pd.DataFrame]] = {}
    if not parent_dir.exists():
        return ranks

    for run_dir in sorted(parent_dir.iterdir()):
        match = RUN_DIR_RE.match(run_dir.name)
        if not match:
            continue
        rank = int(match.group("rank"))
        bias_path = run_dir / "evaluation" / "bias_diagnostics.csv"
        calib_path = run_dir / "evaluation" / "calibration_coverage.csv"
        if not bias_path.exists() or not calib_path.exists():
            logger.warning(
                "[fig_epistemic_diagnosis] rank %d: cached CSVs missing,"
                " skipping", rank,
            )
            continue
        ranks[rank] = {
            "bias": pd.read_csv(bias_path),
            "calib": pd.read_csv(calib_path),
        }
    return ranks


def _rank_colors(ranks: list[int]) -> dict[int, tuple[float, float, float, float]]:
    """Assign perceptually ordered viridis colors to an ordered rank list."""
    if len(ranks) == 1:
        return {ranks[0]: tuple(cm.viridis(0.5))}
    stops = np.linspace(0.15, 0.85, len(ranks))
    return {r: tuple(cm.viridis(s)) for r, s in zip(ranks, stops)}


def plot(
    data: EnsembleResultsData,
    config: dict,
) -> matplotlib.figure.Figure | None:
    """Generate the bias-variance landscape figure.

    Args:
        data: Loaded experiment data. Only ``run_dir`` is consulted; all
            other inputs come from sibling ranks' cached CSVs.
        config: Figure config block from ``config.yaml``.

    Returns:
        The Figure object, or None if fewer than two ranks are available.
    """
    parent_dir = data.run_dir.parent
    rank_filter = config.get("ranks")
    all_rank_data = _load_rank_diagnostics(parent_dir)
    if rank_filter:
        all_rank_data = {r: v for r, v in all_rank_data.items() if r in rank_filter}

    if len(all_rank_data) < 2:
        logger.warning(
            "[fig_epistemic_diagnosis] only %d rank(s) available under %s;"
            " skipping (need >= 2 for a cross-rank landscape).",
            len(all_rank_data), parent_dir,
        )
        return None

    ranks = sorted(all_rank_data.keys())
    colors = _rank_colors(ranks)

    figsize = config.get("figsize", [10.5, 4.2])
    use_log = bool(config.get("log_scale", True))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.35, 1.0],
        wspace=0.28,
    )
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_calib = fig.add_subplot(gs[0, 1])

    _plot_bias_variance_landscape(
        ax_scatter, all_rank_data, colors, log_scale=use_log,
    )
    _plot_calibration_curves(ax_calib, all_rank_data, colors)

    # External shared legend — synthetic Line2D handles so colors/markers
    # render clearly (scatter handles are too faint at the legend's scale).
    rank_handles = [
        Line2D(
            [0], [0], marker="D", markersize=9,
            markerfacecolor=colors[r], markeredgecolor="black",
            markeredgewidth=0.9, linestyle="none",
            label=f"r = {r}",
        )
        for r in ranks
    ]
    extras = [
        Line2D([0], [0], color="black", lw=0.9, linestyle="--",
               label="|bias| = std"),
        Line2D([0], [0], color="#D55E00", lw=6, alpha=0.2,
               label="bias-contaminated zone"),
    ]
    handles = rank_handles + extras
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="center left", bbox_to_anchor=(0.865, 0.5),
        frameon=False, fontsize=8,
        handlelength=1.8, borderaxespad=0.0,
        title="LoRA rank",
        title_fontsize=9,
        alignment="left",
    )

    # Leave room on the right for the out-of-plot legend.
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    return fig


def _plot_bias_variance_landscape(
    ax,
    rank_data: dict[int, dict[str, pd.DataFrame]],
    colors: dict[int, tuple],
    *,
    log_scale: bool,
) -> None:
    """Main panel: |bias| vs procedural std, colored by LoRA rank."""
    floor = 1e-4
    all_std: list[np.ndarray] = []
    all_bias: list[np.ndarray] = []

    ranks_sorted = sorted(rank_data.keys())

    # Compute global axis limits up front so shading / region labels get
    # the correct extent before data is plotted on top.
    for rank in ranks_sorted:
        bias_df = rank_data[rank]["bias"]
        all_std.append(np.clip(
            bias_df["logvol_ensemble_std"].to_numpy(), floor, None,
        ))
        all_bias.append(np.clip(
            bias_df["logvol_abs_bias"].to_numpy(), floor, None,
        ))
    combined_std = np.concatenate(all_std)
    combined_bias = np.concatenate(all_bias)
    lim_min = min(combined_std.min(), combined_bias.min()) * 0.8
    lim_max = max(combined_std.max(), combined_bias.max()) * 1.2

    # Shaded bias-contamination region (above diagonal) — drawn first so
    # scatter points sit on top.
    n_shade = 128
    log_grid = np.logspace(np.log10(lim_min), np.log10(lim_max), n_shade)
    ax.fill_between(
        log_grid, log_grid, np.full_like(log_grid, lim_max),
        color="#D55E00", alpha=0.07, zorder=1, linewidth=0,
    )

    # Diagonal decision boundary.
    diag = np.array([lim_min, lim_max])
    ax.plot(
        diag, diag, "k--", lw=0.9, alpha=0.7, zorder=3,
    )

    # Background scatter layer (one color per rank, low alpha).
    for rank, std, abs_bias in zip(ranks_sorted, all_std, all_bias):
        ax.scatter(
            std, abs_bias,
            s=10, color=colors[rank], alpha=0.30, edgecolors="none",
            zorder=2,
        )

    # Per-rank medians — large diamonds. Labels go in the external legend.
    median_points: list[tuple[int, float, float]] = []
    for rank in ranks_sorted:
        bias_df = rank_data[rank]["bias"]
        med_std = max(float(bias_df["logvol_ensemble_std"].median()), floor)
        med_bias = max(float(bias_df["logvol_abs_bias"].median()), floor)
        median_points.append((rank, med_std, med_bias))
        ax.scatter(
            [med_std], [med_bias],
            s=170, marker="D", color=colors[rank],
            edgecolors="black", linewidths=1.3,
            zorder=6,
        )

    # Arrow from lowest to highest rank showing the trade-off direction.
    _, x_lo, y_lo = median_points[0]
    _, x_hi, y_hi = median_points[-1]
    ax.add_patch(FancyArrowPatch(
        (x_lo, y_lo), (x_hi, y_hi),
        arrowstyle="-|>", mutation_scale=14,
        color="black", lw=1.3, alpha=0.85, zorder=5,
        shrinkA=10, shrinkB=10,
    ))
    # Trade-off annotation placed near the arrow midpoint (geometric mid
    # in log-space).
    mid_x = np.sqrt(x_lo * x_hi)
    mid_y = np.sqrt(y_lo * y_hi)
    ax.annotate(
        "rank ↑\n(bias ↓, var ↑)",
        xy=(mid_x, mid_y),
        xytext=(mid_x * 2.2, mid_y * 0.35),
        fontsize=7.5, color="black",
        ha="center", va="center",
        arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.6),
        zorder=7,
    )

    # Region labels in the corners so they never touch data.
    ax.text(
        lim_min * 1.5, lim_max * 0.65,
        "BIAS-CONTAMINATED\n|bias| > std",
        fontsize=7.5, color="#8B0000", alpha=0.9, fontweight="bold",
        ha="left", va="top", zorder=3,
    )
    ax.text(
        lim_max * 0.95, lim_min * 1.5,
        "PROCEDURAL-DOMINATED\n|bias| ≤ std",
        fontsize=7.5, color="#00468B", alpha=0.9, fontweight="bold",
        ha="right", va="bottom", zorder=3,
    )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Procedural std  Std[log(V)]   (variance)")
    ax.set_ylabel("|Bias|   |mean log(V) − log(V_GT)|   (bias)")
    ax.set_title("Bias–variance landscape across LoRA ranks",
                 fontweight="bold")


def _plot_calibration_curves(
    ax,
    rank_data: dict[int, dict[str, pd.DataFrame]],
    colors: dict[int, tuple],
) -> None:
    """Right panel: one calibration curve per rank."""
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)

    for rank in sorted(rank_data.keys()):
        calib_df = rank_data[rank]["calib"].sort_values("nominal_level")
        nominal = calib_df["nominal_level"].to_numpy()
        empirical = calib_df["empirical_coverage"].to_numpy()
        ax.plot(
            nominal, empirical,
            marker="o", color=colors[rank], lw=1.5, ms=5,
        )
        # 95% deficit label near the last point.
        if len(nominal):
            deficit = float(calib_df["coverage_deficit"].iloc[-1])
            ax.text(
                nominal[-1] + 0.01, empirical[-1],
                f"Δ={deficit:+.2f}",
                fontsize=6.5, color=colors[rank],
                va="center", ha="left",
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration per rank", fontweight="bold")
    ax.text(
        0.5, 0.92,
        "below diagonal =\nundercoverage",
        transform=ax.transAxes, fontsize=6.5,
        ha="center", va="top", color="gray",
    )
