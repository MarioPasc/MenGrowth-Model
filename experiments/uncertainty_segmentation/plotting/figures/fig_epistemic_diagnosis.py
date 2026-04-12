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

    # Plot individual scans per rank (background layer, low alpha).
    for rank in ranks_sorted:
        bias_df = rank_data[rank]["bias"]
        std = np.clip(bias_df["logvol_ensemble_std"].to_numpy(), floor, None)
        abs_bias = np.clip(bias_df["logvol_abs_bias"].to_numpy(), floor, None)
        all_std.append(std)
        all_bias.append(abs_bias)

        color = colors[rank]
        ax.scatter(
            std, abs_bias,
            s=10, color=color, alpha=0.28, edgecolors="none",
            label=f"r = {rank}",
            zorder=2,
        )

    # Plot per-rank medians as large diamond markers (foreground).
    median_points: list[tuple[int, float, float]] = []
    for rank in ranks_sorted:
        bias_df = rank_data[rank]["bias"]
        med_std = float(bias_df["logvol_ensemble_std"].median())
        med_bias = float(bias_df["logvol_abs_bias"].median())
        med_std_clip = max(med_std, floor)
        med_bias_clip = max(med_bias, floor)
        median_points.append((rank, med_std_clip, med_bias_clip))
        ax.scatter(
            [med_std_clip], [med_bias_clip],
            s=180, marker="D", color=colors[rank],
            edgecolors="black", linewidths=1.3,
            zorder=5,
        )
        ax.annotate(
            f"r={rank}",
            xy=(med_std_clip, med_bias_clip),
            xytext=(10, -3), textcoords="offset points",
            fontsize=8, fontweight="bold", color="black",
            zorder=6,
        )

    # Connect medians with an arrow to show the trade-off direction.
    for (r_a, x_a, y_a), (r_b, x_b, y_b) in zip(median_points[:-1], median_points[1:]):
        ax.add_patch(FancyArrowPatch(
            (x_a, y_a), (x_b, y_b),
            arrowstyle="-|>", mutation_scale=12,
            color="black", lw=1.1, alpha=0.7,
            zorder=4,
        ))

    # Decision boundary |bias| = std.
    combined_std = np.concatenate(all_std)
    combined_bias = np.concatenate(all_bias)
    lim_min = min(combined_std.min(), combined_bias.min()) * 0.8
    lim_max = max(combined_std.max(), combined_bias.max()) * 1.2
    diag = np.array([lim_min, lim_max])
    ax.plot(
        diag, diag, "k--", lw=0.9, alpha=0.7,
        label="|bias| = std  (decision boundary)",
        zorder=3,
    )

    # Shaded bias-contamination region (above diagonal).
    ax.fill_between(
        diag, diag, [lim_max, lim_max],
        color="#D55E00", alpha=0.07, zorder=1,
    )
    # Region labels.
    mid = np.sqrt(lim_min * lim_max)
    ax.text(
        mid * 0.25, mid * 4, "bias-contaminated\n(|bias| > std)",
        fontsize=7.5, color="#8B0000", alpha=0.8,
        ha="left", va="center", zorder=1,
    )
    ax.text(
        mid * 1.8, mid * 0.25, "procedural-dominated\n(|bias| ≤ std)",
        fontsize=7.5, color="#00468B", alpha=0.8,
        ha="center", va="center", zorder=1,
    )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Procedural std  Std[log(V)]  (variance-like)")
    ax.set_ylabel("|Bias|  |mean[log(V)] − log(V_GT)|  (bias-like)")
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
