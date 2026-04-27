"""Dice vs LoRA rank — boxplot panel + split pairwise heatmap.

Upper triangle: Wilcoxon p-values (RdYlGn log-scale).
Lower triangle: Cohen's d effect size (viridis).
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

from ..io_layer import InterLoraData
from ..style import (
    DOUBLE_COL_MM,
    LABEL_COLORS,
    MM_TO_INCH,
    REGION_DISPLAY,
    REGION_KEYS,
)

logger = logging.getLogger(__name__)

_BOX_WIDTH: float = 0.22
_LABEL_OFFSETS: dict[str, float] = {"tc": -0.25, "wt": 0.0, "et": 0.25}
_GAP: float = 1.5


def _build_positions(n_ranks: int) -> tuple[list[float], float]:
    """Build nominal x-positions: BSF at 0, gap, then ranks."""
    bsf_pos = 0.0
    rank_positions = [_GAP + i for i in range(n_ranks)]
    return [bsf_pos] + rank_positions, bsf_pos


def _draw_boxplot(
    ax: plt.Axes,
    values: np.ndarray,
    pos: float,
    color: str,
    rng: np.random.Generator,
) -> None:
    """Draw a single boxplot (black frame) + jittered dots (label-coloured)."""
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return

    bp = ax.boxplot(
        clean,
        positions=[pos],
        widths=_BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(0.7)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(1.0)

    jitter = rng.uniform(-_BOX_WIDTH * 0.35, _BOX_WIDTH * 0.35, size=len(clean))
    ax.scatter(
        pos + jitter,
        clean,
        color=color,
        alpha=0.25,
        s=6,
        linewidths=0,
        zorder=4,
    )


def _compute_pairwise(
    data: InterLoraData,
    label: str = "mean",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute pairwise Wilcoxon p-values and Cohen's d across all conditions.

    Returns:
        (p_matrix, d_matrix, condition_labels).
    """
    col = f"dice_{label}" if label != "mean" else "dice_mean"

    conditions: list[tuple[str, pd.Series]] = []
    baseline_run = data.ranks[0]
    bas = baseline_run.baseline_dice.set_index("scan_id")[col]
    conditions.append(("Frozen BSF", bas))

    for rank in data.rank_values:
        rr = data.get_rank(rank)
        ens = rr.ensemble_dice.set_index("scan_id")[col]
        conditions.append((f"$r={rank}$", ens))

    n = len(conditions)
    p_matrix = np.ones((n, n), dtype=float)
    d_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            _, vals_i = conditions[i]
            _, vals_j = conditions[j]
            common = vals_i.index.intersection(vals_j.index)
            if len(common) < 5:
                continue
            x = vals_i.loc[common].values
            y = vals_j.loc[common].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 5:
                continue
            xv, yv = x[valid], y[valid]
            try:
                _, p = stats.wilcoxon(xv, yv, alternative="two-sided")
            except ValueError:
                p = 1.0
            p_matrix[i, j] = p
            p_matrix[j, i] = p

            diff = xv - yv
            sd = diff.std(ddof=1)
            d_val = float(diff.mean() / sd) if sd > 1e-15 else 0.0
            d_matrix[i, j] = d_val
            d_matrix[j, i] = -d_val

    cond_labels = [c[0] for c in conditions]
    return p_matrix, d_matrix, cond_labels


def _draw_split_heatmap(
    ax_heat: plt.Axes,
    ax_cbar_p: plt.Axes,
    ax_cbar_d: plt.Axes,
    p_matrix: np.ndarray,
    d_matrix: np.ndarray,
    labels: list[str],
) -> None:
    """Draw split heatmap: upper-tri = p-values, lower-tri = Cohen's d."""
    n = len(labels)

    p_norm = LogNorm(vmin=1e-30, vmax=1.0)
    d_abs_max = max(np.abs(d_matrix).max(), 0.5)
    d_norm = Normalize(vmin=-d_abs_max, vmax=d_abs_max)

    p_cmap = plt.cm.RdYlGn
    d_cmap = plt.cm.viridis

    canvas = np.full((n, n, 4), np.nan)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j > i:
                val = p_norm(p_matrix[i, j])
                canvas[i, j] = p_cmap(val)
            else:
                val = d_norm(d_matrix[i, j])
                canvas[i, j] = d_cmap(val)

    diag_color = np.array([0.85, 0.85, 0.85, 1.0])
    for i in range(n):
        canvas[i, i] = diag_color

    ax_heat.imshow(canvas, aspect="equal", interpolation="nearest")
    ax_heat.set_rasterized(True)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j > i:
                p = p_matrix[i, j]
                if p < 0.001:
                    txt = f"{p:.0e}"
                elif p < 0.05:
                    txt = f"{p:.3f}"
                else:
                    txt = f"{p:.2f}"
            else:
                d_val = d_matrix[i, j]
                txt = f"{d_val:.2f}"

            ax_heat.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=5,
                color="black",
            )

    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax_heat.set_yticklabels(labels, fontsize=6)

    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    sm_p = plt.cm.ScalarMappable(cmap=p_cmap, norm=p_norm)
    sm_p.set_array([])
    cbar_p = ax_heat.figure.colorbar(sm_p, cax=ax_cbar_p)
    cbar_p.set_label("$p$-value (upper tri.)", fontsize=6)
    cbar_p.ax.tick_params(labelsize=5)

    sm_d = plt.cm.ScalarMappable(cmap=d_cmap, norm=d_norm)
    sm_d.set_array([])
    cbar_d = ax_heat.figure.colorbar(sm_d, cax=ax_cbar_d, orientation="horizontal")
    cbar_d.set_label("Cohen's $d$ (lower tri.)", fontsize=6)
    cbar_d.ax.tick_params(labelsize=5)


def plot(data: InterLoraData, config: dict[str, Any]) -> Figure | None:
    """Boxplot + split pairwise heatmap (p-values upper / Cohen's d lower).

    Args:
        data: Aggregated inter-rank data.
        config: Figure configuration.

    Returns:
        Figure or None if insufficient data.
    """
    rank_values = data.rank_values
    if len(rank_values) < 2:
        logger.warning("quant1: need >= 2 ranks, got %d", len(rank_values))
        return None

    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    all_positions, bsf_pos = _build_positions(len(rank_values))
    rank_positions = all_positions[1:]

    fig_w = DOUBLE_COL_MM * MM_TO_INCH
    fig_h = 85.0 * MM_TO_INCH
    figsize = tuple(config.get("figsize", [fig_w, fig_h]))

    n_conds = len(rank_values) + 1
    heatmap_rel = n_conds / 6.0
    cbar_p_rel = 0.06
    gap_rel = 0.08

    fig = plt.figure(figsize=figsize)

    gs_top = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=[2.2, gap_rel, heatmap_rel, cbar_p_rel],
        wspace=0.05,
        left=0.08,
        right=0.97,
        top=0.88,
        bottom=0.22,
    )

    ax_box = fig.add_subplot(gs_top[0, 0])
    ax_heat = fig.add_subplot(gs_top[0, 2])
    ax_cbar_p = fig.add_subplot(gs_top[0, 3])

    heat_pos = ax_heat.get_position()
    ax_cbar_d = fig.add_axes(
        [
            heat_pos.x0,
            0.08,
            heat_pos.width,
            0.03,
        ]
    )

    baseline_run = data.ranks[0]
    baseline_df = baseline_run.baseline_dice

    # ── Left panel: Boxplots ──
    for region in REGION_KEYS:
        col = f"dice_{region}"
        color = LABEL_COLORS[region]
        offset = _LABEL_OFFSETS[region]

        if col in baseline_df.columns:
            bsf_vals = baseline_df[col].dropna().values
            _draw_boxplot(ax_box, bsf_vals, bsf_pos + offset, color, rng)

        for r_idx, rank in enumerate(rank_values):
            rr = data.get_rank(rank)
            pos = rank_positions[r_idx] + offset
            if col in rr.ensemble_dice.columns:
                vals = rr.ensemble_dice[col].dropna().values
                _draw_boxplot(ax_box, vals, pos, color, rng)

    sep_x = (bsf_pos + rank_positions[0]) / 2
    ax_box.axvline(sep_x, color="#cccccc", linewidth=0.5, linestyle="--", zorder=0)

    all_labels = ["Frozen\nBSF"] + [f"$r={r}$" for r in rank_values]
    ax_box.set_xticks(all_positions)
    ax_box.set_xticklabels(all_labels, fontsize=7)
    ax_box.set_ylabel("Dice coefficient", fontsize=8)
    ax_box.set_ylim(top=1.02)

    rank_center = (rank_positions[0] + rank_positions[-1]) / 2
    y_label_pos = -0.18
    ax_box.annotate(
        "LoRA rank",
        xy=(rank_center, 0),
        xycoords=("data", "axes fraction"),
        xytext=(rank_center, y_label_pos),
        textcoords=("data", "axes fraction"),
        fontsize=8,
        ha="center",
        va="top",
        annotation_clip=False,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=LABEL_COLORS[r],
            linewidth=4,
            alpha=0.4,
            label=REGION_DISPLAY[r],
        )
        for r in REGION_KEYS
    ]
    ax_box.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=7,
    )

    # ── Right panel: Split heatmap ──
    p_matrix, d_matrix, cond_labels = _compute_pairwise(data, label="mean")

    fig.canvas.draw()
    heat_pos = ax_heat.get_position()
    ax_cbar_d.set_position(
        [
            heat_pos.x0,
            0.08,
            heat_pos.width,
            0.03,
        ]
    )

    _draw_split_heatmap(ax_heat, ax_cbar_p, ax_cbar_d, p_matrix, d_matrix, cond_labels)

    logger.info("quant1: rendered %d ranks + BSF, 3 labels + split heatmap", len(rank_values))
    return fig
