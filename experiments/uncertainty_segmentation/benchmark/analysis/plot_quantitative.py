"""Quantitative figures: per-metric (boxplot + pairwise heatmap) and combined panel.

Style mirrors ``inter_lora/figures/quant1_dice_vs_rank.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiments.uncertainty_segmentation.plotting.inter_lora.style import (
    DOUBLE_COL_MM,
    LABEL_COLORS as _BASE_LABEL_COLORS,
    MM_TO_INCH,
    save_figure,
)
from experiments.uncertainty_segmentation.plotting.style import setup_style

from .io import DEFAULT_ANALYSIS_ROOT, OURS_MODEL_ID
from .metrics import LABELS, METRICS

logger = logging.getLogger(__name__)

# Five-label palette: tc/wt/et reuse the inter_lora colours.
LABEL_COLORS_5: dict[str, str] = {
    "TC": _BASE_LABEL_COLORS["tc"],
    "WT": _BASE_LABEL_COLORS["wt"],
    "ET": _BASE_LABEL_COLORS["et"],
    "NETC": "#CC79A7",
    "SNFH": "#F0E442",
}

METRIC_YLABEL: dict[str, str] = {
    "dice": "Dice coefficient",
    "hd95": "HD95 (mm)",
    "lesion_recall": "Lesion recall",
}

METRIC_TITLE: dict[str, str] = {
    "dice": "Dice",
    "hd95": "HD95",
    "lesion_recall": "Lesion recall",
}

_BOX_WIDTH: float = 0.16
# Five offsets per model column, keeping TC/WT/ET in the middle for visual continuity with quant1.
_LABEL_OFFSETS: dict[str, float] = {
    "NETC": -0.36,
    "TC": -0.18,
    "WT": 0.0,
    "ET": 0.18,
    "SNFH": 0.36,
}


def _draw_boxplot(
    ax: plt.Axes,
    values: np.ndarray,
    pos: float,
    color: str,
    rng: np.random.Generator,
) -> None:
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
        s=5,
        linewidths=0,
        zorder=4,
    )


def _draw_pairwise_heatmap(
    ax_heat: plt.Axes,
    ax_cbar_p: plt.Axes,
    ax_cbar_d: plt.Axes,
    p_matrix: np.ndarray,
    d_matrix: np.ndarray,
    labels: list[str],
) -> None:
    """Compound heatmap: upper triangle = p-values (RdYlGn log), lower = Cohen's d (viridis)."""
    n = len(labels)
    p_norm = LogNorm(vmin=1e-30, vmax=1.0)
    d_abs_max = max(float(np.abs(d_matrix).max()), 0.5)
    d_norm = Normalize(vmin=-d_abs_max, vmax=d_abs_max)
    p_cmap = plt.cm.RdYlGn
    d_cmap = plt.cm.viridis
    canvas = np.full((n, n, 4), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j > i:
                canvas[i, j] = p_cmap(p_norm(p_matrix[i, j]))
            else:
                canvas[i, j] = d_cmap(d_norm(d_matrix[i, j]))
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
                pv = p_matrix[i, j]
                if pv < 0.001:
                    txt = f"{pv:.0e}"
                elif pv < 0.05:
                    txt = f"{pv:.3f}"
                else:
                    txt = f"{pv:.2f}"
            else:
                txt = f"{d_matrix[i, j]:.2f}"
            ax_heat.text(j, i, txt, ha="center", va="center", fontsize=5, color="black")
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


def _draw_metric_panel(
    fig: Figure,
    gs_row,
    df_metric: pd.DataFrame,
    metric: str,
    model_order: list[str],
    p_matrix: np.ndarray,
    d_matrix: np.ndarray,
    heatmap_label: str,
    rng: np.random.Generator,
) -> tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    """Draw one (boxplot, heatmap, two cbars) row inside a parent figure."""
    n_models = len(model_order)
    heatmap_rel = max(n_models / 6.0, 0.6)
    cbar_p_rel = 0.06
    gap_rel = 0.08
    sub = gs_row.subgridspec(
        nrows=1,
        ncols=4,
        width_ratios=[2.6, gap_rel, heatmap_rel, cbar_p_rel],
        wspace=0.05,
    )
    ax_box = fig.add_subplot(sub[0, 0])
    ax_heat = fig.add_subplot(sub[0, 2])
    ax_cbar_p = fig.add_subplot(sub[0, 3])

    # Boxplot ---------------------------------------------------------------
    positions = list(range(n_models))
    for col_idx, model in enumerate(model_order):
        df_m = df_metric[df_metric["model"] == model]
        for label in LABELS:
            offset = _LABEL_OFFSETS[label]
            color = LABEL_COLORS_5[label]
            vals = df_m[df_m["label"] == label][metric].to_numpy(dtype=float)
            _draw_boxplot(ax_box, vals, positions[col_idx] + offset, color, rng)

    # Separator before "Ours"
    if OURS_MODEL_ID in model_order:
        ours_idx = model_order.index(OURS_MODEL_ID)
        if ours_idx > 0:
            sep_x = (positions[ours_idx - 1] + positions[ours_idx]) / 2
            ax_box.axvline(sep_x, color="#cccccc", linewidth=0.5, linestyle="--", zorder=0)

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(model_order, fontsize=7, rotation=20, ha="right")
    ax_box.set_ylabel(METRIC_YLABEL[metric], fontsize=8)
    if metric == "dice" or metric == "lesion_recall":
        ax_box.set_ylim(-0.02, 1.02)
    else:
        # HD95 — clamp the visible range to the 99th percentile to avoid extreme outliers
        finite = df_metric[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            top = float(np.nanpercentile(finite.to_numpy(), 99))
            ax_box.set_ylim(0.0, max(top * 1.05, 1.0))

    legend_elements = [
        Line2D([0], [0], color=LABEL_COLORS_5[lbl], linewidth=4, alpha=0.4, label=lbl)
        for lbl in LABELS
    ]
    ax_box.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=5,
        frameon=False,
        fontsize=7,
    )

    # Heatmap (placed below boxplot via add_axes for the horizontal d-cbar) -
    fig.canvas.draw()
    heat_pos = ax_heat.get_position()
    ax_cbar_d = fig.add_axes(
        [heat_pos.x0, heat_pos.y0 - 0.06, heat_pos.width, 0.025]
    )
    _draw_pairwise_heatmap(ax_heat, ax_cbar_p, ax_cbar_d, p_matrix, d_matrix, model_order)
    ax_heat.set_title(f"Pairwise {METRIC_TITLE[metric]} on {heatmap_label}", fontsize=7, pad=4)
    return ax_box, ax_heat, ax_cbar_p, ax_cbar_d


def make_metric_figure(
    df: pd.DataFrame,
    metric: str,
    model_order: list[str],
    p_matrix: np.ndarray,
    d_matrix: np.ndarray,
    heatmap_label: str = "TC",
    seed: int = 42,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Per-metric quantitative figure (boxplot + heatmap)."""
    rng = np.random.default_rng(seed)
    fig_w = DOUBLE_COL_MM * MM_TO_INCH
    fig_h = 95.0 * MM_TO_INCH
    fig = plt.figure(figsize=figsize or (fig_w, fig_h))
    gs = fig.add_gridspec(nrows=1, ncols=1, left=0.08, right=0.97, top=0.85, bottom=0.30)
    df_metric = df[["model", "case_id", "label", metric]].copy()
    _draw_metric_panel(
        fig, gs[0, 0], df_metric, metric, model_order, p_matrix, d_matrix, heatmap_label, rng
    )
    return fig


def make_combined_figure(
    df: pd.DataFrame,
    pairwise: dict[str, dict[str, dict[str, Any]]],
    model_order: list[str],
    heatmap_label: str = "TC",
    seed: int = 42,
) -> Figure:
    """Combined N-row figure: one row per metric, two columns (boxplot | heatmap)."""
    rng = np.random.default_rng(seed)
    fig_w = DOUBLE_COL_MM * MM_TO_INCH
    fig_h = 95.0 * MM_TO_INCH * len(METRICS)
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        nrows=len(METRICS),
        ncols=1,
        left=0.08,
        right=0.97,
        top=1.0 - 0.05 / len(METRICS),
        bottom=0.05 / len(METRICS),
        hspace=0.55,
    )
    df_local = df.copy()
    for row_idx, metric in enumerate(METRICS):
        block = pairwise[metric][heatmap_label]
        p_mat = block["p"].reindex(index=model_order, columns=model_order).to_numpy()
        d_mat = block["d"].reindex(index=model_order, columns=model_order).to_numpy()
        _draw_metric_panel(
            fig,
            outer[row_idx, 0],
            df_local[["model", "case_id", "label", metric]],
            metric,
            model_order,
            p_mat,
            d_mat,
            heatmap_label,
            rng,
        )
    return fig


def write_quantitative(
    df: pd.DataFrame,
    pairwise: dict[str, dict[str, dict[str, Any]]],
    model_order: list[str],
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    heatmap_label: str = "TC",
    seed: int = 42,
) -> list[Path]:
    """Write per-metric figures and the combined panel as PDF + PNG."""
    setup_style()
    out_dir = analysis_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for metric in METRICS:
        block = pairwise[metric][heatmap_label]
        p_mat = block["p"].reindex(index=model_order, columns=model_order).to_numpy()
        d_mat = block["d"].reindex(index=model_order, columns=model_order).to_numpy()
        fig = make_metric_figure(
            df,
            metric=metric,
            model_order=model_order,
            p_matrix=p_mat,
            d_matrix=d_mat,
            heatmap_label=heatmap_label,
            seed=seed,
        )
        for ext in (".pdf", ".png"):
            target = out_dir / f"quant_{metric}{ext}"
            save_figure(
                fig,
                target,
                title=f"BraTS-MEN benchmark: {METRIC_TITLE[metric]}",
                description=f"Pairwise heatmap on label={heatmap_label}; n_models={len(model_order)}",
            )
            paths.append(target)
        plt.close(fig)
    fig = make_combined_figure(df, pairwise, model_order, heatmap_label=heatmap_label, seed=seed)
    for ext in (".pdf", ".png"):
        target = out_dir / f"quant_combined{ext}"
        save_figure(
            fig,
            target,
            title="BraTS-MEN benchmark: combined panel",
            description=f"All metrics × labels, heatmap on label={heatmap_label}",
        )
        paths.append(target)
    plt.close(fig)
    logger.info("plot: wrote %d quantitative figures to %s", len(paths), out_dir)
    return paths
