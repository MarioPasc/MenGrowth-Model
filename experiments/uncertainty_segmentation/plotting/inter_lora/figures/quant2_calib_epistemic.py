"""Calibration and Epistemic Diagnostics vs LoRA Rank (Quant2).

2×2 grid figure showing:
  (a) ECE + Brier score vs rank
  (b) Coverage deficit for nominal levels {0.50, 0.80, 0.90, 0.95}
  (c) Bias dominance taxonomy fractions vs rank
  (d) Inter-member agreement ICC vs rank

All panels share a log₂(rank) x-axis.  A cross-panel consensus band
marks the median of the four per-panel optimal ranks.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..io_layer import InterLoraData
from ..style import (
    DOUBLE_COL_MM,
    LABEL_COLORS,
    MM_TO_INCH,
    REGION_DISPLAY,
    setup_inter_lora_style,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coverage-level appearance
# ---------------------------------------------------------------------------
_COVERAGE_LEVELS: list[float] = [0.50, 0.80, 0.90, 0.95]
_COVERAGE_MARKERS: list[str] = ["o", "s", "^", "D"]
_COVERAGE_COLORS: list[str] = ["#4477AA", "#66CCEE", "#228833", "#AA3377"]

# Bias dominance series appearance
_BIAS_COLORS: dict[str, str] = {
    "pct_scans_k_star_eq_1": "#CC3311",  # red
    "pct_scans_k_star_exceeds_M": "#E69F00",  # orange
    "pct_scans_degenerate_ensemble": "#999999",  # grey
}
_BIAS_LABELS: dict[str, str] = {
    "pct_scans_k_star_eq_1": r"$k^*=1$ (single-mode)",
    "pct_scans_k_star_exceeds_M": r"$k^*>M$ (over-dispersed)",
    "pct_scans_degenerate_ensemble": "degenerate ensemble",
}

# ICC region appearance (use spec-mandated LABEL_COLORS)
_ICC_KEYS: dict[str, str] = {
    "icc_tc": "tc",
    "icc_wt": "wt",
    "icc_et": "et",
}

# Tolerance for ICC flattening detection
_ICC_FLATTEN_TOL: float = 0.005


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _extract_calibration_series(
    data: InterLoraData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ECE and Brier score arrays aligned to rank_values.

    Args:
        data: Aggregated inter-rank data.

    Returns:
        Tuple (log2_ranks, ece_arr, brier_arr) as float arrays.
    """
    rank_values = data.rank_values
    log2_ranks = np.array([np.log2(r) for r in rank_values], dtype=float)
    ece_arr = np.full(len(rank_values), np.nan)
    brier_arr = np.full(len(rank_values), np.nan)

    for i, r in enumerate(rank_values):
        rr = data.get_rank(r)
        cal = rr.calibration
        if cal:
            ece_arr[i] = float(cal.get("ece", np.nan))
            brier_arr[i] = float(cal.get("brier_score", np.nan))

    return log2_ranks, ece_arr, brier_arr


def _extract_coverage_series(
    data: InterLoraData,
) -> dict[float, np.ndarray]:
    """Extract coverage deficit for each nominal level across ranks.

    Args:
        data: Aggregated inter-rank data.

    Returns:
        Dict mapping nominal_level -> deficit array (shape [n_ranks]).
    """
    rank_values = data.rank_values
    result: dict[float, np.ndarray] = {
        lv: np.full(len(rank_values), np.nan) for lv in _COVERAGE_LEVELS
    }

    for i, r in enumerate(rank_values):
        rr = data.get_rank(r)
        df = rr.calibration_coverage
        if df.empty:
            # Fall back to epistemic_taxonomy calibration dict
            tax = rr.epistemic_taxonomy
            cal_node = tax.get("calibration", {})
            _map = {
                0.50: "coverage_50",
                0.80: "coverage_80",
                0.90: "coverage_90",
                0.95: "coverage_deficit_95",
            }
            for lv in _COVERAGE_LEVELS:
                key = _map.get(lv, "")
                if key and key in cal_node:
                    if lv == 0.95:
                        # already a deficit
                        result[lv][i] = float(cal_node[key])
                    else:
                        emp = float(cal_node[key])
                        result[lv][i] = emp - lv
            continue

        for lv in _COVERAGE_LEVELS:
            row = df[np.isclose(df["nominal_level"], lv)]
            if row.empty:
                continue
            if "coverage_deficit" in df.columns:
                result[lv][i] = float(row["coverage_deficit"].iloc[0])
            elif "empirical_coverage" in df.columns:
                emp = float(row["empirical_coverage"].iloc[0])
                result[lv][i] = emp - lv

    return result


def _extract_bias_series(
    data: InterLoraData,
) -> dict[str, np.ndarray]:
    """Extract bias dominance fractions across ranks.

    Args:
        data: Aggregated inter-rank data.

    Returns:
        Dict mapping series_key -> fraction array (shape [n_ranks]).
    """
    rank_values = data.rank_values
    keys = list(_BIAS_COLORS.keys())
    result: dict[str, np.ndarray] = {k: np.full(len(rank_values), np.nan) for k in keys}

    for i, r in enumerate(rank_values):
        rr = data.get_rank(r)
        tax = rr.epistemic_taxonomy
        bd = tax.get("taxonomy", {}).get("estimation_bias", {}).get("bias_dominance", {})
        for k in keys:
            if k in bd:
                result[k][i] = float(bd[k])

    return result


def _extract_icc_series(
    data: InterLoraData,
) -> dict[str, np.ndarray]:
    """Extract ICC per region across ranks.

    Args:
        data: Aggregated inter-rank data.

    Returns:
        Dict mapping icc_key -> ICC array (shape [n_ranks]).
    """
    rank_values = data.rank_values
    result: dict[str, np.ndarray] = {k: np.full(len(rank_values), np.nan) for k in _ICC_KEYS}

    for i, r in enumerate(rank_values):
        rr = data.get_rank(r)
        ss = rr.statistical_summary
        ima = ss.get("inter_member_agreement", {})
        for k in _ICC_KEYS:
            if k in ima:
                result[k][i] = float(ima[k])

    return result


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _panel_calibration(
    ax: plt.Axes,
    log2_ranks: np.ndarray,
    ece_arr: np.ndarray,
    brier_arr: np.ndarray,
    rank_values: list[int],
) -> float | None:
    """Render panel (a): ECE (left y) + Brier score (right y) vs log₂ rank.

    Args:
        ax: Primary axes for ECE.
        log2_ranks: log₂-transformed rank values.
        ece_arr: ECE per rank.
        brier_arr: Brier score per rank.
        rank_values: Raw integer rank values for x-tick labels.

    Returns:
        log₂(rank) of ECE minimum, or None if no valid data.
    """
    color_ece = "#1f77b4"
    color_brier = "#ff7f0e"

    # Left axis: ECE
    valid_ece = ~np.isnan(ece_arr)
    if valid_ece.sum() > 0:
        ax.plot(
            log2_ranks[valid_ece],
            ece_arr[valid_ece],
            color=color_ece,
            linewidth=1.4,
            linestyle="-",
            marker="o",
            markersize=5,
            label="ECE",
            zorder=3,
        )
    ax.set_ylabel("ECE", fontsize=8, color=color_ece)
    ax.tick_params(axis="y", labelcolor=color_ece)

    # Right axis: Brier
    ax2 = ax.twinx()
    valid_brier = ~np.isnan(brier_arr)
    if valid_brier.sum() > 0:
        ax2.plot(
            log2_ranks[valid_brier],
            brier_arr[valid_brier],
            color=color_brier,
            linewidth=1.4,
            linestyle="--",
            marker="^",
            markersize=5,
            label="Brier",
            zorder=3,
        )
    ax2.set_ylabel("Brier score", fontsize=8, color=color_brier)
    ax2.tick_params(axis="y", labelcolor=color_brier)
    # Keep right spine visible for twin axis
    ax2.spines["right"].set_visible(True)

    # Annotate ECE minimum
    opt_log2: float | None = None
    if valid_ece.sum() > 0:
        idx_min = int(np.nanargmin(ece_arr))
        opt_log2 = float(log2_ranks[idx_min])
        ax.axvline(
            opt_log2,
            color=color_ece,
            linewidth=0.9,
            linestyle=":",
            alpha=0.75,
            zorder=2,
        )
        ax.text(
            opt_log2 + 0.05,
            ax.get_ylim()[1],
            rf"$r^*$={rank_values[idx_min]}",
            fontsize=7,
            color=color_ece,
            va="top",
            ha="left",
        )

    # Combined legend
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=7, loc="upper right")

    ax.set_title("(a) Calibration Error", fontsize=9, pad=4)
    _apply_rank_xticks(ax, log2_ranks, rank_values)

    return opt_log2


def _panel_coverage(
    ax: plt.Axes,
    log2_ranks: np.ndarray,
    coverage_series: dict[float, np.ndarray],
    rank_values: list[int],
) -> float | None:
    """Render panel (b): Coverage deficit for 4 nominal levels vs log₂ rank.

    Args:
        ax: Axes to render into.
        log2_ranks: log₂-transformed rank values.
        coverage_series: Dict {nominal_level: deficit_array}.
        rank_values: Raw integer rank values for x-tick labels.

    Returns:
        log₂(rank) minimising |deficit| at 0.95 level, or None.
    """
    # Perfect calibration reference
    ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle=":", zorder=1)

    opt_log2: float | None = None

    for lv, marker, color in zip(_COVERAGE_LEVELS, _COVERAGE_MARKERS, _COVERAGE_COLORS):
        arr = coverage_series[lv]
        valid = ~np.isnan(arr)
        if valid.sum() == 0:
            continue
        ax.plot(
            log2_ranks[valid],
            arr[valid],
            color=color,
            linewidth=1.2,
            linestyle="-",
            marker=marker,
            markersize=5,
            label=f"{int(lv * 100)}%",
            zorder=3,
        )
        if lv == 0.95 and valid.sum() > 0:
            idx_min = int(np.argmin(np.abs(arr[valid])))
            # Map back to full index
            full_indices = np.where(valid)[0]
            full_idx = full_indices[idx_min]
            opt_log2 = float(log2_ranks[full_idx])
            ax.axvline(
                opt_log2,
                color=color,
                linewidth=0.9,
                linestyle=":",
                alpha=0.75,
                zorder=2,
            )
            ax.text(
                opt_log2 + 0.05,
                ax.get_ylim()[1] if ax.get_ylim()[1] > -0.5 else 0.0,
                rf"$r^*$={rank_values[full_idx]}",
                fontsize=7,
                color=color,
                va="top",
                ha="left",
            )

    ax.set_ylabel("Coverage deficit", fontsize=8)
    ax.legend(title="Nominal", fontsize=7, title_fontsize=7, loc="best")
    ax.set_title("(b) Coverage Deficit", fontsize=9, pad=4)
    _apply_rank_xticks(ax, log2_ranks, rank_values)

    return opt_log2


def _panel_bias(
    ax: plt.Axes,
    log2_ranks: np.ndarray,
    bias_series: dict[str, np.ndarray],
    rank_values: list[int],
) -> float | None:
    """Render panel (c): Bias dominance taxonomy fractions vs log₂ rank.

    Args:
        ax: Axes to render into.
        log2_ranks: log₂-transformed rank values.
        bias_series: Dict {series_key: fraction_array}.
        rank_values: Raw integer rank values for x-tick labels.

    Returns:
        log₂(rank) at minimum k*=1 fraction, or None.
    """
    ax.axhline(0.5, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)

    opt_log2: float | None = None

    for key, color in _BIAS_COLORS.items():
        arr = bias_series[key]
        valid = ~np.isnan(arr)
        if valid.sum() == 0:
            continue
        ax.plot(
            log2_ranks[valid],
            arr[valid],
            color=color,
            linewidth=1.2,
            linestyle="-",
            marker="o",
            markersize=4,
            label=_BIAS_LABELS[key],
            zorder=3,
        )
        if key == "pct_scans_k_star_eq_1" and valid.sum() > 0:
            idx_min = int(np.nanargmin(arr[valid]))
            full_indices = np.where(valid)[0]
            full_idx = full_indices[idx_min]
            opt_log2 = float(log2_ranks[full_idx])

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of scans", fontsize=8)
    ax.legend(fontsize=6, loc="best")
    ax.set_title("(c) Bias Dominance", fontsize=9, pad=4)
    _apply_rank_xticks(ax, log2_ranks, rank_values)

    return opt_log2


def _panel_icc(
    ax: plt.Axes,
    log2_ranks: np.ndarray,
    icc_series: dict[str, np.ndarray],
    rank_values: list[int],
) -> float | None:
    """Render panel (d): Inter-member ICC per region vs log₂ rank.

    Annotates the 'flattening rank': first rank where
    |ICC(r) - ICC(r_max)| < _ICC_FLATTEN_TOL for each region.

    Args:
        ax: Axes to render into.
        log2_ranks: log₂-transformed rank values.
        icc_series: Dict {icc_key: ICC_array}.
        rank_values: Raw integer rank values for x-tick labels.

    Returns:
        log₂(rank) of the earliest per-region flattening rank (median of
        valid flatten ranks), or None.
    """
    flatten_log2_vals: list[float] = []

    for icc_key, region_key in _ICC_KEYS.items():
        arr = icc_series[icc_key]
        valid = ~np.isnan(arr)
        if valid.sum() == 0:
            continue

        color = LABEL_COLORS[region_key]
        label = REGION_DISPLAY.get(region_key, region_key.upper())

        ax.plot(
            log2_ranks[valid],
            arr[valid],
            color=color,
            linewidth=1.2,
            linestyle="-",
            marker="o",
            markersize=4,
            label=label,
            zorder=3,
        )

        # Detect flattening: first rank where |ICC - ICC_max| < tol
        arr_valid = arr[valid]
        icc_max = float(np.nanmax(arr_valid))
        full_indices = np.where(valid)[0]
        for j, val in enumerate(arr_valid):
            if abs(val - icc_max) < _ICC_FLATTEN_TOL:
                flatten_log2_vals.append(float(log2_ranks[full_indices[j]]))
                ax.axvline(
                    log2_ranks[full_indices[j]],
                    color=color,
                    linewidth=0.7,
                    linestyle=":",
                    alpha=0.55,
                    zorder=2,
                )
                break

    opt_log2: float | None = float(np.median(flatten_log2_vals)) if flatten_log2_vals else None

    ax.set_ylabel("ICC", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(d) Inter-member Agreement (ICC)", fontsize=9, pad=4)
    _apply_rank_xticks(ax, log2_ranks, rank_values)

    return opt_log2


# ---------------------------------------------------------------------------
# Shared x-axis helper
# ---------------------------------------------------------------------------


def _apply_rank_xticks(
    ax: plt.Axes,
    log2_ranks: np.ndarray,
    rank_values: list[int],
) -> None:
    """Set x-ticks at log₂(rank) positions with rank integer labels.

    Args:
        ax: Target axes.
        log2_ranks: log₂-transformed rank positions.
        rank_values: Integer rank values for tick labels.
    """
    ax.set_xticks(log2_ranks)
    ax.set_xticklabels([str(r) for r in rank_values], fontsize=7)
    ax.set_xlabel(r"$\log_2(\mathrm{rank})$", fontsize=8)
    ax.tick_params(axis="both", which="both", length=3)


# ---------------------------------------------------------------------------
# Cross-panel consensus band
# ---------------------------------------------------------------------------


def _add_consensus_band(
    axes: list[plt.Axes],
    opt_log2_per_panel: list[float | None],
    log2_ranks: np.ndarray,
    rank_values: list[int],
) -> None:
    """Draw a dashed vertical band at the median consensus rank.

    The consensus rank is defined as the median (in log₂ space) of the four
    per-panel optimal log₂ ranks.  Only panels with a valid optimum contribute.

    Args:
        axes: All 4 panel axes.
        opt_log2_per_panel: Per-panel optimal log₂(rank), None if unavailable.
        log2_ranks: Array of valid log₂ rank positions.
        rank_values: Integer rank values aligned to log2_ranks.
    """
    valid_opts = [v for v in opt_log2_per_panel if v is not None]
    if not valid_opts:
        return

    consensus_log2 = float(np.median(valid_opts))

    # Snap to nearest observed rank for interpretability
    nearest_idx = int(np.argmin(np.abs(log2_ranks - consensus_log2)))
    consensus_log2_snapped = float(log2_ranks[nearest_idx])
    consensus_rank = rank_values[nearest_idx]

    for ax in axes:
        ax.axvspan(
            consensus_log2_snapped - 0.08,
            consensus_log2_snapped + 0.08,
            color="#888888",
            alpha=0.12,
            zorder=0,
        )
        ax.axvline(
            consensus_log2_snapped,
            color="#555555",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            zorder=1,
        )

    # Label on the first (top-left) panel only to avoid clutter
    if axes:
        axes[0].text(
            consensus_log2_snapped + 0.10,
            axes[0].get_ylim()[1],
            rf"$r^*_{{\mathrm{{consensus}}}}$={consensus_rank}",
            fontsize=7,
            color="#555555",
            va="top",
            ha="left",
        )

    logger.debug(
        "Consensus rank: r=%d (log2=%.2f) from panel optima=%s",
        consensus_rank,
        consensus_log2_snapped,
        valid_opts,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot(data: InterLoraData, config: dict[str, Any]) -> Figure | None:
    """Render the 2×2 Calibration and Epistemic Diagnostics vs Rank figure.

    Layout (all panels share the log₂(rank) x-axis):
      (a) ECE (left y, solid circles) + Brier score (right y, dashed triangles)
      (b) Coverage deficit for nominal levels {0.50, 0.80, 0.90, 0.95}
      (c) Bias dominance taxonomy fractions (k*=1, k*>M, degenerate)
      (d) Inter-member ICC for TC, WT, ET with flattening annotation
    A dashed consensus band marks the median optimal rank across all panels.

    Args:
        data: Aggregated inter-rank data (io_layer.InterLoraData).
        config: Dict from the orchestrator config, section
            ``quant2_calib_epistemic`` (or the top-level config).
            Recognised keys:
            - ``figsize``: [width_inch, height_inch] override.

    Returns:
        matplotlib Figure, or None if fewer than 2 non-baseline ranks are
        available (degenerate: cannot draw curves).
    """
    setup_inter_lora_style(config.get("style"))

    rank_values: list[int] = data.rank_values
    if len(rank_values) < 2:
        logger.warning(
            "quant2_calib_epistemic: only %d non-baseline rank(s) found; "
            "need >= 2 to draw curves. Skipping.",
            len(rank_values),
        )
        return None

    default_w = DOUBLE_COL_MM * MM_TO_INCH
    default_h = 140.0 * MM_TO_INCH
    figsize: tuple[float, float] = tuple(config.get("figsize", [default_w, default_h]))  # type: ignore[assignment]

    log2_ranks = np.array([np.log2(r) for r in rank_values], dtype=float)

    # Extract all series
    _, ece_arr, brier_arr = _extract_calibration_series(data)
    coverage_series = _extract_coverage_series(data)
    bias_series = _extract_bias_series(data)
    icc_series = _extract_icc_series(data)

    fig, axes_2d = plt.subplots(
        2,
        2,
        figsize=figsize,
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c, ax_d = (
        axes_2d[0, 0],
        axes_2d[0, 1],
        axes_2d[1, 0],
        axes_2d[1, 1],
    )
    all_axes: list[plt.Axes] = [ax_a, ax_b, ax_c, ax_d]

    opt_a = _panel_calibration(ax_a, log2_ranks, ece_arr, brier_arr, rank_values)
    opt_b = _panel_coverage(ax_b, log2_ranks, coverage_series, rank_values)
    opt_c = _panel_bias(ax_c, log2_ranks, bias_series, rank_values)
    opt_d = _panel_icc(ax_d, log2_ranks, icc_series, rank_values)

    _add_consensus_band(all_axes, [opt_a, opt_b, opt_c, opt_d], log2_ranks, rank_values)

    logger.info(
        "quant2_calib_epistemic: rendered %d ranks, panel optima: "
        "ECE_min=r%s, cov95_min=r%s, bias_min=r%s, icc_flat=r%s",
        len(rank_values),
        _log2_to_rank_str(opt_a, rank_values, log2_ranks),
        _log2_to_rank_str(opt_b, rank_values, log2_ranks),
        _log2_to_rank_str(opt_c, rank_values, log2_ranks),
        _log2_to_rank_str(opt_d, rank_values, log2_ranks),
    )
    return fig


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _log2_to_rank_str(
    log2_val: float | None,
    rank_values: list[int],
    log2_ranks: np.ndarray,
) -> str:
    """Convert log₂(rank) back to nearest rank label, or '?' if None.

    Args:
        log2_val: log₂-transformed rank value, or None.
        rank_values: Integer rank values.
        log2_ranks: log₂-transformed rank array.

    Returns:
        String rank label.
    """
    if log2_val is None:
        return "?"
    idx = int(np.argmin(np.abs(log2_ranks - log2_val)))
    return str(rank_values[idx])
