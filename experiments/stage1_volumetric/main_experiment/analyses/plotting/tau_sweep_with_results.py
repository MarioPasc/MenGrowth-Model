"""Composite figure: τ-sweep input distributions (3-D surface) + IS@95 results.

Left panel (re-uses ``tau_sweep_surface.build_surface`` /
``_plot_surface`` / ``_overlay_slice_lines``): 3-D density surface of the
empirical-bootstrap σ²_v shifted by τ, with the four highlighted slices
(saturated-low, empirical, balanced, saturated-high) overlaid as 3-D lines
in the same colours used elsewhere in the package.

Right panel: per-cell IS@95 plotted against the per-cell median
$\\log\\sigma^2_v$ on the x-axis. Each (τ, seed) cell is one scatter point
coloured by τ (viridis for non-highlighted τ; SLICE_COLORS for the four
highlighted slices). Per-τ medians are shown with vertical error bars whose
half-width is the median across-seed BCa 95 % CI of ΔIS@95 (LME homo →
LMEHetero@injected) anchored on the LME-homo IS@95. The LME-homo IS@95 is
drawn as a horizontal black dashed line (zero-effect reference).

A τ-cluster is annotated with ``✱`` when ≥ 50 % of its 20 bootstrap seeds
reject ΔIS = 0 under BH-FDR (q = 0.05).

Usage
-----
::

    python -m experiments.stage1_volumetric.main_experiment.analyses.plotting.tau_sweep_with_results \\
        --config experiments/stage1_volumetric/main_experiment/configs/local.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from experiments.stage1_volumetric.engine.data import load_config
from experiments.stage1_volumetric.main_experiment.analyses.plotting import style
from experiments.stage1_volumetric.main_experiment.analyses.plotting.tau_sweep_surface import (
    _overlay_slice_lines,
    _plot_surface,
    build_surface,
    select_slices,
)
from experiments.stage1_volumetric.main_experiment.modules.cohort import load_cohort
from experiments.stage1_volumetric.main_experiment.modules.sigma_v_generators import (
    build_tau_grid,
    compute_tau_endpoints,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------


def collect_cell_results(output_root: Path) -> list[dict]:
    """Walk ``runs/`` and return one dict per (τ, seed) cell."""
    rows: list[dict] = []
    runs_dir = output_root / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs/ directory missing: {runs_dir}")

    prefix = "empirical_shift_tau_"
    for cell_dir in sorted(runs_dir.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith(prefix):
            continue
        try:
            tau = float(cell_dir.name[len(prefix) :])
        except ValueError:
            continue
        for seed_dir in sorted(cell_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.split("_")[1])
            sv2_path = seed_dir / "sigma_v_sq_injected.npy"
            mm_path = seed_dir / "marginal_metrics.json"
            if not (sv2_path.exists() and mm_path.exists()):
                continue
            sv2 = np.load(sv2_path)
            log_sv2 = np.log(np.maximum(sv2, 1e-15))
            with open(mm_path) as f:
                m = json.load(f)
            rows.append(
                {
                    "tau": tau,
                    "seed": seed,
                    "log_sv2_median": float(np.median(log_sv2)),
                    "log_sv2_q25": float(np.quantile(log_sv2, 0.25)),
                    "log_sv2_q75": float(np.quantile(log_sv2, 0.75)),
                    "is_95": float(m["is_95"]),
                    "cov_95": float(m["cov_95"]),
                }
            )
    return rows


def load_bootstrap_per_tau(output_root: Path) -> dict[float, dict[str, float]]:
    """Per-τ median ΔIS@95 BCa CI (LME → LMEHetero@injected, marginal)."""
    path = output_root / "aggregated" / "bootstrap_results.json"
    if not path.exists():
        logger.warning("bootstrap_results.json not found; CI bars will be omitted")
        return {}
    bp = json.load(open(path))["results"]
    by_tau: dict[float, list[dict]] = defaultdict(list)
    for r in bp:
        if (
            r.get("transition") == "LME__LMEHetero_Injected"
            and r.get("metric") == "is_95"
            and r.get("tertile") == "marginal"
        ):
            by_tau[float(r["level_value"])].append(r)
    out: dict[float, dict[str, float]] = {}
    for tau, items in by_tau.items():
        deltas = np.asarray([x["delta"] for x in items])
        cis_lo = np.asarray([x["ci_lower"] for x in items])
        cis_hi = np.asarray([x["ci_upper"] for x in items])
        rej = np.asarray([bool(x.get("rejected_bh", False)) for x in items])
        out[tau] = {
            "delta_med": float(np.median(deltas)),
            "ci_lo_med": float(np.median(cis_lo)),
            "ci_hi_med": float(np.median(cis_hi)),
            "rej_frac": float(rej.mean()),
            "n_seeds": int(len(items)),
        }
    return out


def load_lme_homo(output_root: Path) -> dict[str, float]:
    with open(output_root / "LME_baseline" / "marginal_metrics.json") as f:
        m = json.load(f)
    return {"is_95": float(m["is_95"]), "cov_95": float(m["cov_95"])}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _slice_color_for_tau(tau: float, slices: dict[str, float]) -> str | None:
    for key, val in slices.items():
        if np.isclose(tau, val, atol=1e-6):
            return style.SLICE_COLORS[key]
    return None


_SLICE_MARKERS: dict[str, str] = {
    "pi_min": "s",  # red square (saturated low)
    "pi_empirical": "*",  # green star  (empirical, τ=0)
    "pi_balanced": "D",  # orange diamond
    "pi_max": "^",  # blue triangle (saturated high)
}
_SLICE_MARKER_SIZE: dict[str, float] = {
    "pi_min": 11,
    "pi_empirical": 18,
    "pi_balanced": 11,
    "pi_max": 13,
}


def _plot_results_panel(
    ax,
    cells: list[dict],
    boot: dict[float, dict[str, float]],
    lme_homo: dict[str, float],
    slices: dict[str, float],
    tau_grid: np.ndarray,
    log_emp: np.ndarray,
    floor_log: float,
    ceil_log: float,
) -> tuple[list, list]:
    homo_is = lme_homo["is_95"]
    norm = plt.Normalize(vmin=float(tau_grid.min()), vmax=float(tau_grid.max()))
    cmap = style.SURFACE_CMAP

    log_emp_mean = float(np.mean(log_emp))
    log_emp_p25 = float(np.percentile(log_emp, 25))
    log_emp_p75 = float(np.percentile(log_emp, 75))

    by_tau: dict[float, list[dict]] = defaultdict(list)
    for c in cells:
        by_tau[c["tau"]].append(c)

    handles: list = []
    labels: list = []

    homo_line = ax.axhline(
        homo_is,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        zorder=3,
    )
    handles.append(homo_line)
    labels.append(rf"LME homo (IS@95 = {homo_is:.2f})")

    # Trend line through per-τ medians (drawn after points, last loop)
    line_xs: list[float] = []
    line_ys: list[float] = []

    used_highlight_keys: set[str] = set()
    for tau in sorted(by_tau):
        items = by_tau[tau]
        slice_key: str | None = None
        for k, v in slices.items():
            if np.isclose(tau, v, atol=1e-6):
                slice_key = k
                break
        slice_color = style.SLICE_COLORS[slice_key] if slice_key else None
        color = slice_color if slice_color is not None else cmap(norm(tau))
        marker = _SLICE_MARKERS[slice_key] if slice_key else "o"
        msize = _SLICE_MARKER_SIZE[slice_key] if slice_key else 8
        edgewidth = 1.3 if slice_key else 0.6

        # x-position of the τ-shifted distribution centroid (pre-clipping).
        # This separates the 9 τ-levels evenly even when the actual injected
        # values pile up against the floor/ceiling clamps.
        x_center = tau + log_emp_mean
        x_p25 = tau + log_emp_p25
        x_p75 = tau + log_emp_p75
        ys = np.asarray([c["is_95"] for c in items])
        y_med = float(np.median(ys))

        # Per-seed scatter, jittered around x_center for readability.
        rng = np.random.default_rng(abs(int(round(tau * 1000))) + 1)
        x_jitter = x_center + rng.normal(0.0, 0.10, size=len(ys))
        ax.scatter(
            x_jitter,
            ys,
            color=color,
            alpha=0.30,
            s=14,
            zorder=2,
            edgecolor="white",
            linewidth=0.3,
        )

        # Horizontal whisker = IQR of the τ-shifted distribution (pre-clip).
        ax.plot(
            [x_p25, x_p75],
            [y_med, y_med],
            color=color,
            linewidth=1.1,
            alpha=0.55,
            zorder=4,
        )

        b = boot.get(tau)
        if b is not None:
            ci_lo_abs = homo_is + b["ci_lo_med"]
            ci_hi_abs = homo_is + b["ci_hi_med"]
            yerr = [[max(y_med - ci_lo_abs, 0.0)], [max(ci_hi_abs - y_med, 0.0)]]
        else:
            yerr = None
        ax.errorbar(
            x_center,
            y_med,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=msize,
            markeredgecolor="black",
            markeredgewidth=edgewidth,
            elinewidth=1.8,
            capsize=4,
            zorder=6,
        )

        line_xs.append(x_center)
        line_ys.append(y_med)

        if slice_key is not None:
            if slice_key not in used_highlight_keys:
                used_highlight_keys.add(slice_key)
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=color,
                        marker=marker,
                        markersize=msize * 0.7,
                        linestyle="none",
                        markeredgecolor="black",
                        markeredgewidth=edgewidth,
                    )
                )
                labels.append(style.SLICE_LABELS[slice_key] + f" (τ={tau:+.2f})")

        if b is not None and b["rej_frac"] >= 0.5:
            ax.annotate(
                "✱",
                xy=(x_center, y_med),
                xytext=(10, 6),
                textcoords="offset points",
                fontsize=15,
                color=color,
                fontweight="bold",
            )

    # Trend line through per-τ medians
    if len(line_xs) >= 2:
        ax.plot(line_xs, line_ys, color="gray", linewidth=1.0, alpha=0.45, zorder=1)

    # Saturation guide shading: regions where the actual injected σ²_v gets
    # clipped to floor or ceiling. Drawn after points so they sit underneath.
    cur_xlim = ax.get_xlim()
    if floor_log > cur_xlim[0]:
        ax.axvspan(cur_xlim[0], floor_log, alpha=0.06, color="gray", zorder=0)
        ax.text(
            (cur_xlim[0] + floor_log) / 2,
            ax.get_ylim()[1] * 0.97,
            "σ²_v ≤ floor\n(saturated low)",
            ha="center",
            va="top",
            fontsize=style.TICK_SIZE - 1,
            color="#666666",
            zorder=1,
        )
    if ceil_log < cur_xlim[1]:
        ax.axvspan(ceil_log, cur_xlim[1], alpha=0.06, color="gray", zorder=0)
        ax.text(
            (ceil_log + cur_xlim[1]) / 2,
            ax.get_ylim()[1] * 0.97,
            "σ²_v ≥ ceiling\n(saturated high)",
            ha="center",
            va="top",
            fontsize=style.TICK_SIZE - 1,
            color="#666666",
            zorder=1,
        )
    ax.set_xlim(cur_xlim)

    # Generic τ-colorbar handle for non-highlighted τ
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label(r"$\tau$ (log-space σ²_v shift)", fontsize=style.LABEL_SIZE)
    cbar.ax.tick_params(labelsize=style.TICK_SIZE)

    ax.set_xlabel(
        r"$\log \sigma^2_v$  (centroid of $\tau$-shifted empirical, mean$+\tau$;"
        r" horizontal whisker = empirical IQR shifted by $\tau$)"
    )
    ax.set_ylabel("IS@95")
    ax.set_title(
        "LMEHetero@injected vs LME homo — IS@95 along the τ-sweep",
        fontsize=style.SECTION_TITLE_SIZE,
    )
    style.style_2d_axes(ax)

    sig_count = sum(1 for v in boot.values() if v.get("rej_frac", 0) >= 0.5)
    info_lines = [
        "BCa B=10000, BH q=0.05",
        f"✱ = ≥50% of 20 seeds reject ΔIS=0 ({sig_count}/{len(boot)} τ levels)",
    ]
    ax.text(
        0.02,
        0.97,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=style.TICK_SIZE,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cccccc",
        },
    )
    return handles, labels


def make_figure(
    cohort_log_emp: np.ndarray,
    tau_grid: np.ndarray,
    cells: list[dict],
    boot: dict[float, dict[str, float]],
    lme_homo: dict[str, float],
    *,
    floor_log: float,
    ceil_log: float,
    n_x: int = 201,
) -> tuple[plt.Figure, dict]:
    style.apply_global_style()

    surface = build_surface(cohort_log_emp, tau_grid, n_x=n_x)
    slices = select_slices(tau_grid)

    fig = plt.figure(figsize=(15.5, 6.5))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1.05, 1.0], wspace=0.18, figure=fig)

    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    _plot_surface(ax_3d, surface)
    _overlay_slice_lines(ax_3d, surface, slices, surface["kde"])
    ax_3d.view_init(elev=22, azim=-58)

    ax_2d = fig.add_subplot(gs[0, 1])
    handles, labels = _plot_results_panel(
        ax_2d,
        cells,
        boot,
        lme_homo,
        slices,
        tau_grid,
        log_emp=cohort_log_emp,
        floor_log=floor_log,
        ceil_log=ceil_log,
    )

    fig.suptitle(
        "σ²_v sweep — input distribution surface (left) and IS@95 outcome (right)",
        fontsize=style.TITLE_SIZE,
        y=0.985,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(len(handles), 5),
        frameon=False,
        fontsize=style.LEGEND_SIZE,
    )
    return fig, surface


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-x", type=int, default=201)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    cfg = load_config(args.config)
    cohort = load_cohort(cfg)
    output_root = Path(cfg["paths"]["output_dir"])
    figures_dir = args.output_dir or (output_root / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    log_emp = np.log(np.maximum(cohort.empirical_sigma_v_sq_flat, 1e-15))

    primary = cfg["sweep"]["primary"]
    if "tau_grid" in primary and primary["tau_grid"] is not None:
        tau_grid = np.asarray(sorted(set(float(t) for t in primary["tau_grid"])), dtype=np.float64)
    else:
        sat = primary.get("saturation", {})
        floor = float(sat.get("sigma_v_sq_floor", cfg["uncertainty"]["floor_variance"]))
        ceil = float(sat.get("sigma_v_sq_ceil", 50.0))
        safety = float(sat.get("safety_margin", 2.0))
        tau_min, tau_max = compute_tau_endpoints(
            log_emp,
            sigma_v_sq_floor=floor,
            sigma_v_sq_ceil=ceil,
            safety_margin=safety,
        )
        n_tau = int(primary.get("n_tau", 9))
        tau_grid = build_tau_grid(n_tau, tau_min, tau_max, include_zero=True)

    cells = collect_cell_results(output_root)
    boot = load_bootstrap_per_tau(output_root)
    lme_homo = load_lme_homo(output_root)
    logger.info(
        "Loaded %d cells across %d τ levels; bootstrap entries: %d τ levels",
        len(cells),
        len({c["tau"] for c in cells}),
        len(boot),
    )

    sat = primary.get("saturation", {})
    floor_v = float(sat.get("sigma_v_sq_floor", cfg["uncertainty"]["floor_variance"]))
    ceil_v = float(sat.get("sigma_v_sq_ceil", 50.0))
    fig, surface = make_figure(
        cohort_log_emp=log_emp,
        tau_grid=tau_grid,
        cells=cells,
        boot=boot,
        lme_homo=lme_homo,
        floor_log=float(np.log(floor_v)),
        ceil_log=float(np.log(ceil_v)),
        n_x=args.n_x,
    )
    png_path = figures_dir / "tau_sweep_with_results.png"
    pdf_path = figures_dir / "tau_sweep_with_results.pdf"
    fig.savefig(png_path, dpi=style.FIG_DPI)
    fig.savefig(pdf_path, dpi=style.FIG_DPI)
    plt.close(fig)
    logger.info("Wrote %s", png_path)
    logger.info("Wrote %s", pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
