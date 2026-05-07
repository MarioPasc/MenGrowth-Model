"""3-D surface + 2-D slices visualisation of the τ-shift σ²_v sweep.

The primary sweep bootstraps the empirical post-QC log σ²_v vector and adds
a global log-space shift τ:

.. math::
   \\sigma^2_{v,k}(\\tau) = \\exp\\!\\bigl(L_k + \\tau\\bigr),
   \\quad L_k \\sim \\widehat{F}_{\\log\\sigma^2_v}^{\\text{post-QC}}.

Top panel: 3-D surface of the implied density $p(\\log\\sigma^2_v\\mid\\tau)$
estimated via Gaussian KDE on the empirical log σ²_v shifted by τ.

Bottom panel: four highlighted slices replotted as 2-D PDFs, with the actual
post-QC empirical histogram overlaid on the τ=0 cell. Slices: τ_min
(saturated confident), τ=0 (empirical match), τ at the maximum of the
sweep grid (saturated uncertain), and a balanced intermediate.

Output
------
``{output_dir}/data/figures/tau_sweep_surface.{png,pdf}``
``{output_dir}/data/tau_sweep_surface_data.npz``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from experiments.stage1_volumetric.engine.data import load_config
from experiments.stage1_volumetric.main_experiment.analyses.plotting import style
from experiments.stage1_volumetric.main_experiment.modules.cohort import load_cohort
from experiments.stage1_volumetric.main_experiment.modules.sigma_v_generators import (
    compute_tau_endpoints,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Density: KDE shifted by τ
# ---------------------------------------------------------------------------


def density_log_sigma_sq(log_x: np.ndarray, tau: float, kde: gaussian_kde) -> np.ndarray:
    """Evaluate the empirical KDE at log_x − τ (shift in log-space)."""
    return kde(log_x - tau)


def build_surface(
    log_emp: np.ndarray,
    tau_grid: np.ndarray,
    *,
    n_x: int = 201,
    x_pad: float = 2.0,
) -> dict[str, np.ndarray]:
    """Compute the (τ × log σ²_v) density grid via shifted KDE."""
    kde = gaussian_kde(log_emp)
    x_lo = float(log_emp.min() + tau_grid.min() - x_pad)
    x_hi = float(log_emp.max() + tau_grid.max() + x_pad)
    log_x = np.linspace(x_lo, x_hi, n_x)
    Z = np.stack([density_log_sigma_sq(log_x, tau, kde) for tau in tau_grid], axis=0)
    TAU, LX = np.meshgrid(tau_grid, log_x, indexing="ij")
    return {
        "tau_grid": tau_grid,
        "log_sigma_sq_grid": log_x,
        "TAU": TAU,
        "LOG_SIGMA_SQ": LX,
        "DENSITY": Z,
        "kde": kde,
    }


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------


def select_slices(tau_grid: np.ndarray) -> dict[str, float]:
    """Numeric τ values for the four highlighted cross-sections."""
    tau_min = float(tau_grid.min())
    tau_max = float(tau_grid.max())
    # Empirical = exact zero; if grid contains 0, use it; else nearest grid point.
    if np.any(np.isclose(tau_grid, 0.0)):
        tau_emp = 0.0
    else:
        tau_emp = float(tau_grid[int(np.argmin(np.abs(tau_grid)))])
    # Balanced = midpoint between τ=0 and τ_max (the "winning" intermediate).
    tau_balanced = float((tau_emp + tau_max) / 2.0)
    return {
        "pi_min": tau_min,  # reuse style keys: pi_min ↔ saturated low
        "pi_empirical": tau_emp,
        "pi_balanced": tau_balanced,
        "pi_max": tau_max,  # pi_max ↔ saturated high
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_surface(ax, surface: dict[str, np.ndarray]) -> None:
    TAU, LX, Z = surface["TAU"], surface["LOG_SIGMA_SQ"], surface["DENSITY"]
    ax.plot_surface(
        LX,
        TAU,
        Z,
        cmap=style.SURFACE_CMAP,
        alpha=style.SURFACE_ALPHA,
        rstride=style.SURFACE_RSTRIDE,
        cstride=style.SURFACE_CSTRIDE,
        linewidth=style.SURFACE_LINEWIDTH,
        antialiased=style.SURFACE_ANTIALIASED,
    )
    ax.set_xlabel(r"$\log \sigma^2_v$")
    ax.set_ylabel(r"$\tau$ (log-space shift)")
    ax.set_zlabel(r"$p(\log \sigma^2_v\mid \tau)$")
    ax.set_title(
        "Density surface — empirical KDE shifted by τ",
        fontsize=style.SECTION_TITLE_SIZE,
    )
    style.style_3d_axes(ax)


_SLICE_SUBLABELS = {
    "pi_min": "saturated confident (τ_min)",
    "pi_empirical": "empirical (τ=0)",
    "pi_balanced": "balanced (τ_mid)",
    "pi_max": "saturated uncertain (τ_max)",
}


def _slice_label(key: str, tau_value: float) -> str:
    return rf"$\tau={tau_value:+.2f}$ — {_SLICE_SUBLABELS[key]}"


def _overlay_slice_lines(
    ax,
    surface: dict[str, np.ndarray],
    slices: dict[str, float],
    kde: gaussian_kde,
) -> None:
    log_x = surface["log_sigma_sq_grid"]
    z_max = float(surface["DENSITY"].max())
    eps = 0.005 * z_max
    for key in style.SLICE_ORDER:
        tau_val = slices[key]
        z_curve = density_log_sigma_sq(log_x, tau_val, kde)
        ax.plot(
            log_x,
            np.full_like(log_x, tau_val),
            z_curve + eps,
            color=style.SLICE_COLORS[key],
            linewidth=style.SLICE_LINE_WIDTH,
            alpha=style.SLICE_LINE_ALPHA,
            label=_slice_label(key, tau_val),
            zorder=5,
        )


def _plot_slice_panel(
    ax,
    surface: dict[str, np.ndarray],
    slices: dict[str, float],
    kde: gaussian_kde,
    empirical_log_sigma_sq: np.ndarray,
    empirical_n_label: str,
    floor_log: float,
    ceil_log: float | None,
) -> tuple[list, list]:
    log_x = surface["log_sigma_sq_grid"]
    handles: list = []
    labels: list = []

    for key in style.SLICE_ORDER:
        tau_val = slices[key]
        pdf = density_log_sigma_sq(log_x, tau_val, kde)
        line = ax.plot(
            log_x,
            pdf,
            color=style.SLICE_COLORS[key],
            linewidth=style.SLICE_LINE_WIDTH,
            alpha=style.SLICE_LINE_ALPHA,
        )[0]
        handles.append(line)
        labels.append(_slice_label(key, tau_val))

    # Histogram of the actual training/testing distribution
    in_view = (empirical_log_sigma_sq >= log_x[0]) & (empirical_log_sigma_sq <= log_x[-1])
    n_visible = int(in_view.sum())
    n_outliers = int((~in_view).sum())
    hist_label = f"Empirical σ²_v ({empirical_n_label}, shown n={n_visible})"
    _, _, patches = ax.hist(
        empirical_log_sigma_sq[in_view],
        bins=style.HIST_BINS,
        density=True,
        range=(float(log_x[0]), float(log_x[-1])),
        color=style.HIST_COLOR,
        alpha=style.HIST_ALPHA,
        edgecolor=style.HIST_EDGE_COLOR,
    )
    handles.append(patches[0])
    labels.append(hist_label)

    # The post-QC vector is floor-clipped at floor_variance, so the histogram
    # has a delta-spike at log(floor). Cap the y-axis so the smooth KDE
    # slices remain readable; annotate the truncation.
    slice_max = 0.0
    for key in style.SLICE_ORDER:
        slice_max = max(slice_max, float(density_log_sigma_sq(log_x, slices[key], kde).max()))
    ax.set_ylim(0.0, slice_max * 1.25)

    # Floor / ceiling guides
    floor_line = ax.axvline(
        floor_log,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        label=r"$\log\sigma^2_v$ floor",
    )
    handles.append(floor_line)
    labels.append(r"$\log$ σ²_v floor (= log floor_variance)")
    if ceil_log is not None:
        ceil_line = ax.axvline(
            ceil_log,
            color="#444444",
            linestyle=":",
            linewidth=1.0,
        )
        handles.append(ceil_line)
        labels.append(r"$\log$ σ²_v ceiling")

    ax.set_xlabel(r"$\log \sigma^2_v$")
    ax.set_ylabel(r"density")
    ax.set_title(
        "Slice cross-sections at highlighted τ values",
        fontsize=style.SECTION_TITLE_SIZE,
    )
    style.style_2d_axes(ax)

    info_lines = [
        rf"$\tau_{{\min}}={slices['pi_min']:+.2f},\ \tau_{{\max}}={slices['pi_max']:+.2f}$",
        f"empirical n={empirical_log_sigma_sq.size}, shown {n_visible}",
    ]
    if n_outliers:
        info_lines.append(f"out-of-view: {n_outliers}")
    ax.text(
        0.02,
        0.97,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=style.TICK_SIZE,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    return handles, labels


def make_figure(
    log_emp: np.ndarray,
    empirical_sigma_v_sq: np.ndarray,
    tau_grid: np.ndarray,
    *,
    floor: float,
    ceil: float | None,
    n_x: int = 201,
    empirical_n_label: str = "n=?",
) -> tuple[plt.Figure, dict[str, np.ndarray], dict[str, float]]:
    style.apply_global_style()
    surface = build_surface(log_emp, tau_grid, n_x=n_x)
    slices = select_slices(tau_grid)
    kde = surface["kde"]

    fig = plt.figure(figsize=style.FIG_SIZE_INCHES)
    gs = GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[1.05, 1.0],
        wspace=0.18,
        figure=fig,
    )

    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    _plot_surface(ax_3d, surface)
    _overlay_slice_lines(ax_3d, surface, slices, kde)
    ax_3d.view_init(elev=22, azim=-58)

    ax_2d = fig.add_subplot(gs[0, 1])
    log_emp_for_hist = np.log(np.maximum(empirical_sigma_v_sq, 1e-15))
    floor_log = float(np.log(floor))
    ceil_log = float(np.log(ceil)) if ceil is not None else None
    handles, labels = _plot_slice_panel(
        ax_2d,
        surface,
        slices,
        kde,
        log_emp_for_hist,
        empirical_n_label=empirical_n_label,
        floor_log=floor_log,
        ceil_log=ceil_log,
    )

    fig.suptitle(
        "σ²_v sweep — empirical bootstrap shifted by τ",
        fontsize=style.TITLE_SIZE,
        y=0.985,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=False,
        fontsize=style.LEGEND_SIZE,
    )
    return fig, surface, slices


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_artifacts(
    fig: plt.Figure,
    surface: dict[str, np.ndarray],
    slices: dict[str, float],
    empirical_sigma_v_sq: np.ndarray,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    png_path = figures_dir / "tau_sweep_surface.png"
    pdf_path = figures_dir / "tau_sweep_surface.pdf"
    fig.savefig(png_path, dpi=style.FIG_DPI)
    fig.savefig(pdf_path, dpi=style.FIG_DPI)

    npz_path = output_dir / "tau_sweep_surface_data.npz"
    slice_keys = list(style.SLICE_ORDER)
    np.savez_compressed(
        npz_path,
        tau_grid=surface["tau_grid"],
        log_sigma_sq_grid=surface["log_sigma_sq_grid"],
        density=surface["DENSITY"],
        slice_keys=np.asarray(slice_keys),
        slice_tau_values=np.asarray([slices[k] for k in slice_keys]),
        empirical_sigma_v_sq=np.asarray(empirical_sigma_v_sq, dtype=np.float64),
    )

    metadata_path = output_dir / "tau_sweep_surface_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "slices": slices,
                "slice_keys_ordered": slice_keys,
                "n_tau": int(surface["tau_grid"].size),
                "n_log_sigma_sq": int(surface["log_sigma_sq_grid"].size),
                "tau_min": float(surface["tau_grid"].min()),
                "tau_max": float(surface["tau_grid"].max()),
            },
            f,
            indent=2,
        )

    logger.info("Wrote figure: %s", png_path)
    logger.info("Wrote figure: %s", pdf_path)
    logger.info("Wrote data:   %s", npz_path)
    logger.info("Wrote meta:   %s", metadata_path)
    return {
        "png": png_path,
        "pdf": pdf_path,
        "npz": npz_path,
        "metadata": metadata_path,
    }


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
    output_dir = args.output_dir or (output_root / "data")

    log_emp = np.log(np.maximum(cohort.empirical_sigma_v_sq_flat, 1e-15))

    sweep_primary = cfg["sweep"]["primary"]
    sat = sweep_primary.get("saturation", {})
    floor = float(
        sat.get("sigma_v_sq_floor", cfg.get("uncertainty", {}).get("floor_variance", 1e-3))
    )
    ceil = sat.get("sigma_v_sq_ceil", None)
    safety = float(sat.get("safety_margin", 2.0))

    if "tau_grid" in sweep_primary and sweep_primary["tau_grid"] is not None:
        tau_grid = np.asarray(
            sorted(set(float(t) for t in sweep_primary["tau_grid"])), dtype=np.float64
        )
    else:
        tau_min, tau_max = compute_tau_endpoints(
            log_emp,
            sigma_v_sq_floor=floor,
            sigma_v_sq_ceil=float(ceil) if ceil is not None else 50.0,
            safety_margin=safety,
        )
        n_tau = int(sweep_primary.get("n_tau", 9))
        tau_grid = np.linspace(tau_min, tau_max, n_tau)
        if not np.any(np.isclose(tau_grid, 0.0)):
            idx = int(np.argmin(np.abs(tau_grid)))
            tau_grid[idx] = 0.0
            tau_grid = np.sort(tau_grid)

    fig, surface, slices = make_figure(
        log_emp=log_emp,
        empirical_sigma_v_sq=cohort.empirical_sigma_v_sq_flat,
        tau_grid=tau_grid,
        floor=floor,
        ceil=float(ceil) if ceil is not None else None,
        n_x=args.n_x,
        empirical_n_label=f"post-QC, n={cohort.empirical_sigma_v_sq_flat.size}",
    )
    save_artifacts(
        fig=fig,
        surface=surface,
        slices=slices,
        empirical_sigma_v_sq=cohort.empirical_sigma_v_sq_flat,
        output_dir=output_dir,
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
