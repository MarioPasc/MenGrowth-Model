"""Figures for the main experiment.

All figures take the long-form aggregated table as input. They focus on
calibration trends along the sweep axis (π for the primary, α for the
ablation) with the empirical pass-through marked as a separate point.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TERTILES = ("low", "mid", "high")


def _setup_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)


def _summary_by_level(
    df: pd.DataFrame,
    *,
    metric: str,
    scope: str,
    tertile: str,
    family: str = "empirical_shift",
) -> pd.DataFrame:
    sub = df[
        (df["family"] == family)
        & (df["scope"] == scope)
        & (df["tertile"] == tertile)
        & (df["metric"] == metric)
    ]
    if sub.empty:
        return pd.DataFrame(columns=["level_value", "median", "q25", "q75"])
    g = (
        sub.groupby("level_value")["value"]
        .agg(median="median", q25=lambda v: v.quantile(0.25), q75=lambda v: v.quantile(0.75))
        .reset_index()
        .sort_values("level_value")
    )
    return g


def figure_metric_vs_tau(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    *,
    title: str | None = None,
    nominal: float | None = None,
) -> None:
    """One panel per tertile (+ marginal); median ± IQR across seeds vs τ."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    panels = [("marginal", "all"), ("tertile", "low"), ("tertile", "mid"), ("tertile", "high")]
    for ax, (scope, tertile) in zip(axes, panels, strict=True):
        g = _summary_by_level(df, metric=metric, scope=scope, tertile=tertile)
        if not g.empty:
            ax.plot(
                g["level_value"], g["median"], marker="o", color="C0", label="LMEHetero@injected"
            )
            ax.fill_between(g["level_value"], g["q25"], g["q75"], color="C0", alpha=0.2)

        # τ=0 (empirical) cell — highlight as a reference line.
        emp = df[
            (df["family"] == "empirical_shift")
            & (df["scope"] == scope)
            & (df["tertile"] == tertile)
            & (df["metric"] == metric)
            & (np.isclose(df["level_value"], 0.0))
        ]["value"]
        if not emp.empty:
            ax.axhline(
                emp.median(),
                color="C1",
                linestyle="--",
                alpha=0.7,
                label=r"$\tau=0$ (empirical)",
            )

        # Baselines
        for base, ls, color in (
            ("LME_baseline", ":", "C2"),
            ("LMEHetero_Zero_baseline", "-.", "C3"),
        ):
            b = df[
                (df["family"] == "baseline")
                & (df["level"] == base)
                & (df["scope"] == scope)
                & (df["tertile"] == tertile)
                & (df["metric"] == metric)
            ]["value"]
            if not b.empty:
                ax.axhline(b.iloc[0], color=color, linestyle=ls, alpha=0.7, label=base)

        if nominal is not None:
            ax.axhline(nominal, color="black", linestyle="-", alpha=0.4, lw=0.6)

        sub_title = "marginal" if scope == "marginal" else f"σ²_v {tertile}"
        _setup_axes(ax, r"$\tau$ (log-space σ²_v shift)", metric, sub_title)

    axes[0].legend(fontsize=7, loc="best")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def figure_sharpness_panel(df: pd.DataFrame, output_path: Path) -> None:
    figure_metric_vs_tau(
        df,
        metric="ci_width_mean",
        output_path=output_path,
        title=r"Mean 95% CI width (sharpness) along the $\tau$-sweep",
    )


def figure_pit_grid(output_root: Path, output_path: Path, n_bins: int = 10) -> None:
    """Per-τ PIT histogram grid using one representative seed per cell."""
    runs_dir = output_root / "runs"
    if not runs_dir.exists():
        return

    cells = []
    for cell_dir in sorted(runs_dir.iterdir()):
        if not cell_dir.name.startswith("empirical_shift_"):
            continue
        # Use seed 0 for visualization
        seed_dir = cell_dir / "seed_000"
        marginal_path = seed_dir / "marginal_metrics.json"
        if not marginal_path.exists():
            continue
        cells.append((cell_dir.name, marginal_path))

    if not cells:
        return

    n = len(cells)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), squeeze=False)

    import json as _json

    for ax, (cell_name, mpath) in zip(axes.ravel(), cells, strict=False):
        with open(mpath) as f:
            data = _json.load(f)
        pit = np.asarray(data.get("pit_values", []), dtype=np.float64)
        if pit.size == 0:
            ax.set_axis_off()
            continue
        ax.hist(pit, bins=n_bins, range=(0, 1), color="C0", edgecolor="white")
        ax.axhline(len(pit) / n_bins, color="black", linestyle="--", lw=0.8, alpha=0.6)
        ks_p = data.get("pit_ks_p", float("nan"))
        ax.set_title(f"{cell_name}\nKS p={ks_p:.2f}", fontsize=9)
        ax.set_xlabel("PIT", fontsize=8)
        ax.set_ylabel("count", fontsize=8)

    for ax in axes.ravel()[len(cells) :]:
        ax.set_axis_off()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def make_all_figures(df: pd.DataFrame, output_root: Path, cfg: dict) -> None:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep = cfg.get("reporting", {}).get("figures", {})

    if rep.get("cov_vs_tau", True):
        figure_metric_vs_tau(
            df,
            metric="cov_95",
            output_path=fig_dir / "cov_vs_tau.png",
            title="Coverage @ 95% along τ-sweep",
            nominal=0.95,
        )
    if rep.get("is_vs_tau", True):
        figure_metric_vs_tau(
            df,
            metric="is_95",
            output_path=fig_dir / "is_vs_tau.png",
            title="Interval Score @ 95% along τ-sweep",
        )
    if rep.get("sharpness_panel", True):
        figure_sharpness_panel(df, output_path=fig_dir / "sharpness_panel.png")
    if rep.get("pit_grid", True):
        figure_pit_grid(output_root, output_path=fig_dir / "pit_grid.png")
