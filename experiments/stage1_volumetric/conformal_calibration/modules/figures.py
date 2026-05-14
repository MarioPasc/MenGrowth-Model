"""Figures for the conformal calibration experiment.

All figures take the long-form aggregated table (from
:func:`~modules.aggregator.collect_runs`) as input.  They compare calibration
layers within a base model and compare base models against each other, with
optional stratification by σ²_v tertile.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LAYERS = ("parametric", "jackknife_plus", "cqr_norm", "cqr_proper")
_LAYER_COLORS = {
    "parametric": "C0",
    "jackknife_plus": "C1",
    "cqr_norm": "C2",
    "cqr_proper": "C3",
}
_TERTILES = ("low", "mid", "high")
_BASE_MODELS = ("lme_homo", "lme_hetero", "ensemble_bma")
_MODEL_COLORS = {
    "lme_homo": "C0",
    "lme_hetero": "C1",
    "ensemble_bma": "C2",
}


def _setup_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)


def _summary_by_model_layer(
    df: pd.DataFrame,
    *,
    metric: str,
    scope: str,
    tertile: str,
) -> pd.DataFrame:
    """Compute median + IQR per (base_model, layer)."""
    sub = df[(df["scope"] == scope) & (df["tertile"] == tertile) & (df["metric"] == metric)]
    if sub.empty:
        return pd.DataFrame(columns=["base_model", "layer", "median", "q25", "q75"])
    g = (
        sub.groupby(["base_model", "layer"])["value"]
        .agg(
            median="median",
            q25=lambda v: v.quantile(0.25),
            q75=lambda v: v.quantile(0.75),
        )
        .reset_index()
    )
    return g


# ---------------------------------------------------------------------------
# Figure 1: IS@95 by model × calibration layer (bar chart)
# ---------------------------------------------------------------------------


def figure_is_by_model_calibration(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar chart: IS@95 per (base_model, layer), marginal only.

    Args:
        df: Long-form aggregated table.
        output_path: Destination PNG path.
    """
    g = _summary_by_model_layer(df, metric="is_95", scope="marginal", tertile="all")
    if g.empty:
        logger.warning("No is_95 data; skipping figure_is_by_model_calibration")
        return

    models = [m for m in _BASE_MODELS if m in g["base_model"].values]
    layers = [l for l in _LAYERS if l in g["layer"].values]
    n_models = len(models)
    n_layers = len(layers)

    if n_models == 0 or n_layers == 0:
        return

    x = np.arange(n_models)
    width = 0.8 / max(n_layers, 1)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_layers)

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_models * n_layers), 4))
    for i, layer in enumerate(layers):
        sub = g[g["layer"] == layer]
        vals = []
        errs_lo = []
        errs_hi = []
        for model in models:
            row = sub[sub["base_model"] == model]
            if row.empty:
                vals.append(np.nan)
                errs_lo.append(0.0)
                errs_hi.append(0.0)
            else:
                med = float(row["median"].iloc[0])
                q25 = float(row["q25"].iloc[0])
                q75 = float(row["q75"].iloc[0])
                vals.append(med)
                errs_lo.append(max(med - q25, 0.0))
                errs_hi.append(max(q75 - med, 0.0))
        ax.bar(
            x + offsets[i],
            vals,
            width * 0.9,
            yerr=[errs_lo, errs_hi],
            label=layer,
            color=_LAYER_COLORS.get(layer, f"C{i}"),
            alpha=0.85,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    _setup_axes(ax, "Base model", "IS@95 (lower is better)", "IS@95 by model × calibration layer")
    ax.legend(title="Layer", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Figure 2: Coverage@95 by model × calibration layer
# ---------------------------------------------------------------------------


def figure_coverage_by_model_calibration(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar chart: coverage@95 per (base_model, layer), marginal only.

    Args:
        df: Long-form aggregated table.
        output_path: Destination PNG path.
    """
    g = _summary_by_model_layer(df, metric="coverage_95", scope="marginal", tertile="all")
    if g.empty:
        logger.warning("No coverage_95 data; skipping figure_coverage_by_model_calibration")
        return

    models = [m for m in _BASE_MODELS if m in g["base_model"].values]
    layers = [l for l in _LAYERS if l in g["layer"].values]
    n_models = len(models)
    n_layers = len(layers)

    if n_models == 0 or n_layers == 0:
        return

    x = np.arange(n_models)
    width = 0.8 / max(n_layers, 1)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_layers)

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_models * n_layers), 4))
    for i, layer in enumerate(layers):
        sub = g[g["layer"] == layer]
        vals = []
        for model in models:
            row = sub[sub["base_model"] == model]
            vals.append(float(row["median"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            x + offsets[i],
            vals,
            width * 0.9,
            label=layer,
            color=_LAYER_COLORS.get(layer, f"C{i}"),
            alpha=0.85,
        )

    ax.axhline(0.95, color="black", linestyle="--", lw=0.8, label="nominal 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    _setup_axes(ax, "Base model", "Coverage@95", "Coverage@95 by model × calibration layer")
    ax.legend(title="Layer", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Figure 3: Tertile panel — IS@95 by σ²_v tertile, layer, base model
# ---------------------------------------------------------------------------


def figure_tertile_panel(
    df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "is_95",
) -> None:
    """4-panel figure (marginal + 3 tertiles) comparing layers per base model.

    Args:
        df: Long-form aggregated table.
        output_path: Destination PNG path.
        metric: Which metric to display on the y-axis.
    """
    panels = [("marginal", "all"), ("tertile", "low"), ("tertile", "mid"), ("tertile", "high")]
    panel_titles = ["marginal", "σ²_v low", "σ²_v mid", "σ²_v high"]

    models = [m for m in _BASE_MODELS if m in df["base_model"].values]
    layers = [l for l in _LAYERS if l in df["layer"].values]
    if not models or not layers:
        return

    n_models = len(models)
    x = np.arange(n_models)
    n_layers = len(layers)
    width = 0.8 / max(n_layers, 1)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_layers)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, (scope, tertile), ptitle in zip(axes, panels, panel_titles, strict=True):
        g = _summary_by_model_layer(df, metric=metric, scope=scope, tertile=tertile)
        for i, layer in enumerate(layers):
            sub = g[g["layer"] == layer] if not g.empty else pd.DataFrame()
            vals = []
            for model in models:
                row = sub[sub["base_model"] == model] if not sub.empty else pd.DataFrame()
                vals.append(float(row["median"].iloc[0]) if not row.empty else np.nan)
            ax.bar(
                x + offsets[i],
                vals,
                width * 0.9,
                label=layer,
                color=_LAYER_COLORS.get(layer, f"C{i}"),
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        _setup_axes(ax, "Base model", metric, ptitle)

    axes[0].legend(title="Layer", fontsize=7, loc="best")
    fig.suptitle(f"{metric} by σ²_v tertile", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Figure 4: PIT grid (one histogram per base_model × seed_000 × layer)
# ---------------------------------------------------------------------------


def figure_pit_grid(output_root: Path, output_path: Path, n_bins: int = 10) -> None:
    """Per-model PIT histogram grid using seed_000.

    Args:
        output_root: Root output directory of the experiment.
        output_path: Destination PNG path.
        n_bins: Number of histogram bins.
    """
    import json as _json

    runs_dir = output_root / "runs"
    if not runs_dir.exists():
        return

    cells = []
    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        seed_dir = model_dir / "seed_000"
        marginal_path = seed_dir / "marginal_metrics.json"
        if not marginal_path.exists():
            continue
        cells.append((model_dir.name, marginal_path))

    if not cells:
        return

    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.5), squeeze=False)

    for ax, (model_name, mpath) in zip(axes[0], cells, strict=False):
        with open(mpath) as f:
            data = _json.load(f)
        # Try to find pit_values inside any layer sub-dict
        pit = np.array([], dtype=np.float64)
        for v in data.values():
            if isinstance(v, dict) and "pit_values" in v:
                pit = np.asarray(v["pit_values"], dtype=np.float64)
                break
        if "pit_values" in data:
            pit = np.asarray(data["pit_values"], dtype=np.float64)

        if pit.size == 0:
            ax.set_axis_off()
            ax.set_title(model_name, fontsize=9)
            continue

        ax.hist(pit, bins=n_bins, range=(0, 1), color="C0", edgecolor="white")
        ax.axhline(len(pit) / n_bins, color="black", linestyle="--", lw=0.8, alpha=0.6)
        ks_p = data.get("pit_ks_p", float("nan"))
        ax.set_title(f"{model_name}\nKS p={ks_p:.2f}", fontsize=9)
        ax.set_xlabel("PIT", fontsize=8)
        ax.set_ylabel("count", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Figure 5: Mean CI width vs σ²_v (scatter, one point per patient × seed)
# ---------------------------------------------------------------------------


def figure_width_vs_sigmav(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter: mean CI width vs σ²_v tertile, faceted by calibration layer.

    Proxy plot using tertile aggregates when per-patient scatter is unavailable.

    Args:
        df: Long-form aggregated table.
        output_path: Destination PNG path.
    """
    layers = [l for l in _LAYERS if l in df["layer"].values]
    models = [m for m in _BASE_MODELS if m in df["base_model"].values]
    if not layers or not models:
        return

    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4), sharey=False)
    if len(layers) == 1:
        axes = [axes]

    tertile_order = ["low", "mid", "high"]
    x_vals = [1, 2, 3]

    for ax, layer in zip(axes, layers, strict=False):
        for model in models:
            sub = df[
                (df["base_model"] == model)
                & (df["layer"] == layer)
                & (df["scope"] == "tertile")
                & (df["metric"] == "mean_width")
            ]
            if sub.empty:
                continue
            y_vals = []
            for tname in tertile_order:
                t_sub = sub[sub["tertile"] == tname]["value"]
                y_vals.append(float(t_sub.median()) if not t_sub.empty else np.nan)
            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                label=model,
                color=_MODEL_COLORS.get(model, "C0"),
            )

        ax.set_xticks(x_vals)
        ax.set_xticklabels(tertile_order)
        _setup_axes(ax, "σ²_v tertile", "Mean CI width", f"{layer}")

    axes[0].legend(title="Model", fontsize=7)
    fig.suptitle("Mean CI width vs σ²_v tertile", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FIGURE_MAP = {
    "is_by_model_calibration": figure_is_by_model_calibration,
    "coverage_by_model_calibration": figure_coverage_by_model_calibration,
    "tertile_panel": figure_tertile_panel,
    "width_vs_sigmav": figure_width_vs_sigmav,
}


def make_all_figures(df: pd.DataFrame, output_root: Path, cfg: dict) -> None:
    """Dispatch all configured figures to ``output_root/figures/``.

    Args:
        df: Long-form aggregated metric table.
        output_root: Root output directory.
        cfg: Full experiment config dict.
    """
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rep_cfg = cfg.get("reporting", {})
    requested: list[str] = rep_cfg.get("figures", list(_FIGURE_MAP.keys()))

    for name in requested:
        fn = _FIGURE_MAP.get(name)
        if fn is None:
            logger.warning("Unknown figure name '%s'; skipping", name)
            continue
        out = fig_dir / f"{name}.png"
        try:
            if name == "pit_grid":
                figure_pit_grid(output_root, out)
            else:
                fn(df, out)
        except Exception as exc:
            logger.warning("Figure '%s' failed: %s", name, exc)
