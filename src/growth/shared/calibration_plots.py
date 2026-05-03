# src/growth/shared/calibration_plots.py
"""Calibration diagnostic plots for probabilistic growth prediction.

Provides PIT histograms and sharpness-calibration scatter plots for
comparing classical analytical uncertainty (NLME σ²_pop) against
propagated segmentation uncertainty (heteroscedastic models).
"""

from __future__ import annotations

import numpy as np


def plot_pit_histogram(
    pit_values: np.ndarray,
    model_name: str,
    ax: object | None = None,
    n_bins: int = 10,
) -> object:
    """Bar chart of PIT histogram with uniform reference and acceptance bands.

    Under a calibrated model, PIT values are U[0,1] so the density
    histogram should be flat at 1.0. Binomial 95% acceptance bands
    show the expected range under calibration.

    Args:
        pit_values: PIT values in [0, 1].
        model_name: Label for the plot title.
        ax: Optional matplotlib Axes. If None, creates a new figure.
        n_bins: Number of histogram bins.

    Returns:
        matplotlib Figure containing the plot.
    """
    import matplotlib.pyplot as plt

    pit_values = np.asarray(pit_values, dtype=np.float64)
    n = len(pit_values)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    else:
        fig = ax.get_figure()

    counts, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0), density=True)
    bin_width = 1.0 / n_bins
    bin_centres = bin_edges[:-1] + bin_width / 2.0

    ax.bar(
        bin_centres, counts, width=bin_width * 0.85, color="#4878CF", alpha=0.8, edgecolor="white"
    )

    # Uniform reference
    ax.axhline(y=1.0, color="k", linestyle="--", linewidth=1.0, label="Uniform")

    # Binomial 95% acceptance bands: p = 1/n_bins, n trials
    p_bin = 1.0 / n_bins
    se = np.sqrt(p_bin * (1.0 - p_bin) / n) * n_bins
    ax.axhspan(1.0 - 1.96 * se, 1.0 + 1.96 * se, color="gray", alpha=0.15, label="95% band")

    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT histogram — {model_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(counts.max() * 1.15, 2.0))
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_pit_histogram_panel(
    pit_dict: dict[str, np.ndarray],
    n_bins: int = 10,
    ncols: int = 3,
) -> object:
    """Panel of PIT histograms, one per model.

    Args:
        pit_dict: Mapping model_name -> PIT values array.
        n_bins: Number of bins per histogram.
        ncols: Number of columns in the panel grid.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n_models = len(pit_dict)
    if n_models == 0:
        fig, _ = plt.subplots(1, 1)
        return fig

    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (model_name, pit_values) in enumerate(pit_dict.items()):
        plot_pit_histogram(pit_values, model_name, ax=axes_flat[idx], n_bins=n_bins)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    return fig


def plot_sharpness_calibration_scatter(
    model_metrics: dict[str, dict[str, float]],
    ax: object | None = None,
    nominal: float = 0.95,
) -> object:
    """Sharpness-calibration scatter: CI width vs empirical coverage.

    Each model is a labelled point. Ideal = (narrow CI, coverage == nominal).
    The Pareto frontier is toward (nominal, 0).

    Args:
        model_metrics: Dict of model_name -> {'coverage_95': float, 'mean_ci_width': float}.
        ax: Optional matplotlib Axes.
        nominal: Target coverage level.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    else:
        fig = ax.get_figure()

    names = list(model_metrics.keys())
    coverages = [model_metrics[n]["coverage_95"] for n in names]
    widths = [model_metrics[n]["mean_ci_width"] for n in names]

    # Colour by model family
    colours = []
    for n in names:
        if "NLME" in n:
            colours.append("#4878CF")
        elif "Hetero" in n:
            colours.append("#D65F5F")
        else:
            colours.append("#6ACC65")

    ax.scatter(widths, coverages, c=colours, s=80, zorder=5, edgecolors="white", linewidth=0.8)

    for name, w, c in zip(names, widths, coverages):
        ax.annotate(
            name,
            (w, c),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )

    ax.axhline(y=nominal, color="k", linestyle="--", linewidth=0.8, label=f"Nominal {nominal:.0%}")
    ax.set_xlabel("Mean 95% CI width (log-volume)")
    ax.set_ylabel("Empirical 95% coverage")
    ax.set_title("Sharpness vs Calibration")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
