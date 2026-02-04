# src/growth/evaluation/visualization.py
"""Visualization utilities for evaluation.

Provides publication-quality plotting functions for:
- Latent space visualization (UMAP)
- Feature variance spectrum
- Prediction scatter plots
- R² comparison bar charts
- Correlation matrices

These functions are designed to be reusable across experiments.

Example:
    >>> from growth.evaluation.visualization import (
    ...     set_publication_style,
    ...     plot_umap,
    ...     save_figure,
    ... )
    >>> set_publication_style()
    >>> fig, ax = plot_umap(features, labels)
    >>> save_figure(fig, "latent_space", output_dir, formats=["pdf", "png"])
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Figure = None
    Axes = None

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

try:
    from umap import UMAP

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    UMAP = None


def set_publication_style(
    font_size: int = 10,
    axes_label_size: int = 11,
    axes_title_size: int = 12,
    tick_label_size: int = 9,
    legend_font_size: int = 9,
    figure_dpi: int = 150,
    save_dpi: int = 300,
) -> None:
    """Configure matplotlib for publication-quality figures.

    Sets font sizes, DPI, and other parameters suitable for academic papers.

    Args:
        font_size: Base font size.
        axes_label_size: Axis label font size.
        axes_title_size: Title font size.
        tick_label_size: Tick label font size.
        legend_font_size: Legend font size.
        figure_dpi: Display DPI.
        save_dpi: Saved figure DPI.

    Raises:
        ImportError: If matplotlib is not available.

    Example:
        >>> set_publication_style()
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": axes_label_size,
            "axes.titlesize": axes_title_size,
            "xtick.labelsize": tick_label_size,
            "ytick.labelsize": tick_label_size,
            "legend.fontsize": legend_font_size,
            "figure.dpi": figure_dpi,
            "savefig.dpi": save_dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def save_figure(
    fig: "Figure",
    name: str,
    output_dir: Union[str, Path],
    formats: Sequence[str] = ("pdf", "png"),
    close: bool = True,
) -> List[Path]:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure to save.
        name: Base filename (without extension).
        output_dir: Directory to save figures.
        formats: File formats to save (e.g., ["pdf", "png"]).
        close: Whether to close the figure after saving.

    Returns:
        List of saved file paths.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> paths = save_figure(fig, "my_plot", "output/figures")
        >>> print(paths)  # [PosixPath('output/figures/my_plot.pdf'), ...]
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path)
        logger.info(f"Saved figure to {path}")
        saved_paths.append(path)

    if close:
        plt.close(fig)

    return saved_paths


def plot_umap(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    color_by: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    title: str = "UMAP Projection",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: Tuple[int, int] = (8, 6),
    point_size: int = 10,
    alpha: float = 0.6,
    label_colors: Optional[Dict[Any, str]] = None,
    colorbar_label: Optional[str] = None,
) -> Tuple["Figure", "Axes"]:
    """Create UMAP visualization of features.

    Args:
        features: Feature array of shape [N, D].
        labels: Optional categorical labels for coloring [N].
        color_by: Optional continuous values for coloring [N].
            If both labels and color_by are provided, labels takes precedence.
        cmap: Colormap for continuous coloring.
        title: Plot title.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed for reproducibility.
        figsize: Figure size (width, height).
        point_size: Scatter point size.
        alpha: Point transparency.
        label_colors: Optional dict mapping labels to colors.
        colorbar_label: Label for colorbar (only used with color_by).

    Returns:
        Tuple of (figure, axes).

    Raises:
        ImportError: If matplotlib or umap is not available.

    Example:
        >>> features = np.random.randn(100, 768)
        >>> labels = np.array(["A"] * 50 + ["B"] * 50)
        >>> fig, ax = plot_umap(features, labels=labels)
        >>> plt.show()
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")
    if not HAS_UMAP:
        raise ImportError("umap-learn is required for UMAP visualization")

    # Fit UMAP
    logger.info(f"Fitting UMAP on {len(features)} samples...")
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = umap.fit_transform(features)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        # Categorical coloring
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            color = label_colors.get(label, None) if label_colors else None
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=color,
                label=str(label),
                s=point_size,
                alpha=alpha,
            )
        ax.legend(markerscale=2)
    elif color_by is not None:
        # Continuous coloring
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=color_by,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    else:
        # No coloring
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=point_size,
            alpha=alpha,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_variance_spectrum(
    features: np.ndarray,
    title: str = "Feature Variance per Dimension",
    figsize: Tuple[int, int] = (10, 4),
    log_scale: bool = True,
    low_variance_threshold: Optional[float] = 0.01,
    color: str = "steelblue",
) -> Tuple["Figure", "Axes"]:
    """Plot variance per feature dimension (sorted descending).

    Useful for identifying collapsed dimensions in latent representations.

    Args:
        features: Feature array of shape [N, D].
        title: Plot title.
        figsize: Figure size (width, height).
        log_scale: Whether to use log scale for y-axis.
        low_variance_threshold: Optional threshold line for low variance.
        color: Line color.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> features = np.random.randn(100, 768)
        >>> fig, ax = plot_variance_spectrum(features)
        >>> plt.show()
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    variance = features.var(axis=0)
    sorted_variance = np.sort(variance)[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(sorted_variance, color=color, alpha=0.8)

    if low_variance_threshold is not None:
        ax.axhline(
            y=low_variance_threshold,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Threshold ({low_variance_threshold})",
        )
        ax.legend()

    ax.set_xlabel("Dimension (sorted by variance)")
    ax.set_ylabel("Variance")
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()
    return fig, ax


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Ground Truth",
    xlabel: str = "Ground Truth",
    ylabel: str = "Prediction",
    figsize: Tuple[int, int] = (6, 6),
    point_size: int = 10,
    alpha: float = 0.3,
    color: str = "steelblue",
    show_identity: bool = True,
    show_r2: bool = True,
) -> Tuple["Figure", "Axes"]:
    """Plot predictions vs ground truth scatter plot.

    Args:
        y_true: Ground truth values [N] or [N, 1].
        y_pred: Predicted values [N] or [N, 1].
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size (width, height).
        point_size: Scatter point size.
        alpha: Point transparency.
        color: Point color.
        show_identity: Whether to show identity line (y=x).
        show_r2: Whether to show R² value.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1])
        >>> fig, ax = plot_prediction_scatter(y_true, y_pred)
        >>> plt.show()
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Flatten if needed
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, s=point_size, alpha=alpha, c=color)

    if show_identity:
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.5, label="Identity")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    if show_r2:
        # Compute R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_r2_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_keys: Sequence[str] = ("r2_volume", "r2_location", "r2_shape"),
    titles: Sequence[str] = ("Volume R²", "Location R²", "Shape R²"),
    figsize: Tuple[int, int] = (12, 4),
    colors: Optional[Dict[str, str]] = None,
) -> Tuple["Figure", np.ndarray]:
    """Plot R² comparison across conditions.

    Args:
        metrics: Dict mapping condition names to metric dicts.
            Example: {"baseline": {"r2_volume": 0.9, ...}, "lora_r8": {...}}
        metric_keys: Metric keys to plot.
        titles: Subplot titles corresponding to metric_keys.
        figsize: Figure size (width, height).
        colors: Optional dict mapping condition names to colors.

    Returns:
        Tuple of (figure, axes array).

    Example:
        >>> metrics = {
        ...     "baseline": {"r2_volume": 0.85, "r2_location": 0.90},
        ...     "lora_r8": {"r2_volume": 0.88, "r2_location": 0.92},
        ... }
        >>> fig, axes = plot_r2_comparison(metrics, ["r2_volume", "r2_location"])
        >>> plt.show()
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    conditions = list(metrics.keys())
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for ax, key, title in zip(axes, metric_keys, titles):
        values = [metrics[cond].get(key, 0) for cond in conditions]

        bar_colors = (
            [colors.get(c, "steelblue") for c in conditions] if colors else "steelblue"
        )

        x = np.arange(len(conditions))
        bars = ax.bar(x, values, color=bar_colors, alpha=0.8)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha="right")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if abs(height) < 10:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=7,
                )

    plt.tight_layout()
    return fig, np.array(axes)


def plot_correlation_matrix(
    matrix: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = True,
    fmt: str = ".2f",
) -> Tuple["Figure", "Axes"]:
    """Plot correlation matrix heatmap.

    Args:
        matrix: Correlation matrix [N, N].
        labels: Optional row/column labels.
        title: Plot title.
        figsize: Figure size (width, height).
        cmap: Colormap.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        annot: Whether to annotate cells with values.
        fmt: Format string for annotations.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> matrix = np.corrcoef(np.random.randn(5, 100))
        >>> fig, ax = plot_correlation_matrix(matrix, labels=["A", "B", "C", "D", "E"])
        >>> plt.show()
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt=fmt,
            square=True,
            xticklabels=labels if labels else False,
            yticklabels=labels if labels else False,
        )
    else:
        # Fallback to matplotlib
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)

        if labels:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)

        if annot:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:{fmt[1:]}}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


__all__ = [
    # Style
    "set_publication_style",
    "save_figure",
    # Plots
    "plot_umap",
    "plot_variance_spectrum",
    "plot_prediction_scatter",
    "plot_r2_comparison",
    "plot_correlation_matrix",
    # Availability flags
    "HAS_MATPLOTLIB",
    "HAS_SEABORN",
    "HAS_UMAP",
]
