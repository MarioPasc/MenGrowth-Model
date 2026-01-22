"""Comparison visualization plots.

Generates overlay plots for comparing multiple experiment runs.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_comparison(
    run_data: Dict[str, Dict[str, Any]],
    output_dir: str,
    dpi: int = 150,
    format: str = "png",
) -> Dict[str, str]:
    """Generate comparison plots for multiple runs.

    Args:
        run_data: Dictionary mapping run_id to loaded experiment data
        output_dir: Directory to save plots
        dpi: Resolution for saved plots
        format: Image format ("png" or "pdf")

    Returns:
        Dictionary mapping plot names to file paths
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping visualization")
        return {}

    if len(run_data) < 2:
        logger.warning("Need at least 2 runs for comparison")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = {}

    # R² comparison
    r2_path = _plot_r2_comparison(run_data, output_path, dpi, format)
    if r2_path:
        plots["comparison_r2"] = str(r2_path)

    # AU comparison
    au_path = _plot_au_comparison(run_data, output_path, dpi, format)
    if au_path:
        plots["comparison_au"] = str(au_path)

    # Radar chart comparison
    radar_path = _plot_radar_comparison(run_data, output_path, dpi, format)
    if radar_path:
        plots["comparison_radar"] = str(radar_path)

    return plots


def _plot_r2_comparison(
    run_data: Dict[str, Dict[str, Any]],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot R² curves for all runs overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(run_data)))
    r2_cols = [("r2_vol", "Volume R²", 0.85), ("r2_loc", "Location R²", 0.90), ("r2_shape", "Shape R²", 0.35)]

    for col_idx, (col, title, target) in enumerate(r2_cols):
        ax = axes[col_idx]

        for run_idx, (run_id, data) in enumerate(run_data.items()):
            semantic_df = data.get("semantic_quality")
            if semantic_df is not None and col in semantic_df.columns:
                df = semantic_df[["epoch", col]].dropna()
                # Use short run_id for legend
                short_id = run_id[:20] + "..." if len(run_id) > 20 else run_id
                ax.plot(df["epoch"], df[col], color=colors[run_idx], linewidth=1.5, label=short_id, alpha=0.8)

        ax.axhline(y=target, color="gray", linestyle="--", alpha=0.5, label=f"Target ({target})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = output_path / f"comparison_r2.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_au_comparison(
    run_data: Dict[str, Dict[str, Any]],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot AU fraction for all runs overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(run_data)))

    for run_idx, (run_id, data) in enumerate(run_data.items()):
        au_df = data.get("au_history")
        if au_df is not None and "au_frac" in au_df.columns:
            short_id = run_id[:20] + "..." if len(run_id) > 20 else run_id
            ax.plot(au_df["epoch"], au_df["au_frac"], color=colors[run_idx], linewidth=1.5, label=short_id, alpha=0.8)

    ax.axhline(y=0.10, color="red", linestyle="--", alpha=0.7, label="Collapse threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Units Fraction")
    ax.set_title("Active Units Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    filepath = output_path / f"comparison_au.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_radar_comparison(
    run_data: Dict[str, Dict[str, Any]],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot radar chart comparing final metrics across runs."""
    from ..statistics.summary import generate_summary

    # Generate summaries for each run
    summaries = {}
    for run_id, data in run_data.items():
        try:
            summaries[run_id] = generate_summary(data)
        except Exception as e:
            logger.warning(f"Failed to generate summary for {run_id}: {e}")

    if len(summaries) < 2:
        return None

    # Metrics for radar chart
    metrics = [
        ("Vol R²", lambda s: s.performance.vol_r2),
        ("Loc R²", lambda s: s.performance.loc_r2),
        ("Shape R²", lambda s: s.performance.shape_r2),
        ("AU Frac", lambda s: s.collapse.au_frac_residual),
        ("Independence", lambda s: s.ode_utility.independence_score),
        ("ODE Ready", lambda s: s.ode_utility.ode_readiness),
    ]

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(summaries)))

    for run_idx, (run_id, summary) in enumerate(summaries.items()):
        values = [func(summary) for _, func in metrics]
        values += values[:1]  # Close the polygon

        short_id = run_id[:15] + "..." if len(run_id) > 15 else run_id
        ax.plot(angles, values, color=colors[run_idx], linewidth=2, label=short_id)
        ax.fill(angles, values, color=colors[run_idx], alpha=0.15)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name for name, _ in metrics])
    ax.set_ylim(0, 1)

    ax.set_title("Multi-Run Comparison Radar", size=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)

    filepath = output_path / f"comparison_radar.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def create_comparison_table(
    run_data: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> Optional[str]:
    """Create comparison table as CSV.

    Args:
        run_data: Dictionary mapping run_id to loaded data
        output_dir: Directory to save table

    Returns:
        Path to saved CSV file
    """
    from ..statistics.summary import generate_summary
    from ..statistics.comparison import create_comparison_dataframe

    summaries = {}
    for run_id, data in run_data.items():
        try:
            summaries[run_id] = generate_summary(data)
        except Exception as e:
            logger.warning(f"Failed to generate summary for {run_id}: {e}")

    if not summaries:
        return None

    df = create_comparison_dataframe(summaries)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / "comparison_table.csv"
    df.to_csv(filepath)

    return str(filepath)
