"""Performance visualization plots.

Generates plots for reconstruction quality and semantic R² metrics.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, visualization disabled")


def plot_performance_metrics(
    data: Dict[str, Any],
    output_dir: str,
    dpi: int = 150,
    format: str = "png",
) -> Dict[str, str]:
    """Generate all performance-related plots.

    Args:
        data: Dictionary from load_experiment_data()
        output_dir: Directory to save plots
        dpi: Resolution for saved plots
        format: Image format ("png" or "pdf")

    Returns:
        Dictionary mapping plot names to file paths
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping visualization")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Reconstruction quality over epochs
    recon_path = _plot_reconstruction_quality(data, output_path, dpi, format)
    if recon_path:
        plots["reconstruction_quality"] = str(recon_path)

    # Semantic R² curves
    semantic_path = _plot_semantic_r2_curves(data, output_path, dpi, format)
    if semantic_path:
        plots["semantic_r2_curves"] = str(semantic_path)

    # Per-modality SSIM
    ssim_path = _plot_per_modality_ssim(data, output_path, dpi, format)
    if ssim_path:
        plots["per_modality_ssim"] = str(ssim_path)

    return plots


def _plot_reconstruction_quality(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot reconstruction MSE over training."""
    metrics_df = data.get("metrics")
    if metrics_df is None or metrics_df.empty:
        return None

    # Find relevant columns
    recon_cols = []
    for col in ["val_epoch/recon", "train_epoch/recon"]:
        if col in metrics_df.columns:
            recon_cols.append(col)

    if not recon_cols:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for col in recon_cols:
        label = "Validation" if "val" in col else "Training"
        df = metrics_df[["epoch", col]].dropna()
        ax.plot(df["epoch"], df[col], label=label, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction Quality Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale if values vary greatly
    if metrics_df[recon_cols[0]].max() / metrics_df[recon_cols[0]].min() > 100:
        ax.set_yscale("log")

    filepath = output_path / f"reconstruction_quality.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_semantic_r2_curves(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot semantic R² values over training."""
    semantic_df = data.get("semantic_quality")
    if semantic_df is None or semantic_df.empty:
        return None

    r2_cols = {
        "Volume": "r2_vol",
        "Location": "r2_loc",
        "Shape": "r2_shape",
    }

    # Check which columns exist
    available_cols = {k: v for k, v in r2_cols.items() if v in semantic_df.columns}
    if not available_cols:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Volume": "#2ecc71", "Location": "#3498db", "Shape": "#9b59b6"}

    for name, col in available_cols.items():
        df = semantic_df[["epoch", col]].dropna()
        ax.plot(df["epoch"], df[col], label=name, color=colors.get(name), linewidth=2)

    # Add target lines
    targets = {"Volume": 0.85, "Location": 0.90, "Shape": 0.35}
    for name in available_cols.keys():
        target = targets.get(name)
        if target:
            ax.axhline(y=target, color=colors.get(name), linestyle="--", alpha=0.5)
            ax.annotate(
                f"{name} target ({target})",
                xy=(ax.get_xlim()[1], target),
                xytext=(-80, 5),
                textcoords="offset points",
                fontsize=8,
                color=colors.get(name),
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("R² Score")
    ax.set_title("Semantic Encoding Quality (R²) Over Training")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    filepath = output_path / f"semantic_r2_curves.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_per_modality_ssim(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot per-modality SSIM over training."""
    metrics_df = data.get("metrics")
    if metrics_df is None or metrics_df.empty:
        return None

    modalities = ["t1c", "t1n", "t2f", "t2w"]
    ssim_cols = {mod: f"val_epoch/ssim_{mod}" for mod in modalities}

    # Check which columns exist
    available_cols = {k: v for k, v in ssim_cols.items() if v in metrics_df.columns}
    if not available_cols:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "t1c": "#e74c3c",
        "t1n": "#f39c12",
        "t2f": "#3498db",
        "t2w": "#2ecc71",
    }

    for mod, col in available_cols.items():
        df = metrics_df[["epoch", col]].dropna()
        if not df.empty:
            ax.plot(
                df["epoch"],
                df[col],
                label=mod.upper(),
                color=colors.get(mod),
                linewidth=1.5,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.set_title("Per-Modality SSIM Over Training")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    filepath = output_path / f"per_modality_ssim.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath
