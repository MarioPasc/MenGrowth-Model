"""Collapse diagnostic plots.

Generates plots for active units, partition activity, and decoder bypass analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_collapse_metrics(
    data: Dict[str, Any],
    output_dir: str,
    dpi: int = 150,
    format: str = "png",
) -> Dict[str, str]:
    """Generate all collapse diagnostic plots.

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

    # AU over time
    au_path = _plot_au_over_time(data, output_path, dpi, format)
    if au_path:
        plots["au_over_time"] = str(au_path)

    # Partition activity heatmap
    heatmap_path = _plot_partition_activity_heatmap(data, output_path, dpi, format)
    if heatmap_path:
        plots["partition_activity_heatmap"] = str(heatmap_path)

    # Decoder bypass test
    bypass_path = _plot_decoder_bypass_test(data, output_path, dpi, format)
    if bypass_path:
        plots["decoder_bypass_test"] = str(bypass_path)

    return plots


def _plot_au_over_time(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot Active Units over training."""
    au_df = data.get("au_history")
    if au_df is None or au_df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AU count
    ax1 = axes[0]
    if "au_count" in au_df.columns:
        ax1.plot(au_df["epoch"], au_df["au_count"], "b-", linewidth=2)
        ax1.fill_between(au_df["epoch"], 0, au_df["au_count"], alpha=0.3)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Active Units Count")
    ax1.set_title("Active Units Over Training")
    ax1.grid(True, alpha=0.3)

    # Add z_dim reference line if available
    if "z_dim" in au_df.columns:
        z_dim = au_df["z_dim"].iloc[0]
        ax1.axhline(y=z_dim, color="r", linestyle="--", alpha=0.5, label=f"z_dim={z_dim}")
        ax1.legend()

    # AU fraction
    ax2 = axes[1]
    if "au_frac" in au_df.columns:
        ax2.plot(au_df["epoch"], au_df["au_frac"], "g-", linewidth=2)
        ax2.fill_between(au_df["epoch"], 0, au_df["au_frac"], alpha=0.3, color="green")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Active Units Fraction")
    ax2.set_title("Active Units Fraction Over Training")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.10, color="r", linestyle="--", alpha=0.5, label="Collapse threshold (10%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = output_path / f"au_over_time.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_partition_activity_heatmap(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot heatmap of partition activity (variance) over training."""
    partition_df = data.get("partition_stats")
    if partition_df is None or partition_df.empty:
        return None

    if "mu_var_mean" not in partition_df.columns:
        return None

    # Pivot to get partitions as columns
    partitions = ["z_vol", "z_loc", "z_shape", "z_residual"]
    available_partitions = [p for p in partitions if p in partition_df["partition"].values]

    if not available_partitions:
        return None

    # Get epochs and create matrix
    epochs = sorted(partition_df["epoch"].unique())
    heatmap_data = np.zeros((len(available_partitions), len(epochs)))

    for i, partition in enumerate(available_partitions):
        part_data = partition_df[partition_df["partition"] == partition]
        for j, epoch in enumerate(epochs):
            epoch_data = part_data[part_data["epoch"] == epoch]
            if not epoch_data.empty:
                heatmap_data[i, j] = epoch_data["mu_var_mean"].values[0]

    # Log scale for better visualization
    heatmap_data = np.log10(heatmap_data + 1e-6)

    fig, ax = plt.subplots(figsize=(12, 4))

    im = ax.imshow(heatmap_data, aspect="auto", cmap="viridis")

    # Labels
    ax.set_yticks(range(len(available_partitions)))
    ax.set_yticklabels([p.replace("z_", "") for p in available_partitions])

    # X-axis: show subset of epochs
    n_ticks = min(10, len(epochs))
    tick_indices = np.linspace(0, len(epochs) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(epochs[i]) for i in tick_indices])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Partition")
    ax.set_title("Partition Activity (log₁₀ variance) Over Training")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log₁₀(variance)")

    filepath = output_path / f"partition_activity_heatmap.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_decoder_bypass_test(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot decoder bypass test (recon_mu_mse vs recon_z0_mse)."""
    metrics_df = data.get("metrics")
    if metrics_df is None or metrics_df.empty:
        return None

    mu_col = "diag/recon_mu_mse"
    z0_col = "diag/recon_z0_mse"

    if mu_col not in metrics_df.columns or z0_col not in metrics_df.columns:
        return None

    df = metrics_df[["epoch", mu_col, z0_col]].dropna()
    if df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Both MSE values over time
    ax1 = axes[0]
    ax1.plot(df["epoch"], df[mu_col], "b-", label="recon(μ)", linewidth=2)
    ax1.plot(df["epoch"], df[z0_col], "r-", label="recon(z=0)", linewidth=2)
    ax1.fill_between(df["epoch"], df[mu_col], df[z0_col], alpha=0.3, color="gray")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.set_title("Decoder Bypass Test: μ vs z=0 Reconstruction")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gap ratio over time
    ax2 = axes[1]
    gap = df[z0_col] - df[mu_col]
    ratio = gap / (df[mu_col] + 1e-8)

    ax2.plot(df["epoch"], ratio, "g-", linewidth=2)
    ax2.axhline(y=0.20, color="r", linestyle="--", alpha=0.7, label="Bypass threshold (20%)")
    ax2.fill_between(df["epoch"], 0, ratio, alpha=0.3, color="green")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("(z=0 MSE - μ MSE) / μ MSE")
    ax2.set_title("Decoder z-Dependence Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Highlight bypass region
    ax2.axhspan(-1, 0.20, alpha=0.1, color="red", label="_bypass region")

    plt.tight_layout()

    filepath = output_path / f"decoder_bypass_test.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath
