"""Training dynamics visualization plots.

Generates plots for loss breakdown, schedule progression, and gradient stability.
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


def plot_training_dynamics(
    data: Dict[str, Any],
    output_dir: str,
    dpi: int = 150,
    format: str = "png",
) -> Dict[str, str]:
    """Generate all training dynamics plots.

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

    # Loss breakdown
    loss_path = _plot_loss_breakdown(data, output_path, dpi, format)
    if loss_path:
        plots["loss_breakdown"] = str(loss_path)

    # Schedule progression
    schedule_path = _plot_schedule_progression(data, output_path, dpi, format)
    if schedule_path:
        plots["schedule_progression"] = str(schedule_path)

    # Gradient stability
    grad_path = _plot_gradient_stability(data, output_path, dpi, format)
    if grad_path:
        plots["gradient_stability"] = str(grad_path)

    return plots


def _plot_loss_breakdown(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot loss components over training."""
    metrics_df = data.get("metrics")
    if metrics_df is None or metrics_df.empty:
        return None

    # Define loss components to plot
    loss_cols = {
        "Total": ["val_epoch/loss", "train_epoch/loss"],
        "Reconstruction": ["val_epoch/recon", "train_epoch/recon"],
        "KL (raw)": ["val_epoch/kl_raw", "train_epoch/kl_raw"],
        "TC": ["val_epoch/tc", "train_epoch/tc"],
        "Semantic": ["val_epoch/semantic_total", "train_epoch/semantic_total"],
        "Cross-partition": ["val_epoch/cross_partition", "train_epoch/cross_partition"],
    }

    # Find which columns exist
    available = {}
    for name, cols in loss_cols.items():
        for col in cols:
            if col in metrics_df.columns:
                available[name] = col
                break

    if len(available) < 2:
        return None

    # Create subplots
    n_plots = len(available)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_plots))

    for idx, (name, col) in enumerate(available.items()):
        ax = axes[idx]
        df = metrics_df[["epoch", col]].dropna()

        if not df.empty:
            ax.plot(df["epoch"], df[col], color=colors[idx], linewidth=1.5)
            ax.fill_between(df["epoch"], 0, df[col], alpha=0.3, color=colors[idx])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

        # Log scale if appropriate
        if df[col].max() / (df[col].min() + 1e-10) > 100:
            ax.set_yscale("log")

    # Hide unused axes
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    filepath = output_path / f"loss_breakdown.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_schedule_progression(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot training schedules (beta, lambdas) over epochs."""
    metrics_df = data.get("metrics")
    if metrics_df is None or metrics_df.empty:
        return None

    # Schedule columns
    schedule_cols = {
        "Beta (KL)": "sched/beta",
        "Lambda Vol": "sched/lambda_vol",
        "Lambda Loc": "sched/lambda_loc",
        "Lambda Shape": "sched/lambda_shape",
        "Lambda TC": "sched/lambda_tc",
        "Lambda Cross": "sched/lambda_cross_partition",
    }

    available = {k: v for k, v in schedule_cols.items() if v in metrics_df.columns}

    if not available:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: KL beta
    ax1 = axes[0]
    if "Beta (KL)" in available:
        col = available["Beta (KL)"]
        df = metrics_df[["epoch", col]].dropna()
        ax1.plot(df["epoch"], df[col], "b-", linewidth=2, label="Beta")
        ax1.set_ylabel("Beta Value")
        ax1.set_title("KL Annealing Schedule")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Lambda schedules
    ax2 = axes[1]
    colors = {"Vol": "#2ecc71", "Loc": "#3498db", "Shape": "#9b59b6", "TC": "#e74c3c", "Cross": "#f39c12"}

    for name, col in available.items():
        if name == "Beta (KL)":
            continue

        short_name = name.replace("Lambda ", "")
        df = metrics_df[["epoch", col]].dropna()
        if not df.empty:
            ax2.plot(
                df["epoch"],
                df[col],
                color=colors.get(short_name, "gray"),
                linewidth=1.5,
                label=short_name,
            )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Lambda Value")
    ax2.set_title("Semantic Loss Weight Schedules")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = output_path / f"schedule_progression.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_gradient_stability(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot gradient norm over training."""
    metrics_df = data.get("metrics")
    if metrics_df is None or metrics_df.empty:
        return None

    # Find gradient norm column
    grad_col = None
    for col in ["train_epoch/grad_norm", "grad_norm", "diag/grad_norm"]:
        if col in metrics_df.columns:
            grad_col = col
            break

    if grad_col is None:
        return None

    df = metrics_df[["epoch", grad_col]].dropna()
    if df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Gradient norm over time
    ax1 = axes[0]
    ax1.plot(df["epoch"], df[grad_col], "b-", linewidth=1, alpha=0.7)
    ax1.axhline(y=10, color="r", linestyle="--", alpha=0.7, label="Explosion threshold")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gradient Norm")
    ax1.set_title("Gradient Norm Over Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark explosions
    explosions = df[df[grad_col] > 10]
    if not explosions.empty:
        ax1.scatter(explosions["epoch"], explosions[grad_col], color="red", s=20, zorder=5)

    # Plot 2: Rolling statistics
    ax2 = axes[1]

    window = min(50, len(df) // 4)
    if window >= 2:
        rolling_mean = df[grad_col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[grad_col].rolling(window=window, min_periods=1).std()

        ax2.plot(df["epoch"], rolling_mean, "b-", linewidth=2, label=f"Mean (window={window})")
        ax2.fill_between(
            df["epoch"],
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.3,
            label="±1 std",
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Gradient Stability (Rolling Statistics)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = output_path / f"gradient_stability.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath
