"""ODE utility visualization plots.

Generates plots for cross-partition correlations, factor independence,
and ODE readiness assessment.
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


def plot_ode_utility(
    data: Dict[str, Any],
    output_dir: str,
    dpi: int = 150,
    format: str = "png",
) -> Dict[str, str]:
    """Generate all ODE utility plots.

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

    # Cross-partition correlation matrix
    matrix_path = _plot_cross_partition_matrix(data, output_path, dpi, format)
    if matrix_path:
        plots["cross_partition_matrix"] = str(matrix_path)

    # Factor independence over time
    independence_path = _plot_independence_over_time(data, output_path, dpi, format)
    if independence_path:
        plots["factor_independence"] = str(independence_path)

    # ODE readiness score over time
    readiness_path = _plot_ode_readiness_over_time(data, output_path, dpi, format)
    if readiness_path:
        plots["ode_readiness"] = str(readiness_path)

    return plots


def _plot_cross_partition_matrix(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot cross-partition correlation matrix at final epoch."""
    cross_df = data.get("cross_correlation")
    metrics_df = data.get("metrics")

    # Try to get correlations from cross_correlation.csv first
    if cross_df is not None and not cross_df.empty:
        # Use final epoch
        final_epoch = cross_df["epoch"].max()
        final_data = cross_df[cross_df["epoch"] == final_epoch].iloc[0]

        partitions = ["z_vol", "z_loc", "z_shape"]
        n = len(partitions)
        corr_matrix = np.eye(n)

        pairs = [
            ("z_vol_z_loc", 0, 1),
            ("z_vol_z_shape", 0, 2),
            ("z_loc_z_shape", 1, 2),
        ]

        for col, i, j in pairs:
            if col in final_data.index and pd.notna(final_data[col]):
                val = float(final_data[col])
                corr_matrix[i, j] = val
                corr_matrix[j, i] = val

    elif metrics_df is not None and not metrics_df.empty:
        # Try to get from metrics columns
        final_epoch = metrics_df["epoch"].max()
        final_data = metrics_df[metrics_df["epoch"] == final_epoch].iloc[0]

        partitions = ["z_vol", "z_loc", "z_shape"]
        n = len(partitions)
        corr_matrix = np.eye(n)

        col_prefixes = ["cross_part/", "val_cross_part/"]
        pairs = [
            ("z_vol_z_loc", 0, 1),
            ("z_vol_z_shape", 0, 2),
            ("z_loc_z_shape", 1, 2),
        ]

        for base_col, i, j in pairs:
            for prefix in col_prefixes:
                col = prefix + base_col
                if col in final_data.index and pd.notna(final_data[col]):
                    val = float(final_data[col])
                    corr_matrix[i, j] = val
                    corr_matrix[j, i] = val
                    break
    else:
        return None

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    # Labels
    labels = ["Volume", "Location", "Shape"]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add correlation values as text
    for i in range(n):
        for j in range(n):
            color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                   ha="center", va="center", color=color, fontsize=12)

    ax.set_title(f"Cross-Partition Correlations (Epoch {final_epoch})")
    plt.colorbar(im, ax=ax, label="Correlation")

    filepath = output_path / f"cross_partition_matrix.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_independence_over_time(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot factor independence score over training."""
    cross_df = data.get("cross_correlation")
    metrics_df = data.get("metrics")

    epochs = []
    max_corrs = []

    if cross_df is not None and not cross_df.empty:
        pairs = ["z_vol_z_loc", "z_vol_z_shape", "z_loc_z_shape"]
        for _, row in cross_df.iterrows():
            corrs = []
            for col in pairs:
                if col in row.index and pd.notna(row[col]):
                    corrs.append(abs(float(row[col])))
            if corrs:
                epochs.append(row["epoch"])
                max_corrs.append(max(corrs))
    elif metrics_df is not None:
        # Extract from metrics columns
        col_prefixes = ["cross_part/", "val_cross_part/"]
        pairs = ["z_vol_z_loc", "z_vol_z_shape", "z_loc_z_shape"]

        for _, row in metrics_df.iterrows():
            corrs = []
            for base_col in pairs:
                for prefix in col_prefixes:
                    col = prefix + base_col
                    if col in row.index and pd.notna(row[col]):
                        corrs.append(abs(float(row[col])))
                        break
            if corrs:
                epochs.append(row["epoch"])
                max_corrs.append(max(corrs))

    if not epochs:
        return None

    # Compute independence score
    independence = [1.0 - mc for mc in max_corrs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Max cross-correlation
    ax1 = axes[0]
    ax1.plot(epochs, max_corrs, "r-", linewidth=2)
    ax1.axhline(y=0.30, color="g", linestyle="--", alpha=0.7, label="Target (<0.30)")
    ax1.fill_between(epochs, 0, max_corrs, alpha=0.3, color="red")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Max |Cross-Partition Correlation|")
    ax1.set_title("Factor Entanglement Over Training")
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Independence score
    ax2 = axes[1]
    ax2.plot(epochs, independence, "g-", linewidth=2)
    ax2.axhline(y=0.70, color="r", linestyle="--", alpha=0.7, label="Target (>0.70)")
    ax2.fill_between(epochs, 0, independence, alpha=0.3, color="green")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Independence Score (1 - max|corr|)")
    ax2.set_title("Factor Independence Over Training")
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = output_path / f"factor_independence.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_ode_readiness_over_time(
    data: Dict[str, Any],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot ODE readiness composite score over training."""
    metrics_df = data.get("metrics")
    semantic_df = data.get("semantic_quality")

    if metrics_df is None or metrics_df.empty:
        return None

    # Check if ode_readiness is logged directly
    if "val_epoch/ode_readiness" in metrics_df.columns:
        df = metrics_df[["epoch", "val_epoch/ode_readiness"]].dropna()
        epochs = df["epoch"].values
        readiness = df["val_epoch/ode_readiness"].values
    elif semantic_df is not None and not semantic_df.empty:
        # Compute from components
        epochs = []
        readiness = []

        for _, row in semantic_df.iterrows():
            vol_r2 = row.get("r2_vol", 0) if pd.notna(row.get("r2_vol")) else 0
            loc_r2 = row.get("r2_loc", 0) if pd.notna(row.get("r2_loc")) else 0

            # Assume independence score of 0.5 if not available
            score = 0.5 * vol_r2 + 0.25 * loc_r2 + 0.25 * 0.5
            epochs.append(row["epoch"])
            readiness.append(score)

        epochs = np.array(epochs)
        readiness = np.array(readiness)
    else:
        return None

    if len(epochs) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, readiness, "b-", linewidth=2)
    ax.fill_between(epochs, 0, readiness, alpha=0.3)

    # Grade thresholds (approximate)
    ax.axhline(y=0.75, color="gold", linestyle="--", alpha=0.7, label="Grade A threshold")
    ax.axhline(y=0.60, color="silver", linestyle="--", alpha=0.7, label="Grade B threshold")
    ax.axhline(y=0.45, color="orange", linestyle="--", alpha=0.7, label="Grade C threshold")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ODE Readiness Score")
    ax.set_title("Neural ODE Readiness Over Training")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Annotate final score
    if len(readiness) > 0:
        final_score = readiness[-1]
        ax.annotate(
            f"Final: {final_score:.3f}",
            xy=(epochs[-1], final_score),
            xytext=(-60, 20),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    filepath = output_path / f"ode_readiness.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return filepath
