"""Dashboard assembly for comprehensive experiment visualization.

Creates a multi-panel summary dashboard combining key metrics and plots.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_dashboard(
    data: Dict[str, Any],
    summary: Any,  # AnalysisSummary
    output_dir: str,
    dpi: int = 150,
    format: str = "png",
) -> Optional[str]:
    """Create comprehensive dashboard visualization.

    Args:
        data: Dictionary from load_experiment_data()
        summary: AnalysisSummary dataclass with computed metrics
        output_dir: Directory to save dashboard
        dpi: Resolution for saved plot
        format: Image format ("png" or "pdf")

    Returns:
        Path to saved dashboard file
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping dashboard")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Row 0: Header with summary info
    _add_header(fig, gs[0, :], summary)

    # Row 1: Performance metrics
    _add_semantic_r2_panel(fig, gs[1, 0:2], data)
    _add_reconstruction_panel(fig, gs[1, 2:4], data)

    # Row 2: Collapse diagnostics
    _add_au_panel(fig, gs[2, 0:2], data)
    _add_partition_variance_panel(fig, gs[2, 2:4], data, summary)

    # Row 3: ODE utility
    _add_cross_correlation_panel(fig, gs[3, 0:2], data, summary)
    _add_ode_readiness_panel(fig, gs[3, 2:4], data, summary)

    # Save
    filepath = output_path / f"dashboard_summary.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Dashboard saved to {filepath}")
    return str(filepath)


def _add_header(fig, gs_spec, summary):
    """Add header with run info and overall grade."""
    ax = fig.add_subplot(gs_spec)
    ax.axis("off")

    # Grade colors
    grade_colors = {
        "A": "#27ae60",
        "B": "#2ecc71",
        "C": "#f39c12",
        "D": "#e67e22",
        "F": "#e74c3c",
    }

    grade = summary.overall_grade.value
    grade_color = grade_colors.get(grade, "#95a5a6")

    # Title
    ax.text(
        0.5, 0.8,
        f"SemiVAE Analysis: {summary.metadata.run_id}",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
    )

    # Subtitle with key info
    subtitle = (
        f"Epochs: {summary.metadata.completed_epochs}/{summary.metadata.max_epochs} | "
        f"z_dim: {summary.metadata.z_dim} | "
        f"Final Loss: {summary.trends.loss_final:.4f}"
    )
    ax.text(
        0.5, 0.55,
        subtitle,
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        color="gray",
    )

    # Grade badge
    grade_text = f"Grade: {grade}"
    ax.text(
        0.5, 0.25,
        grade_text,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        ha="center",
        color=grade_color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=grade_color, alpha=0.2),
    )

    # ODE ready status
    ready_text = "Ready for Neural ODE" if summary.ready_for_ode else "Not Ready for Neural ODE"
    ready_color = "#27ae60" if summary.ready_for_ode else "#e74c3c"
    ax.text(
        0.5, 0.05,
        ready_text,
        transform=ax.transAxes,
        fontsize=14,
        ha="center",
        color=ready_color,
    )


def _add_semantic_r2_panel(fig, gs_spec, data):
    """Add semantic R² panel."""
    ax = fig.add_subplot(gs_spec)

    semantic_df = data.get("semantic_quality")
    if semantic_df is None or semantic_df.empty:
        ax.text(0.5, 0.5, "No semantic data", ha="center", va="center")
        ax.set_title("Semantic R² Over Training")
        return

    colors = {"r2_vol": "#2ecc71", "r2_loc": "#3498db", "r2_shape": "#9b59b6"}
    labels = {"r2_vol": "Volume", "r2_loc": "Location", "r2_shape": "Shape"}
    targets = {"r2_vol": 0.85, "r2_loc": 0.90, "r2_shape": 0.35}

    for col, color in colors.items():
        if col in semantic_df.columns:
            df = semantic_df[["epoch", col]].dropna()
            ax.plot(df["epoch"], df[col], color=color, linewidth=2, label=labels[col])
            ax.axhline(y=targets[col], color=color, linestyle="--", alpha=0.4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("Semantic Encoding Quality")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def _add_reconstruction_panel(fig, gs_spec, data):
    """Add reconstruction quality panel."""
    ax = fig.add_subplot(gs_spec)

    metrics_df = data.get("metrics")
    if metrics_df is None or metrics_df.empty:
        ax.text(0.5, 0.5, "No metrics data", ha="center", va="center")
        ax.set_title("Reconstruction Quality")
        return

    for col, label, color in [
        ("val_epoch/recon", "Validation", "#3498db"),
        ("train_epoch/recon", "Training", "#2ecc71"),
    ]:
        if col in metrics_df.columns:
            df = metrics_df[["epoch", col]].dropna()
            ax.plot(df["epoch"], df[col], color=color, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _add_au_panel(fig, gs_spec, data):
    """Add active units panel."""
    ax = fig.add_subplot(gs_spec)

    au_df = data.get("au_history")
    if au_df is None or au_df.empty:
        ax.text(0.5, 0.5, "No AU data", ha="center", va="center")
        ax.set_title("Active Units")
        return

    if "au_frac" in au_df.columns:
        ax.plot(au_df["epoch"], au_df["au_frac"], "g-", linewidth=2)
        ax.fill_between(au_df["epoch"], 0, au_df["au_frac"], alpha=0.3, color="green")
        ax.axhline(y=0.10, color="r", linestyle="--", alpha=0.7, label="Collapse threshold")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AU Fraction")
    ax.set_title("Active Units (Residual)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _add_partition_variance_panel(fig, gs_spec, data, summary):
    """Add partition variance summary panel."""
    ax = fig.add_subplot(gs_spec)

    # Bar chart of final partition variances
    partitions = ["z_vol", "z_loc", "z_shape", "z_residual"]
    variances = [
        summary.collapse.z_vol_var,
        summary.collapse.z_loc_var,
        summary.collapse.z_shape_var,
        summary.collapse.z_residual_var,
    ]

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#95a5a6"]

    x = np.arange(len(partitions))
    bars = ax.bar(x, variances, color=colors, alpha=0.8)

    # Log scale for better visualization
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("z_", "") for p in partitions])
    ax.set_ylabel("Variance (log scale)")
    ax.set_title("Final Partition Variances")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, variances):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def _add_cross_correlation_panel(fig, gs_spec, data, summary):
    """Add cross-partition correlation panel."""
    ax = fig.add_subplot(gs_spec)

    # Create mini correlation matrix
    corr_matrix = np.array([
        [1.0, summary.ode_utility.corr_vol_loc, summary.ode_utility.corr_vol_shape],
        [summary.ode_utility.corr_vol_loc, 1.0, summary.ode_utility.corr_loc_shape],
        [summary.ode_utility.corr_vol_shape, summary.ode_utility.corr_loc_shape, 1.0],
    ])

    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    labels = ["Vol", "Loc", "Shape"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add values
    for i in range(3):
        for j in range(3):
            color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color=color)

    ax.set_title(f"Cross-Partition Correlations (max: {summary.ode_utility.max_cross_corr:.2f})")
    plt.colorbar(im, ax=ax, shrink=0.8)


def _add_ode_readiness_panel(fig, gs_spec, data, summary):
    """Add ODE readiness summary panel."""
    ax = fig.add_subplot(gs_spec)
    ax.axis("off")

    # Create text summary
    metrics = [
        ("Volume R²", summary.performance.vol_r2, 0.85, summary.ode_utility.vol_ready),
        ("Location R²", summary.performance.loc_r2, 0.90, summary.ode_utility.loc_ready),
        ("Shape R²", summary.performance.shape_r2, 0.35, summary.ode_utility.shape_ready),
        ("Max Cross-Corr", summary.ode_utility.max_cross_corr, 0.30, summary.ode_utility.factors_independent),
        ("Independence Score", summary.ode_utility.independence_score, 0.70, summary.ode_utility.factors_independent),
        ("ODE Readiness", summary.ode_utility.ode_readiness, 0.60, summary.ready_for_ode),
    ]

    y_pos = 0.9
    for name, value, target, passed in metrics:
        color = "#27ae60" if passed else "#e74c3c"
        symbol = "✓" if passed else "✗"

        # For max cross-corr, target is "< 0.30" not ">="
        if "Cross-Corr" in name:
            target_str = f"< {target}"
        else:
            target_str = f">= {target}"

        text = f"{symbol} {name}: {value:.3f} (target: {target_str})"
        ax.text(0.1, y_pos, text, transform=ax.transAxes, fontsize=11, color=color)
        y_pos -= 0.12

    ax.set_title("ODE Readiness Checklist", fontsize=14, fontweight="bold", pad=20)

    # Add recommendations count
    n_rec = len(summary.recommendations)
    n_warn = len(summary.warnings)
    ax.text(
        0.1, 0.05,
        f"Recommendations: {n_rec} | Warnings: {n_warn}",
        transform=ax.transAxes,
        fontsize=10,
        color="gray",
    )
