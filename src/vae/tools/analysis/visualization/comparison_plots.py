"""Comparison visualization plots.

Generates overlay plots for comparing multiple experiment runs.
Produces 11 multi-panel figures for comprehensive scientific comparison.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _get_run_style(run_idx: int, n_runs: int) -> Tuple:
    """Return consistent (color, linestyle, marker) for a run index."""
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_runs, 10)))
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    return colors[run_idx % 10], linestyles[run_idx % 5]


def _get_display_name(run_id: str, name_map: Optional[Dict[str, str]]) -> str:
    """Get display name for a run, falling back to truncated run_id."""
    if name_map and run_id in name_map:
        return name_map[run_id]
    return run_id[:25] + "..." if len(run_id) > 25 else run_id


def plot_comparison(
    run_data: Dict[str, Dict[str, Any]],
    output_dir: str,
    name_map: Optional[Dict[str, str]] = None,
    dpi: int = 150,
    format: str = "png",
) -> Dict[str, str]:
    """Generate all comparison plots for multiple runs.

    Args:
        run_data: Dictionary mapping run_id to loaded experiment data
        output_dir: Directory to save plots
        name_map: Optional mapping from run_id to display name
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

    if name_map is None:
        name_map = {k: k for k in run_data.keys()}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = {}
    n_runs = len(run_data)

    # A1: Loss dynamics
    path = _plot_loss_dynamics(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_loss_dynamics"] = str(path)

    # A2: Schedules
    path = _plot_schedules(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_schedules"] = str(path)

    # B1: R2 detailed (replaces old comparison_r2)
    path = _plot_r2_detailed(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_r2_detailed"] = str(path)

    # B2: Per-feature R2
    path = _plot_per_feature_r2(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_per_feature_r2"] = str(path)

    # B3: Reconstruction quality
    path = _plot_reconstruction(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_reconstruction"] = str(path)

    # C1: Partition health
    path = _plot_partition_health(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_partition_health"] = str(path)

    # C2: Independence
    path = _plot_independence(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_independence"] = str(path)

    # D1: ODE readiness
    path = _plot_ode_readiness(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_ode_readiness"] = str(path)

    # D2: Radar chart
    path = _plot_radar_comparison(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_radar"] = str(path)

    # E1: Summary heatmap table
    path = _plot_summary_heatmap(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_summary_heatmap"] = str(path)

    # E2: Summary ranked table
    path = _plot_summary_ranked(run_data, name_map, output_path, dpi, format)
    if path:
        plots["comparison_summary_ranked"] = str(path)

    return plots


# ---------------------------------------------------------------------------
# A1: Loss Dynamics
# ---------------------------------------------------------------------------

def _plot_loss_dynamics(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot training loss curves comparison (2x3 grid)."""
    loss_cols = [
        ("val_epoch/loss", "Total Loss"),
        ("val_epoch/recon", "Reconstruction"),
        ("val_epoch/kl_raw", "KL Divergence"),
        ("val_epoch/tc", "Total Correlation"),
        ("val_epoch/semantic_total", "Semantic Total"),
        ("val_epoch/cross_partition", "Cross-Partition"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    n_runs = len(run_data)
    has_data = False

    for col_idx, (col, title) in enumerate(loss_cols):
        ax = axes[col_idx]
        for run_idx, (run_id, data) in enumerate(run_data.items()):
            metrics_df = data.get("metrics")
            if metrics_df is None or metrics_df.empty or col not in metrics_df.columns:
                continue
            color, ls = _get_run_style(run_idx, n_runs)
            name = _get_display_name(run_id, name_map)
            df = metrics_df[["epoch", col]].dropna()
            if not df.empty:
                ax.plot(df["epoch"], df[col], color=color, linestyle=ls,
                        linewidth=1.5, label=name, alpha=0.85)
                has_data = True

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    if not has_data:
        plt.close(fig)
        return None

    plt.suptitle("Training Loss Dynamics Comparison", fontsize=14, y=0.98)
    plt.tight_layout()

    filepath = output_path / f"comparison_loss_dynamics.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# A2: Schedules
# ---------------------------------------------------------------------------

def _plot_schedules(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot schedule (beta + lambdas) comparison."""
    fig, (ax_beta, ax_lambda) = plt.subplots(1, 2, figsize=(16, 5))
    n_runs = len(run_data)
    has_data = False

    # Left: Beta annealing
    for run_idx, (run_id, data) in enumerate(run_data.items()):
        metrics_df = data.get("metrics")
        if metrics_df is None or metrics_df.empty:
            continue
        color, ls = _get_run_style(run_idx, n_runs)
        name = _get_display_name(run_id, name_map)
        if "sched/beta" in metrics_df.columns:
            df = metrics_df[["epoch", "sched/beta"]].dropna()
            if not df.empty:
                ax_beta.plot(df["epoch"], df["sched/beta"], color=color,
                             linestyle=ls, linewidth=1.5, label=name, alpha=0.85)
                has_data = True

    ax_beta.set_xlabel("Epoch")
    ax_beta.set_ylabel("Beta")
    ax_beta.set_title("Beta Annealing Schedule")
    ax_beta.legend(fontsize=8)
    ax_beta.grid(True, alpha=0.3)

    # Right: Lambda schedules
    lambda_cols = [
        ("sched/lambda_vol", "vol", "tab:blue"),
        ("sched/lambda_loc", "loc", "tab:orange"),
        ("sched/lambda_shape", "shape", "tab:green"),
        ("sched/lambda_tc", "tc", "tab:red"),
        ("sched/lambda_cross_partition", "cross_part", "tab:purple"),
    ]

    for run_idx, (run_id, data) in enumerate(run_data.items()):
        metrics_df = data.get("metrics")
        if metrics_df is None or metrics_df.empty:
            continue
        _, ls = _get_run_style(run_idx, n_runs)
        name = _get_display_name(run_id, name_map)

        for col, label, color in lambda_cols:
            if col in metrics_df.columns:
                df = metrics_df[["epoch", col]].dropna()
                if not df.empty and df[col].abs().sum() > 0:
                    display_label = f"{label} ({name})" if run_idx == 0 else f"_{label} ({name})"
                    ax_lambda.plot(df["epoch"], df[col], color=color,
                                   linestyle=ls, linewidth=1.2, alpha=0.7,
                                   label=f"{label} [{name}]")
                    has_data = True

    ax_lambda.set_xlabel("Epoch")
    ax_lambda.set_ylabel("Lambda")
    ax_lambda.set_title("Semantic Lambda Schedules")
    ax_lambda.legend(fontsize=6, ncol=2, loc="upper left")
    ax_lambda.grid(True, alpha=0.3)

    if not has_data:
        plt.close(fig)
        return None

    plt.tight_layout()
    filepath = output_path / f"comparison_schedules.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# B1: R2 Detailed
# ---------------------------------------------------------------------------

def _plot_r2_detailed(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot R2 and MSE comparison (2x3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    n_runs = len(run_data)
    has_data = False

    r2_info = [
        ("z_vol", "Volume R\u00b2", 0.85),
        ("z_loc", "Location R\u00b2", 0.90),
        ("z_shape", "Shape R\u00b2", 0.35),
    ]
    mse_info = [
        ("z_vol", "Volume MSE"),
        ("z_loc", "Location MSE"),
        ("z_shape", "Shape MSE"),
    ]

    for col_idx, (partition, title, target) in enumerate(r2_info):
        ax = axes[0, col_idx]
        for run_idx, (run_id, data) in enumerate(run_data.items()):
            semantic_df = data.get("semantic_quality")
            if semantic_df is None or semantic_df.empty:
                continue
            color, ls = _get_run_style(run_idx, n_runs)
            name = _get_display_name(run_id, name_map)

            if "partition" in semantic_df.columns:
                part_data = semantic_df[semantic_df["partition"] == partition]
                if not part_data.empty and "r2" in part_data.columns:
                    df = part_data[["epoch", "r2"]].dropna()
                    if not df.empty:
                        ax.plot(df["epoch"], df["r2"], color=color, linestyle=ls,
                                linewidth=1.5, label=name, alpha=0.85)
                        has_data = True

                        # Annotate first crossing
                        crossing = df[df["r2"] >= target]
                        if not crossing.empty:
                            first_epoch = crossing["epoch"].iloc[0]
                            ax.axvline(x=first_epoch, color=color, alpha=0.3,
                                       linestyle=":", linewidth=1)

        ax.axhline(y=target, color="gray", linestyle="--", alpha=0.5,
                   label=f"Target ({target})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R\u00b2")
        ax.set_title(title)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for col_idx, (partition, title) in enumerate(mse_info):
        ax = axes[1, col_idx]
        for run_idx, (run_id, data) in enumerate(run_data.items()):
            semantic_df = data.get("semantic_quality")
            if semantic_df is None or semantic_df.empty:
                continue
            color, ls = _get_run_style(run_idx, n_runs)
            name = _get_display_name(run_id, name_map)

            if "partition" in semantic_df.columns:
                part_data = semantic_df[semantic_df["partition"] == partition]
                if not part_data.empty and "mse" in part_data.columns:
                    df = part_data[["epoch", "mse"]].dropna()
                    if not df.empty:
                        ax.plot(df["epoch"], df["mse"], color=color, linestyle=ls,
                                linewidth=1.5, label=name, alpha=0.85)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    if not has_data:
        plt.close(fig)
        return None

    plt.suptitle("Semantic Supervision: R\u00b2 and MSE Comparison", fontsize=14, y=0.98)
    plt.tight_layout()
    filepath = output_path / f"comparison_r2_detailed.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# B2: Per-Feature R2
# ---------------------------------------------------------------------------

def _plot_per_feature_r2(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot per-feature R2 comparison (3x4 grid)."""
    # Expected features
    features = [
        "vol_total", "vol_ncr", "vol_ed", "vol_et",
        "loc_x", "loc_y", "loc_z", "sphericity_et",
        "surface_area_et", "aspect_ratio_xy_et", "aspect_ratio_xz_et", "solidity_et",
    ]

    # Check which runs have semantic_tracking data
    runs_with_tracking = {}
    for run_id, data in run_data.items():
        tracking_df = data.get("semantic_tracking")
        if tracking_df is not None and not tracking_df.empty:
            runs_with_tracking[run_id] = tracking_df

    if not runs_with_tracking:
        logger.info("No runs have semantic_tracking.csv, skipping per-feature R2 plot")
        return None

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    n_runs = len(run_data)
    has_any_data = False

    for feat_idx, feature in enumerate(features):
        ax = axes[feat_idx]

        for run_idx, (run_id, data) in enumerate(run_data.items()):
            tracking_df = data.get("semantic_tracking")
            if tracking_df is None or tracking_df.empty:
                continue

            color, ls = _get_run_style(run_idx, n_runs)
            name = _get_display_name(run_id, name_map)

            # Filter for this feature
            if "feature_name" in tracking_df.columns:
                feat_data = tracking_df[tracking_df["feature_name"] == feature]
                if not feat_data.empty and "r2" in feat_data.columns:
                    df = feat_data[["epoch", "r2"]].dropna()
                    if not df.empty:
                        ax.plot(df["epoch"], df["r2"], color=color, linestyle=ls,
                                linewidth=1.2, label=name, alpha=0.8)
                        has_any_data = True

        ax.set_title(feature.replace("_", " ").title(), fontsize=9)
        ax.set_ylim(-0.5, 1.05)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.2)
        if feat_idx == 0:
            ax.legend(fontsize=6)

    if not has_any_data:
        plt.close(fig)
        return None

    # Note which runs are missing
    missing_runs = set(run_data.keys()) - set(runs_with_tracking.keys())
    if missing_runs:
        missing_names = [_get_display_name(r, name_map) for r in missing_runs]
        fig.text(0.5, 0.01, f"Missing data: {', '.join(missing_names)}",
                 ha="center", fontsize=8, style="italic", color="gray")

    plt.suptitle("Per-Feature R\u00b2 Comparison", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    filepath = output_path / f"comparison_per_feature_r2.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# B3: Reconstruction Quality
# ---------------------------------------------------------------------------

def _plot_reconstruction(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot per-modality SSIM/PSNR comparison (2x2)."""
    modalities = [
        ("t1c", "T1c"),
        ("t1n", "T1n"),
        ("t2f", "T2-FLAIR"),
        ("t2w", "T2w"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    n_runs = len(run_data)
    has_data = False

    for mod_idx, (mod_key, mod_name) in enumerate(modalities):
        ax = axes[mod_idx]
        ax2 = ax.twinx()

        ssim_col = f"val_epoch/ssim_{mod_key}"
        psnr_col = f"val_epoch/psnr_{mod_key}"

        for run_idx, (run_id, data) in enumerate(run_data.items()):
            metrics_df = data.get("metrics")
            if metrics_df is None or metrics_df.empty:
                continue
            color, ls = _get_run_style(run_idx, n_runs)
            name = _get_display_name(run_id, name_map)

            if ssim_col in metrics_df.columns:
                df = metrics_df[["epoch", ssim_col]].dropna()
                if not df.empty:
                    ax.plot(df["epoch"], df[ssim_col], color=color, linestyle=ls,
                            linewidth=1.5, label=f"{name} (SSIM)", alpha=0.85)
                    has_data = True

            if psnr_col in metrics_df.columns:
                df = metrics_df[["epoch", psnr_col]].dropna()
                if not df.empty:
                    ax2.plot(df["epoch"], df[psnr_col], color=color, linestyle=ls,
                             linewidth=1.0, alpha=0.4)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("SSIM", color="black")
        ax2.set_ylabel("PSNR (dB)", color="gray", alpha=0.7)
        ax.set_title(mod_name)
        ax.grid(True, alpha=0.3)
        if mod_idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    if not has_data:
        plt.close(fig)
        return None

    plt.suptitle("Reconstruction Quality: Per-Modality SSIM/PSNR", fontsize=14, y=0.98)
    plt.tight_layout()
    filepath = output_path / f"comparison_reconstruction.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# C1: Partition Health
# ---------------------------------------------------------------------------

def _plot_partition_health(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot per-partition AU, variance, KL comparison (2x2)."""
    partitions = ["z_vol", "z_loc", "z_shape", "z_residual"]
    partition_colors = {"z_vol": "tab:blue", "z_loc": "tab:orange",
                        "z_shape": "tab:green", "z_residual": "tab:red"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    n_runs = len(run_data)
    has_data = False

    # Panel 1: AU count per partition
    ax = axes[0, 0]
    for run_idx, (run_id, data) in enumerate(run_data.items()):
        partition_df = data.get("partition_stats")
        if partition_df is None or partition_df.empty:
            continue
        _, ls = _get_run_style(run_idx, n_runs)
        name = _get_display_name(run_id, name_map)

        for partition in partitions:
            part_data = partition_df[partition_df["partition"] == partition]
            if not part_data.empty and "au_count" in part_data.columns:
                df = part_data[["epoch", "au_count"]].dropna()
                if not df.empty:
                    pcolor = partition_colors[partition]
                    label = f"{partition} [{name}]" if run_idx == 0 else f"_{partition} [{name}]"
                    ax.plot(df["epoch"], df["au_count"], color=pcolor,
                            linestyle=ls, linewidth=1.2, alpha=0.7, label=label)
                    has_data = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Units")
    ax.set_title("AU Count per Partition")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Mean variance per partition
    ax = axes[0, 1]
    for run_idx, (run_id, data) in enumerate(run_data.items()):
        partition_df = data.get("partition_stats")
        if partition_df is None or partition_df.empty:
            continue
        _, ls = _get_run_style(run_idx, n_runs)
        name = _get_display_name(run_id, name_map)

        for partition in partitions:
            part_data = partition_df[partition_df["partition"] == partition]
            if not part_data.empty and "mu_var_mean" in part_data.columns:
                df = part_data[["epoch", "mu_var_mean"]].dropna()
                if not df.empty:
                    pcolor = partition_colors[partition]
                    ax.plot(df["epoch"], df["mu_var_mean"], color=pcolor,
                            linestyle=ls, linewidth=1.2, alpha=0.7,
                            label=f"{partition} [{name}]")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Posterior Variance")
    ax.set_title("Posterior Variance per Partition")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: KL per partition
    ax = axes[1, 0]
    for run_idx, (run_id, data) in enumerate(run_data.items()):
        partition_df = data.get("partition_stats")
        if partition_df is None or partition_df.empty:
            continue
        _, ls = _get_run_style(run_idx, n_runs)
        name = _get_display_name(run_id, name_map)

        for partition in partitions:
            part_data = partition_df[partition_df["partition"] == partition]
            if not part_data.empty and "kl_mean" in part_data.columns:
                df = part_data[["epoch", "kl_mean"]].dropna()
                if not df.empty:
                    pcolor = partition_colors[partition]
                    ax.plot(df["epoch"], df["kl_mean"], color=pcolor,
                            linestyle=ls, linewidth=1.2, alpha=0.7,
                            label=f"{partition} [{name}]")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean KL per Dim")
    ax.set_title("KL Divergence per Partition")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 4: z_residual AU fraction with collapse threshold
    ax = axes[1, 1]
    for run_idx, (run_id, data) in enumerate(run_data.items()):
        partition_df = data.get("partition_stats")
        if partition_df is None or partition_df.empty:
            continue
        color, ls = _get_run_style(run_idx, n_runs)
        name = _get_display_name(run_id, name_map)

        residual = partition_df[partition_df["partition"] == "z_residual"]
        if not residual.empty and "au_frac" in residual.columns:
            df = residual[["epoch", "au_frac"]].dropna()
            if not df.empty:
                ax.plot(df["epoch"], df["au_frac"], color=color,
                        linestyle=ls, linewidth=1.5, label=name, alpha=0.85)

    ax.axhline(y=0.10, color="red", linestyle="--", alpha=0.7, label="Collapse threshold (10%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AU Fraction")
    ax.set_title("z_residual Active Units Fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if not has_data:
        plt.close(fig)
        return None

    plt.suptitle("Latent Partition Health Comparison", fontsize=14, y=0.98)
    plt.tight_layout()
    filepath = output_path / f"comparison_partition_health.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# C2: Independence
# ---------------------------------------------------------------------------

def _plot_independence(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot cross-partition independence comparison (1x3)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    n_runs = len(run_data)
    has_data = False

    # Panel 1: Max cross-correlation over epochs
    ax = axes[0]
    for run_idx, (run_id, data) in enumerate(run_data.items()):
        cross_df = data.get("cross_correlation")
        if cross_df is None or cross_df.empty:
            continue
        color, ls = _get_run_style(run_idx, n_runs)
        name = _get_display_name(run_id, name_map)

        if "partition_i" in cross_df.columns:
            epochs = sorted(cross_df["epoch"].unique())
            max_corrs = []
            for ep in epochs:
                ep_data = cross_df[cross_df["epoch"] == ep]
                corr_col = "abs_correlation" if "abs_correlation" in ep_data.columns else "correlation"
                if corr_col in ep_data.columns:
                    max_corrs.append(ep_data[corr_col].abs().max())
                else:
                    max_corrs.append(np.nan)

            if max_corrs:
                ax.plot(epochs, max_corrs, color=color, linestyle=ls,
                        linewidth=1.5, label=name, alpha=0.85)
                has_data = True

    ax.axhline(y=0.30, color="red", linestyle="--", alpha=0.7, label="Target (<0.30)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max |Correlation|")
    ax.set_title("Max Cross-Partition Correlation")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Individual pair correlations
    ax = axes[1]
    pairs = [("z_vol", "z_loc", "vol-loc"), ("z_vol", "z_shape", "vol-shape"),
             ("z_loc", "z_shape", "loc-shape")]
    pair_colors = ["tab:blue", "tab:orange", "tab:green"]

    for run_idx, (run_id, data) in enumerate(run_data.items()):
        cross_df = data.get("cross_correlation")
        if cross_df is None or cross_df.empty or "partition_i" not in cross_df.columns:
            continue
        _, ls = _get_run_style(run_idx, n_runs)
        name = _get_display_name(run_id, name_map)

        for pair_idx, (p1, p2, pair_label) in enumerate(pairs):
            mask = ((cross_df["partition_i"] == p1) & (cross_df["partition_j"] == p2)) | \
                   ((cross_df["partition_i"] == p2) & (cross_df["partition_j"] == p1))
            pair_data = cross_df[mask].copy()
            if not pair_data.empty:
                corr_col = "abs_correlation" if "abs_correlation" in pair_data.columns else "correlation"
                pair_data = pair_data.sort_values("epoch")
                ax.plot(pair_data["epoch"], pair_data[corr_col].abs(),
                        color=pair_colors[pair_idx], linestyle=ls, linewidth=1.2,
                        alpha=0.7, label=f"{pair_label} [{name}]")

    ax.axhline(y=0.30, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Correlation|")
    ax.set_title("Pairwise Partition Correlations")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: Final-epoch correlation heatmaps side by side
    ax = axes[2]
    run_ids = list(run_data.keys())
    n = len(run_ids)
    width = 0.8 / n

    partition_labels = ["z_vol", "z_loc", "z_shape"]
    bar_data = []

    for run_idx, run_id in enumerate(run_ids):
        cross_df = run_data[run_id].get("cross_correlation")
        if cross_df is None or cross_df.empty or "partition_i" not in cross_df.columns:
            continue

        final_epoch = cross_df["epoch"].max()
        final_data = cross_df[cross_df["epoch"] == final_epoch]
        name = _get_display_name(run_id, name_map)

        for pair_idx, (p1, p2, pair_label) in enumerate(pairs):
            mask = ((final_data["partition_i"] == p1) & (final_data["partition_j"] == p2)) | \
                   ((final_data["partition_i"] == p2) & (final_data["partition_j"] == p1))
            pair_row = final_data[mask]
            if not pair_row.empty:
                corr_col = "abs_correlation" if "abs_correlation" in pair_row.columns else "correlation"
                val = pair_row[corr_col].abs().iloc[0]
                bar_data.append((pair_label, name, val, run_idx))

    if bar_data:
        pair_labels_unique = list(dict.fromkeys(d[0] for d in bar_data))
        x = np.arange(len(pair_labels_unique))
        for run_idx, run_id in enumerate(run_ids):
            name = _get_display_name(run_id, name_map)
            color, _ = _get_run_style(run_idx, n_runs)
            vals = []
            for pl in pair_labels_unique:
                matching = [d[2] for d in bar_data if d[0] == pl and d[1] == name]
                vals.append(matching[0] if matching else 0)
            ax.bar(x + run_idx * width - (n - 1) * width / 2, vals, width,
                   color=color, alpha=0.7, label=name)

        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels_unique)
        ax.axhline(y=0.30, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel("|Correlation|")
        ax.set_title("Final Epoch: Partition Correlations")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    if not has_data:
        plt.close(fig)
        return None

    plt.suptitle("Cross-Partition Independence Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = output_path / f"comparison_independence.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# D1: ODE Readiness
# ---------------------------------------------------------------------------

def _plot_ode_readiness(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot ODE readiness score + milestone timeline (1x2)."""
    from ..statistics.comparison import _get_metric_history, _compute_convergence_epochs

    fig, (ax_score, ax_milestones) = plt.subplots(1, 2, figsize=(16, 6))
    n_runs = len(run_data)
    has_data = False

    # Left: Expanded ODE readiness over epochs
    for run_idx, (run_id, data) in enumerate(run_data.items()):
        ode_history = _get_metric_history(data, "ode_readiness_expanded")
        if len(ode_history) > 0:
            # Get epochs from semantic_quality
            semantic_df = data.get("semantic_quality")
            if semantic_df is not None and "partition" in semantic_df.columns:
                epochs_available = sorted(semantic_df[semantic_df["partition"] == "z_vol"]["epoch"].dropna().unique())
                min_len = min(len(epochs_available), len(ode_history))
                epochs_plot = epochs_available[:min_len]
                ode_plot = ode_history[:min_len]
            else:
                epochs_plot = np.arange(len(ode_history))
                ode_plot = ode_history

            color, ls = _get_run_style(run_idx, n_runs)
            name = _get_display_name(run_id, name_map)
            ax_score.plot(epochs_plot, ode_plot, color=color, linestyle=ls,
                          linewidth=2, label=name, alpha=0.85)
            has_data = True

    # Grade threshold lines
    grade_thresholds = [
        (0.70, "D", "lightcoral"),
        (0.55, "C", "lightyellow"),
        (0.40, "F", "lightgray"),
    ]
    ax_score.axhline(y=0.85, color="green", linestyle="--", alpha=0.5, label="Grade A")
    ax_score.axhline(y=0.70, color="blue", linestyle="--", alpha=0.4, label="Grade B")
    ax_score.axhline(y=0.55, color="orange", linestyle="--", alpha=0.4, label="Grade C")

    ax_score.set_xlabel("Epoch")
    ax_score.set_ylabel("ODE Readiness (Expanded)")
    ax_score.set_title("ODE Readiness Score Over Training")
    ax_score.set_ylim(0, 1.0)
    ax_score.legend(fontsize=7)
    ax_score.grid(True, alpha=0.3)

    # Right: Milestone timeline bar chart
    convergence = _compute_convergence_epochs(run_data)
    milestones = ["vol_r2", "loc_r2", "shape_r2", "max_cross_corr", "residual_au_frac"]
    milestone_labels = ["Vol R\u00b2>0.85", "Loc R\u00b2>0.90", "Shape R\u00b2>0.35",
                        "CrossCorr<0.30", "Resid AU>10%"]
    milestone_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    run_ids = list(run_data.keys())
    y_positions = np.arange(len(run_ids))

    # Find max epoch for normalization
    max_epoch = 0
    for data in run_data.values():
        metrics_df = data.get("metrics")
        if metrics_df is not None and "epoch" in metrics_df.columns:
            max_epoch = max(max_epoch, metrics_df["epoch"].max())
    if max_epoch == 0:
        max_epoch = 1000

    for run_idx, run_id in enumerate(run_ids):
        name = _get_display_name(run_id, name_map)
        run_conv = convergence.get(run_id, {})

        for m_idx, milestone in enumerate(milestones):
            epoch_met = run_conv.get(milestone, -1)
            if epoch_met >= 0:
                ax_milestones.barh(run_idx, epoch_met, left=0, height=0.15,
                                   color=milestone_colors[m_idx], alpha=0.7)
                ax_milestones.plot(epoch_met, run_idx, marker="o",
                                   color=milestone_colors[m_idx], markersize=6)

    ax_milestones.set_yticks(y_positions)
    ax_milestones.set_yticklabels([_get_display_name(r, name_map) for r in run_ids])
    ax_milestones.set_xlabel("Epoch")
    ax_milestones.set_title("Milestone Timeline (first epoch met)")
    ax_milestones.set_xlim(0, max_epoch * 1.1)

    # Legend for milestones
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, alpha=0.7, label=l)
                      for c, l in zip(milestone_colors, milestone_labels)]
    ax_milestones.legend(handles=legend_patches, fontsize=7, loc="lower right")
    ax_milestones.grid(True, alpha=0.3, axis="x")

    if not has_data:
        plt.close(fig)
        return None

    plt.suptitle("Neural ODE Readiness Assessment", fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = output_path / f"comparison_ode_readiness.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# D2: Radar Chart
# ---------------------------------------------------------------------------

def _plot_radar_comparison(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot enhanced radar chart comparing final metrics."""
    from ..statistics.summary import generate_summary

    summaries = {}
    for run_id, data in run_data.items():
        try:
            summaries[run_id] = generate_summary(data)
        except Exception as e:
            logger.warning(f"Failed to generate summary for {run_id}: {e}")

    if len(summaries) < 2:
        return None

    metrics = [
        ("Vol R\u00b2", lambda s: max(0, s.performance.vol_r2)),
        ("Loc R\u00b2", lambda s: max(0, s.performance.loc_r2)),
        ("Shape R\u00b2", lambda s: max(0, s.performance.shape_r2)),
        ("AU Fraction", lambda s: s.collapse.au_frac_residual),
        ("Independence", lambda s: s.ode_utility.independence_score),
        ("Recon (SSIM)", lambda s: s.performance.ssim_mean),
        ("ODE Readiness", lambda s: s.ode_utility.ode_readiness_expanded),
    ]

    # Target values for reference polygon
    targets = [0.85, 0.90, 0.35, 0.10, 0.70, 0.8, 0.70]

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    n_runs = len(summaries)

    # Plot target polygon
    target_values = targets + targets[:1]
    ax.plot(angles, target_values, color="gray", linestyle="--", linewidth=1.5,
            alpha=0.5, label="Target")
    ax.fill(angles, target_values, color="gray", alpha=0.05)

    # Plot each run
    for run_idx, (run_id, summary) in enumerate(summaries.items()):
        values = [func(summary) for _, func in metrics]
        values += values[:1]

        color, _ = _get_run_style(run_idx, n_runs)
        name = _get_display_name(run_id, name_map)
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name for name, _ in metrics], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Run Comparison Radar", size=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=9)

    filepath = output_path / f"comparison_radar.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# E1: Summary Heatmap Table
# ---------------------------------------------------------------------------

def _plot_summary_heatmap(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot heatmap-style summary table."""
    from ..statistics.summary import generate_summary

    summaries = {}
    for run_id, data in run_data.items():
        try:
            summaries[run_id] = generate_summary(data)
        except Exception:
            pass

    if len(summaries) < 2:
        return None

    # Metrics to display
    metric_defs = [
        ("Vol R\u00b2", lambda s: s.performance.vol_r2, True),    # higher is better
        ("Loc R\u00b2", lambda s: s.performance.loc_r2, True),
        ("Shape R\u00b2", lambda s: s.performance.shape_r2, True),
        ("SSIM", lambda s: s.performance.ssim_mean, True),
        ("AU Frac (resid)", lambda s: s.collapse.au_frac_residual, True),
        ("Independence", lambda s: s.ode_utility.independence_score, True),
        ("Max Cross-Corr", lambda s: s.ode_utility.max_cross_corr, False),  # lower is better
        ("ODE Score", lambda s: s.ode_utility.ode_readiness_expanded, True),
    ]

    run_ids = list(summaries.keys())
    n_runs = len(run_ids)
    n_metrics = len(metric_defs)

    # Build data matrix
    data_matrix = np.zeros((n_runs, n_metrics))
    for i, run_id in enumerate(run_ids):
        for j, (_, func, _) in enumerate(metric_defs):
            data_matrix[i, j] = func(summaries[run_id])

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), max(3, n_runs * 0.8 + 2)))

    # Normalize each column for coloring
    for j in range(n_metrics):
        col = data_matrix[:, j]
        col_min, col_max = col.min(), col.max()
        _, _, higher_better = metric_defs[j]

        if col_max > col_min:
            norm_col = (col - col_min) / (col_max - col_min)
        else:
            norm_col = np.ones(n_runs) * 0.5

        if not higher_better:
            norm_col = 1.0 - norm_col

        for i in range(n_runs):
            # Color: green (good) to red (bad)
            r = 1.0 - norm_col[i] * 0.7
            g = 0.3 + norm_col[i] * 0.7
            b = 0.3
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor=(r, g, b, 0.4), edgecolor="white", linewidth=2))
            ax.text(j, i, f"{data_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold" if norm_col[i] > 0.9 else "normal")

    ax.set_xlim(-0.5, n_metrics - 0.5)
    ax.set_ylim(-0.5, n_runs - 0.5)
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([m[0] for m in metric_defs], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_runs))
    ax.set_yticklabels([_get_display_name(r, name_map) for r in run_ids], fontsize=10)
    ax.set_title("Comparison Summary (green=best, red=worst per metric)", fontsize=12, pad=15)
    ax.invert_yaxis()

    plt.tight_layout()
    filepath = output_path / f"comparison_summary_heatmap.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# E2: Summary Ranked Table
# ---------------------------------------------------------------------------

def _plot_summary_ranked(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    format: str,
) -> Optional[Path]:
    """Plot ranked summary table with arrows indicating best."""
    from ..statistics.summary import generate_summary

    summaries = {}
    for run_id, data in run_data.items():
        try:
            summaries[run_id] = generate_summary(data)
        except Exception:
            pass

    if len(summaries) < 2:
        return None

    metric_defs = [
        ("Vol R\u00b2", lambda s: s.performance.vol_r2, True, 0.85),
        ("Loc R\u00b2", lambda s: s.performance.loc_r2, True, 0.90),
        ("Shape R\u00b2", lambda s: s.performance.shape_r2, True, 0.35),
        ("SSIM", lambda s: s.performance.ssim_mean, True, None),
        ("AU Frac", lambda s: s.collapse.au_frac_residual, True, 0.10),
        ("Independence", lambda s: s.ode_utility.independence_score, True, 0.70),
        ("Cross-Corr", lambda s: s.ode_utility.max_cross_corr, False, 0.30),
        ("ODE Score", lambda s: s.ode_utility.ode_readiness_expanded, True, None),
        ("Grade", lambda s: s.overall_grade.value, None, None),
    ]

    run_ids = list(summaries.keys())
    n_runs = len(run_ids)
    n_metrics = len(metric_defs)

    fig, ax = plt.subplots(figsize=(max(12, n_metrics * 1.8), max(3, n_runs * 0.9 + 2.5)))
    ax.set_xlim(-0.5, n_metrics - 0.5)
    ax.set_ylim(-1, n_runs)

    # Header row
    for j, (name, _, _, target) in enumerate(metric_defs):
        header = name
        if target is not None:
            header += f"\n(tgt: {target})"
        ax.text(j, -0.7, header, ha="center", va="center", fontsize=8,
                fontweight="bold", color="navy")

    # Data rows
    for i, run_id in enumerate(run_ids):
        summary = summaries[run_id]
        display_name = _get_display_name(run_id, name_map)

        for j, (_, func, higher_better, target) in enumerate(metric_defs):
            val = func(summary)

            if isinstance(val, str):
                text = val
                color = "black"
            else:
                text = f"{val:.3f}"
                color = "black"

                # Check if meets target
                if target is not None:
                    if higher_better and val >= target:
                        color = "darkgreen"
                    elif not higher_better and val <= target:
                        color = "darkgreen"
                    else:
                        color = "darkred"

                # Bold if best across runs
                if higher_better is not None:
                    all_vals = [func(summaries[r]) for r in run_ids if not isinstance(func(summaries[r]), str)]
                    if all_vals:
                        if higher_better and val == max(all_vals):
                            text = f"*{text}*"
                        elif not higher_better and val == min(all_vals):
                            text = f"*{text}*"

            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold" if "*" in str(text) else "normal")

    # Run name labels on left
    ax.set_yticks(range(n_runs))
    ax.set_yticklabels([_get_display_name(r, name_map) for r in run_ids], fontsize=10)
    ax.set_xticks([])

    # Grid lines
    for i in range(n_runs + 1):
        ax.axhline(y=i - 0.5, color="lightgray", linewidth=0.5)
    for j in range(n_metrics + 1):
        ax.axvline(x=j - 0.5, color="lightgray", linewidth=0.5)

    ax.axhline(y=-0.5, color="navy", linewidth=1.5)
    ax.set_title("Ranked Comparison Table (* = best, green = meets target, red = below target)",
                 fontsize=11, pad=15)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    filepath = output_path / f"comparison_summary_ranked.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


# ---------------------------------------------------------------------------
# Legacy API
# ---------------------------------------------------------------------------

def create_comparison_table(
    run_data: Dict[str, Dict[str, Any]],
    output_dir: str,
    name_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Create comparison table as CSV.

    Args:
        run_data: Dictionary mapping run_id to loaded data
        output_dir: Directory to save table
        name_map: Optional run name mapping

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

    df = create_comparison_dataframe(summaries, name_map=name_map)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / "comparison_table.csv"
    df.to_csv(filepath)

    return str(filepath)
