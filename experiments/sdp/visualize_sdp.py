#!/usr/bin/env python
# experiments/sdp/visualize_sdp.py
"""Publication-ready figures for SDP evaluation.

Generates 7 figures for thesis/paper from a completed SDP run:
1. Training curves (loss terms + R² + dCor over epochs)
2. Latent UMAP (3-panel: vol, loc, shape coloring)
3. Disentanglement matrices (Pearson + dCor 4x4 heatmaps)
4. Variance spectrum (per-dim std with partition coloring)
5. Prediction scatter (3-panel pred vs GT)
6. R² comparison bars (self-probe vs cross-probe)
7. DCI importance matrix heatmap

All figures use IEEE publication style with Paul Tol colorblind-safe palettes.

Usage:
    python -m experiments.sdp.visualize_sdp --run-dir outputs/sdp/my_run/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from experiments.sdp.output_paths import load_run_paths
from experiments.utils.settings import (
    CURRICULUM_PHASES,
    LOSS_TERM_COLORS,
    PARTITION_COLORS,
    PARTITION_LABELS,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
)
from growth.evaluation.visualization import save_figure
from growth.models.projection.partition import DEFAULT_PARTITIONS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_csv_metrics(training_dir: Path) -> Optional["pd.DataFrame"]:
    """Load training metrics from CSV logger output.

    Args:
        training_dir: Training log directory.

    Returns:
        DataFrame with metrics, or None if not found.
    """
    import pandas as pd

    # Search for metrics.csv in csv_log subdirectories
    csv_files = list(training_dir.glob("csv_log/version_*/metrics.csv"))
    if not csv_files:
        # Fall back to older layout
        csv_files = list(training_dir.glob("**/metrics.csv"))

    if not csv_files:
        logger.warning(f"No metrics.csv found in {training_dir}")
        return None

    # Use most recent version
    csv_path = sorted(csv_files)[-1]
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def _load_latent_data(latent_dir: Path, split: str = "val") -> dict[str, np.ndarray] | None:
    """Load latent data for a split.

    Args:
        latent_dir: Latent vector directory.
        split: Split name to load.

    Returns:
        Dict with 'z', 'partitions/*', 'targets/*', or None.
    """
    # Try exact match first, then fuzzy
    candidates = list(latent_dir.glob(f"latent_*{split}*.h5"))
    if not candidates:
        logger.warning(f"No latent file found for split '{split}'")
        return None

    h5_path = candidates[0]
    data = {}
    with h5py.File(h5_path, "r") as f:
        data["z"] = np.array(f["z"])
        for grp in ["partitions", "predictions", "targets"]:
            if grp in f:
                for key in f[grp]:
                    data[f"{grp}/{key}"] = np.array(f[grp][key])

    return data


def plot_training_curves(
    training_dir: Path,
    figures_dir: Path,
    curriculum_config: dict | None = None,
) -> None:
    """Figure 1: Training curves — Thesis Fig 4.

    2x3 panel layout showing loss terms, R², dCor, and variance health.

    Args:
        training_dir: Training log directory with CSV.
        figures_dir: Output directory for figures.
        curriculum_config: Curriculum phase boundaries.
    """
    import matplotlib.pyplot as plt

    df = _load_csv_metrics(training_dir)
    if df is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=get_figure_size("double", 0.6))

    # Helper to add curriculum phase markers
    def _add_curriculum_lines(ax: "Axes") -> None:
        if curriculum_config is None:
            return
        for phase_info in CURRICULUM_PHASES.values():
            if phase_info["end"] is not None:
                ax.axvline(
                    x=phase_info["end"],
                    color=phase_info["color"],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=0.8,
                )

    # Helper to safely plot a column
    def _plot_col(ax: "Axes", df: "pd.DataFrame", col: str, **kwargs) -> None:
        if col in df.columns:
            valid = df[["epoch", col]].dropna()
            if len(valid) > 0:
                ax.plot(valid["epoch"], valid[col], **kwargs)

    # (a) Semantic losses
    ax = axes[0, 0]
    for key in ["mse_vol", "mse_loc", "mse_shape"]:
        col = f"train/{key}"
        color = LOSS_TERM_COLORS.get(key, None)
        _plot_col(ax, df, col, label=key, color=color, linewidth=1.0)
    ax.set_title("(a) Semantic losses")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=7)
    _add_curriculum_lines(ax)

    # (b) Regularization losses
    ax = axes[0, 1]
    for key in ["loss_var", "loss_cov", "loss_dcor"]:
        col = f"train/{key}"
        color = LOSS_TERM_COLORS.get(key, None)
        _plot_col(ax, df, col, label=key, color=color, linewidth=1.0)
    ax.set_title("(b) Regularization")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7)
    _add_curriculum_lines(ax)

    # (c) Total loss
    ax = axes[0, 2]
    _plot_col(ax, df, "train/loss_total", label="Train", color=LOSS_TERM_COLORS["loss_total"])
    _plot_col(
        ax, df, "val/loss_total", label="Val", color=LOSS_TERM_COLORS["loss_total"], linestyle="--"
    )
    ax.set_title("(c) Total loss")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7)
    _add_curriculum_lines(ax)

    # (d) R² evolution
    ax = axes[1, 0]
    for key in ["vol", "loc", "shape"]:
        col = f"val/r2_{key}"
        color = PARTITION_COLORS.get(key, None)
        _plot_col(ax, df, col, label=PARTITION_LABELS.get(key, key), color=color)
    ax.axhline(y=0.80, color="#EE6677", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.axhline(y=0.85, color="#4477AA", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.axhline(y=0.30, color="#228833", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.set_title(r"(d) $R^2$ evolution")
    ax.set_ylabel(r"$R^2$")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    _add_curriculum_lines(ax)

    # (e) dCor evolution
    ax = axes[1, 1]
    for name_i, name_j in [("vol", "loc"), ("vol", "shape"), ("loc", "shape")]:
        col = f"val/dcor_{name_i}_{name_j}"
        _plot_col(ax, df, col, label=f"dCor({name_i},{name_j})", linewidth=1.0)
    ax.set_title("(e) dCor evolution")
    ax.set_ylabel("dCor")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    _add_curriculum_lines(ax)

    # (f) Variance health
    ax = axes[1, 2]
    _plot_col(ax, df, "val/effective_rank", label="Effective rank", color="#332288")
    ax2 = ax.twinx()
    _plot_col(ax2, df, "val/min_dim_std", label="Min dim std", color="#EE6677", linestyle="--")
    ax.set_title("(f) Variance health")
    ax.set_ylabel("Effective rank")
    ax2.set_ylabel("Min std", color="#EE6677")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    _add_curriculum_lines(ax)

    fig.tight_layout()
    save_figure(fig, "training_curves", str(figures_dir))
    logger.info("Saved: training_curves.pdf")


def plot_latent_umap(
    latent_data: dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    """Figure 2: Latent UMAP — Thesis Fig 6.

    3-panel UMAP colored by (a) log-volume, (b) centroid z, (c) sphericity.

    Args:
        latent_data: Latent data dict with 'z' and 'targets/*'.
        figures_dir: Output directory.
    """
    import matplotlib.pyplot as plt

    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP figure")
        return

    z = latent_data["z"]

    # Fit UMAP once
    logger.info(f"Fitting UMAP on {len(z)} samples...")
    embedding = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z)

    # Determine color values from targets
    color_configs = []
    if "targets/volume" in latent_data:
        vol = latent_data["targets/volume"]
        color_configs.append(("(a) Log-volume", vol[:, 0], "viridis", "Log-volume"))
    if "targets/location" in latent_data:
        loc = latent_data["targets/location"]
        # Use z-coordinate (index 0 = cz in our convention)
        color_configs.append(("(b) Centroid z", loc[:, 0], "coolwarm", "Centroid z"))
    if "targets/shape" in latent_data:
        shp = latent_data["targets/shape"]
        color_configs.append(("(c) Sphericity", shp[:, 0], "plasma", "Sphericity"))

    if not color_configs:
        logger.warning("No targets found for UMAP coloring, skipping")
        return

    n_panels = len(color_configs)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3.2, 3.0))
    if n_panels == 1:
        axes = [axes]

    for ax, (title, values, cmap, cbar_label) in zip(axes, color_configs):
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=values,
            cmap=cmap,
            s=PLOT_SETTINGS["scatter_size"],
            alpha=PLOT_SETTINGS["scatter_alpha"],
            edgecolors="none",
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig, "latent_umap", str(figures_dir))
    logger.info("Saved: latent_umap.pdf")


def plot_disentanglement_matrices(
    latent_data: dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    """Figure 3: Disentanglement matrices — Thesis Fig 5.

    Side-by-side 4x4 heatmaps: Pearson correlation and distance correlation.

    Args:
        latent_data: Latent data dict with partition arrays.
        figures_dir: Output directory.
    """
    import matplotlib.pyplot as plt

    partition_names = ["vol", "loc", "shape", "residual"]
    partition_data = {}
    for name in partition_names:
        key = f"partitions/{name}"
        if key in latent_data:
            partition_data[name] = latent_data[key]

    if len(partition_data) < 2:
        logger.warning("Not enough partitions for disentanglement matrices")
        return

    names = list(partition_data.keys())
    n = len(names)

    # Pearson: mean absolute cross-partition correlation
    pearson_matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                pearson_matrix[i, j] = 1.0
            else:
                zi, zj = partition_data[ni], partition_data[nj]
                corr = np.corrcoef(np.hstack([zi, zj]).T)
                di = zi.shape[1]
                cross_block = corr[:di, di:]
                pearson_matrix[i, j] = np.abs(cross_block).mean()

    # dCor matrix
    from growth.evaluation.latent_quality import distance_correlation as np_dcor

    dcor_matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                dcor_matrix[i, j] = 1.0
            elif j > i:
                dcor_val = np_dcor(partition_data[ni], partition_data[nj])
                dcor_matrix[i, j] = dcor_val
                dcor_matrix[j, i] = dcor_val

    labels = [PARTITION_LABELS.get(n, n) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size("double", 0.45))

    # Pearson heatmap
    im1 = ax1.imshow(pearson_matrix, cmap="coolwarm", vmin=-0.3, vmax=0.3)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_title("(a) Mean |Pearson|", fontsize=10)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Annotate
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{pearson_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    # dCor heatmap
    im2 = ax2.imshow(dcor_matrix, cmap="Reds", vmin=0, vmax=0.3)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_title("(b) dCor", fontsize=10)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{dcor_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    save_figure(fig, "disentanglement_matrices", str(figures_dir))
    logger.info("Saved: disentanglement_matrices.pdf")


def plot_variance_spectrum(
    latent_data: dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    """Figure 4: Variance spectrum with partition coloring.

    Per-dim std sorted within each partition, colored by partition.

    Args:
        latent_data: Latent data dict with 'z'.
        figures_dir: Output directory.
    """
    import matplotlib.pyplot as plt

    from growth.evaluation.latent_quality import compute_effective_rank

    z = latent_data["z"]
    z_std = z.std(axis=0)

    fig, ax = plt.subplots(figsize=get_figure_size("double", 0.4))

    # Plot sorted std per partition with coloring
    partition_indices = {name: (spec.start, spec.end) for name, spec in DEFAULT_PARTITIONS.items()}

    dim_idx = 0
    for name in ["vol", "loc", "shape", "residual"]:
        start, end = partition_indices[name]
        part_std = np.sort(z_std[start:end])[::-1]
        x_range = np.arange(dim_idx, dim_idx + len(part_std))
        ax.bar(
            x_range,
            part_std,
            color=PARTITION_COLORS[name],
            label=PARTITION_LABELS.get(name, name),
            alpha=0.8,
            width=1.0,
        )
        dim_idx += len(part_std)

    # Threshold lines
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="Threshold 0.3")
    ax.axhline(
        y=0.5, color="orange", linestyle=":", alpha=0.5, linewidth=0.8, label="Threshold 0.5"
    )

    # Annotate effective rank
    eff_rank = compute_effective_rank(z)
    ax.text(
        0.95,
        0.95,
        f"Eff. rank = {eff_rank:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Dimension (grouped by partition)")
    ax.set_ylabel("Std")
    ax.set_title("Per-dimension standard deviation")
    ax.legend(fontsize=7, ncol=3)

    fig.tight_layout()
    save_figure(fig, "variance_spectrum", str(figures_dir))
    logger.info("Saved: variance_spectrum.pdf")


def plot_prediction_scatter(
    latent_data: dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    """Figure 5: Prediction scatter (3-panel pred vs GT).

    Args:
        latent_data: Latent data dict with 'predictions/*' and 'targets/*'.
        figures_dir: Output directory.
    """
    import matplotlib.pyplot as plt

    panels = [
        ("vol", "volume", 0, "Total volume (log)"),
        ("loc", "location", 0, "Centroid z"),
        ("shape", "shape", 0, "Sphericity"),
    ]

    available = []
    for part, tgt, idx, label in panels:
        if f"predictions/{part}" in latent_data and f"targets/{tgt}" in latent_data:
            available.append((part, tgt, idx, label))

    if not available:
        logger.warning("No prediction/target pairs found for scatter plot")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.0, 3.0))
    if n == 1:
        axes = [axes]

    for ax, (part, tgt, idx, label) in zip(axes, available):
        pred = latent_data[f"predictions/{part}"][:, idx]
        gt = latent_data[f"targets/{tgt}"][:, idx]

        color = PARTITION_COLORS.get(part, "steelblue")
        ax.scatter(gt, pred, s=10, alpha=0.4, c=color, edgecolors="none")

        # Identity line
        lims = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)

        # R²
        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - gt.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0
        ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top", fontsize=9)

        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Prediction")
        ax.set_title(label, fontsize=10)

    fig.tight_layout()
    save_figure(fig, "prediction_scatter", str(figures_dir))
    logger.info("Saved: prediction_scatter.pdf")


def plot_r2_comparison(
    eval_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure 6: R² comparison bars — self-probe vs cross-probe.

    Args:
        eval_dir: Evaluation directory with full_metrics.json and cross_probing.json.
        figures_dir: Output directory.
    """
    import matplotlib.pyplot as plt

    full_path = eval_dir / "full_metrics.json"
    cross_path = eval_dir / "cross_probing.json"

    if not full_path.exists() or not cross_path.exists():
        logger.warning("Missing full_metrics.json or cross_probing.json, skipping R² comparison")
        return

    with open(full_path) as f:
        full_metrics = json.load(f)
    with open(cross_path) as f:
        cross_probing = json.load(f)

    partitions = ["vol", "loc", "shape"]
    fig, ax = plt.subplots(figsize=get_figure_size("double", 0.45))

    x = np.arange(len(partitions))
    width = 0.18

    # Self-probe linear
    self_linear = [full_metrics[p]["r2_linear"] for p in partitions]
    ax.bar(
        x - 1.5 * width,
        self_linear,
        width,
        label="Self (Linear)",
        color=[PARTITION_COLORS[p] for p in partitions],
        alpha=0.9,
    )

    # Self-probe MLP
    self_mlp = [full_metrics[p]["r2_mlp"] for p in partitions]
    ax.bar(
        x - 0.5 * width,
        self_mlp,
        width,
        label="Self (MLP)",
        color=[PARTITION_COLORS[p] for p in partitions],
        alpha=0.6,
        edgecolor=[PARTITION_COLORS[p] for p in partitions],
        linewidth=1.2,
    )

    # Best cross-probe (from other partitions, not self)
    best_cross = []
    for tgt in partitions:
        r2_vals = []
        for src in ["vol", "loc", "shape", "residual"]:
            if src != tgt:
                key = f"{src}_to_{tgt}"
                if key in cross_probing:
                    r2_vals.append(cross_probing[key]["r2"])
        best_cross.append(max(r2_vals) if r2_vals else 0.0)

    ax.bar(x + 0.5 * width, best_cross, width, label="Best cross-probe", color="#BBBBBB", alpha=0.8)

    # Residual probe
    residual_r2 = []
    for tgt in partitions:
        key = f"residual_to_{tgt}"
        if key in cross_probing:
            residual_r2.append(cross_probing[key]["r2"])
        else:
            residual_r2.append(0.0)
    ax.bar(
        x + 1.5 * width,
        residual_r2,
        width,
        label="Residual probe",
        color="#DDDDDD",
        alpha=0.8,
        hatch="//",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([PARTITION_LABELS.get(p, p) for p in partitions], fontsize=8)
    ax.set_ylabel(r"$R^2$")
    ax.set_title(r"Self-probe vs cross-probe $R^2$")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    save_figure(fig, "r2_comparison", str(figures_dir))
    logger.info("Saved: r2_comparison.pdf")


def plot_dci_importance(
    eval_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure 7: DCI importance matrix heatmap.

    Args:
        eval_dir: Evaluation directory with dci_scores.json.
        figures_dir: Output directory.
    """
    import matplotlib.pyplot as plt

    dci_path = eval_dir / "dci_scores.json"
    if not dci_path.exists():
        logger.warning("Missing dci_scores.json, skipping DCI importance plot")
        return

    with open(dci_path) as f:
        dci = json.load(f)

    importance = np.array(dci["importance_matrix"])
    n_factors, n_dims = importance.shape

    fig, ax = plt.subplots(figsize=get_figure_size("double", 0.5))

    im = ax.imshow(importance, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="LASSO |coef|")

    # Partition boundaries as vertical lines
    partition_indices = {name: (spec.start, spec.end) for name, spec in DEFAULT_PARTITIONS.items()}
    for name, (start, end) in partition_indices.items():
        if end < n_dims:
            ax.axvline(x=end - 0.5, color="white", linewidth=1.5)

    # Factor labels
    factor_labels = (
        [f"vol_{i}" for i in range(4)]
        + [f"loc_{i}" for i in range(3)]
        + [f"shape_{i}" for i in range(3)]
    )
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels(factor_labels[:n_factors], fontsize=7)

    # Add partition labels at top
    for name, (start, end) in partition_indices.items():
        mid = (start + end) / 2
        if mid < n_dims:
            ax.text(
                mid,
                -0.8,
                name,
                ha="center",
                va="bottom",
                fontsize=7,
                color=PARTITION_COLORS.get(name, "black"),
                fontweight="bold",
            )

    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Target factor")
    ax.set_title(
        f"DCI importance matrix (D={dci['disentanglement']:.2f}, C={dci['completeness']:.2f})"
    )

    fig.tight_layout()
    save_figure(fig, "dci_importance", str(figures_dir))
    logger.info("Saved: dci_importance.pdf")


def main(run_dir: str) -> None:
    """Generate all publication figures for an SDP run.

    Args:
        run_dir: Path to completed SDP run directory.
    """
    paths = load_run_paths(run_dir)
    logger.info(f"Generating figures for: {paths.root}")

    apply_ieee_style()

    # Load latent data (prefer val, fall back to test)
    latent_data = _load_latent_data(paths.latent, "val")
    if latent_data is None:
        latent_data = _load_latent_data(paths.latent, "test")
    if latent_data is None:
        logger.error("No latent data found. Cannot generate figures.")
        return

    logger.info(f"Loaded latent data: z shape = {latent_data['z'].shape}")

    # Figure 1: Training curves
    plot_training_curves(paths.training, paths.figures)

    # Figure 2: Latent UMAP
    plot_latent_umap(latent_data, paths.figures)

    # Figure 3: Disentanglement matrices
    plot_disentanglement_matrices(latent_data, paths.figures)

    # Figure 4: Variance spectrum
    plot_variance_spectrum(latent_data, paths.figures)

    # Figure 5: Prediction scatter
    plot_prediction_scatter(latent_data, paths.figures)

    # Figure 6: R² comparison bars
    plot_r2_comparison(paths.evaluation, paths.figures)

    # Figure 7: DCI importance
    plot_dci_importance(paths.evaluation, paths.figures)

    logger.info("All figures generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SDP (Phase 2)")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.run_dir)
