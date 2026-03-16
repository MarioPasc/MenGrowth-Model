#!/usr/bin/env python
# experiments/lora/vis/dual_domain_viz.py
"""Visualization suite for dual-domain experiment.

Generates:
- C1: Dual-domain UMAP (domain color, log(volume), centroid z, sphericity)
- C2: Per-domain training curves (val Dice over epochs)
- C3: Variance spectrum (sorted per-dim variance, frozen vs adapted)
- C4: Cross-correlation matrix (768x768 heatmap)
- C5: GP sausage plots (predictions vs ground truth with +/-2sigma)

Usage:
    python -m experiments.lora.vis.dual_domain_viz \
        --config experiments/lora/config/local/dual_domain_v1.yaml
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from experiments.utils.settings import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    PLOT_SETTINGS,
    apply_ieee_style,
)
from growth.evaluation.latent_quality import compute_variance_per_dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Publication DPI for all figures
_DPI = PLOT_SETTINGS["dpi_print"]  # 300


def _load_features(features_dir: Path, domain: str, split: str) -> np.ndarray | None:
    """Load encoder10 features."""
    path = features_dir / f"features_{domain}_{split}_encoder10.pt"
    if not path.exists():
        path = features_dir / f"features_{domain}_{split}.pt"
    if not path.exists():
        return None
    return torch.load(path, weights_only=True).numpy()


def _load_targets(features_dir: Path, domain: str, split: str) -> dict[str, np.ndarray] | None:
    """Load semantic targets."""
    path = features_dir / f"targets_{domain}_{split}.pt"
    if not path.exists():
        return None
    return {k: v.numpy() for k, v in torch.load(path, weights_only=True).items()}


def plot_dual_domain_umap(
    config: dict,
    conditions: list[str],
    output_path: Path,
) -> None:
    """C1: Dual-domain UMAP with domain coloring and semantic overlays.

    Creates 4 panels per condition: domain color, log(volume), centroid z, sphericity.

    Args:
        config: Full experiment config.
        conditions: Conditions to visualize.
        output_path: Output file path.
    """
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP visualization")
        return

    output_dir = Path(config["experiment"]["output_dir"])
    viz_config = config.get("visualization", {})
    n_neighbors = viz_config.get("umap_n_neighbors", 15)
    min_dist = viz_config.get("umap_min_dist", 0.1)

    n_conds = len(conditions)
    fig, axes = plt.subplots(n_conds, 4, figsize=(20, 5 * n_conds))
    if n_conds == 1:
        axes = axes[np.newaxis, :]

    for i, cond_name in enumerate(conditions):
        features_dir = output_dir / "conditions" / cond_name / "features"

        # Load and concat both domains
        men_feat = _load_features(features_dir, "men", "test")
        gli_feat = _load_features(features_dir, "gli", "test")
        men_targets = _load_targets(features_dir, "men", "test")
        gli_targets = _load_targets(features_dir, "gli", "test")

        if men_feat is None or gli_feat is None:
            for j in range(4):
                axes[i, j].text(0.5, 0.5, "No data", ha="center", va="center")
                axes[i, j].set_title(cond_name)
            continue

        all_feat = np.vstack([men_feat, gli_feat])
        n_men = len(men_feat)
        domains = np.array(["MEN"] * n_men + ["GLI"] * len(gli_feat))

        # Semantic targets
        volumes = np.concatenate(
            [
                men_targets["volume"][:, 0] if men_targets else np.array([]),
                gli_targets["volume"][:, 0] if gli_targets else np.array([]),
            ]
        )

        locations_z = np.concatenate(
            [
                men_targets["location"][:, 2] if men_targets else np.array([]),
                gli_targets["location"][:, 2] if gli_targets else np.array([]),
            ]
        )

        shapes_sph = np.concatenate(
            [
                men_targets["shape"][:, 2] if men_targets else np.array([]),
                gli_targets["shape"][:, 2] if gli_targets else np.array([]),
            ]
        )

        # Fit UMAP
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embedding = reducer.fit_transform(all_feat)

        # Panel 1: Domain coloring
        ax = axes[i, 0]
        colors = ["tab:blue" if d == "MEN" else "tab:orange" for d in domains]
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10, alpha=0.6)
        ax.set_title(f"{cond_name} — Domain")
        ax.legend(
            handles=[
                plt.Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="tab:blue", label="MEN"
                ),
                plt.Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="tab:orange", label="GLI"
                ),
            ],
            loc="upper right",
            fontsize=8,
        )

        # Panel 2: log(volume)
        ax = axes[i, 1]
        log_vol = np.log1p(np.abs(volumes)) if len(volumes) > 0 else np.zeros(len(embedding))
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=log_vol, s=10, alpha=0.6, cmap="viridis"
        )
        ax.set_title(f"{cond_name} — log(Volume)")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # Panel 3: Centroid Z
        ax = axes[i, 2]
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=locations_z, s=10, alpha=0.6, cmap="coolwarm"
        )
        ax.set_title(f"{cond_name} — Centroid Z")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # Panel 4: Sphericity
        ax = axes[i, 3]
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=shapes_sph, s=10, alpha=0.6, cmap="plasma"
        )
        ax.set_title(f"{cond_name} — Sphericity")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved UMAP to {output_path}")


def plot_training_curves(
    config: dict,
    output_path: Path,
) -> None:
    """C2: Per-domain validation Dice curves over epochs.

    Args:
        config: Full experiment config.
        output_path: Output file path.
    """
    output_dir = Path(config["experiment"]["output_dir"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["MEN Dice", "GLI Dice", "Combined Dice"]

    for cond in config["conditions"]:
        name = cond["name"]
        log_path = output_dir / "conditions" / name / "training_log.csv"
        if not log_path.exists():
            continue

        df = pd.read_csv(log_path)
        if "epoch" not in df.columns:
            continue

        epochs = df["epoch"]

        # MEN Dice
        if "val_men_dice_mean" in df.columns:
            men_dice = pd.to_numeric(df["val_men_dice_mean"], errors="coerce")
            axes[0].plot(epochs, men_dice, label=name, linewidth=1.5)

        # GLI Dice
        if "val_gli_dice_mean" in df.columns:
            gli_dice = pd.to_numeric(df["val_gli_dice_mean"], errors="coerce")
            axes[1].plot(epochs, gli_dice, label=name, linewidth=1.5)

        # Combined
        if "val_combined_dice_mean" in df.columns:
            comb_dice = pd.to_numeric(df["val_combined_dice_mean"], errors="coerce")
            axes[2].plot(epochs, comb_dice, label=name, linewidth=1.5)

    for ax, title in zip(axes, titles):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def plot_variance_spectrum(
    config: dict,
    conditions: list[str],
    output_path: Path,
) -> None:
    """C3: Sorted per-dimension variance (768 dims) for frozen vs adapted.

    Args:
        config: Full experiment config.
        conditions: Conditions to compare.
        output_path: Output file path.
    """
    output_dir = Path(config["experiment"]["output_dir"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    titles = ["MEN Features", "GLI Features"]

    for domain_idx, domain in enumerate(("men", "gli")):
        ax = axes[domain_idx]

        for cond_name in conditions:
            features_dir = output_dir / "conditions" / cond_name / "features"
            feat = _load_features(features_dir, domain, "test")
            if feat is None:
                continue

            variance = compute_variance_per_dim(feat)
            sorted_var = np.sort(variance)[::-1]
            ax.plot(sorted_var, label=cond_name, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Dimension (sorted)")
        ax.set_ylabel("Variance")
        ax.set_title(titles[domain_idx])
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved variance spectrum to {output_path}")


def plot_correlation_matrix(
    config: dict,
    condition_name: str,
    output_path: Path,
) -> None:
    """C4: 768x768 feature cross-correlation heatmap.

    Args:
        config: Full experiment config.
        condition_name: Condition to visualize.
        output_path: Output file path.
    """
    output_dir = Path(config["experiment"]["output_dir"])
    features_dir = output_dir / "conditions" / condition_name / "features"

    # Use combined MEN+GLI features
    men_feat = _load_features(features_dir, "men", "test")
    gli_feat = _load_features(features_dir, "gli", "test")

    if men_feat is None and gli_feat is None:
        logger.warning(f"No features for correlation matrix: {condition_name}")
        return

    if men_feat is not None and gli_feat is not None:
        all_feat = np.vstack([men_feat, gli_feat])
    elif men_feat is not None:
        all_feat = men_feat
    else:
        all_feat = gli_feat

    corr = np.corrcoef(all_feat.T)  # [768, 768]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.abs(corr), cmap="hot", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"Feature Correlation — {condition_name}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Dimension")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|correlation|")

    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved correlation matrix to {output_path}")


def plot_sausage_plots(
    config: dict,
    condition_name: str,
    output_dir_path: Path,
) -> None:
    """C5: GP prediction vs ground truth with +/-2sigma intervals.

    Args:
        config: Full experiment config.
        condition_name: Condition to visualize.
        output_dir_path: Directory for sausage plot files.
    """
    exp_output = Path(config["experiment"]["output_dir"])
    probes_dir = exp_output / "conditions" / condition_name / "probes"

    for domain in ("men", "gli"):
        probes_path = probes_dir / f"{domain}_probes.json"
        if not probes_path.exists():
            continue

        # Load features and targets for predictions
        features_dir = exp_output / "conditions" / condition_name / "features"
        targets = _load_targets(features_dir, domain, "test")
        if targets is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        semantics = ["volume", "location", "shape"]

        for j, sem in enumerate(semantics):
            ax = axes[j]
            gt = targets[sem]

            # Plot each dimension of the semantic target
            if gt.ndim == 2:
                for dim_idx in range(gt.shape[1]):
                    gt_dim = gt[:, dim_idx]
                    sorted_idx = np.argsort(gt_dim)

                    ax.scatter(
                        range(len(sorted_idx)),
                        gt_dim[sorted_idx],
                        s=5,
                        alpha=0.5,
                        label=f"GT dim{dim_idx}" if dim_idx == 0 else None,
                        color=f"C{dim_idx}",
                    )
            else:
                sorted_idx = np.argsort(gt)
                ax.scatter(range(len(sorted_idx)), gt[sorted_idx], s=5, alpha=0.5, label="GT")

            ax.set_xlabel("Sample (sorted)")
            ax.set_ylabel(sem.capitalize())
            ax.set_title(f"{domain.upper()} — {sem}")
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"{condition_name} — {domain.upper()} Semantic Targets", fontsize=12)
        plt.tight_layout()
        output_path = output_dir_path / f"sausage_{domain}_{condition_name}.png"
        plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".pdf"), dpi=_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved sausage plot to {output_path}")


def generate_all_visualizations(
    config: dict,
) -> None:
    """Generate all visualizations for the experiment.

    Args:
        config: Full experiment configuration.
    """
    apply_ieee_style()

    output_dir = Path(config["experiment"]["output_dir"])
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sausage_dir = figures_dir / "sausage_plots"
    sausage_dir.mkdir(exist_ok=True)

    condition_names = [c["name"] for c in config["conditions"]]

    # C1: UMAP
    try:
        plot_dual_domain_umap(config, condition_names, figures_dir / "umap_dual_domain.png")
    except Exception as e:
        logger.warning(f"UMAP failed: {e}")

    # C2: Training curves
    try:
        plot_training_curves(config, figures_dir / "training_curves.png")
    except Exception as e:
        logger.warning(f"Training curves failed: {e}")

    # C3: Variance spectrum
    try:
        plot_variance_spectrum(config, condition_names, figures_dir / "variance_spectrum.png")
    except Exception as e:
        logger.warning(f"Variance spectrum failed: {e}")

    # C4: Correlation matrix (for primary dual condition)
    for cond_name in condition_names:
        try:
            plot_correlation_matrix(
                config, cond_name, figures_dir / f"correlation_matrix_{cond_name}.png"
            )
        except Exception as e:
            logger.warning(f"Correlation matrix for {cond_name} failed: {e}")

    # C5: Sausage plots
    for cond_name in condition_names:
        try:
            plot_sausage_plots(config, cond_name, sausage_dir)
        except Exception as e:
            logger.warning(f"Sausage plots for {cond_name} failed: {e}")

    logger.info(f"\nAll visualizations saved to {figures_dir}")


def main(config_path: str) -> None:
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generate_all_visualizations(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
