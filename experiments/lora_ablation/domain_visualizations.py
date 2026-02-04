# experiments/lora_ablation/domain_visualizations.py
"""Domain shift visualizations for LoRA ablation analysis.

Visualizations focused on domain generalization from meningioma to glioma:
- Feature distribution comparisons (KDE plots)
- Domain Dice comparison (per-class degradation)
- Domain retention analysis
- UMAP with domain labels
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

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

logger = logging.getLogger(__name__)

# Color scheme for domains
DOMAIN_COLORS = {
    "meningioma": "#2166ac",  # Blue
    "glioma": "#b2182b",      # Red
}

# Color scheme for conditions
CONDITION_COLORS = {
    "baseline": "#808080",
    "baseline_frozen": "#a0a0a0",
    "lora_r2": "#a6cee3",
    "lora_r4": "#1f78b4",
    "lora_r8": "#33a02c",
    "lora_r16": "#ff7f00",
    "lora_r32": "#e31a1c",
}


def plot_domain_feature_distributions(
    men_features: np.ndarray,
    gli_features: np.ndarray,
    output_path: Path,
    n_dims: int = 6,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Plot feature distribution comparison between domains.

    Shows KDE plots for top variance dimensions, comparing how feature
    distributions differ between meningioma and glioma.

    Args:
        men_features: Meningioma features [N_men, D].
        gli_features: Glioma features [N_gli, D].
        output_path: Path to save figure (without extension).
        n_dims: Number of dimensions to visualize.
        figsize: Figure size.
    """
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        logger.warning("Matplotlib/Seaborn required for distribution plots")
        return

    # Select top-variance dimensions
    combined = np.vstack([men_features, gli_features])
    variances = combined.var(axis=0)
    top_dims = np.argsort(variances)[::-1][:n_dims]

    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, dim in enumerate(top_dims):
        ax = axes[i]

        # Plot KDEs
        sns.kdeplot(
            men_features[:, dim],
            ax=ax,
            color=DOMAIN_COLORS["meningioma"],
            label="Meningioma",
            fill=True,
            alpha=0.3,
        )
        sns.kdeplot(
            gli_features[:, dim],
            ax=ax,
            color=DOMAIN_COLORS["glioma"],
            label="Glioma",
            fill=True,
            alpha=0.3,
        )

        ax.set_xlabel(f"Dimension {dim}")
        ax.set_ylabel("Density")
        ax.set_title(f"Dim {dim} (var={variances[dim]:.2f})")

        if i == 0:
            ax.legend()

    # Hide empty subplots
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Feature Distributions: Meningioma vs Glioma", y=1.02)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = output_path.parent / f"{output_path.stem}.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def plot_domain_dice_comparison(
    dice_men: Dict[str, Dict],
    dice_gli: Dict[str, Dict],
    output_path: Path,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Plot Dice score comparison between domains by condition.

    Shows how segmentation performance degrades when moving from
    meningioma (in-domain) to glioma (out-of-domain).

    Args:
        dice_men: Dict[condition -> {dice_mean, dice_NCR, dice_ED, dice_ET}].
        dice_gli: Dict[condition -> {dice_mean, dice_NCR, dice_ED, dice_ET}].
        output_path: Path to save figure.
        figsize: Figure size.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib required for Dice comparison plots")
        return

    conditions = list(dice_men.keys())
    classes = ["mean", "NCR", "ED", "ET"]

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    for ax, cls in zip(axes, classes):
        key = f"dice_{cls}" if cls != "mean" else "dice_mean"

        men_scores = [dice_men[c].get(key, 0) for c in conditions]
        gli_scores = [dice_gli[c].get(key, 0) for c in conditions]

        x = np.arange(len(conditions))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, men_scores, width,
            label="Meningioma", color=DOMAIN_COLORS["meningioma"], alpha=0.8
        )
        bars2 = ax.bar(
            x + width / 2, gli_scores, width,
            label="Glioma", color=DOMAIN_COLORS["glioma"], alpha=0.8
        )

        ax.set_ylabel("Dice Score")
        ax.set_title(f"{cls.upper() if cls != 'mean' else 'Mean'}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [c.replace("lora_", "r").replace("baseline", "base") for c in conditions],
            rotation=45, ha="right"
        )
        ax.set_ylim(0, 1)

        if ax == axes[0]:
            ax.legend()

    plt.suptitle("Segmentation Performance: Meningioma vs Glioma", y=1.02)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = output_path.parent / f"{output_path.stem}.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def plot_domain_retention(
    dice_men: Dict[str, Dict],
    dice_gli: Dict[str, Dict],
    output_path: Path,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """Plot domain retention ratio by condition.

    Retention = Dice_glioma / Dice_meningioma
    Higher retention indicates better domain generalization.

    Args:
        dice_men: Meningioma Dice scores by condition.
        dice_gli: Glioma Dice scores by condition.
        output_path: Path to save figure.
        figsize: Figure size.
    """
    if not HAS_MATPLOTLIB:
        return

    conditions = list(dice_men.keys())

    # Compute retention ratios
    retention = []
    delta_dice = []
    for c in conditions:
        men = dice_men[c].get("dice_mean", 0)
        gli = dice_gli[c].get("dice_mean", 0)
        retention.append(gli / men if men > 0 else 0)
        delta_dice.append(men - gli)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    x = np.arange(len(conditions))
    colors = [CONDITION_COLORS.get(c, "#808080") for c in conditions]

    # Retention ratio
    bars1 = ax1.bar(x, retention, color=colors, alpha=0.8)
    ax1.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="Perfect retention")
    ax1.set_ylabel("Retention Ratio (GLI / MEN)")
    ax1.set_xlabel("Condition")
    ax1.set_title("Domain Retention")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [c.replace("lora_", "r").replace("baseline", "base") for c in conditions],
        rotation=45, ha="right"
    )

    # Add value labels
    for bar, val in zip(bars1, retention):
        ax1.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8
        )

    # Delta Dice (performance drop)
    bars2 = ax2.bar(x, delta_dice, color=colors, alpha=0.8)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_ylabel("Dice Drop (MEN - GLI)")
    ax2.set_xlabel("Condition")
    ax2.set_title("Performance Degradation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [c.replace("lora_", "r").replace("baseline", "base") for c in conditions],
        rotation=45, ha="right"
    )

    for bar, val in zip(bars2, delta_dice):
        ax2.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = output_path.parent / f"{output_path.stem}.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def plot_domain_umap(
    men_features: np.ndarray,
    gli_features: np.ndarray,
    output_path: Path,
    condition_name: str = "",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_samples: int = 300,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """Plot UMAP embedding colored by domain.

    Args:
        men_features: Meningioma features.
        gli_features: Glioma features.
        output_path: Path to save figure.
        condition_name: Condition name for title.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        max_samples: Max samples per domain.
        figsize: Figure size.
    """
    if not HAS_MATPLOTLIB or not HAS_UMAP:
        logger.warning("Matplotlib and UMAP required for domain UMAP")
        return

    # Subsample
    if len(men_features) > max_samples:
        idx = np.random.choice(len(men_features), max_samples, replace=False)
        men_features = men_features[idx]
    if len(gli_features) > max_samples:
        idx = np.random.choice(len(gli_features), max_samples, replace=False)
        gli_features = gli_features[idx]

    # Combine and fit UMAP
    combined = np.vstack([men_features, gli_features])
    labels = ["meningioma"] * len(men_features) + ["glioma"] * len(gli_features)

    logger.info(f"Fitting UMAP for domain visualization ({len(combined)} samples)...")
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = umap.fit_transform(combined)

    # Split back
    men_emb = embedding[: len(men_features)]
    gli_emb = embedding[len(men_features):]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: colored by domain
    ax1.scatter(
        men_emb[:, 0], men_emb[:, 1],
        c=DOMAIN_COLORS["meningioma"], label="Meningioma",
        s=10, alpha=0.6
    )
    ax1.scatter(
        gli_emb[:, 0], gli_emb[:, 1],
        c=DOMAIN_COLORS["glioma"], label="Glioma",
        s=10, alpha=0.6
    )
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_title("Feature Space by Domain")
    ax1.legend()

    # Right: density comparison
    if HAS_SEABORN:
        sns.kdeplot(
            x=men_emb[:, 0], y=men_emb[:, 1], ax=ax2,
            color=DOMAIN_COLORS["meningioma"], levels=5, alpha=0.5
        )
        sns.kdeplot(
            x=gli_emb[:, 0], y=gli_emb[:, 1], ax=ax2,
            color=DOMAIN_COLORS["glioma"], levels=5, alpha=0.5
        )
    else:
        ax2.scatter(men_emb[:, 0], men_emb[:, 1], c=DOMAIN_COLORS["meningioma"], s=5, alpha=0.3)
        ax2.scatter(gli_emb[:, 0], gli_emb[:, 1], c=DOMAIN_COLORS["glioma"], s=5, alpha=0.3)

    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.set_title("Domain Density Overlap")

    title = "Domain Shift in Feature Space"
    if condition_name:
        title += f" ({condition_name})"
    plt.suptitle(title, y=1.02)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = output_path.parent / f"{output_path.stem}.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def plot_domain_shift_summary(
    metrics_by_condition: Dict[str, Dict],
    output_path: Path,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """Plot comprehensive domain shift summary.

    Three-panel figure showing:
    1. Dice (MEN vs GLI)
    2. Retention ratio
    3. R² comparison

    Args:
        metrics_by_condition: Dict with dice_men, dice_gli, r2 metrics.
        output_path: Path to save figure.
        figsize: Figure size.
    """
    if not HAS_MATPLOTLIB:
        return

    conditions = list(metrics_by_condition.keys())
    x = np.arange(len(conditions))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Dice comparison
    men_dice = [metrics_by_condition[c].get("dice_men", 0) for c in conditions]
    gli_dice = [metrics_by_condition[c].get("dice_gli", 0) for c in conditions]

    width = 0.35
    ax1.bar(x - width / 2, men_dice, width, label="MEN", color=DOMAIN_COLORS["meningioma"])
    ax1.bar(x + width / 2, gli_dice, width, label="GLI", color=DOMAIN_COLORS["glioma"])
    ax1.set_ylabel("Dice Score")
    ax1.set_title("Segmentation Performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("lora_", "r").replace("baseline", "base") for c in conditions], rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Panel 2: Retention ratio
    retention = [g / m if m > 0 else 0 for m, g in zip(men_dice, gli_dice)]
    colors = [CONDITION_COLORS.get(c, "#808080") for c in conditions]
    ax2.bar(x, retention, color=colors)
    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Retention (GLI/MEN)")
    ax2.set_title("Domain Generalization")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("lora_", "r").replace("baseline", "base") for c in conditions], rotation=45, ha="right")

    # Panel 3: R² mean
    r2_mean = [metrics_by_condition[c].get("r2_mean", 0) for c in conditions]
    ax3.bar(x, r2_mean, color=colors)
    ax3.set_ylabel("R² (mean)")
    ax3.set_title("Feature Informativeness")
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace("lora_", "r").replace("baseline", "base") for c in conditions], rotation=45, ha="right")
    ax3.set_ylim(0, 1)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = output_path.parent / f"{output_path.stem}.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def generate_domain_figures(
    output_dir: Path,
    config: dict,
) -> None:
    """Generate all domain shift visualizations.

    Args:
        output_dir: Experiment output directory.
        config: Experiment configuration.
    """
    import torch

    if not HAS_MATPLOTLIB:
        logger.error("Matplotlib required for visualizations")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    conditions = [c["name"] for c in config["conditions"]]

    # Collect metrics
    dice_men = {}
    dice_gli = {}
    metrics_summary = {}

    for cond in conditions:
        cond_dir = output_dir / "conditions" / cond

        # Load Dice scores
        men_path = cond_dir / "test_dice_men.json"
        gli_path = cond_dir / "test_dice_gli.json"

        if men_path.exists():
            with open(men_path) as f:
                dice_men[cond] = json.load(f)
        if gli_path.exists():
            with open(gli_path) as f:
                dice_gli[cond] = json.load(f)

        # Load metrics
        metrics_path = cond_dir / "metrics_enhanced.json"
        if not metrics_path.exists():
            metrics_path = cond_dir / "metrics.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
                metrics_summary[cond] = {
                    "dice_men": dice_men.get(cond, {}).get("dice_mean", 0),
                    "dice_gli": dice_gli.get(cond, {}).get("dice_mean", 0),
                    "r2_mean": m.get("r2_mean", 0),
                }

    # Generate figures
    if dice_men and dice_gli:
        logger.info("Generating domain Dice comparison...")
        plot_domain_dice_comparison(
            dice_men, dice_gli,
            figures_dir / "domain_dice_comparison"
        )

        logger.info("Generating domain retention plot...")
        plot_domain_retention(
            dice_men, dice_gli,
            figures_dir / "domain_retention"
        )

    if metrics_summary:
        logger.info("Generating domain shift summary...")
        plot_domain_shift_summary(
            metrics_summary,
            figures_dir / "domain_shift_summary"
        )

    # Generate UMAP for select conditions
    for cond in ["baseline", "lora_r8"]:
        cond_dir = output_dir / "conditions" / cond
        men_path = cond_dir / "features_meningioma.pt"
        gli_path = cond_dir / "features_glioma.pt"

        if men_path.exists() and gli_path.exists():
            logger.info(f"Generating domain UMAP for {cond}...")
            men_feat = torch.load(men_path).numpy()
            gli_feat = torch.load(gli_path).numpy()

            plot_domain_umap(
                men_feat, gli_feat,
                figures_dir / f"domain_umap_{cond}",
                condition_name=cond
            )

            # Feature distribution comparison
            plot_domain_feature_distributions(
                men_feat, gli_feat,
                figures_dir / f"domain_distributions_{cond}"
            )

    logger.info(f"Domain figures saved to {figures_dir}")
