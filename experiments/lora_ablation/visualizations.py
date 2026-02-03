#!/usr/bin/env python
# experiments/lora_ablation/visualizations.py
"""Visualizations for LoRA ablation analysis.

Features:
- UMAP colored by semantic features (volume, location)
- Variance per dimension plots
- Prediction vs ground truth scatter plots
- Linear vs MLP R² comparison
- Publication-quality figures

Usage:
    python -m experiments.lora_ablation.visualizations \
        --config experiments/lora_ablation/config/ablation.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette for conditions
CONDITION_COLORS = {
    'baseline': '#808080',  # Gray
    'lora_r2': '#a6cee3',   # Light blue
    'lora_r4': '#1f78b4',   # Blue
    'lora_r8': '#33a02c',   # Green
    'lora_r16': '#ff7f00',  # Orange
    'lora_r32': '#e31a1c',  # Red
}


def load_condition_data(
    condition_dir: Path,
    feature_level: str = "encoder10",
) -> Dict:
    """Load all data for a condition."""
    data = {}

    # Load features
    feat_path = condition_dir / f"features_test_{feature_level}.pt"
    if not feat_path.exists():
        feat_path = condition_dir / "features_test.pt"
    if feat_path.exists():
        data['features'] = torch.load(feat_path).numpy()

    # Load targets
    tgt_path = condition_dir / "targets_test.pt"
    if tgt_path.exists():
        targets = torch.load(tgt_path)
        data['targets'] = {k: v.numpy() for k, v in targets.items() if k != 'all'}

    # Load metrics
    metrics_path = condition_dir / "metrics_enhanced.json"
    if not metrics_path.exists():
        metrics_path = condition_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data['metrics'] = json.load(f)

    # Load predictions
    pred_path = condition_dir / "predictions_enhanced.json"
    if pred_path.exists():
        with open(pred_path) as f:
            data['predictions'] = json.load(f)

    return data


def plot_r2_comparison_enhanced(
    metrics_by_condition: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Plot R² comparison with both linear and MLP probes."""
    conditions = list(metrics_by_condition.keys())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    feature_types = ['volume', 'location', 'shape']
    titles = ['Volume R²', 'Location R²', 'Shape R²']

    for ax, feat, title in zip(axes, feature_types, titles):
        linear_r2 = []
        mlp_r2 = []

        for cond in conditions:
            m = metrics_by_condition[cond]
            linear_r2.append(m.get(f'r2_{feat}_linear', m.get(f'r2_{feat}', 0)))
            mlp_r2.append(m.get(f'r2_{feat}_mlp', 0))

        x = np.arange(len(conditions))
        width = 0.35

        bars1 = ax.bar(x - width/2, linear_r2, width, label='Linear',
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, mlp_r2, width, label='MLP',
                      color='darkorange', alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('R²')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('lora_', 'r=').replace('baseline', 'base')
                          for c in conditions], rotation=45, ha='right')
        ax.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if abs(height) < 10:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height > 0 else -10),
                           textcoords="offset points",
                           ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=7)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = output_dir / f"r2_comparison_enhanced.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def plot_umap_by_semantic(
    data_by_condition: Dict[str, Dict],
    output_dir: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """Plot UMAP colored by semantic features."""
    if not HAS_UMAP:
        logger.warning("UMAP not available. Skipping UMAP visualization.")
        return

    # Select conditions to show
    conditions_to_show = ['baseline', 'lora_r4', 'lora_r8', 'lora_r16']
    conditions_to_show = [c for c in conditions_to_show if c in data_by_condition]

    if not conditions_to_show:
        logger.warning("No conditions with data for UMAP")
        return

    # Prepare data
    all_features = []
    all_volumes = []
    all_condition_labels = []

    for cond in conditions_to_show:
        data = data_by_condition[cond]
        if 'features' not in data or 'targets' not in data:
            continue

        features = data['features'][:100]  # Subsample
        volumes = data['targets']['volume'][:100, 0]  # Total volume

        all_features.append(features)
        all_volumes.append(volumes)
        all_condition_labels.extend([cond] * len(features))

    if not all_features:
        return

    all_features = np.vstack(all_features)
    all_volumes = np.concatenate(all_volumes)

    # Fit UMAP
    logger.info(f"Fitting UMAP on {len(all_features)} samples...")
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = umap.fit_transform(all_features)

    # Plot: by condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: colored by condition
    ax = axes[0]
    for cond in conditions_to_show:
        mask = np.array(all_condition_labels) == cond
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=CONDITION_COLORS.get(cond, 'gray'),
                  label=cond, s=10, alpha=0.6)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Latent Space by Condition')
    ax.legend(markerscale=2)

    # Right: colored by volume
    ax = axes[1]
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=all_volumes, cmap='viridis', s=10, alpha=0.6)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Latent Space by Tumor Volume')
    plt.colorbar(scatter, ax=ax, label='log(Volume+1)')

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = output_dir / f"umap_semantic.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def plot_variance_per_dim(
    data_by_condition: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Plot feature variance per dimension."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for cond, data in data_by_condition.items():
        if 'features' not in data:
            continue

        variance = data['features'].var(axis=0)
        ax.plot(np.sort(variance)[::-1], label=cond,
               color=CONDITION_COLORS.get(cond, 'gray'), alpha=0.8)

    ax.set_xlabel('Dimension (sorted by variance)')
    ax.set_ylabel('Variance')
    ax.set_title('Feature Variance per Dimension')
    ax.set_yscale('log')
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5,
              label='Low variance threshold')
    ax.legend()

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = output_dir / f"variance_per_dim.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def plot_predictions_scatter(
    data_by_condition: Dict[str, Dict],
    output_dir: Path,
    condition: str = "lora_r8",
) -> None:
    """Plot predictions vs ground truth scatter plots."""
    if condition not in data_by_condition:
        logger.warning(f"Condition {condition} not found")
        return

    data = data_by_condition[condition]
    if 'predictions' not in data:
        logger.warning(f"No predictions for {condition}")
        return

    preds = data['predictions']

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    feature_types = ['volume', 'location', 'shape']
    dim_labels = {
        'volume': ['Total', 'NCR', 'ED', 'ET'],
        'location': ['X', 'Y', 'Z'],
        'shape': ['Sphericity', 'Surface', 'Solidity'],
    }

    for col, feat in enumerate(feature_types):
        gt = np.array(preds[feat]['ground_truth'])
        linear_pred = np.array(preds[feat]['linear'])
        mlp_pred = np.array(preds[feat]['mlp'])

        # Use first dimension for visualization
        dim_idx = 0

        # Linear
        ax = axes[0, col]
        ax.scatter(gt[:, dim_idx], linear_pred[:, dim_idx], alpha=0.3, s=10)
        lims = [min(gt[:, dim_idx].min(), linear_pred[:, dim_idx].min()),
               max(gt[:, dim_idx].max(), linear_pred[:, dim_idx].max())]
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel(f'Ground Truth ({dim_labels[feat][dim_idx]})')
        ax.set_ylabel('Linear Prediction')
        ax.set_title(f'{feat.capitalize()} - Linear')

        # MLP
        ax = axes[1, col]
        ax.scatter(gt[:, dim_idx], mlp_pred[:, dim_idx], alpha=0.3, s=10,
                  color='darkorange')
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel(f'Ground Truth ({dim_labels[feat][dim_idx]})')
        ax.set_ylabel('MLP Prediction')
        ax.set_title(f'{feat.capitalize()} - MLP')

    plt.suptitle(f'Predictions vs Ground Truth ({condition})', y=1.02)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = output_dir / f"scatter_{condition}.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def plot_nonlinearity_gap(
    metrics_by_condition: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Plot the nonlinearity gap (MLP R² - Linear R²)."""
    conditions = list(metrics_by_condition.keys())

    feature_types = ['volume', 'location', 'shape']
    gaps = {feat: [] for feat in feature_types}

    for cond in conditions:
        m = metrics_by_condition[cond]
        for feat in feature_types:
            linear = m.get(f'r2_{feat}_linear', m.get(f'r2_{feat}', 0))
            mlp = m.get(f'r2_{feat}_mlp', 0)
            gaps[feat].append(mlp - linear)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(conditions))
    width = 0.25

    for i, feat in enumerate(feature_types):
        ax.bar(x + i * width, gaps[feat], width, label=feat.capitalize())

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Nonlinearity Gap (MLP R² - Linear R²)')
    ax.set_xlabel('Condition')
    ax.set_title('How Much Information is Nonlinearly Encoded?')
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace('lora_', 'r=').replace('baseline', 'base')
                       for c in conditions], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = output_dir / f"nonlinearity_gap.{ext}"
        plt.savefig(path)
        logger.info(f"Saved {path}")

    plt.close()


def generate_all_figures(config: dict) -> None:
    """Generate all enhanced visualizations."""
    if not HAS_PLOTTING:
        logger.error("Matplotlib not available")
        return

    output_dir = Path(config["experiment"]["output_dir"])
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data for all conditions
    conditions = [c["name"] for c in config["conditions"]]
    data_by_condition = {}
    metrics_by_condition = {}

    feature_level = config.get("feature_extraction", {}).get("level", "encoder10")

    for cond in conditions:
        cond_dir = output_dir / "conditions" / cond
        if not cond_dir.exists():
            continue

        data = load_condition_data(cond_dir, feature_level)
        if data:
            data_by_condition[cond] = data
            if 'metrics' in data:
                metrics_by_condition[cond] = data['metrics']

    if not metrics_by_condition:
        logger.warning("No metrics found. Run evaluation first.")
        return

    logger.info(f"Generating figures for {len(data_by_condition)} conditions...")

    # Generate figures
    plot_r2_comparison_enhanced(metrics_by_condition, figures_dir)
    plot_variance_per_dim(data_by_condition, figures_dir)
    plot_nonlinearity_gap(metrics_by_condition, figures_dir)

    # UMAP (if available)
    viz_config = config.get("visualization", {})
    if viz_config.get("color_by_semantic", True):
        plot_umap_by_semantic(
            data_by_condition, figures_dir,
            n_neighbors=viz_config.get("umap_n_neighbors", 15),
            min_dist=viz_config.get("umap_min_dist", 0.1),
        )

    # Scatter plots
    if viz_config.get("show_scatter_plots", True):
        for cond in ['baseline', 'lora_r8', 'lora_r16']:
            if cond in data_by_condition:
                plot_predictions_scatter(data_by_condition, figures_dir, cond)

    logger.info(f"All figures saved to {figures_dir}")


def main(config_path: str) -> None:
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generate_all_figures(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate enhanced visualizations")
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    main(args.config)
