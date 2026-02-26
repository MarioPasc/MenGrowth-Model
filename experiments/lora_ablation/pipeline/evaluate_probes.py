#!/usr/bin/env python
# experiments/lora_ablation/evaluate_probes.py
"""Probe evaluation with linear and MLP probes.

Supports configurable probe types via config:
- Linear probes (always enabled): Ridge regression
- MLP probes (optional via use_mlp_probes): For detecting nonlinear encoding

Usage:
    python -m experiments.lora_ablation.evaluate_probes \
        --config experiments/lora_ablation/config/ablation.yaml \
        --condition lora_r8
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml

from growth.evaluation.enhanced_probes import (
    EnhancedSemanticProbes,
    analyze_feature_quality,
)
from growth.evaluation.latent_quality import compute_variance_per_dim
from growth.utils.seed import set_seed

from .data_splits import load_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_features_and_targets(
    condition_dir: Path,
    feature_level: str = "encoder10",
) -> Dict[str, torch.Tensor]:
    """Load features and targets with multi-scale support.

    Args:
        condition_dir: Directory containing feature files.
        feature_level: 'encoder10', 'multi_scale', or specific layer.

    Returns:
        Dict with features and targets.
    """
    data = {}

    # Try to load specific feature level first
    features_probe_path = condition_dir / f"features_probe_{feature_level}.pt"
    if not features_probe_path.exists():
        # Fall back to default path
        features_probe_path = condition_dir / "features_probe.pt"

    features_test_path = condition_dir / f"features_test_{feature_level}.pt"
    if not features_test_path.exists():
        features_test_path = condition_dir / "features_test.pt"

    if not features_probe_path.exists():
        raise FileNotFoundError(f"Features not found: {features_probe_path}")

    data["features_probe"] = torch.load(features_probe_path)
    data["targets_probe"] = torch.load(condition_dir / "targets_probe.pt")
    data["features_test"] = torch.load(features_test_path)
    data["targets_test"] = torch.load(condition_dir / "targets_test.pt")

    logger.info(f"Loaded features ({feature_level}): "
                f"probe={data['features_probe'].shape}, "
                f"test={data['features_test'].shape}")

    return data


def evaluate_probes_enhanced(
    condition_name: str,
    config: dict,
    device: str = "cuda",
) -> Dict[str, float]:
    """Enhanced probe evaluation with linear + MLP.

    Args:
        condition_name: Condition to evaluate.
        config: Experiment configuration.
        device: Device (for potential GPU acceleration).

    Returns:
        Dict with comprehensive metrics.
    """
    logger.info(f"Enhanced probe evaluation for: {condition_name}")

    # Set up paths
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    # Get probe config
    probe_config = config.get("probe", {})
    feature_level = config.get("feature_extraction", {}).get("level", "encoder10")
    alpha_linear = probe_config.get("alpha_linear", 1.0)
    alpha_mlp = probe_config.get("alpha_mlp", 1e-4)
    hidden_sizes = tuple(probe_config.get("hidden_sizes", [256, 128]))
    normalize_targets = probe_config.get("normalize_targets", True)

    logger.info(f"Feature level: {feature_level}")
    logger.info(f"Normalize targets: {normalize_targets}")

    # Load data
    data = load_features_and_targets(condition_dir, feature_level)

    # Convert to numpy
    X_probe = data["features_probe"].numpy()
    X_test = data["features_test"].numpy()

    targets_probe = {k: v.numpy() for k, v in data["targets_probe"].items()
                    if k != 'all'}
    targets_test = {k: v.numpy() for k, v in data["targets_test"].items()
                   if k != 'all'}

    # Create and train enhanced probes
    logger.info("Training enhanced probes (linear + MLP)...")
    probes = EnhancedSemanticProbes(
        input_dim=X_probe.shape[1],
        alpha_linear=alpha_linear,
        alpha_mlp=alpha_mlp,
        hidden_sizes=hidden_sizes,
        normalize_targets=normalize_targets,
    )
    probes.fit(X_probe, targets_probe)

    # Evaluate
    logger.info("Evaluating on test set...")
    results = probes.evaluate(X_test, targets_test)
    summary = probes.get_summary(results)

    # Additional diagnostics
    variance = compute_variance_per_dim(X_test)
    summary["variance_mean"] = float(np.mean(variance))
    summary["variance_min"] = float(np.min(variance))
    summary["variance_max"] = float(np.max(variance))
    summary["variance_std"] = float(np.std(variance))
    summary["num_low_variance_dims"] = int((variance < 0.01).sum())
    summary["num_collapsed_dims"] = int((variance < 1e-6).sum())

    # Per-dimension details
    summary["r2_volume_per_dim_linear"] = results['volume'].r2_per_dim_linear.tolist()
    summary["r2_volume_per_dim_mlp"] = results['volume'].r2_per_dim_mlp.tolist()
    summary["r2_location_per_dim_linear"] = results['location'].r2_per_dim_linear.tolist()
    summary["r2_location_per_dim_mlp"] = results['location'].r2_per_dim_mlp.tolist()
    summary["r2_shape_per_dim_linear"] = results['shape'].r2_per_dim_linear.tolist()
    summary["r2_shape_per_dim_mlp"] = results['shape'].r2_per_dim_mlp.tolist()

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info(f"Enhanced Probe Results for {condition_name}")
    logger.info("=" * 60)
    logger.info("\nLinear Probes:")
    logger.info(f"  R² Volume:   {summary['r2_volume_linear']:.4f}")
    logger.info(f"  R² Location: {summary['r2_location_linear']:.4f}")
    logger.info(f"  R² Shape:    {summary['r2_shape_linear']:.4f}")
    logger.info(f"  R² Mean:     {summary['r2_mean_linear']:.4f}")
    logger.info("\nMLP Probes:")
    logger.info(f"  R² Volume:   {summary['r2_volume_mlp']:.4f}")
    logger.info(f"  R² Location: {summary['r2_location_mlp']:.4f}")
    logger.info(f"  R² Shape:    {summary['r2_shape_mlp']:.4f}")
    logger.info(f"  R² Mean:     {summary['r2_mean_mlp']:.4f}")
    logger.info("\nNonlinearity Gap (MLP - Linear):")
    logger.info(f"  Volume:   {summary['nonlinearity_gap_volume']:.4f}")
    logger.info(f"  Location: {summary['nonlinearity_gap_location']:.4f}")
    logger.info(f"  Shape:    {summary['nonlinearity_gap_shape']:.4f}")
    logger.info("\nFeature Diagnostics:")
    logger.info(f"  Variance (mean): {summary['variance_mean']:.6f}")
    logger.info(f"  Variance (min):  {summary['variance_min']:.6f}")
    logger.info(f"  Low variance dims (<0.01): {summary['num_low_variance_dims']}")
    logger.info(f"  Collapsed dims (<1e-6): {summary['num_collapsed_dims']}")
    logger.info("=" * 60 + "\n")

    # Save metrics
    metrics_path = condition_dir / "metrics_enhanced.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved enhanced metrics to {metrics_path}")

    # Also update the original metrics.json with key results
    original_metrics = {
        "r2_volume": summary['r2_volume_linear'],
        "r2_location": summary['r2_location_linear'],
        "r2_shape": summary['r2_shape_linear'],
        "r2_mean": summary['r2_mean_linear'],
        "r2_volume_mlp": summary['r2_volume_mlp'],
        "r2_location_mlp": summary['r2_location_mlp'],
        "r2_shape_mlp": summary['r2_shape_mlp'],
        "r2_mean_mlp": summary['r2_mean_mlp'],
        "r2_volume_per_dim": summary['r2_volume_per_dim_linear'],
        "r2_location_per_dim": summary['r2_location_per_dim_linear'],
        "r2_shape_per_dim": summary['r2_shape_per_dim_linear'],
        "mse_volume": summary['mse_volume_linear'],
        "mse_location": summary['mse_location_linear'],
        "mse_shape": summary['mse_shape_linear'],
        "variance_mean": summary['variance_mean'],
        "variance_min": summary['variance_min'],
        "variance_std": summary['variance_std'],
    }
    with open(condition_dir / "metrics.json", "w") as f:
        json.dump(original_metrics, f, indent=2)

    # Save trained probes
    probes_path = condition_dir / "probes_enhanced.pkl"
    with open(probes_path, "wb") as f:
        pickle.dump(probes, f)
    logger.info(f"Saved enhanced probes to {probes_path}")

    # Save predictions for visualization
    predictions = {
        'volume': {
            'linear': results['volume'].predictions_linear.tolist(),
            'mlp': results['volume'].predictions_mlp.tolist(),
            'ground_truth': targets_test['volume'].tolist(),
        },
        'location': {
            'linear': results['location'].predictions_linear.tolist(),
            'mlp': results['location'].predictions_mlp.tolist(),
            'ground_truth': targets_test['location'].tolist(),
        },
        'shape': {
            'linear': results['shape'].predictions_linear.tolist(),
            'mlp': results['shape'].predictions_mlp.tolist(),
            'ground_truth': targets_test['shape'].tolist(),
        },
    }
    with open(condition_dir / "predictions_enhanced.json", "w") as f:
        json.dump(predictions, f)

    return summary


def main(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])

    evaluate_probes_enhanced(
        condition_name=condition,
        config=config,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced probe evaluation with MLP and target normalization"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
