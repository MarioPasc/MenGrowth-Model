#!/usr/bin/env python
# experiments/lora_ablation/evaluate_probes.py
"""Probe evaluation with GP-based linear and RBF probes.

Supports configurable probe types via config:
- GP-Linear probes (always enabled): equivalent to Ridge regression + uncertainty
- GP-RBF probes (always enabled): nonlinear probe with automatic complexity control

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

import numpy as np
import torch
import yaml

from growth.evaluation.gp_probes import GPSemanticProbes
from growth.evaluation.latent_quality import compute_variance_per_dim
from growth.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_features_and_targets(
    condition_dir: Path,
    feature_level: str = "encoder10",
) -> dict[str, torch.Tensor]:
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

    data["features_probe"] = torch.load(features_probe_path, weights_only=True)
    data["targets_probe"] = torch.load(condition_dir / "targets_probe.pt", weights_only=True)
    data["features_test"] = torch.load(features_test_path, weights_only=True)
    data["targets_test"] = torch.load(condition_dir / "targets_test.pt", weights_only=True)

    logger.info(
        f"Loaded features ({feature_level}): "
        f"probe={data['features_probe'].shape}, "
        f"test={data['features_test'].shape}"
    )

    return data


def evaluate_probes_enhanced(
    condition_name: str,
    config: dict,
    device: str = "cuda",
) -> dict[str, float]:
    """GP probe evaluation with linear + RBF kernels.

    Args:
        condition_name: Condition to evaluate.
        config: Experiment configuration.
        device: Device (for potential GPU acceleration).

    Returns:
        Dict with comprehensive metrics.
    """
    logger.info(f"GP probe evaluation for: {condition_name}")

    # Set up paths
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    # Get probe config
    probe_config = config.get("probe", {})
    feature_level = config.get("feature_extraction", {}).get("level", "encoder10")
    normalize_targets = probe_config.get("normalize_targets", True)
    n_restarts = probe_config.get("n_restarts", 3)
    r2_ci_samples = probe_config.get("r2_ci_samples", 500)

    logger.info(f"Feature level: {feature_level}")
    logger.info(f"Normalize targets: {normalize_targets}")
    logger.info(f"GP restarts: {n_restarts}, CI samples: {r2_ci_samples}")

    # Load data
    data = load_features_and_targets(condition_dir, feature_level)

    # Convert to numpy
    X_probe = data["features_probe"].numpy()
    X_test = data["features_test"].numpy()

    targets_probe = {k: v.numpy() for k, v in data["targets_probe"].items() if k != "all"}
    targets_test = {k: v.numpy() for k, v in data["targets_test"].items() if k != "all"}

    # Create and train GP probes
    logger.info("Training GP probes (linear + RBF)...")
    probes = GPSemanticProbes(
        input_dim=X_probe.shape[1],
        normalize_targets=normalize_targets,
        n_restarts=n_restarts,
        r2_ci_samples=r2_ci_samples,
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
    summary["r2_volume_per_dim_linear"] = results.linear["volume"].r2_per_dim.tolist()
    summary["r2_volume_per_dim_rbf"] = results.rbf["volume"].r2_per_dim.tolist()
    summary["r2_location_per_dim_linear"] = results.linear["location"].r2_per_dim.tolist()
    summary["r2_location_per_dim_rbf"] = results.rbf["location"].r2_per_dim.tolist()
    summary["r2_shape_per_dim_linear"] = results.linear["shape"].r2_per_dim.tolist()
    summary["r2_shape_per_dim_rbf"] = results.rbf["shape"].r2_per_dim.tolist()

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info(f"GP Probe Results for {condition_name}")
    logger.info("=" * 60)
    logger.info("\nGP-Linear Probes:")
    logger.info(f"  R² Volume:   {summary['r2_volume_linear']:.4f}")
    logger.info(f"  R² Location: {summary['r2_location_linear']:.4f}")
    logger.info(f"  R² Shape:    {summary['r2_shape_linear']:.4f}")
    logger.info(f"  R² Mean:     {summary['r2_mean_linear']:.4f}")
    logger.info("\nGP-RBF Probes:")
    logger.info(f"  R² Volume:   {summary['r2_volume_rbf']:.4f}")
    logger.info(f"  R² Location: {summary['r2_location_rbf']:.4f}")
    logger.info(f"  R² Shape:    {summary['r2_shape_rbf']:.4f}")
    logger.info(f"  R² Mean:     {summary['r2_mean_rbf']:.4f}")
    logger.info("\nNonlinearity Evidence (delta LML, RBF - Linear):")
    logger.info(f"  Volume:   {summary['nonlinearity_evidence_volume']:.2f} nats")
    logger.info(f"  Location: {summary['nonlinearity_evidence_location']:.2f} nats")
    logger.info(f"  Shape:    {summary['nonlinearity_evidence_shape']:.2f} nats")
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
    logger.info(f"Saved GP metrics to {metrics_path}")

    # Also update the original metrics.json with key results
    original_metrics = {
        "r2_volume": summary["r2_volume_linear"],
        "r2_location": summary["r2_location_linear"],
        "r2_shape": summary["r2_shape_linear"],
        "r2_mean": summary["r2_mean_linear"],
        "r2_volume_rbf": summary["r2_volume_rbf"],
        "r2_location_rbf": summary["r2_location_rbf"],
        "r2_shape_rbf": summary["r2_shape_rbf"],
        "r2_mean_rbf": summary["r2_mean_rbf"],
        "r2_volume_per_dim": summary["r2_volume_per_dim_linear"],
        "r2_location_per_dim": summary["r2_location_per_dim_linear"],
        "r2_shape_per_dim": summary["r2_shape_per_dim_linear"],
        "mse_volume": summary["mse_volume_linear"],
        "mse_location": summary["mse_location_linear"],
        "mse_shape": summary["mse_shape_linear"],
        "variance_mean": summary["variance_mean"],
        "variance_min": summary["variance_min"],
        "variance_std": summary["variance_std"],
    }
    with open(condition_dir / "metrics.json", "w") as f:
        json.dump(original_metrics, f, indent=2)

    # Save trained probes
    probes_path = condition_dir / "probes_gp.pkl"
    with open(probes_path, "wb") as f:
        pickle.dump(probes, f)
    logger.info(f"Saved GP probes to {probes_path}")

    # Save predictions for visualization
    predictions = {
        "volume": {
            "linear_mean": results.linear["volume"].predictions.tolist(),
            "linear_std": results.linear["volume"].predictive_std.tolist(),
            "rbf_mean": results.rbf["volume"].predictions.tolist(),
            "rbf_std": results.rbf["volume"].predictive_std.tolist(),
            "ground_truth": targets_test["volume"].tolist(),
        },
        "location": {
            "linear_mean": results.linear["location"].predictions.tolist(),
            "linear_std": results.linear["location"].predictive_std.tolist(),
            "rbf_mean": results.rbf["location"].predictions.tolist(),
            "rbf_std": results.rbf["location"].predictive_std.tolist(),
            "ground_truth": targets_test["location"].tolist(),
        },
        "shape": {
            "linear_mean": results.linear["shape"].predictions.tolist(),
            "linear_std": results.linear["shape"].predictive_std.tolist(),
            "rbf_mean": results.rbf["shape"].predictions.tolist(),
            "rbf_std": results.rbf["shape"].predictive_std.tolist(),
            "ground_truth": targets_test["shape"].tolist(),
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
    parser = argparse.ArgumentParser(description="GP probe evaluation with linear and RBF kernels")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
