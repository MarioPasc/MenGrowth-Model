#!/usr/bin/env python
# experiments/lora_ablation/evaluate_probes.py
"""Train linear probes and compute R² metrics for a condition.

This script:
1. Loads pre-extracted features (probe_train and test)
2. Trains separate linear probes for volume, location, and shape
3. Evaluates on test set and computes R² scores
4. Saves results and trained probe models

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

from growth.evaluation.latent_quality import (
    LinearProbe,
    SemanticProbes,
    ProbeResults,
    compute_variance_per_dim,
)
from growth.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_features_and_targets(condition_dir: Path) -> Dict[str, torch.Tensor]:
    """Load pre-extracted features and targets.

    Returns:
        Dict with 'features_probe', 'targets_probe', 'features_test', 'targets_test'.
    """
    data = {}

    # Load probe (training) data
    features_probe_path = condition_dir / "features_probe.pt"
    targets_probe_path = condition_dir / "targets_probe.pt"

    if not features_probe_path.exists():
        raise FileNotFoundError(
            f"Features not found at {features_probe_path}. "
            "Run extract_features.py first."
        )

    data["features_probe"] = torch.load(features_probe_path)
    data["targets_probe"] = torch.load(targets_probe_path)

    # Load test data
    features_test_path = condition_dir / "features_test.pt"
    targets_test_path = condition_dir / "targets_test.pt"

    data["features_test"] = torch.load(features_test_path)
    data["targets_test"] = torch.load(targets_test_path)

    logger.info(f"Loaded features: probe={data['features_probe'].shape}, "
                f"test={data['features_test'].shape}")

    return data


def evaluate_probes(
    condition_name: str,
    config: dict,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train and evaluate linear probes for a condition.

    Args:
        condition_name: Name of the condition.
        config: Full experiment configuration.
        device: Device (unused, probes are CPU-based).

    Returns:
        Dict with R² metrics and MSE values.
    """
    logger.info(f"Evaluating probes for condition: {condition_name}")

    # Set up paths
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    # Load features and targets
    data = load_features_and_targets(condition_dir)

    # Convert to numpy
    X_probe = data["features_probe"].numpy()
    X_test = data["features_test"].numpy()

    targets_probe = {k: v.numpy() for k, v in data["targets_probe"].items()}
    targets_test = {k: v.numpy() for k, v in data["targets_test"].items()}

    # Get probe configuration
    probe_config = config.get("probe", {})
    alpha = probe_config.get("alpha", 1.0)

    # Create and train probes
    logger.info("Training linear probes...")
    probes = SemanticProbes(input_dim=X_probe.shape[1], alpha=alpha)
    probes.fit(X_probe, targets_probe)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    results = probes.evaluate(X_test, targets_test)

    # Get summary metrics
    metrics = probes.get_summary(results)

    # Add per-dimension R² for each target type
    for name, res in results.items():
        metrics[f"r2_{name}_per_dim"] = res.r2_per_dim.tolist()
        metrics[f"mse_{name}"] = res.mse

    # Compute feature variance statistics
    variance = compute_variance_per_dim(X_test)
    metrics["variance_mean"] = float(np.mean(variance))
    metrics["variance_min"] = float(np.min(variance))
    metrics["variance_std"] = float(np.std(variance))

    # Log results
    logger.info("\n" + "=" * 50)
    logger.info(f"Linear Probe Results for {condition_name}")
    logger.info("=" * 50)
    logger.info(f"  R² Volume:   {metrics['r2_volume']:.4f}")
    logger.info(f"  R² Location: {metrics['r2_location']:.4f}")
    logger.info(f"  R² Shape:    {metrics['r2_shape']:.4f}")
    logger.info(f"  R² Mean:     {metrics['r2_mean']:.4f}")
    logger.info("-" * 50)
    logger.info(f"  Variance (mean): {metrics['variance_mean']:.4f}")
    logger.info(f"  Variance (min):  {metrics['variance_min']:.4f}")
    logger.info("=" * 50 + "\n")

    # Save metrics
    metrics_path = condition_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save trained probes for later use
    probes_path = condition_dir / "probes.pkl"
    with open(probes_path, "wb") as f:
        pickle.dump(probes, f)
    logger.info(f"Saved probes to {probes_path}")

    # Save detailed predictions for analysis
    predictions = {
        name: res.predictions.tolist()
        for name, res in results.items()
    }
    predictions_path = condition_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions, f)
    logger.info(f"Saved predictions to {predictions_path}")

    return metrics


def main(
    config_path: str,
    condition: str,
) -> None:
    """Main entry point for probe evaluation.

    Args:
        config_path: Path to ablation.yaml.
        condition: Condition name.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Evaluate probes
    evaluate_probes(
        condition_name=condition,
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate linear probes for a condition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["baseline", "lora_r4", "lora_r8", "lora_r16"],
        help="Condition to evaluate",
    )

    args = parser.parse_args()
    main(args.config, args.condition)
