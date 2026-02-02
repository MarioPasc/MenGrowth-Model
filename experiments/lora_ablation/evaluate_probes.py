#!/usr/bin/env python
# experiments/lora_ablation/evaluate_probes.py
"""Train linear probes and compute R² metrics for a condition.

This script:
1. Loads pre-extracted features (probe_train and test)
2. Trains separate linear probes for volume, location, and shape
3. Evaluates on test set and computes R² scores
4. Evaluates Dice scores on test set (segmentation quality)
5. Saves results and trained probe models

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
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from growth.data.bratsmendata import BraTSMENDataset
from growth.data.transforms import get_val_transforms
from growth.evaluation.latent_quality import (
    LinearProbe,
    SemanticProbes,
    ProbeResults,
    compute_variance_per_dim,
)
from growth.losses.segmentation import DiceMetric
from growth.utils.seed import set_seed

from .data_splits import load_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_segmentation_model(
    condition_name: str,
    config: dict,
    device: str = "cuda",
) -> nn.Module:
    """Load trained segmentation model for a condition.

    Args:
        condition_name: Name of the condition (baseline, lora_r4, etc.).
        config: Experiment configuration.
        device: Device to load model on.

    Returns:
        Trained segmentation model in eval mode.
    """
    from growth.models.encoder.swin_loader import load_swin_encoder
    from growth.models.segmentation.seg_head import (
        SegmentationHead,
        LoRASegmentationModel,
    )

    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    # Get condition config
    cond_config = None
    for c in config["conditions"]:
        if c["name"] == condition_name:
            cond_config = c
            break

    if cond_config is None:
        raise ValueError(f"Unknown condition: {condition_name}")

    # Load base encoder
    encoder = load_swin_encoder(config["paths"]["checkpoint"])

    # Create model based on condition
    lora_rank = cond_config.get("lora_rank")

    if lora_rank is None:
        # Baseline: just encoder + segmentation head
        from .train_condition import BaselineSegmentationModel
        model = BaselineSegmentationModel(encoder, out_channels=4)
    else:
        # LoRA model
        lora_alpha = cond_config.get("lora_alpha", lora_rank * 2)
        model = LoRASegmentationModel(
            encoder=encoder,
            out_channels=4,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    # Load checkpoint (with fallback for different naming conventions)
    checkpoint_path = condition_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = condition_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found in {condition_dir}. "
            "Expected 'best_model.pt' or 'checkpoint.pt'."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle checkpoint format: train_condition.py saves {"decoder_state_dict": ..., "epoch": ..., "metrics": ...}
    if isinstance(checkpoint, dict) and "decoder_state_dict" in checkpoint:
        # Load decoder state
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        logger.info(f"Loaded decoder from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")

        # Load LoRA adapter if applicable
        if lora_rank is not None:
            adapter_dir = condition_dir / "adapter"
            if adapter_dir.exists():
                model.encoder.load_lora(adapter_dir)
                logger.info(f"Loaded LoRA adapter from {adapter_dir}")
            else:
                logger.warning(f"LoRA adapter directory not found: {adapter_dir}")
    else:
        # Legacy format: raw state dict
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from {checkpoint_path} (legacy format)")

    model.to(device)
    model.eval()

    return model


def evaluate_test_dice(
    condition_name: str,
    config: dict,
    config_path: str,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate Dice scores on the test set.

    Args:
        condition_name: Name of the condition.
        config: Experiment configuration.
        config_path: Path to config file (for loading splits).
        device: Device to use.

    Returns:
        Dict with test Dice scores: 'test_dice_mean', 'test_dice_NCR', etc.
    """
    logger.info(f"Evaluating test Dice for {condition_name}...")

    # Load model
    try:
        model = load_segmentation_model(condition_name, config, device)
    except FileNotFoundError as e:
        logger.warning(f"Could not load model: {e}")
        return {}

    # Load test split
    splits = load_splits(config_path)
    test_subjects = splits["test"]

    # Create test dataset
    dataset = BraTSMENDataset(
        data_root=config["paths"]["data_root"],
        subject_ids=test_subjects,
        transform=get_val_transforms(),
        compute_semantic=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # Evaluate Dice
    dice_metric = DiceMetric()
    all_dice_scores = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            segs = batch["seg"].to(device)

            # Forward pass
            outputs = model(images)

            # Handle deep supervision (use main output)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            # Compute Dice
            pred = torch.softmax(outputs, dim=1)
            dice_scores = dice_metric(pred, segs)
            all_dice_scores.append(dice_scores.cpu())

    # Aggregate
    dice_tensor = torch.stack(all_dice_scores).mean(dim=0)

    results = {
        "test_dice_mean": dice_tensor.mean().item(),
        "test_dice_NCR": dice_tensor[0].item(),
        "test_dice_ED": dice_tensor[1].item(),
        "test_dice_ET": dice_tensor[2].item(),
    }

    logger.info(f"  Test Dice Mean: {results['test_dice_mean']:.4f}")
    logger.info(f"  Test Dice NCR:  {results['test_dice_NCR']:.4f}")
    logger.info(f"  Test Dice ED:   {results['test_dice_ED']:.4f}")
    logger.info(f"  Test Dice ET:   {results['test_dice_ET']:.4f}")

    return results


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
    config_path: str = "experiments/lora_ablation/config/ablation.yaml",
    device: str = "cuda",
    skip_dice: bool = False,
) -> Dict[str, float]:
    """Train and evaluate linear probes for a condition.

    Args:
        condition_name: Name of the condition.
        config: Full experiment configuration.
        config_path: Path to config file (for loading splits for Dice eval).
        device: Device for Dice evaluation.
        skip_dice: If True, skip test Dice evaluation.

    Returns:
        Dict with R² metrics, MSE values, and test Dice scores.
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

    # Evaluate test Dice (segmentation quality on test set)
    if not skip_dice:
        logger.info("Evaluating test set Dice scores...")
        try:
            dice_metrics = evaluate_test_dice(
                condition_name, config, config_path, device
            )
            metrics.update(dice_metrics)
        except Exception as e:
            logger.warning(f"Test Dice evaluation failed: {e}")
            # Continue without Dice metrics
    else:
        logger.info("Skipping test Dice evaluation (--skip-dice)")

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
    device: str = "cuda",
    skip_dice: bool = False,
) -> None:
    """Main entry point for probe evaluation.

    Args:
        config_path: Path to ablation.yaml.
        condition: Condition name.
        device: Device for Dice evaluation.
        skip_dice: If True, skip test Dice evaluation.
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
        config_path=config_path,
        device=device,
        skip_dice=skip_dice,
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
        choices=["baseline", "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32"],
        help="Condition to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Dice evaluation",
    )
    parser.add_argument(
        "--skip-dice",
        action="store_true",
        help="Skip test Dice evaluation (faster)",
    )

    args = parser.parse_args()
    main(args.config, args.condition, args.device, args.skip_dice)
