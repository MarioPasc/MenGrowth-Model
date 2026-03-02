#!/usr/bin/env python
# experiments/lora_ablation/evaluate_dice.py
"""Test Dice evaluation for LoRA ablation conditions.

This module evaluates segmentation Dice scores on the BraTS-MEN test set
for all trained conditions.

Usage:
    # Evaluate single condition on BraTS-MEN test set
    python -m experiments.lora_ablation.evaluate_dice \\
        --config experiments/lora_ablation/config/ablation.yaml \\
        --condition lora_r8

    # Evaluate all conditions
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        test-dice-all
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from growth.data.bratsmendata import BraTSMENDataset
from growth.data.transforms import FEATURE_ROI_SIZE, get_val_transforms
from growth.losses.segmentation import DiceMetric3Ch

from .model_factory import create_ablation_model, get_condition_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestDiceEvaluator:
    """Evaluate segmentation Dice on BraTS-MEN test set.

    Args:
        checkpoint_path: Path to BrainSegFounder checkpoint.
        device: Device to run evaluation on.
        batch_size: Batch size for evaluation.
        num_workers: Number of data loader workers.

    Example:
        >>> evaluator = TestDiceEvaluator(checkpoint_path, device="cuda")
        >>> results = evaluator.evaluate_condition("lora_r8", config, device)
        >>> print(f"BraTS-MEN Dice: {results['dice_mean']:.4f}")
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dice_metric = DiceMetric3Ch()

    def _load_model(
        self,
        condition_config: dict,
        training_config: dict,
        condition_dir: Path,
    ) -> nn.Module:
        """Load trained model from checkpoint."""
        # Create model architecture
        model = create_ablation_model(
            condition_config=condition_config,
            training_config=training_config,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        # Load trained weights (skip for skip_training conditions)
        skip_training = condition_config.get("skip_training", False)
        if not skip_training:
            checkpoint_file = condition_dir / "best_model.pt"
            if not checkpoint_file.exists():
                checkpoint_file = condition_dir / "checkpoint.pt"

            if checkpoint_file.exists():
                state_dict = torch.load(checkpoint_file, map_location=self.device, weights_only=True)
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "decoder_state_dict" in state_dict:
                    # Old format with separate decoder state
                    model.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded weights from {checkpoint_file}")
            else:
                logger.warning(f"No checkpoint found at {condition_dir}")

        model.eval()
        return model

    def _create_dataloader(
        self,
        data_root: str,
        subject_ids: list[str],
        h5_path: str | None = None,
        h5_split: str | None = None,
    ) -> DataLoader:
        """Create data loader for evaluation."""
        if h5_path:
            from growth.data.bratsmendata import BraTSMENDatasetH5
            from growth.data.transforms import get_h5_val_transforms

            dataset = BraTSMENDatasetH5(
                h5_path=h5_path,
                split=h5_split or "test",
                transform=get_h5_val_transforms(roi_size=FEATURE_ROI_SIZE),
                compute_semantic=False,
            )
        else:
            # 192³ center crop for evaluation (100% tumor containment)
            dataset = BraTSMENDataset(
                data_root=data_root,
                subject_ids=subject_ids,
                transform=get_val_transforms(roi_size=FEATURE_ROI_SIZE),
                compute_semantic=False,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @torch.no_grad()
    def _evaluate_dataset(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        desc: str = "Evaluating",
    ) -> dict[str, float]:
        """Evaluate model on a dataset.

        Returns:
            Dict with 'dice_mean', 'dice_TC', 'dice_WT', 'dice_ET', 'dice_std'.
        """
        model.eval()
        all_dice_scores = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, leave=False):
                images = batch["image"].to(self.device)
                segs = batch["seg"].to(self.device)

                # Forward pass (plain forward — no semantic heads needed for eval)
                if hasattr(model, "model"):
                    pred = model.model(images)
                else:
                    pred = model(images)

                # Compute Dice per class
                dice_scores = self.dice_metric(pred, segs)
                all_dice_scores.append(dice_scores.cpu())

        # Aggregate: each element is [B_i, 3], concat to [N_samples, 3]
        dice_tensor = torch.cat(all_dice_scores, dim=0)  # [N_samples, 3]
        dice_mean_per_sample = dice_tensor.mean(dim=1)  # [N_samples]

        return {
            "dice_mean": float(dice_tensor.mean()),
            "dice_TC": float(dice_tensor[:, 0].mean()),
            "dice_WT": float(dice_tensor[:, 1].mean()),
            "dice_ET": float(dice_tensor[:, 2].mean()),
            "dice_std": float(dice_mean_per_sample.std()),
            "num_samples": len(dice_tensor),
        }

    def evaluate_condition(
        self,
        condition_name: str,
        config: dict,
        subject_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate Dice for a single condition on BraTS-MEN test set.

        Args:
            condition_name: Name of condition (e.g., "lora_r8").
            config: Full experiment configuration.
            subject_ids: Optional list of subject IDs (uses test split if None).

        Returns:
            Dict with Dice scores.
        """
        # Get condition config
        condition_config = get_condition_config(config, condition_name)
        training_config = config["training"]
        output_dir = Path(config["experiment"]["output_dir"])
        condition_dir = output_dir / "conditions" / condition_name

        # Determine data root and subjects
        data_root = config["paths"]["data_root"]
        if subject_ids is None:
            # Load splits from output directory
            output_dir = Path(config["experiment"]["output_dir"])
            splits_path = output_dir / "data_splits.json"
            if not splits_path.exists():
                raise FileNotFoundError(
                    f"Splits not found at {splits_path}. Run splits generation first."
                )
            import json

            with open(splits_path) as f:
                splits = json.load(f)
            subject_ids = splits["test"]

        logger.info(f"Evaluating {condition_name} on BraTS-MEN")
        logger.info(f"  - Data root: {data_root}")
        logger.info(f"  - Subjects: {len(subject_ids)}")

        # Load model
        model = self._load_model(condition_config, training_config, condition_dir)

        # Create dataloader
        h5_path = config.get("paths", {}).get("h5_file")
        dataloader = self._create_dataloader(data_root, subject_ids, h5_path=h5_path)

        # Evaluate
        results = self._evaluate_dataset(
            model, dataloader, desc=f"{condition_name}"
        )

        logger.info(
            f"  - Dice: {results['dice_mean']:.4f} "
            f"(TC={results['dice_TC']:.4f}, WT={results['dice_WT']:.4f}, "
            f"ET={results['dice_ET']:.4f})"
        )

        return results

    def evaluate_all_conditions(
        self,
        config: dict,
    ) -> dict[str, dict]:
        """Evaluate all conditions on BraTS-MEN.

        Args:
            config: Full experiment configuration.

        Returns:
            Dict mapping condition name -> Dice metrics.
        """
        men_results = {}

        # Evaluate each condition
        for cond in config["conditions"]:
            condition_name = cond["name"]

            try:
                men_results[condition_name] = self.evaluate_condition(
                    condition_name, config
                )
            except Exception as e:
                logger.warning(f"Failed to evaluate {condition_name}: {e}")
                men_results[condition_name] = {"error": str(e)}

        return men_results


def save_dice_results(
    results: dict[str, float],
    output_path: Path,
) -> None:
    """Save Dice results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved Dice results to {output_path}")


def generate_dice_summary(
    config: dict,
    men_results: dict[str, dict],
) -> str:
    """Generate summary CSV of Dice results.

    Returns:
        CSV string with all Dice metrics.
    """
    import pandas as pd

    rows = []
    for cond in config["conditions"]:
        name = cond["name"]
        row = {"condition": name}

        # MEN results
        if name in men_results and "error" not in men_results[name]:
            m = men_results[name]
            row["dice_mean"] = m.get("dice_mean")
            row["dice_TC"] = m.get("dice_TC")
            row["dice_WT"] = m.get("dice_WT")
            row["dice_ET"] = m.get("dice_ET")
            row["dice_std"] = m.get("dice_std")

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def main(
    config_path: str,
    condition: str | None = None,
    device: str = "cuda",
) -> None:
    """Main entry point for Dice evaluation.

    Args:
        config_path: Path to ablation configuration.
        condition: Specific condition to evaluate (None = all).
        device: Device to run on.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    checkpoint_path = config["paths"]["checkpoint"]

    # Check CUDA
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    evaluator = TestDiceEvaluator(
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=config.get("feature_extraction", {}).get("batch_size", 4),
        num_workers=config["training"]["num_workers"],
    )

    if condition is not None:
        # Evaluate single condition
        results = evaluator.evaluate_condition(condition, config)

        # Save results
        cond_dir = output_dir / "conditions" / condition
        save_dice_results(results, cond_dir / "test_dice_men.json")

    else:
        # Evaluate all conditions
        logger.info("=" * 60)
        logger.info("Test Dice Evaluation (All Conditions)")
        logger.info("=" * 60)

        men_results = evaluator.evaluate_all_conditions(config)

        # Save per-condition results
        for cond in config["conditions"]:
            name = cond["name"]
            cond_dir = output_dir / "conditions" / name
            cond_dir.mkdir(parents=True, exist_ok=True)

            if name in men_results and "error" not in men_results[name]:
                save_dice_results(men_results[name], cond_dir / "test_dice_men.json")

        # Generate summary CSV
        summary_csv = generate_dice_summary(config, men_results)
        summary_path = output_dir / "test_dice_summary.csv"
        with open(summary_path, "w") as f:
            f.write(summary_csv)
        logger.info(f"Saved Dice summary to {summary_path}")

        # Print summary table
        print("\n" + "=" * 60)
        print("TEST DICE SUMMARY")
        print("=" * 60)
        print(f"\n{'Condition':<15} {'Dice':>10}")
        print("-" * 30)
        for cond in config["conditions"]:
            name = cond["name"]
            dice = men_results.get(name, {}).get("dice_mean", float("nan"))
            print(f"{name:<15} {dice:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Test Dice for LoRA ablation conditions")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Specific condition to evaluate (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
