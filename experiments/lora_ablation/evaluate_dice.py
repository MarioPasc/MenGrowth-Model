#!/usr/bin/env python
# experiments/lora_ablation/evaluate_dice.py
"""Test Dice evaluation for LoRA ablation conditions.

This module evaluates segmentation Dice scores on test sets for all trained
conditions, including domain shift evaluation on BraTS-GLI (glioma).

Usage:
    # Evaluate single condition on BraTS-MEN test set
    python -m experiments.lora_ablation.evaluate_dice \
        --config experiments/lora_ablation/config/ablation.yaml \
        --condition lora_r8 \
        --dataset men

    # Evaluate all conditions on both datasets
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        test-dice-all
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from growth.data.bratsmendata import BraTSMENDataset
from growth.losses.segmentation import DiceMetric3Ch
from growth.evaluation.segmentation_metrics import SegmentationEvaluator

from .model_factory import create_ablation_model, get_condition_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestDiceEvaluator:
    """Evaluate segmentation Dice on test sets.

    Supports evaluation on:
    - BraTS-MEN (meningioma): In-domain test set
    - BraTS-GLI (glioma): Out-of-domain for domain shift analysis

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
                state_dict = torch.load(checkpoint_file, map_location=self.device)
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
        subject_ids: List[str],
    ) -> DataLoader:
        """Create data loader for evaluation."""
        dataset = BraTSMENDataset(
            data_root=data_root,
            subject_ids=subject_ids,
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
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.

        Returns:
            Dict with 'dice_mean', 'dice_NCR', 'dice_ED', 'dice_ET', 'dice_std'.
        """
        model.eval()
        all_dice_scores = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch["image"].to(self.device)
            segs = batch["seg"].to(self.device)

            # Forward pass
            if hasattr(model, 'forward_with_semantics'):
                outputs = model.forward_with_semantics(images)
                pred = outputs['logits']
            else:
                pred = model(images)

            # Compute Dice per class
            dice_scores = self.dice_metric(pred, segs)
            all_dice_scores.append(dice_scores.cpu())

        # Aggregate
        dice_tensor = torch.stack(all_dice_scores)  # [N, 3]
        dice_mean_per_sample = dice_tensor.mean(dim=1)  # [N]

        return {
            "dice_mean": float(dice_tensor.mean()),
            "dice_NCR": float(dice_tensor[:, 0].mean()),
            "dice_ED": float(dice_tensor[:, 1].mean()),
            "dice_ET": float(dice_tensor[:, 2].mean()),
            "dice_std": float(dice_mean_per_sample.std()),
            "num_samples": len(dice_tensor),
        }

    def evaluate_condition(
        self,
        condition_name: str,
        config: dict,
        dataset_name: str = "men",
        subject_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate Dice for a single condition on specified dataset.

        Args:
            condition_name: Name of condition (e.g., "lora_r8").
            config: Full experiment configuration.
            dataset_name: "men" for BraTS-MEN, "gli" for BraTS-GLI.
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
        if dataset_name == "men":
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
        elif dataset_name == "gli":
            data_root = config["paths"].get("glioma_root")
            if data_root is None:
                raise ValueError("glioma_root not specified in config paths")
            if subject_ids is None:
                # Use all available glioma subjects for test
                glioma_path = Path(data_root)
                subject_ids = sorted([d.name for d in glioma_path.iterdir() if d.is_dir()])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Use 'men' or 'gli'.")

        logger.info(f"Evaluating {condition_name} on BraTS-{dataset_name.upper()}")
        logger.info(f"  - Data root: {data_root}")
        logger.info(f"  - Subjects: {len(subject_ids)}")

        # Load model
        model = self._load_model(condition_config, training_config, condition_dir)

        # Create dataloader
        dataloader = self._create_dataloader(data_root, subject_ids)

        # Evaluate
        results = self._evaluate_dataset(
            model, dataloader, desc=f"{condition_name} ({dataset_name})"
        )

        logger.info(
            f"  - Dice: {results['dice_mean']:.4f} "
            f"(NCR={results['dice_NCR']:.4f}, ED={results['dice_ED']:.4f}, "
            f"ET={results['dice_ET']:.4f})"
        )

        return results

    def evaluate_all_conditions(
        self,
        config: dict,
        include_glioma: bool = True,
        glioma_test_size: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Evaluate all conditions on BraTS-MEN and optionally BraTS-GLI.

        Args:
            config: Full experiment configuration.
            include_glioma: Whether to evaluate on BraTS-GLI.
            glioma_test_size: Number of glioma subjects to use (None = all).

        Returns:
            Tuple of (men_results, gli_results) dicts mapping condition -> metrics.
        """
        men_results = {}
        gli_results = {}

        # Get glioma subjects if needed
        gli_subjects = None
        if include_glioma:
            glioma_root = config["paths"].get("glioma_root")
            if glioma_root and Path(glioma_root).exists():
                glioma_path = Path(glioma_root)
                gli_subjects = sorted([d.name for d in glioma_path.iterdir() if d.is_dir()])
                if glioma_test_size is not None:
                    gli_subjects = gli_subjects[:glioma_test_size]
                logger.info(f"Glioma test set: {len(gli_subjects)} subjects")
            else:
                logger.warning("glioma_root not found, skipping BraTS-GLI evaluation")
                include_glioma = False

        # Evaluate each condition
        for cond in config["conditions"]:
            condition_name = cond["name"]

            # BraTS-MEN (in-domain)
            try:
                men_results[condition_name] = self.evaluate_condition(
                    condition_name, config, dataset_name="men"
                )
            except Exception as e:
                logger.warning(f"Failed to evaluate {condition_name} on MEN: {e}")
                men_results[condition_name] = {"error": str(e)}

            # BraTS-GLI (domain shift)
            if include_glioma and gli_subjects:
                try:
                    gli_results[condition_name] = self.evaluate_condition(
                        condition_name, config, dataset_name="gli",
                        subject_ids=gli_subjects
                    )
                except Exception as e:
                    logger.warning(f"Failed to evaluate {condition_name} on GLI: {e}")
                    gli_results[condition_name] = {"error": str(e)}

        return men_results, gli_results


def save_dice_results(
    results: Dict[str, float],
    output_path: Path,
) -> None:
    """Save Dice results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved Dice results to {output_path}")


def generate_dice_summary(
    config: dict,
    men_results: Dict[str, Dict],
    gli_results: Optional[Dict[str, Dict]] = None,
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
            row["men_dice_mean"] = m.get("dice_mean")
            row["men_dice_NCR"] = m.get("dice_NCR")
            row["men_dice_ED"] = m.get("dice_ED")
            row["men_dice_ET"] = m.get("dice_ET")
            row["men_dice_std"] = m.get("dice_std")

        # GLI results
        if gli_results and name in gli_results and "error" not in gli_results[name]:
            g = gli_results[name]
            row["gli_dice_mean"] = g.get("dice_mean")
            row["gli_dice_NCR"] = g.get("dice_NCR")
            row["gli_dice_ED"] = g.get("dice_ED")
            row["gli_dice_ET"] = g.get("dice_ET")
            row["gli_dice_std"] = g.get("dice_std")

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def main(
    config_path: str,
    condition: Optional[str] = None,
    dataset: str = "men",
    glioma_test_size: Optional[int] = None,
    device: str = "cuda",
) -> None:
    """Main entry point for Dice evaluation.

    Args:
        config_path: Path to ablation configuration.
        condition: Specific condition to evaluate (None = all).
        dataset: "men", "gli", or "all" for both.
        glioma_test_size: Number of glioma subjects (None = all available).
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
        results = evaluator.evaluate_condition(
            condition, config, dataset_name=dataset
        )

        # Save results
        cond_dir = output_dir / "conditions" / condition
        save_dice_results(results, cond_dir / f"test_dice_{dataset}.json")

    else:
        # Evaluate all conditions
        logger.info("=" * 60)
        logger.info("Test Dice Evaluation (All Conditions)")
        logger.info("=" * 60)

        include_glioma = dataset in ("gli", "all")
        men_results, gli_results = evaluator.evaluate_all_conditions(
            config,
            include_glioma=include_glioma,
            glioma_test_size=glioma_test_size,
        )

        # Save per-condition results
        for cond in config["conditions"]:
            name = cond["name"]
            cond_dir = output_dir / "conditions" / name
            cond_dir.mkdir(parents=True, exist_ok=True)

            if name in men_results and "error" not in men_results[name]:
                save_dice_results(men_results[name], cond_dir / "test_dice_men.json")

            if name in gli_results and "error" not in gli_results[name]:
                save_dice_results(gli_results[name], cond_dir / "test_dice_gli.json")

        # Generate summary CSV
        summary_csv = generate_dice_summary(config, men_results, gli_results)
        summary_path = output_dir / "test_dice_summary.csv"
        with open(summary_path, "w") as f:
            f.write(summary_csv)
        logger.info(f"Saved Dice summary to {summary_path}")

        # Print summary table
        print("\n" + "=" * 60)
        print("TEST DICE SUMMARY")
        print("=" * 60)
        print(f"\n{'Condition':<15} {'MEN Dice':>10} {'GLI Dice':>10} {'Δ (GLI-MEN)':>12}")
        print("-" * 50)
        for cond in config["conditions"]:
            name = cond["name"]
            men_dice = men_results.get(name, {}).get("dice_mean", float("nan"))
            gli_dice = gli_results.get(name, {}).get("dice_mean", float("nan"))
            delta = gli_dice - men_dice if not (np.isnan(men_dice) or np.isnan(gli_dice)) else float("nan")
            print(f"{name:<15} {men_dice:>10.4f} {gli_dice:>10.4f} {delta:>+12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Test Dice for LoRA ablation conditions"
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
        default=None,
        help="Specific condition to evaluate (default: all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["men", "gli", "all"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--glioma-test-size",
        type=int,
        default=200,
        help="Number of glioma subjects for test (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )

    args = parser.parse_args()
    main(args.config, args.condition, args.dataset, args.glioma_test_size, args.device)
