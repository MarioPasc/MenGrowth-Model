#!/usr/bin/env python
# experiments/lora/eval/evaluate_dice.py
"""Segmentation Dice evaluation (single-domain and per-domain).

Supports:
- DualDomainDiceEvaluator: Evaluates on both MEN and GLI test sets
- TestDiceEvaluator: Evaluates on MEN test set only (lora_ablation compat)

Both use domain-aware label conversion (TC/WT/ET).

Usage:
    python -m experiments.lora.run --config <yaml> dice --condition <name>
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from growth.data.bratsmendata import BraTSDatasetH5
from growth.data.transforms import FEATURE_ROI_SIZE, get_h5_val_transforms
from growth.losses.segmentation import DiceMetric3Ch

from ..engine.model_factory import create_ablation_model, get_condition_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dual-Domain Dice Evaluator (primary)
# =============================================================================


class DualDomainDiceEvaluator:
    """Evaluate segmentation Dice on both MEN and GLI test sets.

    Uses domain-aware label conversion to compute per-domain Dice scores.

    Args:
        checkpoint_path: Path to BrainSegFounder checkpoint.
        device: Device.
        batch_size: Evaluation batch size.
        num_workers: DataLoader workers.
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
        model = create_ablation_model(
            condition_config=condition_config,
            training_config=training_config,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        skip_training = condition_config.get("skip_training", False)
        if not skip_training:
            checkpoint_file = condition_dir / "best_model.pt"
            if not checkpoint_file.exists():
                checkpoint_file = condition_dir / "checkpoint.pt"

            if checkpoint_file.exists():
                state_dict = torch.load(
                    checkpoint_file, map_location=self.device, weights_only=True
                )
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded weights from {checkpoint_file}")
            else:
                logger.warning(f"No checkpoint found at {condition_dir}")

        model.eval()
        return model

    def _create_dataloader(
        self,
        h5_path: str,
        h5_split: str = "test",
    ) -> DataLoader:
        """Create evaluation DataLoader for a domain."""
        dataset = BraTSDatasetH5(
            h5_path=h5_path,
            split=h5_split,
            transform=get_h5_val_transforms(roi_size=FEATURE_ROI_SIZE),
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
    def _evaluate_domain(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        domain: str,
        desc: str = "Evaluating",
    ) -> dict[str, float]:
        """Evaluate model on a single domain.

        Args:
            model: Model in eval mode.
            dataloader: Domain-specific DataLoader.
            domain: Domain string ("MEN" or "GLI").
            desc: Progress bar description.

        Returns:
            Dict with per-class and mean Dice scores.
        """
        model.eval()
        all_dice_scores: list[torch.Tensor] = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch["image"].to(self.device)
            segs = batch["seg"].to(self.device)

            if hasattr(model, "model"):
                pred = model.model(images)
            else:
                pred = model(images)

            dice_scores = self.dice_metric(pred, segs, domain=domain)
            all_dice_scores.append(dice_scores.cpu())

        dice_tensor = torch.cat(all_dice_scores, dim=0)  # [N, 3]
        dice_mean_per_sample = dice_tensor.mean(dim=1)  # [N]

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
    ) -> dict[str, dict[str, float]]:
        """Evaluate Dice for a condition on both MEN and GLI test sets.

        Args:
            condition_name: Condition name.
            config: Full experiment configuration.

        Returns:
            Dict with "men" and "gli" sub-dicts of Dice scores.
        """
        condition_config = get_condition_config(config, condition_name)
        training_config = config["training"]
        output_dir = Path(config["experiment"]["output_dir"])
        condition_dir = output_dir / "conditions" / condition_name

        model = self._load_model(condition_config, training_config, condition_dir)

        results: dict[str, dict[str, float]] = {}

        for domain, h5_key in [("men", "men_h5_file"), ("gli", "gli_h5_file")]:
            h5_path = config["paths"].get(h5_key)
            if h5_path is None:
                continue
            logger.info(f"Evaluating {condition_name} on {domain.upper()} test set")

            dataloader = self._create_dataloader(h5_path, "test")
            domain_results = self._evaluate_domain(
                model, dataloader, domain.upper(), desc=f"{condition_name}-{domain.upper()}"
            )

            results[domain] = domain_results
            logger.info(
                f"  {domain.upper()} Dice: {domain_results['dice_mean']:.4f} "
                f"(TC={domain_results['dice_TC']:.4f}, "
                f"WT={domain_results['dice_WT']:.4f}, "
                f"ET={domain_results['dice_ET']:.4f})"
            )

        return results

    def evaluate_all_conditions(
        self,
        config: dict,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Evaluate all conditions.

        Returns:
            Dict: condition_name -> {men: {...}, gli: {...}}.
        """
        all_results: dict[str, dict] = {}

        for cond in config["conditions"]:
            condition_name = cond["name"]
            try:
                all_results[condition_name] = self.evaluate_condition(condition_name, config)
            except Exception as e:
                logger.warning(f"Failed to evaluate {condition_name}: {e}")
                all_results[condition_name] = {"error": str(e)}

        return all_results


# =============================================================================
# Single-Domain Dice Evaluator (lora_ablation compat)
# =============================================================================


class TestDiceEvaluator:
    """Evaluate segmentation Dice on BraTS-MEN test set only.

    Single-domain evaluator for backward compatibility with lora_ablation configs
    that use h5_file instead of men_h5_file/gli_h5_file.

    Args:
        checkpoint_path: Path to BrainSegFounder checkpoint.
        device: Device to run evaluation on.
        batch_size: Batch size for evaluation.
        num_workers: Number of data loader workers.
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
        model = create_ablation_model(
            condition_config=condition_config,
            training_config=training_config,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        skip_training = condition_config.get("skip_training", False)
        if not skip_training:
            checkpoint_file = condition_dir / "best_model.pt"
            if not checkpoint_file.exists():
                checkpoint_file = condition_dir / "checkpoint.pt"

            if checkpoint_file.exists():
                state_dict = torch.load(
                    checkpoint_file, map_location=self.device, weights_only=True
                )
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded weights from {checkpoint_file}")
            else:
                logger.warning(f"No checkpoint found at {condition_dir}")

        model.eval()
        return model

    def _create_dataloader(
        self,
        h5_path: str,
        h5_split: str = "test",
    ) -> DataLoader:
        """Create data loader for evaluation."""
        dataset = BraTSDatasetH5(
            h5_path=h5_path,
            split=h5_split,
            transform=get_h5_val_transforms(roi_size=FEATURE_ROI_SIZE),
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
        all_dice_scores: list[torch.Tensor] = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch["image"].to(self.device)
            segs = batch["seg"].to(self.device)

            if hasattr(model, "model"):
                pred = model.model(images)
            else:
                pred = model(images)

            dice_scores = self.dice_metric(pred, segs)
            all_dice_scores.append(dice_scores.cpu())

        dice_tensor = torch.cat(all_dice_scores, dim=0)  # [N_samples, 3]
        dice_mean_per_sample = dice_tensor.mean(dim=1)

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
        """Evaluate Dice for a single condition on BraTS-MEN test set."""
        condition_config = get_condition_config(config, condition_name)
        training_config = config["training"]
        output_dir = Path(config["experiment"]["output_dir"])
        condition_dir = output_dir / "conditions" / condition_name

        logger.info(f"Evaluating {condition_name} on BraTS-MEN")

        model = self._load_model(condition_config, training_config, condition_dir)

        h5_path = config.get("paths", {}).get("h5_file")
        dataloader = self._create_dataloader(h5_path)

        results = self._evaluate_dataset(model, dataloader, desc=f"{condition_name}")

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
        """Evaluate all conditions on BraTS-MEN."""
        men_results: dict[str, dict] = {}

        for cond in config["conditions"]:
            condition_name = cond["name"]
            try:
                men_results[condition_name] = self.evaluate_condition(condition_name, config)
            except Exception as e:
                logger.warning(f"Failed to evaluate {condition_name}: {e}")
                men_results[condition_name] = {"error": str(e)}

        return men_results


# =============================================================================
# Utility Functions
# =============================================================================


def save_dice_results(
    results: dict,
    output_path: Path,
) -> None:
    """Save Dice results to JSON file.

    Handles both single-domain (flat dict) and dual-domain (nested dict) results.

    Args:
        results: Dice results dict.
        output_path: Output path (file or directory).
    """
    if output_path.suffix == ".json":
        # Single file save
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved Dice results to {output_path}")
    else:
        # Directory save (per-domain)
        dice_dir = output_path / "dice" if "dice" not in output_path.name else output_path
        dice_dir.mkdir(parents=True, exist_ok=True)

        for domain, scores in results.items():
            if isinstance(scores, dict) and "error" not in scores:
                with open(dice_dir / f"{domain}_test_dice.json", "w") as f:
                    json.dump(scores, f, indent=2)

        with open(dice_dir / "dice_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved Dice results to {dice_dir}")


def generate_dice_summary(
    config: dict,
    results: dict[str, dict],
) -> str:
    """Generate summary CSV of Dice results.

    Args:
        config: Experiment configuration.
        results: Per-condition results (flat or nested).

    Returns:
        CSV string with all Dice metrics.
    """
    import pandas as pd

    rows = []
    for cond in config["conditions"]:
        name = cond["name"]
        row: dict[str, object] = {"condition": name}

        if name in results and "error" not in results[name]:
            cond_results = results[name]

            # Detect if nested (dual-domain) or flat (single-domain)
            if "men" in cond_results or "gli" in cond_results:
                # Dual-domain nested format
                for domain in ("men", "gli"):
                    if domain in cond_results:
                        for k, v in cond_results[domain].items():
                            row[f"{domain}_{k}"] = v
            else:
                # Single-domain flat format
                for k, v in cond_results.items():
                    row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# =============================================================================
# Main Entry Point
# =============================================================================


def main(
    config_path: str,
    condition: str | None = None,
    device: str = "cuda",
) -> None:
    """Main entry point for Dice evaluation.

    Automatically detects dual-domain vs single-domain config.

    Args:
        config_path: Path to experiment configuration.
        condition: Specific condition (None = all).
        device: Device to run on.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    checkpoint_path = config["paths"]["checkpoint"]

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Detect dual-domain config
    is_dual = "men_h5_file" in config.get("paths", {}) and "gli_h5_file" in config.get("paths", {})

    if is_dual:
        evaluator = DualDomainDiceEvaluator(
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=config.get("feature_extraction", {}).get("batch_size", 4),
            num_workers=config["training"]["num_workers"],
        )

        if condition is not None:
            results = evaluator.evaluate_condition(condition, config)
            cond_dir = output_dir / "conditions" / condition
            save_dice_results(results, cond_dir)
        else:
            all_results = evaluator.evaluate_all_conditions(config)

            for cond in config["conditions"]:
                name = cond["name"]
                cond_dir = output_dir / "conditions" / name
                cond_dir.mkdir(parents=True, exist_ok=True)
                if name in all_results and "error" not in all_results[name]:
                    save_dice_results(all_results[name], cond_dir)

            summary_csv = generate_dice_summary(config, all_results)
            summary_path = output_dir / "test_dice_summary.csv"
            with open(summary_path, "w") as f:
                f.write(summary_csv)
            logger.info(f"Saved summary to {summary_path}")
    else:
        evaluator = TestDiceEvaluator(
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=config.get("feature_extraction", {}).get("batch_size", 4),
            num_workers=config["training"]["num_workers"],
        )

        if condition is not None:
            results = evaluator.evaluate_condition(condition, config)
            cond_dir = output_dir / "conditions" / condition
            save_dice_results(results, cond_dir / "test_dice_men.json")
        else:
            all_results = evaluator.evaluate_all_conditions(config)

            for cond in config["conditions"]:
                name = cond["name"]
                cond_dir = output_dir / "conditions" / name
                cond_dir.mkdir(parents=True, exist_ok=True)
                if name in all_results and "error" not in all_results[name]:
                    save_dice_results(all_results[name], cond_dir / "test_dice_men.json")

            summary_csv = generate_dice_summary(config, all_results)
            summary_path = output_dir / "test_dice_summary.csv"
            with open(summary_path, "w") as f:
                f.write(summary_csv)
            logger.info(f"Saved Dice summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
