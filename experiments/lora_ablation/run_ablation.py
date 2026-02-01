#!/usr/bin/env python
# experiments/lora_ablation/run_ablation.py
"""Main orchestrator for the LoRA ablation experiment.

This script runs the complete ablation study:
1. Generate/load data splits
2. For each condition (baseline, lora_r4, lora_r8, lora_r16):
   a. Train segmentation model
   b. Extract features
   c. Evaluate linear probes
3. Generate comparison table
4. Print summary and recommendation

Usage:
    # Full run (all conditions)
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml

    # Skip training (use existing checkpoints)
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        --skip-training

    # Single condition
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        --conditions lora_r8

    # Quick test (2 epochs)
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        --max-epochs 2
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from growth.utils.seed import set_seed

from .data_splits import main as generate_splits, load_splits
from .train_condition import train_condition
from .extract_features import extract_features
from .evaluate_probes import evaluate_probes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_ablation(
    config_path: str,
    conditions: Optional[List[str]] = None,
    skip_training: bool = False,
    skip_extraction: bool = False,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """Run the complete LoRA ablation experiment.

    Args:
        config_path: Path to ablation.yaml configuration.
        conditions: List of conditions to run. If None, runs all.
        skip_training: Skip training (use existing checkpoints).
        skip_extraction: Skip feature extraction (use existing features).
        max_epochs: Override max epochs (for testing).
        device: Device to use.

    Returns:
        DataFrame with comparison results.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Get output directory
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate/load data splits
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Data Splits")
    logger.info("=" * 60)

    splits = generate_splits(config_path, force=False)

    # Determine which conditions to run
    all_conditions = [c["name"] for c in config["conditions"]]
    if conditions is None:
        conditions = all_conditions
    else:
        # Validate conditions
        for c in conditions:
            if c not in all_conditions:
                raise ValueError(
                    f"Unknown condition: {c}. Available: {all_conditions}"
                )

    logger.info(f"Running conditions: {conditions}")

    # Step 2: Run each condition
    results = {}

    for i, condition_name in enumerate(conditions):
        logger.info("\n" + "=" * 60)
        logger.info(f"Step 2.{i+1}: Condition '{condition_name}'")
        logger.info("=" * 60)

        condition_dir = output_dir / "conditions" / condition_name

        # 2a: Training
        if not skip_training:
            logger.info(f"\n--- Training {condition_name} ---")
            train_condition(
                condition_name=condition_name,
                config=config,
                splits=splits,
                max_epochs=max_epochs,
                device=device,
            )
        else:
            logger.info(f"Skipping training for {condition_name}")

        # 2b: Feature extraction
        if not skip_extraction:
            logger.info(f"\n--- Extracting features for {condition_name} ---")
            extract_features(
                condition_name=condition_name,
                config=config,
                splits=splits,
                device=device,
            )
        else:
            logger.info(f"Skipping feature extraction for {condition_name}")

        # 2c: Probe evaluation
        logger.info(f"\n--- Evaluating probes for {condition_name} ---")
        metrics = evaluate_probes(
            condition_name=condition_name,
            config=config,
        )

        # Load training summary
        summary_path = condition_dir / "training_summary.yaml"
        if summary_path.exists():
            with open(summary_path) as f:
                training_summary = yaml.safe_load(f)
            metrics["val_dice"] = training_summary.get("best_val_dice", None)
            metrics["best_epoch"] = training_summary.get("best_epoch", None)

        results[condition_name] = metrics

    # Step 3: Generate comparison table
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Comparison Table")
    logger.info("=" * 60)

    comparison_df = create_comparison_table(results)
    print("\n" + comparison_df.to_string() + "\n")

    # Save comparison table
    comparison_path = output_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_path)
    logger.info(f"Saved comparison table to {comparison_path}")

    # Step 4: Generate recommendation
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Recommendation")
    logger.info("=" * 60)

    recommendation = generate_recommendation(results)
    print(recommendation)

    # Save recommendation
    recommendation_path = output_dir / "recommendation.txt"
    with open(recommendation_path, "w") as f:
        f.write(recommendation)

    return comparison_df


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a comparison table from results.

    Args:
        results: Dict mapping condition names to metrics dicts.

    Returns:
        DataFrame with comparison.
    """
    rows = []
    for condition_name, metrics in results.items():
        row = {
            "Condition": condition_name,
            "R²_vol": metrics.get("r2_volume", None),
            "R²_loc": metrics.get("r2_location", None),
            "R²_shape": metrics.get("r2_shape", None),
            "R²_mean": metrics.get("r2_mean", None),
            "Val_Dice": metrics.get("val_dice", None),
            "Var_mean": metrics.get("variance_mean", None),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Format numeric columns
    for col in ["R²_vol", "R²_loc", "R²_shape", "R²_mean", "Val_Dice", "Var_mean"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")

    return df


def generate_recommendation(results: Dict[str, Dict]) -> str:
    """Generate a recommendation based on results.

    Args:
        results: Dict mapping condition names to metrics dicts.

    Returns:
        Recommendation text.
    """
    # Extract key metrics
    baseline_metrics = results.get("baseline", {})
    baseline_r2_vol = baseline_metrics.get("r2_volume", 0)
    baseline_r2_mean = baseline_metrics.get("r2_mean", 0)

    lines = [
        "\n" + "=" * 60,
        "RECOMMENDATION",
        "=" * 60,
        "",
    ]

    # Find best LoRA condition
    best_lora = None
    best_r2_mean = baseline_r2_mean
    best_delta = 0

    for condition_name, metrics in results.items():
        if condition_name == "baseline":
            continue

        r2_mean = metrics.get("r2_mean", 0)
        delta = r2_mean - baseline_r2_mean

        if r2_mean > best_r2_mean:
            best_lora = condition_name
            best_r2_mean = r2_mean
            best_delta = delta

    # Generate recommendation based on decision matrix
    lines.append(f"Baseline R²_vol: {baseline_r2_vol:.4f}")
    lines.append(f"Baseline R²_mean: {baseline_r2_mean:.4f}")
    lines.append("")

    if best_lora is None:
        lines.append("No LoRA condition outperformed baseline.")
        lines.append("")
        lines.append("DECISION: Skip LoRA adaptation")
        lines.append("")
        lines.append("The baseline encoder features are already well-suited for")
        lines.append("meningioma semantic prediction. LoRA adaptation provides")
        lines.append("no benefit and adds unnecessary complexity.")
    else:
        lines.append(f"Best LoRA condition: {best_lora}")
        lines.append(f"Best LoRA R²_mean: {best_r2_mean:.4f}")
        lines.append(f"Improvement over baseline: {best_delta * 100:.1f}%")
        lines.append("")

        # Decision based on improvement magnitude
        if best_delta < 0.03:
            lines.append("DECISION: Skip LoRA (marginal improvement)")
            lines.append("")
            lines.append(f"The improvement from {best_lora} ({best_delta * 100:.1f}%) is marginal.")
            lines.append("The added complexity of LoRA adaptation is not justified.")
        elif best_delta < 0.05:
            lines.append(f"DECISION: Consider {best_lora} (moderate improvement)")
            lines.append("")
            lines.append(f"The improvement from {best_lora} ({best_delta * 100:.1f}%) is moderate.")
            lines.append("Consider using LoRA if the improved semantic prediction")
            lines.append("is critical for downstream growth modeling.")
        else:
            lines.append(f"DECISION: Use {best_lora} (significant improvement)")
            lines.append("")
            lines.append(f"The improvement from {best_lora} ({best_delta * 100:.1f}%) is significant.")
            lines.append("LoRA adaptation provides meaningful enhancement of encoder")
            lines.append("features for meningioma semantic prediction.")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the complete LoRA ablation experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        choices=["baseline", "lora_r4", "lora_r8", "lora_r16"],
        help="Conditions to run (default: all)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing checkpoints)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip feature extraction (use existing features)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    run_ablation(
        config_path=args.config,
        conditions=args.conditions,
        skip_training=args.skip_training,
        skip_extraction=args.skip_extraction,
        max_epochs=args.max_epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
