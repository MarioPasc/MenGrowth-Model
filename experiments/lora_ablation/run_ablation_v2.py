#!/usr/bin/env python
# experiments/lora_ablation/run_ablation_v2.py
"""Enhanced LoRA ablation orchestrator with original decoder and semantic heads.

This script runs the complete enhanced ablation pipeline:
1. Generate data splits (if not exist)
2. Train all conditions with original decoder
3. Extract multi-scale features
4. Evaluate with linear + MLP probes
5. Generate enhanced visualizations
6. Statistical analysis and recommendations

Key improvements over v1:
- Uses original SwinUNETR decoder for stronger gradients
- Optional auxiliary semantic losses during training
- MLP probes to detect nonlinear encoding
- Target normalization for stable evaluation
- Multi-scale feature extraction

Usage:
    python -m experiments.lora_ablation.run_ablation_v2 \
        --config experiments/lora_ablation/config/ablation_v2.yaml \
        run-all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import yaml

from growth.utils.seed import set_seed

from .data_splits import main as generate_splits, load_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_splits(config_path: str) -> None:
    """Generate data splits."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating Data Splits")
    logger.info("=" * 60)
    generate_splits(config_path)


def run_train(
    config_path: str,
    condition: str,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
) -> None:
    """Train a single condition."""
    from .train_condition_v2 import main as train_main

    logger.info(f"Training condition: {condition}")
    train_main(config_path, condition, max_epochs, device)


def run_train_all(
    config_path: str,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
) -> None:
    """Train all conditions."""
    logger.info("=" * 60)
    logger.info("STEP 2: Training All Conditions")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    conditions = [c["name"] for c in config["conditions"]]

    for condition in conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {condition}")
        logger.info(f"{'='*60}\n")
        run_train(config_path, condition, max_epochs, device)


def run_extract(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Extract features for a condition."""
    from .extract_features_v2 import main as extract_main

    logger.info(f"Extracting features for: {condition}")
    extract_main(config_path, condition, device)


def run_extract_all(
    config_path: str,
    device: str = "cuda",
) -> None:
    """Extract features for all conditions."""
    logger.info("=" * 60)
    logger.info("STEP 3: Extracting Features")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    conditions = [c["name"] for c in config["conditions"]]

    for condition in conditions:
        logger.info(f"\nExtracting: {condition}")
        try:
            run_extract(config_path, condition, device)
        except Exception as e:
            logger.warning(f"Failed to extract {condition}: {e}")


def run_probes(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Evaluate probes for a condition."""
    from .evaluate_probes_v2 import main as probes_main

    logger.info(f"Evaluating probes for: {condition}")
    probes_main(config_path, condition, device)


def run_probes_all(
    config_path: str,
    device: str = "cuda",
) -> None:
    """Evaluate probes for all conditions."""
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluating Probes")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    conditions = [c["name"] for c in config["conditions"]]

    for condition in conditions:
        logger.info(f"\nEvaluating: {condition}")
        try:
            run_probes(config_path, condition, device)
        except Exception as e:
            logger.warning(f"Failed to evaluate {condition}: {e}")


def run_visualize(config_path: str) -> None:
    """Generate visualizations."""
    logger.info("=" * 60)
    logger.info("STEP 5: Generating Visualizations")
    logger.info("=" * 60)

    from .visualizations_v2 import main as viz_main
    viz_main(config_path)


def run_analysis(config_path: str) -> None:
    """Run statistical analysis."""
    logger.info("=" * 60)
    logger.info("STEP 6: Statistical Analysis")
    logger.info("=" * 60)

    # Import original analysis (it still works with enhanced metrics)
    from .analyze_results import main as analyze_main
    analyze_main(config_path, with_glioma=False)


def run_all(
    config_path: str,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
    skip_training: bool = False,
) -> None:
    """Run complete enhanced ablation pipeline."""
    logger.info("=" * 60)
    logger.info("ENHANCED LORA ABLATION PIPELINE V2")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Key improvements:")
    logger.info("  - Original SwinUNETR decoder (not lightweight)")
    logger.info("  - Auxiliary semantic prediction losses")
    logger.info("  - MLP probes for nonlinear analysis")
    logger.info("  - Target normalization")
    logger.info("  - Multi-scale feature extraction")
    logger.info("")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])

    # Step 1: Generate splits
    run_splits(config_path)

    # Step 2: Train all conditions
    if not skip_training:
        run_train_all(config_path, max_epochs, device)
    else:
        logger.info("Skipping training (--skip-training)")

    # Step 3: Extract features
    run_extract_all(config_path, device)

    # Step 4: Evaluate probes
    run_probes_all(config_path, device)

    # Step 5: Generate visualizations
    run_visualize(config_path)

    # Step 6: Statistical analysis
    run_analysis(config_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {config['experiment']['output_dir']}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced LoRA ablation with original decoder"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation_v2.yaml",
        help="Path to configuration file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # splits
    subparsers.add_parser("splits", help="Generate data splits")

    # train
    train_parser = subparsers.add_parser("train", help="Train a condition")
    train_parser.add_argument("--condition", type=str, required=True)
    train_parser.add_argument("--max-epochs", type=int, default=None)
    train_parser.add_argument("--device", type=str, default="cuda")

    # train-all
    train_all_parser = subparsers.add_parser("train-all", help="Train all conditions")
    train_all_parser.add_argument("--max-epochs", type=int, default=None)
    train_all_parser.add_argument("--device", type=str, default="cuda")

    # extract
    extract_parser = subparsers.add_parser("extract", help="Extract features")
    extract_parser.add_argument("--condition", type=str, required=True)
    extract_parser.add_argument("--device", type=str, default="cuda")

    # extract-all
    extract_all_parser = subparsers.add_parser("extract-all", help="Extract all")
    extract_all_parser.add_argument("--device", type=str, default="cuda")

    # probes
    probes_parser = subparsers.add_parser("probes", help="Evaluate probes")
    probes_parser.add_argument("--condition", type=str, required=True)
    probes_parser.add_argument("--device", type=str, default="cuda")

    # probes-all
    probes_all_parser = subparsers.add_parser("probes-all", help="Evaluate all probes")
    probes_all_parser.add_argument("--device", type=str, default="cuda")

    # visualize
    subparsers.add_parser("visualize", help="Generate visualizations")

    # analyze
    subparsers.add_parser("analyze", help="Run statistical analysis")

    # run-all
    run_all_parser = subparsers.add_parser("run-all", help="Run complete pipeline")
    run_all_parser.add_argument("--max-epochs", type=int, default=None)
    run_all_parser.add_argument("--device", type=str, default="cuda")
    run_all_parser.add_argument("--skip-training", action="store_true",
                               help="Skip training (use existing checkpoints)")

    args = parser.parse_args()

    if args.command == "splits":
        run_splits(args.config)
    elif args.command == "train":
        run_train(args.config, args.condition, args.max_epochs, args.device)
    elif args.command == "train-all":
        run_train_all(args.config, args.max_epochs, args.device)
    elif args.command == "extract":
        run_extract(args.config, args.condition, args.device)
    elif args.command == "extract-all":
        run_extract_all(args.config, args.device)
    elif args.command == "probes":
        run_probes(args.config, args.condition, args.device)
    elif args.command == "probes-all":
        run_probes_all(args.config, args.device)
    elif args.command == "visualize":
        run_visualize(args.config)
    elif args.command == "analyze":
        run_analysis(args.config)
    elif args.command == "run-all":
        run_all(args.config, args.max_epochs, args.device, args.skip_training)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
