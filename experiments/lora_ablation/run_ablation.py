#!/usr/bin/env python
# experiments/lora_ablation/run_ablation.py
"""Unified LoRA ablation orchestrator with configurable decoder.

This script runs the complete ablation pipeline:
1. Generate data splits (if not exist)
2. Train all conditions
3. Extract multi-scale features
4. Extract domain features (Glioma + Meningioma) for UMAP (optional)
5. Evaluate with linear + MLP probes
6. Generate enhanced visualizations
7. Statistical analysis and recommendations

Supports both decoder architectures via decoder_type config:
- "lightweight": Custom SegmentationHead (~2M params)
- "original": Full SwinUNETR decoder (~30M params, recommended)

Usage:
    # Run complete pipeline
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        run-all

    # With domain shift analysis
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        run-all --domain-features

    # Train single condition
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        train --condition lora_r8
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from growth.utils.seed import set_seed

from .data_splits import main as generate_splits, load_splits
from .extract_domain_features import extract_domain_features
from .evaluate_dice import TestDiceEvaluator, generate_dice_summary, save_dice_results
from .generate_tables import (
    load_all_metrics as load_all_metrics_tables,
    generate_comprehensive_csv,
    generate_comprehensive_latex,
    generate_simplified_latex,
    generate_domain_shift_csv,
)

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
    from .train_condition import main as train_main

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
    decoder_type = config.get("training", {}).get("decoder_type", "original")
    logger.info(f"Decoder type: {decoder_type}")

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
    from .extract_features import main as extract_main

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


def run_domain(
    config_path: str,
    condition: str,
    n_glioma: int = 200,
    n_meningioma: int = 200,
    device: str = "cuda",
) -> None:
    """Extract domain features (Glioma + Meningioma) for a condition."""
    logger.info(f"Extracting domain features for: {condition}")
    extract_domain_features(
        config_path=config_path,
        condition_name=condition,
        n_glioma=n_glioma,
        n_meningioma=n_meningioma,
        device=device,
    )


def run_domain_all(
    config_path: str,
    n_glioma: int = 200,
    n_meningioma: int = 200,
    device: str = "cuda",
) -> None:
    """Extract domain features for all conditions."""
    logger.info("=" * 60)
    logger.info("STEP 4: Extracting Domain Features (Glioma vs Meningioma)")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    conditions = [c["name"] for c in config["conditions"]]

    for condition in conditions:
        logger.info(f"\nExtracting domain features: {condition}")
        try:
            run_domain(config_path, condition, n_glioma, n_meningioma, device)
        except Exception as e:
            logger.warning(f"Failed to extract domain features for {condition}: {e}")


def run_probes(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Evaluate probes for a condition."""
    from .evaluate_probes import main as probes_main

    logger.info(f"Evaluating probes for: {condition}")
    probes_main(config_path, condition, device)


def run_probes_all(
    config_path: str,
    device: str = "cuda",
) -> None:
    """Evaluate probes for all conditions."""
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluating Probes")
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
    logger.info("STEP 7: Generating Visualizations")
    logger.info("=" * 60)

    from .visualizations import main as viz_main
    viz_main(config_path)


def run_test_dice(
    config_path: str,
    condition: str,
    dataset: str = "men",
    device: str = "cuda",
) -> None:
    """Evaluate test Dice for a single condition."""
    from .evaluate_dice import main as evaluate_dice_main

    logger.info(f"Evaluating test Dice for: {condition}")
    evaluate_dice_main(config_path, condition, dataset, device=device)


def run_test_dice_all(
    config_path: str,
    device: str = "cuda",
    glioma_test_size: int = 200,
) -> None:
    """Evaluate test Dice for all conditions on both datasets."""
    logger.info("=" * 60)
    logger.info("STEP 6: Test Dice Evaluation")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    checkpoint_path = config["paths"]["checkpoint"]

    evaluator = TestDiceEvaluator(
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=config.get("feature_extraction", {}).get("batch_size", 4),
        num_workers=config["training"]["num_workers"],
    )

    men_results, gli_results = evaluator.evaluate_all_conditions(
        config,
        include_glioma=True,
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

    # Generate summary
    summary_csv = generate_dice_summary(config, men_results, gli_results)
    summary_path = output_dir / "test_dice_summary.csv"
    with open(summary_path, "w") as f:
        f.write(summary_csv)
    logger.info(f"Saved Dice summary to {summary_path}")


def run_generate_tables(config_path: str) -> None:
    """Generate comprehensive result tables."""
    logger.info("=" * 60)
    logger.info("STEP 8: Generating Comprehensive Tables")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])

    # Load all metrics
    metrics = load_all_metrics_tables(config)

    if not metrics:
        logger.warning("No metrics found for table generation")
        return

    # Generate all tables
    generate_comprehensive_csv(metrics, config, output_dir / "comprehensive_results.csv")
    generate_comprehensive_latex(metrics, config, output_dir / "comprehensive_table.tex")
    generate_simplified_latex(metrics, config, output_dir / "simplified_table.tex")
    generate_domain_shift_csv(metrics, config, output_dir / "domain_shift_analysis.csv")

    logger.info("Generated comprehensive tables:")
    logger.info("  - comprehensive_results.csv")
    logger.info("  - comprehensive_table.tex")
    logger.info("  - simplified_table.tex")
    logger.info("  - domain_shift_analysis.csv")


def run_analysis(config_path: str, glioma_features_path: Optional[str] = None) -> None:
    """Run statistical analysis."""
    logger.info("=" * 60)
    logger.info("STEP 9: Statistical Analysis")
    logger.info("=" * 60)

    from .analyze_results import analyze_results
    analyze_results(config_path, glioma_features_path=glioma_features_path)


def run_all(
    config_path: str,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
    skip_training: bool = False,
    domain_features: bool = False,
    n_glioma: int = 200,
    n_meningioma: int = 200,
    glioma_test_size: int = 200,
) -> None:
    """Run complete ablation pipeline.

    Pipeline Steps:
    1. Generate data splits
    2. Train all conditions
    3. Extract features
    4. Extract domain features (optional)
    5. Evaluate probes
    6. Evaluate test Dice (BraTS-MEN + BraTS-GLI)
    7. Generate visualizations
    8. Generate comprehensive tables
    9. Statistical analysis
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    decoder_type = config.get("training", {}).get("decoder_type", "original")

    logger.info("=" * 60)
    logger.info("LORA ABLATION PIPELINE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  - Decoder type: {decoder_type}")
    if decoder_type == "original":
        logger.info("  - Original SwinUNETR decoder (pretrained)")
        logger.info("  - Auxiliary semantic prediction losses")
        logger.info("  - MLP probes for nonlinear analysis")
        logger.info("  - Multi-scale feature extraction")
    else:
        logger.info("  - Lightweight SegmentationHead decoder")
        logger.info("  - Linear probes only")
        logger.info("  - Single-scale features")
    if domain_features:
        logger.info("  - Domain shift analysis (Glioma vs Meningioma)")
    logger.info(f"  - Glioma test size: {glioma_test_size}")
    logger.info("")

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

    # Step 4: Extract domain features (optional)
    glioma_features_path = None
    if domain_features:
        run_domain_all(config_path, n_glioma, n_meningioma, device)
        # Get path to glioma features from first condition with them
        output_dir = Path(config["experiment"]["output_dir"])
        for cond in config["conditions"]:
            glioma_path = output_dir / "conditions" / cond["name"] / "features_glioma.pt"
            if glioma_path.exists():
                glioma_features_path = str(glioma_path)
                break
    else:
        logger.info("Skipping domain features (use --domain-features to enable)")

    # Step 5: Evaluate probes
    run_probes_all(config_path, device)

    # Step 6: Evaluate test Dice (BraTS-MEN + BraTS-GLI)
    run_test_dice_all(config_path, device, glioma_test_size)

    # Step 7: Generate visualizations
    run_visualize(config_path)

    # Step 8: Generate comprehensive tables
    run_generate_tables(config_path)

    # Step 9: Statistical analysis
    run_analysis(config_path, glioma_features_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {config['experiment']['output_dir']}")
    logger.info("")
    logger.info("Key output files:")
    logger.info("  - comprehensive_results.csv (all metrics)")
    logger.info("  - comprehensive_table.tex (LaTeX table)")
    logger.info("  - test_dice_summary.csv (Dice scores)")
    logger.info("  - domain_shift_analysis.csv (MEN vs GLI)")
    logger.info("  - analysis_report.md (full report)")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LoRA ablation with configurable decoder"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
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

    # domain (extract domain features for one condition)
    domain_parser = subparsers.add_parser(
        "domain", help="Extract domain features (Glioma + Meningioma)"
    )
    domain_parser.add_argument("--condition", type=str, required=True)
    domain_parser.add_argument("--n-glioma", type=int, default=200,
                               help="Number of glioma samples")
    domain_parser.add_argument("--n-meningioma", type=int, default=200,
                               help="Number of meningioma samples")
    domain_parser.add_argument("--device", type=str, default="cuda")

    # domain-all (extract domain features for all conditions)
    domain_all_parser = subparsers.add_parser(
        "domain-all", help="Extract domain features for all conditions"
    )
    domain_all_parser.add_argument("--n-glioma", type=int, default=200)
    domain_all_parser.add_argument("--n-meningioma", type=int, default=200)
    domain_all_parser.add_argument("--device", type=str, default="cuda")

    # visualize
    subparsers.add_parser("visualize", help="Generate visualizations")

    # test-dice
    dice_parser = subparsers.add_parser("test-dice", help="Evaluate test Dice for one condition")
    dice_parser.add_argument("--condition", type=str, required=True)
    dice_parser.add_argument("--dataset", type=str, default="men", choices=["men", "gli"])
    dice_parser.add_argument("--device", type=str, default="cuda")

    # test-dice-all
    dice_all_parser = subparsers.add_parser("test-dice-all", help="Evaluate test Dice for all conditions")
    dice_all_parser.add_argument("--device", type=str, default="cuda")
    dice_all_parser.add_argument("--glioma-test-size", type=int, default=200,
                                 help="Number of glioma subjects for test")

    # generate-tables
    subparsers.add_parser("generate-tables", help="Generate comprehensive result tables")

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Run statistical analysis")
    analyze_parser.add_argument("--glioma-features", type=str, default=None,
                               help="Path to glioma features for domain shift")

    # run-all
    run_all_parser = subparsers.add_parser("run-all", help="Run complete pipeline")
    run_all_parser.add_argument("--max-epochs", type=int, default=None)
    run_all_parser.add_argument("--device", type=str, default="cuda")
    run_all_parser.add_argument("--skip-training", action="store_true",
                               help="Skip training (use existing checkpoints)")
    run_all_parser.add_argument("--domain-features", action="store_true",
                               help="Extract domain features for UMAP")
    run_all_parser.add_argument("--n-glioma", type=int, default=200)
    run_all_parser.add_argument("--n-meningioma", type=int, default=200)
    run_all_parser.add_argument("--glioma-test-size", type=int, default=200,
                               help="Number of glioma subjects for test Dice")

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
    elif args.command == "domain":
        run_domain(args.config, args.condition, args.n_glioma,
                  args.n_meningioma, args.device)
    elif args.command == "domain-all":
        run_domain_all(args.config, args.n_glioma, args.n_meningioma, args.device)
    elif args.command == "probes":
        run_probes(args.config, args.condition, args.device)
    elif args.command == "probes-all":
        run_probes_all(args.config, args.device)
    elif args.command == "visualize":
        run_visualize(args.config)
    elif args.command == "test-dice":
        run_test_dice(args.config, args.condition, args.dataset, args.device)
    elif args.command == "test-dice-all":
        run_test_dice_all(args.config, args.device, args.glioma_test_size)
    elif args.command == "generate-tables":
        run_generate_tables(args.config)
    elif args.command == "analyze":
        run_analysis(args.config, args.glioma_features)
    elif args.command == "run-all":
        run_all(
            args.config, args.max_epochs, args.device, args.skip_training,
            args.domain_features, args.n_glioma, args.n_meningioma,
            args.glioma_test_size
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
