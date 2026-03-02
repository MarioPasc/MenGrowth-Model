#!/usr/bin/env python
# experiments/lora_ablation/run_ablation.py
"""Unified LoRA ablation orchestrator with configurable decoder.

This script runs the complete ablation pipeline:
1. Generate data splits (if not exist)
2. Train all conditions
3. Extract multi-scale features
4. Evaluate with GP probes (linear + RBF kernels)
5. Generate enhanced visualizations
6. Statistical analysis and recommendations

Supports both decoder architectures via decoder_type config:
- "lightweight": Custom SegmentationHead (~2M params)
- "original": Full SwinUNETR decoder (~30M params, recommended)

Usage:
    # Run complete pipeline
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        run-all

    # Train single condition
    python -m experiments.lora_ablation.run_ablation \
        --config experiments/lora_ablation/config/ablation.yaml \
        train --condition lora_r8
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from growth.utils.seed import set_seed
from growth.utils.reproducibility import save_reproducibility_artifacts

from .pipeline.data_splits import main as generate_splits, load_splits
from .pipeline.evaluate_dice import TestDiceEvaluator, generate_dice_summary, save_dice_results
from .analysis.generate_tables import (
    load_all_metrics as load_all_metrics_tables,
    generate_comprehensive_csv,
    generate_comprehensive_latex,
    generate_simplified_latex,
)

# Initial basic logging (will be enhanced with file handler after config is loaded)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Track whether file logging has been set up
_file_logging_initialized = False


def setup_file_logging(output_dir: Path, log_filename: Optional[str] = None) -> Path:
    """Set up file logging in addition to console logging.

    Creates a log file in the experiment output directory that captures all
    log messages. The file is useful for post-hoc debugging with grep:
        grep -E "WARNING|ERROR" experiment.log

    Args:
        output_dir: Experiment output directory.
        log_filename: Optional custom log filename. Defaults to timestamped name.

    Returns:
        Path to the log file.
    """
    global _file_logging_initialized

    if _file_logging_initialized:
        return output_dir / "experiment.log"  # Already set up

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"experiment_{timestamp}.log"

    log_path = output_dir / log_filename

    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)  # Capture all levels to file
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    # Add file handler to root logger (captures all loggers)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Also create a symlink to latest log for convenience
    latest_link = output_dir / "experiment.log"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(log_path.name)
    except OSError:
        pass  # Symlinks may not work on all systems

    _file_logging_initialized = True

    logger.info(f"Logging to file: {log_path}")
    logger.info(f"To check for errors: grep -E 'WARNING|ERROR' {log_path}")

    return log_path


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
    from .pipeline.train_condition import main as train_main

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
    from .pipeline.extract_features import main as extract_main

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
    from .pipeline.evaluate_probes import main as probes_main

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

    from .analysis.visualizations import main as viz_main
    viz_main(config_path)


def run_test_dice(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Evaluate test Dice for a single condition."""
    from .pipeline.evaluate_dice import main as evaluate_dice_main

    logger.info(f"Evaluating test Dice for: {condition}")
    evaluate_dice_main(config_path, condition, device=device)


def run_test_dice_all(
    config_path: str,
    device: str = "cuda",
) -> None:
    """Evaluate test Dice for all conditions on BraTS-MEN."""
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

    men_results = evaluator.evaluate_all_conditions(config)

    # Save per-condition results
    for cond in config["conditions"]:
        name = cond["name"]
        cond_dir = output_dir / "conditions" / name
        cond_dir.mkdir(parents=True, exist_ok=True)

        if name in men_results and "error" not in men_results[name]:
            save_dice_results(men_results[name], cond_dir / "test_dice_men.json")

    # Generate summary
    summary_csv = generate_dice_summary(config, men_results)
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

    logger.info("Generated comprehensive tables:")
    logger.info("  - comprehensive_results.csv")
    logger.info("  - comprehensive_table.tex")
    logger.info("  - simplified_table.tex")


def run_analysis(config_path: str) -> None:
    """Run statistical analysis."""
    logger.info("=" * 60)
    logger.info("STEP 9: Statistical Analysis")
    logger.info("=" * 60)

    from .analysis.analyze_results import analyze_results
    analyze_results(config_path)


def run_feature_quality(
    config_path: str,
    condition: str | None = None,
) -> None:
    """Run feature quality evaluation."""
    logger.info("=" * 60)
    logger.info("STEP: Feature Quality Evaluation")
    logger.info("=" * 60)

    from .pipeline.evaluate_feature_quality import main as feature_quality_main
    feature_quality_main(config_path, condition)


def run_feature_quality_all(config_path: str) -> None:
    """Run feature quality evaluation for all conditions."""
    logger.info("=" * 60)
    logger.info("STEP: Feature Quality Evaluation (All Conditions)")
    logger.info("=" * 60)

    from .pipeline.evaluate_feature_quality import main as feature_quality_main
    feature_quality_main(config_path, condition=None)


def run_regenerate(
    config_path: str,
    skip_cache: bool = False,
    figures_only: bool = False,
    tables_only: bool = False,
) -> None:
    """Regenerate all analysis outputs (figures, tables, reports)."""
    logger.info("=" * 60)
    logger.info("STEP: Regenerate Analysis")
    logger.info("=" * 60)

    from .analysis.regenerate_analysis import main as regenerate_main
    regenerate_main(config_path, skip_cache, figures_only, tables_only)


def run_enhanced_diagnostics(config_path: str) -> None:
    """Run enhanced diagnostics analysis."""
    logger.info("=" * 60)
    logger.info("STEP 10: Enhanced Diagnostics")
    logger.info("=" * 60)

    from .analysis.enhanced_diagnostics import run_comprehensive_diagnostics
    run_comprehensive_diagnostics(config_path)


def run_analyze_only(
    config_path: str,
    device: str = "cuda",
    skip_extraction: bool = False,
    skip_probes: bool = False,
    skip_dice: bool = False,
) -> None:
    """Run analysis-only pipeline on already-trained conditions.

    This is useful when training is complete but analysis needs to be re-run
    with fixes or enhancements. Re-computes features, probes, and tables.

    Pipeline Steps (analysis-only):
    1. Extract features (unless --skip-extraction)
    2. Evaluate probes (unless --skip-probes)
    3. Evaluate test Dice (unless --skip-dice)
    4. Generate visualizations
    5. Generate comprehensive tables
    6. Statistical analysis with enhanced diagnostics
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])

    logger.info("=" * 60)
    logger.info("ANALYSIS-ONLY PIPELINE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This mode skips training and re-runs analysis steps.")
    logger.info(f"  - Skip feature extraction: {skip_extraction}")
    logger.info(f"  - Skip probe evaluation: {skip_probes}")
    logger.info(f"  - Skip Dice evaluation: {skip_dice}")
    logger.info("")

    set_seed(config["experiment"]["seed"])

    # Step 1: Extract features (optional)
    if not skip_extraction:
        run_extract_all(config_path, device)

    # Step 2: Evaluate probes (optional)
    if not skip_probes:
        run_probes_all(config_path, device)

    # Step 2b: Feature quality evaluation
    if not skip_probes:
        run_feature_quality_all(config_path)

    # Step 3: Evaluate test Dice (optional)
    if not skip_dice:
        run_test_dice_all(config_path, device)

    # Step 4: Generate visualizations
    run_visualize(config_path)

    # Step 5: Generate comprehensive tables
    run_generate_tables(config_path)

    # Step 6: Statistical analysis
    run_analysis(config_path)

    # Step 7: Enhanced diagnostics (includes gradient analysis)
    run_enhanced_diagnostics(config_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {config['experiment']['output_dir']}")


def run_all(
    config_path: str,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
    skip_training: bool = False,
) -> None:
    """Run complete ablation pipeline.

    Pipeline Steps:
    1. Generate data splits
    2. Train all conditions
    3. Extract features
    4. Evaluate probes
    5. Evaluate test Dice (BraTS-MEN)
    6. Generate visualizations
    7. Generate comprehensive tables
    8. Statistical analysis
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    decoder_type = config.get("training", {}).get("decoder_type", "original")

    # Save reproducibility artifacts
    logger.info("Saving reproducibility artifacts...")
    save_reproducibility_artifacts(output_dir, config, config_path)

    logger.info("=" * 60)
    logger.info("LORA ABLATION PIPELINE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  - Decoder type: {decoder_type}")
    if decoder_type == "original":
        logger.info("  - Original SwinUNETR decoder (pretrained)")
        logger.info("  - Auxiliary semantic prediction losses")
        logger.info("  - GP probes (linear + RBF) for nonlinear analysis")
        logger.info("  - Multi-scale feature extraction")
    else:
        logger.info("  - Lightweight SegmentationHead decoder")
        logger.info("  - Linear probes only")
        logger.info("  - Single-scale features")
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

    # Step 4: Evaluate probes
    run_probes_all(config_path, device)

    # Step 4b: Feature quality evaluation
    run_feature_quality_all(config_path)

    # Step 5: Evaluate test Dice (BraTS-MEN)
    run_test_dice_all(config_path, device)

    # Step 6: Generate visualizations
    run_visualize(config_path)

    # Step 7: Generate comprehensive tables
    run_generate_tables(config_path)

    # Step 8: Statistical analysis
    run_analysis(config_path)

    # Step 9: Enhanced diagnostics (gradient analysis, feature quality, etc.)
    run_enhanced_diagnostics(config_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {config['experiment']['output_dir']}")
    logger.info("")
    logger.info("Key output files:")
    logger.info("  - meta/run_manifest.json (reproducibility)")
    logger.info("  - comprehensive_results.csv (all metrics)")
    logger.info("  - comprehensive_table.tex (LaTeX table)")
    logger.info("  - test_dice_summary.csv (Dice scores)")
    logger.info("  - analysis_report.md (full report)")
    logger.info("  - diagnostics_*.csv (enhanced diagnostics)")
    logger.info("  - diagnostics_report.txt (diagnostic summary)")


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
        "domain", help="[REMOVED] Use experiments.domain_gap instead"
    )

    # domain-all (extract domain features for all conditions)
    domain_all_parser = subparsers.add_parser(
        "domain-all", help="[REMOVED] Use experiments.domain_gap instead"
    )

    # visualize
    subparsers.add_parser("visualize", help="Generate visualizations")

    # test-dice
    dice_parser = subparsers.add_parser("test-dice", help="Evaluate test Dice for one condition")
    dice_parser.add_argument("--condition", type=str, required=True)
    dice_parser.add_argument("--device", type=str, default="cuda")

    # test-dice-all
    dice_all_parser = subparsers.add_parser("test-dice-all", help="Evaluate test Dice for all conditions")
    dice_all_parser.add_argument("--device", type=str, default="cuda")

    # generate-tables
    subparsers.add_parser("generate-tables", help="Generate comprehensive result tables")

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Run statistical analysis")

    # feature-quality
    fq_parser = subparsers.add_parser("feature-quality", help="Evaluate feature quality")
    fq_parser.add_argument("--condition", type=str, default=None,
                           help="Single condition (if omitted, evaluates all)")

    # enhanced-diagnostics
    subparsers.add_parser(
        "enhanced-diagnostics",
        help="Run enhanced diagnostics (gradient analysis, feature quality, etc.)"
    )

    # regenerate
    regen_parser = subparsers.add_parser(
        "regenerate",
        help="Regenerate analysis outputs (figures, tables, reports) from existing data"
    )
    regen_parser.add_argument("--skip-cache", action="store_true",
                               help="Reuse existing figure cache")
    regen_parser.add_argument("--figures-only", action="store_true",
                               help="Only regenerate figures")
    regen_parser.add_argument("--tables-only", action="store_true",
                               help="Only regenerate tables")

    # run-all
    run_all_parser = subparsers.add_parser("run-all", help="Run complete pipeline")
    run_all_parser.add_argument("--max-epochs", type=int, default=None)
    run_all_parser.add_argument("--device", type=str, default="cuda")
    run_all_parser.add_argument("--skip-training", action="store_true",
                               help="Skip training (use existing checkpoints)")

    # analyze-only
    analyze_only_parser = subparsers.add_parser(
        "analyze-only",
        help="Re-run analysis on already-trained conditions (skips training)"
    )
    analyze_only_parser.add_argument("--device", type=str, default="cuda")
    analyze_only_parser.add_argument("--skip-extraction", action="store_true",
                                     help="Skip feature extraction (use existing)")
    analyze_only_parser.add_argument("--skip-probes", action="store_true",
                                     help="Skip probe evaluation (use existing)")
    analyze_only_parser.add_argument("--skip-dice", action="store_true",
                                     help="Skip Dice evaluation (use existing)")

    args = parser.parse_args()

    # Set up file logging if a command is specified
    if args.command and args.command != "help":
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f)
            output_dir = Path(config["experiment"]["output_dir"])
            setup_file_logging(output_dir)
        except Exception as e:
            # Don't fail if logging setup fails, just warn
            logger.warning(f"Could not set up file logging: {e}")

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
    elif args.command in ("domain", "domain-all"):
        logger.error(
            "Domain gap analysis has been moved to experiments.domain_gap. "
            "Use: python -m experiments.domain_gap.run_domain_gap --config <path>"
        )
        sys.exit(1)
    elif args.command == "probes":
        run_probes(args.config, args.condition, args.device)
    elif args.command == "probes-all":
        run_probes_all(args.config, args.device)
    elif args.command == "visualize":
        run_visualize(args.config)
    elif args.command == "test-dice":
        run_test_dice(args.config, args.condition, device=args.device)
    elif args.command == "test-dice-all":
        run_test_dice_all(args.config, args.device)
    elif args.command == "generate-tables":
        run_generate_tables(args.config)
    elif args.command == "analyze":
        run_analysis(args.config)
    elif args.command == "feature-quality":
        run_feature_quality(args.config, args.condition)
    elif args.command == "enhanced-diagnostics":
        run_enhanced_diagnostics(args.config)
    elif args.command == "regenerate":
        run_regenerate(args.config, args.skip_cache, args.figures_only, args.tables_only)
    elif args.command == "run-all":
        run_all(
            args.config, args.max_epochs, args.device, args.skip_training,
        )
    elif args.command == "analyze-only":
        run_analyze_only(
            args.config, args.device,
            args.skip_extraction, args.skip_probes, args.skip_dice
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
