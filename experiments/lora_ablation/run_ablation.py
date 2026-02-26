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
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from growth.utils.seed import set_seed
from growth.utils.reproducibility import save_reproducibility_artifacts

from .pipeline.data_splits import main as generate_splits, load_splits
from .analysis.domain_visualizations import generate_domain_figures
from .pipeline.extract_domain_features import extract_domain_features
from .pipeline.evaluate_dice import TestDiceEvaluator, generate_dice_summary, save_dice_results
from .analysis.generate_tables import (
    load_all_metrics as load_all_metrics_tables,
    generate_comprehensive_csv,
    generate_comprehensive_latex,
    generate_simplified_latex,
    generate_domain_shift_csv,
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
    dataset: str = "men",
    device: str = "cuda",
) -> None:
    """Evaluate test Dice for a single condition."""
    from .pipeline.evaluate_dice import main as evaluate_dice_main

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

    from .analysis.analyze_results import analyze_results
    analyze_results(config_path, glioma_features_path=glioma_features_path)


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
    domain_features: bool = False,
    n_glioma: int = 200,
    n_meningioma: int = 200,
    glioma_test_size: int = 200,
    skip_extraction: bool = False,
    skip_probes: bool = False,
    skip_dice: bool = False,
) -> None:
    """Run analysis-only pipeline on already-trained conditions.

    This is useful when training is complete but analysis needs to be re-run
    with fixes or enhancements. Re-computes features, probes, and tables.

    Pipeline Steps (analysis-only):
    1. Extract features (unless --skip-extraction)
    2. Extract domain features (if --domain-features)
    3. Evaluate probes (unless --skip-probes)
    4. Evaluate test Dice (unless --skip-dice)
    5. Generate visualizations
    6. Generate comprehensive tables
    7. Statistical analysis with enhanced diagnostics
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
    logger.info(f"  - Domain features: {domain_features}")
    logger.info("")

    set_seed(config["experiment"]["seed"])

    # Step 1: Extract features (optional)
    if not skip_extraction:
        run_extract_all(config_path, device)

    # Step 2: Extract domain features (optional)
    glioma_features_path = None
    if domain_features:
        run_domain_all(config_path, n_glioma, n_meningioma, device)
        for cond in config["conditions"]:
            glioma_path = output_dir / "conditions" / cond["name"] / "features_glioma.pt"
            if glioma_path.exists():
                glioma_features_path = str(glioma_path)
                break

    # Step 3: Evaluate probes (optional)
    if not skip_probes:
        run_probes_all(config_path, device)

    # Step 3b: Feature quality evaluation
    if not skip_probes:
        run_feature_quality_all(config_path)

    # Step 4: Evaluate test Dice (optional)
    if not skip_dice:
        run_test_dice_all(config_path, device, glioma_test_size)

    # Step 5: Generate visualizations
    run_visualize(config_path)

    # Step 5b: Generate domain shift visualizations
    if domain_features:
        logger.info("Generating domain shift visualizations...")
        generate_domain_figures(output_dir, config)

    # Step 6: Generate comprehensive tables
    run_generate_tables(config_path)

    # Step 7: Statistical analysis
    run_analysis(config_path, glioma_features_path)

    # Step 8: Enhanced diagnostics (includes gradient analysis)
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

    # Step 5b: Feature quality evaluation
    run_feature_quality_all(config_path)

    # Step 6: Evaluate test Dice (BraTS-MEN + BraTS-GLI)
    run_test_dice_all(config_path, device, glioma_test_size)

    # Step 7: Generate visualizations
    run_visualize(config_path)

    # Step 7b: Generate domain shift visualizations
    if domain_features:
        logger.info("Generating domain shift visualizations...")
        generate_domain_figures(output_dir, config)

    # Step 8: Generate comprehensive tables
    run_generate_tables(config_path)

    # Step 9: Statistical analysis
    run_analysis(config_path, glioma_features_path)

    # Step 10: Enhanced diagnostics (gradient analysis, feature quality, etc.)
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
    logger.info("  - domain_shift_analysis.csv (MEN vs GLI)")
    logger.info("  - figures/domain_*.png (domain shift plots)")
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
    run_all_parser.add_argument("--domain-features", action="store_true",
                               help="Extract domain features for UMAP")
    run_all_parser.add_argument("--n-glioma", type=int, default=200)
    run_all_parser.add_argument("--n-meningioma", type=int, default=200)
    run_all_parser.add_argument("--glioma-test-size", type=int, default=200,
                               help="Number of glioma subjects for test Dice")

    # analyze-only (NEW)
    analyze_only_parser = subparsers.add_parser(
        "analyze-only",
        help="Re-run analysis on already-trained conditions (skips training)"
    )
    analyze_only_parser.add_argument("--device", type=str, default="cuda")
    analyze_only_parser.add_argument("--domain-features", action="store_true",
                                     help="Extract domain features for UMAP")
    analyze_only_parser.add_argument("--n-glioma", type=int, default=200)
    analyze_only_parser.add_argument("--n-meningioma", type=int, default=200)
    analyze_only_parser.add_argument("--glioma-test-size", type=int, default=200,
                                     help="Number of glioma subjects for test Dice")
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
    elif args.command == "feature-quality":
        run_feature_quality(args.config, args.condition)
    elif args.command == "enhanced-diagnostics":
        run_enhanced_diagnostics(args.config)
    elif args.command == "regenerate":
        run_regenerate(args.config, args.skip_cache, args.figures_only, args.tables_only)
    elif args.command == "run-all":
        run_all(
            args.config, args.max_epochs, args.device, args.skip_training,
            args.domain_features, args.n_glioma, args.n_meningioma,
            args.glioma_test_size
        )
    elif args.command == "analyze-only":
        run_analyze_only(
            args.config, args.device, args.domain_features,
            args.n_glioma, args.n_meningioma, args.glioma_test_size,
            args.skip_extraction, args.skip_probes, args.skip_dice
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
