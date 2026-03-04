#!/usr/bin/env python
# experiments/lora/run.py
"""Unified LoRA experiment orchestrator.

Handles both single-domain (rank ablation) and dual-domain (MEN+GLI) experiments
via YAML config. All subcommands detect the experiment type automatically.

Usage:
    # Run complete pipeline
    python -m experiments.lora.run --config <yaml> run-all

    # Train single condition
    python -m experiments.lora.run --config <yaml> train --condition <name>

    # Analysis only (skip training)
    python -m experiments.lora.run --config <yaml> analyze-only
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from growth.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_file_logging_initialized = False


def setup_file_logging(output_dir: Path, log_filename: str | None = None) -> Path:
    """Set up file logging in addition to console logging.

    Args:
        output_dir: Experiment output directory.
        log_filename: Optional custom log filename. Defaults to timestamped name.

    Returns:
        Path to the log file.
    """
    global _file_logging_initialized

    if _file_logging_initialized:
        return output_dir / "experiment.log"

    output_dir.mkdir(parents=True, exist_ok=True)

    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"experiment_{timestamp}.log"

    log_path = output_dir / log_filename

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    latest_link = output_dir / "experiment.log"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(log_path.name)
    except OSError:
        pass

    _file_logging_initialized = True

    logger.info(f"Logging to file: {log_path}")
    logger.info(f"To check for errors: grep -E 'WARNING|ERROR' {log_path}")

    return log_path


# =============================================================================
# Pipeline Steps
# =============================================================================


def run_splits(config_path: str) -> None:
    """Generate data splits."""
    from .engine.data_splits import main as generate_splits

    logger.info("=" * 60)
    logger.info("STEP: Generating Data Splits")
    logger.info("=" * 60)
    generate_splits(config_path)


def run_train(
    config_path: str,
    condition: str,
    max_epochs: int | None = None,
    device: str = "cuda",
) -> None:
    """Train a single condition."""
    from .engine.train_condition import main as train_main

    logger.info(f"Training condition: {condition}")
    train_main(config_path, condition, max_epochs, device)


def run_train_all(
    config_path: str,
    max_epochs: int | None = None,
    device: str = "cuda",
) -> None:
    """Train all conditions."""
    logger.info("=" * 60)
    logger.info("STEP: Training All Conditions")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    for cond in config["conditions"]:
        name = cond["name"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training: {name}")
        logger.info(f"{'=' * 60}\n")
        run_train(config_path, name, max_epochs, device)


def run_extract(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Extract features for a condition."""
    from .engine.extract_features import main as extract_main

    logger.info(f"Extracting features for: {condition}")
    extract_main(config_path, condition, device)


def run_extract_all(
    config_path: str,
    device: str = "cuda",
) -> None:
    """Extract features for all conditions."""
    logger.info("=" * 60)
    logger.info("STEP: Extracting Features")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    for cond in config["conditions"]:
        try:
            run_extract(config_path, cond["name"], device)
        except Exception as e:
            logger.warning(f"Failed to extract {cond['name']}: {e}")


def run_probes(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Evaluate probes for a condition."""
    from .eval.evaluate_probes import main as probes_main

    logger.info(f"Evaluating probes for: {condition}")
    probes_main(config_path, condition, device)


def run_probes_all(
    config_path: str,
    device: str = "cuda",
) -> None:
    """Evaluate probes for all conditions."""
    logger.info("=" * 60)
    logger.info("STEP: GP Probe Evaluation")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    for cond in config["conditions"]:
        try:
            run_probes(config_path, cond["name"], device)
        except Exception as e:
            logger.warning(f"Failed probes for {cond['name']}: {e}")


def run_dice(
    config_path: str,
    condition: str | None = None,
    device: str = "cuda",
) -> None:
    """Evaluate per-domain Dice."""
    from .eval.evaluate_dice import main as dice_main

    logger.info("=" * 60)
    logger.info("STEP: Dice Evaluation")
    logger.info("=" * 60)

    dice_main(config_path, condition, device)


def run_dice_all(
    config_path: str,
    device: str = "cuda",
) -> None:
    """Evaluate Dice for all conditions."""
    from .eval.evaluate_dice import main as dice_main

    logger.info("=" * 60)
    logger.info("STEP: Dice Evaluation (All Conditions)")
    logger.info("=" * 60)

    dice_main(config_path, condition=None, device=device)


def run_domain_gap(
    config_path: str,
    condition: str | None = None,
    device: str = "cuda",
) -> None:
    """Evaluate domain gap metrics."""
    from .eval.evaluate_domain_gap import main as gap_main

    logger.info("=" * 60)
    logger.info("STEP: Domain Gap Evaluation")
    logger.info("=" * 60)

    gap_main(config_path, condition, device)


def run_feature_quality(
    config_path: str,
    condition: str | None = None,
) -> None:
    """Run feature quality evaluation."""
    from .eval.evaluate_feature_quality import main as feature_quality_main

    logger.info("=" * 60)
    logger.info("STEP: Feature Quality Evaluation")
    logger.info("=" * 60)

    feature_quality_main(config_path, condition)


def run_visualize(config_path: str) -> None:
    """Generate visualizations."""
    logger.info("=" * 60)
    logger.info("STEP: Generating Visualizations")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    is_dual = "men_h5_file" in config.get("paths", {})

    if is_dual:
        from .vis.dual_domain_viz import main as viz_main

        viz_main(config_path)
    else:
        from .vis.visualizations import main as viz_main

        viz_main(config_path)


def run_generate_tables(config_path: str) -> None:
    """Generate comprehensive result tables."""
    logger.info("=" * 60)
    logger.info("STEP: Generating Tables")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    is_dual = "men_h5_file" in config.get("paths", {})

    if is_dual:
        # Dual-domain table generation (inline from run_experiment)
        import pandas as pd

        output_dir = Path(config["experiment"]["output_dir"])
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for cond in config["conditions"]:
            name = cond["name"]
            cond_dir = output_dir / "conditions" / name
            row: dict[str, object] = {"condition": name}

            # Load Dice results
            dice_path = cond_dir / "dice" / "dice_summary.json"
            if dice_path.exists():
                with open(dice_path) as f:
                    dice = json.load(f)
                for domain in ("men", "gli"):
                    if domain in dice:
                        for k, v in dice[domain].items():
                            row[f"{domain}_{k}"] = v

            # Load probe results
            probes_path = cond_dir / "probes" / "all_probes.json"
            if probes_path.exists():
                with open(probes_path) as f:
                    probes = json.load(f)
                for domain in ("men", "gli"):
                    if domain in probes.get("per_domain", {}):
                        pd_data = probes["per_domain"][domain]
                        row[f"{domain}_r2_mean_linear"] = pd_data.get("r2_mean_linear")
                        row[f"{domain}_r2_mean_rbf"] = pd_data.get("r2_mean_rbf")
                        row[f"{domain}_effective_rank"] = pd_data.get("effective_rank")
                        row[f"{domain}_n_dead_dims"] = pd_data.get("n_dead_dims")

                for direction in ("gli_to_men", "men_to_gli"):
                    if direction in probes.get("cross_domain", {}):
                        cd = probes["cross_domain"][direction]
                        row[f"cross_{direction}_r2_mean"] = cd.get("r2_mean_linear")

            # Load domain gap
            gap_path = cond_dir / "domain_gap" / "domain_gap_metrics.json"
            if gap_path.exists():
                with open(gap_path) as f:
                    gap = json.load(f)
                row["mmd_squared"] = gap.get("mmd_squared")
                row["pad"] = gap.get("proxy_a_distance")
                row["cka_men_vs_gli"] = gap.get("cka_men_vs_gli")

            rows.append(row)

        df = pd.DataFrame(rows)

        csv_path = tables_dir / "comprehensive_results.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")
        logger.info(f"Saved {csv_path}")

        try:
            latex_path = tables_dir / "comprehensive_results.tex"
            df.to_latex(latex_path, index=False, float_format="%.4f", escape=True)
            logger.info(f"Saved {latex_path}")
        except Exception as e:
            logger.warning(f"LaTeX generation failed: {e}")
    else:
        # Single-domain table generation
        from .analysis.generate_tables import (
            generate_comprehensive_csv,
            generate_comprehensive_latex,
            generate_simplified_latex,
        )
        from .analysis.generate_tables import (
            load_all_metrics as load_all_metrics_tables,
        )

        output_dir = Path(config["experiment"]["output_dir"])
        metrics = load_all_metrics_tables(config)

        if not metrics:
            logger.warning("No metrics found for table generation")
            return

        generate_comprehensive_csv(metrics, config, output_dir / "comprehensive_results.csv")
        generate_comprehensive_latex(metrics, config, output_dir / "comprehensive_table.tex")
        generate_simplified_latex(metrics, config, output_dir / "simplified_table.tex")


def run_analysis(config_path: str) -> None:
    """Run statistical analysis."""
    logger.info("=" * 60)
    logger.info("STEP: Statistical Analysis")
    logger.info("=" * 60)

    from .analysis.analyze_results import analyze_results

    analyze_results(config_path)


def run_enhanced_diagnostics(config_path: str) -> None:
    """Run enhanced diagnostics analysis."""
    logger.info("=" * 60)
    logger.info("STEP: Enhanced Diagnostics")
    logger.info("=" * 60)

    from .analysis.enhanced_diagnostics import run_comprehensive_diagnostics

    run_comprehensive_diagnostics(config_path)


def run_regenerate(
    config_path: str,
    skip_cache: bool = False,
    figures_only: bool = False,
    tables_only: bool = False,
) -> None:
    """Regenerate all analysis outputs."""
    logger.info("=" * 60)
    logger.info("STEP: Regenerate Analysis")
    logger.info("=" * 60)

    from .analysis.regenerate_analysis import main as regenerate_main

    regenerate_main(config_path, skip_cache, figures_only, tables_only)


# =============================================================================
# Composite Commands
# =============================================================================


def run_all(
    config_path: str,
    max_epochs: int | None = None,
    device: str = "cuda",
    skip_training: bool = False,
) -> None:
    """Run complete experiment pipeline.

    Args:
        config_path: Path to experiment config.
        max_epochs: Override max epochs.
        device: Device.
        skip_training: Skip training step.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    setup_file_logging(output_dir)

    is_dual = "men_h5_file" in config.get("paths", {})

    logger.info("=" * 60)
    logger.info("LORA EXPERIMENT PIPELINE")
    logger.info(f"Mode: {'Dual-domain' if is_dual else 'Single-domain'}")
    logger.info("=" * 60)

    set_seed(config["experiment"]["seed"])

    # Save reproducibility artifacts (if available)
    try:
        from growth.utils.reproducibility import save_reproducibility_artifacts

        save_reproducibility_artifacts(output_dir, config, config_path)
    except ImportError:
        pass

    # Step 1: Generate splits (single-domain only)
    if not is_dual:
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
    if not is_dual:
        run_feature_quality(config_path)

    # Step 5: Evaluate Dice
    run_dice(config_path, device=device)

    # Step 5b: Domain gap (dual-domain only)
    if is_dual:
        run_domain_gap(config_path, device=device)

    # Step 6: Generate visualizations
    run_visualize(config_path)

    # Step 7: Generate tables
    run_generate_tables(config_path)

    # Step 8: Statistical analysis (single-domain only)
    if not is_dual:
        run_analysis(config_path)
        run_enhanced_diagnostics(config_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {config['experiment']['output_dir']}")


def run_analyze_only(
    config_path: str,
    device: str = "cuda",
    skip_extraction: bool = False,
    skip_probes: bool = False,
    skip_dice: bool = False,
) -> None:
    """Re-run analysis on already-trained conditions.

    Args:
        config_path: Path to experiment config.
        device: Device.
        skip_extraction: Skip feature extraction.
        skip_probes: Skip probe evaluation.
        skip_dice: Skip Dice evaluation.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    setup_file_logging(output_dir)

    is_dual = "men_h5_file" in config.get("paths", {})

    logger.info("=" * 60)
    logger.info("ANALYSIS-ONLY PIPELINE")
    logger.info(f"Mode: {'Dual-domain' if is_dual else 'Single-domain'}")
    logger.info("=" * 60)

    set_seed(config["experiment"]["seed"])

    if not skip_extraction:
        run_extract_all(config_path, device)

    if not skip_dice:
        run_dice(config_path, device=device)

    if not skip_probes:
        run_probes_all(config_path, device)
        if not is_dual:
            run_feature_quality(config_path)

    if is_dual:
        run_domain_gap(config_path, device=device)

    run_visualize(config_path)
    run_generate_tables(config_path)

    if not is_dual:
        run_analysis(config_path)
        run_enhanced_diagnostics(config_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Unified LoRA experiment orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora/config/ablation.yaml",
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

    # dice
    dice_parser = subparsers.add_parser("dice", help="Evaluate Dice")
    dice_parser.add_argument("--condition", type=str, default=None)
    dice_parser.add_argument("--device", type=str, default="cuda")

    # dice-all
    dice_all_parser = subparsers.add_parser("dice-all", help="Evaluate Dice for all")
    dice_all_parser.add_argument("--device", type=str, default="cuda")

    # domain-gap
    gap_parser = subparsers.add_parser("domain-gap", help="Domain gap metrics")
    gap_parser.add_argument("--condition", type=str, default=None)
    gap_parser.add_argument("--device", type=str, default="cuda")

    # feature-quality
    fq_parser = subparsers.add_parser("feature-quality", help="Evaluate feature quality")
    fq_parser.add_argument("--condition", type=str, default=None)

    # visualize
    subparsers.add_parser("visualize", help="Generate visualizations")

    # generate-tables
    subparsers.add_parser("generate-tables", help="Generate comprehensive result tables")

    # analyze
    subparsers.add_parser("analyze", help="Run statistical analysis")

    # enhanced-diagnostics
    subparsers.add_parser("enhanced-diagnostics", help="Run enhanced diagnostics")

    # regenerate
    regen_parser = subparsers.add_parser("regenerate", help="Regenerate analysis outputs")
    regen_parser.add_argument("--skip-cache", action="store_true")
    regen_parser.add_argument("--figures-only", action="store_true")
    regen_parser.add_argument("--tables-only", action="store_true")

    # run-all
    run_all_parser = subparsers.add_parser("run-all", help="Run complete pipeline")
    run_all_parser.add_argument("--max-epochs", type=int, default=None)
    run_all_parser.add_argument("--device", type=str, default="cuda")
    run_all_parser.add_argument("--skip-training", action="store_true")

    # train-ddp
    train_ddp_parser = subparsers.add_parser("train-ddp", help="Train a condition with DDP")
    train_ddp_parser.add_argument("--condition", type=str, required=True)
    train_ddp_parser.add_argument("--max-epochs", type=int, default=None)

    # analyze-only
    ao_parser = subparsers.add_parser("analyze-only", help="Re-run analysis only")
    ao_parser.add_argument("--device", type=str, default="cuda")
    ao_parser.add_argument("--skip-extraction", action="store_true")
    ao_parser.add_argument("--skip-probes", action="store_true")
    ao_parser.add_argument("--skip-dice", action="store_true")

    args = parser.parse_args()

    # Set up file logging
    if args.command and args.command != "help":
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f)
            output_dir = Path(config["experiment"]["output_dir"])
            setup_file_logging(output_dir)
        except Exception as e:
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
    elif args.command == "probes":
        run_probes(args.config, args.condition, args.device)
    elif args.command == "probes-all":
        run_probes_all(args.config, args.device)
    elif args.command == "dice":
        run_dice(args.config, args.condition, args.device)
    elif args.command == "dice-all":
        run_dice_all(args.config, args.device)
    elif args.command == "domain-gap":
        run_domain_gap(args.config, args.condition, args.device)
    elif args.command == "feature-quality":
        run_feature_quality(args.config, args.condition)
    elif args.command == "visualize":
        run_visualize(args.config)
    elif args.command == "generate-tables":
        run_generate_tables(args.config)
    elif args.command == "analyze":
        run_analysis(args.config)
    elif args.command == "enhanced-diagnostics":
        run_enhanced_diagnostics(args.config)
    elif args.command == "regenerate":
        run_regenerate(args.config, args.skip_cache, args.figures_only, args.tables_only)
    elif args.command == "run-all":
        run_all(args.config, args.max_epochs, args.device, args.skip_training)
    elif args.command == "train-ddp":
        from .engine.train_condition import main_ddp

        main_ddp(args.config, args.condition, args.max_epochs)
    elif args.command == "analyze-only":
        run_analyze_only(
            args.config,
            args.device,
            args.skip_extraction,
            args.skip_probes,
            args.skip_dice,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
