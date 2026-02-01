#!/usr/bin/env python
# experiments/lora_ablation/cli.py
"""Professional CLI for LoRA Ablation Experiment.

This module provides a unified command-line interface for the LoRA ablation study:

    growth-exp-lora-ablation <command> [options]

Commands:
    splits      Generate data splits for the experiment
    run         Train a single experimental condition
    run-all     Run complete experiment (all conditions + analysis)
    extract     Extract features for probe evaluation
    domain      Extract domain features (glioma + meningioma) for UMAP
    probes      Evaluate linear probes on extracted features
    analyse     Run statistical analysis and generate visualizations

Examples:
    # Generate data splits
    growth-exp-lora-ablation splits

    # Train baseline condition
    growth-exp-lora-ablation run --condition baseline

    # Run complete experiment
    growth-exp-lora-ablation run-all

    # Run analysis with domain visualization
    growth-exp-lora-ablation analyse --glioma-features

    # Quick test (2 epochs, baseline only)
    growth-exp-lora-ablation run --condition baseline --max-epochs 2
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lora_ablation")

# Default config path
DEFAULT_CONFIG = "experiments/lora_ablation/config/ablation.yaml"


def get_config_path(args) -> str:
    """Get config path from args or default."""
    return getattr(args, "config", DEFAULT_CONFIG)


def check_cuda(device: str) -> str:
    """Check CUDA availability and return appropriate device."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return "cpu"
    return device


# =============================================================================
# Subcommand Implementations
# =============================================================================

def cmd_splits(args) -> int:
    """Generate data splits."""
    from .data_splits import main as generate_splits

    logger.info("Generating data splits...")
    generate_splits(get_config_path(args), force=args.force)
    return 0


def cmd_run(args) -> int:
    """Train a single condition."""
    from .data_splits import load_splits
    from .train_condition import train_condition

    config_path = get_config_path(args)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Ensure splits exist
    try:
        splits = load_splits(config_path)
    except FileNotFoundError:
        logger.error("Data splits not found. Run 'splits' command first.")
        return 1

    device = check_cuda(args.device)

    logger.info(f"Training condition: {args.condition}")
    train_condition(
        condition_name=args.condition,
        config=config,
        splits=splits,
        max_epochs=args.max_epochs,
        device=device,
    )

    # Optionally extract features and evaluate probes
    if args.evaluate:
        from .extract_features import extract_features
        from .evaluate_probes import evaluate_probes

        logger.info("Extracting features...")
        extract_features(args.condition, config, splits, device)

        logger.info("Evaluating probes...")
        evaluate_probes(args.condition, config, config_path, device)

    return 0


def cmd_extract(args) -> int:
    """Extract features for a condition."""
    from .data_splits import load_splits
    from .extract_features import extract_features

    config_path = get_config_path(args)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    splits = load_splits(config_path)
    device = check_cuda(args.device)

    if args.condition == "all":
        conditions = [c["name"] for c in config["conditions"]]
    else:
        conditions = [args.condition]

    for cond in conditions:
        logger.info(f"Extracting features for {cond}")
        extract_features(cond, config, splits, device)

    return 0


def cmd_domain(args) -> int:
    """Extract domain features for UMAP visualization."""
    from .extract_domain_features import extract_domain_features

    config_path = get_config_path(args)
    device = check_cuda(args.device)

    if args.condition == "all":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        conditions = [c["name"] for c in config["conditions"]]
    else:
        conditions = [args.condition]

    for cond in conditions:
        logger.info(f"Extracting domain features for {cond}")
        extract_domain_features(
            config_path=config_path,
            condition_name=cond,
            n_glioma=args.n_glioma,
            n_meningioma=args.n_meningioma,
            device=device,
        )

    return 0


def cmd_probes(args) -> int:
    """Evaluate linear probes."""
    from .evaluate_probes import evaluate_probes

    config_path = get_config_path(args)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = check_cuda(args.device)
    skip_dice = getattr(args, "skip_dice", False)

    if args.condition == "all":
        conditions = [c["name"] for c in config["conditions"]]
    else:
        conditions = [args.condition]

    for cond in conditions:
        logger.info(f"Evaluating probes for {cond}")
        evaluate_probes(cond, config, config_path, device, skip_dice)

    return 0


def cmd_analyse(args) -> int:
    """Run statistical analysis and generate visualizations."""
    from .analyze_results import analyze_results

    config_path = get_config_path(args)

    # Determine glioma features path
    glioma_features_path = None
    if args.glioma_features:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Use baseline condition's glioma features by default
        output_dir = Path(config["experiment"]["output_dir"])
        glioma_path = output_dir / "conditions" / "baseline" / "features_glioma.pt"

        if glioma_path.exists():
            glioma_features_path = str(glioma_path)
            logger.info(f"Using glioma features from {glioma_path}")
        else:
            logger.warning(
                f"Glioma features not found at {glioma_path}. "
                "Run 'domain' command first to extract them."
            )

    analyze_results(
        config_path=config_path,
        glioma_features_path=glioma_features_path,
        skip_figures=args.skip_figures,
    )

    return 0


def cmd_run_all(args) -> int:
    """Run complete experiment: all conditions + analysis."""
    from .data_splits import main as generate_splits, load_splits
    from .train_condition import train_condition
    from .extract_features import extract_features
    from .evaluate_probes import evaluate_probes
    from .extract_domain_features import extract_domain_features
    from .analyze_results import analyze_results

    config_path = get_config_path(args)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = check_cuda(args.device)
    conditions = [c["name"] for c in config["conditions"]]

    # Override conditions if specified
    if args.conditions:
        conditions = args.conditions

    # Step 1: Generate splits
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Generating Data Splits")
    logger.info("=" * 70)
    generate_splits(config_path, force=False)
    splits = load_splits(config_path)

    # Step 2: Train all conditions
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Training Conditions")
    logger.info("=" * 70)

    for i, cond in enumerate(conditions):
        logger.info(f"\n[{i+1}/{len(conditions)}] Training {cond}")
        train_condition(
            condition_name=cond,
            config=config,
            splits=splits,
            max_epochs=args.max_epochs,
            device=device,
        )

    # Step 3: Extract features
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Extracting Features")
    logger.info("=" * 70)

    for cond in conditions:
        logger.info(f"Extracting features for {cond}")
        extract_features(cond, config, splits, device)

    # Step 4: Evaluate probes
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Evaluating Linear Probes")
    logger.info("=" * 70)

    for cond in conditions:
        logger.info(f"Evaluating probes for {cond}")
        evaluate_probes(cond, config, config_path, device)

    # Step 5: Extract domain features (optional)
    glioma_features_path = None
    if args.domain_features:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: Extracting Domain Features for UMAP")
        logger.info("=" * 70)

        for cond in conditions:
            logger.info(f"Extracting domain features for {cond}")
            paths = extract_domain_features(
                config_path=config_path,
                condition_name=cond,
                n_glioma=args.n_domain_samples,
                n_meningioma=args.n_domain_samples,
                device=device,
            )
            if cond == "baseline" and paths.get("glioma"):
                glioma_features_path = str(paths["glioma"])

    # Step 6: Run analysis
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Running Analysis")
    logger.info("=" * 70)

    analyze_results(
        config_path=config_path,
        glioma_features_path=glioma_features_path,
        skip_figures=False,
    )

    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {config['experiment']['output_dir']}")

    return 0


# =============================================================================
# CLI Parser Setup
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""

    # Main parser
    parser = argparse.ArgumentParser(
        prog="growth-exp-lora-ablation",
        description="LoRA Ablation Experiment CLI for Meningioma Encoder Adaptation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s splits                          Generate data splits
  %(prog)s run --condition baseline        Train baseline condition
  %(prog)s run-all                         Run complete experiment
  %(prog)s analyse --glioma-features       Analyse with domain visualization

For more information on a specific command:
  %(prog)s <command> --help
  
  ┌────────────────────────┬─────────────────────────────────────────────────────────────────┐
  │        Command         │                           Description                           │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ splits                 │ Generate train/val/test data splits                             │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ run --condition <name> │ Train a single condition (baseline, lora_r2/4/8/16/32)          │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ extract                │ Extract encoder features for probe evaluation                   │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ domain                 │ Extract glioma + meningioma features for UMAP                   │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ probes                 │ Evaluate linear probes on extracted features                    │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ analyse                │ Run statistical analysis and generate figures                   │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ run-all                │ Run complete experiment pipeline                                │
  └────────────────────────┴─────────────────────────────────────────────────────────────────┘

""",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG})",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>",
    )

    # -------------------------------------------------------------------------
    # splits subcommand
    # -------------------------------------------------------------------------
    sp_splits = subparsers.add_parser(
        "splits",
        help="Generate train/val/test data splits",
        description="Generate and save fixed data splits for reproducible experiments.",
    )
    sp_splits.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if splits exist",
    )
    sp_splits.set_defaults(func=cmd_splits)

    # -------------------------------------------------------------------------
    # run subcommand
    # -------------------------------------------------------------------------
    sp_run = subparsers.add_parser(
        "run",
        help="Train a single experimental condition",
        description="Train baseline or LoRA-adapted encoder with segmentation objective.",
    )
    sp_run.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["baseline", "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32"],
        help="Condition to train",
    )
    sp_run.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs (for quick testing)",
    )
    sp_run.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    sp_run.add_argument(
        "--evaluate",
        action="store_true",
        help="Also extract features and evaluate probes after training",
    )
    sp_run.set_defaults(func=cmd_run)

    # -------------------------------------------------------------------------
    # extract subcommand
    # -------------------------------------------------------------------------
    sp_extract = subparsers.add_parser(
        "extract",
        help="Extract encoder features for probe evaluation",
        description="Extract 768-dim features from probe_train and test splits.",
    )
    sp_extract.add_argument(
        "--condition",
        type=str,
        default="all",
        help="Condition to extract (default: all)",
    )
    sp_extract.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    sp_extract.set_defaults(func=cmd_extract)

    # -------------------------------------------------------------------------
    # domain subcommand
    # -------------------------------------------------------------------------
    sp_domain = subparsers.add_parser(
        "domain",
        help="Extract domain features for UMAP visualization",
        description="Extract glioma and meningioma feature subsets for domain shift analysis.",
    )
    sp_domain.add_argument(
        "--condition",
        type=str,
        default="all",
        help="Condition to extract (default: all)",
    )
    sp_domain.add_argument(
        "--n-glioma",
        type=int,
        default=200,
        help="Number of glioma samples (default: 200)",
    )
    sp_domain.add_argument(
        "--n-meningioma",
        type=int,
        default=200,
        help="Number of meningioma samples (default: 200)",
    )
    sp_domain.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    sp_domain.set_defaults(func=cmd_domain)

    # -------------------------------------------------------------------------
    # probes subcommand
    # -------------------------------------------------------------------------
    sp_probes = subparsers.add_parser(
        "probes",
        help="Evaluate linear probes on extracted features",
        description="Train and evaluate linear probes to compute R² metrics.",
    )
    sp_probes.add_argument(
        "--condition",
        type=str,
        default="all",
        help="Condition to evaluate (default: all)",
    )
    sp_probes.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Dice evaluation (default: cuda)",
    )
    sp_probes.add_argument(
        "--skip-dice",
        action="store_true",
        help="Skip test Dice evaluation (faster)",
    )
    sp_probes.set_defaults(func=cmd_probes)

    # -------------------------------------------------------------------------
    # analyse subcommand
    # -------------------------------------------------------------------------
    sp_analyse = subparsers.add_parser(
        "analyse",
        aliases=["analyze"],
        help="Run statistical analysis and generate visualizations",
        description="Comprehensive analysis with statistical tests and publication figures.",
    )
    sp_analyse.add_argument(
        "--glioma-features",
        action="store_true",
        help="Include glioma features for domain shift UMAP (requires 'domain' command first)",
    )
    sp_analyse.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation (faster)",
    )
    sp_analyse.set_defaults(func=cmd_analyse)

    # -------------------------------------------------------------------------
    # run-all subcommand
    # -------------------------------------------------------------------------
    sp_run_all = subparsers.add_parser(
        "run-all",
        help="Run complete experiment (all conditions + analysis)",
        description="Full pipeline: splits → train → extract → evaluate → analyse",
    )
    sp_run_all.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        choices=["baseline", "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32"],
        help="Specific conditions to run (default: all)",
    )
    sp_run_all.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs (for quick testing)",
    )
    sp_run_all.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    sp_run_all.add_argument(
        "--domain-features",
        action="store_true",
        help="Also extract domain features for UMAP visualization",
    )
    sp_run_all.add_argument(
        "--n-domain-samples",
        type=int,
        default=200,
        help="Number of domain samples for UMAP (default: 200)",
    )
    sp_run_all.set_defaults(func=cmd_run_all)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Show help if no command given
    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
