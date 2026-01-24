"""Command-line interface for analysis pipeline.

Usage:
    python -m vae.tools.analysis analyze /path/to/run_dir
    python -m vae.tools.analysis stats /path/to/run_dir
    python -m vae.tools.analysis visualize /path/to/run_dir
    python -m vae.tools.analysis compare run1_dir run2_dir --output comparison/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .pipeline import run_stage1, run_stage2, run_full_pipeline, run_comparison
from .loaders import validate_experiment_directory

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run full analysis pipeline."""
    run_dir = args.run_dir

    # Validate
    is_valid, found, missing = validate_experiment_directory(run_dir)
    if not is_valid:
        logger.error(f"Invalid experiment directory: {run_dir}")
        logger.error(f"Missing required files: {missing}")
        return 1

    logger.info(f"Found {len(found)} experiment files")

    try:
        result = run_full_pipeline(
            run_dir=run_dir,
            output_dir=args.output,
            dpi=args.dpi,
            format=args.format,
        )

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Output directory: {result['output_dir']}")
        print(f"Stage 1 files: {len(result['files'])}")
        print(f"Stage 2 plots: {len(result['plots'])}")

        # Print summary
        summary = result["summary"]
        print(f"\nOverall Grade: {summary.overall_grade.value}")
        print(f"ODE Ready: {'Yes' if summary.ready_for_ode else 'No'}")
        print(f"\nKey Metrics:")
        print(f"  Volume R²: {summary.performance.vol_r2:.3f} (target: 0.85)")
        print(f"  Location R²: {summary.performance.loc_r2:.3f} (target: 0.90)")
        print(f"  Shape R²: {summary.performance.shape_r2:.3f} (target: 0.35)")
        print(f"  Max Cross-Corr: {summary.ode_utility.max_cross_corr:.3f} (target: <0.30)")

        if summary.warnings:
            print(f"\nWarnings ({len(summary.warnings)}):")
            for w in summary.warnings[:3]:
                print(f"  - {w[:80]}...")

        if summary.recommendations:
            print(f"\nRecommendations ({len(summary.recommendations)}):")
            for r in summary.recommendations[:3]:
                print(f"  - {r[:80]}...")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Run Stage 1 only (statistics)."""
    try:
        result = run_stage1(
            run_dir=args.run_dir,
            output_dir=args.output,
        )

        print(f"\nStage 1 complete. Generated {len(result['files'])} files.")
        print(f"Output: {result['output_dir']}")

        return 0

    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_visualize(args: argparse.Namespace) -> int:
    """Run Stage 2 only (visualization)."""
    try:
        result = run_stage2(
            run_dir=args.run_dir,
            output_dir=args.output,
            dpi=args.dpi,
            format=args.format,
        )

        print(f"\nStage 2 complete. Generated {len(result['plots'])} plots.")
        print(f"Output: {result['output_dir']}/plots/")

        return 0

    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Run comparison across multiple runs."""
    run_dirs = args.run_dirs
    run_names = args.names

    if len(run_dirs) < 2:
        logger.error("Need at least 2 run directories for comparison")
        return 1

    if run_names is not None and len(run_names) != len(run_dirs):
        logger.error(
            f"--names count ({len(run_names)}) must match "
            f"run_dirs count ({len(run_dirs)})"
        )
        return 1

    try:
        result = run_comparison(
            run_dirs=run_dirs,
            output_dir=args.output,
            run_names=run_names,
            include_statistical_tests=not args.no_stats,
            dpi=args.dpi,
            format=args.format,
        )

        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)

        comparison = result["comparison"]
        name_map = result.get("name_map", {})

        print(f"Compared {len(comparison.run_ids)} runs")
        print(f"Output: {result['output_dir']}")

        best_id = result["best_run"]
        best_name = name_map.get(best_id, best_id)
        print(f"\nBest run: {best_name}")

        # Print comparison table
        print("\nRun Grades:")
        for run_id, summary in comparison.summaries.items():
            display_name = name_map.get(run_id, run_id)
            short_name = display_name[:30] + "..." if len(display_name) > 30 else display_name
            print(f"  {short_name}: {summary.overall_grade.value}")

        # Print key metrics
        print("\nKey Metrics:")
        print(f"  {'Run':<20} {'Vol R2':>8} {'Loc R2':>8} {'Shape R2':>9} {'ODE Score':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
        for run_id, summary in comparison.summaries.items():
            display_name = name_map.get(run_id, run_id)
            short_name = display_name[:20]
            print(
                f"  {short_name:<20} "
                f"{summary.performance.vol_r2:>8.3f} "
                f"{summary.performance.loc_r2:>8.3f} "
                f"{summary.performance.shape_r2:>9.3f} "
                f"{summary.ode_utility.ode_readiness:>10.3f}"
            )

        return 0

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="vae.tools.analysis",
        description="SemiVAE Experiment Analysis Pipeline",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run full analysis pipeline (Stage 1 + Stage 2)",
    )
    analyze_parser.add_argument(
        "run_dir",
        help="Path to experiment run directory",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: {run_dir}/analysis)",
    )
    analyze_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot resolution (default: 150)",
    )
    analyze_parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Image format (default: png)",
    )

    # stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Run Stage 1 only (statistics)",
    )
    stats_parser.add_argument(
        "run_dir",
        help="Path to experiment run directory",
    )
    stats_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory",
    )

    # visualize command
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Run Stage 2 only (visualization)",
    )
    viz_parser.add_argument(
        "run_dir",
        help="Path to experiment run directory",
    )
    viz_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory",
    )
    viz_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot resolution",
    )
    viz_parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Image format",
    )

    # compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple experiment runs",
    )
    compare_parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Paths to experiment run directories (at least 2)",
    )
    compare_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for comparison results",
    )
    compare_parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip statistical tests",
    )
    compare_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot resolution",
    )
    compare_parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Image format",
    )
    compare_parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Human-readable names for runs (must match number of run_dirs)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    # Dispatch to command handler
    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "visualize":
        return cmd_visualize(args)
    elif args.command == "compare":
        return cmd_compare(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
