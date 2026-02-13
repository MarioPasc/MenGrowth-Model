"""CLI entry point for the LoRA ablation report generator.

Orchestrates data loading, figure generation, narrative construction,
and HTML report building.

Usage:
    python -m experiments.lora_ablation.generate_report \
        --results-dir /path/to/LoRA_Adaptation \
        --output-dir ./report_output \
        --mode both \
        --compare-semantic \
        --skip-umap
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from experiments.lora_ablation.report.data_loader import (
    ExperimentData,
    load_all_experiments,
)
from experiments.lora_ablation.report.figures import (
    FigureResult,
    generate_all_figures,
)
from experiments.lora_ablation.report.html_builder import build_report
from experiments.lora_ablation.report.narrative import (
    SectionContent,
    generate_all_sections,
)
from experiments.lora_ablation.report.style import EXPERIMENT_LABELS

logger = logging.getLogger(__name__)


def _export_summary_json(
    experiments: List[ExperimentData],
    output_path: Path,
) -> None:
    """Export aggregated metrics as JSON for programmatic access.

    Args:
        experiments: Loaded experiments.
        output_path: Path to write summary.json.
    """
    summary: Dict = {"experiments": []}

    for exp in experiments:
        exp_data: Dict = {
            "name": exp.name,
            "adapter_type": exp.adapter_type,
            "semantic_heads": exp.semantic_heads,
            "conditions": {},
        }

        for cond_name, cond in exp.conditions.items():
            exp_data["conditions"][cond_name] = {
                "dice_men": cond.dice_men,
                "dice_gli": cond.dice_gli,
                "domain_metrics": cond.domain_metrics,
                "r2_mean_linear": cond.metrics_enhanced.get("r2_mean_linear"),
                "r2_mean_mlp": cond.metrics_enhanced.get("r2_mean_mlp"),
                "best_val_dice": cond.training_summary.get("best_val_dice"),
                "best_epoch": cond.training_summary.get("best_epoch"),
                "training_time_minutes": cond.training_summary.get("training_time_minutes"),
                "param_counts": cond.training_summary.get("param_counts"),
            }

        summary["experiments"].append(exp_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary JSON written to %s", output_path)


def _export_tables(
    experiments: List[ExperimentData],
    output_dir: Path,
) -> Dict[str, str]:
    """Generate CSV and LaTeX table exports.

    Args:
        experiments: Loaded experiments.
        output_dir: Report output directory.

    Returns:
        Dict of table name -> HTML table string for embedding.
    """
    import pandas as pd

    from experiments.lora_ablation.report.html_builder import html_table

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    html_tables: Dict[str, str] = {}

    for exp in experiments:
        # Build comprehensive results table
        rows = []
        for cond_name, cond in exp.conditions.items():
            row = {
                "Condition": cond_name,
                "Dice MEN": cond.dice_men.get("dice_mean"),
                "Dice GLI": cond.dice_gli.get("dice_mean"),
                "R² Linear": cond.metrics_enhanced.get("r2_mean_linear"),
                "R² MLP": cond.metrics_enhanced.get("r2_mean_mlp"),
                "MMD²": cond.domain_metrics.get("mmd"),
                "Best Epoch": cond.training_summary.get("best_epoch"),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        label = EXPERIMENT_LABELS.get(exp.name, exp.name)

        # Save CSV
        csv_path = tables_dir / f"{exp.name}_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved %s", csv_path)

        # Generate HTML table
        html_tables[f"results_{exp.name}"] = html_table(df, caption=f"Results: {label}")

    return html_tables


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LoRA ablation scientific report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Top-level results directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report_output"),
        help="Output directory for report (default: ./report_output)",
    )
    parser.add_argument(
        "--mode",
        choices=["lora", "dora", "both"],
        default="both",
        help="Which adapter types to include (default: both)",
    )
    parser.add_argument(
        "--compare-semantic",
        action="store_true",
        help="Include semantic vs no-semantic head analysis",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip slow UMAP computation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for report generation."""
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("LoRA Ablation Report Generator")
    logger.info("=" * 60)
    logger.info("Results dir: %s", args.results_dir)
    logger.info("Output dir:  %s", args.output_dir)
    logger.info("Mode:        %s", args.mode)
    logger.info("Semantic:    %s", args.compare_semantic)
    logger.info("Skip UMAP:   %s", args.skip_umap)

    # 1. Load experiments
    logger.info("\n--- Loading experiments ---")
    experiments = load_all_experiments(
        args.results_dir,
        mode=args.mode,
        compare_semantic=args.compare_semantic,
    )

    if not experiments:
        logger.error("No experiments found in %s", args.results_dir)
        return

    logger.info("Loaded %d experiment(s)", len(experiments))

    # 2. Generate figures
    logger.info("\n--- Generating figures ---")
    figures = generate_all_figures(
        experiments=experiments,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        skip_umap=args.skip_umap,
        compare_semantic=args.compare_semantic,
    )
    logger.info("Generated %d figures", len(figures))

    # 3. Generate narrative
    logger.info("\n--- Generating narrative ---")
    sections = generate_all_sections(
        experiments=experiments,
        compare_semantic=args.compare_semantic,
    )
    logger.info("Generated %d sections", len(sections))

    # 4. Generate tables
    logger.info("\n--- Generating tables ---")
    tables = _export_tables(experiments, args.output_dir)
    logger.info("Generated %d tables", len(tables))

    # 5. Build HTML report
    logger.info("\n--- Building HTML report ---")
    report_path = build_report(
        sections=sections,
        figures=figures,
        tables=tables,
        output_path=args.output_dir / "report.html",
        mode=args.mode,
        num_experiments=len(experiments),
    )
    logger.info("Report: %s", report_path)

    # 6. Export summary JSON
    logger.info("\n--- Exporting summary ---")
    _export_summary_json(experiments, args.output_dir / "data" / "summary.json")

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info("Report:  %s", report_path)
    logger.info("Figures: %s", args.output_dir / "figures")
    logger.info("Tables:  %s", args.output_dir / "tables")
    logger.info("Data:    %s", args.output_dir / "data" / "summary.json")
