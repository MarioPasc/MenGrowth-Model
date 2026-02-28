#!/usr/bin/env python
# experiments/lora_ablation/regenerate_analysis.py
"""Standalone entry point for regenerating all v3 analysis outputs.

Reorganizes per-condition raw data into a clean output structure with
numbered figures, LaTeX tables, aggregated CSVs, and Markdown reports.

Pipeline:
    1. Create output directories
    2. Precompute figure cache (from raw .pt / .json / .csv)
    3. Generate 8 thesis-quality figures
    4. Aggregate CSVs (comprehensive, feature_quality, dice_summary)
    5. Generate LaTeX tables
    6. Copy/generate reports

Usage:
    python -m experiments.lora_ablation.regenerate_analysis \
        --config experiments/lora_ablation/config/ablation_v3.yaml

    python -m experiments.lora_ablation.regenerate_analysis \
        --config experiments/lora_ablation/config/ablation_v3.yaml \
        --skip-cache          # reuse existing cache
        --figures-only        # only regenerate figures
        --tables-only         # only regenerate tables
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import yaml

from .v3_cache import precompute_all
from .v3_figures import generate_all_v3_figures
from .generate_tables import (
    load_all_metrics,
    generate_comprehensive_csv,
    generate_comprehensive_latex,
    generate_simplified_latex,
    generate_domain_shift_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Directory creation
# ============================================================================

def _create_output_dirs(output_dir: Path) -> None:
    """Create the reorganized output directory tree."""
    dirs = [
        output_dir / "results" / "figure_cache",
        output_dir / "tables",
        output_dir / "figures" / "png",
        output_dir / "reports",
        output_dir / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Table generation (redirected to tables/ dir)
# ============================================================================

def _generate_tables(config: dict, output_dir: Path) -> None:
    """Generate LaTeX and CSV tables in the tables/ directory."""
    metrics = load_all_metrics(config)
    if not metrics:
        logger.warning("No metrics found for table generation.")
        return

    tables_dir = output_dir / "tables"
    results_dir = output_dir / "results"

    # CSVs go to results/
    generate_comprehensive_csv(
        metrics, config, results_dir / "comprehensive.csv"
    )
    generate_domain_shift_csv(
        metrics, config, results_dir / "domain_shift.csv"
    )

    # Also generate feature quality and dice CSVs if data exists
    _generate_feature_quality_csv(config, output_dir, results_dir)
    _generate_dice_summary_csv(config, output_dir, results_dir)

    # LaTeX goes to tables/
    generate_comprehensive_latex(
        metrics, config, tables_dir / "tab_comprehensive.tex"
    )
    generate_simplified_latex(
        metrics, config, tables_dir / "tab_main_results.tex"
    )

    # Feature quality LaTeX
    _generate_feature_quality_latex(config, output_dir, tables_dir)

    logger.info(f"Tables saved to {tables_dir}")


def _generate_feature_quality_csv(
    config: dict,
    output_dir: Path,
    results_dir: Path,
) -> None:
    """Aggregate feature_quality.json from all conditions into a single CSV."""
    import csv

    rows = []
    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        fq_path = output_dir / "conditions" / name / "feature_quality.json"
        if not fq_path.exists():
            continue
        with open(fq_path) as f:
            fq = json.load(f)
        row = {"condition": name}
        row.update(fq)
        rows.append(row)

    if not rows:
        return

    out_path = results_dir / "feature_quality.csv"
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {out_path}")


def _generate_dice_summary_csv(
    config: dict,
    output_dir: Path,
    results_dir: Path,
) -> None:
    """Aggregate test_dice_men.json from all conditions into a summary CSV."""
    import csv

    rows = []
    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        dice_path = output_dir / "conditions" / name / "test_dice_men.json"
        if not dice_path.exists():
            continue
        with open(dice_path) as f:
            dice = json.load(f)
        row = {"condition": name}
        row.update(dice)
        rows.append(row)

    if not rows:
        return

    out_path = results_dir / "dice_summary.csv"
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {out_path}")


def _generate_feature_quality_latex(
    config: dict,
    output_dir: Path,
    tables_dir: Path,
) -> None:
    """Generate feature quality LaTeX table."""
    from datetime import datetime
    from experiments.utils.settings import get_label

    fq_by_cond = {}
    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        fq_path = output_dir / "conditions" / name / "feature_quality.json"
        if fq_path.exists():
            with open(fq_path) as f:
                fq_by_cond[name] = json.load(f)

    if not fq_by_cond:
        return

    lines = [
        r"% Auto-generated by regenerate_analysis.py",
        f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        r"",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Feature quality metrics across encoder adaptation conditions.}",
        r"\label{tab:feature_quality}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Condition & Eff. Rank & Mean $|r|$ & PCA@50 & Collapsed & DCI-D & DCI-C \\",
        r"\midrule",
    ]

    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        if name not in fq_by_cond:
            continue
        fq = fq_by_cond[name]
        label = get_label(name)

        rank = fq.get("effective_rank", 0)
        corr = fq.get("mean_interdim_corr", fq.get("mean_abs_correlation", 0))
        pca50 = fq.get("pca_explained_variance", {}).get("50", 0)
        collapsed = fq.get("collapsed_dims", fq.get("n_collapsed", 0))
        dci_d = fq.get("dci_disentanglement", fq.get("dci_D", 0))
        dci_c = fq.get("dci_completeness", fq.get("dci_C", 0))

        def fmt(v: float, decimals: int = 3) -> str:
            return f"{v:.{decimals}f}" if v is not None else "---"

        lines.append(
            f"{label} & {fmt(rank, 1)} & {fmt(corr)} & {fmt(pca50)} & "
            f"{int(collapsed) if collapsed else 0} & {fmt(dci_d)} & {fmt(dci_c)} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = tables_dir / "tab_feature_quality.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved {out_path}")


# ============================================================================
# Report generation
# ============================================================================

def _generate_reports(config: dict, output_dir: Path) -> None:
    """Copy/generate analysis reports into the reports/ directory."""
    reports_dir = output_dir / "reports"

    # Copy existing analysis_report.md if present
    src_report = output_dir / "analysis_report.md"
    if src_report.exists():
        dst = reports_dir / "analysis.md"
        dst.write_text(src_report.read_text())
        logger.info(f"Copied analysis report to {dst}")

    # Copy diagnostics
    src_diag = output_dir / "diagnostics_report.txt"
    if src_diag.exists():
        dst = reports_dir / "diagnostics.txt"
        dst.write_text(src_diag.read_text())

    # Generate recommendation
    _generate_recommendation(config, output_dir, reports_dir)


def _generate_recommendation(
    config: dict,
    output_dir: Path,
    reports_dir: Path,
) -> None:
    """Generate a short recommendation based on probe metrics."""
    from experiments.utils.settings import get_label

    best_name: Optional[str] = None
    best_r2: float = -float("inf")

    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        metrics_path = output_dir / "conditions" / name / "metrics_enhanced.json"
        if not metrics_path.exists():
            metrics_path = output_dir / "conditions" / name / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        r2 = m.get("r2_mean", m.get("r2_mean_mlp", -1.0))
        if r2 is not None and r2 > best_r2:
            best_r2 = r2
            best_name = name

    lines = [
        "Recommendation",
        "=" * 40,
        "",
    ]
    if best_name:
        lines.append(f"Best condition: {get_label(best_name)} (r2_mean={best_r2:.4f})")
        lines.append(f"  Config name: {best_name}")
    else:
        lines.append("No metrics found. Run evaluation pipeline first.")

    out_path = reports_dir / "recommendation.txt"
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved {out_path}")


# ============================================================================
# Main pipeline
# ============================================================================

def main(
    config_path: str,
    skip_cache: bool = False,
    figures_only: bool = False,
    tables_only: bool = False,
) -> None:
    """Run the full regeneration pipeline.

    Args:
        config_path: Path to experiment YAML config.
        skip_cache: If True, reuse existing figure cache.
        figures_only: Only regenerate figures.
        tables_only: Only regenerate tables.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])

    logger.info("=" * 60)
    logger.info("REGENERATE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")

    # 1. Create directories
    _create_output_dirs(output_dir)

    # 2. Precompute cache
    cache_dir = output_dir / "results" / "figure_cache"
    if not skip_cache and not tables_only:
        precompute_all(config, output_dir)

    # 3. Generate figures
    if not tables_only:
        figures_dir = output_dir / "figures"
        generate_all_v3_figures(cache_dir, figures_dir, config)

    # 4. Generate tables
    if not figures_only:
        _generate_tables(config, output_dir)

    # 5. Generate reports
    if not figures_only and not tables_only:
        _generate_reports(config, output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("REGENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  figures/    : 11 PDF + PNG")
    logger.info(f"  tables/     : LaTeX tables")
    logger.info(f"  results/    : aggregated CSVs")
    logger.info(f"  reports/    : analysis + recommendation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate all v3 analysis outputs (figures, tables, reports)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--skip-cache", action="store_true",
        help="Reuse existing figure cache (skip precomputation)",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Only regenerate figures",
    )
    parser.add_argument(
        "--tables-only", action="store_true",
        help="Only regenerate tables",
    )
    args = parser.parse_args()
    main(args.config, args.skip_cache, args.figures_only, args.tables_only)
