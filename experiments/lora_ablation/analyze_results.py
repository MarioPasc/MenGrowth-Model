#!/usr/bin/env python
# experiments/lora_ablation/analyze_results.py
"""Statistical analysis and final comparison of LoRA ablation results.

This script:
1. Loads metrics from all conditions
2. Creates detailed comparison tables
3. Computes statistical significance (if applicable)
4. Generates a comprehensive analysis report

Usage:
    python -m experiments.lora_ablation.analyze_results \
        --config experiments/lora_ablation/config/ablation.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_all_metrics(config: dict) -> Dict[str, Dict]:
    """Load metrics from all conditions.

    Returns:
        Dict mapping condition names to metrics dicts.
    """
    output_dir = Path(config["experiment"]["output_dir"])
    all_metrics = {}

    for cond in config["conditions"]:
        condition_name = cond["name"]
        metrics_path = output_dir / "conditions" / condition_name / "metrics.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                all_metrics[condition_name] = json.load(f)

            # Also load training summary if available
            summary_path = output_dir / "conditions" / condition_name / "training_summary.yaml"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = yaml.safe_load(f)
                all_metrics[condition_name].update({
                    "val_dice": summary.get("best_val_dice"),
                    "best_epoch": summary.get("best_epoch"),
                    "training_time_minutes": summary.get("training_time_minutes"),
                    "param_counts": summary.get("param_counts"),
                })
        else:
            logger.warning(f"Metrics not found for {condition_name}")

    return all_metrics


def compute_deltas(metrics: Dict[str, Dict]) -> pd.DataFrame:
    """Compute differences from baseline for all conditions.

    Returns:
        DataFrame with delta values.
    """
    if "baseline" not in metrics:
        logger.warning("Baseline not found, cannot compute deltas")
        return pd.DataFrame()

    baseline = metrics["baseline"]
    rows = []

    for condition_name, cond_metrics in metrics.items():
        if condition_name == "baseline":
            continue

        row = {
            "Condition": condition_name,
            "ΔR²_vol": cond_metrics.get("r2_volume", 0) - baseline.get("r2_volume", 0),
            "ΔR²_loc": cond_metrics.get("r2_location", 0) - baseline.get("r2_location", 0),
            "ΔR²_shape": cond_metrics.get("r2_shape", 0) - baseline.get("r2_shape", 0),
            "ΔR²_mean": cond_metrics.get("r2_mean", 0) - baseline.get("r2_mean", 0),
            "ΔVal_Dice": (cond_metrics.get("val_dice", 0) or 0) - (baseline.get("val_dice", 0) or 0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def generate_analysis_report(config: dict, metrics: Dict[str, Dict]) -> str:
    """Generate comprehensive analysis report.

    Args:
        config: Configuration dict.
        metrics: Dict of all condition metrics.

    Returns:
        Markdown-formatted report.
    """
    lines = [
        "# LoRA Ablation Analysis Report",
        "",
        "## Experiment Configuration",
        "",
        f"- **Seed**: {config['experiment']['seed']}",
        f"- **Data Root**: {config['paths']['data_root']}",
        f"- **Checkpoint**: {config['paths']['checkpoint']}",
        "",
        "### Data Splits",
        f"- LoRA Training: {config['data_splits']['lora_train']} subjects",
        f"- LoRA Validation: {config['data_splits']['lora_val']} subjects",
        f"- Probe Training: {config['data_splits']['probe_train']} subjects",
        f"- Test: {config['data_splits']['test']} subjects",
        "",
        "---",
        "",
        "## Results Summary",
        "",
    ]

    # Main results table
    lines.append("### Linear Probe R² Scores (Test Set)")
    lines.append("")
    lines.append("| Condition | R²_vol | R²_loc | R²_shape | R²_mean | Val Dice |")
    lines.append("|-----------|--------|--------|----------|---------|----------|")

    for cond in config["conditions"]:
        name = cond["name"]
        if name in metrics:
            m = metrics[name]
            r2_vol = f"{m.get('r2_volume', 0):.4f}"
            r2_loc = f"{m.get('r2_location', 0):.4f}"
            r2_shape = f"{m.get('r2_shape', 0):.4f}"
            r2_mean = f"{m.get('r2_mean', 0):.4f}"
            val_dice = f"{m.get('val_dice', 0):.4f}" if m.get('val_dice') else "N/A"
            lines.append(f"| {name} | {r2_vol} | {r2_loc} | {r2_shape} | {r2_mean} | {val_dice} |")

    lines.append("")

    # Delta table
    if "baseline" in metrics:
        lines.append("### Improvement over Baseline")
        lines.append("")
        lines.append("| Condition | ΔR²_vol | ΔR²_loc | ΔR²_shape | ΔR²_mean |")
        lines.append("|-----------|---------|---------|-----------|----------|")

        baseline = metrics["baseline"]
        for cond in config["conditions"]:
            name = cond["name"]
            if name == "baseline" or name not in metrics:
                continue

            m = metrics[name]
            delta_vol = m.get('r2_volume', 0) - baseline.get('r2_volume', 0)
            delta_loc = m.get('r2_location', 0) - baseline.get('r2_location', 0)
            delta_shape = m.get('r2_shape', 0) - baseline.get('r2_shape', 0)
            delta_mean = m.get('r2_mean', 0) - baseline.get('r2_mean', 0)

            sign_vol = "+" if delta_vol >= 0 else ""
            sign_loc = "+" if delta_loc >= 0 else ""
            sign_shape = "+" if delta_shape >= 0 else ""
            sign_mean = "+" if delta_mean >= 0 else ""

            lines.append(
                f"| {name} | {sign_vol}{delta_vol:.4f} | {sign_loc}{delta_loc:.4f} | "
                f"{sign_shape}{delta_shape:.4f} | {sign_mean}{delta_mean:.4f} |"
            )

        lines.append("")

    # Parameter counts
    lines.append("### Model Parameters")
    lines.append("")
    lines.append("| Condition | Trainable Params | Description |")
    lines.append("|-----------|-----------------|-------------|")

    for cond in config["conditions"]:
        name = cond["name"]
        desc = cond.get("description", "")
        if name in metrics and metrics[name].get("param_counts"):
            params = metrics[name]["param_counts"]["total"]
            lines.append(f"| {name} | {params:,} | {desc} |")
        else:
            lines.append(f"| {name} | N/A | {desc} |")

    lines.append("")

    # Feature analysis
    lines.append("### Feature Quality Analysis")
    lines.append("")
    lines.append("| Condition | Variance (mean) | Variance (min) |")
    lines.append("|-----------|-----------------|----------------|")

    for cond in config["conditions"]:
        name = cond["name"]
        if name in metrics:
            m = metrics[name]
            var_mean = f"{m.get('variance_mean', 0):.4f}"
            var_min = f"{m.get('variance_min', 0):.6f}"
            lines.append(f"| {name} | {var_mean} | {var_min} |")

    lines.append("")

    # Per-dimension analysis
    lines.append("### Per-Dimension R² (Volume)")
    lines.append("")
    lines.append("Volume targets: [Total, NCR, ED, ET]")
    lines.append("")

    for cond in config["conditions"]:
        name = cond["name"]
        if name in metrics:
            m = metrics[name]
            per_dim = m.get("r2_volume_per_dim", [])
            if per_dim:
                formatted = ", ".join([f"{x:.4f}" for x in per_dim])
                lines.append(f"- **{name}**: [{formatted}]")

    lines.append("")

    # Decision
    lines.append("---")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")

    # Find best condition
    best_name = "baseline"
    best_r2_mean = metrics.get("baseline", {}).get("r2_mean", 0)

    for name, m in metrics.items():
        if m.get("r2_mean", 0) > best_r2_mean:
            best_name = name
            best_r2_mean = m.get("r2_mean", 0)

    baseline_r2_mean = metrics.get("baseline", {}).get("r2_mean", 0)
    improvement = best_r2_mean - baseline_r2_mean

    if best_name == "baseline" or improvement < 0.03:
        lines.append("**RECOMMENDATION: Use Baseline (No LoRA)**")
        lines.append("")
        lines.append("The baseline encoder already provides strong semantic features.")
        lines.append("LoRA adaptation does not provide sufficient improvement to justify")
        lines.append("the added complexity and training cost.")
    elif improvement < 0.05:
        lines.append(f"**RECOMMENDATION: Consider {best_name} (Marginal Improvement)**")
        lines.append("")
        lines.append(f"LoRA adaptation with {best_name} provides {improvement*100:.1f}% improvement")
        lines.append("over baseline. This is a marginal gain that may or may not be")
        lines.append("worth the additional training cost depending on your use case.")
    else:
        lines.append(f"**RECOMMENDATION: Use {best_name} (Significant Improvement)**")
        lines.append("")
        lines.append(f"LoRA adaptation with {best_name} provides {improvement*100:.1f}% improvement")
        lines.append("over baseline. This significant gain justifies the additional")
        lines.append("training complexity for downstream growth modeling.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by analyze_results.py*")

    return "\n".join(lines)


def analyze_results(config_path: str) -> None:
    """Run analysis and generate report.

    Args:
        config_path: Path to ablation.yaml.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])

    # Load all metrics
    logger.info("Loading metrics from all conditions...")
    metrics = load_all_metrics(config)

    if not metrics:
        logger.error("No metrics found. Run the experiment first.")
        return

    logger.info(f"Loaded metrics for {len(metrics)} conditions")

    # Generate report
    logger.info("Generating analysis report...")
    report = generate_analysis_report(config, metrics)

    # Save report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved report to {report_path}")

    # Print report
    print("\n" + report)

    # Also save comparison CSV
    rows = []
    for cond in config["conditions"]:
        name = cond["name"]
        if name in metrics:
            m = metrics[name]
            row = {
                "condition": name,
                "r2_volume": m.get("r2_volume"),
                "r2_location": m.get("r2_location"),
                "r2_shape": m.get("r2_shape"),
                "r2_mean": m.get("r2_mean"),
                "val_dice": m.get("val_dice"),
                "variance_mean": m.get("variance_mean"),
                "variance_min": m.get("variance_min"),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison table to {csv_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze LoRA ablation experiment results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )

    args = parser.parse_args()
    analyze_results(args.config)


if __name__ == "__main__":
    main()
