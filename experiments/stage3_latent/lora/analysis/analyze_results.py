#!/usr/bin/env python
# experiments/lora/analysis/analyze_results.py
"""Comprehensive analysis and report generation for LoRA ablation study.

This script integrates:
1. Statistical analysis with hypothesis testing
2. Publication-quality visualizations
3. LaTeX-ready tables and figures
4. Evidence-based recommendation for thesis

Usage:
    # Full analysis with all outputs
    python -m experiments.lora.analysis.analyze_results \
        --config experiments/lora_ablation/config/ablation.yaml
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .statistical_analysis import (
    AblationStatistics,
    run_statistical_analysis,
)
from .statistical_analysis import (
    save_results as save_statistical_results,
)
from .visualizations import generate_all_figures

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Canonical ordered list of all possible semantic features.
_ALL_SEMANTIC_FEATURES = ["volume", "location", "shape"]

# Short labels used in LaTeX / Markdown column headers.
_FEATURE_SHORT = {"volume": "Vol", "location": "Loc", "shape": "Shape"}


def _discover_available_features(metrics: dict[str, dict], suffix: str = "") -> list[str]:
    """Return the subset of semantic features present in *any* condition's metrics.

    A feature is considered available if at least one condition has a non-None,
    non-zero value for the key ``r2_{feature}{suffix}``.

    Args:
        metrics: Mapping of condition name to its metrics dict.
        suffix: Optional key suffix, e.g. ``"_rbf"`` for GP-RBF probe keys.

    Returns:
        Ordered list of available feature names (subset of ``_ALL_SEMANTIC_FEATURES``).
    """
    available: list[str] = []
    for feat in _ALL_SEMANTIC_FEATURES:
        key = f"r2_{feat}{suffix}"
        if any(m.get(key) not in (None, 0) for m in metrics.values()):
            available.append(feat)
    return available


def validate_experiment_outputs(config: dict) -> dict[str, dict[str, bool]]:
    """Pre-flight check: validate which experiment outputs exist.

    Returns a dict mapping condition -> {file_type: exists}.
    """
    output_dir = Path(config["experiment"]["output_dir"])
    status = {}

    # Base expected files for all conditions
    base_expected_files = [
        ("training_log", "training_log.csv"),
        ("training_summary", "training_summary.yaml"),
        ("features_probe", "features_probe.pt"),
        ("features_test", "features_test.pt"),
        ("targets_probe", "targets_probe.pt"),
        ("targets_test", "targets_test.pt"),
        ("metrics", "metrics.json"),
    ]

    logger.info("Pre-flight validation of experiment outputs:")
    logger.info("-" * 60)

    all_complete = True
    for cond in config["conditions"]:
        name = cond["name"]
        cond_dir = output_dir / "conditions" / name
        status[name] = {}

        # Build expected files list based on condition type
        expected_files = list(base_expected_files)

        # Checkpoint: accept either best_model.pt or checkpoint.pt
        checkpoint_exists = (cond_dir / "best_model.pt").exists() or (
            cond_dir / "checkpoint.pt"
        ).exists()
        status[name]["checkpoint"] = checkpoint_exists

        # Adapter: only expected for LoRA conditions (lora_rank is not None)
        lora_rank = cond.get("lora_rank")
        if lora_rank is not None:
            expected_files.append(("adapter", "adapter"))

        missing = []
        if not checkpoint_exists:
            missing.append("checkpoint")

        for file_type, filename in expected_files:
            path = cond_dir / filename
            exists = path.exists()
            status[name][file_type] = exists
            if not exists:
                missing.append(file_type)

        if missing:
            all_complete = False
            logger.warning(f"  {name}: INCOMPLETE - missing {missing}")
        else:
            logger.info(f"  {name}: OK (all files present)")

    logger.info("-" * 60)
    if not all_complete:
        logger.warning("Some conditions are incomplete. Analysis will proceed with available data.")

    return status


def load_all_metrics(config: dict) -> dict[str, dict]:
    """Load metrics from all conditions."""
    output_dir = Path(config["experiment"]["output_dir"])
    all_metrics = {}

    for cond in config["conditions"]:
        condition_name = cond["name"]
        cond_dir = output_dir / "conditions" / condition_name

        metrics = {}

        # Load probe metrics
        metrics_path = cond_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics.update(json.load(f))

        # Load training summary
        summary_path = cond_dir / "training_summary.yaml"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = yaml.safe_load(f)
            metrics.update(
                {
                    "val_dice": summary.get("best_val_dice"),
                    "best_epoch": summary.get("best_epoch"),
                    "training_time_minutes": summary.get("training_time_minutes"),
                    "param_counts": summary.get("param_counts"),
                }
            )

        if metrics:
            all_metrics[condition_name] = metrics
            logger.info(f"Loaded metrics for {condition_name}")

    return all_metrics


def generate_latex_table(
    metrics: dict[str, dict],
    config: dict,
    stats: AblationStatistics | None = None,
    include_rbf: bool = True,
) -> str:
    """Generate LaTeX table for thesis.

    Args:
        metrics: Dict of condition -> metrics.
        config: Experiment configuration.
        stats: Optional statistical analysis results.
        include_rbf: Whether to include GP-RBF probe results (creates wider table).

    Returns:
        Publication-ready LaTeX code.
    """
    # Discover which semantic features are present in the metrics
    lin_features = _discover_available_features(metrics)
    has_rbf_data = include_rbf and any("r2_mean_rbf" in m for m in metrics.values())
    rbf_features = _discover_available_features(metrics, suffix="_rbf") if has_rbf_data else []

    n_lin = len(lin_features)
    n_rbf = len(rbf_features)

    if has_rbf_data and rbf_features:
        # Extended table with both Linear and GP-RBF probes
        lin_col_spec = "c" * n_lin
        rbf_col_spec = "c" * n_rbf
        lin_header_cells = " & ".join(_FEATURE_SHORT[f] for f in lin_features)
        rbf_header_cells = " & ".join(_FEATURE_SHORT[f] for f in rbf_features)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Comparison of encoder adaptation strategies for meningioma feature learning. "
            r"Both linear and GP-RBF probe $R^2$ scores are reported. "
            r"$\Delta$ indicates improvement over baseline.}",
            r"\label{tab:lora_ablation}",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{l{lin_col_spec}|{rbf_col_spec}|cc}}",
            r"\toprule",
            rf" & \multicolumn{{{n_lin}}}{{c|}}{{Linear Probe $R^2$}} & \multicolumn{{{n_rbf}}}{{c|}}{{GP-RBF Probe $R^2$}} & & \\",
            rf"Condition & {lin_header_cells} & {rbf_header_cells} & $R^2_\mathrm{{mean}}$ (Lin) & Params \\",
            r"\midrule",
        ]
    else:
        # Table without GP-RBF
        feat_col_spec = "c" * n_lin
        feat_header = " & ".join(
            rf"$R^2_\mathrm{{{_FEATURE_SHORT[f].lower()}}}$" for f in lin_features
        )
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Comparison of encoder adaptation strategies for meningioma feature learning. "
            r"Linear probe $R^2$ scores measure semantic feature predictability from encoder features. "
            r"$\Delta$ indicates improvement over baseline. Statistical significance tested with "
            r"Wilcoxon signed-rank test (Holm-Bonferroni corrected).}",
            r"\label{tab:lora_ablation}",
            rf"\begin{{tabular}}{{l{feat_col_spec}ccc}}",
            r"\toprule",
            rf"Condition & {feat_header} & "
            r"$R^2_\mathrm{mean}$ & Test Dice & Params \\",
            r"\midrule",
        ]

    baseline_r2 = metrics.get("baseline", {}).get("r2_mean", 0)

    for cond in config["conditions"]:
        name = cond["name"]
        if name not in metrics:
            continue

        m = metrics[name]

        # Format linear R² values for available features
        lin_cells = [f"{m.get(f'r2_{feat}', 0):.3f}" for feat in lin_features]
        r2_mean = f"{m.get('r2_mean', 0):.3f}"

        # Prefer test Dice, fall back to validation Dice
        test_dice = m.get("test_dice_mean") or m.get("val_dice")
        dice_str = f"{test_dice:.3f}" if test_dice else "---"

        # Format params
        params = m.get("param_counts", {})
        if isinstance(params, dict):
            total_params = params.get("total", 0)
            if total_params > 1e6:
                params_str = f"{total_params / 1e6:.1f}M"
            else:
                params_str = f"{total_params / 1e3:.0f}K"
        else:
            params_str = "---"

        # Add delta annotation for non-baseline
        if name != "baseline":
            delta = m.get("r2_mean", 0) - baseline_r2
            delta_sign = "+" if delta >= 0 else ""
            r2_mean = f"{m.get('r2_mean', 0):.3f} ({delta_sign}{delta:.3f})"

        # Condition label
        if name == "baseline":
            label = "Baseline (Frozen)"
        else:
            rank = cond.get("lora_rank", "?")
            label = f"LoRA $r={rank}$"

        lin_str = " & ".join(lin_cells)

        if has_rbf_data and rbf_features:
            rbf_cells = [f"{m.get(f'r2_{feat}_rbf', 0):.3f}" for feat in rbf_features]
            rbf_str = " & ".join(rbf_cells)
            lines.append(f"{label} & {lin_str} & {rbf_str} & {r2_mean} & {params_str} \\\\")
        else:
            lines.append(f"{label} & {lin_str} & {r2_mean} & {dice_str} & {params_str} \\\\")

    if has_rbf_data and rbf_features:
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}}",
                r"\end{table}",
            ]
        )
    else:
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

    return "\n".join(lines)


def generate_markdown_report(
    config: dict,
    metrics: dict[str, dict],
    stats: AblationStatistics | None = None,
) -> str:
    """Generate comprehensive Markdown report for thesis appendix."""
    lines = [
        "# LoRA Ablation Study: Comprehensive Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Experiment Configuration",
        "",
        f"- **Seed:** {config['experiment']['seed']}",
        f"- **Data Root:** `{config['paths']['data_root']}`",
        f"- **Checkpoint:** `{config['paths']['checkpoint']}`",
        "",
        "### Data Splits",
        f"- LoRA Training: {config['data_splits']['lora_train']} subjects",
        f"- LoRA Validation: {config['data_splits']['lora_val']} subjects",
        f"- Test Set: {config['data_splits']['test']} subjects",
        "",
        "### Experimental Conditions",
        "",
        "| Condition | LoRA Rank | LoRA Alpha | Description |",
        "|-----------|-----------|------------|-------------|",
    ]

    for cond in config["conditions"]:
        name = cond["name"]
        rank = cond.get("lora_rank", "None")
        alpha = cond.get("lora_alpha", "N/A")
        desc = cond.get("description", "")
        lines.append(f"| {name} | {rank} | {alpha} | {desc} |")

    lin_features = _discover_available_features(metrics)
    feat_names_str = ", ".join(lin_features)

    # Build dynamic table header
    feat_header_cells = " | ".join(f"R²_{_FEATURE_SHORT[f].lower()}" for f in lin_features)
    feat_separator = " | ".join("--------" for _ in lin_features)
    header_row = f"| Condition | {feat_header_cells} | R²_mean | ΔR²_mean |"
    sep_row = f"|-----------|{feat_separator}|---------|----------|"

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Primary Results: Linear Probe R²",
            "",
            "The primary evaluation metric is R² from linear probes trained to predict ",
            f"semantic features ({feat_names_str}) from encoder features.",
            "",
            header_row,
            sep_row,
        ]
    )

    baseline_r2 = metrics.get("baseline", {}).get("r2_mean", 0)

    for cond in config["conditions"]:
        name = cond["name"]
        if name not in metrics:
            continue

        m = metrics[name]
        feat_cells = " | ".join(f"{m.get(f'r2_{feat}', 0):.4f}" for feat in lin_features)
        r2_mean = f"{m.get('r2_mean', 0):.4f}"

        if name == "baseline":
            delta = "---"
        else:
            d = m.get("r2_mean", 0) - baseline_r2
            sign = "+" if d >= 0 else ""
            delta = (
                f"{sign}{d:.4f} ({sign}{d / abs(baseline_r2) * 100:.1f}%)"
                if baseline_r2 != 0
                else f"{sign}{d:.4f}"
            )

        lines.append(f"| {name} | {feat_cells} | {r2_mean} | {delta} |")

    # Check if GP-RBF probe data is available
    has_rbf_data = any("r2_mean_rbf" in m for m in metrics.values())

    if has_rbf_data:
        rbf_features = _discover_available_features(metrics, suffix="_rbf")
        if rbf_features:
            rbf_header_cells = " | ".join(
                f"R²_{_FEATURE_SHORT[f].lower()} (GP-RBF)" for f in rbf_features
            )
            rbf_sep = " | ".join(
                "-" * len(f"R²_{_FEATURE_SHORT[f].lower()} (GP-RBF)") for f in rbf_features
            )
            header_row = f"| Condition | {rbf_header_cells} | R²_mean (GP-RBF) |"
            sep_row = f"|-----------|{rbf_sep}|------------------|"

            lines.extend(
                [
                    "",
                    "### GP-RBF Probe R² (Nonlinear)",
                    "",
                    "GP-RBF probes capture nonlinear relationships in the feature space.",
                    "",
                    header_row,
                    sep_row,
                ]
            )

            for cond in config["conditions"]:
                name = cond["name"]
                if name not in metrics:
                    continue

                m = metrics[name]
                rbf_cells = " | ".join(f"{m.get(f'r2_{feat}_rbf', 0):.4f}" for feat in rbf_features)
                r2_mean_rbf = f"{m.get('r2_mean_rbf', 0):.4f}"

                lines.append(f"| {name} | {rbf_cells} | {r2_mean_rbf} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Secondary Results: Segmentation Dice",
            "",
            "| Condition | Test Dice (Mean) | Test Dice (TC) | Test Dice (WT) | Test Dice (ET) |",
            "|-----------|------------------|----------------|----------------|----------------|",
        ]
    )

    for cond in config["conditions"]:
        name = cond["name"]
        if name not in metrics:
            continue

        m = metrics[name]
        # Use test dice from metrics.json
        dice_mean = f"{m.get('test_dice_mean', 0):.4f}" if m.get("test_dice_mean") else "N/A"
        dice_tc = f"{m.get('test_dice_TC', 0):.4f}" if m.get("test_dice_TC") else "N/A"
        dice_wt = f"{m.get('test_dice_WT', 0):.4f}" if m.get("test_dice_WT") else "N/A"
        dice_et = f"{m.get('test_dice_ET', 0):.4f}" if m.get("test_dice_ET") else "N/A"

        lines.append(f"| {name} | {dice_mean} | {dice_tc} | {dice_wt} | {dice_et} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Statistical Analysis",
            "",
        ]
    )

    if stats and stats.recommendation:
        lines.append("### Statistical Test Results")
        lines.append("")
        lines.append("```")
        lines.append(stats.recommendation)
        lines.append("```")
    else:
        lines.append("*Run statistical_analysis.py for detailed statistical tests.*")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Feature Quality Analysis",
            "",
            "### Per-Dimension R² Breakdown",
            "",
        ]
    )

    for cond in config["conditions"]:
        name = cond["name"]
        if name not in metrics:
            continue

        m = metrics[name]
        lines.append(f"**{name}:**")

        for feat in _ALL_SEMANTIC_FEATURES:
            per_dim = m.get(f"r2_{feat}_per_dim", [])
            if per_dim:
                formatted = ", ".join([f"{x:.4f}" for x in per_dim])
                lines.append(f"- {feat}: [{formatted}]")

        lines.append("")

    lines.extend(
        [
            "### Feature Variance",
            "",
            "| Condition | Variance (mean) | Variance (min) |",
            "|-----------|-----------------|----------------|",
        ]
    )

    for cond in config["conditions"]:
        name = cond["name"]
        if name not in metrics:
            continue

        m = metrics[name]
        var_mean = f"{m.get('variance_mean', 0):.4f}"
        var_min = f"{m.get('variance_min', 0):.6f}"
        lines.append(f"| {name} | {var_mean} | {var_min} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Recommendation",
            "",
        ]
    )

    # Generate recommendation
    best_name = "baseline"
    best_r2_mean = baseline_r2

    for name, m in metrics.items():
        if m.get("r2_mean", 0) > best_r2_mean:
            best_name = name
            best_r2_mean = m.get("r2_mean", 0)

    improvement = best_r2_mean - baseline_r2
    improvement_pct = improvement / baseline_r2 * 100 if baseline_r2 > 0 else 0

    if best_name == "baseline" or improvement < 0.03:
        lines.extend(
            [
                "### **RECOMMENDATION: Use Baseline (No LoRA)**",
                "",
                "**Rationale:**",
                f"- Baseline R²_mean: {baseline_r2:.4f}",
                f"- Best LoRA R²_mean: {best_r2_mean:.4f} ({best_name})",
                f"- Improvement: {improvement:.4f} ({improvement_pct:.1f}%)",
                "",
                "The improvement from LoRA adaptation is marginal (<3%) and does not justify:",
                "1. Additional training complexity",
                "2. Increased parameter count",
                "3. Risk of overfitting to meningioma-specific features",
                "",
                "The baseline encoder already provides strong features for meningioma semantic ",
                "prediction, demonstrating good cross-tumor generalization.",
            ]
        )
    elif improvement < 0.05:
        lines.extend(
            [
                f"### **RECOMMENDATION: Consider {best_name} (Marginal Benefit)**",
                "",
                "**Rationale:**",
                f"- Baseline R²_mean: {baseline_r2:.4f}",
                f"- {best_name} R²_mean: {best_r2_mean:.4f}",
                f"- Improvement: {improvement:.4f} ({improvement_pct:.1f}%)",
                "",
                "The improvement is modest (3-5%). Consider:",
                "1. If computational resources allow, use LoRA",
                "2. If simplicity is prioritized, baseline is acceptable",
                "3. Statistical significance should guide final decision",
            ]
        )
    else:
        lines.extend(
            [
                f"### **RECOMMENDATION: Use {best_name}**",
                "",
                "**Rationale:**",
                f"- Baseline R²_mean: {baseline_r2:.4f}",
                f"- {best_name} R²_mean: {best_r2_mean:.4f}",
                f"- Improvement: {improvement:.4f} ({improvement_pct:.1f}%)",
                "",
                "LoRA adaptation provides meaningful improvement (>5%) in semantic feature ",
                "prediction, justifying the additional training complexity for downstream ",
                "growth modeling tasks.",
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "*Report generated by experiments/lora_ablation/analyze_results.py*",
        ]
    )

    return "\n".join(lines)


def analyze_results(
    config_path: str,
    skip_figures: bool = False,
) -> None:
    """Run complete analysis pipeline."""
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])

    logger.info("=" * 60)
    logger.info("LoRA Ablation Analysis Pipeline")
    logger.info("=" * 60)

    # Step 0: Pre-flight validation
    logger.info("\n[0/5] Validating experiment outputs...")
    validate_experiment_outputs(config)

    # Step 1: Load metrics
    logger.info("\n[1/5] Loading experiment metrics...")
    metrics = load_all_metrics(config)

    if not metrics:
        logger.error("No metrics found. Run the experiment first.")
        return

    # Step 2: Statistical analysis
    logger.info("\n[2/5] Running statistical analysis...")
    try:
        stats = run_statistical_analysis(config_path)
        save_statistical_results(stats, output_dir)
    except Exception as e:
        logger.warning(f"Statistical analysis failed: {e}")
        stats = None

    # Step 3: Generate visualizations
    if not skip_figures:
        logger.info("\n[3/5] Generating publication figures...")
        try:
            generate_all_figures(config)
        except Exception as e:
            logger.warning(f"Figure generation failed: {e}")
    else:
        logger.info("\n[3/5] Skipping figure generation (--skip-figures)")

    # Step 4: Generate reports
    logger.info("\n[4/5] Generating reports...")

    # Markdown report
    md_report = generate_markdown_report(config, metrics, stats)
    md_path = output_dir / "analysis_report.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    logger.info(f"Saved Markdown report to {md_path}")

    # LaTeX table
    latex_table = generate_latex_table(metrics, config, stats)
    latex_path = output_dir / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    logger.info(f"Saved LaTeX table to {latex_path}")

    # Summary CSV — include only features that are present in the data
    csv_lin_features = _discover_available_features(metrics)
    csv_rbf_features = _discover_available_features(metrics, suffix="_rbf")

    rows = []
    for cond in config["conditions"]:
        name = cond["name"]
        if name in metrics:
            m = metrics[name]
            row: dict = {"condition": name}
            # Linear probe R² (dynamic)
            for feat in csv_lin_features:
                row[f"r2_{feat}"] = m.get(f"r2_{feat}")
            row["r2_mean"] = m.get("r2_mean")
            # GP-RBF probe R² (dynamic, only if available)
            for feat in csv_rbf_features:
                row[f"r2_{feat}_rbf"] = m.get(f"r2_{feat}_rbf")
            if csv_rbf_features:
                row["r2_mean_rbf"] = m.get("r2_mean_rbf")
            # Test Dice
            row["test_dice_mean"] = m.get("test_dice_mean")
            row["test_dice_TC"] = m.get("test_dice_TC")
            row["test_dice_WT"] = m.get("test_dice_WT")
            row["test_dice_ET"] = m.get("test_dice_ET")
            row["val_dice"] = m.get("val_dice")
            row["variance_mean"] = m.get("variance_mean")
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison table to {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {md_path.name} (Markdown report)")
    print(f"  - {latex_path.name} (LaTeX table)")
    print(f"  - {csv_path.name} (CSV summary)")
    print("  - figures/ (Publication figures)")
    print(f"  - statistical_*.{'{json,csv,txt}'} (Statistical analysis)")

    if stats:
        print("\n" + stats.recommendation)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive analysis for LoRA ablation study")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation (faster for quick analysis)",
    )

    args = parser.parse_args()
    analyze_results(args.config, args.skip_figures)


if __name__ == "__main__":
    main()
