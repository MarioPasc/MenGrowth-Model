#!/usr/bin/env python
# experiments/lora_ablation/statistical_analysis.py
"""Comprehensive statistical analysis for LoRA ablation study.

This module provides rigorous statistical analysis to support thesis conclusions:
1. Bootstrap confidence intervals for R² and Dice
2. Paired statistical tests (Wilcoxon signed-rank, paired t-test)
3. Effect size computation (Cohen's d)
4. Multiple comparison correction (Bonferroni, Holm-Bonferroni)
5. Publication-quality summary tables

Usage:
    python -m experiments.lora_ablation.statistical_analysis \
        --config experiments/lora_ablation/config/ablation.yaml
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval results."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_bootstrap: int = 1000

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"


@dataclass
class StatisticalTest:
    """Results from a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    effect_interpretation: str
    significant_005: bool
    significant_001: bool
    n1: int
    n2: int

    def __str__(self) -> str:
        stars = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "n.s."
        return f"{self.test_name}: p={self.p_value:.4f} {stars}, d={self.effect_size:.3f} ({self.effect_interpretation})"


@dataclass
class ConditionComparison:
    """Comparison between baseline and a LoRA condition."""
    condition: str
    metric: str
    baseline_ci: BootstrapCI
    condition_ci: BootstrapCI
    delta_mean: float
    delta_ci: BootstrapCI
    test_result: StatisticalTest
    p_corrected: float = None  # After multiple comparison correction


@dataclass
class AblationStatistics:
    """Complete statistical analysis results."""
    comparisons: Dict[str, Dict[str, ConditionComparison]] = field(default_factory=dict)
    summary_table: pd.DataFrame = None
    recommendation: str = ""


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = "mean",
    random_state: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval.

    Args:
        data: 1D array of values.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (default 0.95 for 95% CI).
        statistic: "mean" or "median".
        random_state: Random seed.

    Returns:
        BootstrapCI with mean, CI bounds, and std.
    """
    rng = np.random.RandomState(random_state)
    n = len(data)

    # Bootstrap sampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = resample(data, n_samples=n, random_state=rng.randint(0, 2**31))
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        else:
            bootstrap_stats.append(np.median(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute percentiles for CI
    alpha = 1 - ci
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    return BootstrapCI(
        mean=np.mean(data) if statistic == "mean" else np.median(data),
        ci_lower=np.percentile(bootstrap_stats, lower_percentile),
        ci_upper=np.percentile(bootstrap_stats, upper_percentile),
        std=np.std(bootstrap_stats),
        n_bootstrap=n_bootstrap,
    )


def bootstrap_delta_ci(
    data1: np.ndarray,
    data2: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> BootstrapCI:
    """Compute bootstrap CI for the difference between two conditions.

    Uses paired bootstrap (same indices for both).

    Args:
        data1: Values from condition 1 (e.g., baseline).
        data2: Values from condition 2 (e.g., LoRA).
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.
        random_state: Random seed.

    Returns:
        BootstrapCI for (data2 - data1).
    """
    rng = np.random.RandomState(random_state)
    n = len(data1)
    assert len(data2) == n, "Arrays must have same length for paired bootstrap"

    deltas = data2 - data1
    bootstrap_deltas = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        bootstrap_deltas.append(np.mean(deltas[indices]))

    bootstrap_deltas = np.array(bootstrap_deltas)

    alpha = 1 - ci
    return BootstrapCI(
        mean=np.mean(deltas),
        ci_lower=np.percentile(bootstrap_deltas, alpha / 2 * 100),
        ci_upper=np.percentile(bootstrap_deltas, (1 - alpha / 2) * 100),
        std=np.std(bootstrap_deltas),
        n_bootstrap=n_bootstrap,
    )


# =============================================================================
# Statistical Tests
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Uses pooled standard deviation.

    Args:
        group1: Values from group 1.
        group2: Values from group 2.

    Returns:
        Cohen's d (positive if group2 > group1).
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group2) - np.mean(group1)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def paired_statistical_test(
    baseline: np.ndarray,
    condition: np.ndarray,
    test_type: str = "auto",
) -> StatisticalTest:
    """Perform paired statistical test.

    Args:
        baseline: Values from baseline condition.
        condition: Values from LoRA condition.
        test_type: "wilcoxon", "ttest", or "auto" (checks normality).

    Returns:
        StatisticalTest result.
    """
    n = len(baseline)
    assert len(condition) == n, "Arrays must have same length"

    differences = condition - baseline

    # Check normality for auto selection
    if test_type == "auto":
        # Shapiro-Wilk test for normality
        if n >= 3:
            _, p_normal = stats.shapiro(differences)
            test_type = "ttest" if p_normal > 0.05 else "wilcoxon"
        else:
            test_type = "wilcoxon"

    # Perform test
    if test_type == "wilcoxon":
        # Wilcoxon signed-rank test (non-parametric)
        try:
            statistic, p_value = stats.wilcoxon(differences, alternative="two-sided")
            test_name = "Wilcoxon signed-rank"
        except ValueError:
            # All zeros - no difference
            statistic, p_value = 0.0, 1.0
            test_name = "Wilcoxon signed-rank"
    else:
        # Paired t-test
        statistic, p_value = stats.ttest_rel(condition, baseline)
        test_name = "Paired t-test"

    # Effect size
    d = cohens_d(baseline, condition)

    return StatisticalTest(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        effect_size=d,
        effect_interpretation=interpret_cohens_d(d),
        significant_005=p_value < 0.05,
        significant_001=p_value < 0.01,
        n1=n,
        n2=n,
    )


# =============================================================================
# Multiple Comparison Correction
# =============================================================================

def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values.
        alpha: Family-wise error rate.

    Returns:
        List of (corrected_p, significant) tuples.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    corrected = []
    for i, p in enumerate(sorted_p):
        corrected_p = p * (n - i)
        corrected.append(min(corrected_p, 1.0))

    # Ensure monotonicity
    for i in range(1, n):
        corrected[i] = max(corrected[i], corrected[i-1])

    # Map back to original order
    result = [None] * n
    for i, orig_idx in enumerate(sorted_indices):
        result[orig_idx] = (corrected[i], corrected[i] < alpha)

    return result


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values.
        alpha: Family-wise error rate.

    Returns:
        List of (corrected_p, significant) tuples.
    """
    n = len(p_values)
    return [(min(p * n, 1.0), p * n < alpha) for p in p_values]


# =============================================================================
# Per-Subject R² Computation
# =============================================================================

def compute_per_subject_r2(
    features: np.ndarray,
    targets: np.ndarray,
    probe_model,
) -> np.ndarray:
    """Compute R² contribution per subject using leave-one-out.

    This gives per-subject "scores" that can be used for paired tests.

    Args:
        features: [N, D] feature array.
        targets: [N, K] target array.
        probe_model: Fitted sklearn model with predict().

    Returns:
        [N] array of per-subject squared errors (negative, so higher is better).
    """
    predictions = probe_model.predict(features)
    # Per-subject MSE (negative so that "higher is better" like R²)
    per_subject_mse = -np.mean((predictions - targets) ** 2, axis=1)
    return per_subject_mse


# =============================================================================
# Main Analysis Functions
# =============================================================================

def load_condition_data(condition_dir: Path) -> Dict:
    """Load all data for a condition.

    Returns dict with features, targets, metrics, predictions.
    """
    import torch

    data = {}

    # Load features
    for split in ["probe", "test"]:
        feat_path = condition_dir / f"features_{split}.pt"
        targ_path = condition_dir / f"targets_{split}.pt"

        if feat_path.exists():
            data[f"features_{split}"] = torch.load(feat_path).numpy()
            targets = torch.load(targ_path)
            data[f"targets_{split}"] = {k: v.numpy() for k, v in targets.items()}

    # Load metrics
    metrics_path = condition_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)

    # Load training summary
    summary_path = condition_dir / "training_summary.yaml"
    if summary_path.exists():
        with open(summary_path) as f:
            data["training_summary"] = yaml.safe_load(f)

    # Load predictions
    pred_path = condition_dir / "predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            data["predictions"] = json.load(f)

    return data


def compute_per_subject_metrics(
    features: np.ndarray,
    targets: Dict[str, np.ndarray],
    predictions: Dict[str, List],
) -> Dict[str, np.ndarray]:
    """Compute per-subject error metrics for statistical tests.

    Returns dict mapping metric names to [N] arrays.
    """
    metrics = {}

    for target_name, target_vals in targets.items():
        if target_name == "all":
            continue

        pred_vals = np.array(predictions[target_name])

        # Per-subject negative MSE (higher is better)
        per_subject_neg_mse = -np.mean((pred_vals - target_vals) ** 2, axis=1)
        metrics[f"{target_name}_neg_mse"] = per_subject_neg_mse

        # Per-subject absolute error
        per_subject_mae = np.mean(np.abs(pred_vals - target_vals), axis=1)
        metrics[f"{target_name}_mae"] = per_subject_mae

    return metrics


def run_statistical_analysis(config_path: str) -> AblationStatistics:
    """Run complete statistical analysis.

    Args:
        config_path: Path to ablation.yaml.

    Returns:
        AblationStatistics with all results.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    conditions = [c["name"] for c in config["conditions"]]

    # Load data for all conditions
    all_data = {}
    for cond in conditions:
        cond_dir = output_dir / "conditions" / cond
        if cond_dir.exists():
            all_data[cond] = load_condition_data(cond_dir)
            logger.info(f"Loaded data for {cond}")

    if "baseline" not in all_data:
        raise ValueError("Baseline condition data not found")

    # Compute per-subject metrics for each condition
    per_subject_metrics = {}
    for cond, data in all_data.items():
        if "features_test" in data and "targets_test" in data and "predictions" in data:
            per_subject_metrics[cond] = compute_per_subject_metrics(
                data["features_test"],
                data["targets_test"],
                data["predictions"],
            )

    # Run comparisons
    results = AblationStatistics()
    results.comparisons = {}

    baseline_data = all_data["baseline"]
    baseline_metrics = per_subject_metrics.get("baseline", {})

    # Metrics to compare
    metric_names = ["volume_neg_mse", "location_neg_mse", "shape_neg_mse"]
    r2_names = ["r2_volume", "r2_location", "r2_shape", "r2_mean"]

    for cond in conditions:
        if cond == "baseline":
            continue

        if cond not in all_data:
            continue

        cond_data = all_data[cond]
        cond_metrics = per_subject_metrics.get(cond, {})
        results.comparisons[cond] = {}

        # Compare per-subject metrics
        for metric_name in metric_names:
            if metric_name not in baseline_metrics or metric_name not in cond_metrics:
                continue

            baseline_vals = baseline_metrics[metric_name]
            cond_vals = cond_metrics[metric_name]

            # Bootstrap CIs
            baseline_ci = bootstrap_ci(baseline_vals)
            condition_ci = bootstrap_ci(cond_vals)
            delta_ci = bootstrap_delta_ci(baseline_vals, cond_vals)

            # Statistical test
            test_result = paired_statistical_test(baseline_vals, cond_vals)

            results.comparisons[cond][metric_name] = ConditionComparison(
                condition=cond,
                metric=metric_name,
                baseline_ci=baseline_ci,
                condition_ci=condition_ci,
                delta_mean=delta_ci.mean,
                delta_ci=delta_ci,
                test_result=test_result,
            )

        # Also store R² comparisons (single values, no per-subject)
        for r2_name in r2_names:
            baseline_r2 = baseline_data.get("metrics", {}).get(r2_name, 0)
            cond_r2 = cond_data.get("metrics", {}).get(r2_name, 0)
            delta_r2 = cond_r2 - baseline_r2

            results.comparisons[cond][r2_name] = {
                "baseline": baseline_r2,
                "condition": cond_r2,
                "delta": delta_r2,
                "delta_pct": delta_r2 / baseline_r2 * 100 if baseline_r2 != 0 else 0,
            }

    # Apply multiple comparison correction
    all_p_values = []
    comparison_keys = []
    for cond, comps in results.comparisons.items():
        for metric, comp in comps.items():
            if isinstance(comp, ConditionComparison):
                all_p_values.append(comp.test_result.p_value)
                comparison_keys.append((cond, metric))

    if all_p_values:
        corrected = holm_bonferroni_correction(all_p_values)
        for (cond, metric), (p_corr, _) in zip(comparison_keys, corrected):
            results.comparisons[cond][metric].p_corrected = p_corr

    # Create summary table
    results.summary_table = create_summary_table(results, all_data)

    # Generate recommendation
    results.recommendation = generate_statistical_recommendation(results, all_data)

    return results


def create_summary_table(
    results: AblationStatistics,
    all_data: Dict,
) -> pd.DataFrame:
    """Create summary table for thesis.

    Returns DataFrame suitable for LaTeX export.
    """
    rows = []

    conditions = ["baseline"] + list(results.comparisons.keys())

    for cond in conditions:
        row = {"Condition": cond}

        if cond in all_data:
            metrics = all_data[cond].get("metrics", {})
            row["R²_vol"] = metrics.get("r2_volume", None)
            row["R²_loc"] = metrics.get("r2_location", None)
            row["R²_shape"] = metrics.get("r2_shape", None)
            row["R²_mean"] = metrics.get("r2_mean", None)

            summary = all_data[cond].get("training_summary", {})
            row["Val_Dice"] = summary.get("best_val_dice", None)

        if cond != "baseline" and cond in results.comparisons:
            # Add delta and significance
            comps = results.comparisons[cond]
            if "r2_mean" in comps and isinstance(comps["r2_mean"], dict):
                row["ΔR²_mean"] = comps["r2_mean"]["delta"]
                row["ΔR²_mean_%"] = comps["r2_mean"]["delta_pct"]

            # Get statistical significance
            if "volume_neg_mse" in comps:
                comp = comps["volume_neg_mse"]
                row["p_value"] = comp.test_result.p_value
                row["p_corrected"] = comp.p_corrected
                row["Cohen_d"] = comp.test_result.effect_size
                row["Effect"] = comp.test_result.effect_interpretation

        rows.append(row)

    return pd.DataFrame(rows)


def generate_statistical_recommendation(
    results: AblationStatistics,
    all_data: Dict,
) -> str:
    """Generate evidence-based recommendation.

    Returns detailed recommendation text for thesis.
    """
    lines = [
        "=" * 70,
        "STATISTICAL ANALYSIS SUMMARY",
        "=" * 70,
        "",
    ]

    baseline_r2 = all_data.get("baseline", {}).get("metrics", {}).get("r2_mean", 0)
    lines.append(f"Baseline R²_mean: {baseline_r2:.4f}")
    lines.append("")

    # Find best condition
    best_cond = None
    best_r2 = baseline_r2
    best_delta = 0
    significant_improvements = []

    for cond, comps in results.comparisons.items():
        if "r2_mean" in comps and isinstance(comps["r2_mean"], dict):
            cond_r2 = comps["r2_mean"]["condition"]
            delta = comps["r2_mean"]["delta"]

            if cond_r2 > best_r2:
                best_cond = cond
                best_r2 = cond_r2
                best_delta = delta

            # Check statistical significance
            if "volume_neg_mse" in comps:
                comp = comps["volume_neg_mse"]
                if comp.p_corrected is not None and comp.p_corrected < 0.05:
                    significant_improvements.append(cond)

    lines.append("Statistical Test Results:")
    lines.append("-" * 40)

    for cond, comps in results.comparisons.items():
        lines.append(f"\n{cond} vs. Baseline:")
        for metric, comp in comps.items():
            if isinstance(comp, ConditionComparison):
                stars = "***" if comp.test_result.p_value < 0.001 else \
                        "**" if comp.test_result.p_value < 0.01 else \
                        "*" if comp.test_result.p_value < 0.05 else "n.s."
                lines.append(
                    f"  {metric}: Δ={comp.delta_mean:.4f}, "
                    f"p={comp.test_result.p_value:.4f} {stars}, "
                    f"d={comp.test_result.effect_size:.3f} ({comp.test_result.effect_interpretation})"
                )
                if comp.p_corrected:
                    lines.append(f"    (corrected p={comp.p_corrected:.4f})")

    lines.append("")
    lines.append("=" * 70)
    lines.append("RECOMMENDATION")
    lines.append("=" * 70)
    lines.append("")

    if not significant_improvements:
        lines.append("DECISION: Use Baseline (No LoRA)")
        lines.append("")
        lines.append("Rationale:")
        lines.append("• No LoRA condition showed statistically significant improvement")
        lines.append("  after multiple comparison correction (Holm-Bonferroni, α=0.05)")
        lines.append("• The baseline encoder features are sufficient for semantic prediction")
        lines.append("• LoRA adaptation adds complexity without proven benefit")
    elif best_cond and best_delta > 0.05:
        lines.append(f"DECISION: Use {best_cond}")
        lines.append("")
        lines.append("Rationale:")
        lines.append(f"• {best_cond} shows significant improvement (p < 0.05 after correction)")
        lines.append(f"• Mean R² improvement: {best_delta:.4f} ({best_delta/baseline_r2*100:.1f}%)")
        lines.append(f"• Effect size indicates {'meaningful' if abs(results.comparisons[best_cond].get('volume_neg_mse', ConditionComparison('','',None,None,0,None,StatisticalTest('',0,1,0,'',False,False,0,0))).test_result.effect_size) > 0.5 else 'modest'} practical significance")
    else:
        lines.append(f"DECISION: Consider {best_cond if best_cond else 'Baseline'} (Marginal)")
        lines.append("")
        lines.append("Rationale:")
        lines.append("• Statistical significance achieved, but effect size is small")
        lines.append("• Practical benefit may not justify increased complexity")
        lines.append("• Consider computational costs vs. performance gain")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_results(
    results: AblationStatistics,
    output_dir: Path,
) -> None:
    """Save statistical analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary table
    if results.summary_table is not None:
        csv_path = output_dir / "statistical_summary.csv"
        results.summary_table.to_csv(csv_path, index=False)
        logger.info(f"Saved summary table to {csv_path}")

        # Also save LaTeX version
        latex_path = output_dir / "statistical_summary.tex"
        results.summary_table.to_latex(
            latex_path,
            index=False,
            float_format="%.4f",
            caption="Statistical comparison of encoder adaptation strategies.",
            label="tab:lora_ablation",
        )
        logger.info(f"Saved LaTeX table to {latex_path}")

    # Save recommendation
    rec_path = output_dir / "statistical_recommendation.txt"
    with open(rec_path, "w") as f:
        f.write(results.recommendation)
    logger.info(f"Saved recommendation to {rec_path}")

    # Save detailed comparisons as JSON
    comparisons_path = output_dir / "statistical_comparisons.json"
    serializable = {}
    for cond, comps in results.comparisons.items():
        serializable[cond] = {}
        for metric, comp in comps.items():
            if isinstance(comp, ConditionComparison):
                serializable[cond][metric] = {
                    "baseline_mean": comp.baseline_ci.mean,
                    "baseline_ci": [comp.baseline_ci.ci_lower, comp.baseline_ci.ci_upper],
                    "condition_mean": comp.condition_ci.mean,
                    "condition_ci": [comp.condition_ci.ci_lower, comp.condition_ci.ci_upper],
                    "delta_mean": comp.delta_mean,
                    "delta_ci": [comp.delta_ci.ci_lower, comp.delta_ci.ci_upper],
                    "p_value": comp.test_result.p_value,
                    "p_corrected": comp.p_corrected,
                    "effect_size": comp.test_result.effect_size,
                    "effect_interpretation": comp.test_result.effect_interpretation,
                }
            else:
                serializable[cond][metric] = comp

    with open(comparisons_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved comparisons to {comparisons_path}")


def main(config_path: str) -> None:
    """Run statistical analysis and save results."""
    # Load config for output dir
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])

    # Run analysis
    logger.info("Running statistical analysis...")
    results = run_statistical_analysis(config_path)

    # Print and save
    print(results.recommendation)

    if results.summary_table is not None:
        print("\nSummary Table:")
        print(results.summary_table.to_string())

    save_results(results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run statistical analysis for LoRA ablation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )

    args = parser.parse_args()
    main(args.config)
