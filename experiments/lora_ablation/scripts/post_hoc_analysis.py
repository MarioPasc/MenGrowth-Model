#!/usr/bin/env python
"""Post-hoc analysis for LoRA ablation experiments.

Generates:
1. Statistical tests with proper file paths (predictions_enhanced.json)
2. Visualization plots
3. Anomaly detection in results

Usage:
    python -m experiments.lora_ablation.post_hoc_analysis \
        --results-dir /path/to/results/lora_ablation_semantic_heads
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True

    # Publication settings
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
except ImportError:
    HAS_PLOTTING = False
    logger.warning("Matplotlib not available - skipping visualizations")


# =============================================================================
# Statistical Analysis (Fixed file paths)
# =============================================================================

def load_predictions(condition_dir: Path) -> Optional[Dict]:
    """Load predictions from predictions_enhanced.json (correct file)."""
    # Try enhanced first (correct file), then fallback
    for fname in ["predictions_enhanced.json", "predictions.json"]:
        path = condition_dir / fname
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def load_metrics(condition_dir: Path) -> Optional[Dict]:
    """Load metrics from metrics.json, normalizing key names."""
    # Prefer standard metrics.json (has consistent r2_volume, r2_mean keys)
    for fname in ["metrics.json", "metrics_enhanced.json"]:
        path = condition_dir / fname
        if path.exists():
            with open(path) as f:
                raw = json.load(f)

            # Normalize key names (enhanced uses _linear suffix)
            metrics = {}
            for key, val in raw.items():
                # Convert r2_volume_linear -> r2_volume etc.
                if key.endswith('_linear') and not key.endswith('per_dim_linear'):
                    normalized = key.replace('_linear', '')
                    metrics[normalized] = val
                metrics[key] = val  # Keep original key too

            return metrics
    return None


def compute_per_subject_errors(predictions: Dict) -> Dict[str, np.ndarray]:
    """Compute per-subject squared errors for statistical tests."""
    errors = {}
    for target_name in ['volume', 'location', 'shape']:
        if target_name not in predictions:
            continue

        gt = np.array(predictions[target_name]['ground_truth'])
        pred = np.array(predictions[target_name]['linear'])

        # Per-subject negative MSE (higher is better, like R²)
        per_subject_neg_mse = -np.mean((pred - gt) ** 2, axis=1)
        errors[f"{target_name}_neg_mse"] = per_subject_neg_mse

    return errors


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group2) - np.mean(group1)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def paired_test(baseline: np.ndarray, condition: np.ndarray) -> Dict:
    """Perform paired statistical test with effect size."""
    n = len(baseline)
    differences = condition - baseline

    # Check normality
    if n >= 3:
        _, p_normal = stats.shapiro(differences)
        use_parametric = p_normal > 0.05
    else:
        use_parametric = False

    # Perform test
    if use_parametric:
        statistic, p_value = stats.ttest_rel(condition, baseline)
        test_name = "Paired t-test"
    else:
        try:
            statistic, p_value = stats.wilcoxon(differences, alternative="two-sided")
            test_name = "Wilcoxon signed-rank"
        except ValueError:
            statistic, p_value = 0.0, 1.0
            test_name = "Wilcoxon (no variance)"

    d = cohens_d(baseline, condition)

    return {
        "test_name": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(d),
        "effect_interpretation": interpret_cohens_d(d),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
        "n": int(n),
    }


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni correction."""
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


def run_statistical_analysis(results_dir: Path) -> Dict:
    """Run complete statistical analysis with correct file paths."""
    logger.info("Running statistical analysis...")

    conditions_dir = results_dir / "conditions"
    conditions = [d.name for d in conditions_dir.iterdir() if d.is_dir()]

    # Load all data
    all_predictions = {}
    all_metrics = {}
    all_errors = {}

    for cond in conditions:
        cond_dir = conditions_dir / cond

        preds = load_predictions(cond_dir)
        if preds:
            all_predictions[cond] = preds
            all_errors[cond] = compute_per_subject_errors(preds)

        metrics = load_metrics(cond_dir)
        if metrics:
            all_metrics[cond] = metrics

    if "baseline" not in all_predictions:
        logger.warning("Baseline predictions not found - limited analysis")
        return {"error": "baseline_missing"}

    baseline_errors = all_errors["baseline"]

    # Run pairwise comparisons
    comparisons = {}
    all_p_values = []
    comparison_keys = []

    for cond in conditions:
        if cond == "baseline" or cond not in all_errors:
            continue

        comparisons[cond] = {}
        cond_errors = all_errors[cond]

        for metric_name in ['volume_neg_mse', 'location_neg_mse', 'shape_neg_mse']:
            if metric_name not in baseline_errors or metric_name not in cond_errors:
                continue

            test_result = paired_test(baseline_errors[metric_name], cond_errors[metric_name])
            comparisons[cond][metric_name] = test_result

            all_p_values.append(test_result['p_value'])
            comparison_keys.append((cond, metric_name))

    # Apply multiple comparison correction
    if all_p_values:
        corrected = holm_bonferroni_correction(all_p_values)
        for (cond, metric), (p_corr, sig) in zip(comparison_keys, corrected):
            comparisons[cond][metric]['p_corrected'] = float(p_corr)
            comparisons[cond][metric]['significant_corrected'] = bool(sig)

    # Generate recommendation
    recommendation = generate_recommendation(comparisons, all_metrics)

    return {
        "comparisons": comparisons,
        "metrics": all_metrics,
        "recommendation": recommendation,
    }


def get_r2_mean(metrics_dict: Dict) -> float:
    """Get R² mean from metrics dict, handling different key names."""
    # Try different key names
    for key in ['r2_mean', 'r2_mean_linear']:
        if key in metrics_dict:
            return metrics_dict[key]
    return 0.0


def generate_recommendation(comparisons: Dict, metrics: Dict) -> str:
    """Generate evidence-based recommendation."""
    lines = [
        "=" * 70,
        "STATISTICAL ANALYSIS SUMMARY (CORRECTED)",
        "=" * 70,
        "",
    ]

    baseline_r2 = get_r2_mean(metrics.get("baseline", {}))
    lines.append(f"Baseline R²_mean: {baseline_r2:.4f}")
    lines.append("")

    # Find best condition and significant improvements
    best_cond = "baseline"
    best_r2 = baseline_r2
    significant_improvements = []

    for cond, comps in comparisons.items():
        cond_r2 = get_r2_mean(metrics.get(cond, {}))
        if cond_r2 > best_r2:
            best_cond = cond
            best_r2 = cond_r2

        # Check if any metric is significant after correction
        for metric, test in comps.items():
            if test.get('significant_corrected', False):
                if cond not in significant_improvements:
                    significant_improvements.append(cond)

    lines.append("Statistical Test Results:")
    lines.append("-" * 40)

    for cond, comps in comparisons.items():
        lines.append(f"\n{cond} vs. Baseline:")
        for metric, test in comps.items():
            stars = "***" if test['p_value'] < 0.001 else \
                    "**" if test['p_value'] < 0.01 else \
                    "*" if test['p_value'] < 0.05 else "n.s."
            lines.append(
                f"  {metric}: p={test['p_value']:.4f} {stars}, "
                f"d={test['effect_size']:.3f} ({test['effect_interpretation']})"
            )
            if 'p_corrected' in test:
                corr_sig = "✓" if test['significant_corrected'] else "✗"
                lines.append(f"    (corrected p={test['p_corrected']:.4f} {corr_sig})")

    lines.append("")
    lines.append("=" * 70)
    lines.append("RECOMMENDATION")
    lines.append("=" * 70)
    lines.append("")

    delta = best_r2 - baseline_r2

    if significant_improvements:
        lines.append(f"DECISION: Use {best_cond}")
        lines.append("")
        lines.append("Rationale:")
        lines.append(f"• {best_cond} shows improvement with R²_mean = {best_r2:.4f}")
        if abs(baseline_r2) > 1e-6:
            lines.append(f"• Improvement over baseline: {delta:.4f} ({delta/abs(baseline_r2)*100:.1f}%)")
        else:
            lines.append(f"• Improvement over baseline: {delta:.4f}")
        lines.append(f"• Statistically significant conditions: {significant_improvements}")
    elif delta > 0.05:
        lines.append(f"DECISION: Consider {best_cond} (Marginal)")
        lines.append("")
        lines.append("Rationale:")
        lines.append(f"• {best_cond} shows numerical improvement: {delta:.4f}")
        if abs(baseline_r2) > 1e-6:
            lines.append(f"• Relative improvement: {delta/abs(baseline_r2)*100:.1f}%")
        lines.append("• Statistical significance not achieved after correction")
        lines.append("• Effect sizes suggest practical relevance may exist")
    else:
        lines.append("DECISION: Use Baseline (No LoRA)")
        lines.append("")
        lines.append("Rationale:")
        lines.append("• No LoRA condition showed statistically significant improvement")
        lines.append("  after multiple comparison correction (Holm-Bonferroni, α=0.05)")
        lines.append("• The baseline encoder features are sufficient for semantic prediction")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# Visualization (run locally)
# =============================================================================

CONDITION_COLORS = {
    'baseline': '#808080',
    'lora_r2': '#a6cee3',
    'lora_r4': '#1f78b4',
    'lora_r8': '#33a02c',
    'lora_r16': '#ff7f00',
    'lora_r32': '#e31a1c',
}


def get_metric(metrics_dict: Dict, key: str) -> float:
    """Get a metric value, handling different key naming conventions."""
    # Try exact key first
    if key in metrics_dict:
        return metrics_dict[key]
    # Try with _linear suffix (enhanced format)
    if f"{key}_linear" in metrics_dict:
        return metrics_dict[f"{key}_linear"]
    return 0.0


def plot_r2_comparison(metrics: Dict, output_dir: Path, title_suffix: str = "") -> None:
    """Plot R² comparison bar chart."""
    if not HAS_PLOTTING:
        return

    # Sort conditions
    order = ['baseline', 'lora_r2', 'lora_r4', 'lora_r8', 'lora_r16', 'lora_r32']
    conditions = [c for c in order if c in metrics]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, (feat, title) in zip(axes, [
        ('r2_volume', 'Volume R²'),
        ('r2_location', 'Location R²'),
        ('r2_shape', 'Shape R²'),
        ('r2_mean', 'Mean R²'),
    ]):
        values = [get_metric(metrics[c], feat) for c in conditions]
        colors = [CONDITION_COLORS.get(c, 'gray') for c in conditions]

        bars = ax.bar(range(len(conditions)), values, color=colors)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('R²')
        ax.set_title(title)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c.replace('lora_', 'r=').replace('baseline', 'base')
                          for c in conditions], rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, val),
                       xytext=(0, 3 if val >= 0 else -10),
                       textcoords="offset points",
                       ha='center', va='bottom' if val >= 0 else 'top',
                       fontsize=8)

    plt.suptitle(f'Linear Probe R² by Condition{title_suffix}', y=1.02)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f"r2_comparison.{ext}")
    plt.close()
    logger.info(f"Saved r2_comparison.{{pdf,png}}")


def plot_linear_vs_mlp(metrics: Dict, output_dir: Path, title_suffix: str = "") -> None:
    """Plot linear vs MLP probe comparison."""
    if not HAS_PLOTTING:
        return

    order = ['baseline', 'lora_r2', 'lora_r4', 'lora_r8', 'lora_r16', 'lora_r32']
    conditions = [c for c in order if c in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, feat in zip(axes, ['volume', 'location', 'shape']):
        linear_r2 = [get_metric(metrics[c], f'r2_{feat}') for c in conditions]
        mlp_r2 = [metrics[c].get(f'r2_{feat}_mlp', 0) for c in conditions]

        x = np.arange(len(conditions))
        width = 0.35

        ax.bar(x - width/2, linear_r2, width, label='Linear', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, mlp_r2, width, label='MLP', color='darkorange', alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('R²')
        ax.set_title(f'{feat.capitalize()} R²')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('lora_', 'r=').replace('baseline', 'base')
                          for c in conditions], rotation=45, ha='right')
        ax.legend()

    plt.suptitle(f'Linear vs MLP Probe Performance{title_suffix}', y=1.02)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f"linear_vs_mlp.{ext}")
    plt.close()
    logger.info(f"Saved linear_vs_mlp.{{pdf,png}}")


def plot_rank_vs_r2(metrics: Dict, output_dir: Path, title_suffix: str = "") -> None:
    """Plot LoRA rank vs R² trend."""
    if not HAS_PLOTTING:
        return

    ranks = [0, 2, 4, 8, 16, 32]
    cond_map = {0: 'baseline', 2: 'lora_r2', 4: 'lora_r4',
                8: 'lora_r8', 16: 'lora_r16', 32: 'lora_r32'}

    fig, ax = plt.subplots(figsize=(8, 5))

    for feat, color, marker in [
        ('r2_volume', 'blue', 'o'),
        ('r2_location', 'green', 's'),
        ('r2_shape', 'red', '^'),
        ('r2_mean', 'black', 'D'),
    ]:
        values = []
        valid_ranks = []
        for r in ranks:
            cond = cond_map[r]
            if cond in metrics:
                values.append(get_metric(metrics[cond], feat))
                valid_ranks.append(r)

        ax.plot(valid_ranks, values, marker=marker, label=feat.replace('r2_', ''),
               color=color, linewidth=2, markersize=8)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('LoRA Rank (0 = baseline)')
    ax.set_ylabel('R²')
    ax.set_title(f'Effect of LoRA Rank on Feature Quality{title_suffix}')
    ax.legend()
    ax.set_xticks(ranks)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f"rank_vs_r2.{ext}")
    plt.close()
    logger.info(f"Saved rank_vs_r2.{{pdf,png}}")


def plot_semantic_vs_nosemantic(
    semantic_metrics: Dict,
    nosemantic_metrics: Dict,
    output_dir: Path
) -> None:
    """Compare semantic heads vs no semantic heads experiments."""
    if not HAS_PLOTTING:
        return

    order = ['baseline', 'lora_r2', 'lora_r4', 'lora_r8', 'lora_r16', 'lora_r32']
    conditions = [c for c in order if c in semantic_metrics and c in nosemantic_metrics]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, (feat, title) in zip(axes, [
        ('r2_volume', 'Volume R²'),
        ('r2_location', 'Location R²'),
        ('r2_shape', 'Shape R²'),
        ('r2_mean', 'Mean R²'),
    ]):
        sem_values = [get_metric(semantic_metrics[c], feat) for c in conditions]
        nosem_values = [get_metric(nosemantic_metrics[c], feat) for c in conditions]

        x = np.arange(len(conditions))
        width = 0.35

        ax.bar(x - width/2, sem_values, width, label='Semantic', color='forestgreen', alpha=0.8)
        ax.bar(x + width/2, nosem_values, width, label='Dice-only', color='steelblue', alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('R²')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('lora_', 'r=').replace('baseline', 'base')
                          for c in conditions], rotation=45, ha='right')
        ax.legend()

    plt.suptitle('Semantic Supervision vs Dice-Only Training', y=1.02)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f"semantic_vs_nosemantic.{ext}")
    plt.close()
    logger.info(f"Saved semantic_vs_nosemantic.{{pdf,png}}")


# =============================================================================
# Anomaly Detection
# =============================================================================

def detect_anomalies(results_dir: Path) -> List[str]:
    """Detect potential issues in experiment results."""
    anomalies = []

    conditions_dir = results_dir / "conditions"
    if not conditions_dir.exists():
        anomalies.append(f"ERROR: conditions directory not found at {conditions_dir}")
        return anomalies

    for cond_dir in conditions_dir.iterdir():
        if not cond_dir.is_dir():
            continue

        cond = cond_dir.name

        # Check training log
        log_path = cond_dir / "training_log.csv"
        if log_path.exists():
            df = pd.read_csv(log_path)

            # Check for NaN losses
            if df['train_loss'].isna().any():
                anomalies.append(f"WARNING [{cond}]: NaN values in training loss")

            # Check for loss explosion
            if df['train_loss'].max() > 100:
                anomalies.append(f"WARNING [{cond}]: Very high training loss (max={df['train_loss'].max():.2f})")

            # Check for early stopping
            epochs = len(df)
            if epochs < 50:
                anomalies.append(f"WARNING [{cond}]: Only {epochs} epochs trained (early stopping?)")

            # Check validation dice trend
            if 'val_dice_mean' in df.columns:
                dice_values = df['val_dice_mean'].dropna()
                if len(dice_values) > 10:
                    early_dice = dice_values.iloc[:10].mean()
                    late_dice = dice_values.iloc[-10:].mean()
                    if late_dice < early_dice - 0.05:
                        anomalies.append(f"WARNING [{cond}]: Dice degraded during training "
                                       f"({early_dice:.3f} → {late_dice:.3f})")

        # Check metrics
        metrics = load_metrics(cond_dir)
        if metrics:
            # Check for extremely negative R²
            for feat in ['volume', 'location', 'shape']:
                r2 = metrics.get(f'r2_{feat}', 0)
                if r2 < -1.0:
                    anomalies.append(f"WARNING [{cond}]: Very negative R²_{feat} = {r2:.3f}")

            # Check MLP vs Linear gap
            for feat in ['volume', 'location', 'shape']:
                linear = metrics.get(f'r2_{feat}', 0)
                mlp = metrics.get(f'r2_{feat}_mlp', 0)
                gap = mlp - linear

                # MLP should generally be >= linear (captures nonlinear patterns)
                # Large negative gap suggests MLP overfitting
                if gap < -0.2:
                    anomalies.append(f"WARNING [{cond}]: MLP underperforms linear for {feat} "
                                   f"(gap={gap:.3f}) - possible MLP overfitting")

            # Check variance
            var_mean = metrics.get('variance_mean', 0)
            if var_mean < 0.01:
                anomalies.append(f"WARNING [{cond}]: Very low feature variance ({var_mean:.6f}) "
                               "- possible feature collapse")

        # Check for missing files
        required_files = ['features_probe.pt', 'features_test.pt', 'targets_probe.pt',
                         'targets_test.pt', 'metrics.json']
        for fname in required_files:
            if not (cond_dir / fname).exists():
                # Check for alternative names
                alt_exists = any((cond_dir / f).exists()
                               for f in [fname.replace('.pt', '_multi_scale.pt'),
                                        fname.replace('.json', '_enhanced.json')])
                if not alt_exists:
                    anomalies.append(f"WARNING [{cond}]: Missing {fname}")

    return anomalies


# =============================================================================
# Main Analysis
# =============================================================================

def main(
    results_dir: str,
    compare_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Run complete post-hoc analysis."""
    results_path = Path(results_dir)

    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = results_path / "post_hoc_analysis"

    out_path.mkdir(parents=True, exist_ok=True)
    figures_dir = out_path / "figures"
    figures_dir.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("POST-HOC ANALYSIS FOR LORA ABLATION")
    logger.info("=" * 70)
    logger.info(f"Results directory: {results_path}")
    logger.info(f"Output directory: {out_path}")

    # 1. Detect anomalies
    logger.info("\n[1/4] Checking for anomalies...")
    anomalies = detect_anomalies(results_path)

    if anomalies:
        logger.warning(f"Found {len(anomalies)} potential issues:")
        for a in anomalies:
            logger.warning(f"  {a}")

        with open(out_path / "anomalies.txt", "w") as f:
            f.write("ANOMALY DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            for a in anomalies:
                f.write(f"{a}\n")
    else:
        logger.info("  No anomalies detected")

    # 2. Run statistical analysis
    logger.info("\n[2/4] Running statistical analysis...")
    stats = run_statistical_analysis(results_path)

    if "error" not in stats:
        # Save recommendation
        with open(out_path / "statistical_recommendation_corrected.txt", "w") as f:
            f.write(stats["recommendation"])
        logger.info(f"Saved statistical recommendation")

        # Save comparisons as JSON
        with open(out_path / "statistical_comparisons_corrected.json", "w") as f:
            json.dump(stats["comparisons"], f, indent=2)

        print("\n" + stats["recommendation"])

    # 3. Generate visualizations
    logger.info("\n[3/4] Generating visualizations...")

    if HAS_PLOTTING and stats.get("metrics"):
        metrics = stats["metrics"]
        exp_name = results_path.name.replace("lora_ablation_", "")
        title_suffix = f" ({exp_name})"

        plot_r2_comparison(metrics, figures_dir, title_suffix)
        plot_linear_vs_mlp(metrics, figures_dir, title_suffix)
        plot_rank_vs_r2(metrics, figures_dir, title_suffix)

    # 4. Compare experiments if second directory provided
    if compare_dir:
        logger.info("\n[4/4] Comparing experiments...")
        compare_path = Path(compare_dir)

        stats2 = run_statistical_analysis(compare_path)

        if "metrics" in stats and "metrics" in stats2:
            if HAS_PLOTTING:
                plot_semantic_vs_nosemantic(
                    stats["metrics"],
                    stats2["metrics"],
                    figures_dir
                )

            # Create comparison table
            rows = []
            for cond in stats["metrics"].keys():
                if cond in stats2["metrics"]:
                    m1 = stats["metrics"][cond]
                    m2 = stats2["metrics"][cond]
                    r2_1 = get_r2_mean(m1)
                    r2_2 = get_r2_mean(m2)
                    rows.append({
                        "condition": cond,
                        "r2_mean_semantic": r2_1,
                        "r2_mean_nosemantic": r2_2,
                        "delta": r2_1 - r2_2,
                    })

            df = pd.DataFrame(rows)
            df.to_csv(out_path / "experiment_comparison.csv", index=False)
            logger.info("Saved experiment comparison")
    else:
        logger.info("\n[4/4] Skipping comparison (no second directory provided)")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Outputs saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-hoc analysis for LoRA ablation")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Path to experiment results directory")
    parser.add_argument("--compare-dir", type=str, default=None,
                       help="Optional: second experiment to compare")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: results_dir/post_hoc_analysis)")

    args = parser.parse_args()
    main(args.results_dir, args.compare_dir, args.output_dir)
