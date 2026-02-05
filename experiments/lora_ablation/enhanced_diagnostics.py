#!/usr/bin/env python
# experiments/lora_ablation/enhanced_diagnostics.py
"""Enhanced diagnostics for LoRA ablation study.

Provides comprehensive analysis including:
1. Gradient dynamics analysis
2. Feature quality metrics (variance, collapse detection)
3. Representation analysis (correlation structure, dimensionality)
4. Training stability analysis
5. Cross-condition comparisons
6. Potential issues detection

Usage:
    python -m experiments.lora_ablation.enhanced_diagnostics \
        --config experiments/lora_ablation/config/ablation.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Gradient Analysis
# =============================================================================

def analyze_gradient_dynamics(
    training_log: pd.DataFrame,
    condition_name: str,
) -> Dict[str, any]:
    """Analyze gradient dynamics from training log.

    Args:
        training_log: DataFrame with training metrics.
        condition_name: Name of condition.

    Returns:
        Dict with gradient analysis results.
    """
    results = {
        'condition': condition_name,
        'has_gradient_data': False,
    }

    if 'encoder_grad_norm' not in training_log.columns:
        return results

    results['has_gradient_data'] = True

    enc_grad = training_log['encoder_grad_norm'].values
    dec_grad = training_log['decoder_grad_norm'].values
    sem_grad = training_log.get('semantic_grad_norm', pd.Series([0] * len(enc_grad))).values

    # Basic statistics
    results['encoder_grad_mean'] = float(enc_grad.mean())
    results['encoder_grad_std'] = float(enc_grad.std())
    results['decoder_grad_mean'] = float(dec_grad.mean())
    results['decoder_grad_std'] = float(dec_grad.std())
    results['semantic_grad_mean'] = float(sem_grad.mean())
    results['semantic_grad_std'] = float(sem_grad.std())

    # Gradient ratio (encoder/decoder)
    ratio = enc_grad / (dec_grad + 1e-8)
    results['grad_ratio_mean'] = float(ratio.mean())
    results['grad_ratio_std'] = float(ratio.std())
    results['grad_ratio_min'] = float(ratio.min())
    results['grad_ratio_max'] = float(ratio.max())

    # Gradient stability (coefficient of variation)
    results['encoder_cv'] = float(enc_grad.std() / (enc_grad.mean() + 1e-8))
    results['decoder_cv'] = float(dec_grad.std() / (dec_grad.mean() + 1e-8))

    # Gradient trend (linear regression slope)
    epochs = np.arange(len(enc_grad))
    if len(epochs) > 5:
        enc_slope, _, _, _, _ = stats.linregress(epochs, enc_grad)
        dec_slope, _, _, _, _ = stats.linregress(epochs, dec_grad)
        results['encoder_grad_trend'] = float(enc_slope)
        results['decoder_grad_trend'] = float(dec_slope)

    # Detect potential issues
    issues = []

    # Issue: Encoder gradient vanishing
    if enc_grad.mean() < 0.01:
        issues.append("ENCODER_GRAD_VANISHING: Encoder gradients very small (<0.01)")

    # Issue: Gradient imbalance
    if ratio.mean() < 0.2:
        issues.append(f"GRADIENT_IMBALANCE: Encoder receives only {ratio.mean():.1%} of decoder gradients")
    elif ratio.mean() > 10:
        issues.append(f"GRADIENT_IMBALANCE: Encoder receives {ratio.mean():.1f}x decoder gradients (may be unstable)")

    # Issue: High variance (unstable training)
    if results['encoder_cv'] > 2.0:
        issues.append(f"ENCODER_INSTABILITY: High gradient variance (CV={results['encoder_cv']:.2f})")

    results['issues'] = issues
    results['num_issues'] = len(issues)

    return results


def analyze_loss_dynamics(
    training_log: pd.DataFrame,
    condition_name: str,
) -> Dict[str, any]:
    """Analyze loss dynamics from training log.

    Args:
        training_log: DataFrame with training metrics.
        condition_name: Name of condition.

    Returns:
        Dict with loss analysis results.
    """
    results = {
        'condition': condition_name,
    }

    # Training loss
    if 'train_loss' in training_log.columns:
        train_loss = training_log['train_loss'].values
        results['train_loss_initial'] = float(train_loss[0])
        results['train_loss_final'] = float(train_loss[-1])
        results['train_loss_reduction'] = float((train_loss[0] - train_loss[-1]) / (train_loss[0] + 1e-8))

    # Validation Dice
    if 'val_dice_mean' in training_log.columns:
        val_dice = training_log['val_dice_mean'].values
        results['val_dice_initial'] = float(val_dice[0])
        results['val_dice_final'] = float(val_dice[-1])
        results['val_dice_best'] = float(val_dice.max())
        results['val_dice_best_epoch'] = int(val_dice.argmax()) + 1

        # Convergence analysis
        # Find epoch where Dice reaches 95% of best
        threshold = val_dice.max() * 0.95
        convergence_epochs = np.where(val_dice >= threshold)[0]
        if len(convergence_epochs) > 0:
            results['convergence_epoch'] = int(convergence_epochs[0]) + 1
        else:
            results['convergence_epoch'] = len(val_dice)

    # Auxiliary losses (if present)
    for loss_name in ['train_vol_loss', 'train_loc_loss', 'train_shape_loss']:
        if loss_name in training_log.columns:
            loss_vals = training_log[loss_name].values
            # Skip zeros at the start (before warmup)
            nonzero_start = np.argmax(loss_vals > 0)
            if nonzero_start < len(loss_vals):
                loss_vals = loss_vals[nonzero_start:]
                short_name = loss_name.replace('train_', '').replace('_loss', '')
                results[f'{short_name}_loss_final'] = float(loss_vals[-1]) if len(loss_vals) > 0 else 0
                results[f'{short_name}_loss_reduction'] = float(
                    (loss_vals[0] - loss_vals[-1]) / (loss_vals[0] + 1e-8)
                ) if len(loss_vals) > 1 else 0

    return results


# =============================================================================
# Feature Quality Analysis
# =============================================================================

def analyze_feature_quality(
    features_path: Path,
    condition_name: str,
) -> Dict[str, any]:
    """Analyze feature quality from extracted features.

    Args:
        features_path: Path to features file.
        condition_name: Name of condition.

    Returns:
        Dict with feature quality metrics.
    """
    results = {
        'condition': condition_name,
    }

    if not features_path.exists():
        results['error'] = f"Features not found: {features_path}"
        return results

    # Load features
    features = torch.load(features_path).numpy()
    n_samples, n_dims = features.shape

    results['n_samples'] = n_samples
    results['n_dims'] = n_dims

    # Per-dimension statistics
    variance = features.var(axis=0)
    mean = features.mean(axis=0)

    results['variance_mean'] = float(variance.mean())
    results['variance_std'] = float(variance.std())
    results['variance_min'] = float(variance.min())
    results['variance_max'] = float(variance.max())

    # Collapse detection
    results['n_low_variance_dims'] = int((variance < 0.01).sum())
    results['n_collapsed_dims'] = int((variance < 1e-6).sum())
    results['collapse_fraction'] = float((variance < 0.01).sum() / n_dims)

    # Effective dimensionality (participation ratio)
    # Higher = more dimensions are being used
    normalized_var = variance / (variance.sum() + 1e-8)
    participation_ratio = 1.0 / (normalized_var ** 2).sum()
    results['effective_dimensionality'] = float(participation_ratio)
    results['dim_utilization'] = float(participation_ratio / n_dims)

    # Feature range (for detecting saturation)
    results['feature_mean'] = float(mean.mean())
    results['feature_std'] = float(features.std())
    results['feature_min'] = float(features.min())
    results['feature_max'] = float(features.max())

    # Correlation structure
    if n_samples > 10:
        corr_matrix = np.corrcoef(features.T)
        # Exclude diagonal
        off_diag = corr_matrix[~np.eye(n_dims, dtype=bool)]
        results['mean_correlation'] = float(np.abs(off_diag).mean())
        results['max_correlation'] = float(np.abs(off_diag).max())

    # Detect issues
    issues = []

    if results['collapse_fraction'] > 0.5:
        issues.append(f"FEATURE_COLLAPSE: {results['collapse_fraction']:.1%} of dimensions have low variance")

    if results['dim_utilization'] < 0.1:
        issues.append(f"LOW_UTILIZATION: Only {results['dim_utilization']:.1%} effective dimensionality")

    if results.get('mean_correlation', 0) > 0.5:
        issues.append(f"HIGH_CORRELATION: Mean |correlation| = {results['mean_correlation']:.2f}")

    results['issues'] = issues
    results['num_issues'] = len(issues)

    return results


# =============================================================================
# Probe Quality Analysis
# =============================================================================

def analyze_probe_quality(
    metrics_path: Path,
    condition_name: str,
) -> Dict[str, any]:
    """Analyze probe quality metrics.

    Args:
        metrics_path: Path to metrics.json.
        condition_name: Name of condition.

    Returns:
        Dict with probe quality analysis.
    """
    results = {
        'condition': condition_name,
    }

    if not metrics_path.exists():
        results['error'] = f"Metrics not found: {metrics_path}"
        return results

    with open(metrics_path) as f:
        metrics = json.load(f)

    # R² scores
    results['r2_volume'] = metrics.get('r2_volume', 0)
    results['r2_location'] = metrics.get('r2_location', 0)
    results['r2_shape'] = metrics.get('r2_shape', 0)
    results['r2_mean'] = metrics.get('r2_mean', 0)

    # MLP R² scores
    results['r2_volume_mlp'] = metrics.get('r2_volume_mlp', 0)
    results['r2_location_mlp'] = metrics.get('r2_location_mlp', 0)
    results['r2_shape_mlp'] = metrics.get('r2_shape_mlp', 0)
    results['r2_mean_mlp'] = metrics.get('r2_mean_mlp', 0)

    # Nonlinearity gap (MLP - Linear)
    results['nonlinearity_gap_volume'] = results['r2_volume_mlp'] - results['r2_volume']
    results['nonlinearity_gap_location'] = results['r2_location_mlp'] - results['r2_location']
    results['nonlinearity_gap_shape'] = results['r2_shape_mlp'] - results['r2_shape']

    # Per-dimension R² (if available)
    for feat in ['volume', 'location', 'shape']:
        per_dim = metrics.get(f'r2_{feat}_per_dim', [])
        if per_dim:
            results[f'{feat}_n_negative_r2'] = sum(1 for r in per_dim if r < 0)
            results[f'{feat}_n_high_r2'] = sum(1 for r in per_dim if r > 0.5)

    # Detect issues
    issues = []

    if results['r2_mean'] < 0:
        issues.append(f"NEGATIVE_R2: Mean R² is negative ({results['r2_mean']:.3f}), worse than mean predictor")

    if results['r2_volume'] < 0.3 and results['r2_location'] < 0.3:
        issues.append("POOR_SEMANTIC_ENCODING: Both volume and location R² < 0.3")

    # Check if MLP significantly outperforms linear (suggests nonlinear encoding)
    avg_gap = (results['nonlinearity_gap_volume'] + results['nonlinearity_gap_location']) / 2
    if avg_gap > 0.2:
        issues.append(f"NONLINEAR_ENCODING: MLP outperforms linear by {avg_gap:.2f} avg (features may be nonlinearly encoded)")

    if avg_gap < -0.2:
        issues.append(f"MLP_UNDERFITTING: MLP underperforms linear by {abs(avg_gap):.2f} (may need more capacity)")

    results['issues'] = issues
    results['num_issues'] = len(issues)

    return results


# =============================================================================
# Comprehensive Diagnostics
# =============================================================================

def run_comprehensive_diagnostics(config_path: str) -> Dict[str, pd.DataFrame]:
    """Run comprehensive diagnostics on all conditions.

    Args:
        config_path: Path to ablation configuration.

    Returns:
        Dict of DataFrames with different diagnostic results.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])

    logger.info("=" * 60)
    logger.info("COMPREHENSIVE DIAGNOSTICS")
    logger.info("=" * 60)

    gradient_results = []
    loss_results = []
    feature_results = []
    probe_results = []
    all_issues = []

    for cond in config["conditions"]:
        name = cond["name"]
        cond_dir = output_dir / "conditions" / name

        logger.info(f"\nAnalyzing: {name}")
        logger.info("-" * 40)

        # 1. Gradient analysis
        log_path = cond_dir / "training_log.csv"
        if log_path.exists():
            training_log = pd.read_csv(log_path)
            grad_analysis = analyze_gradient_dynamics(training_log, name)
            loss_analysis = analyze_loss_dynamics(training_log, name)

            gradient_results.append(grad_analysis)
            loss_results.append(loss_analysis)

            if grad_analysis.get('has_gradient_data'):
                logger.info(f"  Gradients: enc={grad_analysis['encoder_grad_mean']:.4f}, "
                           f"dec={grad_analysis['decoder_grad_mean']:.4f}, "
                           f"ratio={grad_analysis['grad_ratio_mean']:.2f}x")

            if grad_analysis.get('issues'):
                for issue in grad_analysis['issues']:
                    logger.warning(f"    ISSUE: {issue}")
                    all_issues.append({'condition': name, 'type': 'gradient', 'issue': issue})

        # 2. Feature analysis
        for level in ['multi_scale', 'encoder10']:
            feat_path = cond_dir / f"features_test_{level}.pt"
            if feat_path.exists():
                feat_analysis = analyze_feature_quality(feat_path, f"{name}_{level}")
                feature_results.append(feat_analysis)

                logger.info(f"  Features ({level}): var={feat_analysis['variance_mean']:.4f}, "
                           f"effective_dim={feat_analysis['effective_dimensionality']:.1f}")

                if feat_analysis.get('issues'):
                    for issue in feat_analysis['issues']:
                        logger.warning(f"    ISSUE: {issue}")
                        all_issues.append({'condition': name, 'type': 'feature', 'issue': issue})
                break  # Only analyze one level

        # 3. Probe analysis
        metrics_path = cond_dir / "metrics.json"
        if metrics_path.exists():
            probe_analysis = analyze_probe_quality(metrics_path, name)
            probe_results.append(probe_analysis)

            logger.info(f"  Probes: R²_mean={probe_analysis['r2_mean']:.4f} "
                       f"(vol={probe_analysis['r2_volume']:.3f}, "
                       f"loc={probe_analysis['r2_location']:.3f}, "
                       f"shape={probe_analysis['r2_shape']:.3f})")

            if probe_analysis.get('issues'):
                for issue in probe_analysis['issues']:
                    logger.warning(f"    ISSUE: {issue}")
                    all_issues.append({'condition': name, 'type': 'probe', 'issue': issue})

    # Create DataFrames
    results = {}

    if gradient_results:
        results['gradients'] = pd.DataFrame(gradient_results)

    if loss_results:
        results['loss'] = pd.DataFrame(loss_results)

    if feature_results:
        results['features'] = pd.DataFrame(feature_results)

    if probe_results:
        results['probes'] = pd.DataFrame(probe_results)

    if all_issues:
        results['issues'] = pd.DataFrame(all_issues)

    # Save results
    for name, df in results.items():
        save_path = output_dir / f"diagnostics_{name}.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Saved {name} diagnostics to {save_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    if all_issues:
        logger.warning(f"\nFound {len(all_issues)} potential issues:")
        issue_df = pd.DataFrame(all_issues)
        issue_counts = issue_df.groupby('type').size()
        for issue_type, count in issue_counts.items():
            logger.warning(f"  - {issue_type}: {count} issues")
    else:
        logger.info("\nNo issues detected!")

    # Print recommendations
    logger.info("\n" + "-" * 40)
    logger.info("RECOMMENDATIONS")
    logger.info("-" * 40)

    if 'probes' in results:
        probe_df = results['probes']
        best_condition = probe_df.loc[probe_df['r2_mean'].idxmax()]
        logger.info(f"\n1. Best R² condition: {best_condition['condition']} "
                   f"(R²_mean={best_condition['r2_mean']:.4f})")

        baseline = probe_df[probe_df['condition'] == 'baseline']
        if not baseline.empty:
            baseline_r2 = baseline['r2_mean'].values[0]
            improvements = probe_df[probe_df['r2_mean'] > baseline_r2 + 0.05]
            if not improvements.empty:
                logger.info(f"\n2. Conditions with >0.05 R² improvement over baseline:")
                for _, row in improvements.iterrows():
                    delta = row['r2_mean'] - baseline_r2
                    logger.info(f"   - {row['condition']}: +{delta:.4f}")

    if 'gradients' in results:
        grad_df = results['gradients']
        balanced = grad_df[(grad_df['grad_ratio_mean'] > 0.5) & (grad_df['grad_ratio_mean'] < 3)]
        if not balanced.empty:
            logger.info(f"\n3. Conditions with balanced gradients: {list(balanced['condition'])}")

    return results


def main(config_path: str) -> None:
    """Run comprehensive diagnostics and save results."""
    results = run_comprehensive_diagnostics(config_path)

    # Generate summary report
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])

    report_path = output_dir / "diagnostics_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE DIAGNOSTICS REPORT\n")
        f.write("=" * 70 + "\n\n")

        for name, df in results.items():
            f.write(f"\n{'='*50}\n")
            f.write(f"{name.upper()} ANALYSIS\n")
            f.write(f"{'='*50}\n\n")
            f.write(df.to_string())
            f.write("\n")

    logger.info(f"\nSaved diagnostic report to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive diagnostics for LoRA ablation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )

    args = parser.parse_args()
    main(args.config)
