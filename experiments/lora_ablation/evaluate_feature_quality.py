#!/usr/bin/env python
# experiments/lora_ablation/evaluate_feature_quality.py
"""Comprehensive feature quality diagnostics for LoRA ablation v3.

Computes per-condition:
- Effective rank (SVD entropy-based)
- PCA explained variance at dims 1, 5, 10, 50, 100
- Inter-dimension correlation (mean |Pearson r|)
- Per-target linear probe R² (vol, loc, shape)
- Feature-target correlation (# dims with |r| > 0.3 per target)
- DCI disentanglement scores

Generates both CSV and LaTeX comparison tables.

Usage:
    python -m experiments.lora_ablation.evaluate_feature_quality \
        --config experiments/lora_ablation/config/ablation_v3.yaml \
        --condition lora_r16_full
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml

from growth.evaluation.latent_quality import (
    compute_dci,
    compute_effective_rank,
    compute_variance_per_dim,
    LinearProbe,
)
from growth.utils.seed import set_seed

from .data_splits import load_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_pca_explained_variance(
    features: np.ndarray,
    dims: tuple[int, ...] = (1, 5, 10, 50, 100),
) -> dict[str, float]:
    """Compute cumulative PCA explained variance at given dims.

    Args:
        features: Feature array [N, D].
        dims: Tuple of dimension counts to report.

    Returns:
        Dict with 'pca_var_at_{d}' for each d in dims.
    """
    features_centered = features - features.mean(axis=0)
    _, s, _ = np.linalg.svd(features_centered, full_matrices=False)

    total_var = (s ** 2).sum()
    cumvar = np.cumsum(s ** 2) / total_var

    results = {}
    for d in dims:
        if d <= len(cumvar):
            results[f"pca_var_at_{d}"] = float(cumvar[d - 1])
        else:
            results[f"pca_var_at_{d}"] = 1.0

    return results


def compute_inter_dim_correlation(features: np.ndarray) -> dict[str, float]:
    """Compute inter-dimension correlation statistics.

    Args:
        features: Feature array [N, D].

    Returns:
        Dict with 'mean_abs_corr', 'median_abs_corr', 'max_abs_corr'.
    """
    corr = np.corrcoef(features.T)
    # Zero out diagonal
    np.fill_diagonal(corr, 0)
    abs_corr = np.abs(corr)

    # Upper triangle only to avoid double-counting
    upper_tri = abs_corr[np.triu_indices_from(abs_corr, k=1)]

    return {
        "mean_abs_corr": float(np.mean(upper_tri)),
        "median_abs_corr": float(np.median(upper_tri)),
        "max_abs_corr": float(np.max(upper_tri)),
    }


def compute_feature_target_correlation(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    threshold: float = 0.3,
) -> dict[str, int]:
    """Count features with |Pearson r| > threshold per target.

    Args:
        features: Feature array [N, D].
        targets: Dict of target arrays.
        threshold: Correlation threshold.

    Returns:
        Dict with 'n_informative_{target}' for each target.
    """
    results = {}

    for target_name, target_arr in targets.items():
        if target_arr.ndim == 1:
            target_arr = target_arr.reshape(-1, 1)

        n_informative = 0
        for d in range(features.shape[1]):
            for t in range(target_arr.shape[1]):
                r = np.corrcoef(features[:, d], target_arr[:, t])[0, 1]
                if abs(r) > threshold:
                    n_informative += 1
                    break  # Count this dim once even if correlated with multiple targets

        results[f"n_informative_{target_name}"] = n_informative

    return results


def evaluate_feature_quality_single(
    condition_name: str,
    config: dict,
    feature_level: str = "encoder10",
) -> dict[str, float]:
    """Evaluate feature quality for a single condition.

    Args:
        condition_name: Condition to evaluate.
        config: Experiment configuration.
        feature_level: Which feature level to evaluate.

    Returns:
        Dict with all quality metrics.
    """
    logger.info(f"Evaluating feature quality for: {condition_name}")

    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    # Load features
    features_path = condition_dir / f"features_probe_{feature_level}.pt"
    if not features_path.exists():
        features_path = condition_dir / "features_probe.pt"

    if not features_path.exists():
        logger.warning(f"Features not found for {condition_name}, skipping")
        return {}

    features = torch.load(features_path, weights_only=True).numpy()
    targets_data = torch.load(condition_dir / "targets_probe.pt", weights_only=True)
    targets = {k: v.numpy() for k, v in targets_data.items() if k != "all"}

    # Also load test set for probe evaluation
    test_features_path = condition_dir / f"features_test_{feature_level}.pt"
    if not test_features_path.exists():
        test_features_path = condition_dir / "features_test.pt"

    test_features = torch.load(test_features_path, weights_only=True).numpy()
    test_targets_data = torch.load(condition_dir / "targets_test.pt", weights_only=True)
    test_targets = {k: v.numpy() for k, v in test_targets_data.items() if k != "all"}

    logger.info(f"  Features: {features.shape}, Test: {test_features.shape}")

    metrics = {"condition": condition_name}

    # 1. Effective rank
    metrics["effective_rank"] = compute_effective_rank(features)
    logger.info(f"  Effective rank: {metrics['effective_rank']:.1f}")

    # 2. PCA explained variance
    pca_metrics = compute_pca_explained_variance(features)
    metrics.update(pca_metrics)
    logger.info(f"  PCA var at 10 dims: {pca_metrics.get('pca_var_at_10', 0):.3f}")

    # 3. Inter-dimension correlation
    corr_metrics = compute_inter_dim_correlation(features)
    metrics.update(corr_metrics)
    logger.info(f"  Mean |r|: {corr_metrics['mean_abs_corr']:.3f}")

    # 4. Per-target linear probe R²
    for target_name in ["volume", "location", "shape"]:
        y_train = targets[target_name]
        y_test = test_targets[target_name]
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

        probe = LinearProbe(
            input_dim=features.shape[1],
            output_dim=y_train.shape[1],
            alpha=1.0,
        )
        probe.fit(features, y_train)
        result = probe.evaluate(test_features, y_test)
        metrics[f"r2_{target_name}"] = result.r2
        metrics[f"r2_{target_name}_per_dim"] = result.r2_per_dim.tolist()
        logger.info(f"  R² {target_name}: {result.r2:.4f}")

    # 5. Feature-target correlation
    ft_corr = compute_feature_target_correlation(features, targets)
    metrics.update(ft_corr)
    for k, v in ft_corr.items():
        logger.info(f"  {k}: {v}")

    # 6. DCI disentanglement
    all_targets = np.concatenate(
        [targets[k] if targets[k].ndim == 2 else targets[k].reshape(-1, 1)
         for k in ["volume", "location", "shape"]],
        axis=1,
    )
    dci = compute_dci(features, all_targets, alpha=0.01)
    metrics["dci_disentanglement"] = dci.disentanglement
    metrics["dci_completeness"] = dci.completeness
    metrics["dci_informativeness"] = dci.informativeness
    logger.info(
        f"  DCI: D={dci.disentanglement:.3f}, "
        f"C={dci.completeness:.3f}, I={dci.informativeness:.3f}"
    )

    # 7. Variance analysis
    variance = compute_variance_per_dim(features)
    metrics["variance_mean"] = float(np.mean(variance))
    metrics["variance_min"] = float(np.min(variance))
    metrics["num_collapsed_dims"] = int((variance < 1e-6).sum())
    metrics["num_low_variance_dims"] = int((variance < 0.01).sum())

    # Save metrics
    metrics_path = condition_dir / "feature_quality.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Saved to {metrics_path}")

    return metrics


def evaluate_all_conditions(
    config: dict,
    feature_level: str = "encoder10",
) -> list[dict[str, float]]:
    """Evaluate feature quality for all conditions.

    Args:
        config: Experiment configuration.
        feature_level: Which feature level to evaluate.

    Returns:
        List of metric dicts, one per condition.
    """
    all_metrics = []

    for cond in config["conditions"]:
        name = cond["name"]
        try:
            metrics = evaluate_feature_quality_single(name, config, feature_level)
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            logger.warning(f"Failed to evaluate {name}: {e}")

    return all_metrics


def generate_comparison_table(
    all_metrics: list[dict],
    output_dir: Path,
) -> None:
    """Generate CSV and LaTeX comparison tables.

    Args:
        all_metrics: List of metric dicts from evaluate_all_conditions.
        output_dir: Directory to save tables.
    """
    if not all_metrics:
        logger.warning("No metrics to generate tables from")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV table
    csv_path = output_dir / "feature_quality_comparison.csv"
    columns = [
        "condition", "effective_rank", "mean_abs_corr",
        "r2_volume", "r2_location", "r2_shape",
        "n_informative_volume", "n_informative_location", "n_informative_shape",
        "dci_disentanglement", "dci_completeness", "dci_informativeness",
        "pca_var_at_10", "pca_var_at_50",
        "num_collapsed_dims",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)

    logger.info(f"Saved comparison CSV to {csv_path}")

    # LaTeX table
    latex_path = output_dir / "feature_quality_table.tex"
    with open(latex_path, "w") as f:
        f.write("\\begin{tabular}{l|ccc|ccc|c}\n")
        f.write("\\hline\n")
        f.write(
            "Condition & Eff. Rank & Mean $|r|$ & "
            "$R^2_{\\text{vol}}$ & $R^2_{\\text{loc}}$ & $R^2_{\\text{shape}}$ & "
            "DCI-D & Collapsed \\\\\n"
        )
        f.write("\\hline\n")

        for m in all_metrics:
            name = m.get("condition", "?")
            f.write(
                f"{name} & "
                f"{m.get('effective_rank', 0):.1f} & "
                f"{m.get('mean_abs_corr', 0):.3f} & "
                f"{m.get('r2_volume', 0):.3f} & "
                f"{m.get('r2_location', 0):.3f} & "
                f"{m.get('r2_shape', 0):.3f} & "
                f"{m.get('dci_disentanglement', 0):.3f} & "
                f"{m.get('num_collapsed_dims', 0)} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    logger.info(f"Saved LaTeX table to {latex_path}")


def main(config_path: str, condition: str | None = None, device: str = "cuda") -> None:
    """Main entry point.

    Args:
        config_path: Path to ablation config.
        condition: Optional single condition (if None, evaluates all).
        device: Device (unused, CPU-only computation).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])
    feature_level = config.get("feature_extraction", {}).get("level", "encoder10")

    if condition:
        evaluate_feature_quality_single(condition, config, feature_level)
    else:
        all_metrics = evaluate_all_conditions(config, feature_level)
        output_dir = Path(config["experiment"]["output_dir"])
        generate_comparison_table(all_metrics, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature quality diagnostics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
