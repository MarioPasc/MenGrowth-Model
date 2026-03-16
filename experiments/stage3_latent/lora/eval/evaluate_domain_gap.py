#!/usr/bin/env python
# experiments/lora/eval/evaluate_domain_gap.py
"""Domain gap evaluation between MEN and GLI feature distributions (B4).

Computes:
- MMD² with permutation test
- CKA (frozen vs adapted, and between domains)
- PAD (Proxy A-distance)
- Effective rank per domain and combined

Usage:
    python -m experiments.lora.eval.evaluate_domain_gap \
        --config experiments/dual_domain_lora/config/dual_domain_v1.yaml \
        --condition dual_r8
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from growth.evaluation.latent_quality import (
    compute_cka,
    compute_domain_classifier_accuracy,
    compute_effective_rank,
    compute_proxy_a_distance,
    compute_variance_per_dim,
    mmd_permutation_test,
)
from growth.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_features(
    features_dir: Path,
    domain: str,
    feature_level: str = "encoder10",
) -> np.ndarray | None:
    """Load test features for a domain.

    Args:
        features_dir: Features directory.
        domain: "men" or "gli".
        feature_level: Feature level.

    Returns:
        Feature array or None.
    """
    feat_path = features_dir / f"features_{domain}_test_{feature_level}.pt"
    if not feat_path.exists():
        feat_path = features_dir / f"features_{domain}_test.pt"

    if not feat_path.exists():
        logger.warning(f"Features not found: {feat_path}")
        return None

    return torch.load(feat_path, weights_only=True).numpy()


def evaluate_domain_gap(
    condition_name: str,
    config: dict,
    baseline_condition: str = "baseline_frozen",
) -> dict[str, float]:
    """Evaluate domain gap metrics for a condition.

    Computes:
    - MMD² between MEN and GLI features (with p-value)
    - Domain classifier accuracy / PAD
    - CKA between frozen and adapted features
    - Effective rank per domain
    - Per-domain variance statistics

    Args:
        condition_name: Condition to evaluate.
        config: Full experiment configuration.
        baseline_condition: Name of baseline condition for CKA comparison.

    Returns:
        Dict with all domain gap metrics.
    """
    logger.info(f"Domain gap evaluation for: {condition_name}")

    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name
    features_dir = condition_dir / "features"
    gap_dir = condition_dir / "domain_gap"
    gap_dir.mkdir(parents=True, exist_ok=True)

    feature_level = config.get("feature_extraction", {}).get("level", "encoder10")

    # Load features
    men_features = load_test_features(features_dir, "men", feature_level)
    gli_features = load_test_features(features_dir, "gli", feature_level)

    if men_features is None or gli_features is None:
        logger.warning("Skipping domain gap — missing features")
        return {}

    results: dict[str, float] = {}

    # MMD² with permutation test
    logger.info("Computing MMD²...")
    mmd_val, mmd_pval = mmd_permutation_test(men_features, gli_features, n_perm=200)
    results["mmd_squared"] = mmd_val
    results["mmd_pvalue"] = mmd_pval
    logger.info(f"  MMD² = {mmd_val:.6f} (p = {mmd_pval:.4f})")

    # Domain classifier accuracy / PAD
    logger.info("Computing PAD...")
    domain_acc = compute_domain_classifier_accuracy(men_features, gli_features)
    pad = compute_proxy_a_distance(men_features, gli_features)
    results["domain_classifier_accuracy"] = domain_acc
    results["proxy_a_distance"] = pad
    logger.info(f"  Domain classifier acc = {domain_acc:.4f}, PAD = {pad:.4f}")

    # Effective rank per domain
    results["men_effective_rank"] = compute_effective_rank(men_features)
    results["gli_effective_rank"] = compute_effective_rank(gli_features)

    combined = np.vstack([men_features, gli_features])
    results["combined_effective_rank"] = compute_effective_rank(combined)
    logger.info(
        f"  Effective rank: MEN={results['men_effective_rank']:.1f}, "
        f"GLI={results['gli_effective_rank']:.1f}, "
        f"combined={results['combined_effective_rank']:.1f}"
    )

    # Per-domain variance
    men_var = compute_variance_per_dim(men_features)
    gli_var = compute_variance_per_dim(gli_features)
    results["men_n_dead_dims"] = int((men_var < 1e-6).sum())
    results["gli_n_dead_dims"] = int((gli_var < 1e-6).sum())
    results["men_variance_mean"] = float(np.mean(men_var))
    results["gli_variance_mean"] = float(np.mean(gli_var))

    # CKA: frozen vs adapted (if baseline available)
    if condition_name != baseline_condition:
        baseline_dir = output_dir / "conditions" / baseline_condition / "features"
        men_frozen = load_test_features(baseline_dir, "men", feature_level)
        gli_frozen = load_test_features(baseline_dir, "gli", feature_level)

        if men_frozen is not None and men_features.shape[0] == men_frozen.shape[0]:
            cka_men = compute_cka(men_frozen, men_features)
            results["cka_men_frozen_vs_adapted"] = cka_men
            logger.info(f"  CKA (MEN frozen vs adapted) = {cka_men:.4f}")

        if gli_frozen is not None and gli_features.shape[0] == gli_frozen.shape[0]:
            cka_gli = compute_cka(gli_frozen, gli_features)
            results["cka_gli_frozen_vs_adapted"] = cka_gli
            logger.info(f"  CKA (GLI frozen vs adapted) = {cka_gli:.4f}")

    # CKA between domains (same condition)
    n_min = min(len(men_features), len(gli_features))
    cka_between = compute_cka(men_features[:n_min], gli_features[:n_min])
    results["cka_men_vs_gli"] = cka_between
    logger.info(f"  CKA (MEN vs GLI) = {cka_between:.4f}")

    # Save results
    with open(gap_dir / "domain_gap_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved domain gap metrics to {gap_dir}")

    return results


def main(
    config_path: str,
    condition: str | None = None,
    device: str = "cuda",
) -> None:
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])

    if condition is not None:
        evaluate_domain_gap(condition, config)
    else:
        for cond in config["conditions"]:
            try:
                evaluate_domain_gap(cond["name"], config)
            except Exception as e:
                logger.warning(f"Failed domain gap for {cond['name']}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain gap evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
