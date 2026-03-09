#!/usr/bin/env python
# experiments/lora/eval/evaluate_probes.py
"""GP probe evaluation (single-domain, per-domain, and cross-domain).

Supports:
- Single-domain GP probes (lora_ablation compat): train/test on MEN
- Per-domain probes (B2): separate MEN and GLI GP probes
- Cross-domain probes (B3): train on GLI, test on MEN and vice versa
- VICReg diagnostics (B5): variance, dead dims, effective rank

Usage:
    python -m experiments.lora.run --config <yaml> probes --condition <name>
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from growth.evaluation.gp_probes import GPSemanticProbes
from growth.evaluation.latent_quality import (
    compute_effective_rank,
    compute_variance_per_dim,
)
from growth.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Feature Loading
# =============================================================================


def load_domain_features(
    features_dir: Path,
    domain: str,
    split: str,
    feature_level: str = "encoder10",
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Load features and targets for a domain/split.

    Args:
        features_dir: Features directory.
        domain: "men" or "gli".
        split: "test" or "probe".
        feature_level: Feature level to load.

    Returns:
        Tuple of (features, targets_dict) or None if not found.
    """
    prefix = f"{domain}_{split}"

    feat_path = features_dir / f"features_{prefix}_{feature_level}.pt"
    if not feat_path.exists():
        feat_path = features_dir / f"features_{prefix}.pt"

    targets_path = features_dir / f"targets_{prefix}.pt"

    if not feat_path.exists() or not targets_path.exists():
        logger.warning(f"Features not found: {feat_path}")
        return None

    features = torch.load(feat_path, weights_only=True).numpy()
    targets = {k: v.numpy() for k, v in torch.load(targets_path, weights_only=True).items()}

    logger.info(f"Loaded {prefix}: features={features.shape}")
    return features, targets


def load_features_and_targets(
    condition_dir: Path,
    feature_level: str = "encoder10",
) -> dict[str, torch.Tensor]:
    """Load features and targets (single-domain compat).

    Args:
        condition_dir: Directory containing feature files.
        feature_level: Feature level to load.

    Returns:
        Dict with features and targets.
    """
    data: dict[str, torch.Tensor] = {}

    features_probe_path = condition_dir / f"features_probe_{feature_level}.pt"
    if not features_probe_path.exists():
        features_probe_path = condition_dir / "features_probe.pt"

    features_test_path = condition_dir / f"features_test_{feature_level}.pt"
    if not features_test_path.exists():
        features_test_path = condition_dir / "features_test.pt"

    if not features_probe_path.exists():
        raise FileNotFoundError(f"Features not found: {features_probe_path}")

    data["features_probe"] = torch.load(features_probe_path, weights_only=True)
    data["targets_probe"] = torch.load(condition_dir / "targets_probe.pt", weights_only=True)
    data["features_test"] = torch.load(features_test_path, weights_only=True)
    data["targets_test"] = torch.load(condition_dir / "targets_test.pt", weights_only=True)

    logger.info(
        f"Loaded features ({feature_level}): "
        f"probe={data['features_probe'].shape}, "
        f"test={data['features_test'].shape}"
    )

    return data


# =============================================================================
# Per-Domain Probes (B2)
# =============================================================================


def evaluate_per_domain_probes(
    features_dir: Path,
    probe_config: dict,
    feature_level: str = "encoder10",
) -> dict[str, dict]:
    """Evaluate GP probes on each domain separately (B2).

    Args:
        features_dir: Directory with per-domain feature files.
        probe_config: Probe configuration dict.
        feature_level: Feature level.

    Returns:
        Dict with per-domain probe results.
    """
    normalize_targets = probe_config.get("normalize_targets", True)
    n_restarts = probe_config.get("n_restarts", 3)
    r2_ci_samples = probe_config.get("r2_ci_samples", 500)

    results: dict[str, dict] = {}

    for domain in ("men", "gli"):
        probe_data = load_domain_features(features_dir, domain, "probe", feature_level)
        test_data = load_domain_features(features_dir, domain, "test", feature_level)

        if probe_data is None or test_data is None:
            logger.warning(f"Skipping {domain} probes — data not found")
            continue

        X_probe, targets_probe = probe_data
        X_test, targets_test = test_data

        logger.info(f"\nTraining {domain.upper()} GP probes...")
        probes = GPSemanticProbes(
            input_dim=X_probe.shape[1],
            normalize_targets=normalize_targets,
            n_restarts=n_restarts,
            r2_ci_samples=r2_ci_samples,
        )
        probes.fit(X_probe, targets_probe)
        gp_results = probes.evaluate(X_test, targets_test)
        summary = probes.get_summary(gp_results)

        # VICReg diagnostics (B5)
        variance = compute_variance_per_dim(X_test)
        summary["effective_rank"] = compute_effective_rank(X_test)
        summary["n_dead_dims"] = int((variance < 1e-6).sum())
        summary["n_low_variance_dims"] = int((variance < 0.01).sum())
        summary["variance_mean"] = float(np.mean(variance))
        summary["variance_min"] = float(np.min(variance))

        results[domain] = summary

        # Log primary metric (volume)
        vol_r2_str = f"vol={summary.get('r2_volume_linear', 0):.4f}"
        logger.info(f"{domain.upper()} GP-Linear R²: {vol_r2_str}")
        # Location is diagnostic (logged at DEBUG)
        if 'r2_location_linear' in summary:
            logger.debug(f"{domain.upper()} GP-Linear R² loc={summary['r2_location_linear']:.4f}")
        logger.info(
            f"{domain.upper()} effective rank: {summary['effective_rank']:.1f}, "
            f"dead dims: {summary['n_dead_dims']}"
        )

    return results


# =============================================================================
# Cross-Domain Probes (B3)
# =============================================================================


def evaluate_cross_domain_probes(
    features_dir: Path,
    probe_config: dict,
    feature_level: str = "encoder10",
) -> dict[str, dict]:
    """Evaluate cross-domain probes (B3): train on one domain, test on other.

    Args:
        features_dir: Directory with per-domain feature files.
        probe_config: Probe configuration.
        feature_level: Feature level.

    Returns:
        Dict with cross-domain results: "gli_to_men" and "men_to_gli".
    """
    normalize_targets = probe_config.get("normalize_targets", True)
    n_restarts = probe_config.get("n_restarts", 3)

    results: dict[str, dict] = {}

    men_probe = load_domain_features(features_dir, "men", "probe", feature_level)
    men_test = load_domain_features(features_dir, "men", "test", feature_level)
    gli_probe = load_domain_features(features_dir, "gli", "probe", feature_level)
    gli_test = load_domain_features(features_dir, "gli", "test", feature_level)

    if any(d is None for d in [men_probe, men_test, gli_probe, gli_test]):
        logger.warning("Skipping cross-domain probes — missing data")
        return results

    X_men_probe, targets_men_probe = men_probe
    X_men_test, targets_men_test = men_test
    X_gli_probe, targets_gli_probe = gli_probe
    X_gli_test, targets_gli_test = gli_test

    # GLI → MEN
    logger.info("\nCross-domain: train on GLI, test on MEN...")
    probes_gli = GPSemanticProbes(
        input_dim=X_gli_probe.shape[1],
        normalize_targets=normalize_targets,
        n_restarts=n_restarts,
        r2_ci_samples=0,
    )
    probes_gli.fit(X_gli_probe, targets_gli_probe)
    gli_on_men = probes_gli.evaluate(X_men_test, targets_men_test)
    results["gli_to_men"] = probes_gli.get_summary(gli_on_men)

    logger.info(
        f"GLI→MEN R²: vol={results['gli_to_men'].get('r2_volume_linear', 0):.4f}"
    )

    # MEN → GLI
    logger.info("\nCross-domain: train on MEN, test on GLI...")
    probes_men = GPSemanticProbes(
        input_dim=X_men_probe.shape[1],
        normalize_targets=normalize_targets,
        n_restarts=n_restarts,
        r2_ci_samples=0,
    )
    probes_men.fit(X_men_probe, targets_men_probe)
    men_on_gli = probes_men.evaluate(X_gli_test, targets_gli_test)
    results["men_to_gli"] = probes_men.get_summary(men_on_gli)

    logger.info(
        f"MEN→GLI R²: vol={results['men_to_gli'].get('r2_volume_linear', 0):.4f}"
    )

    return results


# =============================================================================
# Single-Domain Probes (lora_ablation compat)
# =============================================================================


def evaluate_probes_enhanced(
    condition_name: str,
    config: dict,
    device: str = "cuda",
) -> dict[str, float]:
    """GP probe evaluation with linear + RBF kernels (single-domain).

    Args:
        condition_name: Condition to evaluate.
        config: Experiment configuration.
        device: Device (for potential GPU acceleration).

    Returns:
        Dict with comprehensive metrics.
    """
    logger.info(f"GP probe evaluation for: {condition_name}")

    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    probe_config = config.get("probe", {})
    feature_level = config.get("feature_extraction", {}).get("level", "encoder10")
    normalize_targets = probe_config.get("normalize_targets", True)
    n_restarts = probe_config.get("n_restarts", 3)
    r2_ci_samples = probe_config.get("r2_ci_samples", 500)

    logger.info(f"Feature level: {feature_level}")
    logger.info(f"Normalize targets: {normalize_targets}")
    logger.info(f"GP restarts: {n_restarts}, CI samples: {r2_ci_samples}")

    data = load_features_and_targets(condition_dir, feature_level)

    X_probe = data["features_probe"].numpy()
    X_test = data["features_test"].numpy()

    targets_probe = {k: v.numpy() for k, v in data["targets_probe"].items() if k != "all"}
    targets_test = {k: v.numpy() for k, v in data["targets_test"].items() if k != "all"}

    logger.info("Training GP probes (linear + RBF)...")
    probes = GPSemanticProbes(
        input_dim=X_probe.shape[1],
        normalize_targets=normalize_targets,
        n_restarts=n_restarts,
        r2_ci_samples=r2_ci_samples,
    )
    probes.fit(X_probe, targets_probe)

    logger.info("Evaluating on test set...")
    results = probes.evaluate(X_test, targets_test)
    summary = probes.get_summary(results)

    # Additional diagnostics
    variance = compute_variance_per_dim(X_test)
    summary["variance_mean"] = float(np.mean(variance))
    summary["variance_min"] = float(np.min(variance))
    summary["variance_max"] = float(np.max(variance))
    summary["variance_std"] = float(np.std(variance))
    summary["num_low_variance_dims"] = int((variance < 0.01).sum())
    summary["num_collapsed_dims"] = int((variance < 1e-6).sum())

    # Per-dimension details (only for available targets)
    for target_name in targets_probe:
        if target_name in results.linear:
            summary[f"r2_{target_name}_per_dim_linear"] = results.linear[target_name].r2_per_dim.tolist()
        if target_name in results.rbf:
            summary[f"r2_{target_name}_per_dim_rbf"] = results.rbf[target_name].r2_per_dim.tolist()

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info(f"GP Probe Results for {condition_name}")
    logger.info("=" * 60)
    logger.info("\nGP-Linear Probes:")
    logger.info(f"  R² Volume:   {summary.get('r2_volume_linear', 0):.4f}")
    if "r2_location_linear" in summary:
        logger.debug(f"  R² Location: {summary['r2_location_linear']:.4f}")
    logger.info(f"  R² Mean:     {summary.get('r2_mean_linear', 0):.4f}")
    logger.info("\nGP-RBF Probes:")
    logger.info(f"  R² Volume:   {summary.get('r2_volume_rbf', 0):.4f}")
    if "r2_location_rbf" in summary:
        logger.debug(f"  R² Location: {summary['r2_location_rbf']:.4f}")
    logger.info(f"  R² Mean:     {summary.get('r2_mean_rbf', 0):.4f}")
    logger.info("=" * 60 + "\n")

    # Save metrics
    metrics_path = condition_dir / "metrics_enhanced.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved GP metrics to {metrics_path}")

    # Also update the original metrics.json (dynamic based on available targets)
    original_metrics = {
        "r2_volume": summary.get("r2_volume_linear", 0),
        "r2_mean": summary.get("r2_mean_linear", 0),
        "r2_volume_rbf": summary.get("r2_volume_rbf", 0),
        "r2_mean_rbf": summary.get("r2_mean_rbf", 0),
        "variance_mean": summary["variance_mean"],
        "variance_min": summary["variance_min"],
        "variance_std": summary["variance_std"],
    }
    # Add per-target metrics if available
    for target_name in targets_probe:
        if f"r2_{target_name}_linear" in summary:
            original_metrics[f"r2_{target_name}"] = summary[f"r2_{target_name}_linear"]
        if f"r2_{target_name}_per_dim_linear" in summary:
            original_metrics[f"r2_{target_name}_per_dim"] = summary[f"r2_{target_name}_per_dim_linear"]
        if f"mse_{target_name}_linear" in summary:
            original_metrics[f"mse_{target_name}"] = summary[f"mse_{target_name}_linear"]
    with open(condition_dir / "metrics.json", "w") as f:
        json.dump(original_metrics, f, indent=2)

    # Save trained probes
    probes_path = condition_dir / "probes_gp.pkl"
    with open(probes_path, "wb") as f:
        pickle.dump(probes, f)

    # Save predictions for visualization (dynamic based on available targets)
    predictions = {}
    for target_name in targets_test:
        if target_name in results.linear and target_name in results.rbf:
            predictions[target_name] = {
                "linear_mean": results.linear[target_name].predictions.tolist(),
                "linear_std": results.linear[target_name].predictive_std.tolist(),
                "rbf_mean": results.rbf[target_name].predictions.tolist(),
                "rbf_std": results.rbf[target_name].predictive_std.tolist(),
                "ground_truth": targets_test[target_name].tolist(),
            }
    with open(condition_dir / "predictions_enhanced.json", "w") as f:
        json.dump(predictions, f)

    return summary


# =============================================================================
# Unified Entry Point
# =============================================================================


def evaluate_probes(
    condition_name: str,
    config: dict,
    device: str = "cuda",
) -> dict[str, dict]:
    """Full probe evaluation for a condition.

    Automatically detects dual-domain vs single-domain from config.

    Args:
        condition_name: Condition name.
        config: Full experiment config.
        device: Device.

    Returns:
        Dict with all probe results.
    """
    is_dual = "men_h5_file" in config.get("paths", {}) and "gli_h5_file" in config.get("paths", {})

    if is_dual:
        logger.info(f"Probe evaluation (dual-domain) for: {condition_name}")

        output_dir = Path(config["experiment"]["output_dir"])
        condition_dir = output_dir / "conditions" / condition_name
        features_dir = condition_dir / "features"
        probes_dir = condition_dir / "probes"
        probes_dir.mkdir(parents=True, exist_ok=True)

        probe_config = config.get("probe", {})
        feature_level = config.get("feature_extraction", {}).get("level", "encoder10")

        # B2: Per-domain probes
        per_domain = evaluate_per_domain_probes(features_dir, probe_config, feature_level)
        for domain, summary in per_domain.items():
            with open(probes_dir / f"{domain}_probes.json", "w") as f:
                json.dump(summary, f, indent=2)

        # B3: Cross-domain probes
        cross_domain = evaluate_cross_domain_probes(features_dir, probe_config, feature_level)
        if cross_domain:
            with open(probes_dir / "cross_domain_probes.json", "w") as f:
                json.dump(cross_domain, f, indent=2)

        all_results = {
            "per_domain": per_domain,
            "cross_domain": cross_domain,
        }

        with open(probes_dir / "all_probes.json", "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Saved probe results to {probes_dir}")
        return all_results
    else:
        # Single-domain evaluation
        summary = evaluate_probes_enhanced(condition_name, config, device)
        return {"single_domain": summary}


def main(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])
    evaluate_probes(condition, config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GP probe evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
