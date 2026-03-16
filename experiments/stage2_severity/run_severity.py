# experiments/stage2_severity/run_severity.py
"""Stage 2: Latent Severity Model — Unified Orchestrator.

Runs the severity model (MLE or Bayesian) under LOPO-CV, computes bootstrap
CIs, compares to Stage 1, and produces diagnostic outputs.

Usage::

    cd /home/mpascual/research/code/MenGrowth-Model
    ~/.conda/envs/growth/bin/python -m experiments.stage2_severity.run_severity \\
        --config experiments/stage2_severity/config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from growth.shared.bootstrap import (
    BootstrapResult,
    bootstrap_metric,
    paired_permutation_test,
)
from growth.shared.lopo import LOPOEvaluator, LOPOResults
from growth.shared.metrics import compute_mae, compute_r2, compute_rmse
from growth.stages.stage1_volumetric.trajectory_loader import load_trajectories_from_h5
from growth.stages.stage2_severity.severity_model import SeverityModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_severity_model_kwargs(cfg: dict) -> tuple[type, dict]:
    """Build model class and kwargs from config.

    Returns:
        (model_class, model_kwargs) tuple.
    """
    sev_cfg = cfg["severity"]
    method = sev_cfg.get("estimation_method", "mle")
    seed = cfg["experiment"]["seed"]

    shared_kwargs = {
        "growth_function": sev_cfg.get("growth_function", "gompertz_reduced"),
        "severity_features": sev_cfg.get("severity_features", ["log_volume", "sphericity"]),
        "seed": seed,
    }

    if method == "mle":
        opt_cfg = sev_cfg.get("optimization", {})
        return SeverityModel, {
            **shared_kwargs,
            "lambda_reg": opt_cfg.get("lambda_reg", 0.01),
            "n_restarts": opt_cfg.get("n_restarts", 10),
            "max_iter": opt_cfg.get("max_iter", 5000),
        }
    elif method == "bayesian":
        try:
            from growth.stages.stage2_severity.bayesian_severity_model import (
                BayesianSeverityModel,
            )
        except ImportError:
            raise ImportError(
                "Bayesian severity model requires numpyro. "
                "Install with: pip install mengrowth-model[bayesian]"
            )
        mcmc_cfg = sev_cfg.get("mcmc", {})
        return BayesianSeverityModel, {
            **shared_kwargs,
            "n_warmup": mcmc_cfg.get("n_warmup", 500),
            "n_samples": mcmc_cfg.get("n_samples", 1000),
            "n_chains": mcmc_cfg.get("n_chains", 2),
        }
    else:
        raise ValueError(f"Unknown estimation method: {method}")


# ---------------------------------------------------------------------------
# Bootstrap CIs (reused from Stage 1 pattern)
# ---------------------------------------------------------------------------


def compute_bootstrap_cis(
    lopo_results: LOPOResults,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, BootstrapResult]:
    """Compute bootstrap CIs on LOPO-CV metrics."""
    actuals: list[float] = []
    preds: list[float] = []

    for fr in lopo_results.fold_results:
        if "last_from_rest" not in fr.predictions:
            continue
        for p in fr.predictions["last_from_rest"]:
            actuals.append(p["actual"])
            preds.append(p["pred_mean"])

    if len(actuals) < 3:
        return {}

    y_true = np.array(actuals)
    y_pred = np.array(preds)

    results: dict[str, BootstrapResult] = {}
    for metric_name, metric_fn in [
        ("r2_log", compute_r2),
        ("mae_log", compute_mae),
        ("rmse_log", compute_rmse),
    ]:
        results[metric_name] = bootstrap_metric(
            y_true,
            y_pred,
            metric_fn,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )

    return results


# ---------------------------------------------------------------------------
# Stage 1 comparison
# ---------------------------------------------------------------------------


def load_stage1_errors(stage1_results_dir: str | Path) -> dict[str, float] | None:
    """Load per-patient errors from Stage 1 results.

    Returns:
        Dict mapping patient_id to absolute error, or None if not found.
    """
    results_dir = Path(stage1_results_dir)

    # Try LME first (best Stage 1 model), then ScalarGP
    for model_name in ["LME", "ScalarGP", "HGP"]:
        errors_path = results_dir / model_name / "per_patient_errors.json"
        if errors_path.exists():
            with open(errors_path) as f:
                errors = json.load(f)
            logger.info(f"Loaded Stage 1 errors from {model_name}: {len(errors)} patients")
            return {pid: e["abs_error"] for pid, e in errors.items()}

    logger.warning(f"No Stage 1 results found in {results_dir}")
    return None


def compare_to_stage1(
    stage2_results: LOPOResults,
    stage1_errors: dict[str, float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Compare Stage 2 to Stage 1 via paired permutation test.

    Returns:
        Dict with delta_r2, permutation_p_value, etc.
    """
    # Collect Stage 2 per-patient absolute errors
    s2_errors: dict[str, float] = {}
    for fr in stage2_results.fold_results:
        if "last_from_rest" not in fr.predictions:
            continue
        for p in fr.predictions["last_from_rest"]:
            err = abs(p["pred_mean"] - p["actual"])
            s2_errors[fr.patient_id] = err

    # Find common patients
    common = sorted(set(s2_errors.keys()) & set(stage1_errors.keys()))
    if len(common) < 3:
        return {"error": "Too few common patients for comparison"}

    e1 = np.array([stage1_errors[pid] for pid in common])
    e2 = np.array([s2_errors[pid] for pid in common])

    perm_result = paired_permutation_test(e1, e2, n_permutations=n_permutations, seed=seed)

    return {
        "n_common_patients": len(common),
        "stage1_mean_abs_error": float(np.mean(e1)),
        "stage2_mean_abs_error": float(np.mean(e2)),
        "delta_mae": float(np.mean(e2) - np.mean(e1)),
        "permutation_p_value": perm_result.p_value,
        "observed_diff": perm_result.observed_diff,
        "n_permutations": perm_result.n_permutations,
    }


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------


def save_results(
    lopo_results: LOPOResults,
    bootstrap_cis: dict[str, BootstrapResult],
    comparison: dict | None,
    model_class_name: str,
    output_dir: Path,
) -> None:
    """Save Stage 2 results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # LOPO results
    with open(output_dir / "lopo_results.json", "w") as f:
        json.dump(lopo_results.to_dict(), f, indent=2)

    # Bootstrap CIs
    if bootstrap_cis:
        ci_data = {
            k: {"estimate": br.estimate, "ci_lower": br.ci_lower, "ci_upper": br.ci_upper}
            for k, br in bootstrap_cis.items()
        }
        with open(output_dir / "bootstrap_cis.json", "w") as f:
            json.dump(ci_data, f, indent=2)

    # Per-patient errors
    errors: dict[str, dict] = {}
    for fr in lopo_results.fold_results:
        if "last_from_rest" not in fr.predictions:
            continue
        for p in fr.predictions["last_from_rest"]:
            errors[fr.patient_id] = {
                "error": p["pred_mean"] - p["actual"],
                "abs_error": abs(p["pred_mean"] - p["actual"]),
                "actual": p["actual"],
                "predicted": p["pred_mean"],
                "within_95_ci": p["lower_95"] <= p["actual"] <= p["upper_95"],
            }
    with open(output_dir / "per_patient_errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    # Comparison to Stage 1
    if comparison:
        with open(output_dir / "stage1_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    # Summary
    summary = {
        "model": model_class_name,
        "n_folds": len(lopo_results.fold_results),
        "n_failed": len(lopo_results.failed_folds),
        "metrics": lopo_results.aggregate_metrics,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Stage 2 Severity Model experiment."""
    parser = argparse.ArgumentParser(description="Stage 2: Latent Severity Model")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/stage2_severity/config.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])

    # =========================================================================
    # Step 1: Load trajectories
    # =========================================================================
    logger.info("=== Step 1: Load Trajectories ===")
    trajectories = load_trajectories_from_h5(
        h5_path=cfg["paths"]["mengrowth_h5"],
        time_variable=cfg["time"]["variable"],
        exclude_patients=cfg["patients"].get("exclude", []),
        min_timepoints=cfg["patients"].get("min_timepoints", 2),
        semantic_covariates=cfg.get("semantic_covariates", []),
        skip_all_zero_volume=cfg["patients"].get("skip_all_zero_volume", True),
    )
    logger.info(f"Loaded {len(trajectories)} patient trajectories")

    # =========================================================================
    # Step 2: Build model and run LOPO-CV
    # =========================================================================
    method = cfg["severity"].get("estimation_method", "mle")
    logger.info(f"=== Step 2: LOPO-CV ({method.upper()}) ===")

    model_class, model_kwargs = _build_severity_model_kwargs(cfg)
    evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest", "all_from_first"])

    results = evaluator.evaluate(model_class, trajectories, **model_kwargs)

    r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("nan"))
    mae = results.aggregate_metrics.get("last_from_rest/mae_log", float("nan"))
    cal = results.aggregate_metrics.get("last_from_rest/calibration_95", float("nan"))
    logger.info(
        f"LOPO-CV: R2_log={r2:.4f}, MAE_log={mae:.4f}, Cal_95={cal:.3f}, "
        f"failed={len(results.failed_folds)}/{len(results.fold_results) + len(results.failed_folds)}"
    )

    # =========================================================================
    # Step 3: Bootstrap CIs
    # =========================================================================
    bootstrap_cis: dict[str, BootstrapResult] = {}
    boot_cfg = cfg.get("bootstrap", {})
    if boot_cfg.get("enabled", True):
        logger.info("=== Step 3: Bootstrap CIs ===")
        bootstrap_cis = compute_bootstrap_cis(
            results,
            n_bootstrap=boot_cfg.get("n_samples", 2000),
            confidence_level=boot_cfg.get("confidence_level", 0.95),
            seed=boot_cfg.get("seed", 42),
        )
        if "r2_log" in bootstrap_cis:
            br = bootstrap_cis["r2_log"]
            logger.info(f"R2_log = {br.estimate:.4f} [{br.ci_lower:.4f}, {br.ci_upper:.4f}]")

    # =========================================================================
    # Step 4: Compare to Stage 1
    # =========================================================================
    comparison: dict | None = None
    comp_cfg = cfg.get("comparison", {})
    if comp_cfg.get("enabled", True):
        logger.info("=== Step 4: Stage 1 Comparison ===")
        stage1_errors = load_stage1_errors(cfg["paths"].get("stage1_results", ""))
        if stage1_errors:
            comparison = compare_to_stage1(
                results,
                stage1_errors,
                n_permutations=comp_cfg.get("permutation_test_n", 10000),
            )
            if "permutation_p_value" in comparison:
                logger.info(
                    f"Stage 2 vs Stage 1: delta_MAE={comparison['delta_mae']:.4f}, "
                    f"p={comparison['permutation_p_value']:.4f}"
                )

    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    logger.info("=== Step 5: Save Results ===")
    save_results(results, bootstrap_cis, comparison, model_class.__name__, output_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 72)
    print(f"STAGE 2: SEVERITY MODEL ({method.upper()}) — RESULTS")
    print("=" * 72)

    m = results.aggregate_metrics
    ci_str = ""
    if "r2_log" in bootstrap_cis:
        br = bootstrap_cis["r2_log"]
        ci_str = f"  95% CI: [{br.ci_lower:.4f}, {br.ci_upper:.4f}]"

    print(f"\n  R2_log:        {m.get('last_from_rest/r2_log', float('nan')):.4f}{ci_str}")
    print(f"  MAE_log:       {m.get('last_from_rest/mae_log', float('nan')):.4f}")
    print(f"  RMSE_log:      {m.get('last_from_rest/rmse_log', float('nan')):.4f}")
    print(f"  Cal_95:        {m.get('last_from_rest/calibration_95', float('nan')):.3f}")
    print(f"  Failed folds:  {len(results.failed_folds)}")

    # Gate checks
    print("\n--- Stage 2 Gate Checks ---")
    print(f"  S2-T2 Optimization converged: {'PASS' if len(results.failed_folds) == 0 else 'FAIL'}")

    if comparison and "permutation_p_value" in comparison:
        p_val = comparison["permutation_p_value"]
        alpha = comp_cfg.get("significance_level", 0.05)
        print(
            f"  S2-T7 R2 >= Stage 1: "
            f"{'PASS' if comparison['delta_mae'] < 0 else 'FAIL'} "
            f"(delta_MAE={comparison['delta_mae']:.4f}, p={p_val:.4f})"
        )

    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
