# experiments/stage1_volumetric/run_stage1.py
"""Stage 1: Volumetric Baseline — LOPO-CV on manual volume trajectories.

Loads per-patient log(V_WT + 1) trajectories directly from the MenGrowth H5,
evaluates three growth models (ScalarGP, LME, HGP) plus an optional Gompertz
ablation under LOPO-CV, and produces bootstrap CIs and per-patient error
distributions.

Usage::

    cd /home/mpascual/research/code/MenGrowth-Model
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_stage1 \\
        --config experiments/stage1_volumetric/config_stage1.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from growth.models.growth.hgp_model import HierarchicalGPModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.models.growth.scalar_gp import ScalarGP
from growth.shared.bootstrap import BootstrapResult, bootstrap_metric
from growth.shared.lopo import LOPOEvaluator, LOPOResults
from growth.shared.metrics import compute_mae, compute_r2, compute_rmse
from growth.stages.stage1_volumetric.trajectory_loader import load_trajectories_from_h5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_model_configs(cfg: dict) -> dict[str, tuple[type, dict]]:
    """Build model class + kwargs from config.

    Returns:
        Dict mapping model display name to (model_class, kwargs).
    """
    gp_cfg = cfg["gp"]
    seed = cfg["experiment"]["seed"]
    models_cfg = cfg.get("models", {})

    cov_cfg = cfg.get("covariates", {})
    cov_enabled = cov_cfg.get("enabled", False)
    cov_features = cov_cfg.get("features", [])
    cov_missing = cov_cfg.get("missing_strategy", "skip")

    cov_kwargs: dict = {}
    if cov_enabled:
        cov_kwargs = {
            "use_covariates": True,
            "covariate_names": cov_features,
            "missing_strategy": cov_missing,
        }

    shared_gp_kwargs = {
        "kernel_type": gp_cfg["kernel"],
        "n_restarts": gp_cfg["n_restarts"],
        "max_iter": gp_cfg["max_iter"],
        "lengthscale_bounds": tuple(gp_cfg["lengthscale_bounds"]),
        "signal_var_bounds": tuple(gp_cfg["signal_var_bounds"]),
        "noise_var_bounds": tuple(gp_cfg["noise_var_bounds"]),
        "seed": seed,
        **cov_kwargs,
    }

    models: dict[str, tuple[type, dict]] = {}

    if models_cfg.get("scalar_gp", True):
        models["ScalarGP"] = (
            ScalarGP,
            {**shared_gp_kwargs, "mean_function": gp_cfg.get("mean_function", "linear")},
        )

    if models_cfg.get("lme", True):
        lme_cfg = cfg.get("lme", {})
        models["LME"] = (
            LMEGrowthModel,
            {"method": lme_cfg.get("method", "reml"), **cov_kwargs},
        )

    if models_cfg.get("hgp", True):
        models["HGP"] = (
            HierarchicalGPModel,
            {**shared_gp_kwargs, "mean_function": "linear"},
        )

    if models_cfg.get("hgp_gompertz", False):
        models["HGP_Gompertz"] = (
            HierarchicalGPModel,
            {**shared_gp_kwargs, "mean_function": "gompertz"},
        )

    return models


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def compute_bootstrap_cis(
    lopo_results: LOPOResults,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, BootstrapResult]:
    """Compute bootstrap CIs on LOPO-CV metrics.

    Resamples per-patient predictions (last_from_rest) and computes
    R^2, MAE, RMSE on each bootstrap sample.

    Args:
        lopo_results: LOPO-CV results for one model.
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: CI confidence level.
        seed: Random seed.

    Returns:
        Dict mapping metric name to BootstrapResult.
    """
    # Collect per-prediction actuals and predictions
    actuals: list[float] = []
    preds: list[float] = []

    for fr in lopo_results.fold_results:
        if "last_from_rest" not in fr.predictions:
            continue
        for p in fr.predictions["last_from_rest"]:
            actuals.append(p["actual"])
            preds.append(p["pred_mean"])

    if len(actuals) < 3:
        logger.warning("Not enough predictions for bootstrap CIs")
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
# Per-patient error analysis
# ---------------------------------------------------------------------------


def compute_per_patient_errors(
    lopo_results: LOPOResults,
) -> dict[str, dict]:
    """Compute per-patient prediction errors from LOPO-CV.

    Returns:
        Dict mapping patient_id to error statistics.
    """
    errors: dict[str, dict] = {}

    for fr in lopo_results.fold_results:
        if "last_from_rest" not in fr.predictions:
            continue
        for p in fr.predictions["last_from_rest"]:
            err = p["pred_mean"] - p["actual"]
            abs_err = abs(err)
            within_ci = p["lower_95"] <= p["actual"] <= p["upper_95"]
            errors[fr.patient_id] = {
                "error": err,
                "abs_error": abs_err,
                "actual": p["actual"],
                "predicted": p["pred_mean"],
                "within_95_ci": within_ci,
                "ci_width": p["upper_95"] - p["lower_95"],
                "n_conditioning": p["n_conditioning"],
            }

    return errors


def summarize_per_patient_errors(errors: dict[str, dict]) -> dict:
    """Compute summary statistics for per-patient error distribution."""
    if not errors:
        return {}

    abs_errors = [e["abs_error"] for e in errors.values()]
    signed_errors = [e["error"] for e in errors.values()]
    ci_widths = [e["ci_width"] for e in errors.values()]

    return {
        "n_patients": len(errors),
        "abs_error_mean": float(np.mean(abs_errors)),
        "abs_error_std": float(np.std(abs_errors)),
        "abs_error_median": float(np.median(abs_errors)),
        "abs_error_min": float(np.min(abs_errors)),
        "abs_error_max": float(np.max(abs_errors)),
        "abs_error_q25": float(np.percentile(abs_errors, 25)),
        "abs_error_q75": float(np.percentile(abs_errors, 75)),
        "signed_error_mean": float(np.mean(signed_errors)),
        "signed_error_std": float(np.std(signed_errors)),
        "ci_width_mean": float(np.mean(ci_widths)),
        "ci_width_std": float(np.std(ci_widths)),
        "fraction_within_ci": float(np.mean([e["within_95_ci"] for e in errors.values()])),
    }


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------


def save_results(
    lopo_results: dict[str, LOPOResults],
    bootstrap_cis: dict[str, dict[str, BootstrapResult]],
    per_patient_errors: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    """Save all Stage 1 results to organized directory structure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-model results
    for model_name, results in lopo_results.items():
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # LOPO results
        with open(model_dir / "lopo_results.json", "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        # Bootstrap CIs
        if model_name in bootstrap_cis:
            ci_data = {}
            for metric, br in bootstrap_cis[model_name].items():
                ci_data[metric] = {
                    "estimate": br.estimate,
                    "ci_lower": br.ci_lower,
                    "ci_upper": br.ci_upper,
                    "confidence_level": br.confidence_level,
                    "n_bootstrap": br.n_bootstrap,
                }
            with open(model_dir / "bootstrap_cis.json", "w") as f:
                json.dump(ci_data, f, indent=2)

        # Per-patient errors
        if model_name in per_patient_errors:
            errors = per_patient_errors[model_name]
            with open(model_dir / "per_patient_errors.json", "w") as f:
                json.dump(errors, f, indent=2, default=str)

            summary = summarize_per_patient_errors(errors)
            with open(model_dir / "error_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    # Model comparison
    comparison = _build_comparison(lopo_results, bootstrap_cis)
    with open(output_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def _build_comparison(
    lopo_results: dict[str, LOPOResults],
    bootstrap_cis: dict[str, dict[str, BootstrapResult]],
) -> dict:
    """Build head-to-head comparison table."""
    comparison: dict = {"models": {}}

    for model_name, results in lopo_results.items():
        entry: dict = {
            "model_name": results.model_name,
            "n_folds": len(results.fold_results),
            "n_failed": len(results.failed_folds),
        }

        for metric_name, val in sorted(results.aggregate_metrics.items()):
            entry[metric_name] = val

        # Add bootstrap CI info
        if model_name in bootstrap_cis:
            ci_info: dict = {}
            for metric, br in bootstrap_cis[model_name].items():
                ci_info[metric] = {
                    "estimate": br.estimate,
                    "ci_lower": br.ci_lower,
                    "ci_upper": br.ci_upper,
                }
            entry["bootstrap_ci"] = ci_info

        comparison["models"][model_name] = entry

    # Rank by R^2
    r2_ranking = sorted(
        comparison["models"].items(),
        key=lambda x: x[1].get("last_from_rest/r2_log", float("-inf")),
        reverse=True,
    )
    comparison["ranking"] = [
        {"model": k, "r2_log": v.get("last_from_rest/r2_log")} for k, v in r2_ranking
    ]

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Stage 1 Volumetric Baseline evaluation."""
    parser = argparse.ArgumentParser(description="Stage 1: Volumetric Baseline LOPO-CV")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/stage1_volumetric/config_stage1.yaml",
        help="Path to Stage 1 config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])

    # =========================================================================
    # Step 1: Load trajectories from H5
    # =========================================================================
    logger.info("=== Step 1: Load Trajectories ===")

    cov_cfg = cfg.get("covariates", {})
    covariate_features = cov_cfg.get("features", []) if cov_cfg.get("enabled", False) else []

    trajectories = load_trajectories_from_h5(
        h5_path=cfg["paths"]["mengrowth_h5"],
        time_variable=cfg["time"]["variable"],
        exclude_patients=cfg["patients"].get("exclude", []),
        min_timepoints=cfg["patients"].get("min_timepoints", 2),
        covariate_features=covariate_features,
        skip_all_zero_volume=cfg["patients"].get("skip_all_zero_volume", True),
    )

    logger.info(f"Loaded {len(trajectories)} patient trajectories")
    for traj in trajectories[:3]:
        logger.info(
            f"  {traj.patient_id}: {traj.n_timepoints} tp, "
            f"obs range [{traj.observations.min():.2f}, {traj.observations.max():.2f}]"
        )

    # =========================================================================
    # Step 2: LOPO-CV with all models
    # =========================================================================
    logger.info("=== Step 2: LOPO-CV ===")
    model_configs = _build_model_configs(cfg)
    evaluator = LOPOEvaluator()

    lopo_results: dict[str, LOPOResults] = {}

    for model_name, (model_cls, kwargs) in model_configs.items():
        logger.info(f"--- {model_name} ---")
        try:
            results = evaluator.evaluate(model_cls, trajectories, **kwargs)
            lopo_results[model_name] = results

            r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("nan"))
            mae = results.aggregate_metrics.get("last_from_rest/mae_log", float("nan"))
            cal = results.aggregate_metrics.get("last_from_rest/calibration_95", float("nan"))
            logger.info(
                f"  R2_log={r2:.4f}, MAE_log={mae:.4f}, Cal_95={cal:.3f}, "
                f"folds={len(results.fold_results)}/"
                f"{len(results.fold_results) + len(results.failed_folds)}"
            )
        except Exception as e:
            logger.error(f"  {model_name} FAILED: {e}", exc_info=True)

    # =========================================================================
    # Step 3: Bootstrap CIs
    # =========================================================================
    bootstrap_cfg = cfg.get("bootstrap", {})
    bootstrap_cis: dict[str, dict[str, BootstrapResult]] = {}

    if bootstrap_cfg.get("enabled", True):
        logger.info("=== Step 3: Bootstrap CIs ===")
        for model_name, results in lopo_results.items():
            cis = compute_bootstrap_cis(
                results,
                n_bootstrap=bootstrap_cfg.get("n_samples", 2000),
                confidence_level=bootstrap_cfg.get("confidence_level", 0.95),
                seed=bootstrap_cfg.get("seed", 42),
            )
            bootstrap_cis[model_name] = cis
            if "r2_log" in cis:
                br = cis["r2_log"]
                logger.info(
                    f"  {model_name}: R2_log = {br.estimate:.4f} "
                    f"[{br.ci_lower:.4f}, {br.ci_upper:.4f}]"
                )

    # =========================================================================
    # Step 4: Per-patient error analysis
    # =========================================================================
    logger.info("=== Step 4: Per-Patient Error Analysis ===")
    per_patient_errors: dict[str, dict[str, dict]] = {}

    for model_name, results in lopo_results.items():
        errors = compute_per_patient_errors(results)
        per_patient_errors[model_name] = errors
        summary = summarize_per_patient_errors(errors)
        if summary:
            logger.info(
                f"  {model_name}: |error| = {summary['abs_error_mean']:.4f} "
                f"+/- {summary['abs_error_std']:.4f}, "
                f"range [{summary['abs_error_min']:.4f}, {summary['abs_error_max']:.4f}]"
            )

    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    logger.info("=== Step 5: Save Results ===")
    save_results(lopo_results, bootstrap_cis, per_patient_errors, output_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 72)
    print("STAGE 1: VOLUMETRIC BASELINE — RESULTS")
    print("=" * 72)
    print(
        f"\n  {'Model':<16} {'R2_log':>8} {'MAE_log':>8} {'RMSE_log':>9} "
        f"{'Cal_95':>7} {'CI_width':>8}   {'R2 95% CI':>20}"
    )
    print("  " + "-" * 80)

    for model_name in model_configs:
        if model_name not in lopo_results:
            continue
        m = lopo_results[model_name].aggregate_metrics

        ci_str = ""
        if model_name in bootstrap_cis and "r2_log" in bootstrap_cis[model_name]:
            br = bootstrap_cis[model_name]["r2_log"]
            ci_str = f"[{br.ci_lower:.4f}, {br.ci_upper:.4f}]"

        print(
            f"  {model_name:<16} "
            f"{m.get('last_from_rest/r2_log', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/mae_log', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/rmse_log', float('nan')):>9.4f} "
            f"{m.get('last_from_rest/calibration_95', float('nan')):>7.3f} "
            f"{m.get('last_from_rest/mean_ci_width_log', float('nan')):>8.4f}   "
            f"{ci_str:>20}"
        )

    # Stage 1 gate checks
    print("\n--- Stage 1 Gate Checks ---")
    best_r2 = float("-inf")
    best_model = None
    for model_name, results in lopo_results.items():
        r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("-inf"))
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name

    # S1-T1: ScalarGP completes without NaN
    if "ScalarGP" in lopo_results:
        n_failed = len(lopo_results["ScalarGP"].failed_folds)
        print(f"  S1-T1 ScalarGP no NaN: {'PASS' if n_failed == 0 else 'FAIL'} ({n_failed} failed)")

    # S1-T2: LME R2 > 0
    if "LME" in lopo_results:
        lme_r2 = lopo_results["LME"].aggregate_metrics.get("last_from_rest/r2_log", -1)
        print(f"  S1-T2 LME R2 > 0: {'PASS' if lme_r2 > 0 else 'FAIL'} (R2={lme_r2:.4f})")

    # S1-T3: HGP >= ScalarGP
    if "HGP" in lopo_results and "ScalarGP" in lopo_results:
        hgp_r2 = lopo_results["HGP"].aggregate_metrics.get("last_from_rest/r2_log", -999)
        sgp_r2 = lopo_results["ScalarGP"].aggregate_metrics.get("last_from_rest/r2_log", -999)
        print(
            f"  S1-T3 HGP >= ScalarGP: "
            f"{'PASS' if hgp_r2 >= sgp_r2 else 'DIAGNOSTIC-FAIL'} "
            f"({hgp_r2:.4f} vs {sgp_r2:.4f})"
        )

    # S1-T5: Bootstrap CI excludes 0
    if best_model and best_model in bootstrap_cis and "r2_log" in bootstrap_cis[best_model]:
        br = bootstrap_cis[best_model]["r2_log"]
        excludes_0 = br.ci_lower > 0
        print(
            f"  S1-T5 Best model CI excludes 0: "
            f"{'PASS' if excludes_0 else 'FAIL'} "
            f"({best_model} [{br.ci_lower:.4f}, {br.ci_upper:.4f}])"
        )

    # S1-T7: Gompertz ablation
    if "HGP_Gompertz" in lopo_results and "HGP" in lopo_results:
        gomp_r2 = lopo_results["HGP_Gompertz"].aggregate_metrics.get("last_from_rest/r2_log", -999)
        hgp_r2 = lopo_results["HGP"].aggregate_metrics.get("last_from_rest/r2_log", -999)
        print(
            f"  S1-T7 Gompertz ablation: "
            f"HGP_Gompertz R2={gomp_r2:.4f} vs HGP_Linear R2={hgp_r2:.4f} "
            f"(delta={gomp_r2 - hgp_r2:+.4f})"
        )

    print(f"\n  Best model: {best_model} (R2={best_r2:.4f})")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
