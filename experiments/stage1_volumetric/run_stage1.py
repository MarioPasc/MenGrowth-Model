# experiments/stage1_volumetric/run_stage1.py
"""Stage 1: Volumetric Baseline — LOPO-CV on manual volume trajectories.

Loads per-patient log(V_ET + 1) trajectories directly from the MenGrowth H5
(enhancing tumor = label 3 in BraTS-MEN; peritumoral edema is excluded),
evaluates three growth models (ScalarGP, LME, HGP) plus an optional Gompertz
ablation under LOPO-CV, and produces bootstrap CIs and per-patient error
distributions.

Usage::

    cd /home/mpascual/research/code/MenGrowth-Model
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_stage1 \\
        --config experiments/stage1_volumetric/config.yaml
"""

import argparse
import logging
from pathlib import Path

import yaml

from experiments.utils.experiment_output import (
    save_experiment_metadata,
    save_stage_results,
)
from growth.models.growth.hgp_model import HierarchicalGPModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.models.growth.scalar_gp import ScalarGP
from growth.shared.bootstrap import BootstrapResult
from growth.shared.lopo import LOPOEvaluator, LOPOResults
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Stage 1 Volumetric Baseline evaluation."""
    parser = argparse.ArgumentParser(description="Stage 1: Volumetric Baseline LOPO-CV")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/stage1_volumetric/config.yaml",
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
        semantic_covariates=cfg.get("semantic_covariates", []),
        skip_all_zero_volume=cfg["patients"].get("skip_all_zero_volume", True),
    )

    logger.info(f"Loaded {len(trajectories)} patient trajectories")

    # =========================================================================
    # Step 2: LOPO-CV with all models
    # =========================================================================
    logger.info("=== Step 2: LOPO-CV ===")
    model_configs = _build_model_configs(cfg)
    evaluator = LOPOEvaluator()

    lopo_results: dict[str, LOPOResults] = {}
    all_bootstrap_cis: dict[str, dict[str, BootstrapResult]] = {}

    bootstrap_cfg = cfg.get("bootstrap", {})
    bootstrap_n = bootstrap_cfg.get("n_samples", 2000)
    bootstrap_seed = bootstrap_cfg.get("seed", 42)

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

            # Save per-model results with shared output module
            ci_results = save_stage_results(
                output_dir=output_dir,
                model_name=model_name,
                lopo_results=results,
                trajectories=trajectories,
                config=cfg,
                stage_name="stage1_volumetric",
                bootstrap_n=bootstrap_n,
                bootstrap_seed=bootstrap_seed,
            )
            if ci_results:
                all_bootstrap_cis[model_name] = ci_results

        except Exception as e:
            logger.error(f"  {model_name} FAILED: {e}", exc_info=True)

    # =========================================================================
    # Step 3: Save experiment-level metadata and comparison
    # =========================================================================
    logger.info("=== Step 3: Save Results ===")
    save_experiment_metadata(
        output_dir=output_dir,
        trajectories=trajectories,
        config=cfg,
        stage_name="stage1_volumetric",
        all_model_results=lopo_results,
        all_bootstrap_cis=all_bootstrap_cis,
    )

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
        if model_name in all_bootstrap_cis and "r2_log" in all_bootstrap_cis[model_name]:
            br = all_bootstrap_cis[model_name]["r2_log"]
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

    # Gate checks
    print("\n--- Stage 1 Gate Checks ---")
    best_r2 = float("-inf")
    best_model = None
    for model_name, results in lopo_results.items():
        r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("-inf"))
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name

    if "ScalarGP" in lopo_results:
        n_failed = len(lopo_results["ScalarGP"].failed_folds)
        print(f"  S1-T1 ScalarGP no NaN: {'PASS' if n_failed == 0 else 'FAIL'} ({n_failed} failed)")

    if "LME" in lopo_results:
        lme_r2 = lopo_results["LME"].aggregate_metrics.get("last_from_rest/r2_log", -1)
        print(f"  S1-T2 LME R2 > 0: {'PASS' if lme_r2 > 0 else 'FAIL'} (R2={lme_r2:.4f})")

    if "HGP" in lopo_results and "ScalarGP" in lopo_results:
        hgp_r2 = lopo_results["HGP"].aggregate_metrics.get("last_from_rest/r2_log", -999)
        sgp_r2 = lopo_results["ScalarGP"].aggregate_metrics.get("last_from_rest/r2_log", -999)
        print(
            f"  S1-T3 HGP >= ScalarGP: "
            f"{'PASS' if hgp_r2 >= sgp_r2 else 'DIAGNOSTIC-FAIL'} "
            f"({hgp_r2:.4f} vs {sgp_r2:.4f})"
        )

    if best_model and best_model in all_bootstrap_cis and "r2_log" in all_bootstrap_cis[best_model]:
        br = all_bootstrap_cis[best_model]["r2_log"]
        excludes_0 = br.ci_lower > 0
        print(
            f"  S1-T5 Best model CI excludes 0: "
            f"{'PASS' if excludes_0 else 'FAIL'} "
            f"({best_model} [{br.ci_lower:.4f}, {br.ci_upper:.4f}])"
        )

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
