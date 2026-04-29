# experiments/stage1_volumetric/run_stage1_uq.py
"""Stage 1: Uncertainty-Propagated Volume Prediction — LOPO-CV.

Loads per-patient log(V_ET + 1) trajectories with per-observation variance
from the LoRA-ensemble uncertainty group, evaluates paired homoscedastic
and heteroscedastic growth models under LOPO-CV, and produces paired
comparisons (ΔR², ΔCRPS, ΔCov_95).

Usage::

    cd /home/mpascual/research/code/MenGrowth-Model
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_stage1_uq \\
        --config experiments/stage1_volumetric/config_uq.yaml

    # With real time:
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_stage1_uq \\
        --config experiments/stage1_volumetric/config_uq.yaml --real-time

    # With robust estimator:
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_stage1_uq \\
        --config experiments/stage1_volumetric/config_uq.yaml --estimator median_mad
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from experiments.utils.experiment_output import (
    save_experiment_metadata,
    save_stage_results,
)
from growth.models.growth.hgp_hetero import HGPHeteroModel
from growth.models.growth.hgp_model import HierarchicalGPModel
from growth.models.growth.lme_hetero import LMEHeteroGrowthModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.models.growth.scalar_gp import ScalarGP
from growth.models.growth.scalar_gp_hetero import ScalarGPHetero
from growth.shared.bootstrap import BootstrapResult, paired_permutation_test
from growth.shared.lopo import LOPOEvaluator, LOPOResults
from growth.shared.metrics import (
    compute_coverage_at_levels,
    compute_crps_gaussian,
    compute_r2,
)
from growth.stages.stage1_volumetric.trajectory_loader import (
    load_uncertainty_trajectories_from_h5,
)

logger = logging.getLogger(__name__)


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
    lme_cfg = cfg.get("lme", {})
    uq_cfg = cfg.get("uncertainty", {})
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

    floor_var = uq_cfg.get("floor_variance", 1e-6)

    shared_gp_kwargs = {
        "n_restarts": gp_cfg["n_restarts"],
        "max_iter": gp_cfg["max_iter"],
        "lengthscale_bounds": tuple(gp_cfg["lengthscale_bounds"]),
        "signal_var_bounds": tuple(gp_cfg["signal_var_bounds"]),
        "noise_var_bounds": tuple(gp_cfg["noise_var_bounds"]),
        "seed": seed,
        **cov_kwargs,
    }

    models: dict[str, tuple[type, dict]] = {}

    # --- Homoscedastic models ---
    if models_cfg.get("scalar_gp", True):
        models["ScalarGP"] = (
            ScalarGP,
            {
                "kernel_type": gp_cfg["kernel"],
                "mean_function": gp_cfg.get("mean_function", "linear"),
                **shared_gp_kwargs,
            },
        )

    if models_cfg.get("lme", True):
        models["LME"] = (
            LMEGrowthModel,
            {"method": lme_cfg.get("method", "reml"), **cov_kwargs},
        )

    if models_cfg.get("hgp", True):
        models["HGP"] = (
            HierarchicalGPModel,
            {
                "kernel_type": gp_cfg["kernel"],
                "mean_function": "linear",
                **shared_gp_kwargs,
            },
        )

    if models_cfg.get("hgp_gompertz", False):
        models["HGP_Gompertz"] = (
            HierarchicalGPModel,
            {
                "kernel_type": gp_cfg["kernel"],
                "mean_function": "gompertz",
                **shared_gp_kwargs,
            },
        )

    # --- Heteroscedastic models ---
    if models_cfg.get("scalar_gp_hetero", True):
        models["ScalarGPHetero"] = (
            ScalarGPHetero,
            {
                "mean_function": gp_cfg.get("mean_function", "linear"),
                "floor_variance": floor_var,
                **shared_gp_kwargs,
            },
        )

    if models_cfg.get("lme_hetero", True):
        models["LMEHetero"] = (
            LMEHeteroGrowthModel,
            {
                "method": lme_cfg.get("method", "reml"),
                "n_restarts": lme_cfg.get("n_restarts", 5),
                "max_iter": lme_cfg.get("max_iter", 1000),
                "seed": seed,
                "floor_variance": floor_var,
                **cov_kwargs,
            },
        )

    if models_cfg.get("hgp_hetero", True):
        models["HGPHetero"] = (
            HGPHeteroModel,
            {
                "mean_function": "linear",
                "floor_variance": floor_var,
                **shared_gp_kwargs,
            },
        )

    if models_cfg.get("hgp_gompertz_hetero", False):
        models["HGP_Gompertz_Hetero"] = (
            HGPHeteroModel,
            {
                "mean_function": "gompertz",
                "floor_variance": floor_var,
                **shared_gp_kwargs,
            },
        )

    return models


def _extract_lopo_predictions(
    results: LOPOResults,
    protocol: str = "last_from_rest",
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned per-patient predictions from LOPO results.

    Returns:
        (patient_ids, y_true, y_pred, pred_var) arrays.
    """
    pids, y_true, y_pred, pred_var = [], [], [], []
    for fr in results.fold_results:
        if protocol not in fr.predictions:
            continue
        for pd in fr.predictions[protocol]:
            pids.append(fr.patient_id)
            y_true.append(pd["actual"])
            y_pred.append(pd["pred_mean"])
            pred_var.append(pd["pred_var"])
    return pids, np.array(y_true), np.array(y_pred), np.array(pred_var)


def _run_paired_comparisons(
    lopo_results: dict[str, LOPOResults],
    pairs: list[list[str]],
    n_permutations: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """Run paired comparisons between homoscedastic and heteroscedastic models.

    Returns:
        List of comparison dicts.
    """
    comparisons = []

    for pair in pairs:
        if len(pair) != 2:
            continue
        name_homo, name_hetero = pair[0], pair[1]

        if name_homo not in lopo_results or name_hetero not in lopo_results:
            logger.warning(f"Skipping pair {pair}: missing results")
            continue

        pids_h, y_true_h, y_pred_h, var_h = _extract_lopo_predictions(lopo_results[name_homo])
        pids_het, y_true_het, y_pred_het, var_het = _extract_lopo_predictions(
            lopo_results[name_hetero]
        )

        if len(y_true_h) == 0 or len(y_true_het) == 0:
            continue

        # Align by patient ID for paired comparison
        common_pids = sorted(set(pids_h) & set(pids_het))
        if len(common_pids) < 3:
            logger.warning(f"Pair {pair}: only {len(common_pids)} common patients, skipping")
            continue

        idx_h = {pid: i for i, pid in enumerate(pids_h)}
        idx_het = {pid: i for i, pid in enumerate(pids_het)}
        sel_h = np.array([idx_h[p] for p in common_pids])
        sel_het = np.array([idx_het[p] for p in common_pids])

        yt_h, yp_h, v_h = y_true_h[sel_h], y_pred_h[sel_h], var_h[sel_h]
        yt_het, yp_het, v_het = y_true_het[sel_het], y_pred_het[sel_het], var_het[sel_het]

        # Point metrics
        r2_homo = compute_r2(yt_h, yp_h)
        r2_hetero = compute_r2(yt_het, yp_het)
        delta_r2 = r2_hetero - r2_homo

        # CRPS
        sigma_h = np.sqrt(np.maximum(v_h, 0.0))
        sigma_het = np.sqrt(np.maximum(v_het, 0.0))
        crps_homo = compute_crps_gaussian(yt_h, yp_h, sigma_h)
        crps_hetero = compute_crps_gaussian(yt_het, yp_het, sigma_het)
        delta_crps = crps_hetero - crps_homo

        # Coverage at 95%
        cov_homo = compute_coverage_at_levels(yt_h, yp_h, sigma_h, (0.95,))
        cov_hetero = compute_coverage_at_levels(yt_het, yp_het, sigma_het, (0.95,))
        delta_cov_95 = cov_hetero[0.95] - cov_homo[0.95]

        # Permutation tests on per-patient absolute errors
        errors_homo = np.abs(yt_h - yp_h)
        errors_hetero = np.abs(yt_het - yp_het)
        perm_result = paired_permutation_test(
            errors_homo,
            errors_hetero,
            n_permutations=n_permutations,
            seed=seed,
        )

        comp = {
            "pair": [name_homo, name_hetero],
            "r2_homo": r2_homo,
            "r2_hetero": r2_hetero,
            "delta_r2": delta_r2,
            "crps_homo": crps_homo,
            "crps_hetero": crps_hetero,
            "delta_crps": delta_crps,
            "coverage_95_homo": cov_homo[0.95],
            "coverage_95_hetero": cov_hetero[0.95],
            "delta_coverage_95": delta_cov_95,
            "p_value_errors": perm_result.p_value,
        }
        comparisons.append(comp)

        logger.info(
            f"Pair {name_homo} → {name_hetero}: "
            f"ΔR²={delta_r2:+.4f}, ΔCRPS={delta_crps:+.4f}, "
            f"ΔCov95={delta_cov_95:+.3f}, p={perm_result.p_value:.4f}"
        )

    return comparisons


def _write_comparison_table(
    comparisons: list[dict],
    output_dir: Path,
) -> None:
    """Write comparison table as JSON and Markdown."""
    # JSON
    with open(output_dir / "comparison_homo_vs_hetero.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    # Markdown
    lines = [
        "# Homoscedastic vs Heteroscedastic Comparison",
        "",
        "| Pair | ΔR² | ΔCRPS | ΔCov_95 | p-value |",
        "|------|-----|-------|---------|---------|",
    ]
    for c in comparisons:
        pair_str = f"{c['pair'][0]} → {c['pair'][1]}"
        lines.append(
            f"| {pair_str} | {c['delta_r2']:+.4f} | "
            f"{c['delta_crps']:+.4f} | {c['delta_coverage_95']:+.3f} | "
            f"{c['p_value_errors']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ΔR² ≈ 0: uncertainty propagation preserves point accuracy",
            "- ΔCRPS < 0: heteroscedastic model is better calibrated (lower is better)",
            "- ΔCov_95 > 0: heteroscedastic intervals achieve better coverage",
            "",
        ]
    )

    with open(output_dir / "comparison_homo_vs_hetero.md", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Run Stage 1 Uncertainty-Propagated Volumetric Prediction."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Uncertainty-Propagated Volume Prediction LOPO-CV"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/stage1_volumetric/config_uq.yaml",
        help="Path to UQ config YAML",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Use days-from-baseline instead of ordinal time",
    )
    parser.add_argument(
        "--estimator",
        choices=["mean_std", "median_mad", "mask_mean"],
        default=None,
        help="Override uncertainty estimator",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])

    # CLI overrides
    if args.real_time:
        cfg["time"]["variable"] = "days_from_baseline"
        logger.info("CLI override: time_variable=days_from_baseline")
    if args.estimator:
        cfg["uncertainty"]["estimator"] = args.estimator
        logger.info(f"CLI override: estimator={args.estimator}")

    uq_cfg = cfg.get("uncertainty", {})
    time_cfg = cfg["time"]

    # =========================================================================
    # Step 1: Load trajectories
    # =========================================================================
    logger.info("=== Step 1: Load Trajectories ===")

    cov_cfg = cfg.get("covariates", {})
    covariate_features = cov_cfg.get("features", []) if cov_cfg.get("enabled", False) else []

    # All models use the SAME trajectories from the uncertainty loader so
    # homo and hetero fit the same target variable (ensemble logvol_mean),
    # making paired comparisons valid. Homo models ignore observation_variance.
    trajectories = load_uncertainty_trajectories_from_h5(
        h5_path=cfg["paths"]["mengrowth_h5"],
        time_variable=time_cfg["variable"],
        estimator=uq_cfg.get("estimator", "mean_std"),
        exclude_patients=cfg["patients"].get("exclude", []),
        min_timepoints=cfg["patients"].get("min_timepoints", 2),
        covariate_features=covariate_features,
        semantic_covariates=cfg.get("semantic_covariates", []),
        skip_all_zero_volume=cfg["patients"].get("skip_all_zero_volume", True),
        missing_date_strategy=time_cfg.get("missing_date_strategy", "mixed"),
        floor_variance=uq_cfg.get("floor_variance", 1e-6),
    )

    logger.info(f"Loaded {len(trajectories)} trajectories (ensemble logvol_mean + variance)")

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

            m = results.aggregate_metrics
            r2 = m.get("last_from_rest/r2_log", float("nan"))
            mae = m.get("last_from_rest/mae_log", float("nan"))
            cal = m.get("last_from_rest/calibration_95", float("nan"))
            crps = m.get("last_from_rest/crps", float("nan"))
            logger.info(
                f"  R2={r2:.4f}, MAE={mae:.4f}, Cal95={cal:.3f}, CRPS={crps:.4f}, "
                f"folds={len(results.fold_results)}/"
                f"{len(results.fold_results) + len(results.failed_folds)}"
            )

            ci_results = save_stage_results(
                output_dir=output_dir,
                model_name=model_name,
                lopo_results=results,
                trajectories=trajectories,
                config=cfg,
                stage_name="stage1_volumetric_uq",
                bootstrap_n=bootstrap_n,
                bootstrap_seed=bootstrap_seed,
            )
            if ci_results:
                all_bootstrap_cis[model_name] = ci_results

        except Exception as e:
            logger.error(f"  {model_name} FAILED: {e}", exc_info=True)

    # =========================================================================
    # Step 3: Paired comparisons
    # =========================================================================
    logger.info("=== Step 3: Paired Comparisons ===")

    vd_cfg = cfg.get("variance_decomposition", {})
    if vd_cfg.get("enabled", True):
        pairs = vd_cfg.get("pairs", [])
        n_perm = vd_cfg.get("n_permutations", 10000)
        comparisons = _run_paired_comparisons(
            lopo_results, pairs, n_permutations=n_perm, seed=cfg["experiment"]["seed"]
        )
        _write_comparison_table(comparisons, output_dir)

    # =========================================================================
    # Step 4: Save experiment metadata
    # =========================================================================
    logger.info("=== Step 4: Save Results ===")
    save_experiment_metadata(
        output_dir=output_dir,
        trajectories=trajectories,
        config=cfg,
        stage_name="stage1_volumetric_uq",
        all_model_results=lopo_results,
        all_bootstrap_cis=all_bootstrap_cis,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 1: UNCERTAINTY-PROPAGATED VOLUME PREDICTION — RESULTS")
    print("=" * 100)

    header = (
        f"  {'Model':<22} {'R2_log':>8} {'MAE_log':>8} {'CRPS':>8} "
        f"{'Cov_50':>7} {'Cov_80':>7} {'Cov_90':>7} {'Cov_95':>7} "
        f"{'IS_95':>8} {'CI_w':>8}"
    )
    print(f"\n{header}")
    print("  " + "-" * 96)

    for model_name in model_configs:
        if model_name not in lopo_results:
            continue
        m = lopo_results[model_name].aggregate_metrics

        print(
            f"  {model_name:<22} "
            f"{m.get('last_from_rest/r2_log', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/mae_log', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/crps', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/coverage_50', float('nan')):>7.3f} "
            f"{m.get('last_from_rest/coverage_80', float('nan')):>7.3f} "
            f"{m.get('last_from_rest/coverage_90', float('nan')):>7.3f} "
            f"{m.get('last_from_rest/coverage_95', float('nan')):>7.3f} "
            f"{m.get('last_from_rest/is_95', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/mean_ci_width_log', float('nan')):>8.4f}"
        )

    if vd_cfg.get("enabled", True) and comparisons:
        print("\n--- Paired Comparisons ---")
        print(f"  {'Pair':<40} {'ΔR²':>8} {'ΔCRPS':>8} {'ΔCov95':>8} {'p':>8}")
        print("  " + "-" * 74)
        for c in comparisons:
            pair_str = f"{c['pair'][0]} → {c['pair'][1]}"
            print(
                f"  {pair_str:<40} "
                f"{c['delta_r2']:>+8.4f} "
                f"{c['delta_crps']:>+8.4f} "
                f"{c['delta_coverage_95']:>+8.3f} "
                f"{c['p_value_errors']:>8.4f}"
            )

    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
