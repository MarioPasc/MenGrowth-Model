# experiments/variance_decomposition/run_decomposition.py
"""Variance decomposition: cross-stage ΔR² analysis.

Runs all models under identical LOPO-CV folds and computes the marginal
contribution of each model transition. Central analytical contribution.

Usage::

    ~/.conda/envs/growth/bin/python -m experiments.variance_decomposition.run_decomposition \\
        --config experiments/variance_decomposition/config.yaml \\
        --trajectories /path/to/trajectories.json \\
        --output-dir /path/to/results
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from growth.evaluation.variance_decomposition import VarianceDecomposition
from growth.shared import LOPOEvaluator, PatientTrajectory, load_trajectories

logger = logging.getLogger(__name__)


def _population_mean_predictions(patients: list[PatientTrajectory]) -> np.ndarray:
    """Baseline M₀: predict the population mean for every patient.

    Uses LOPO-CV compatible mean: for each held-out patient, the mean
    is computed from the remaining patients' last observations.

    Args:
        patients: All patient trajectories.

    Returns:
        Per-patient predictions, shape ``[N]``.
    """
    preds = np.empty(len(patients))
    for i in range(len(patients)):
        train_obs = [float(p.observations[-1, 0]) for j, p in enumerate(patients) if j != i]
        preds[i] = float(np.mean(train_obs))
    return preds


def main() -> None:
    """Run variance decomposition across all configured models."""
    parser = argparse.ArgumentParser(description="Variance Decomposition")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trajectories", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    cfg = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patients = load_trajectories(args.trajectories)
    logger.info(f"Loaded {len(patients)} patients for variance decomposition")

    # Collect per-patient predictions from each model
    y_true = np.array([float(p.observations[-1, 0]) for p in patients])
    model_predictions: dict[str, np.ndarray] = {}
    model_order: list[str] = []

    for model_cfg in cfg.models:
        name = model_cfg.name
        model_order.append(name)
        logger.info(f"Running model: {name} (type={model_cfg.type})")

        if model_cfg.type == "population_mean":
            model_predictions[name] = _population_mean_predictions(patients)
        else:
            # For GP/LME/Severity models, run LOPO-CV and extract predictions
            model_predictions[name] = _run_lopo_for_model(model_cfg, patients)

    # Run variance decomposition
    vd = VarianceDecomposition(
        n_permutations=cfg.statistics.n_permutations,
        n_bootstrap=cfg.statistics.n_bootstrap,
        alpha=cfg.statistics.alpha,
        seed=cfg.seed,
    )
    result = vd.decompose(y_true, model_predictions, model_order)

    # Save results
    result_path = output_dir / "variance_decomposition.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Print summary table
    logger.info("=" * 60)
    logger.info("VARIANCE DECOMPOSITION RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'R²':>8} {'ΔR²':>8} {'p-value':>10} {'Sig':>5}")
    logger.info("-" * 60)
    for i, name in enumerate(result.models):
        if i == 0:
            logger.info(f"{name:<20} {result.r2_values[i]:>8.4f} {'—':>8} {'—':>10} {'—':>5}")
        else:
            t = result.transitions[i - 1]
            sig = (
                "***"
                if t.p_value < 0.001
                else "**"
                if t.p_value < 0.01
                else "*"
                if t.p_value < 0.05
                else ""
            )
            logger.info(
                f"{name:<20} {result.r2_values[i]:>8.4f} {t.delta_r2:>+8.4f} "
                f"{t.p_value:>10.4f} {sig:>5}"
            )
    logger.info("=" * 60)

    logger.info(f"Results saved to {result_path}")


def _run_lopo_for_model(
    model_cfg: object,
    patients: list[PatientTrajectory],
) -> np.ndarray:
    """Run LOPO-CV for a single model and extract per-patient predictions.

    Args:
        model_cfg: Model configuration from OmegaConf.
        patients: All patient trajectories.

    Returns:
        Per-patient predictions, shape ``[N]``.
    """
    from growth.models.growth.hgp_model import HierarchicalGPModel
    from growth.models.growth.lme_model import LMEGrowthModel
    from growth.models.growth.scalar_gp import ScalarGP
    from growth.stages.stage2_severity import SeverityModel

    model_type = model_cfg.type
    model_kwargs: dict = {}

    if model_type == "scalar_gp":
        model_class = ScalarGP
        model_kwargs = {
            "kernel_type": getattr(model_cfg, "kernel", "matern52"),
            "mean_function": getattr(model_cfg, "mean_function", "linear"),
        }
    elif model_type == "lme":
        model_class = LMEGrowthModel
    elif model_type == "hgp":
        model_class = HierarchicalGPModel
        model_kwargs = {"kernel_type": getattr(model_cfg, "kernel", "matern52")}
    elif model_type == "severity":
        model_class = SeverityModel
        model_kwargs = {
            "growth_function": getattr(model_cfg, "growth_function", "gompertz_reduced"),
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
    results = evaluator.evaluate(model_class, patients, **model_kwargs)

    # Extract per-patient predictions
    preds = np.empty(len(patients))
    patient_id_to_idx = {p.patient_id: i for i, p in enumerate(patients)}

    for fr in results.fold_results:
        idx = patient_id_to_idx[fr.patient_id]
        if "last_from_rest" in fr.predictions and fr.predictions["last_from_rest"]:
            preds[idx] = fr.predictions["last_from_rest"][0]["pred_mean"]
        else:
            preds[idx] = float("nan")

    # Fill failed folds with population mean (may bias ΔR² upward)
    nan_mask = np.isnan(preds)
    if np.any(nan_mask):
        n_nan = int(np.sum(nan_mask))
        failed_ids = [patients[i].patient_id for i in np.where(nan_mask)[0]]
        logger.warning(
            f"Imputing {n_nan} failed folds with population mean "
            f"(patients: {failed_ids}). This may bias ΔR² upward."
        )
        mean_pred = np.nanmean(preds)
        preds = np.where(nan_mask, mean_pred, preds)

    return preds


if __name__ == "__main__":
    main()
