# experiments/utils/experiment_output.py
"""Standardized experiment output saving for all stages.

Produces a consistent output structure optimized for downstream statistical
analysis and figure generation.  Both Stage 1 and Stage 2 use this module
to ensure comparable, machine-readable outputs.

Output directory layout per model::

    {output_dir}/{model_name}/
        lopo_results.json          # Full LOPO-CV fold-level results
        raw_predictions.json       # Aligned (patient, time, actual, predicted, CI) per protocol
        per_patient_errors.json    # Per-patient signed/abs errors + CI coverage
        error_summary.json         # Aggregate error statistics
        bootstrap_cis.json         # Bootstrap CI per metric
        bootstrap_samples.json     # Full bootstrap distribution (for custom reanalysis)
        hyperparameters.json       # Per-fold fitted hyperparameters

    {output_dir}/
        model_comparison.json      # Head-to-head model comparison
        run_metadata.json          # Config snapshot, timestamps, git hash
        trajectories_used.json     # Input trajectories for reproducibility
"""

import datetime
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from growth.shared.bootstrap import BootstrapResult
from growth.shared.growth_models import PatientTrajectory
from growth.shared.lopo import LOPOResults
from growth.shared.metrics import compute_mae, compute_r2, compute_rmse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw prediction extraction
# ---------------------------------------------------------------------------


def extract_raw_predictions(lopo_results: LOPOResults) -> dict[str, list[dict]]:
    """Extract aligned prediction arrays from LOPO results per protocol.

    Returns a dict mapping protocol name to a list of prediction records,
    each containing all fields needed for scatter plots, calibration curves,
    and error analysis.

    Returns:
        ``{"last_from_rest": [...], "all_from_first": [...]}``
    """
    raw: dict[str, list[dict]] = {}

    for fr in lopo_results.fold_results:
        for protocol, preds in fr.predictions.items():
            if protocol not in raw:
                raw[protocol] = []
            for p in preds:
                raw[protocol].append(
                    {
                        "patient_id": fr.patient_id,
                        "n_timepoints": fr.n_timepoints,
                        "n_train_patients": fr.n_train_patients,
                        "n_conditioning": p.get("n_conditioning"),
                        "time": p["time"],
                        "actual": p["actual"],
                        "predicted": p["pred_mean"],
                        "variance": p["pred_var"],
                        "lower_95": p["lower_95"],
                        "upper_95": p["upper_95"],
                    }
                )

    return raw


def extract_per_patient_errors(lopo_results: LOPOResults) -> dict[str, dict]:
    """Extract per-patient prediction errors from last_from_rest protocol.

    Returns:
        Dict mapping patient_id to error dict.
    """
    errors: dict[str, dict] = {}

    for fr in lopo_results.fold_results:
        if "last_from_rest" not in fr.predictions:
            continue
        preds = fr.predictions["last_from_rest"]
        assert len(preds) == 1, (
            f"last_from_rest should return exactly 1 prediction per fold, "
            f"got {len(preds)} for patient {fr.patient_id}"
        )
        for p in preds:
            err = p["pred_mean"] - p["actual"]
            errors[fr.patient_id] = {
                "error": float(err),
                "abs_error": float(abs(err)),
                "actual": float(p["actual"]),
                "predicted": float(p["pred_mean"]),
                "variance": float(p["pred_var"]),
                "lower_95": float(p["lower_95"]),
                "upper_95": float(p["upper_95"]),
                "within_95_ci": bool(p["lower_95"] <= p["actual"] <= p["upper_95"]),
                "ci_width": float(p["upper_95"] - p["lower_95"]),
                "n_conditioning": p.get("n_conditioning"),
                "n_timepoints": fr.n_timepoints,
            }

    return errors


def summarize_errors(errors: dict[str, dict]) -> dict:
    """Compute summary statistics from per-patient errors."""
    if not errors:
        return {}

    abs_errors = [e["abs_error"] for e in errors.values()]
    signed_errors = [e["error"] for e in errors.values()]
    ci_widths = [e["ci_width"] for e in errors.values()]
    within = [e["within_95_ci"] for e in errors.values()]

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
        "calibration_95": float(np.mean(within)),
    }


# ---------------------------------------------------------------------------
# Bootstrap with distribution
# ---------------------------------------------------------------------------


def compute_bootstrap_cis_with_distribution(
    lopo_results: LOPOResults,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[dict[str, BootstrapResult], dict[str, list[float]]]:
    """Compute bootstrap CIs and save the full bootstrap distribution.

    Returns:
        (ci_dict, distribution_dict) where distribution_dict maps metric
        name to the list of bootstrap sample values.
    """
    actuals: list[float] = []
    preds: list[float] = []

    for fr in lopo_results.fold_results:
        if "last_from_rest" not in fr.predictions:
            continue
        for p in fr.predictions["last_from_rest"]:
            actuals.append(p["actual"])
            preds.append(p["pred_mean"])

    if len(actuals) < 3:
        logger.warning(
            f"Only {len(actuals)} predictions available (need >= 3 for bootstrap CIs). "
            f"Returning empty CIs."
        )
        return {}, {}

    y_true = np.array(actuals)
    y_pred = np.array(preds)

    ci_results: dict[str, BootstrapResult] = {}
    distributions: dict[str, list[float]] = {}

    rng = np.random.default_rng(seed)
    n = len(y_true)

    for metric_name, metric_fn in [
        ("r2_log", compute_r2),
        ("mae_log", compute_mae),
        ("rmse_log", compute_rmse),
    ]:
        point_estimate = metric_fn(y_true, y_pred)
        boot_values = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_values.append(float(metric_fn(y_true[idx], y_pred[idx])))

        alpha = 1 - confidence_level
        ci_lower = float(np.percentile(boot_values, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))

        ci_results[metric_name] = BootstrapResult(
            estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        )
        distributions[metric_name] = boot_values

    return ci_results, distributions


# ---------------------------------------------------------------------------
# Hyperparameter extraction
# ---------------------------------------------------------------------------


def extract_per_fold_hyperparameters(lopo_results: LOPOResults) -> list[dict]:
    """Extract per-fold fitted hyperparameters for stability analysis.

    Returns:
        List of dicts, one per fold, with patient_id and hyperparameters.
    """
    hypers: list[dict] = []
    for fr in lopo_results.fold_results:
        hypers.append(
            {
                "patient_id": fr.patient_id,
                "n_train_patients": fr.n_train_patients,
                "n_train_observations": fr.n_train_observations,
                "fit_time_s": fr.fit_time_s,
                "log_marginal_likelihood": fr.fit_result.log_marginal_likelihood,
                "hyperparameters": fr.fit_result.hyperparameters,
            }
        )
    return hypers


# ---------------------------------------------------------------------------
# Trajectory serialization
# ---------------------------------------------------------------------------


def serialize_trajectories(trajectories: list[PatientTrajectory]) -> list[dict]:
    """Serialize trajectories for reproducibility."""
    return [
        {
            "patient_id": t.patient_id,
            "n_timepoints": t.n_timepoints,
            "times": t.times.tolist(),
            "observations": t.observations[:, 0].tolist(),
            "covariates": t.covariates,
        }
        for t in trajectories
    ]


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def build_run_metadata(config: dict, stage_name: str) -> dict:
    """Build run metadata with config snapshot, timestamp, git hash, and environment."""
    import sys

    meta: dict = {
        "stage": stage_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "python_version": sys.version,
    }

    # Key package versions (best-effort)
    for pkg in ("numpy", "scipy", "GPy", "statsmodels", "torch", "monai"):
        try:
            mod = __import__(pkg)
            meta.setdefault("package_versions", {})[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # Git hash (best-effort)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            meta["git_hash"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return meta


# ---------------------------------------------------------------------------
# Main save function
# ---------------------------------------------------------------------------


def save_stage_results(
    output_dir: Path,
    model_name: str,
    lopo_results: LOPOResults,
    trajectories: list[PatientTrajectory],
    config: dict,
    stage_name: str,
    bootstrap_n: int = 2000,
    bootstrap_seed: int = 42,
    extra_data: dict | None = None,
) -> dict[str, BootstrapResult] | None:
    """Save comprehensive results for one model.

    Creates the standard output directory structure with all raw data
    needed for downstream statistical analysis and figure generation.

    Args:
        output_dir: Root output directory.
        model_name: Name for this model's subdirectory.
        lopo_results: LOPO-CV results.
        trajectories: Input trajectories used.
        config: Configuration dict (snapshotted).
        stage_name: E.g. ``"stage1_volumetric"`` or ``"stage2_severity"``.
        bootstrap_n: Number of bootstrap samples.
        bootstrap_seed: Seed for bootstrap.
        extra_data: Additional stage-specific data to save (e.g., severity values).
    """
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full LOPO results
    with open(model_dir / "lopo_results.json", "w") as f:
        json.dump(lopo_results.to_dict(), f, indent=2)

    # 2. Raw predictions (aligned arrays for plotting)
    raw_preds = extract_raw_predictions(lopo_results)
    with open(model_dir / "raw_predictions.json", "w") as f:
        json.dump(raw_preds, f, indent=2)

    # 3. Per-patient errors
    errors = extract_per_patient_errors(lopo_results)
    with open(model_dir / "per_patient_errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    # 4. Error summary
    summary = summarize_errors(errors)
    with open(model_dir / "error_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 5. Bootstrap CIs + full distribution
    ci_results, boot_dist = compute_bootstrap_cis_with_distribution(
        lopo_results,
        n_bootstrap=bootstrap_n,
        seed=bootstrap_seed,
    )
    if ci_results:
        ci_data = {
            k: {
                "estimate": br.estimate,
                "ci_lower": br.ci_lower,
                "ci_upper": br.ci_upper,
                "confidence_level": br.confidence_level,
                "n_bootstrap": br.n_bootstrap,
            }
            for k, br in ci_results.items()
        }
        with open(model_dir / "bootstrap_cis.json", "w") as f:
            json.dump(ci_data, f, indent=2)

        with open(model_dir / "bootstrap_samples.json", "w") as f:
            json.dump(boot_dist, f, indent=2)

    # 6. Per-fold hyperparameters
    hypers = extract_per_fold_hyperparameters(lopo_results)
    with open(model_dir / "hyperparameters.json", "w") as f:
        json.dump(hypers, f, indent=2)

    # 7. Extra stage-specific data
    if extra_data:
        with open(model_dir / "stage_specific.json", "w") as f:
            json.dump(extra_data, f, indent=2, default=str)

    logger.info(f"Saved {model_name} results to {model_dir}")

    return ci_results


def save_experiment_metadata(
    output_dir: Path,
    trajectories: list[PatientTrajectory],
    config: dict,
    stage_name: str,
    all_model_results: dict[str, LOPOResults],
    all_bootstrap_cis: dict[str, dict[str, BootstrapResult]],
) -> None:
    """Save experiment-level metadata and model comparison.

    Call this ONCE after all models have been evaluated.

    Args:
        output_dir: Root output directory.
        trajectories: Input trajectories.
        config: Configuration dict.
        stage_name: Stage identifier.
        all_model_results: Dict mapping model_name to LOPOResults.
        all_bootstrap_cis: Dict mapping model_name to bootstrap CIs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run metadata
    meta = build_run_metadata(config, stage_name)
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Trajectories used
    traj_data = serialize_trajectories(trajectories)
    with open(output_dir / "trajectories_used.json", "w") as f:
        json.dump(traj_data, f, indent=2)

    # Model comparison
    comparison: dict = {"models": {}, "ranking": []}
    for model_name, results in all_model_results.items():
        entry: dict = {
            "model_name": results.model_name,
            "n_folds": len(results.fold_results),
            "n_failed": len(results.failed_folds),
            "failed_folds": results.failed_folds,
        }
        for metric, val in sorted(results.aggregate_metrics.items()):
            entry[metric] = val

        if model_name in all_bootstrap_cis:
            entry["bootstrap_ci"] = {
                k: {"estimate": br.estimate, "ci_lower": br.ci_lower, "ci_upper": br.ci_upper}
                for k, br in all_bootstrap_cis[model_name].items()
            }

        comparison["models"][model_name] = entry

    # Rank by R2
    comparison["ranking"] = sorted(
        [
            {"model": k, "r2_log": v.get("last_from_rest/r2_log", float("-inf"))}
            for k, v in comparison["models"].items()
        ],
        key=lambda x: x["r2_log"],
        reverse=True,
    )

    with open(output_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Experiment metadata saved to {output_dir}")
