"""Task runner for the conformal calibration experiment.

One task = one (base_model_key, seed) pair. The runner:

1. Builds the base model instance from the key (lme_homo / lme_hetero / ensemble_bma).
2. Instantiates :class:`~growth.evaluation.conformal_lopo.ConformalLOPOEvaluator` with the
   four calibration layers requested in config.
3. Runs the nested LOPO evaluation → :class:`ConformalLOPOResults`.
4. Computes per-layer IS@95, coverage@95, mean width, R²_log, and CRPS.
5. Stratifies by σ²_v tertile (pinned to the empirical distribution).
6. Writes ``conformal_lopo_results.json``, ``marginal_metrics.json``,
   ``tertile_metrics.json``, ``per_patient_metrics.json`` under
   ``{output_root}/runs/{base_model}/seed_{NNN}/``.

``per_patient_metrics.json`` is the long-form per-patient, per-layer record
(one row per patient × calibration layer): point prediction, interval bounds,
width, coverage flag, the Winkler interval score, the per-target σ²_v and its
cohort-pinned tertile. It is the substrate for the per-patient interval figure
(prediction interval + IS@95 per held-out patient) and the per-patient paired
comparison table.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from growth.evaluation.conformal_lopo import (
    ConformalLOPOEvaluator,
    ConformalLOPOResults,
)
from growth.models.growth.ensemble_lme import EnsembleLMEGrowthModel
from growth.models.growth.lme_hetero import LMEHeteroGrowthModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.shared.growth_models import GrowthModel, PatientTrajectory
from growth.shared.metrics import (
    compute_coverage_at_levels,
    compute_crps_gaussian,
    compute_interval_score,
    compute_r2,
)

from .cohort import Cohort

logger = logging.getLogger(__name__)

_BASE_MODEL_KEYS = ("lme_homo", "lme_hetero", "ensemble_bma")
_ALL_LAYERS = ("parametric", "jackknife_plus", "cqr_norm", "cqr_proper")


@dataclass(frozen=True)
class TaskSpec:
    """Identifies a single (base_model, seed) task in the experiment manifest."""

    base_model: str  # "lme_homo" | "lme_hetero" | "ensemble_bma"
    seed: int

    @property
    def model_dirname(self) -> str:
        return self.base_model

    @property
    def seed_dirname(self) -> str:
        return f"seed_{self.seed:03d}"


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_model(base_model_key: str, cfg: dict, seed: int) -> GrowthModel:
    """Instantiate a base growth model from its config key.

    Args:
        base_model_key: One of ``"lme_homo"``, ``"lme_hetero"``,
            ``"ensemble_bma"``.
        cfg: Full experiment config dict.
        seed: Seed for the LMEHetero optimiser restarts.

    Returns:
        Instantiated (unfitted) :class:`GrowthModel`.

    Raises:
        ValueError: If ``base_model_key`` is not recognised.
    """
    eval_cfg = cfg.get("evaluation", {})
    n_restarts = int(eval_cfg.get("n_restarts", 5))
    uq_cfg = cfg.get("uncertainty", {})
    floor_variance = float(uq_cfg.get("floor_variance", 1e-6))
    ens_cfg = cfg.get("ensemble", {})
    n_members: int | None = ens_cfg.get("n_members", None)

    if base_model_key == "lme_homo":
        return LMEGrowthModel(method="reml")
    if base_model_key == "lme_hetero":
        return LMEHeteroGrowthModel(
            floor_variance=floor_variance,
            n_restarts=n_restarts,
        )
    if base_model_key == "ensemble_bma":
        return EnsembleLMEGrowthModel(
            method="reml",
            n_members=n_members,
        )
    raise ValueError(
        f"Unknown base_model key '{base_model_key}'. Must be one of {_BASE_MODEL_KEYS}"
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _layer_metrics(
    results: ConformalLOPOResults,
    layer: str,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute calibration metrics for one layer over all fold results."""
    if not results.fold_results:
        return {}

    actual = np.array([fr.actual for fr in results.fold_results])
    pred = np.array([fr.parametric_mean for fr in results.fold_results])
    pred_var = np.array([fr.parametric_var for fr in results.fold_results])
    lower = np.array([fr.intervals[layer][0] for fr in results.fold_results])
    upper = np.array([fr.intervals[layer][1] for fr in results.fold_results])

    n = len(actual)
    sigma = np.sqrt(np.maximum(pred_var, 1e-15))
    covered = int(np.sum((actual >= lower) & (actual <= upper)))
    cov_dict = compute_coverage_at_levels(actual, pred, sigma)

    from growth.shared.conformal import beta_binomial_coverage_ci

    ci_lo, ci_hi = beta_binomial_coverage_ci(covered, n, confidence=0.95)

    return {
        "n": n,
        "r2_log": float(compute_r2(actual, pred)),
        "is_95": float(compute_interval_score(actual, lower, upper, alpha=alpha)),
        "coverage_95": float(covered / n),
        "coverage_95_ci_low": float(ci_lo),
        "coverage_95_ci_high": float(ci_hi),
        "mean_width": float(np.mean(upper - lower)),
        "crps": float(compute_crps_gaussian(actual, pred, sigma)),
        "cov_50": float(cov_dict.get(0.50, float("nan"))),
        "cov_80": float(cov_dict.get(0.80, float("nan"))),
        "cov_90": float(cov_dict.get(0.90, float("nan"))),
        "cov_95_base": float(cov_dict.get(0.95, float("nan"))),
    }


def _tertile_layer_metrics(
    results: ConformalLOPOResults,
    layer: str,
    cuts: tuple[float, float],
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Stratify layer metrics by σ²_v tertile."""
    q33, q66 = cuts
    sv2 = np.array([fr.sigma_v_sq_target for fr in results.fold_results])
    actual = np.array([fr.actual for fr in results.fold_results])
    pred = np.array([fr.parametric_mean for fr in results.fold_results])
    pred_var = np.array([fr.parametric_var for fr in results.fold_results])
    lower = np.array([fr.intervals[layer][0] for fr in results.fold_results])
    upper = np.array([fr.intervals[layer][1] for fr in results.fold_results])

    out: dict[str, dict[str, float]] = {}
    masks = {
        "low": sv2 <= q33,
        "mid": (sv2 > q33) & (sv2 <= q66),
        "high": sv2 > q66,
    }
    for name, mask in masks.items():
        if mask.sum() < 2:
            out[name] = {"n": int(mask.sum())}
            continue
        a_m = actual[mask]
        p_m = pred[mask]
        pv_m = pred_var[mask]
        lo_m = lower[mask]
        hi_m = upper[mask]
        sigma_m = np.sqrt(np.maximum(pv_m, 1e-15))
        covered_m = int(np.sum((a_m >= lo_m) & (a_m <= hi_m)))
        n_m = int(mask.sum())
        out[name] = {
            "n": n_m,
            "r2_log": float(compute_r2(a_m, p_m)),
            "is_95": float(compute_interval_score(a_m, lo_m, hi_m, alpha=alpha)),
            "coverage_95": float(covered_m / n_m),
            "mean_width": float(np.mean(hi_m - lo_m)),
            "crps": float(compute_crps_gaussian(a_m, p_m, sigma_m)),
            "sigma_v_sq_mean": float(np.mean(sv2[mask])),
        }
    return out


def _empirical_tertile_cuts(sigma_v_sq_flat: np.ndarray) -> tuple[float, float]:
    """Tertile cuts pinned to the full cohort empirical σ²_v distribution."""
    sv = np.asarray(sigma_v_sq_flat, dtype=np.float64)
    return float(np.quantile(sv, 1.0 / 3.0)), float(np.quantile(sv, 2.0 / 3.0))


def _assign_tertile(sigma_v_sq: float, cuts: tuple[float, float]) -> str:
    """Map a scalar σ²_v to its cohort-pinned tertile label.

    Args:
        sigma_v_sq: Per-target measurement variance.
        cuts: ``(q33, q66)`` cohort-empirical tertile edges.

    Returns:
        One of ``"low"``, ``"mid"``, ``"high"``, or ``"nan"`` if ``sigma_v_sq``
        is not finite.
    """
    q33, q66 = cuts
    if not np.isfinite(sigma_v_sq):
        return "nan"
    if sigma_v_sq <= q33:
        return "low"
    if sigma_v_sq <= q66:
        return "mid"
    return "high"


def _build_per_patient_rows(
    results: ConformalLOPOResults,
    spec: TaskSpec,
    cuts: tuple[float, float],
) -> list[dict[str, Any]]:
    """Long-form per-patient, per-layer records for one (base_model, seed) task.

    Wraps :meth:`ConformalLOPOResults.per_patient_table` (which already carries
    the per-patient interval, width, coverage flag and Winkler interval score)
    and tags every row with the task identity and the cohort-pinned σ²_v
    tertile, so the analysis phase can build the per-patient comparison table
    and the per-patient interval figure without re-deriving anything.

    Args:
        results: Completed nested-LOPO results for one task.
        spec: Task identifier (base model + seed).
        cuts: ``(q33, q66)`` cohort-empirical σ²_v tertile edges.

    Returns:
        One dict per (patient, calibration layer).
    """
    rows: list[dict[str, Any]] = []
    for row in results.per_patient_table():
        rows.append(
            {
                "base_model": spec.base_model,
                "seed": spec.seed,
                "model_name": results.model_name,
                "tertile": _assign_tertile(float(row["sigma_v_sq_target"]), cuts),
                **row,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------


def run_task(
    spec: TaskSpec,
    cohort: Cohort,
    cfg: dict,
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run one (base_model, seed) conformal LOPO task.

    Args:
        spec: Task identifier.
        cohort: Loaded cohort with ensemble and variance fields.
        cfg: Full experiment config dict.
        output_root: Root output directory; results written under
            ``{output_root}/runs/{base_model}/seed_{NNN}/``.
        force: Re-run even if cached results exist.

    Returns:
        Dict with output paths and per-layer marginal metrics.
    """
    task_dir = output_root / "runs" / spec.model_dirname / spec.seed_dirname
    task_dir.mkdir(parents=True, exist_ok=True)

    lopo_path = task_dir / "conformal_lopo_results.json"
    marginal_path = task_dir / "marginal_metrics.json"
    tertile_path = task_dir / "tertile_metrics.json"
    per_patient_path = task_dir / "per_patient_metrics.json"

    if (
        not force
        and lopo_path.exists()
        and marginal_path.exists()
        and tertile_path.exists()
        and per_patient_path.exists()
    ):
        logger.info("CACHED task %s/%s", spec.model_dirname, spec.seed_dirname)
        with open(marginal_path) as fh:
            marginal = json.load(fh)
        return {
            "task_dir": str(task_dir),
            "marginal_metrics": marginal,
            "cached": True,
        }

    conf_cfg = cfg.get("conformal", {})
    alpha = float(conf_cfg.get("alpha", 0.05))
    layers_list = list(conf_cfg.get("layers", _ALL_LAYERS))
    jp_score = conf_cfg.get("jackknife_plus", {}).get("score", "signed")
    cqr_frac = float(conf_cfg.get("cqr_proper", {}).get("calib_fraction", 0.33))

    # Select trajectories based on base model needs.
    trajectories = _select_trajectories(spec.base_model, cohort)

    evaluator = ConformalLOPOEvaluator(
        alpha=alpha,
        layers=tuple(layers_list),
        jackknife_score=jp_score,
        cqr_calib_fraction=cqr_frac,
        seed=spec.seed,
    )

    model_class = _model_class(spec.base_model)
    model_kwargs = _model_kwargs(spec.base_model, cfg, spec.seed)

    logger.info(
        "Running task %s/seed_%03d (model=%s, layers=%s)",
        spec.base_model,
        spec.seed,
        spec.base_model,
        layers_list,
    )
    results: ConformalLOPOResults = evaluator.evaluate(model_class, trajectories, **model_kwargs)

    # Persist full per-patient JSON.
    with open(lopo_path, "w") as fh:
        json.dump(results.to_dict(), fh, indent=2)

    # Marginal and tertile metrics per layer.
    cuts = _empirical_tertile_cuts(cohort.sigma_v_sq_flat)
    marginal: dict[str, Any] = {}
    tertile: dict[str, Any] = {}

    for layer in layers_list:
        if results.fold_results and layer in results.fold_results[0].intervals:
            marginal[layer] = _layer_metrics(results, layer, alpha=alpha)
            tertile[layer] = _tertile_layer_metrics(results, layer, cuts, alpha=alpha)

    marginal["r2_log"] = (
        float(
            compute_r2(
                np.array([fr.actual for fr in results.fold_results]),
                np.array([fr.parametric_mean for fr in results.fold_results]),
            )
        )
        if results.fold_results
        else float("nan")
    )

    with open(marginal_path, "w") as fh:
        json.dump(marginal, fh, indent=2)
    with open(tertile_path, "w") as fh:
        json.dump({"cuts_q33_q66": list(cuts), "strata_by_layer": tertile}, fh, indent=2)

    # Long-form per-patient / per-layer records (intervals + IS@95 + σ²_v +
    # tertile). The headline per-patient interval figure consumes this.
    per_patient_rows = _build_per_patient_rows(results, spec, cuts)
    with open(per_patient_path, "w") as fh:
        json.dump(
            {
                "base_model": spec.base_model,
                "seed": spec.seed,
                "alpha": alpha,
                "cuts_q33_q66": list(cuts),
                "n_patients": len(results.fold_results),
                "failed_folds": results.failed_folds,
                "rows": per_patient_rows,
            },
            fh,
            indent=2,
        )

    # Log headline metrics.
    for layer in ("parametric", "jackknife_plus"):
        if layer in marginal:
            m = marginal[layer]
            logger.info(
                "  %s/%s [%s]: IS@95=%.3f cov95=%.3f width=%.3f R²=%.3f",
                spec.base_model,
                spec.seed_dirname,
                layer,
                m.get("is_95", float("nan")),
                m.get("coverage_95", float("nan")),
                m.get("mean_width", float("nan")),
                marginal.get("r2_log", float("nan")),
            )

    return {
        "task_dir": str(task_dir),
        "marginal_metrics": marginal,
        "cached": False,
    }


def _select_trajectories(
    base_model_key: str,
    cohort: Cohort,
) -> list[PatientTrajectory]:
    """Return the trajectory list appropriate for the base model.

    All three models share the same trajectories from the cohort — the base
    model classes only access ``observation_ensemble`` (EnsembleLME) or
    ``observation_variance`` (LMEHetero) as needed; LMEGrowthModel ignores both.
    """
    return cohort.trajectories


def _model_class(base_model_key: str) -> type[GrowthModel]:
    return {
        "lme_homo": LMEGrowthModel,
        "lme_hetero": LMEHeteroGrowthModel,
        "ensemble_bma": EnsembleLMEGrowthModel,
    }[base_model_key]


def _model_kwargs(base_model_key: str, cfg: dict, seed: int) -> dict[str, Any]:
    """Build keyword arguments for the model class constructor."""
    eval_cfg = cfg.get("evaluation", {})
    n_restarts = int(eval_cfg.get("n_restarts", 5))
    uq_cfg = cfg.get("uncertainty", {})
    floor_variance = float(uq_cfg.get("floor_variance", 1e-6))
    ens_cfg = cfg.get("ensemble", {})
    n_members: int | None = ens_cfg.get("n_members", None)

    if base_model_key == "lme_homo":
        return {"method": "reml"}
    if base_model_key == "lme_hetero":
        return {
            "floor_variance": floor_variance,
            "n_restarts": n_restarts,
        }
    if base_model_key == "ensemble_bma":
        return {
            "method": "reml",
            "n_members": n_members,
        }
    raise ValueError(f"Unknown base_model key: {base_model_key}")


# ---------------------------------------------------------------------------
# Task manifest helpers
# ---------------------------------------------------------------------------


def iter_task_specs(cfg: dict) -> list[TaskSpec]:
    """Yield all (base_model, seed) task specs from config.

    Args:
        cfg: Full experiment config dict.

    Returns:
        List of :class:`TaskSpec` in deterministic order.
    """
    models_cfg = cfg.get("models", {})
    eval_cfg = cfg.get("evaluation", {})
    n_seeds = int(eval_cfg.get("n_seeds", 20))

    specs: list[TaskSpec] = []
    for key in _BASE_MODEL_KEYS:
        if models_cfg.get(key, True):
            for seed in range(n_seeds):
                specs.append(TaskSpec(base_model=key, seed=seed))
    return specs
