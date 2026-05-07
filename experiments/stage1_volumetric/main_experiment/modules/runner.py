"""Cell-level runner: one (family, level, seed) → LMEHetero LOPO + metrics.

This module is the orchestration layer between the sweep generators in
``sigma_v_generators`` and the existing LOPO evaluator + LMEHetero model class.
It does not re-implement the model, the LOPO loop, or the metrics — those live
in ``growth.shared.lopo``, ``growth.models.growth.lme_hetero``, and
``growth.shared.metrics``.

Layout per cell::

    {runs_dir}/{family}_{level}/seed_{NNN}/
        sigma_v_sq_injected.npy
        lopo_results.json
        marginal_metrics.json
        tertile_metrics.json
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stage1_volumetric.synthetic_uq.run_synthetic_uq import inject_sigma_v
from growth.models.growth.lme_hetero import LMEHeteroGrowthModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.shared.lopo import LOPOEvaluator, LOPOResults
from growth.shared.metrics import (
    compute_coverage_at_levels,
    compute_crps_gaussian,
    compute_dawid_sebastiani,
    compute_interval_score,
    compute_log_score,
    compute_pit,
    compute_r2,
)

from .cohort import Cohort
from .sigma_v_generators import (
    build_tau_grid,
    compute_tau_endpoints,
    sample_beta_alpha,
    sample_shifted_empirical,
)

logger = logging.getLogger(__name__)

PROTOCOL = "last_from_rest"


@dataclass(frozen=True)
class CellSpec:
    """Identifies a single (family, level, seed) cell in the sweep."""

    family: str  # "empirical_shift" | "beta_alpha"
    level: str  # e.g. "tau_+0.000", "alpha_0.000"
    level_value: float  # τ (or α) for the cell
    seed: int

    @property
    def cell_dirname(self) -> str:
        return f"{self.family}_{self.level}"

    @property
    def seed_dirname(self) -> str:
        return f"seed_{self.seed:03d}"


# ---------------------------------------------------------------------------
# σ²_v sampling dispatch
# ---------------------------------------------------------------------------


def generate_sigma_v(
    spec: CellSpec,
    cohort: Cohort,
    cfg: dict,
) -> np.ndarray:
    """Dispatch to the appropriate sampler given a cell spec."""
    rng = np.random.default_rng(spec.seed)
    n = cohort.n_scans_total
    floor = cfg.get("uncertainty", {}).get("floor_variance", 1e-3)

    if spec.family == "empirical_shift":
        primary = cfg.get("sweep", {}).get("primary", {})
        ceil = primary.get("saturation", {}).get("sigma_v_sq_ceil")
        log_emp = np.log(np.maximum(cohort.empirical_sigma_v_sq_flat, 1e-15))
        return sample_shifted_empirical(
            tau=spec.level_value,
            n=n,
            log_empirical_sigma_v_sq=log_emp,
            rng=rng,
            sigma_v_sq_floor=floor,
            sigma_v_sq_ceil=ceil,
        )

    if spec.family == "beta_alpha":
        ab = cfg["sweep"]["ablation"]
        return sample_beta_alpha(
            alpha=spec.level_value,
            n=n,
            rng=rng,
            sigma_v_sq_max=ab.get("sigma_v_sq_max", 1.5),
            steepness=ab.get("steepness", 9.0),
        )

    raise ValueError(f"Unknown sweep family: {spec.family}")


# ---------------------------------------------------------------------------
# Metric helpers (mirror synthetic_uq.run_synthetic_uq pattern)
# ---------------------------------------------------------------------------


def _flatten_predictions(results: LOPOResults) -> tuple[np.ndarray, ...]:
    pids: list[str] = []
    pm, pa, pl, pu, pv, sv2 = [], [], [], [], [], []
    for fr in results.fold_results:
        if PROTOCOL not in fr.predictions:
            continue
        for pred in fr.predictions[PROTOCOL]:
            pids.append(fr.patient_id)
            pm.append(pred["pred_mean"])
            pa.append(pred["actual"])
            pl.append(pred["lower_95"])
            pu.append(pred["upper_95"])
            pv.append(pred["pred_var"])
            sv2.append(pred.get("sigma_v_sq_target", float("nan")))
    return (
        np.asarray(pids),
        np.asarray(pm),
        np.asarray(pa),
        np.asarray(pl),
        np.asarray(pu),
        np.asarray(pv),
        np.asarray(sv2),
    )


def _calibration_battery(
    pred: np.ndarray,
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    var: np.ndarray,
) -> dict[str, float]:
    sigma_sq = np.maximum(var, 1e-15)
    sigma = np.sqrt(sigma_sq)
    cov = compute_coverage_at_levels(actual, pred, sigma)
    pit_vals = compute_pit(actual, pred, sigma)

    from scipy.stats import kstest

    ks_stat, ks_p = kstest(pit_vals, "uniform")

    return {
        "n": int(len(actual)),
        "r2_log": float(compute_r2(actual, pred)),
        "ci_width_mean": float(np.mean(upper - lower)),
        "cov_50": float(cov[0.50]),
        "cov_80": float(cov[0.80]),
        "cov_90": float(cov[0.90]),
        "cov_95": float(cov[0.95]),
        "crps": float(compute_crps_gaussian(actual, pred, sigma)),
        "is_95": float(compute_interval_score(actual, lower, upper, alpha=0.05)),
        "nlpd": float(compute_log_score(actual, pred, sigma_sq)),
        "dss": float(compute_dawid_sebastiani(actual, pred, sigma_sq)),
        "pred_var_mean": float(np.mean(np.maximum(var, 0.0))),
        "pit_ks_stat": float(ks_stat),
        "pit_ks_p": float(ks_p),
        "pit_values": pit_vals.tolist(),
    }


def _tertile_battery(
    pred: np.ndarray,
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    var: np.ndarray,
    sigma_v_sq: np.ndarray,
    cuts: tuple[float, float],
) -> dict[str, dict[str, float]]:
    q33, q66 = cuts
    masks = {
        "low": sigma_v_sq <= q33,
        "mid": (sigma_v_sq > q33) & (sigma_v_sq <= q66),
        "high": sigma_v_sq > q66,
    }
    out: dict[str, dict[str, float]] = {}
    for name, m in masks.items():
        if m.sum() < 2:
            out[name] = {"n": int(m.sum())}
            continue
        battery = _calibration_battery(pred[m], actual[m], lower[m], upper[m], var[m])
        battery.pop("pit_values", None)  # keep tertile JSON small
        battery["sigma_v_sq_mean"] = float(np.mean(sigma_v_sq[m]))
        out[name] = battery
    return out


def empirical_tertile_cuts(empirical_sigma_v_sq_flat: np.ndarray) -> tuple[float, float]:
    """Tertile cuts pinned to the empirical σ²_v distribution.

    Locking the cuts to the empirical distribution (and not to each cell's
    injected σ²_v) keeps the high/mid/low strata comparable across cells.
    """
    sv = np.asarray(empirical_sigma_v_sq_flat, dtype=np.float64)
    return float(np.quantile(sv, 1 / 3)), float(np.quantile(sv, 2 / 3))


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------


def _save_results_with_sigma_v(
    results: LOPOResults,
    sigma_v_sq_flat: np.ndarray,
    cohort: Cohort,
    output_path: Path,
) -> None:
    """Stamp ``sigma_v_sq_target`` onto each fold prediction and serialize.

    The LOPO evaluator already records the target σ²_v on the held-out scan,
    but only for trajectories whose ``observation_variance`` is set at fit
    time. Stamping again here is defensive and ensures downstream tertile
    bootstrap can read the exact injected value even if the model class did
    not propagate it.
    """
    pid_to_n = dict(zip(cohort.patient_ids, cohort.n_timepoints_per_patient, strict=True))
    pid_to_offset: dict[str, int] = {}
    cursor = 0
    for pid in cohort.patient_ids:
        pid_to_offset[pid] = cursor
        cursor += pid_to_n[pid]

    for fr in results.fold_results:
        offset = pid_to_offset.get(fr.patient_id)
        n = pid_to_n.get(fr.patient_id)
        if offset is None or n is None:
            continue
        target_var = float(sigma_v_sq_flat[offset + n - 1])
        for protocol_preds in fr.predictions.values():
            for pred in protocol_preds:
                pred["sigma_v_sq_target"] = target_var

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)


def run_cell(
    spec: CellSpec,
    cohort: Cohort,
    cfg: dict,
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run one LMEHetero LOPO cell with injected σ²_v.

    Args:
        spec: Cell identifier.
        cohort: Loaded cohort with empirical σ²_v.
        cfg: Full config dict.
        output_root: Base directory; the cell writes under
            ``{output_root}/runs/{family}_{level}/seed_{NNN}/``.
        force: Re-run even if cached results exist.

    Returns:
        Dict with paths and aggregate metrics (cached or freshly computed).
    """
    cell_dir = output_root / "runs" / spec.cell_dirname / spec.seed_dirname
    cell_dir.mkdir(parents=True, exist_ok=True)

    sigma_path = cell_dir / "sigma_v_sq_injected.npy"
    lopo_path = cell_dir / "lopo_results.json"
    marginal_path = cell_dir / "marginal_metrics.json"
    tertile_path = cell_dir / "tertile_metrics.json"

    # Resume support
    if not force and lopo_path.exists() and marginal_path.exists() and tertile_path.exists():
        logger.info("CACHED cell %s/%s", spec.cell_dirname, spec.seed_dirname)
        with open(marginal_path) as f:
            marginal = json.load(f)
        return {
            "cell_dir": str(cell_dir),
            "marginal_metrics": marginal,
            "cached": True,
        }

    # 1. Sample σ²_v
    sigma_v_sq = generate_sigma_v(spec, cohort, cfg)
    np.save(sigma_path, sigma_v_sq)

    # 2. Inject and run LOPO
    new_trajs, _ = inject_sigma_v(cohort.trajectories, sigma_v_sq)
    floor_var = cfg.get("uncertainty", {}).get("floor_variance", 1e-3)

    evaluator = LOPOEvaluator(prediction_protocols=[PROTOCOL])
    results = evaluator.evaluate(
        LMEHeteroGrowthModel,
        new_trajs,
        floor_variance=floor_var,
    )
    _save_results_with_sigma_v(results, sigma_v_sq, cohort, lopo_path)

    # 3. Compute metrics
    pids, pm, pa, pl, pu, pv, sv2 = _flatten_predictions(results)
    if pm.size == 0:
        raise RuntimeError(f"Cell {spec.cell_dirname}/{spec.seed_dirname} produced no predictions")

    marginal = _calibration_battery(pm, pa, pl, pu, pv)
    cuts = empirical_tertile_cuts(cohort.empirical_sigma_v_sq_flat)
    tertile = _tertile_battery(pm, pa, pl, pu, pv, sv2, cuts)

    with open(marginal_path, "w") as f:
        json.dump(marginal, f, indent=2)
    with open(tertile_path, "w") as f:
        json.dump({"cuts_q33_q66": list(cuts), "strata": tertile}, f, indent=2)

    logger.info(
        "  cell %s/%s: R²=%.3f, IS@95=%.3f, cov95=%.3f",
        spec.cell_dirname,
        spec.seed_dirname,
        marginal["r2_log"],
        marginal["is_95"],
        marginal["cov_95"],
    )

    return {
        "cell_dir": str(cell_dir),
        "marginal_metrics": marginal,
        "cached": False,
    }


# ---------------------------------------------------------------------------
# Baselines (cached once globally)
# ---------------------------------------------------------------------------


def _run_baseline(
    name: str,
    model_cls: type,
    model_kwargs: dict,
    trajectories,
    output_root: Path,
    cohort: Cohort,
    *,
    force: bool = False,
) -> dict[str, Any]:
    base_dir = output_root / name
    base_dir.mkdir(parents=True, exist_ok=True)

    lopo_path = base_dir / "lopo_results.json"
    marginal_path = base_dir / "marginal_metrics.json"
    tertile_path = base_dir / "tertile_metrics.json"

    if not force and lopo_path.exists() and marginal_path.exists():
        logger.info("CACHED baseline %s", name)
        with open(marginal_path) as f:
            marginal = json.load(f)
        return {"baseline": name, "marginal_metrics": marginal, "cached": True}

    evaluator = LOPOEvaluator(prediction_protocols=[PROTOCOL])
    results = evaluator.evaluate(model_cls, trajectories, **model_kwargs)

    # Stamp empirical σ²_v_target so tertile bootstrap pairing works.
    _save_results_with_sigma_v(results, cohort.empirical_sigma_v_sq_flat, cohort, lopo_path)

    _, pm, pa, pl, pu, pv, sv2 = _flatten_predictions(results)
    marginal = _calibration_battery(pm, pa, pl, pu, pv)
    cuts = empirical_tertile_cuts(cohort.empirical_sigma_v_sq_flat)
    tertile = _tertile_battery(pm, pa, pl, pu, pv, sv2, cuts)

    with open(marginal_path, "w") as f:
        json.dump(marginal, f, indent=2)
    with open(tertile_path, "w") as f:
        json.dump({"cuts_q33_q66": list(cuts), "strata": tertile}, f, indent=2)

    logger.info(
        "  baseline %s: R²=%.3f, IS@95=%.3f, cov95=%.3f",
        name,
        marginal["r2_log"],
        marginal["is_95"],
        marginal["cov_95"],
    )
    return {"baseline": name, "marginal_metrics": marginal, "cached": False}


def run_baseline_lme(
    cohort: Cohort, cfg: dict, output_root: Path, *, force: bool = False
) -> dict[str, Any]:
    """Run the homoscedastic LME baseline once. Independent of σ²_v."""
    return _run_baseline(
        name="LME_baseline",
        model_cls=LMEGrowthModel,
        model_kwargs={"method": cfg.get("lme", {}).get("method", "reml")},
        trajectories=cohort.trajectories,
        output_root=output_root,
        cohort=cohort,
        force=force,
    )


def run_baseline_lme_hetero_zero(
    cohort: Cohort, cfg: dict, output_root: Path, *, force: bool = False
) -> dict[str, Any]:
    """Run LMEHetero with σ²_v ≡ floor_variance (controlled-homo baseline)."""
    floor_var = cfg.get("uncertainty", {}).get("floor_variance", 1e-3)
    zero_trajs = []
    for traj in cohort.trajectories:
        new_traj = deepcopy(traj)
        new_traj.observation_variance = np.full(traj.n_timepoints, floor_var, dtype=np.float64)
        zero_trajs.append(new_traj)

    return _run_baseline(
        name="LMEHetero_Zero_baseline",
        model_cls=LMEHeteroGrowthModel,
        model_kwargs={"floor_variance": floor_var},
        trajectories=zero_trajs,
        output_root=output_root,
        cohort=cohort,
        force=force,
    )


# ---------------------------------------------------------------------------
# Sweep iterators
# ---------------------------------------------------------------------------


def resolve_tau_grid(cfg: dict, cohort: Cohort) -> np.ndarray:
    """Build the τ-grid from config.

    If ``sweep.primary.tau_grid`` is provided explicitly, it is used as-is
    after sorting and deduplication. Otherwise, ``sweep.primary.saturation``
    is used to compute (τ_min, τ_max) from the empirical p5/p95, and an
    evenly spaced grid of ``n_tau`` values (default 9) is built that always
    contains τ=0 (the empirical-match cell).
    """
    primary = cfg["sweep"]["primary"]
    if "tau_grid" in primary and primary["tau_grid"] is not None:
        grid = np.asarray(sorted(set(float(t) for t in primary["tau_grid"])), dtype=np.float64)
        return grid

    saturation = primary.get("saturation", {})
    floor = float(
        saturation.get(
            "sigma_v_sq_floor",
            cfg.get("uncertainty", {}).get("floor_variance", 1e-3),
        )
    )
    ceil = float(saturation.get("sigma_v_sq_ceil", 50.0))
    safety = float(saturation.get("safety_margin", 2.0))

    log_emp = np.log(np.maximum(cohort.empirical_sigma_v_sq_flat, 1e-15))
    tau_min, tau_max = compute_tau_endpoints(
        log_emp,
        sigma_v_sq_floor=floor,
        sigma_v_sq_ceil=ceil,
        safety_margin=safety,
    )
    n_tau = int(primary.get("n_tau", 9))
    return build_tau_grid(n_tau, tau_min, tau_max, include_zero=True)


def iter_primary_specs(cfg: dict, cohort: Cohort) -> list[CellSpec]:
    """Yield all (τ, seed) cells of the empirical-shift primary sweep.

    τ=0 is the empirical-match cell (deterministic up to bootstrap noise);
    extreme τ values saturate at the floor / ceiling.
    """
    primary = cfg["sweep"]["primary"]
    n_seeds = int(cfg["evaluation"]["n_seeds"])
    tau_grid = resolve_tau_grid(cfg, cohort)

    family = primary.get("family", "empirical_shift")
    specs: list[CellSpec] = []
    for tau in tau_grid:
        level = f"tau_{tau:+.3f}"
        for s in range(n_seeds):
            specs.append(
                CellSpec(
                    family=family,
                    level=level,
                    level_value=float(tau),
                    seed=s,
                )
            )
    return specs


def iter_ablation_specs(cfg: dict) -> list[CellSpec]:
    """Yield all (α, seed) cells of the Beta(α) ablation sweep."""
    ablation = cfg["sweep"].get("ablation", {})
    if not ablation.get("enabled", False):
        return []

    n_seeds = int(ablation.get("n_seeds_override", cfg["evaluation"]["n_seeds"]))
    alpha_grid = list(ablation["alpha_grid"])

    specs: list[CellSpec] = []
    for a in alpha_grid:
        level = f"alpha_{a:+.3f}"
        for s in range(n_seeds):
            specs.append(
                CellSpec(
                    family="beta_alpha",
                    level=level,
                    level_value=float(a),
                    seed=s,
                )
            )
    return specs
