"""End-to-end driver for the synthetic σ²_v stress test.

Procedure (one call runs the full sweep):

  1. Load trajectories with empirical σ²_v from MenGrowth.h5
     (`load_uncertainty_trajectories_from_h5`, last_from_rest protocol,
     ordinal time, no covariates — matches the published baseline).
  2. For every (profile, level, seed):
     a. Generate synthetic σ²_v of length ``n_scans_total``.
     b. Slice the array per patient and overwrite
        ``traj.observation_variance``.
     c. Run LOPO for LMEHetero (the only model that consumes σ²_v).
     d. Cache LME results once globally (homo doesn't read σ²_v, so its
        predictions are identical across (profile, level, seed)).
     e. Compute marginal + per-tertile conditional metrics; persist
        per-fold predictions.
  3. Aggregate.py turns the per-run dumps into tables and figures.

Usage::

    cd /home/mpascual/research/code/MenGrowth-Model
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.synthetic_uq.run_synthetic_uq \\
        --output-dir /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/uncertainty_propagation_volume_prediction/synthetic_uq \\
        --n-seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from growth.models.growth.lme_hetero import LMEHeteroGrowthModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.shared.lopo import LOPOEvaluator, LOPOResults
from growth.shared.metrics import (
    compute_coverage_at_levels,
    compute_crps_gaussian,
    compute_interval_score,
    compute_log_score,
    compute_r2,
)
from growth.stages.stage1_volumetric.trajectory_loader import (
    load_uncertainty_trajectories_from_h5,
)

from .sample_profiles import (
    get_default_profiles,
    sample_profile,
)

logger = logging.getLogger(__name__)

DEFAULT_H5 = "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/source/MenGrowth.h5"
DEFAULT_OUTPUT = (
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction/synthetic_uq"
)

# ---------------------------------------------------------------------------
# Profile injection
# ---------------------------------------------------------------------------


def inject_sigma_v(
    trajectories: list,
    sigma_v_sq_flat: np.ndarray,
) -> tuple[list, np.ndarray]:
    """Slice a flat σ²_v array per-patient and write to ``observation_variance``.

    Returns deep-copied trajectories (so the caller can iterate safely) and
    the per-trajectory σ²_v array used (for diagnostic logging).
    """
    out: list = []
    used = np.zeros_like(sigma_v_sq_flat)
    cursor = 0
    for traj in trajectories:
        n_i = traj.n_timepoints
        sv = sigma_v_sq_flat[cursor : cursor + n_i].astype(np.float64).copy()
        used[cursor : cursor + n_i] = sv
        new_traj = deepcopy(traj)
        new_traj.observation_variance = sv
        out.append(new_traj)
        cursor += n_i
    if cursor != len(sigma_v_sq_flat):
        raise RuntimeError(
            f"σ²_v vector length mismatch: cursor={cursor}, expected {len(sigma_v_sq_flat)}"
        )
    return out, used


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _flatten_predictions(results: LOPOResults, protocol: str = "last_from_rest"):
    """Return arrays for the predictive distribution and the σ²_v stratifier."""
    pm, pa, pl, pu, pv, sv2 = [], [], [], [], [], []
    for fr in results.fold_results:
        if protocol not in fr.predictions:
            continue
        for pred in fr.predictions[protocol]:
            pm.append(pred["pred_mean"])
            pa.append(pred["actual"])
            pl.append(pred["lower_95"])
            pu.append(pred["upper_95"])
            pv.append(pred["pred_var"])
            sv2.append(pred.get("sigma_v_sq_target", float("nan")))
    return (
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
    sigma = np.sqrt(np.maximum(var, 1e-15))
    cov = compute_coverage_at_levels(actual, pred, sigma)
    return {
        "n": int(len(actual)),
        "r2_log": compute_r2(actual, pred),
        "ci_width_mean": float(np.mean(upper - lower)),
        "cov_50": float(cov[0.50]),
        "cov_80": float(cov[0.80]),
        "cov_90": float(cov[0.90]),
        "cov_95": float(cov[0.95]),
        "crps": compute_crps_gaussian(actual, pred, sigma),
        "is_95": compute_interval_score(actual, lower, upper, alpha=0.05),
        "nlpd": compute_log_score(actual, pred, np.maximum(var, 1e-15)),
        "pred_var_mean": float(np.mean(np.maximum(var, 0.0))),
    }


def _conditional_battery(
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
        if m.sum() == 0:
            out[name] = {"n": 0}
            continue
        out[name] = _calibration_battery(pred[m], actual[m], lower[m], upper[m], var[m])
        out[name]["sigma_v_sq_mean"] = float(np.mean(sigma_v_sq[m]))
    return out


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------


def run_one(
    trajectories,
    sigma_v_sq_flat: np.ndarray,
    *,
    fit_lme: bool,
    fit_lme_hetero: bool,
    n_restarts: int,
    seed: int,
) -> dict[str, Any]:
    """Inject σ²_v, run LOPO for the requested models, return per-fold results.

    LME (homo) results are independent of σ²_v; the caller can short-circuit
    by setting ``fit_lme=False`` after the first run.
    """
    new_trajs, _ = inject_sigma_v(trajectories, sigma_v_sq_flat)
    eval_ = LOPOEvaluator(prediction_protocols=["last_from_rest"])

    out: dict[str, Any] = {}
    if fit_lme:
        t0 = time.monotonic()
        lme_results = eval_.evaluate(
            LMEGrowthModel,
            new_trajs,
            method="reml",
            use_covariates=False,
        )
        out["LME"] = lme_results.to_dict()
        out["LME"]["wall_time_s"] = time.monotonic() - t0

    if fit_lme_hetero:
        t0 = time.monotonic()
        lmh_results = eval_.evaluate(
            LMEHeteroGrowthModel,
            new_trajs,
            method="reml",
            n_restarts=n_restarts,
            use_covariates=False,
            floor_variance=1e-6,
            seed=seed,
        )
        out["LMEHetero"] = lmh_results.to_dict()
        out["LMEHetero"]["wall_time_s"] = time.monotonic() - t0

    return out


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Subset of profile names to run (A B C D E). Default = all.",
    )
    parser.add_argument(
        "--levels", nargs="*", default=None, help="Subset of level tags to run. Default = all."
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=2,
        help="LMEHetero L-BFGS-B restarts (default 2; the published "
        "config uses 5). Lower restarts are fine here because we "
        "average over many seeds.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only profiles {A:c0.42, C:p0.4, E:empirical} with n_seeds=2 for smoke-testing.",
    )
    parser.add_argument(
        "--max-logvol-std",
        type=float,
        default=None,
        help="Optional QC filter passed to the trajectory loader.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    # Reduce noise from per-fold logs
    logging.getLogger("growth.shared.lopo").setLevel(logging.WARNING)
    logging.getLogger("growth.models.growth.lme_hetero").setLevel(logging.WARNING)
    logging.getLogger("growth.models.growth.lme_model").setLevel(logging.WARNING)
    logging.getLogger("growth.stages.stage1_volumetric.trajectory_loader").setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- step 1
    logger.info("Loading trajectories from %s", args.h5)
    base_trajs = load_uncertainty_trajectories_from_h5(
        h5_path=args.h5,
        time_variable="ordinal",
        estimator="mean_std",
        exclude_patients=["MenGrowth-0028"],
        min_timepoints=2,
        skip_all_zero_volume=True,
        floor_variance=1e-6,
        max_logvol_std=args.max_logvol_std,
    )
    n_patients = len(base_trajs)
    n_scans_total = sum(t.n_timepoints for t in base_trajs)
    empirical_sv2 = np.concatenate(
        [np.asarray(t.observation_variance, dtype=np.float64) for t in base_trajs]
    )
    logger.info(
        "Loaded %d patients, %d scans (cohort empirical mean σ²_v = %.4f)",
        n_patients,
        n_scans_total,
        float(empirical_sv2.mean()),
    )

    # ------------------------------------------------------------- step 2
    profiles = get_default_profiles()
    if args.smoke:
        wanted = {("A", "c0.01"), ("C", "p0.4"), ("E", "empirical")}
        profiles = [s for s in profiles if (s.name, s.level) in wanted]
        args.n_seeds = 2
    else:
        if args.profiles:
            profiles = [s for s in profiles if s.name in set(args.profiles)]
        if args.levels:
            profiles = [s for s in profiles if s.level in set(args.levels)]

    logger.info(
        "Will run %d (profile, level) combos × n_seeds=%d for LMEHetero",
        len(profiles),
        args.n_seeds,
    )
    for s in profiles:
        logger.info("  - %s/%s :: %s", s.name, s.level, s.description)

    # ------------------------------------------------------------- step 3
    # Run LME once with the empirical σ²_v: homo LME ignores observation_variance,
    # so its predictions are constant across all (profile, level, seed). We cache
    # them and reuse to save compute.
    logger.info("Caching LME (homo) baseline (independent of σ²_v) ...")
    lme_cache_path = output_dir / "lme_baseline.json"
    if lme_cache_path.exists():
        logger.info("  reusing cached LME results from %s", lme_cache_path)
        with open(lme_cache_path) as f:
            lme_cached = json.load(f)
    else:
        cached = run_one(
            base_trajs,
            empirical_sv2,
            fit_lme=True,
            fit_lme_hetero=False,
            n_restarts=args.n_restarts,
            seed=0,
        )
        lme_cached = cached["LME"]
        with open(lme_cache_path, "w") as f:
            json.dump(lme_cached, f, indent=2)
        logger.info(
            "  saved LME baseline to %s (wall=%.1fs)", lme_cache_path, lme_cached["wall_time_s"]
        )

    # ------------------------------------------------------------- step 4
    # Tertile cuts: locked to empirical σ²_v so all metrics are comparable
    # across profiles. (Stratifying by *injected* σ²_v would conflate the
    # propagation effect with the choice of tertile boundaries.)
    q33, q66 = np.quantile(empirical_sv2, [1.0 / 3, 2.0 / 3])
    logger.info("Empirical σ²_v tertile cuts: q33=%.4g, q66=%.4g", q33, q66)

    # Save the empirical tertile assignment per scan, plus patient mapping.
    cohort_meta = {
        "n_patients": n_patients,
        "n_scans_total": int(n_scans_total),
        "patient_ids": [t.patient_id for t in base_trajs],
        "n_timepoints_per_patient": [int(t.n_timepoints) for t in base_trajs],
        "empirical_sigma_v_sq_flat": empirical_sv2.tolist(),
        "empirical_sigma_v_sq_target_per_patient": [
            float(t.observation_variance[-1]) for t in base_trajs
        ],
        "tertile_cuts_empirical": [float(q33), float(q66)],
        "n_seeds": args.n_seeds,
        "n_restarts": args.n_restarts,
        "h5_path": args.h5,
        "max_logvol_std": args.max_logvol_std,
    }
    with open(output_dir / "cohort_meta.json", "w") as f:
        json.dump(cohort_meta, f, indent=2)

    # ------------------------------------------------------------- step 5
    summary_rows: list[dict[str, Any]] = []

    for spec in profiles:
        for seed in range(args.n_seeds):
            run_dir = output_dir / "runs" / f"{spec.name}_{spec.level}" / f"seed{seed:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            done_marker = run_dir / "DONE"
            if done_marker.exists():
                logger.info("[skip] %s seed=%d already done", run_dir, seed)
                with open(run_dir / "marginal.json") as f:
                    summary_rows.extend(json.load(f)["rows"])
                continue

            rng = np.random.default_rng(int(1e6) + 1000 * (ord(spec.name) - 65) + seed)
            sv2 = sample_profile(spec, n_scans_total, rng, empirical_sigma_v_sq=empirical_sv2)

            t0 = time.monotonic()
            res = run_one(
                base_trajs,
                sv2,
                fit_lme=False,
                fit_lme_hetero=True,
                n_restarts=args.n_restarts,
                seed=seed,
            )
            elapsed = time.monotonic() - t0

            # ---- compute metrics from cached LME and fresh LMEHetero ----
            rows = []
            for model_name, results_dict in [
                ("LME", lme_cached),
                ("LMEHetero", res["LMEHetero"]),
            ]:
                lr = LOPOResults.from_dict(results_dict)
                pm, pa, pl, pu, pv, sv2_target = _flatten_predictions(lr)
                marg = _calibration_battery(pm, pa, pl, pu, pv)
                marg.update(
                    {
                        "model": model_name,
                        "profile": spec.name,
                        "level": spec.level,
                        "seed": seed,
                        "wall_time_s": float(results_dict.get("wall_time_s", 0.0)),
                        "injected_sigma_v_sq_mean": float(np.mean(sv2)),
                        "injected_sigma_v_sq_p90": float(np.percentile(sv2, 90)),
                        "injected_sigma_v_sq_high_frac": float(np.mean(sv2 > q66)),
                    }
                )
                cond = _conditional_battery(pm, pa, pl, pu, pv, sv2_target, (q33, q66))
                rows.append({**marg, "conditional": cond})

            with open(run_dir / "marginal.json", "w") as f:
                json.dump({"rows": rows}, f, indent=2)
            # Persist LMEHetero LOPO results so downstream tools can re-stratify
            # by the *empirical* sigma_v_sq_target (joining via patient_id +
            # cohort_meta) — see `re_aggregate_empirical.py`.
            with open(run_dir / "lopo_results_lmehetero.json", "w") as f:
                json.dump(res["LMEHetero"], f)
            np.save(run_dir / "sigma_v_sq_injected.npy", sv2)
            done_marker.write_text("ok\n")
            summary_rows.extend(rows)
            logger.info(
                "[%s/%s seed=%d] LMEHetero LOPO wall=%.1fs", spec.name, spec.level, seed, elapsed
            )

    # ------------------------------------------------------------- step 6
    summary_path = output_dir / "summary_rows.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info("Wrote %d rows to %s", len(summary_rows), summary_path)
    logger.info("Sweep complete.")


if __name__ == "__main__":
    main()
