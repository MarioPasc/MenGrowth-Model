"""Controlled-homo baseline: LMEHetero with σ²_v ≡ 0 (floor only).

Setting every observation_variance entry to ``floor_variance`` reduces the
heteroscedastic residual covariance ``R_i = σ²_n·I + diag(σ²_v)`` to
``R_i ≈ σ²_n·I``, which is the homoscedastic LME likelihood. The custom
L-BFGS-B REML optimiser used by ``LMEHeteroGrowthModel`` then converges
on the same hyperparameters a homo LME would find — but using the *same*
implementation as ``LMEHetero@empirical``.

This makes the comparison ``LMEHetero@σ²_v=0`` vs ``LMEHetero@empirical``
the **clean propagation effect**, with all implementation choices held
fixed. The drift between this baseline and the statsmodels-based
``LMEGrowthModel`` quantifies the structural REML implementation gap
identified in the synthetic stress test (Profile A).

Usage::

    cd /home/mpascual/research/code/MenGrowth-Model
    ~/.conda/envs/growth/bin/python -m \\
        experiments.stage1_volumetric.run_lme_hetero_zero \\
        --output-dir /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/uncertainty_propagation_volume_prediction
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from growth.models.growth.lme_hetero import LMEHeteroGrowthModel
from growth.shared.bootstrap import bootstrap_metric
from growth.shared.lopo import LOPOEvaluator, LOPOResults
from growth.shared.metrics import (
    compute_coverage_at_levels,
    compute_crps_gaussian,
    compute_interval_score,
    compute_mae,
    compute_r2,
    compute_rmse,
)
from growth.stages.stage1_volumetric.trajectory_loader import (
    load_uncertainty_trajectories_from_h5,
)

logger = logging.getLogger(__name__)

DEFAULT_H5 = (
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/source/MenGrowth.h5"
)
DEFAULT_OUTPUT = (
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction"
)
MODEL_DIR_NAME = "LMEHetero_Zero"


def zero_out_observation_variance(trajectories, floor_value: float = 1e-6) -> list:
    """Deep-copy trajectories with every σ²_v entry set to ``floor_value``.

    The L-BFGS-B REML optimiser inside ``LMEHeteroGrowthModel`` requires
    a strictly positive variance to keep V_i invertible. Using the
    floor (1e-6 by default) is numerically equivalent to σ²_v ≡ 0
    because it is six orders of magnitude below σ²_n ≈ 0.55.
    """
    out = []
    for traj in trajectories:
        new_traj = deepcopy(traj)
        n = traj.n_timepoints
        new_traj.observation_variance = np.full(n, float(floor_value), dtype=np.float64)
        out.append(new_traj)
    return out


def _flatten_predictions(results: LOPOResults, protocol: str = "last_from_rest"):
    pm, pa, pl, pu, pv = [], [], [], [], []
    for fr in results.fold_results:
        if protocol not in fr.predictions:
            continue
        for pred in fr.predictions[protocol]:
            pm.append(pred["pred_mean"])
            pa.append(pred["actual"])
            pl.append(pred["lower_95"])
            pu.append(pred["upper_95"])
            pv.append(pred["pred_var"])
    return (np.asarray(pm), np.asarray(pa), np.asarray(pl), np.asarray(pu), np.asarray(pv))


def _save_lopo_artifacts(
    results: LOPOResults,
    out_dir: Path,
    bootstrap_n: int,
    bootstrap_seed: int,
) -> None:
    """Persist LOPO results, bootstrap CIs, and an error summary mirroring
    the layout used by the main ``run_all`` for other models."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "lopo_results.json", "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    with open(out_dir / "hyperparameters.json", "w") as f:
        per_fold = []
        for fr in results.fold_results:
            entry = {
                "patient_id": fr.patient_id,
                "log_marginal_likelihood": fr.fit_result.log_marginal_likelihood,
                "hyperparameters": fr.fit_result.hyperparameters,
            }
            per_fold.append(entry)
        json.dump({"per_fold": per_fold}, f, indent=2)

    pm, pa, pl, pu, pv = _flatten_predictions(results, "last_from_rest")
    sigma = np.sqrt(np.maximum(pv, 1e-15))
    cov = compute_coverage_at_levels(pa, pm, sigma)
    abs_err = np.abs(pa - pm)
    summary = {
        "n_patients": int(len(pa)),
        "abs_error_mean": float(abs_err.mean()),
        "abs_error_std": float(abs_err.std()),
        "abs_error_median": float(np.median(abs_err)),
        "abs_error_min": float(abs_err.min()),
        "abs_error_max": float(abs_err.max()),
        "abs_error_q25": float(np.quantile(abs_err, 0.25)),
        "abs_error_q75": float(np.quantile(abs_err, 0.75)),
        "signed_error_mean": float((pa - pm).mean()),
        "signed_error_std": float((pa - pm).std()),
        "ci_width_mean": float((pu - pl).mean()),
        "ci_width_std": float((pu - pl).std()),
        "calibration_95": float(cov[0.95]),
        "is_95": float(compute_interval_score(pa, pl, pu, alpha=0.05)),
        "crps": compute_crps_gaussian(pa, pm, sigma),
    }
    with open(out_dir / "error_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    boot = {
        "r2_log": bootstrap_metric(pa, pm, compute_r2, n_bootstrap=bootstrap_n, seed=bootstrap_seed),
        "mae_log": bootstrap_metric(pa, pm, compute_mae, n_bootstrap=bootstrap_n, seed=bootstrap_seed),
        "rmse_log": bootstrap_metric(pa, pm, compute_rmse, n_bootstrap=bootstrap_n, seed=bootstrap_seed),
    }
    with open(out_dir / "bootstrap_cis.json", "w") as f:
        json.dump(
            {
                k: {
                    "estimate": v.estimate,
                    "ci_lower": v.ci_lower,
                    "ci_upper": v.ci_upper,
                    "confidence_level": v.confidence_level,
                    "n_bootstrap": v.n_bootstrap,
                }
                for k, v in boot.items()
            },
            f,
            indent=2,
        )

    raw = []
    for fr in results.fold_results:
        for pred in fr.predictions.get("last_from_rest", []):
            raw.append(
                {
                    "patient_id": fr.patient_id,
                    "time": pred["time"],
                    "pred_mean": pred["pred_mean"],
                    "pred_var": pred["pred_var"],
                    "actual": pred["actual"],
                    "lower_95": pred["lower_95"],
                    "upper_95": pred["upper_95"],
                    "sigma_v_sq_target": pred.get("sigma_v_sq_target", 0.0),
                }
            )
    with open(out_dir / "raw_predictions.json", "w") as f:
        json.dump(raw, f, indent=2)


def _load_lopo_results(model_dir: Path) -> LOPOResults | None:
    """Load a sibling model's LOPO results from disk, if available."""
    p = model_dir / "lopo_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return LOPOResults.from_dict(json.load(f))


def _per_tertile_metrics(
    pa: np.ndarray,
    pm: np.ndarray,
    pl: np.ndarray,
    pu: np.ndarray,
    pv: np.ndarray,
    sigma_v_sq_target: np.ndarray,
    edges: tuple[float, float],
) -> dict[str, dict[str, float]]:
    """Compute (R², IS@95, cov@95, CRPS, CI width) per σ²_v_target tertile."""
    q33, q66 = edges
    masks = {
        "low": sigma_v_sq_target <= q33,
        "mid": (sigma_v_sq_target > q33) & (sigma_v_sq_target <= q66),
        "high": sigma_v_sq_target > q66,
    }
    out: dict[str, dict[str, float]] = {}
    for name, m in masks.items():
        if m.sum() == 0:
            out[name] = {"n": 0}
            continue
        sigma_t = np.sqrt(np.maximum(pv[m], 1e-15))
        cov = compute_coverage_at_levels(pa[m], pm[m], sigma_t)
        out[name] = {
            "n": int(m.sum()),
            "sigma_v_sq_mean": float(np.mean(sigma_v_sq_target[m])),
            "r2_log": compute_r2(pa[m], pm[m]),
            "ci_width_mean": float(np.mean(pu[m] - pl[m])),
            "coverage_95": float(cov[0.95]),
            "coverage_90": float(cov[0.90]),
            "coverage_80": float(cov[0.80]),
            "crps": compute_crps_gaussian(pa[m], pm[m], sigma_t),
            "interval_score_95": float(compute_interval_score(pa[m], pl[m], pu[m], alpha=0.05)),
        }
    return out


def _compare_to_existing(
    zero_results: LOPOResults,
    output_dir: Path,
) -> None:
    """Build a 3-way comparison table: LME (statsmodels), LMEHetero_Zero,
    LMEHetero@empirical. Per-tertile cuts anchored to LMEHetero@empirical."""
    other_dirs = {
        "LME": output_dir / "LME",
        "LMEHetero": output_dir / "LMEHetero",
    }
    other = {name: _load_lopo_results(d) for name, d in other_dirs.items()}
    if any(v is None for v in other.values()):
        logger.warning(
            "Could not load sibling models (missing %s); skipping comparison.",
            [k for k, v in other.items() if v is None],
        )
        return

    # Tertile cuts: use the empirical σ²_v target distribution from
    # LMEHetero@empirical so the partition is identical across all models.
    pm_e, pa_e, pl_e, pu_e, pv_e = _flatten_predictions(other["LMEHetero"])
    sigma_v_sq_target = []
    for fr in other["LMEHetero"].fold_results:
        for pred in fr.predictions.get("last_from_rest", []):
            sigma_v_sq_target.append(pred.get("sigma_v_sq_target", float("nan")))
    sigma_v_sq_target = np.asarray(sigma_v_sq_target)
    finite = np.isfinite(sigma_v_sq_target)
    q33, q66 = np.quantile(sigma_v_sq_target[finite], [1 / 3, 2 / 3])
    edges = (float(q33), float(q66))

    # The LMEHetero_Zero predictions come in the same patient order as the
    # other models because all three use the same trajectory list. We pair
    # by patient_id to be safe.
    def by_pid(results: LOPOResults) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for fr in results.fold_results:
            for pred in fr.predictions.get("last_from_rest", []):
                out[fr.patient_id] = pred
        return out

    pid_to_pred = {
        "LME": by_pid(other["LME"]),
        "LMEHetero": by_pid(other["LMEHetero"]),
        "LMEHetero_Zero": by_pid(zero_results),
    }
    common_pids = sorted(set.intersection(*[set(d.keys()) for d in pid_to_pred.values()]))
    pid_to_sv = {fr.patient_id: pred.get("sigma_v_sq_target", float("nan"))
                 for fr in other["LMEHetero"].fold_results
                 for pred in fr.predictions.get("last_from_rest", [])}

    sv_aligned = np.asarray([pid_to_sv.get(pid, float("nan")) for pid in common_pids])

    table: dict[str, dict] = {}
    for model_name, lookup in pid_to_pred.items():
        pa = np.asarray([lookup[pid]["actual"] for pid in common_pids])
        pm = np.asarray([lookup[pid]["pred_mean"] for pid in common_pids])
        pl = np.asarray([lookup[pid]["lower_95"] for pid in common_pids])
        pu = np.asarray([lookup[pid]["upper_95"] for pid in common_pids])
        pv = np.asarray([lookup[pid]["pred_var"] for pid in common_pids])
        sigma = np.sqrt(np.maximum(pv, 1e-15))
        cov = compute_coverage_at_levels(pa, pm, sigma)

        table[model_name] = {
            "marginal": {
                "n": len(common_pids),
                "r2_log": compute_r2(pa, pm),
                "ci_width_mean": float((pu - pl).mean()),
                "coverage_95": float(cov[0.95]),
                "crps": compute_crps_gaussian(pa, pm, sigma),
                "interval_score_95": float(compute_interval_score(pa, pl, pu, alpha=0.05)),
            },
            "per_tertile": _per_tertile_metrics(pa, pm, pl, pu, pv, sv_aligned, edges),
        }

    # Compute deltas for the two key contrasts.
    contrasts = []
    base_pairs = [
        ("LME", "LMEHetero_Zero", "implementation drift (statsmodels vs custom REML)"),
        ("LMEHetero_Zero", "LMEHetero", "clean propagation effect (σ²_v=0 vs empirical)"),
        ("LME", "LMEHetero", "headline (statsmodels homo vs custom-REML hetero)"),
    ]
    for a, b, label in base_pairs:
        delta_marg = {
            k: table[b]["marginal"][k] - table[a]["marginal"][k]
            for k in ("r2_log", "ci_width_mean", "coverage_95", "crps", "interval_score_95")
        }
        delta_high = {}
        for k in ("r2_log", "ci_width_mean", "coverage_95", "crps", "interval_score_95"):
            ah = table[a]["per_tertile"]["high"]
            bh = table[b]["per_tertile"]["high"]
            if ah.get("n", 0) > 0 and bh.get("n", 0) > 0 and k in ah and k in bh:
                delta_high[k] = bh[k] - ah[k]
        contrasts.append(
            {
                "from": a,
                "to": b,
                "label": label,
                "delta_marginal": delta_marg,
                "delta_high_tertile": delta_high,
            }
        )

    payload = {
        "tertile_edges_sigma_v_sq": list(edges),
        "n_common_patients": len(common_pids),
        "models": table,
        "contrasts": contrasts,
    }
    with open(output_dir / "comparison_lme_hetero_zero.json", "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# LMEHetero@σ²_v=0 — Implementation Drift & Clean Propagation Effect",
        "",
        "Three-way comparison among:",
        "- **LME** : statsmodels MixedLM (literature-default homo).",
        "- **LMEHetero_Zero** : custom L-BFGS-B REML with σ²_v ≡ 1e-6 (controlled homo).",
        "- **LMEHetero** : same custom REML with empirical σ²_v from the M=20 LoRA ensemble.",
        "",
        f"Tertile cuts on σ²_v_target (from LMEHetero@empirical): "
        f"q33={edges[0]:.4g}, q66={edges[1]:.4g}.",
        "",
        "## Marginal calibration",
        "",
        "| Model | R²_log | CI width | cov@95 | CRPS | IS@95 |",
        "|---|---|---|---|---|---|",
    ]
    for model_name, entry in table.items():
        m = entry["marginal"]
        lines.append(
            f"| {model_name} | {m['r2_log']:+.4f} | {m['ci_width_mean']:.3f} | "
            f"{m['coverage_95']:.3f} | {m['crps']:.4f} | {m['interval_score_95']:.3f} |"
        )

    lines += [
        "",
        "## High σ²_v tertile",
        "",
        "| Model | n | R²_log | CI width | cov@95 | CRPS | IS@95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for model_name, entry in table.items():
        h = entry["per_tertile"]["high"]
        if h.get("n", 0) == 0:
            continue
        lines.append(
            f"| {model_name} | {h['n']} | {h['r2_log']:+.4f} | {h['ci_width_mean']:.3f} | "
            f"{h['coverage_95']:.3f} | {h['crps']:.4f} | {h['interval_score_95']:.3f} |"
        )

    lines += [
        "",
        "## Decomposition",
        "",
    ]
    for c in contrasts:
        lines.append(f"### {c['from']} → {c['to']} ({c['label']})")
        lines.append("")
        lines.append("| Region | ΔR² | ΔCI w | Δcov_95 | ΔCRPS | ΔIS@95 |")
        lines.append("|---|---|---|---|---|---|")
        for region, deltas in (("marginal", c["delta_marginal"]), ("high tertile", c["delta_high_tertile"])):
            if not deltas:
                continue
            lines.append(
                f"| {region} | {deltas.get('r2_log', float('nan')):+.4f} | "
                f"{deltas.get('ci_width_mean', float('nan')):+.3f} | "
                f"{deltas.get('coverage_95', float('nan')):+.3f} | "
                f"{deltas.get('crps', float('nan')):+.4f} | "
                f"{deltas.get('interval_score_95', float('nan')):+.3f} |"
            )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- `LME → LMEHetero_Zero`: structural baseline — both ignore σ²_v, "
        "differ only in REML implementation (statsmodels vs custom L-BFGS-B). "
        "Δ on the high tertile bounds the implementation drift.",
        "- `LMEHetero_Zero → LMEHetero`: clean propagation effect — same "
        "implementation, σ²_v turned on. Δ on the high tertile is the "
        "honest propagation gain.",
        "- `LME → LMEHetero`: headline — sum of the two above.",
        "",
    ]
    with open(output_dir / "comparison_lme_hetero_zero.md", "w") as f:
        f.write("\n".join(lines))

    logger.info("Wrote comparison artifacts to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--floor-variance", type=float, default=1e-6)
    parser.add_argument("--bootstrap-n", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--max-logvol-std",
        type=float,
        default=1.0,
        help="Match the production config (config_uq.yaml) QC filter. "
        "Pass --no-max-logvol-std to disable (matches the published "
        "56-patient LME / LMEHetero runs).",
    )
    parser.add_argument(
        "--no-max-logvol-std",
        action="store_true",
        help="Disable the σ_v outlier filter (use the full 56-patient cohort).",
    )
    parser.add_argument(
        "--model-dir-name",
        default=MODEL_DIR_NAME,
        help="Override the output sub-directory (default: LMEHetero_Zero).",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip the 3-way comparison against LME and LMEHetero on disk.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    logging.getLogger("growth.shared.lopo").setLevel(logging.WARNING)
    logging.getLogger("growth.models.growth.lme_hetero").setLevel(logging.WARNING)
    logging.getLogger("growth.stages.stage1_volumetric.trajectory_loader").setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    model_dir = output_dir / args.model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)

    max_logvol_std = None if args.no_max_logvol_std else args.max_logvol_std
    logger.info(
        "Loading trajectories from %s (max_logvol_std=%s)", args.h5, max_logvol_std
    )
    trajs = load_uncertainty_trajectories_from_h5(
        h5_path=args.h5,
        time_variable="ordinal",
        estimator="mean_std",
        exclude_patients=["MenGrowth-0028"],
        min_timepoints=2,
        skip_all_zero_volume=True,
        floor_variance=args.floor_variance,
        max_logvol_std=max_logvol_std,
    )
    logger.info(
        "Loaded %d patients (%d scans).",
        len(trajs),
        sum(t.n_timepoints for t in trajs),
    )

    zero_trajs = zero_out_observation_variance(trajs, floor_value=args.floor_variance)
    cohort_sv = np.concatenate([t.observation_variance for t in zero_trajs])
    logger.info("σ²_v after zero-out: max=%.3g, mean=%.3g.", cohort_sv.max(), cohort_sv.mean())

    eval_ = LOPOEvaluator(prediction_protocols=["last_from_rest", "all_from_first"])

    t0 = time.monotonic()
    results = eval_.evaluate(
        LMEHeteroGrowthModel,
        zero_trajs,
        method="reml",
        n_restarts=args.n_restarts,
        use_covariates=False,
        floor_variance=args.floor_variance,
        seed=42,
    )
    elapsed = time.monotonic() - t0
    logger.info(
        "LMEHetero@σ²_v=0 LOPO: %d folds, %d failed, wall=%.1fs",
        len(results.fold_results),
        len(results.failed_folds),
        elapsed,
    )

    _save_lopo_artifacts(results, model_dir, args.bootstrap_n, args.bootstrap_seed)
    logger.info("Saved %s artifacts to %s", MODEL_DIR_NAME, model_dir)

    if not args.no_comparison:
        _compare_to_existing(results, output_dir)


if __name__ == "__main__":
    main()
