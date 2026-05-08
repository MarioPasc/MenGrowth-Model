"""Stage 2 cell-level runner for the candidate-uncertainty-signal diagnostic.

One task = one (candidate, scaling) cell, or one negative control. The
runner wraps the existing main_experiment infrastructure:

* ``inject_sigma_v`` from ``synthetic_uq``
* ``LOPOEvaluator`` + ``LMEHeteroGrowthModel`` / ``LMEGrowthModel``
* ``_save_results_with_sigma_v``, ``_flatten_predictions``,
  ``_calibration_battery``, ``_tertile_battery``,
  ``empirical_tertile_cuts`` from ``main_experiment.modules.runner``

so that downstream aggregation, bootstrap, and figure code from the main
experiment can be reused without modification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.stage1_volumetric.main_experiment.modules.cohort import Cohort
from experiments.stage1_volumetric.main_experiment.modules.runner import (
    PROTOCOL,
    _calibration_battery,
    _flatten_predictions,
    _save_results_with_sigma_v,
    _tertile_battery,
    empirical_tertile_cuts,
)
from experiments.stage1_volumetric.synthetic_uq.run_synthetic_uq import inject_sigma_v
from growth.models.growth.lme_hetero import LMEHeteroGrowthModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.shared.lopo import LOPOEvaluator

from .candidates import (
    CANDIDATE_REGISTRY,
    CONTROL_NAMES,
    SCALING_REGISTRY,
    apply_floor,
    build_control,
)

logger = logging.getLogger(__name__)

# Special task name for the LME-homo sanity baseline (no σ²_v injection).
HOMO_SANITY_NAME = "homo_sanity"


@dataclass(frozen=True)
class TaskSpec:
    """Identifier for one diagnostic cell."""

    candidate: str
    scaling: str

    @property
    def cell_dirname(self) -> str:
        return f"candidate_{self.candidate}_scaling_{self.scaling}"


@dataclass
class DiagnosticManifest:
    """Ordered list of ``TaskSpec`` for the SLURM array."""

    tasks: list[TaskSpec] = field(default_factory=list)

    @classmethod
    def from_cfg(cls, cfg: dict) -> DiagnosticManifest:
        diag = cfg.get("diagnostic", {})
        tasks: list[TaskSpec] = []
        candidates = diag.get("candidates", [])
        scalings = diag.get("scalings", ["mean_matched"])
        for c in candidates:
            if c not in CANDIDATE_REGISTRY:
                logger.warning("Unknown candidate %s — skipping", c)
                continue
            for s in scalings:
                if s not in SCALING_REGISTRY:
                    logger.warning("Unknown scaling %s — skipping", s)
                    continue
                tasks.append(TaskSpec(candidate=c, scaling=s))

        for c in diag.get("controls", []):
            if c not in CONTROL_NAMES:
                logger.warning("Unknown control %s — skipping", c)
                continue
            tasks.append(TaskSpec(candidate=c, scaling="raw"))

        # Sanity check: re-run LME homo so we can verify the pipeline matches
        # the existing main_experiment LME_baseline.
        tasks.append(TaskSpec(candidate=HOMO_SANITY_NAME, scaling="raw"))
        return cls(tasks=tasks)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                [{"candidate": t.candidate, "scaling": t.scaling} for t in self.tasks],
                f,
                indent=2,
            )

    @classmethod
    def from_json(cls, path: Path) -> DiagnosticManifest:
        with open(path) as f:
            data = json.load(f)
        return cls(tasks=[TaskSpec(**d) for d in data])


# ---------------------------------------------------------------------------
# σ²_v vector assembly from the candidate CSV
# ---------------------------------------------------------------------------


def filter_candidates_to_cohort(
    candidates_df: pd.DataFrame,
    cohort: Cohort,
    cfg: dict,
) -> pd.DataFrame:
    """Apply the same QC filter as ``load_cohort`` and reorder rows to match
    cohort scan order. Returns a frame of ``cohort.n_scans_total`` rows.
    """
    df = candidates_df.copy()
    excl = cfg["patients"].get("exclude", [])
    df = df[~df["patient_id"].isin(excl)]

    max_lvs = cfg["patients"].get("max_logvol_std", None)
    if max_lvs is not None:
        df = df[df["logvol_var"] <= float(max_lvs) ** 2]

    if cfg["patients"].get("skip_all_zero_volume", True):
        # Per-patient: drop patient if all of its retained scans have y = 0
        per_pat = df.groupby("patient_id")["logvol_mean"]
        nonzero = per_pat.transform(lambda s: (s > 0).any())
        df = df[nonzero]

    min_tp = int(cfg["patients"].get("min_timepoints", 2))
    counts = df.groupby("patient_id").size()
    df = df[df["patient_id"].isin(counts[counts >= min_tp].index)]

    pid_order = {pid: i for i, pid in enumerate(cohort.patient_ids)}
    df = df.assign(_pid_rank=df["patient_id"].map(pid_order))
    df = df.dropna(subset=["_pid_rank"]).copy()
    df["_pid_rank"] = df["_pid_rank"].astype(int)
    df = df.sort_values(["_pid_rank", "timepoint_idx"]).reset_index(drop=True)

    if len(df) != cohort.n_scans_total:
        raise RuntimeError(
            f"Candidate CSV after QC has {len(df)} scans but cohort has "
            f"{cohort.n_scans_total}. Cohort patient_ids unique = "
            f"{len(set(cohort.patient_ids))}; CSV unique pids = "
            f"{df['patient_id'].nunique()}"
        )
    return df


def build_sigma_v_vector(
    spec: TaskSpec,
    cohort: Cohort,
    candidates_df: pd.DataFrame,
    cfg: dict,
) -> np.ndarray:
    """Assemble per-scan σ²_v for the given task spec, in cohort scan order."""
    if spec.candidate in CONTROL_NAMES:
        seed = int(cfg.get("diagnostic", {}).get("permuted_seed", 17))
        return build_control(
            spec.candidate, cohort.n_scans_total, cohort.empirical_sigma_v_sq_flat, seed=seed
        )

    if spec.candidate == HOMO_SANITY_NAME:
        return cohort.empirical_sigma_v_sq_flat.copy()

    aligned = filter_candidates_to_cohort(candidates_df, cohort, cfg)
    if spec.candidate not in aligned.columns:
        raise KeyError(
            f"candidate column {spec.candidate!r} not in candidate_signals.csv "
            f"— available: {[c for c in aligned.columns if c not in ('patient_id', 'timepoint_idx')]}"
        )
    raw_vec = aligned[spec.candidate].to_numpy(dtype=np.float64)
    if np.isnan(raw_vec).any():
        n_nan = int(np.isnan(raw_vec).sum())
        logger.warning(
            "candidate %s has %d / %d NaN entries — filling with 0 (Stage-0 repair recommended)",
            spec.candidate,
            n_nan,
            raw_vec.size,
        )
        raw_vec = np.nan_to_num(raw_vec, nan=0.0)
    scaling_fn = SCALING_REGISTRY[spec.scaling]
    return scaling_fn(raw_vec, cohort.empirical_sigma_v_sq_flat)


# ---------------------------------------------------------------------------
# Trajectory subsetting for smoke runs
# ---------------------------------------------------------------------------


def subset_cohort(cohort: Cohort, max_folds: int | None) -> Cohort:
    """Keep only the first ``max_folds`` patients (smoke testing)."""
    if max_folds is None or max_folds >= len(cohort.trajectories):
        return cohort
    keep_idx = list(range(max_folds))
    keep_pids = [cohort.patient_ids[i] for i in keep_idx]
    keep_trajs = [cohort.trajectories[i] for i in keep_idx]
    keep_n = [cohort.n_timepoints_per_patient[i] for i in keep_idx]
    n_scans_kept = sum(keep_n)
    cursor = 0
    chunks = []
    for i in range(len(cohort.patient_ids)):
        n = cohort.n_timepoints_per_patient[i]
        if i in keep_idx:
            chunks.append(cohort.empirical_sigma_v_sq_flat[cursor : cursor + n])
        cursor += n
    flat = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
    return Cohort(
        trajectories=keep_trajs,
        empirical_sigma_v_sq_flat=flat,
        raw_sigma_v_sq_flat=flat.copy(),
        mixture_fit_sigma_v_sq_flat=flat.copy(),
        patient_ids=keep_pids,
        n_timepoints_per_patient=keep_n,
        n_scans_total=n_scans_kept,
    )


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------


def run_task(
    spec: TaskSpec,
    cohort: Cohort,
    candidates_df: pd.DataFrame | None,
    cfg: dict,
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run one (candidate, scaling) cell or control."""
    cell_dir = output_root / "runs" / spec.cell_dirname
    cell_dir.mkdir(parents=True, exist_ok=True)
    sigma_path = cell_dir / "sigma_v_sq_injected.npy"
    lopo_path = cell_dir / "lopo_results.json"
    marginal_path = cell_dir / "marginal_metrics.json"
    tertile_path = cell_dir / "tertile_metrics.json"

    if not force and lopo_path.exists() and marginal_path.exists() and tertile_path.exists():
        logger.info("CACHED %s", spec.cell_dirname)
        with open(marginal_path) as f:
            marginal = json.load(f)
        return {"cell_dir": str(cell_dir), "marginal_metrics": marginal, "cached": True}

    floor_var = float(cfg.get("uncertainty", {}).get("floor_variance", 1e-3))
    sigma_v_sq_raw = build_sigma_v_vector(spec, cohort, candidates_df, cfg)
    sigma_v_sq = apply_floor(sigma_v_sq_raw, floor_var)
    np.save(sigma_path, sigma_v_sq)

    if spec.candidate == HOMO_SANITY_NAME:
        evaluator = LOPOEvaluator(prediction_protocols=[PROTOCOL])
        results = evaluator.evaluate(
            LMEGrowthModel,
            cohort.trajectories,
            method=cfg.get("lme", {}).get("method", "reml"),
        )
    else:
        new_trajs, _ = inject_sigma_v(cohort.trajectories, sigma_v_sq)
        evaluator = LOPOEvaluator(prediction_protocols=[PROTOCOL])
        results = evaluator.evaluate(
            LMEHeteroGrowthModel,
            new_trajs,
            floor_variance=floor_var,
        )

    _save_results_with_sigma_v(results, sigma_v_sq, cohort, lopo_path)

    pids, pm, pa, pl, pu, pv, sv2 = _flatten_predictions(results)
    if pm.size == 0:
        raise RuntimeError(f"{spec.cell_dirname} produced no predictions")

    marginal = _calibration_battery(pm, pa, pl, pu, pv)
    cuts = empirical_tertile_cuts(cohort.empirical_sigma_v_sq_flat)
    tertile = _tertile_battery(pm, pa, pl, pu, pv, sv2, cuts)

    with open(marginal_path, "w") as f:
        json.dump(marginal, f, indent=2)
    with open(tertile_path, "w") as f:
        json.dump({"cuts_q33_q66": list(cuts), "strata": tertile}, f, indent=2)

    logger.info(
        "  %s: R²=%.3f IS@95=%.3f cov95=%.3f W=%.3f",
        spec.cell_dirname,
        marginal["r2_log"],
        marginal["is_95"],
        marginal["cov_95"],
        marginal["ci_width_mean"],
    )
    return {"cell_dir": str(cell_dir), "marginal_metrics": marginal, "cached": False}
