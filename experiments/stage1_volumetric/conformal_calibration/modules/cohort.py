"""Cohort loader for the conformal calibration experiment.

Loads trajectories with both per-scan variance (``observation_variance``) and
per-member ensemble (``observation_ensemble``) from the MenGrowth H5 file, so
all three base models (lme_homo, lme_hetero, ensemble_bma) can be run from a
single cohort load.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from growth.shared.growth_models import PatientTrajectory
from growth.stages.stage1_volumetric.trajectory_loader import (
    load_ensemble_trajectories_from_h5,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cohort:
    """Materialised QC-filtered cohort for the conformal calibration experiment.

    Attributes:
        trajectories: Per-patient trajectories with ``observation_variance``
            (floored σ²_v) and ``observation_ensemble`` ([n_i, M]) populated.
        sigma_v_sq_flat: Concatenated per-scan σ²_v (floored) in patient order.
            Used for tertile stratification.
        patient_ids: Patient IDs matching ``trajectories``.
        n_timepoints_per_patient: Per-patient scan count.
        n_scans_total: Sum of ``n_timepoints_per_patient``.
        signal_name: Identifier of the σ²_v source actually loaded.
        scaling: How the signal was placed on a variance scale
            (``"raw"`` or ``"mean_matched"``).
        n_members: Ensemble size M.
    """

    trajectories: list[PatientTrajectory]
    sigma_v_sq_flat: np.ndarray
    patient_ids: list[str]
    n_timepoints_per_patient: list[int]
    n_scans_total: int
    signal_name: str
    scaling: str
    n_members: int


def load_cohort(cfg: dict) -> Cohort:
    """Load and materialise the QC-filtered cohort from config.

    Calls :func:`~growth.stages.stage1_volumetric.trajectory_loader.load_ensemble_trajectories_from_h5`
    which populates both ``observation_variance`` (floored σ²_v) and
    ``observation_ensemble`` (log1p-transformed per-member volumes).  The
    optional ``patients.max_patients`` key truncates the cohort for smoke tests.

    Args:
        cfg: Parsed YAML configuration dict.

    Returns:
        Materialised :class:`Cohort`.

    Raises:
        RuntimeError: If the cohort loader returns zero trajectories or if any
            trajectory is missing the required ensemble/variance fields.
    """
    uq_cfg = cfg.get("uncertainty", {})
    time_cfg = cfg["time"]
    pat_cfg = cfg.get("patients", {})
    ens_cfg = cfg.get("ensemble", {})

    variance_key = uq_cfg.get("signal", "logvol_var") or "logvol_var"
    mean_key = uq_cfg.get("mean_signal", "logvol_mean") or "logvol_mean"
    scaling = uq_cfg.get("scaling", "raw") or "raw"
    floor_variance = float(uq_cfg.get("floor_variance", 1e-6))
    n_members_cfg: int | None = ens_cfg.get("n_members", None)

    trajectories = load_ensemble_trajectories_from_h5(
        h5_path=cfg["paths"]["mengrowth_h5"],
        time_variable=time_cfg["variable"],
        variance_key=variance_key,
        mean_key=mean_key,
        scaling=scaling,
        floor_variance=floor_variance,
        exclude=pat_cfg.get("exclude", []),
        min_timepoints=int(pat_cfg.get("min_timepoints", 2)),
        skip_all_zero_volume=bool(pat_cfg.get("skip_all_zero_volume", True)),
        max_logvol_std=pat_cfg.get("max_logvol_std", None),
        missing_date_strategy=time_cfg.get("missing_date_strategy", "skip"),
    )

    if not trajectories:
        raise RuntimeError("Cohort loader returned 0 trajectories")

    # Enforce max_patients (smoke / debug mode).
    max_patients: int | None = pat_cfg.get("max_patients", None)
    if max_patients is not None:
        trajectories = trajectories[: int(max_patients)]
        logger.info("max_patients=%d applied; using %d patients", max_patients, len(trajectories))

    # Truncate ensemble to n_members if requested.
    if n_members_cfg is not None:
        truncated: list[PatientTrajectory] = []
        for traj in trajectories:
            if (
                traj.observation_ensemble is not None
                and traj.observation_ensemble.shape[1] > n_members_cfg
            ):
                import copy

                t2 = copy.copy(traj)
                object.__setattr__(
                    t2, "observation_ensemble", traj.observation_ensemble[:, :n_members_cfg]
                )
                truncated.append(t2)
            else:
                truncated.append(traj)
        trajectories = truncated

    # Validate required fields.
    for traj in trajectories:
        if traj.observation_variance is None:
            raise RuntimeError(
                f"Trajectory {traj.patient_id} has no observation_variance — "
                "conformal calibration experiment requires σ²_v"
            )
        if traj.observation_ensemble is None:
            raise RuntimeError(
                f"Trajectory {traj.patient_id} has no observation_ensemble — "
                "conformal calibration experiment requires the per-member array"
            )

    # Build flat σ²_v vector for tertile stratification.
    chunks: list[np.ndarray] = []
    pids: list[str] = []
    n_per: list[int] = []
    for traj in trajectories:
        chunks.append(np.asarray(traj.observation_variance, dtype=np.float64))
        pids.append(str(traj.patient_id))
        n_per.append(int(traj.n_timepoints))

    flat = np.concatenate(chunks)
    n_members = trajectories[0].ensemble_size if trajectories else 0

    logger.info(
        "Cohort: %d patients, %d scans, M=%d members (signal=%s, scaling=%s); "
        "σ²_v range [%.3e, %.3e] mean=%.3e",
        len(trajectories),
        int(flat.size),
        n_members,
        variance_key,
        scaling,
        float(flat.min()),
        float(flat.max()),
        float(flat.mean()),
    )

    return Cohort(
        trajectories=trajectories,
        sigma_v_sq_flat=flat,
        patient_ids=pids,
        n_timepoints_per_patient=n_per,
        n_scans_total=int(flat.size),
        signal_name=variance_key,
        scaling=scaling,
        n_members=n_members,
    )


def write_cohort_meta(cohort: Cohort, path: Path | str) -> None:
    """Persist cohort summary as JSON for traceability.

    Args:
        cohort: Materialised cohort.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = cohort.sigma_v_sq_flat
    payload = {
        "signal_name": cohort.signal_name,
        "scaling": cohort.scaling,
        "n_members": cohort.n_members,
        "n_patients": len(cohort.patient_ids),
        "patient_ids": cohort.patient_ids,
        "n_timepoints_per_patient": cohort.n_timepoints_per_patient,
        "n_scans_total": cohort.n_scans_total,
        "sigma_v_sq_stats": {
            "mean": float(flat.mean()),
            "median": float(np.median(flat)),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "q33": float(np.quantile(flat, 1.0 / 3.0)),
            "q66": float(np.quantile(flat, 2.0 / 3.0)),
        },
        "sigma_v_sq_flat": flat.tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote cohort meta to %s", path)
