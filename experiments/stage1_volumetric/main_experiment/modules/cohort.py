"""QC-filtered cohort loader for the main experiment.

Wraps ``engine.data.load_trajectories`` (which itself wraps
``load_uncertainty_trajectories_from_h5``) and exposes the empirical σ²_v vector
in scan order so the sweep generators can be aligned per-scan.

Two σ²_v vectors are exposed:

* ``empirical_sigma_v_sq_flat`` — floored at ``uncertainty.floor_variance``;
  this is what the LMEHetero model actually consumes through
  ``observation_variance``.
* ``raw_sigma_v_sq_flat`` — same QC filter, but with the floor lowered to
  ``1e-12``. Used as the EM-fit target so the mixture is fitted to the
  *natural* empirical distribution and not to a delta spike at the floor.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.stage1_volumetric.engine.data import load_trajectories
from growth.shared.growth_models import PatientTrajectory
from growth.stages.stage1_volumetric.trajectory_loader import (  # noqa: F401
    load_uncertainty_trajectories_from_h5 as _load_uncertainty_trajectories_from_h5,
)

logger = logging.getLogger(__name__)

# Re-export under the original name so the formatter keeps the import alive.
load_uncertainty_trajectories_from_h5 = _load_uncertainty_trajectories_from_h5

_RAW_FLOOR = 1e-12


@dataclass(frozen=True)
class Cohort:
    """Materialised QC-filtered cohort.

    Attributes:
        trajectories: Per-patient trajectories with ``observation_variance``
            populated from the empirical M=20 LoRA ensemble (floored).
        empirical_sigma_v_sq_flat: Concatenated per-scan σ²_v (floored)
            in patient order. Cursor matches :func:`inject_sigma_v`.
        raw_sigma_v_sq_flat: Same vector before flooring; used by the EM
            fitter to characterise the natural empirical distribution.
        patient_ids: Patient IDs matching ``trajectories``.
        n_timepoints_per_patient: Per-patient scan count.
        n_scans_total: Sum of ``n_timepoints_per_patient``.
    """

    trajectories: list[PatientTrajectory]
    empirical_sigma_v_sq_flat: np.ndarray
    raw_sigma_v_sq_flat: np.ndarray
    mixture_fit_sigma_v_sq_flat: np.ndarray
    patient_ids: list[str]
    n_timepoints_per_patient: list[int]
    n_scans_total: int


def load_cohort(cfg: dict) -> Cohort:
    """Load and materialise the QC-filtered cohort from config.

    Loads the H5 twice with the same QC filter:
      * Pass 1 (used for the experiment): floors σ²_v at
        ``cfg.uncertainty.floor_variance``. Drives ``observation_variance``.
      * Pass 2 (used for the EM mixture fit): floors at ``1e-12``. Captures
        the natural empirical distribution without the floor-induced spike.
    """
    trajectories = load_trajectories(cfg)
    if not trajectories:
        raise RuntimeError("Cohort loader returned 0 trajectories")

    empirical_chunks: list[np.ndarray] = []
    pids: list[str] = []
    n_per_patient: list[int] = []
    for traj in trajectories:
        if traj.observation_variance is None:
            raise RuntimeError(
                f"Trajectory {traj.patient_id} has no observation_variance — "
                "main experiment requires σ²_v from H5 uncertainty group"
            )
        empirical_chunks.append(np.asarray(traj.observation_variance, dtype=np.float64))
        pids.append(str(traj.patient_id))
        n_per_patient.append(int(traj.n_timepoints))

    flat = np.concatenate(empirical_chunks)
    n_scans_total = int(flat.size)

    # Pass 2: raw values for EM fitting (floor lowered).
    uq_cfg = cfg.get("uncertainty", {})
    time_cfg = cfg["time"]
    raw_trajs = load_uncertainty_trajectories_from_h5(
        h5_path=cfg["paths"]["mengrowth_h5"],
        time_variable=time_cfg["variable"],
        estimator=uq_cfg.get("estimator", "mean_std"),
        exclude_patients=cfg["patients"].get("exclude", []),
        min_timepoints=cfg["patients"].get("min_timepoints", 2),
        skip_all_zero_volume=cfg["patients"].get("skip_all_zero_volume", True),
        missing_date_strategy=time_cfg.get("missing_date_strategy", "mixed"),
        floor_variance=_RAW_FLOOR,
        max_logvol_std=cfg["patients"].get("max_logvol_std", None),
    )
    if len(raw_trajs) != len(trajectories):
        raise RuntimeError(
            f"Raw and floored trajectory loads disagree: {len(raw_trajs)} vs {len(trajectories)}"
        )
    raw_chunks = [np.asarray(traj.observation_variance, dtype=np.float64) for traj in raw_trajs]
    raw_flat = np.concatenate(raw_chunks)
    if raw_flat.size != flat.size:
        raise RuntimeError(
            f"Raw σ²_v vector length {raw_flat.size} disagrees with floored {flat.size}"
        )

    # Pass 3: pre-QC values for the EM mixture fit. The QC filter on
    # max_logvol_std drops the high-noise tail (segmentation failures), but
    # those scans encode the true high-σ²_v regime the sweep must reach. If
    # we fit only on post-QC scans both components collapse together and
    # the bimodal structure disappears.
    pre_qc_trajs = load_uncertainty_trajectories_from_h5(
        h5_path=cfg["paths"]["mengrowth_h5"],
        time_variable=time_cfg["variable"],
        estimator=uq_cfg.get("estimator", "mean_std"),
        exclude_patients=cfg["patients"].get("exclude", []),
        min_timepoints=cfg["patients"].get("min_timepoints", 2),
        skip_all_zero_volume=cfg["patients"].get("skip_all_zero_volume", True),
        missing_date_strategy=time_cfg.get("missing_date_strategy", "mixed"),
        floor_variance=_RAW_FLOOR,
        max_logvol_std=None,
    )
    mixture_chunks = [
        np.asarray(traj.observation_variance, dtype=np.float64) for traj in pre_qc_trajs
    ]
    mixture_flat = np.concatenate(mixture_chunks)

    logger.info(
        "Cohort: %d patients, %d total scans (max_logvol_std=%.2f); raw σ²_v range "
        "[%.2e, %.2e]; pre-QC σ²_v range [%.2e, %.2e] (n=%d)",
        len(trajectories),
        n_scans_total,
        cfg["patients"].get("max_logvol_std", float("nan")),
        float(raw_flat.min()),
        float(raw_flat.max()),
        float(mixture_flat.min()),
        float(mixture_flat.max()),
        int(mixture_flat.size),
    )

    return Cohort(
        trajectories=trajectories,
        empirical_sigma_v_sq_flat=flat,
        raw_sigma_v_sq_flat=raw_flat,
        mixture_fit_sigma_v_sq_flat=mixture_flat,
        patient_ids=pids,
        n_timepoints_per_patient=n_per_patient,
        n_scans_total=n_scans_total,
    )


def write_cohort_meta(cohort: Cohort, path: Path | str) -> None:
    """Persist patient ids, scan counts, and empirical σ²_v as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "patient_ids": cohort.patient_ids,
        "n_timepoints_per_patient": cohort.n_timepoints_per_patient,
        "n_scans_total": cohort.n_scans_total,
        "empirical_sigma_v_sq_flat": cohort.empirical_sigma_v_sq_flat.tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
