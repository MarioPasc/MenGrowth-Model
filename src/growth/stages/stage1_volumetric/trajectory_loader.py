# src/growth/stages/stage1_volumetric/trajectory_loader.py
"""Load per-patient volume trajectories directly from MenGrowth H5.

This module reads the pre-computed semantic features (log-volumes) and
longitudinal structure from the H5 file, bypassing the full segmentation
pipeline.  It supports two time modes:

- ``ordinal``: uses integer timepoint indices (0, 1, 2, ...)
- ``days_from_baseline``: uses ``time_delta_days`` from H5 metadata
  (when timestamps become available)

Covariates (age, sex) are attached when present in the H5 metadata group.

Usage::

    from growth.stages.stage1_volumetric.trajectory_loader import (
        load_trajectories_from_h5,
    )
    trajectories = load_trajectories_from_h5(
        h5_path="path/to/MenGrowth.h5",
        time_variable="ordinal",
        exclude_patients=["MenGrowth-0028"],
    )
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from growth.shared.growth_models import PatientTrajectory

logger = logging.getLogger(__name__)

# Index into semantic/volume for whole-tumor log-volume
_WT_VOL_IDX = 0  # column 0 = log(V_total + 1)


def load_trajectories_from_h5(
    h5_path: str | Path,
    time_variable: str = "ordinal",
    exclude_patients: list[str] | None = None,
    min_timepoints: int = 2,
    covariate_features: list[str] | None = None,
    semantic_covariates: list[str] | None = None,
    skip_all_zero_volume: bool = True,
) -> list[PatientTrajectory]:
    """Load per-patient volume trajectories from MenGrowth H5 file.

    Reads ``semantic/volume`` (pre-computed log1p volumes) and the
    longitudinal index to build ``PatientTrajectory`` objects.

    Args:
        h5_path: Path to MenGrowth H5 file (v2.0 schema).
        time_variable: ``"ordinal"`` (timepoint index) or
            ``"days_from_baseline"`` (requires ``time_delta_days`` in H5).
        exclude_patients: Patient IDs to exclude.
        min_timepoints: Minimum observations required per patient.
        covariate_features: List of covariate names to attach from
            ``metadata/`` group (e.g. ``["age", "sex"]``). Only patients
            with non-missing values will have these attached.
        semantic_covariates: List of semantic feature names to attach as
            covariates from the ``semantic/`` group. Supported:
            ``"sphericity"`` (from ``semantic/shape[:, 0]``). These have
            100% coverage and are always attached.
        skip_all_zero_volume: If True, skip patients where all WT
            volumes are zero (empty segmentations at all timepoints).

    Returns:
        List of PatientTrajectory sorted by patient_id.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    exclude_patients = set(exclude_patients or [])
    covariate_features = covariate_features or []
    semantic_covariates = semantic_covariates or []

    trajectories: list[PatientTrajectory] = []

    # Mapping from semantic covariate name to (group, column_index)
    _SEMANTIC_COV_MAP = {
        "sphericity": ("shape", 0),
        "enhancement_ratio": ("shape", 1),
        "infiltration_index": ("shape", 2),
    }

    with h5py.File(h5_path, "r") as f:
        # Validate schema
        version = f.attrs.get("version", "unknown")
        if version != "2.0":
            logger.warning(f"Expected H5 schema v2.0, got {version}")

        # Load datasets
        semantic_vol = f["semantic"]["volume"][:]  # [N_scans, 4]
        timepoint_idx = f["timepoint_idx"][:].astype(int)  # [N_scans]
        patient_ids = [
            pid.decode() if isinstance(pid, bytes) else str(pid) for pid in f["patient_ids"][:]
        ]

        # Longitudinal structure
        patient_list = [
            p.decode() if isinstance(p, bytes) else str(p)
            for p in f["longitudinal"]["patient_list"][:]
        ]
        patient_offsets = f["longitudinal"]["patient_offsets"][:].astype(int)

        # Load temporal metadata if needed
        time_delta_days: np.ndarray | None = None
        if time_variable == "days_from_baseline":
            if "time_delta_days" in f:
                time_delta_days = f["time_delta_days"][:].astype(np.float64)
            elif "metadata" in f and "time_delta_days" in f["metadata"]:
                time_delta_days = f["metadata"]["time_delta_days"][:].astype(np.float64)
            else:
                logger.warning(
                    "time_variable='days_from_baseline' but no time_delta_days "
                    "in H5. Falling back to ordinal."
                )
                time_variable = "ordinal"

        # Load covariates from metadata group
        metadata_arrays: dict[str, np.ndarray] = {}
        if covariate_features and "metadata" in f:
            for feat in covariate_features:
                if feat in f["metadata"]:
                    raw = f["metadata"][feat][:]
                    if raw.dtype.kind in ("U", "S", "O"):
                        # String data (e.g., sex)
                        metadata_arrays[feat] = np.array(
                            [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
                        )
                    else:
                        metadata_arrays[feat] = raw.astype(np.float64)
                else:
                    logger.warning(f"Covariate '{feat}' not found in H5 metadata")

        # Load semantic covariates from semantic/ group
        semantic_arrays: dict[str, np.ndarray] = {}
        if semantic_covariates and "semantic" in f:
            for sem_name in semantic_covariates:
                if sem_name in _SEMANTIC_COV_MAP:
                    group_name, col_idx = _SEMANTIC_COV_MAP[sem_name]
                    if group_name in f["semantic"]:
                        sem_data = f["semantic"][group_name][:]
                        if col_idx < sem_data.shape[1]:
                            semantic_arrays[sem_name] = sem_data[:, col_idx].astype(np.float64)
                        else:
                            logger.warning(
                                f"Semantic covariate '{sem_name}': column {col_idx} "
                                f"out of range for semantic/{group_name}"
                            )
                    else:
                        logger.warning(f"Semantic group '{group_name}' not found in H5")
                else:
                    logger.warning(
                        f"Unknown semantic covariate '{sem_name}'. "
                        f"Available: {list(_SEMANTIC_COV_MAP.keys())}"
                    )

    # Build per-patient trajectories using CSR offsets
    n_patients = len(patient_list)

    for i in range(n_patients):
        pid = patient_list[i]
        if pid in exclude_patients:
            continue

        start = patient_offsets[i]
        end = patient_offsets[i + 1]
        n_scans = end - start

        if n_scans < min_timepoints:
            logger.debug(f"Skipping {pid}: {n_scans} < {min_timepoints} timepoints")
            continue

        scan_indices = list(range(start, end))

        # Sort by timepoint index
        tp_indices = timepoint_idx[scan_indices]
        sort_order = np.argsort(tp_indices)
        scan_indices = [scan_indices[j] for j in sort_order]

        # Time variable
        if time_variable == "days_from_baseline" and time_delta_days is not None:
            times = time_delta_days[scan_indices]
        else:
            times = timepoint_idx[scan_indices].astype(np.float64)

        # WT log-volume (column 0 of semantic/volume)
        obs = semantic_vol[scan_indices, _WT_VOL_IDX].astype(np.float64)

        # Skip patients with all-zero volumes
        if skip_all_zero_volume and np.all(obs == 0.0):
            logger.debug(f"Skipping {pid}: all WT volumes are zero")
            continue

        # Build covariates dict from metadata + semantic features
        covariates = _build_covariates(scan_indices[0], covariate_features, metadata_arrays)
        sem_covs = _build_semantic_covariates(scan_indices[0], semantic_covariates, semantic_arrays)
        if sem_covs:
            if covariates is None:
                covariates = sem_covs
            else:
                covariates.update(sem_covs)

        trajectories.append(
            PatientTrajectory(
                patient_id=pid,
                times=times,
                observations=obs,
                covariates=covariates,
            )
        )

    trajectories.sort(key=lambda t: t.patient_id)

    logger.info(
        f"Loaded {len(trajectories)} trajectories from {h5_path.name} "
        f"(time={time_variable}, excluded={len(exclude_patients)}, "
        f"min_tp={min_timepoints})"
    )
    return trajectories


def _build_covariates(
    scan_idx: int,
    covariate_features: list[str],
    metadata_arrays: dict[str, np.ndarray],
) -> dict[str, float] | None:
    """Extract covariates for a patient from the first scan's metadata.

    Returns None if any requested covariate is missing/invalid.
    """
    if not covariate_features or not metadata_arrays:
        return None

    covariates: dict[str, float] = {}

    for feat in covariate_features:
        if feat not in metadata_arrays:
            return None

        raw = metadata_arrays[feat]

        if raw.dtype.kind in ("U", "S", "O"):
            # String covariate (e.g., sex) — encode as numeric
            val_str = str(raw[scan_idx])
            if val_str.lower() in ("unknown", "nan", "", "none"):
                return None
            if feat == "sex":
                covariates[feat] = 1.0 if val_str.upper() == "M" else 0.0
            else:
                return None  # Unknown string covariate
        else:
            val = float(raw[scan_idx])
            if np.isnan(val):
                return None
            covariates[feat] = val

    return covariates


def _build_semantic_covariates(
    scan_idx: int,
    semantic_covariates: list[str],
    semantic_arrays: dict[str, np.ndarray],
) -> dict[str, float] | None:
    """Extract semantic covariates for a patient's baseline scan.

    Unlike metadata covariates, semantic covariates are always available
    (100% coverage) so missing values are not expected.

    Returns:
        Dict mapping covariate name to value, or None if no covariates requested.
    """
    if not semantic_covariates or not semantic_arrays:
        return None

    covariates: dict[str, float] = {}
    for name in semantic_covariates:
        if name in semantic_arrays:
            val = float(semantic_arrays[name][scan_idx])
            if np.isfinite(val):
                covariates[name] = val

    return covariates if covariates else None


def compute_wt_volumes_from_segs(
    h5_path: str | Path,
) -> dict[str, np.ndarray]:
    """Compute WT volumes directly from segmentation masks in H5.

    Use this as a validation path — computes volumes fresh from the
    ``segs`` dataset instead of relying on pre-computed ``semantic/volume``.

    Args:
        h5_path: Path to MenGrowth H5 file.

    Returns:
        Dict mapping patient_id to array of log1p(WT_volume) per scan.
    """
    h5_path = Path(h5_path)
    result: dict[str, np.ndarray] = {}

    with h5py.File(h5_path, "r") as f:
        patient_list = [
            p.decode() if isinstance(p, bytes) else str(p)
            for p in f["longitudinal"]["patient_list"][:]
        ]
        patient_offsets = f["longitudinal"]["patient_offsets"][:].astype(int)
        segs = f["segs"]  # [N, 1, D, H, W]

        for i, pid in enumerate(patient_list):
            start = patient_offsets[i]
            end = patient_offsets[i + 1]

            volumes = []
            for idx in range(start, end):
                seg = segs[idx, 0]  # [D, H, W]
                wt_count = int(np.sum(seg > 0))
                volumes.append(np.log1p(float(wt_count)))

            result[pid] = np.array(volumes)

    return result
