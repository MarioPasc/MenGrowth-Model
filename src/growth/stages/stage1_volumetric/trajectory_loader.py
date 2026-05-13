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

# Index into semantic/volume for the meningioma growth target.
#
# semantic/volume columns are:
#   [log(V_total+1), log(V_NCR+1), log(V_ED+1), log(V_ET+1), log(V_MEN+1)]
# (see src/growth/data/semantic_features.py).
#
# V_MEN = V_NCR + V_ET (labels 1|3) = meningioma mass excluding peritumoral
# edema (SNFH, label 2). This matches the ensemble volume target (BSF ch0,
# TC channel) and the canonical definition in the project memory.
_MEN_VOL_IDX = 4  # column 4 = log(V_MEN + 1) = meningioma mass volume


def load_trajectories_from_h5(
    h5_path: str | Path,
    time_variable: str = "ordinal",
    exclude_patients: list[str] | None = None,
    min_timepoints: int = 2,
    covariate_features: list[str] | None = None,
    semantic_covariates: list[str] | None = None,
    skip_all_zero_volume: bool = True,
) -> list[PatientTrajectory]:
    """Load per-patient meningioma-mass volume trajectories from MenGrowth H5.

    Reads ``semantic/volume[:, 4]`` (pre-computed log1p of V_MEN, labels 1|3)
    and the longitudinal index to build ``PatientTrajectory`` objects. The
    meningioma mass volume = NCR (label 1) + ET (label 3) in BraTS-MEN,
    excluding peritumoral edema (SNFH, label 2). This matches the canonical
    volume target (BSF ch0 = TC channel) and the ensemble prediction target.

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
        skip_all_zero_volume: If True, skip patients where all ET
            volumes are zero (empty enhancing tumor at all timepoints).

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

        # Meningioma mass log-volume — column 4 of semantic/volume (labels 1|3).
        obs = semantic_vol[scan_indices, _MEN_VOL_IDX].astype(np.float64)

        # Skip patients with all-zero volumes
        if skip_all_zero_volume and np.all(obs == 0.0):
            logger.debug(f"Skipping {pid}: all ET volumes are zero")
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


def _compute_deltas_from_dates(
    study_dates: np.ndarray,
    patient_offsets: np.ndarray,
) -> tuple[np.ndarray, list[bool]]:
    """Compute days-from-baseline from YYYY-MM-DD study dates.

    Args:
        study_dates: Per-scan date strings [N_scans].
        patient_offsets: CSR offsets [N_patients + 1].

    Returns:
        (delta_days [N_scans] float64, has_dates [N_patients] bool).
        Patients with missing/malformed dates get NaN entries and
        has_dates[i] = False.
    """
    from datetime import datetime

    n_scans = len(study_dates)
    n_patients = len(patient_offsets) - 1
    delta_days = np.full(n_scans, np.nan, dtype=np.float64)
    has_dates: list[bool] = []

    for i in range(n_patients):
        start = patient_offsets[i]
        end = patient_offsets[i + 1]
        patient_ok = True
        dates: list[datetime | None] = []

        for idx in range(start, end):
            raw = study_dates[idx]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            raw = str(raw).strip()
            if not raw or raw.lower() in ("", "nan", "none", "unknown"):
                patient_ok = False
                dates.append(None)
                continue
            try:
                dates.append(datetime.strptime(raw, "%Y-%m-%d"))
            except ValueError:
                patient_ok = False
                dates.append(None)

        has_dates.append(patient_ok)

        if patient_ok and dates:
            baseline = dates[0]
            for j, idx in enumerate(range(start, end)):
                delta_days[idx] = (dates[j] - baseline).days

    return delta_days, has_dates


def load_uncertainty_trajectories_from_h5(
    h5_path: str | Path,
    *,
    time_variable: str = "ordinal",
    estimator: str = "mean_std",
    variance_key: str | None = None,
    mean_key: str | None = None,
    exclude_patients: list[str] | None = None,
    min_timepoints: int = 2,
    covariate_features: list[str] | None = None,
    semantic_covariates: list[str] | None = None,
    skip_all_zero_volume: bool = True,
    missing_date_strategy: str = "mixed",
    floor_variance: float = 1e-6,
    max_logvol_std: float | None = None,
) -> list[PatientTrajectory]:
    """Load trajectories with per-observation log-volume variance.

    Reads ``uncertainty/logvol_*`` and (optionally) ``metadata/study_date``,
    returns trajectories with ``observation_variance`` populated.

    Two read modes:

    1. **Legacy estimator mode** (``variance_key=None``): use the
       ``estimator`` parameter to look up a (mean_key, std_key) pair in
       ``_ESTIMATOR_MAP``; per-scan σ²_v is ``std**2``.
    2. **Direct signal mode** (``variance_key`` is set): read
       ``uncertainty/<variance_key>`` *as the variance itself* (no
       squaring) and use ``mean_key`` (defaulting to ``logvol_mean``) for
       the y values. This is the entry point for candidate uncertainty
       signals like ``men_mean_entropy`` injected as σ²_v.

    Args:
        h5_path: Path to MenGrowth H5 file (v2.0 schema).
        time_variable: ``"ordinal"`` or ``"days_from_baseline"``.
        estimator: Which (y, sigma^2) pair to use (legacy mode only):
            ``"mean_std"`` -> logvol_mean, logvol_std^2;
            ``"median_mad"`` -> logvol_median, logvol_mad_scaled^2;
            ``"mask_mean"`` -> logvol_ensemble, logvol_std^2.
        variance_key: If set, read ``uncertainty/<variance_key>`` directly
            as σ²_v (no squaring). Overrides ``estimator``. NaN entries are
            replaced by 0 before the ``floor_variance`` clip.
        mean_key: Mean dataset name when ``variance_key`` is set. Defaults
            to ``"logvol_mean"``. Ignored when ``variance_key is None``.
        exclude_patients: Patient IDs to exclude.
        min_timepoints: Minimum observations required per patient.
        covariate_features: Covariate names from metadata group.
        semantic_covariates: Semantic feature names to attach as covariates.
        skip_all_zero_volume: Skip patients with all-zero volumes.
        missing_date_strategy: How to handle missing dates when
            time_variable="days_from_baseline": ``"skip"`` excludes
            patients, ``"mixed"`` uses ordinal for those patients,
            ``"fail"`` raises ValueError.
        floor_variance: Minimum allowed variance (avoids singular V_i).
        max_logvol_std: If set, drop any scan whose LoRA-ensemble
            ``logvol_std`` exceeds this threshold. These scans are
            segmentation failures (zero / near-vanishing tumours where the
            ensemble disagrees about whether to predict any mask), not
            measurement noise. Filtering them prevents the inflated mean
            sigma_v from being absorbed into REML's sigma_n. After
            filtering, patients that fall below ``min_timepoints`` are also
            dropped. Set to ``None`` (default) to disable. The QC always
            uses ``uncertainty/logvol_std`` (the canonical segmentation-
            quality signal), regardless of which ``variance_key`` drives
            σ²_v.

    Returns:
        List of PatientTrajectory sorted by patient_id, with
        observation_variance populated.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    exclude_patients_set = set(exclude_patients or [])
    covariate_features = covariate_features or []
    semantic_covariates = semantic_covariates or []

    _SEMANTIC_COV_MAP = {
        "sphericity": ("shape", 0),
        "enhancement_ratio": ("shape", 1),
        "infiltration_index": ("shape", 2),
    }

    _ESTIMATOR_MAP = {
        "mean_std": ("logvol_mean", "logvol_std"),
        "median_mad": ("logvol_median", "logvol_mad_scaled"),
        "mask_mean": ("logvol_ensemble", "logvol_std"),
    }

    if variance_key is None and estimator not in _ESTIMATOR_MAP:
        raise ValueError(
            f"Unknown estimator '{estimator}'. Must be one of {list(_ESTIMATOR_MAP.keys())}"
        )

    trajectories: list[PatientTrajectory] = []

    with h5py.File(h5_path, "r") as f:
        version = f.attrs.get("version", "unknown")
        if isinstance(version, bytes):
            version = version.decode()
        if version != "2.0":
            logger.warning(f"Expected H5 schema v2.0, got {version}")

        if "uncertainty" not in f:
            raise ValueError(
                f"H5 file {h5_path} has no 'uncertainty' group. "
                f"Run the LoRA-ensemble merge pipeline first."
            )

        # Load uncertainty data
        uq = f["uncertainty"]
        if variance_key is not None:
            y_key = mean_key or "logvol_mean"
            if y_key not in uq:
                raise ValueError(
                    f"H5 uncertainty group missing mean dataset '{y_key}'"
                )
            if variance_key not in uq:
                raise ValueError(
                    f"H5 uncertainty group missing variance dataset '{variance_key}'. "
                    f"Available: {sorted(uq.keys())}"
                )
            y_values = uq[y_key][:].astype(np.float64)
            raw_var = np.nan_to_num(
                uq[variance_key][:].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
            var_values = np.maximum(raw_var, floor_variance)
            # QC always operates on logvol_std (segmentation-quality signal),
            # regardless of which variance_key drives σ²_v.
            s_values = (
                uq["logvol_std"][:].astype(np.float64) if "logvol_std" in uq else None
            )
        else:
            y_key, s_key = _ESTIMATOR_MAP[estimator]
            y_values = uq[y_key][:].astype(np.float64)
            s_values = uq[s_key][:].astype(np.float64)
            var_values = np.maximum(s_values**2, floor_variance)

        timepoint_idx = f["timepoint_idx"][:].astype(int)
        patient_ids_raw = f["patient_ids"][:]

        patient_list = [
            p.decode() if isinstance(p, bytes) else str(p)
            for p in f["longitudinal"]["patient_list"][:]
        ]
        patient_offsets = f["longitudinal"]["patient_offsets"][:].astype(int)

        # Time handling
        time_delta_days: np.ndarray | None = None
        has_dates: list[bool] | None = None

        if time_variable == "days_from_baseline":
            if "time_delta_days" in f:
                time_delta_days = f["time_delta_days"][:].astype(np.float64)
            elif "metadata" in f and "time_delta_days" in f["metadata"]:
                time_delta_days = f["metadata"]["time_delta_days"][:].astype(np.float64)
            elif "metadata" in f and "study_date" in f["metadata"]:
                study_dates = f["metadata"]["study_date"][:]
                time_delta_days, has_dates = _compute_deltas_from_dates(
                    study_dates, patient_offsets
                )
                n_with = sum(has_dates)
                n_without = len(has_dates) - n_with
                if n_without > 0:
                    logger.info(
                        f"study_date: {n_with} patients with dates, "
                        f"{n_without} without (strategy={missing_date_strategy})"
                    )
                    if missing_date_strategy == "fail":
                        missing_pids = [patient_list[i] for i, ok in enumerate(has_dates) if not ok]
                        raise ValueError(
                            f"missing_date_strategy='fail' but {n_without} patients "
                            f"lack dates: {missing_pids}"
                        )
            else:
                logger.warning(
                    "time_variable='days_from_baseline' but no date data found. "
                    "Falling back to ordinal."
                )
                time_variable = "ordinal"

        # Load metadata covariates
        metadata_arrays: dict[str, np.ndarray] = {}
        if covariate_features and "metadata" in f:
            for feat in covariate_features:
                if feat in f["metadata"]:
                    raw = f["metadata"][feat][:]
                    if raw.dtype.kind in ("U", "S", "O"):
                        metadata_arrays[feat] = np.array(
                            [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
                        )
                    else:
                        metadata_arrays[feat] = raw.astype(np.float64)

        # Load semantic covariates
        semantic_arrays: dict[str, np.ndarray] = {}
        if semantic_covariates and "semantic" in f:
            for sem_name in semantic_covariates:
                if sem_name in _SEMANTIC_COV_MAP:
                    group_name, col_idx = _SEMANTIC_COV_MAP[sem_name]
                    if group_name in f["semantic"]:
                        sem_data = f["semantic"][group_name][:]
                        if col_idx < sem_data.shape[1]:
                            semantic_arrays[sem_name] = sem_data[:, col_idx].astype(np.float64)

    # Build per-patient trajectories
    n_patients = len(patient_list)
    qc_dropped_scans: list[tuple[str, int, float]] = []  # (pid, tp_idx, sigma_v)
    qc_dropped_patients: list[str] = []

    for i in range(n_patients):
        pid = patient_list[i]
        if pid in exclude_patients_set:
            continue

        start = patient_offsets[i]
        end = patient_offsets[i + 1]
        n_scans = end - start

        if n_scans < min_timepoints:
            continue

        scan_indices = list(range(start, end))

        # Sort by timepoint index
        tp_indices = timepoint_idx[scan_indices]
        sort_order = np.argsort(tp_indices)
        scan_indices = [scan_indices[j] for j in sort_order]

        # QC filter: drop scans whose raw logvol_std exceeds the threshold.
        # Applied to the raw s_values (pre-floor) so the floor cannot mask a
        # genuinely catastrophic ensemble disagreement.
        if max_logvol_std is not None:
            if s_values is None:
                logger.warning(
                    "max_logvol_std=%s requested but uncertainty/logvol_std missing — "
                    "skipping segmentation-quality QC",
                    max_logvol_std,
                )
            else:
                kept = []
                for sidx in scan_indices:
                    s_raw = float(s_values[sidx])
                    if s_raw > max_logvol_std:
                        qc_dropped_scans.append((pid, int(timepoint_idx[sidx]), s_raw))
                    else:
                        kept.append(sidx)
                scan_indices = kept

        if len(scan_indices) < min_timepoints:
            qc_dropped_patients.append(pid)
            continue

        # Time variable
        if time_variable == "days_from_baseline" and time_delta_days is not None:
            if has_dates is not None and not has_dates[i]:
                if missing_date_strategy == "skip":
                    continue
                # "mixed": fall back to ordinal for this patient
                times = timepoint_idx[scan_indices].astype(np.float64)
            else:
                times = time_delta_days[scan_indices]
        else:
            times = timepoint_idx[scan_indices].astype(np.float64)

        obs = y_values[scan_indices]
        obs_var = var_values[scan_indices]

        if skip_all_zero_volume and np.all(obs == 0.0):
            continue

        # Build covariates
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
                observation_variance=obs_var,
            )
        )

    trajectories.sort(key=lambda t: t.patient_id)

    if max_logvol_std is not None:
        logger.info(
            f"QC filter (max_logvol_std={max_logvol_std}): dropped "
            f"{len(qc_dropped_scans)} scan(s) and {len(qc_dropped_patients)} "
            f"patient(s) (fell below min_timepoints={min_timepoints} after "
            f"filter)"
        )
        for pid, tp_idx, s_raw in qc_dropped_scans:
            logger.info(f"  dropped scan: {pid} tp={tp_idx} logvol_std={s_raw:.4f}")
        for pid in qc_dropped_patients:
            logger.info(f"  dropped patient (post-filter): {pid}")

    source_desc = (
        f"variance_key={variance_key}, mean_key={mean_key or 'logvol_mean'}"
        if variance_key is not None
        else f"estimator={estimator}"
    )
    logger.info(
        f"Loaded {len(trajectories)} uncertainty trajectories from {h5_path.name} "
        f"(time={time_variable}, {source_desc}, "
        f"excluded={len(exclude_patients_set)}, floor_var={floor_variance}, "
        f"max_logvol_std={max_logvol_std})"
    )
    return trajectories


def compute_et_volumes_from_segs(
    h5_path: str | Path,
) -> dict[str, np.ndarray]:
    """Compute meningioma-mass (merged ET) volumes directly from segmentation masks in H5.

    Use this as a validation path — computes volumes fresh from the
    ``segs`` dataset instead of relying on pre-computed ``semantic/volume``.

    The "meningioma mass" target follows the BSF 2-label translation: NETC
    (label 1) is part of the solid mass and is **merged into ET** (label 3).
    This makes the helper agnostic to whether the input segs are:
      * Raw BraTS-MEN GT with separate {1=NETC, 3=ET}, or
      * Predicted segs from the LoRA ensemble (where label 3 already
        contains the merged mass and label 1 is never produced).
    In both cases the count `(seg == 1) | (seg == 3)` returns the
    meningioma-mass voxel count.

    Args:
        h5_path: Path to MenGrowth H5 file.

    Returns:
        Dict mapping patient_id to array of log1p(meningioma_mass_volume) per scan.
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
                mass_count = int(np.sum((seg == 1) | (seg == 3)))
                volumes.append(np.log1p(float(mass_count)))

            result[pid] = np.array(volumes)

    return result
