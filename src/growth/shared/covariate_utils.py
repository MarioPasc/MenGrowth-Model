# src/growth/shared/covariate_utils.py
"""Shared covariate utilities for growth prediction models.

Provides centralized logic for collecting, validating, and handling missing
covariates from PatientTrajectory objects. Used by LME, H-GP, ScalarGP,
and SeverityModel to ensure consistent covariate handling across all stages.

Missing-data strategies:
    - ``"skip"``: Silently ignore patients without covariates during fitting.
    - ``"impute_mean"``: Replace missing values with training-set means.
    - ``"drop_patient"``: Exclude patients with any missing covariate entirely.

This is the canonical location. Backward-compatible re-exports exist at
``growth.models.growth.covariate_utils``.
"""

import logging

import numpy as np

from growth.shared.growth_models import PatientTrajectory

logger = logging.getLogger(__name__)

VALID_MISSING_STRATEGIES = ("skip", "impute_mean", "drop_patient")


def collect_covariates(
    patients: list[PatientTrajectory],
    covariate_names: list[str],
    missing_strategy: str = "skip",
) -> tuple[dict[str, np.ndarray], list[str], list[PatientTrajectory]]:
    """Collect covariate values from patients, handling missing data.

    Args:
        patients: Training patient trajectories.
        covariate_names: Ordered list of covariate names to extract.
        missing_strategy: How to handle patients with missing covariates.
            ``"skip"`` ignores them, ``"impute_mean"`` fills from training
            means, ``"drop_patient"`` excludes them entirely.

    Returns:
        Tuple of:
            - ``cov_values``: Dict mapping patient_id to covariate array
              ``[n_covariates]``.
            - ``used_names``: Covariate names actually used (may be fewer
              than requested if all patients lack a covariate).
            - ``filtered_patients``: Patients remaining after applying the
              missing strategy.

    Raises:
        ValueError: If ``missing_strategy`` is invalid.
    """
    if missing_strategy not in VALID_MISSING_STRATEGIES:
        raise ValueError(
            f"Invalid missing_strategy '{missing_strategy}'. "
            f"Must be one of {VALID_MISSING_STRATEGIES}"
        )

    if not covariate_names:
        return {}, [], patients

    # Step 1: Collect raw values, track which patients have complete data
    raw: dict[str, dict[str, float | None]] = {}
    for p in patients:
        vals: dict[str, float | None] = {}
        for name in covariate_names:
            if p.covariates is not None and name in p.covariates:
                vals[name] = p.covariates[name]
            else:
                vals[name] = None
        raw[p.patient_id] = vals

    # Step 2: Determine which covariates have at least one non-None value
    used_names: list[str] = []
    for name in covariate_names:
        n_present = sum(1 for v in raw.values() if v[name] is not None)
        if n_present > 0:
            used_names.append(name)
        else:
            logger.warning(f"Covariate '{name}' missing for ALL patients, skipping")

    if not used_names:
        logger.warning("No covariates available for any patient")
        return {}, [], patients

    # Step 3: Compute means from available values (for imputation)
    means: dict[str, float] = {}
    for name in used_names:
        vals = [raw[pid][name] for pid in raw if raw[pid][name] is not None]
        means[name] = float(np.mean(vals)) if vals else 0.0

    # Step 4: Apply missing strategy
    cov_values: dict[str, np.ndarray] = {}
    filtered_patients: list[PatientTrajectory] = []
    n_imputed = 0
    n_dropped = 0

    for p in patients:
        vals = raw[p.patient_id]
        has_missing = any(vals[name] is None for name in used_names)

        if has_missing:
            if missing_strategy == "drop_patient":
                n_dropped += 1
                continue
            elif missing_strategy == "skip":
                # Patient included but gets no covariates
                filtered_patients.append(p)
                continue
            elif missing_strategy == "impute_mean":
                n_imputed += 1

        # Build covariate array
        cov_arr = np.array(
            [vals[name] if vals[name] is not None else means[name] for name in used_names],
            dtype=np.float64,
        )
        cov_values[p.patient_id] = cov_arr
        filtered_patients.append(p)

    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped}/{len(patients)} patients due to missing covariates")
    if n_imputed > 0:
        logger.info(f"Imputed covariates for {n_imputed}/{len(patients)} patients (means: {means})")

    n_with_covs = len(cov_values)
    logger.info(
        f"Covariates collected: {len(used_names)} features, "
        f"{n_with_covs}/{len(filtered_patients)} patients have values"
    )

    return cov_values, used_names, filtered_patients


def get_patient_covariate_vector(
    patient: PatientTrajectory,
    covariate_names: list[str],
    training_means: dict[str, float] | None = None,
) -> np.ndarray | None:
    """Extract a covariate vector for a single patient.

    Args:
        patient: Patient trajectory.
        covariate_names: Ordered list of covariate names.
        training_means: If provided, used to impute missing values.
            If ``None`` and any covariate is missing, returns ``None``.

    Returns:
        Covariate array ``[n_covariates]`` or ``None`` if unavailable.
    """
    if not covariate_names:
        return None

    values: list[float] = []
    for name in covariate_names:
        if patient.covariates is not None and name in patient.covariates:
            values.append(patient.covariates[name])
        elif training_means is not None and name in training_means:
            values.append(training_means[name])
        else:
            return None

    return np.array(values, dtype=np.float64)
