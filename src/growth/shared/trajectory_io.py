# src/growth/shared/trajectory_io.py
"""Trajectory I/O utilities for loading and saving patient trajectories.

Provides functions to load longitudinal trajectory data from JSON/dict format
into PatientTrajectory objects used by all growth models and LOPO-CV.

This is the canonical location. Backward-compatible re-exports exist at
``growth.data.trajectory_dataset``.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from growth.shared.growth_models import PatientTrajectory

logger = logging.getLogger(__name__)


def load_trajectories(
    path: str | Path,
    obs_key: str = "log_volume",
    time_key: str = "t",
) -> list[PatientTrajectory]:
    """Load patient trajectories from a JSON file.

    Expected JSON format::

        [
            {
                "patient_id": "P001",
                "timepoints": [
                    {"t": 0.0, "log_volume": 5.2, "age": 65, ...},
                    {"t": 6.5, "log_volume": 5.4, ...},
                ]
            },
            ...
        ]

    Args:
        path: Path to JSON file.
        obs_key: Key in each timepoint dict for the observation value.
        time_key: Key in each timepoint dict for the time value.

    Returns:
        List of PatientTrajectory objects.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    trajectories: list[PatientTrajectory] = []
    for entry in data:
        patient_id = entry["patient_id"]
        timepoints = entry["timepoints"]

        times = np.array([tp[time_key] for tp in timepoints], dtype=np.float64)
        observations = np.array([tp[obs_key] for tp in timepoints], dtype=np.float64)

        # Extract static covariates from the first timepoint (if available)
        covariates: dict[str, float] | None = None
        first_tp = timepoints[0]
        cov_keys = [k for k in first_tp if k not in (time_key, obs_key, "date")]
        if cov_keys:
            covariates = {
                k: float(first_tp[k]) for k in cov_keys if isinstance(first_tp[k], (int, float))
            }

        trajectories.append(
            PatientTrajectory(
                patient_id=patient_id,
                times=times,
                observations=observations,
                covariates=covariates or None,
            )
        )

    logger.info(f"Loaded {len(trajectories)} trajectories from {path}")
    return trajectories


def save_trajectories(
    trajectories: list[PatientTrajectory],
    path: str | Path,
    obs_key: str = "log_volume",
    time_key: str = "t",
) -> None:
    """Save patient trajectories to a JSON file.

    Args:
        trajectories: List of PatientTrajectory objects.
        path: Output JSON path.
        obs_key: Key name for the observation value.
        time_key: Key name for the time value.
    """
    path = Path(path)
    data = []
    for traj in trajectories:
        timepoints = []
        for i in range(traj.n_timepoints):
            tp: dict = {
                time_key: float(traj.times[i]),
                obs_key: float(traj.observations[i, 0]),
            }
            timepoints.append(tp)

        entry: dict = {
            "patient_id": traj.patient_id,
            "timepoints": timepoints,
        }
        if traj.covariates:
            entry["covariates"] = traj.covariates

        data.append(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(trajectories)} trajectories to {path}")


def trajectories_to_dataframe(trajectories: list[PatientTrajectory]) -> pd.DataFrame:
    """Convert trajectories to a pandas DataFrame for analysis.

    Returns a long-format DataFrame with columns: patient_id, time, observation,
    plus any covariate columns.

    Args:
        trajectories: List of PatientTrajectory objects.

    Returns:
        DataFrame with one row per observation.
    """
    rows: list[dict] = []
    for traj in trajectories:
        for i in range(traj.n_timepoints):
            row: dict = {
                "patient_id": traj.patient_id,
                "time": float(traj.times[i]),
                "observation": float(traj.observations[i, 0]),
            }
            if traj.covariates:
                row.update(traj.covariates)
            rows.append(row)

    return pd.DataFrame(rows)
