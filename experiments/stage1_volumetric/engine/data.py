# experiments/stage1_volumetric/engine/data.py
"""Configuration loading and trajectory data preparation."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from growth.shared.growth_models import PatientTrajectory
from growth.stages.stage1_volumetric.trajectory_loader import (
    load_uncertainty_trajectories_from_h5,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load YAML config.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Parsed configuration dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_trajectories(cfg: dict) -> list[PatientTrajectory]:
    """Load patient trajectories from H5 based on config settings.

    All models (homo, hetero, NLME) use the same trajectories so that
    paired comparisons are valid. Homoscedastic models ignore
    observation_variance; heteroscedastic models consume it.

    Args:
        cfg: Parsed YAML config dict.

    Returns:
        List of PatientTrajectory with observation_variance populated.
    """
    uq_cfg = cfg.get("uncertainty", {})
    time_cfg = cfg["time"]
    cov_cfg = cfg.get("covariates", {})
    covariate_features = cov_cfg.get("features", []) if cov_cfg.get("enabled", False) else []

    trajectories = load_uncertainty_trajectories_from_h5(
        h5_path=cfg["paths"]["mengrowth_h5"],
        time_variable=time_cfg["variable"],
        estimator=uq_cfg.get("estimator", "mean_std"),
        variance_key=uq_cfg.get("signal"),
        mean_key=uq_cfg.get("mean_signal"),
        exclude_patients=cfg["patients"].get("exclude", []),
        min_timepoints=cfg["patients"].get("min_timepoints", 2),
        covariate_features=covariate_features,
        semantic_covariates=cfg.get("semantic_covariates", []),
        skip_all_zero_volume=cfg["patients"].get("skip_all_zero_volume", True),
        missing_date_strategy=time_cfg.get("missing_date_strategy", "mixed"),
        floor_variance=uq_cfg.get("floor_variance", 1e-6),
        max_logvol_std=cfg["patients"].get("max_logvol_std", None),
    )

    logger.info(f"Loaded {len(trajectories)} trajectories (ensemble logvol_mean + variance)")
    return trajectories
