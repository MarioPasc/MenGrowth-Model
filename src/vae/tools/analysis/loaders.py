"""Data loading utilities for experiment analysis.

This module provides functions to load experiment artifacts (CSVs, configs, etc.)
and validate experiment directory structure.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import yaml

from .schemas import ExperimentMetadata

logger = logging.getLogger(__name__)


# Expected files in a valid experiment directory
REQUIRED_FILES = [
    "config/config.yaml",
]

OPTIONAL_FILES = [
    "config/run_meta.json",
    "metrics.csv",
    "logs/metrics.csv",
    "test_metrics.csv",
    "logs/test_metrics.csv",
    "diagnostics/active_units/au_history.csv",
    "diagnostics/semivae/semantic_quality.csv",
    "diagnostics/semivae/semantic_tracking.csv",
    "diagnostics/semivae/partition_stats.csv",
    "diagnostics/semivae/cross_correlation.csv",
    "diagnostics/gradients/grad_stats.csv",
]


def validate_experiment_directory(run_dir: str) -> Tuple[bool, List[str], List[str]]:
    """Validate that a directory contains expected experiment artifacts.

    Args:
        run_dir: Path to experiment run directory

    Returns:
        Tuple of (is_valid, found_files, missing_required)
    """
    run_path = Path(run_dir)

    if not run_path.exists():
        return False, [], [f"Directory does not exist: {run_dir}"]

    found_files = []
    missing_required = []

    # Check required files
    for rel_path in REQUIRED_FILES:
        full_path = run_path / rel_path
        if full_path.exists():
            found_files.append(rel_path)
        else:
            missing_required.append(rel_path)

    # Check optional files
    for rel_path in OPTIONAL_FILES:
        full_path = run_path / rel_path
        if full_path.exists():
            found_files.append(rel_path)

    is_valid = len(missing_required) == 0
    return is_valid, found_files, missing_required


def load_config(run_dir: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML.

    Args:
        run_dir: Path to experiment run directory

    Returns:
        Configuration dictionary
    """
    config_path = Path(run_dir) / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_run_metadata(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load run metadata from JSON if available.

    Args:
        run_dir: Path to experiment run directory

    Returns:
        Metadata dictionary or None if not found
    """
    meta_path = Path(run_dir) / "config" / "run_meta.json"

    if not meta_path.exists():
        return None

    with open(meta_path, "r") as f:
        return json.load(f)


def load_metrics_csv(
    run_dir: str,
    filename: str = "metrics.csv",
) -> Optional[pd.DataFrame]:
    """Load metrics CSV file.

    Searches in multiple locations: root, logs/, and data/ subdirectories.

    Args:
        run_dir: Path to experiment run directory
        filename: Name of metrics file (default: metrics.csv)

    Returns:
        DataFrame with metrics or None if not found
    """
    run_path = Path(run_dir)

    # Search in multiple locations
    search_paths = [
        run_path / filename,
        run_path / "logs" / filename,
        run_path / "data" / filename,
    ]

    metrics_path = None
    for path in search_paths:
        if path.exists():
            metrics_path = path
            break

    if metrics_path is None:
        logger.warning(f"Metrics file {filename} not found in {run_dir}")
        return None

    try:
        df = pd.read_csv(metrics_path)
        logger.debug(f"Loaded metrics from {metrics_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {metrics_path}: {e}")
        return None


def load_diagnostics_csv(
    run_dir: str,
    subdir: str,
    filename: str,
) -> Optional[pd.DataFrame]:
    """Load a diagnostics CSV file.

    Args:
        run_dir: Path to experiment run directory
        subdir: Subdirectory under diagnostics/ (e.g., "active_units", "semivae")
        filename: CSV filename

    Returns:
        DataFrame or None if not found
    """
    csv_path = Path(run_dir) / "diagnostics" / subdir / filename

    if not csv_path.exists():
        logger.debug(f"Diagnostics file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return None


def load_experiment_data(run_dir: str) -> Dict[str, Any]:
    """Load all available experiment data from a run directory.

    This is the main entry point for loading experiment artifacts.
    Loads config, metadata, and all available CSV files.

    Args:
        run_dir: Path to experiment run directory

    Returns:
        Dictionary with all loaded data:
        {
            "config": {...},
            "metadata": {...},
            "metrics": pd.DataFrame,
            "test_metrics": pd.DataFrame,
            "au_history": pd.DataFrame,
            "semantic_quality": pd.DataFrame,
            "partition_stats": pd.DataFrame,
            "cross_correlation": pd.DataFrame,
        }
    """
    run_path = Path(run_dir)

    # Validate directory
    is_valid, found_files, missing = validate_experiment_directory(run_dir)
    if not is_valid:
        raise ValueError(f"Invalid experiment directory. Missing: {missing}")

    logger.info(f"Loading experiment data from: {run_dir}")
    logger.info(f"Found files: {found_files}")

    data = {
        "run_dir": str(run_path.absolute()),
        "found_files": found_files,
    }

    # Load config (required)
    data["config"] = load_config(run_dir)

    # Load metadata (optional)
    data["metadata"] = load_run_metadata(run_dir)

    # Load main metrics
    data["metrics"] = load_metrics_csv(run_dir, "metrics.csv")
    data["test_metrics"] = load_metrics_csv(run_dir, "test_metrics.csv")

    # Load diagnostics CSVs
    data["au_history"] = load_diagnostics_csv(run_dir, "active_units", "au_history.csv")
    data["semantic_quality"] = load_diagnostics_csv(run_dir, "semivae", "semantic_quality.csv")
    data["semantic_tracking"] = load_diagnostics_csv(run_dir, "semivae", "semantic_tracking.csv")
    data["partition_stats"] = load_diagnostics_csv(run_dir, "semivae", "partition_stats.csv")
    data["cross_correlation"] = load_diagnostics_csv(run_dir, "semivae", "cross_correlation.csv")
    data["grad_stats"] = load_diagnostics_csv(run_dir, "gradients", "grad_stats.csv")

    return data


def extract_experiment_metadata(data: Dict[str, Any]) -> ExperimentMetadata:
    """Extract structured metadata from loaded experiment data.

    Args:
        data: Dictionary from load_experiment_data()

    Returns:
        ExperimentMetadata dataclass
    """
    config = data.get("config", {})
    run_meta = data.get("metadata", {})

    # Extract run_id from metadata or directory name
    run_id = run_meta.get("run_id", Path(data["run_dir"]).name)

    # Extract partition dimensions from config
    partitioning = config.get("model", {}).get("latent_partitioning", {})
    z_vol_dim = partitioning.get("z_vol", {}).get("dim", 0)
    z_loc_dim = partitioning.get("z_loc", {}).get("dim", 0)
    z_shape_dim = partitioning.get("z_shape", {}).get("dim", 0)
    z_residual_dim = partitioning.get("z_residual", {}).get("dim", 0)

    # Get completed epochs from metrics
    completed_epochs = 0
    if data.get("metrics") is not None and "epoch" in data["metrics"].columns:
        completed_epochs = int(data["metrics"]["epoch"].max())

    # Hardware info
    hardware = run_meta.get("hardware", {})
    gpu_models = hardware.get("gpu_model", [])
    gpu_model = gpu_models[0] if gpu_models else None

    return ExperimentMetadata(
        run_id=run_id,
        run_dir=data["run_dir"],
        experiment_type=config.get("model", {}).get("variant", "unknown"),
        start_time=run_meta.get("start_time"),
        end_time=run_meta.get("end_time"),
        status=run_meta.get("status", "unknown"),
        max_epochs=config.get("train", {}).get("max_epochs", 0),
        completed_epochs=completed_epochs,
        gpu_model=gpu_model,
        gpu_count=hardware.get("gpu_count", 1),
        z_dim=config.get("model", {}).get("z_dim", 128),
        batch_size=config.get("data", {}).get("batch_size", 2),
        learning_rate=config.get("train", {}).get("lr", 1e-4),
        z_vol_dim=z_vol_dim,
        z_loc_dim=z_loc_dim,
        z_shape_dim=z_shape_dim,
        z_residual_dim=z_residual_dim,
    )
