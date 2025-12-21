"""I/O utilities for experiment management.

This module provides functions for:
- Creating timestamped run directories
- Saving configuration files
- Saving train/val split CSVs
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def create_run_dir(save_dir: str, experiment_name: str = "exp1_baseline_vae") -> Path:
    """Create a timestamped run directory.

    Creates directory structure:
        <save_dir>/<timestamp>_<experiment_name>/
            checkpoints/
            logs/
            recon/
            splits/

    Args:
        save_dir: Base directory for experiment runs.
        experiment_name: Name to append to timestamp.

    Returns:
        Path to created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{experiment_name}"
    run_dir = Path(save_dir) / run_name

    # Create subdirectories
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "recon").mkdir(parents=True, exist_ok=True)
    (run_dir / "splits").mkdir(parents=True, exist_ok=True)

    logger.info(f"Created run directory: {run_dir}")
    return run_dir


def save_config(cfg: DictConfig, run_dir: str, filename: str = "config_resolved.yaml") -> Path:
    """Save resolved configuration to YAML file.

    Args:
        cfg: OmegaConf configuration object.
        run_dir: Path to run directory.
        filename: Name of config file.

    Returns:
        Path to saved config file.
    """
    config_path = Path(run_dir) / filename

    # Save with resolve to expand any interpolations
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f, resolve=True)

    logger.info(f"Saved config to {config_path}")
    return config_path


def save_split_csvs(
    train_subjects: List[Dict[str, str]],
    val_subjects: List[Dict[str, str]],
    run_dir: str,
) -> tuple:
    """Save train/val split information to CSV files.

    Each CSV contains columns: id, t1c, t1n, t2f, t2w, seg

    Args:
        train_subjects: List of training subject dicts.
        val_subjects: List of validation subject dicts.
        run_dir: Path to run directory.

    Returns:
        Tuple of (train_csv_path, val_csv_path).
    """
    splits_dir = Path(run_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train_subjects.csv"
    val_csv = splits_dir / "val_subjects.csv"

    # Determine columns from first subject
    if train_subjects:
        columns = list(train_subjects[0].keys())
    elif val_subjects:
        columns = list(val_subjects[0].keys())
    else:
        columns = ["id"]

    # Ensure 'id' is first
    if "id" in columns:
        columns.remove("id")
        columns = ["id"] + sorted(columns)

    # Write train CSV
    with open(train_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(train_subjects)

    # Write val CSV
    with open(val_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(val_subjects)

    logger.info(f"Saved train split ({len(train_subjects)} subjects) to {train_csv}")
    logger.info(f"Saved val split ({len(val_subjects)} subjects) to {val_csv}")

    return train_csv, val_csv
