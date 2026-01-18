"""I/O utilities for experiment management.

This module provides functions for:
- Creating timestamped run directories with informative names
- Saving configuration files
- Saving train/val split CSVs
- Managing runs index for experiment tracking
- Collecting hardware information
"""

import csv
import json
import logging
import os
import platform
import socket
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def get_hardware_info() -> Dict[str, Any]:
    """Collect hardware and environment information.

    Returns:
        Dictionary with hardware info including hostname, GPU, CUDA, PyTorch versions.
    """
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }

    # CUDA info
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()

        # Get GPU model(s)
        gpu_models = []
        total_memory_gb = 0.0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_models.append(props.name)
            total_memory_gb += props.total_memory / (1024**3)

        info["gpu_model"] = gpu_models[0] if len(gpu_models) == 1 else gpu_models
        info["total_gpu_memory_gb"] = round(total_memory_gb, 2)
    else:
        info["cuda_available"] = False
        info["gpu_count"] = 0

    return info


def generate_run_id(
    cfg: DictConfig,
    experiment_type: str,
    timestamp: Optional[str] = None,
) -> str:
    """Generate an informative run ID from configuration.

    Format: {YYYYMMDD_HHMMSS}_s{seed}_b{batch}_z{zdim}[_sbd][_tag]

    Args:
        cfg: OmegaConf configuration object.
        experiment_type: Experiment type ("exp1_baseline_vae" or "exp2_dipvae").
        timestamp: Optional timestamp string. If None, generates current timestamp.

    Returns:
        Run ID string.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract key hyperparameters
    seed = cfg.train.seed
    batch_size = cfg.data.batch_size
    z_dim = cfg.model.z_dim

    # Build base run ID
    run_id = f"{timestamp}_s{seed}_b{batch_size}_z{z_dim}"

    # Add SBD suffix if using Spatial Broadcast Decoder
    use_sbd = cfg.model.get("use_sbd", False)
    if use_sbd:
        run_id += "_sbd"

    # Add custom tag if provided
    run_tag = cfg.logging.get("run_tag", None)
    if run_tag:
        run_id += f"_{run_tag}"

    return run_id


def create_run_dir(
    save_dir: str,
    cfg: DictConfig,
    experiment_type: str = "exp1_baseline_vae",
) -> Path:
    """Create a run directory with informative naming and organized structure.

    Creates directory structure:
        <save_dir>/<experiment_type>/runs/<run_id>/
            config/           # Configuration files
            checkpoints/      # Model checkpoints
            logs/             # Training logs and metrics
            visualizations/   # Reconstruction images
                reconstructions/
            diagnostics/      # Latent space diagnostics
                active_units/
                latent_probes/
                gradients/
            data/             # Data splits
                splits/
            external/         # External tool outputs (wandb)

    Args:
        save_dir: Base directory for experiment runs.
        cfg: OmegaConf configuration object for extracting hyperparameters.
        experiment_type: Experiment type ("exp1_baseline_vae" or "exp2_dipvae").

    Returns:
        Path to created run directory.
    """
    # Generate informative run ID
    run_id = generate_run_id(cfg, experiment_type)

    # Build path: save_dir/experiment_type/runs/run_id/
    run_dir = Path(save_dir) / experiment_type / "runs" / run_id

    # Create organized subdirectory structure
    subdirs = [
        "config",
        "checkpoints",
        "logs",
        "visualizations/reconstructions",
        "diagnostics/active_units",
        "diagnostics/latent_probes",
        "diagnostics/gradients",
        "data/splits",
        "external",
    ]

    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Created run directory: {run_dir}")
    logger.info(f"  Run ID: {run_id}")

    return run_dir


def save_config(
    cfg: DictConfig,
    run_dir: Union[str, Path],
    filename: str = "config.yaml",
) -> Path:
    """Save resolved configuration to YAML file.

    Args:
        cfg: OmegaConf configuration object.
        run_dir: Path to run directory.
        filename: Name of config file (saved in config/ subdir).

    Returns:
        Path to saved config file.
    """
    config_dir = Path(run_dir) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / filename

    # Save with resolve to expand any interpolations
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f, resolve=True)

    logger.info(f"Saved config to {config_path}")
    return config_path


def save_split_csvs(
    train_subjects: List[Dict[str, str]],
    val_subjects: List[Dict[str, str]],
    run_dir: Union[str, Path],
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
    splits_dir = Path(run_dir) / "data" / "splits"
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


def update_runs_index(
    run_dir: Union[str, Path],
    cfg: DictConfig,
    experiment_type: str,
    status: str = "running",
    best_val_loss: Optional[float] = None,
    best_epoch: Optional[int] = None,
    final_au_count: Optional[int] = None,
    final_au_frac: Optional[float] = None,
) -> Path:
    """Update the runs index CSV for experiment tracking.

    Creates or updates {save_dir}/{experiment_type}/runs_index.csv with run metadata.

    Args:
        run_dir: Path to run directory.
        cfg: OmegaConf configuration object.
        experiment_type: Experiment type ("exp1_baseline_vae" or "exp2_dipvae").
        status: Run status ("running", "completed", "failed").
        best_val_loss: Best validation loss (optional, for completed runs).
        best_epoch: Epoch with best validation loss (optional).
        final_au_count: Final active units count (optional).
        final_au_frac: Final active units fraction (optional).

    Returns:
        Path to runs index CSV.
    """
    run_dir = Path(run_dir)
    run_id = run_dir.name

    # Index path is at experiment level
    experiment_dir = run_dir.parent.parent  # runs/<run_id> -> <experiment_type>
    index_path = experiment_dir / "runs_index.csv"

    # Build row data
    hardware_info = get_hardware_info()

    row = {
        "run_id": run_id,
        "timestamp": run_id.split("_")[0] + "_" + run_id.split("_")[1],  # Extract timestamp
        "seed": cfg.train.seed,
        "batch_size": cfg.data.batch_size,
        "z_dim": cfg.model.z_dim,
        "max_epochs": cfg.train.max_epochs,
        "use_sbd": cfg.model.get("use_sbd", False),
        "kl_free_bits": cfg.train.get("kl_free_bits", 0.0),
        "kl_beta": cfg.train.get("kl_beta", 1.0),
        "lambda_od": cfg.loss.get("lambda_od", None) if "loss" in cfg else None,
        "lambda_d": cfg.loss.get("lambda_d", None) if "loss" in cfg else None,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_au_count": final_au_count,
        "final_au_frac": final_au_frac,
        "status": status,
        "hostname": hardware_info["hostname"],
        "gpu_model": hardware_info.get("gpu_model", "N/A"),
        "run_dir": str(run_dir),
    }

    # Load existing index or create new
    if index_path.exists():
        df = pd.read_csv(index_path)
        # Remove existing row for this run_id if present (for updates)
        df = df[df["run_id"] != run_id]
    else:
        df = pd.DataFrame()

    # Append new row
    df_new = pd.DataFrame([row])
    df = pd.concat([df, df_new], ignore_index=True)

    # Sort by timestamp descending (most recent first)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # Atomic write
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=index_path.parent, suffix=".tmp"
    ) as tmp_f:
        tmp_path = Path(tmp_f.name)
        df.to_csv(tmp_f, index=False)

    tmp_path.replace(index_path)

    logger.info(f"Updated runs index: {index_path} (status={status})")
    return index_path


def get_experiment_dir(run_dir: Union[str, Path]) -> Path:
    """Get the experiment directory from a run directory path.

    Args:
        run_dir: Path to run directory.

    Returns:
        Path to experiment directory (parent of runs/).
    """
    run_dir = Path(run_dir)
    # run_dir structure: {save_dir}/{experiment_type}/runs/{run_id}
    # experiment_dir: {save_dir}/{experiment_type}
    return run_dir.parent.parent
