"""Performance metrics computation.

Computes reconstruction quality and semantic encoding R² metrics
from experiment data.
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from ..schemas import PerformanceMetrics

logger = logging.getLogger(__name__)

# Expected column names in metrics.csv
RECON_MSE_COLS = ["val_epoch/recon", "train_epoch/recon"]
SSIM_COLS = ["val_epoch/ssim_t1c", "val_epoch/ssim_t1n", "val_epoch/ssim_t2f", "val_epoch/ssim_t2w"]
PSNR_COLS = ["val_epoch/psnr_t1c", "val_epoch/psnr_t1n", "val_epoch/psnr_t2f", "val_epoch/psnr_t2w"]

# Semantic quality partition names (in long-format CSV)
SEMANTIC_PARTITIONS = ["z_vol", "z_loc", "z_shape"]


def compute_performance_metrics(
    data: Dict[str, Any],
    epoch: Optional[int] = None,
) -> PerformanceMetrics:
    """Compute performance metrics for a specific epoch or final.

    Args:
        data: Dictionary from load_experiment_data()
        epoch: Specific epoch to analyze (None = use best/final)

    Returns:
        PerformanceMetrics dataclass
    """
    metrics_df = data.get("metrics")
    semantic_df = data.get("semantic_quality")

    result = PerformanceMetrics()

    if metrics_df is None or metrics_df.empty:
        logger.warning("No metrics data available")
        return result

    # Determine epoch to use
    if epoch is None:
        # Use the final epoch (curriculum learning means early epochs may have low loss
        # but poor semantic encoding, so final epoch is typically best)
        epoch = int(metrics_df["epoch"].max())

    result.epoch = epoch

    # Filter to specific epoch
    epoch_data = metrics_df[metrics_df["epoch"] == epoch]
    if epoch_data.empty:
        # Try closest epoch
        available_epochs = metrics_df["epoch"].unique()
        closest = min(available_epochs, key=lambda x: abs(x - epoch))
        epoch_data = metrics_df[metrics_df["epoch"] == closest]
        result.epoch = int(closest)

    if epoch_data.empty:
        logger.warning(f"No data found for epoch {epoch}")
        return result

    row = epoch_data.iloc[0]

    # Reconstruction MSE
    for col in RECON_MSE_COLS:
        if col in row.index and pd.notna(row[col]):
            result.recon_mse = float(row[col])
            break

    # Per-modality MSE
    modalities = ["t1c", "t1n", "t2f", "t2w"]
    for mod in modalities:
        col = f"val_epoch/recon_{mod}"
        if col in row.index and pd.notna(row[col]):
            result.recon_mse_per_modality[mod] = float(row[col])

    # SSIM
    ssim_values = []
    for mod in modalities:
        col = f"val_epoch/ssim_{mod}"
        if col in row.index and pd.notna(row[col]):
            val = float(row[col])
            result.ssim_per_modality[mod] = val
            ssim_values.append(val)
    if ssim_values:
        result.ssim_mean = float(np.mean(ssim_values))

    # PSNR
    psnr_values = []
    for mod in modalities:
        col = f"val_epoch/psnr_{mod}"
        if col in row.index and pd.notna(row[col]):
            val = float(row[col])
            result.psnr_per_modality[mod] = val
            psnr_values.append(val)
    if psnr_values:
        result.psnr_mean = float(np.mean(psnr_values))

    # Semantic R² from semantic_quality.csv
    if semantic_df is not None and not semantic_df.empty:
        # Handle both long format (partition column) and wide format (r2_vol columns)
        if "partition" in semantic_df.columns:
            # Long format: each row is one partition
            # Find data for this epoch
            sem_epoch_data = semantic_df[semantic_df["epoch"] == result.epoch]
            if sem_epoch_data.empty:
                # Use closest available
                available = semantic_df["epoch"].unique()
                if len(available) > 0:
                    closest = min(available, key=lambda x: abs(x - result.epoch))
                    sem_epoch_data = semantic_df[semantic_df["epoch"] == closest]

            if not sem_epoch_data.empty:
                for partition in SEMANTIC_PARTITIONS:
                    part_row = sem_epoch_data[sem_epoch_data["partition"] == partition]
                    if not part_row.empty:
                        row = part_row.iloc[0]
                        short_name = partition.replace("z_", "")  # vol, loc, shape
                        if "r2" in row.index and pd.notna(row["r2"]):
                            setattr(result, f"{short_name}_r2", float(row["r2"]))
                        if "mse" in row.index and pd.notna(row["mse"]):
                            setattr(result, f"{short_name}_mse", float(row["mse"]))
        else:
            # Wide format: columns like r2_vol, r2_loc, etc.
            sem_epoch_data = semantic_df[semantic_df["epoch"] == result.epoch]
            if sem_epoch_data.empty:
                available = semantic_df["epoch"].unique()
                if len(available) > 0:
                    closest = min(available, key=lambda x: abs(x - result.epoch))
                    sem_epoch_data = semantic_df[semantic_df["epoch"] == closest]

            if not sem_epoch_data.empty:
                sem_row = sem_epoch_data.iloc[0]
                for short_name in ["vol", "loc", "shape"]:
                    r2_col = f"r2_{short_name}"
                    mse_col = f"mse_{short_name}"
                    if r2_col in sem_row.index and pd.notna(sem_row[r2_col]):
                        setattr(result, f"{short_name}_r2", float(sem_row[r2_col]))
                    if mse_col in sem_row.index and pd.notna(sem_row[mse_col]):
                        setattr(result, f"{short_name}_mse", float(sem_row[mse_col]))

    return result


def compute_performance_history(
    data: Dict[str, Any],
    epochs: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute performance metrics over training history.

    Args:
        data: Dictionary from load_experiment_data()
        epochs: Specific epochs to include (None = all)

    Returns:
        DataFrame with performance metrics per epoch
    """
    metrics_df = data.get("metrics")
    semantic_df = data.get("semantic_quality")

    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    # Get unique epochs
    all_epochs = sorted(metrics_df["epoch"].unique())
    if epochs is not None:
        all_epochs = [e for e in all_epochs if e in epochs]

    records = []
    for epoch in all_epochs:
        metrics = compute_performance_metrics(data, epoch=epoch)
        record = {
            "epoch": metrics.epoch,
            "recon_mse": metrics.recon_mse,
            "ssim_mean": metrics.ssim_mean,
            "psnr_mean": metrics.psnr_mean,
            "vol_r2": metrics.vol_r2,
            "loc_r2": metrics.loc_r2,
            "shape_r2": metrics.shape_r2,
            "vol_mse": metrics.vol_mse,
            "loc_mse": metrics.loc_mse,
            "shape_mse": metrics.shape_mse,
        }
        # Add per-modality metrics
        for mod, val in metrics.ssim_per_modality.items():
            record[f"ssim_{mod}"] = val
        for mod, val in metrics.psnr_per_modality.items():
            record[f"psnr_{mod}"] = val
        records.append(record)

    return pd.DataFrame(records)
