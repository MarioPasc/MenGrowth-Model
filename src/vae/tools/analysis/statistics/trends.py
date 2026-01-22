"""Training dynamics and trend analysis.

Computes convergence metrics, stability analysis, and gradient health
indicators from training logs.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from ..schemas import TrendMetrics

logger = logging.getLogger(__name__)


# Thresholds for trend analysis
GRAD_EXPLOSION_THRESHOLD = 10.0  # Gradient norm > 10 is considered explosion
STABILITY_WINDOW = 50  # Last N epochs for stability calculation
CONVERGENCE_THRESHOLD = 0.90  # 90% of final performance


def compute_trend_metrics(data: Dict[str, Any]) -> TrendMetrics:
    """Compute training dynamics and trend metrics.

    Args:
        data: Dictionary from load_experiment_data()

    Returns:
        TrendMetrics dataclass
    """
    metrics_df = data.get("metrics")

    result = TrendMetrics()

    if metrics_df is None or metrics_df.empty:
        logger.warning("No metrics data available for trend analysis")
        return result

    # Sort by epoch
    df = metrics_df.sort_values("epoch").copy()

    # Loss metrics
    if "val_epoch/loss" in df.columns:
        loss_col = "val_epoch/loss"
    elif "train_epoch/loss" in df.columns:
        loss_col = "train_epoch/loss"
    else:
        logger.warning("No loss column found in metrics")
        return result

    # Remove NaN values for analysis
    loss_series = df[loss_col].dropna()
    if loss_series.empty:
        return result

    result.loss_final = float(loss_series.iloc[-1])
    result.loss_best = float(loss_series.min())
    result.loss_best_epoch = int(df.loc[loss_series.idxmin(), "epoch"])

    # Convergence check: is final loss within 5% of best?
    if result.loss_best > 0:
        relative_diff = (result.loss_final - result.loss_best) / result.loss_best
        result.loss_converged = relative_diff < 0.05

    # Epochs to reach 90% of final improvement
    if len(loss_series) > 1:
        initial_loss = float(loss_series.iloc[0])
        improvement_needed = initial_loss - result.loss_best
        target_loss = initial_loss - (CONVERGENCE_THRESHOLD * improvement_needed)

        below_target = loss_series[loss_series <= target_loss]
        if not below_target.empty:
            result.epochs_to_90pct = int(df.loc[below_target.index[0], "epoch"])

    # Stability: std/mean in last N epochs
    n_epochs = min(STABILITY_WINDOW, len(loss_series))
    last_losses = loss_series.iloc[-n_epochs:]
    if len(last_losses) > 1:
        mean_loss = last_losses.mean()
        if mean_loss > 1e-8:
            result.loss_stability = float(last_losses.std() / mean_loss)

    # Gradient health
    grad_col = None
    for col in ["train_epoch/grad_norm", "grad_norm", "diag/grad_norm"]:
        if col in df.columns:
            grad_col = col
            break

    if grad_col is not None:
        grad_series = df[grad_col].dropna()
        if not grad_series.empty:
            result.grad_norm_mean = float(grad_series.mean())
            result.grad_norm_max = float(grad_series.max())
            result.grad_explosions = int((grad_series > GRAD_EXPLOSION_THRESHOLD).sum())

    # Learning rate at end
    lr_col = None
    for col in ["lr", "learning_rate", "sched/lr"]:
        if col in df.columns:
            lr_col = col
            break

    if lr_col is not None:
        lr_series = df[lr_col].dropna()
        if not lr_series.empty:
            result.final_lr = float(lr_series.iloc[-1])

    return result


def compute_loss_history(data: Dict[str, Any]) -> pd.DataFrame:
    """Extract loss history over training.

    Args:
        data: Dictionary from load_experiment_data()

    Returns:
        DataFrame with loss components per epoch
    """
    metrics_df = data.get("metrics")

    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    # Select relevant columns
    loss_cols = [
        "epoch",
        "val_epoch/loss",
        "train_epoch/loss",
        "val_epoch/recon",
        "train_epoch/recon",
        "val_epoch/kl_raw",
        "train_epoch/kl_raw",
        "val_epoch/tc",
        "train_epoch/tc",
        "val_epoch/semantic_total",
        "train_epoch/semantic_total",
        "val_epoch/cross_partition",
        "train_epoch/cross_partition",
    ]

    available_cols = [c for c in loss_cols if c in metrics_df.columns]
    return metrics_df[available_cols].copy()


def compute_schedule_history(data: Dict[str, Any]) -> pd.DataFrame:
    """Extract training schedules (beta, lambda values) over training.

    Args:
        data: Dictionary from load_experiment_data()

    Returns:
        DataFrame with schedule values per epoch
    """
    metrics_df = data.get("metrics")

    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    # Select schedule columns
    sched_cols = [
        "epoch",
        "sched/beta",
        "sched/lambda_vol",
        "sched/lambda_loc",
        "sched/lambda_shape",
        "sched/lambda_tc",
        "sched/lambda_cross_partition",
        "sched/free_bits",
    ]

    available_cols = [c for c in sched_cols if c in metrics_df.columns]
    if len(available_cols) <= 1:  # Only epoch
        return pd.DataFrame()

    return metrics_df[available_cols].copy()
