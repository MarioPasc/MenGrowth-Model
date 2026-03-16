# src/growth/shared/metrics.py
"""Standalone metric functions for growth prediction evaluation.

All functions operate on numpy arrays and are used by LOPOEvaluator,
variance decomposition, and stage-specific evaluation scripts.
"""

import numpy as np


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 (coefficient of determination).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        R^2 score. Can be negative if predictions are worse than mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_calibration(
    y_true: np.ndarray,
    lower_95: np.ndarray,
    upper_95: np.ndarray,
) -> float:
    """Compute calibration as fraction of observations within 95% CI.

    Args:
        y_true: Ground truth values.
        lower_95: Lower bound of 95% credible interval.
        upper_95: Upper bound of 95% credible interval.

    Returns:
        Fraction in [0, 1]. Target: 0.90-0.98 for well-calibrated models.
    """
    within = (y_true >= lower_95) & (y_true <= upper_95)
    return float(np.mean(within))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        eps: Small constant to avoid division by zero.

    Returns:
        MAPE as a fraction (not percentage).
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))
