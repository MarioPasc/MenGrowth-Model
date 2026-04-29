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


def compute_crps_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Closed-form CRPS for Gaussian predictive distribution.

    Gneiting & Raftery (2007), JASA. With omega = (y - mu) / sigma:

        CRPS = sigma * [omega * (2*Phi(omega) - 1) + 2*phi(omega) - 1/sqrt(pi)]

    Args:
        y_true: Ground truth values.
        mu: Predictive means.
        sigma: Predictive standard deviations (must be > 0).

    Returns:
        Mean CRPS over the batch. Lower is better.
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-15)

    omega = (y_true - mu) / sigma
    crps_per = sigma * (
        omega * (2.0 * norm.cdf(omega) - 1.0) + 2.0 * norm.pdf(omega) - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_per))


def compute_coverage_at_levels(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> dict[float, float]:
    """Empirical coverage at each nominal level (Gaussian assumption).

    Args:
        y_true: Ground truth values.
        mu: Predictive means.
        sigma: Predictive standard deviations.
        levels: Nominal coverage levels to evaluate.

    Returns:
        Dict mapping nominal level to empirical coverage fraction.
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-15)

    result: dict[float, float] = {}
    for level in levels:
        z = norm.ppf((1.0 + level) / 2.0)
        lower = mu - z * sigma
        upper = mu + z * sigma
        within = (y_true >= lower) & (y_true <= upper)
        result[level] = float(np.mean(within))
    return result


def compute_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Winkler interval score for (1-alpha) prediction intervals.

    IS_alpha = (u-l) + (2/alpha)*(l-y)*1[y<l] + (2/alpha)*(y-u)*1[y>u]

    Gneiting & Raftery (2007), section 6.2. Lower is better.

    Args:
        y_true: Ground truth values.
        lower: Lower bound of prediction interval.
        upper: Upper bound of prediction interval.
        alpha: Significance level (e.g. 0.05 for 95% interval).

    Returns:
        Mean interval score. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    width = upper - lower
    penalty_lower = (2.0 / alpha) * np.maximum(lower - y_true, 0.0)
    penalty_upper = (2.0 / alpha) * np.maximum(y_true - upper, 0.0)

    return float(np.mean(width + penalty_lower + penalty_upper))
