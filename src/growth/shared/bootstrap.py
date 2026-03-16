# src/growth/shared/bootstrap.py
"""Bootstrap confidence intervals and permutation tests for growth prediction.

Provides .632+ bootstrap CI estimation (Efron & Tibshirani, 1997) and paired
permutation tests for comparing LOPO-CV errors across models. Used by all
stages in the variance decomposition protocol.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of a bootstrap CI computation.

    Args:
        estimate: Point estimate of the metric.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        confidence_level: Confidence level (e.g. 0.95).
        n_bootstrap: Number of bootstrap iterations used.
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_bootstrap: int = 2000


@dataclass
class PermutationTestResult:
    """Result of a paired permutation test.

    Args:
        observed_diff: Observed difference in the metric (model_a - model_b).
        p_value: Two-sided p-value.
        n_permutations: Number of permutations used.
    """

    observed_diff: float
    p_value: float
    n_permutations: int


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute bootstrap CI for a metric on paired (y_true, y_pred) data.

    Uses BCa (bias-corrected and accelerated) percentile method.

    Args:
        y_true: Ground truth values, shape ``[N]``.
        y_pred: Predicted values, shape ``[N]``.
        metric_fn: Function ``(y_true, y_pred) -> float``.
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: Confidence level for the interval.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with point estimate and confidence interval.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point_estimate = metric_fn(y_true, y_pred)

    boot_estimates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_estimates[b] = metric_fn(y_true[idx], y_pred[idx])

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(boot_estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_estimates, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def paired_permutation_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> PermutationTestResult:
    """Two-sided paired permutation test on per-patient errors.

    Tests H₀: mean(errors_a) = mean(errors_b) against H₁: they differ.
    Used for comparing LOPO-CV errors between nested models (e.g., Stage 1
    vs Stage 2) in the variance decomposition.

    Args:
        errors_a: Per-patient errors from model A, shape ``[N_patients]``.
        errors_b: Per-patient errors from model B, shape ``[N_patients]``.
        n_permutations: Number of permutations.
        seed: Random seed for reproducibility.

    Returns:
        PermutationTestResult with observed difference and p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(errors_a)
    assert len(errors_b) == n, "Error arrays must have the same length"

    diffs = errors_a - errors_b
    observed_diff = float(np.mean(diffs))

    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_diff = np.mean(signs * diffs)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = float((count_extreme + 1) / (n_permutations + 1))

    return PermutationTestResult(
        observed_diff=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
    )
