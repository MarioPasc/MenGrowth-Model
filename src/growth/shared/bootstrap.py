# src/growth/shared/bootstrap.py
"""Bootstrap confidence intervals and permutation tests for growth prediction.

Provides .632+ bootstrap CI estimation (Efron & Tibshirani, 1997) and paired
permutation tests for comparing LOPO-CV errors across models. Used by all
stages in the variance decomposition protocol.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as _scipy_stats

logger = logging.getLogger(__name__)

__all__ = [
    "BootstrapResult",
    "PermutationTestResult",
    "bootstrap_metric",
    "paired_permutation_test",
    "cohen_d_paired",
    "benjamini_hochberg",
    "paired_bootstrap_ci",
]


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


def cohen_d_paired(differences: np.ndarray) -> float:
    """Paired Cohen's d effect size for a vector of within-unit differences.

    For paired measurements (e.g. per-patient metric of model A minus model B),
    the standardised effect size is ``mean(d) / std(d)`` with the sample
    standard deviation (``ddof=1``). This is the effect size that pairs with a
    Wilcoxon signed-rank or paired permutation test.

    Args:
        differences: Per-unit paired differences, shape ``[N]``.

    Returns:
        Paired Cohen's d. ``0.0`` if the differences have (near-)zero variance.
    """
    d = np.asarray(differences, dtype=np.float64).ravel()
    if d.size < 2:
        return 0.0
    sd = float(np.std(d, ddof=1))
    if sd < 1e-15:
        return 0.0
    return float(np.mean(d) / sd)


def benjamini_hochberg(
    p_values: np.ndarray,
    q: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg false-discovery-rate correction.

    Controls the expected proportion of false positives among the rejected
    hypotheses at level ``q`` (Benjamini & Hochberg, JRSS-B 1995). Used to
    correct the family of paired comparisons in the conformal-calibration
    experiment.

    Args:
        p_values: Raw p-values, shape ``[M]``.
        q: Target false-discovery rate.

    Returns:
        Tuple ``(rejected, p_adjusted)``: a boolean array of rejections and the
        BH-adjusted p-values (monotone, clipped to ``[0, 1]``), both shape
        ``[M]`` and aligned to the input order.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

    order = np.argsort(p)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    p_sorted = p[order]

    # Adjusted p-values: enforce monotonicity from the largest rank downward.
    adj_sorted = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    p_adjusted = np.empty(m, dtype=np.float64)
    p_adjusted[order] = adj_sorted
    rejected = p_adjusted <= q
    return rejected, p_adjusted


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """BCa bootstrap CI for the mean paired difference ``mean(a) - mean(b)``.

    Resamples the paired units (e.g. patients) with replacement and applies the
    bias-corrected and accelerated (BCa) percentile method (Efron 1987). For a
    per-patient metric such as IS@95, ``mean(a) - mean(b)`` is exactly the mean
    metric difference; a CI excluding zero indicates a significant difference.

    Args:
        values_a: Per-unit metric for model A, shape ``[N]``.
        values_b: Per-unit metric for model B, shape ``[N]``, paired with A.
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: Confidence level for the interval.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with point estimate ``mean(a) - mean(b)`` and the BCa
        confidence interval. Falls back to the percentile interval if the BCa
        acceleration is degenerate (zero-variance jackknife).
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"values_a {a.shape} and values_b {b.shape} must be paired")
    diffs = a - b
    n = diffs.size
    point_estimate = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[k] = np.mean(diffs[idx])

    alpha = 1.0 - confidence_level

    # Bias-correction z0.
    prop_less = float(np.mean(boot < point_estimate))
    prop_less = min(max(prop_less, 1.0 / (n_bootstrap + 1)), 1.0 - 1.0 / (n_bootstrap + 1))
    z0 = float(_scipy_stats.norm.ppf(prop_less))

    # Acceleration a_hat via the jackknife over the paired differences.
    jack = np.array([np.mean(np.delete(diffs, i)) for i in range(n)])
    jack_mean = float(np.mean(jack))
    num = float(np.sum((jack_mean - jack) ** 3))
    den = 6.0 * float(np.sum((jack_mean - jack) ** 2)) ** 1.5

    if den < 1e-15 or not np.isfinite(z0):
        # Degenerate acceleration / bias term: fall back to the percentile CI.
        ci_lower = float(np.percentile(boot, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    else:
        a_hat = num / den
        z_lo = _scipy_stats.norm.ppf(alpha / 2)
        z_hi = _scipy_stats.norm.ppf(1 - alpha / 2)
        a1 = _scipy_stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
        a2 = _scipy_stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))
        a1 = min(max(float(a1), 1e-6), 1.0 - 1e-6)
        a2 = min(max(float(a2), 1e-6), 1.0 - 1e-6)
        ci_lower = float(np.percentile(boot, 100 * a1))
        ci_upper = float(np.percentile(boot, 100 * a2))

    return BootstrapResult(
        estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )
