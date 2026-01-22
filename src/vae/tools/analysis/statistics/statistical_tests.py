"""Statistical tests for experiment analysis.

Provides Mann-Whitney U, bootstrap confidence intervals, Spearman correlation,
and Levene's variance test for comparing experiments and assessing significance.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

from ..schemas import StatisticalTestResults

logger = logging.getLogger(__name__)


def mann_whitney_u_test(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    alternative: str = "two-sided",
) -> StatisticalTestResults:
    """Perform Mann-Whitney U test for comparing two independent samples.

    Non-parametric test for comparing distributions. Useful for comparing
    AU counts or R² values between runs.

    Args:
        group1: First sample
        group2: Second sample
        alternative: "two-sided", "less", or "greater"

    Returns:
        StatisticalTestResults with U statistic and p-value
    """
    try:
        from scipy import stats
    except ImportError:
        logger.warning("scipy not available, returning empty results")
        return StatisticalTestResults(
            test_name="Mann-Whitney U",
            notes="scipy not available",
        )

    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 2 or len(group2) < 2:
        return StatisticalTestResults(
            test_name="Mann-Whitney U",
            notes="Insufficient data (need at least 2 samples per group)",
        )

    try:
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative=alternative
        )

        # Compute effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        return StatisticalTestResults(
            test_name="Mann-Whitney U",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < 0.05,
            effect_size=float(effect_size),
            notes=f"n1={n1}, n2={n2}, alternative={alternative}",
        )
    except Exception as e:
        logger.error(f"Mann-Whitney U test failed: {e}")
        return StatisticalTestResults(
            test_name="Mann-Whitney U",
            notes=f"Test failed: {e}",
        )


def bootstrap_confidence_interval(
    data: Union[List[float], np.ndarray],
    statistic_func: callable = np.mean,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = 42,
) -> StatisticalTestResults:
    """Compute bootstrap confidence interval for a statistic.

    Uses BCa (bias-corrected and accelerated) bootstrap when scipy is available,
    falls back to percentile method otherwise.

    Args:
        data: Sample data
        statistic_func: Function to compute statistic (default: np.mean)
        confidence_level: Confidence level (default: 0.95)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility

    Returns:
        StatisticalTestResults with point estimate and confidence interval
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if len(data) < 2:
        return StatisticalTestResults(
            test_name="Bootstrap CI",
            notes="Insufficient data (need at least 2 samples)",
        )

    rng = np.random.default_rng(random_state)

    # Point estimate
    point_estimate = float(statistic_func(data))

    # Bootstrap samples
    bootstrap_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile confidence interval
    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return StatisticalTestResults(
        test_name="Bootstrap CI",
        statistic=point_estimate,
        p_value=0.0,  # Not applicable
        significant=False,  # Not applicable
        confidence_interval=(lower, upper),
        notes=f"n={len(data)}, n_bootstrap={n_bootstrap}, confidence={confidence_level}",
    )


def spearman_correlation(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
) -> StatisticalTestResults:
    """Compute Spearman rank correlation.

    Measures monotonic relationship between two variables.
    Useful for assessing if metrics are consistently improving over epochs.

    Args:
        x: First variable (e.g., epoch)
        y: Second variable (e.g., R² values)

    Returns:
        StatisticalTestResults with correlation coefficient and p-value
    """
    try:
        from scipy import stats
    except ImportError:
        logger.warning("scipy not available, returning empty results")
        return StatisticalTestResults(
            test_name="Spearman Correlation",
            notes="scipy not available",
        )

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove pairs with NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return StatisticalTestResults(
            test_name="Spearman Correlation",
            notes="Insufficient data (need at least 3 pairs)",
        )

    try:
        correlation, p_value = stats.spearmanr(x, y)

        return StatisticalTestResults(
            test_name="Spearman Correlation",
            statistic=float(correlation),
            p_value=float(p_value),
            significant=p_value < 0.05,
            effect_size=float(correlation),  # Correlation is its own effect size
            notes=f"n={len(x)}",
        )
    except Exception as e:
        logger.error(f"Spearman correlation failed: {e}")
        return StatisticalTestResults(
            test_name="Spearman Correlation",
            notes=f"Test failed: {e}",
        )


def levene_variance_test(
    *groups: Union[List[float], np.ndarray],
    center: str = "median",
) -> StatisticalTestResults:
    """Perform Levene's test for equality of variances.

    Tests if multiple groups have equal variances. Useful for checking
    if different partitions have similar variance levels.

    Args:
        *groups: Two or more groups to compare
        center: "mean", "median", or "trimmed"

    Returns:
        StatisticalTestResults with W statistic and p-value
    """
    try:
        from scipy import stats
    except ImportError:
        logger.warning("scipy not available, returning empty results")
        return StatisticalTestResults(
            test_name="Levene's Test",
            notes="scipy not available",
        )

    if len(groups) < 2:
        return StatisticalTestResults(
            test_name="Levene's Test",
            notes="Need at least 2 groups",
        )

    # Clean groups
    cleaned_groups = []
    for g in groups:
        g = np.asarray(g)
        g = g[~np.isnan(g)]
        if len(g) >= 2:
            cleaned_groups.append(g)

    if len(cleaned_groups) < 2:
        return StatisticalTestResults(
            test_name="Levene's Test",
            notes="Insufficient data after removing NaN",
        )

    try:
        statistic, p_value = stats.levene(*cleaned_groups, center=center)

        group_sizes = [len(g) for g in cleaned_groups]
        group_vars = [float(np.var(g, ddof=1)) for g in cleaned_groups]

        return StatisticalTestResults(
            test_name="Levene's Test",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < 0.05,
            notes=f"n_groups={len(cleaned_groups)}, sizes={group_sizes}, variances={[f'{v:.4f}' for v in group_vars]}",
        )
    except Exception as e:
        logger.error(f"Levene's test failed: {e}")
        return StatisticalTestResults(
            test_name="Levene's Test",
            notes=f"Test failed: {e}",
        )


def test_convergence_monotonicity(
    epochs: Union[List[int], np.ndarray],
    values: Union[List[float], np.ndarray],
    increasing: bool = True,
) -> StatisticalTestResults:
    """Test if a metric shows monotonic improvement over epochs.

    Uses Spearman correlation to assess if values consistently
    increase (or decrease) with epoch.

    Args:
        epochs: Epoch numbers
        values: Metric values
        increasing: If True, test for positive correlation (improvement = increase)
                   If False, test for negative correlation (improvement = decrease)

    Returns:
        StatisticalTestResults indicating if monotonic trend is significant
    """
    result = spearman_correlation(epochs, values)

    if result.statistic is not None and result.statistic != 0:
        # Check direction matches expectation
        expected_sign = 1.0 if increasing else -1.0
        correct_direction = (result.statistic * expected_sign) > 0

        result.notes += f", expected_direction={'increasing' if increasing else 'decreasing'}"
        result.notes += f", correct_direction={correct_direction}"

        # Only significant if both p-value is low AND direction is correct
        result.significant = result.p_value < 0.05 and correct_direction

    result.test_name = f"Convergence Monotonicity ({'increasing' if increasing else 'decreasing'})"
    return result
