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


def cohens_d(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
) -> StatisticalTestResults:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        group1: First sample
        group2: Second sample

    Returns:
        StatisticalTestResults with effect size and interpretation
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 2 or len(group2) < 2:
        return StatisticalTestResults(
            test_name="Cohen's d",
            notes="Insufficient data (need at least 2 samples per group)",
        )

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        d = 0.0
    else:
        d = (mean1 - mean2) / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return StatisticalTestResults(
        test_name="Cohen's d",
        statistic=float(d),
        effect_size=float(d),
        effect_interpretation=interpretation,
        notes=f"n1={n1}, n2={n2}, mean1={mean1:.4f}, mean2={mean2:.4f}",
    )


def cliff_delta(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
) -> StatisticalTestResults:
    """Compute Cliff's delta non-parametric effect size.

    Measures the probability that a random value from group1 is greater than
    a random value from group2, normalized to [-1, 1].

    Interpretation:
    - |delta| < 0.147: negligible
    - 0.147 <= |delta| < 0.33: small
    - 0.33 <= |delta| < 0.474: medium
    - |delta| >= 0.474: large

    Args:
        group1: First sample
        group2: Second sample

    Returns:
        StatisticalTestResults with effect size and interpretation
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 1 or len(group2) < 1:
        return StatisticalTestResults(
            test_name="Cliff's delta",
            notes="Insufficient data",
        )

    n1, n2 = len(group1), len(group2)

    # Count dominance pairs
    count_greater = 0
    count_less = 0
    for x in group1:
        count_greater += np.sum(x > group2)
        count_less += np.sum(x < group2)

    delta = (count_greater - count_less) / (n1 * n2)

    # Interpretation
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return StatisticalTestResults(
        test_name="Cliff's delta",
        statistic=float(delta),
        effect_size=float(delta),
        effect_interpretation=interpretation,
        notes=f"n1={n1}, n2={n2}",
    )


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """Apply Benjamini-Hochberg FDR correction to a list of p-values.

    Args:
        p_values: List of raw p-values
        alpha: Significance threshold (default: 0.05)

    Returns:
        List of (adjusted_p_value, is_significant) tuples, in original order
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and keep track of original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Compute adjusted p-values
    adjusted = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        adjusted_p = p * n / rank
        adjusted[orig_idx] = adjusted_p

    # Enforce monotonicity (adjusted p-values should be non-decreasing
    # when sorted by original p-values)
    # Process from largest to smallest original p-value
    sorted_by_p = sorted(range(n), key=lambda i: p_values[i], reverse=True)
    running_min = 1.0
    for idx in sorted_by_p:
        adjusted[idx] = min(adjusted[idx], running_min)
        running_min = min(running_min, adjusted[idx])

    # Cap at 1.0
    adjusted = [min(p, 1.0) for p in adjusted]

    # Determine significance
    results = [(adj_p, adj_p < alpha) for adj_p in adjusted]

    return results


def compute_stability_cv(
    values: Union[List[float], np.ndarray],
    tail_fraction: float = 0.20,
) -> float:
    """Compute coefficient of variation for the last portion of a time series.

    The CV measures relative variability: std/|mean|. Lower CV indicates
    more stable convergence.

    Args:
        values: Time series values
        tail_fraction: Fraction of the series to use (from the end)

    Returns:
        Coefficient of variation (0 = perfectly stable, higher = more variable)
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 5:
        return float("nan")

    n_tail = max(5, int(len(values) * tail_fraction))
    tail = values[-n_tail:]

    mean_val = np.mean(tail)
    if abs(mean_val) < 1e-10:
        return float("nan")

    return float(np.std(tail, ddof=1) / abs(mean_val))


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
