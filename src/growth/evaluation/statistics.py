# src/growth/evaluation/statistics.py
"""General-purpose statistical analysis utilities.

This module provides reusable statistical functions for:
- Bootstrap confidence intervals
- Effect size computation (Cohen's d)
- Paired statistical tests (Wilcoxon, paired t-test)
- Multiple comparison correction (Bonferroni, Holm-Bonferroni)

These functions are general-purpose and can be used across different experiments.

Example:
    >>> from growth.evaluation.statistics import bootstrap_ci, paired_statistical_test
    >>> ci = bootstrap_ci(data, n_bootstrap=1000)
    >>> print(f"Mean: {ci.mean:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")
    >>> result = paired_statistical_test(baseline, condition)
    >>> print(f"p={result.p_value:.4f}, d={result.effect_size:.3f}")
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from scipy import stats
from sklearn.utils import resample


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval results.

    Attributes:
        mean: Point estimate (mean or median of original data).
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        std: Standard deviation of bootstrap distribution.
        n_bootstrap: Number of bootstrap samples used.
    """

    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_bootstrap: int = 1000

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"

    def __repr__(self) -> str:
        return (
            f"BootstrapCI(mean={self.mean:.4f}, "
            f"ci=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"std={self.std:.4f})"
        )


@dataclass
class PairedTestResult:
    """Results from a paired statistical test.

    Attributes:
        test_name: Name of the statistical test used.
        statistic: Test statistic value.
        p_value: Two-sided p-value.
        effect_size: Cohen's d effect size.
        effect_interpretation: Interpretation of effect size magnitude.
        significant_005: Whether significant at alpha=0.05.
        significant_001: Whether significant at alpha=0.01.
        n1: Sample size of first group.
        n2: Sample size of second group.
    """

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_interpretation: str
    significant_005: bool
    significant_001: bool
    n1: int
    n2: int

    def __str__(self) -> str:
        stars = (
            "***"
            if self.p_value < 0.001
            else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "n.s."
        )
        return (
            f"{self.test_name}: p={self.p_value:.4f} {stars}, "
            f"d={self.effect_size:.3f} ({self.effect_interpretation})"
        )

    def __repr__(self) -> str:
        return (
            f"PairedTestResult(test={self.test_name}, p={self.p_value:.4f}, "
            f"d={self.effect_size:.3f})"
        )


# Backward compatibility alias
StatisticalTest = PairedTestResult


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = "mean",
    random_state: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval.

    Args:
        data: 1D array of values.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (default 0.95 for 95% CI).
        statistic: "mean" or "median".
        random_state: Random seed for reproducibility.

    Returns:
        BootstrapCI with mean, CI bounds, and std.

    Example:
        >>> data = np.array([1.2, 1.5, 1.8, 2.0, 2.3])
        >>> result = bootstrap_ci(data, n_bootstrap=1000, ci=0.95)
        >>> print(f"{result.mean:.2f} [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    """
    rng = np.random.RandomState(random_state)
    n = len(data)

    # Compute statistic function
    stat_func: Callable[[np.ndarray], float] = (
        np.mean if statistic == "mean" else np.median
    )

    # Bootstrap sampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = resample(data, n_samples=n, random_state=rng.randint(0, 2**31))
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute percentiles for CI
    alpha = 1 - ci
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    return BootstrapCI(
        mean=stat_func(data),
        ci_lower=float(np.percentile(bootstrap_stats, lower_percentile)),
        ci_upper=float(np.percentile(bootstrap_stats, upper_percentile)),
        std=float(np.std(bootstrap_stats)),
        n_bootstrap=n_bootstrap,
    )


def bootstrap_delta_ci(
    data1: np.ndarray,
    data2: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> BootstrapCI:
    """Compute bootstrap CI for the difference between two paired conditions.

    Uses paired bootstrap (same indices for both arrays).

    Args:
        data1: Values from condition 1 (e.g., baseline).
        data2: Values from condition 2 (e.g., treatment).
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.
        random_state: Random seed for reproducibility.

    Returns:
        BootstrapCI for (data2 - data1).

    Raises:
        ValueError: If arrays have different lengths.

    Example:
        >>> baseline = np.array([1.0, 1.2, 1.1, 1.3, 1.4])
        >>> treatment = np.array([1.5, 1.7, 1.6, 1.8, 1.9])
        >>> delta = bootstrap_delta_ci(baseline, treatment)
        >>> print(f"Delta: {delta.mean:.2f}")
    """
    if len(data1) != len(data2):
        raise ValueError(
            f"Arrays must have same length for paired bootstrap. "
            f"Got {len(data1)} and {len(data2)}."
        )

    rng = np.random.RandomState(random_state)
    n = len(data1)

    deltas = data2 - data1
    bootstrap_deltas = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        bootstrap_deltas.append(np.mean(deltas[indices]))

    bootstrap_deltas = np.array(bootstrap_deltas)

    alpha = 1 - ci
    return BootstrapCI(
        mean=float(np.mean(deltas)),
        ci_lower=float(np.percentile(bootstrap_deltas, alpha / 2 * 100)),
        ci_upper=float(np.percentile(bootstrap_deltas, (1 - alpha / 2) * 100)),
        std=float(np.std(bootstrap_deltas)),
        n_bootstrap=n_bootstrap,
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Uses pooled standard deviation.

    Args:
        group1: Values from group 1.
        group2: Values from group 2.

    Returns:
        Cohen's d (positive if group2 > group1).

    Example:
        >>> control = np.array([1.0, 1.2, 1.1, 1.3])
        >>> treatment = np.array([1.5, 1.7, 1.6, 1.8])
        >>> d = cohens_d(control, treatment)
        >>> print(f"d = {d:.3f}")  # Large positive effect
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group2) - np.mean(group1)) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size magnitude.

    Uses conventional thresholds:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def paired_statistical_test(
    baseline: np.ndarray,
    condition: np.ndarray,
    test_type: str = "auto",
) -> PairedTestResult:
    """Perform paired statistical test.

    Args:
        baseline: Values from baseline condition.
        condition: Values from comparison condition.
        test_type: "wilcoxon", "ttest", or "auto" (checks normality).

    Returns:
        PairedTestResult with test statistics and effect size.

    Raises:
        ValueError: If arrays have different lengths.

    Example:
        >>> baseline = np.array([1.0, 1.2, 1.1, 1.3, 1.4])
        >>> treatment = np.array([1.5, 1.7, 1.6, 1.8, 1.9])
        >>> result = paired_statistical_test(baseline, treatment)
        >>> print(result)  # Shows p-value and effect size
    """
    n = len(baseline)
    if len(condition) != n:
        raise ValueError(
            f"Arrays must have same length. Got {n} and {len(condition)}."
        )

    differences = condition - baseline

    # Check normality for auto selection
    if test_type == "auto":
        if n >= 3:
            _, p_normal = stats.shapiro(differences)
            test_type = "ttest" if p_normal > 0.05 else "wilcoxon"
        else:
            test_type = "wilcoxon"

    # Perform test
    if test_type == "wilcoxon":
        try:
            statistic, p_value = stats.wilcoxon(differences, alternative="two-sided")
            test_name = "Wilcoxon signed-rank"
        except ValueError:
            # All zeros - no difference
            statistic, p_value = 0.0, 1.0
            test_name = "Wilcoxon signed-rank"
    else:
        statistic, p_value = stats.ttest_rel(condition, baseline)
        test_name = "Paired t-test"

    # Effect size
    d = cohens_d(baseline, condition)

    return PairedTestResult(
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=d,
        effect_interpretation=interpret_cohens_d(d),
        significant_005=p_value < 0.05,
        significant_001=p_value < 0.01,
        n1=n,
        n2=n,
    )


def holm_bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    The Holm-Bonferroni method is a step-down procedure that is uniformly
    more powerful than Bonferroni while still controlling FWER.

    Args:
        p_values: List of p-values.
        alpha: Family-wise error rate.

    Returns:
        List of (corrected_p, significant) tuples in original order.

    Example:
        >>> p_values = [0.01, 0.04, 0.03, 0.06]
        >>> corrected = holm_bonferroni_correction(p_values, alpha=0.05)
        >>> for p, sig in corrected:
        ...     print(f"p={p:.4f}, significant={sig}")
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    corrected = []
    for i, p in enumerate(sorted_p):
        corrected_p = p * (n - i)
        corrected.append(min(corrected_p, 1.0))

    # Ensure monotonicity
    for i in range(1, n):
        corrected[i] = max(corrected[i], corrected[i - 1])

    # Map back to original order
    result: List[Tuple[float, bool]] = [(0.0, False)] * n
    for i, orig_idx in enumerate(sorted_indices):
        result[orig_idx] = (corrected[i], corrected[i] < alpha)

    return result


def bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> List[Tuple[float, bool]]:
    """Apply Bonferroni correction for multiple comparisons.

    Simple but conservative method that multiplies each p-value
    by the number of tests.

    Args:
        p_values: List of p-values.
        alpha: Family-wise error rate.

    Returns:
        List of (corrected_p, significant) tuples.

    Example:
        >>> p_values = [0.01, 0.04, 0.03]
        >>> corrected = bonferroni_correction(p_values, alpha=0.05)
        >>> for p, sig in corrected:
        ...     print(f"p={p:.4f}, significant={sig}")
    """
    n = len(p_values)
    return [(min(p * n, 1.0), p * n < alpha) for p in p_values]


__all__ = [
    "BootstrapCI",
    "PairedTestResult",
    "StatisticalTest",  # Backward compatibility alias
    "bootstrap_ci",
    "bootstrap_delta_ci",
    "cohens_d",
    "interpret_cohens_d",
    "paired_statistical_test",
    "holm_bonferroni_correction",
    "bonferroni_correction",
]
