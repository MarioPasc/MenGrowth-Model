# tests/growth/test_metrics_crps.py
"""Tests for CRPS, coverage, and interval score metrics.

Markers: phase4, unit
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


class TestCRPSGaussian:
    """Tests for compute_crps_gaussian."""

    def test_crps_at_mean_equals_expected(self) -> None:
        """CRPS(N(mu, sigma^2), mu) = sigma * [2*phi(0) - 1/sqrt(pi)]."""
        from scipy.stats import norm

        from growth.shared.metrics import compute_crps_gaussian

        mu = np.array([5.0, 10.0])
        sigma = np.array([1.0, 2.0])
        y = mu.copy()

        crps = compute_crps_gaussian(y, mu, sigma)
        # omega=0: CRPS = sigma * (2*phi(0) - 1/sqrt(pi))
        per_crps = sigma * (2.0 * norm.pdf(0.0) - 1.0 / np.sqrt(np.pi))
        expected = float(np.mean(per_crps))
        np.testing.assert_allclose(crps, expected, rtol=1e-6)

    def test_crps_reduces_to_mae_at_zero_variance(self) -> None:
        """With sigma -> 0, CRPS -> |y - mu|."""
        from growth.shared.metrics import compute_crps_gaussian

        y = np.array([1.0, 3.0, 5.0])
        mu = np.array([2.0, 2.0, 4.0])
        sigma = np.full(3, 1e-12)

        crps = compute_crps_gaussian(y, mu, sigma)
        expected_mae = float(np.mean(np.abs(y - mu)))
        np.testing.assert_allclose(crps, expected_mae, atol=1e-6)

    def test_crps_numerical_matches_quadrature(self) -> None:
        """Check closed-form against numerical integration for one sample."""
        from scipy.integrate import quad
        from scipy.stats import norm

        from growth.shared.metrics import compute_crps_gaussian

        y_val = 3.0
        mu_val = 2.0
        sigma_val = 1.5

        def crps_integrand(x: float) -> float:
            return (norm.cdf(x, mu_val, sigma_val) - (1.0 if x >= y_val else 0.0)) ** 2

        numerical, _ = quad(crps_integrand, mu_val - 10 * sigma_val, mu_val + 10 * sigma_val)
        analytical = compute_crps_gaussian(
            np.array([y_val]), np.array([mu_val]), np.array([sigma_val])
        )
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5)


class TestCoverageAtLevels:
    """Tests for compute_coverage_at_levels."""

    def test_coverage_at_known_levels(self) -> None:
        """Large Gaussian sample: empirical coverage ≈ nominal."""
        from growth.shared.metrics import compute_coverage_at_levels

        rng = np.random.default_rng(42)
        n = 10000
        mu = np.zeros(n)
        sigma = np.ones(n)
        y = rng.normal(mu, sigma)

        coverage = compute_coverage_at_levels(y, mu, sigma)
        for level, cov in coverage.items():
            np.testing.assert_allclose(cov, level, atol=0.02)


class TestIntervalScore:
    """Tests for compute_interval_score."""

    def test_inside_interval_equals_width(self) -> None:
        """When y is inside [l, u], IS = width only."""
        from growth.shared.metrics import compute_interval_score

        y = np.array([5.0])
        lower = np.array([3.0])
        upper = np.array([7.0])

        is_val = compute_interval_score(y, lower, upper, alpha=0.05)
        np.testing.assert_allclose(is_val, 4.0, rtol=1e-10)

    def test_outside_interval_adds_penalty(self) -> None:
        """When y is below l, IS = width + (2/alpha)*(l-y)."""
        from growth.shared.metrics import compute_interval_score

        y = np.array([1.0])
        lower = np.array([3.0])
        upper = np.array([7.0])
        alpha = 0.05

        expected = 4.0 + (2.0 / alpha) * (3.0 - 1.0)
        is_val = compute_interval_score(y, lower, upper, alpha=alpha)
        np.testing.assert_allclose(is_val, expected, rtol=1e-10)
