# tests/growth/test_metrics_dss.py
"""Tests for Dawid-Sebastiani Score, NLPD, and PIT calibration metrics."""

import numpy as np
import pytest
from scipy.stats import norm

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


class TestDawidSebastiani:
    """Tests for compute_dawid_sebastiani."""

    def test_dss_at_mean_unit_variance(self) -> None:
        """When y == mu and sigma_sq == 1, DSS = log(1) = 0."""
        from growth.shared.metrics import compute_dawid_sebastiani

        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        sigma_sq = np.ones(3)
        assert compute_dawid_sebastiani(y, mu, sigma_sq) == pytest.approx(0.0, abs=1e-10)

    def test_dss_penalises_overconfidence(self) -> None:
        """Smaller sigma_sq with same residual -> higher DSS (worse)."""
        from growth.shared.metrics import compute_dawid_sebastiani

        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([0.5, 1.5, 2.5])

        dss_wide = compute_dawid_sebastiani(y, mu, np.ones(3) * 4.0)
        dss_narrow = compute_dawid_sebastiani(y, mu, np.ones(3) * 0.01)

        assert dss_narrow > dss_wide

    def test_dss_penalises_underconfidence(self) -> None:
        """At the mean (no residual), larger sigma_sq -> higher DSS due to log(sigma_sq) term."""
        from growth.shared.metrics import compute_dawid_sebastiani

        y = np.array([1.0, 2.0, 3.0])
        mu = y.copy()

        dss_small = compute_dawid_sebastiani(y, mu, np.ones(3) * 0.1)
        dss_large = compute_dawid_sebastiani(y, mu, np.ones(3) * 100.0)

        assert dss_large > dss_small

    def test_dss_strictly_proper(self) -> None:
        """DSS is minimised when predictive variance equals squared error + noise."""
        from growth.shared.metrics import compute_dawid_sebastiani

        rng = np.random.default_rng(42)
        n = 5000
        true_sigma = 1.5
        mu = rng.normal(0, 3, size=n)
        y = mu + rng.normal(0, true_sigma, size=n)

        sigma_sq_true = np.ones(n) * true_sigma**2
        sigma_sq_too_small = np.ones(n) * 0.1
        sigma_sq_too_large = np.ones(n) * 100.0

        dss_true = compute_dawid_sebastiani(y, mu, sigma_sq_true)
        dss_small = compute_dawid_sebastiani(y, mu, sigma_sq_too_small)
        dss_large = compute_dawid_sebastiani(y, mu, sigma_sq_too_large)

        assert dss_true < dss_small
        assert dss_true < dss_large

    def test_dss_scalar_inputs(self) -> None:
        """DSS works with scalar inputs."""
        from growth.shared.metrics import compute_dawid_sebastiani

        result = compute_dawid_sebastiani(np.array([1.0]), np.array([0.0]), np.array([1.0]))
        expected = 1.0 + 0.0  # (1-0)^2/1 + log(1)
        assert result == pytest.approx(expected, abs=1e-10)


class TestLogScore:
    """Tests for compute_log_score (NLPD)."""

    def test_nlpd_equals_dss_plus_half_log2pi(self) -> None:
        """NLPD = 0.5*(DSS + log(2*pi))."""
        from growth.shared.metrics import compute_dawid_sebastiani, compute_log_score

        rng = np.random.default_rng(42)
        y = rng.normal(0, 2, size=50)
        mu = rng.normal(0, 1, size=50)
        sigma_sq = rng.uniform(0.1, 5, size=50)

        dss = compute_dawid_sebastiani(y, mu, sigma_sq)
        nlpd = compute_log_score(y, mu, sigma_sq)

        expected = 0.5 * (dss + np.log(2 * np.pi))
        assert nlpd == pytest.approx(expected, rel=1e-10)

    def test_nlpd_matches_scipy_logpdf(self) -> None:
        """NLPD should equal mean(-log p(y|mu, sigma))."""
        from growth.shared.metrics import compute_log_score

        rng = np.random.default_rng(42)
        y = rng.normal(0, 2, size=100)
        mu = rng.normal(0, 1, size=100)
        sigma = rng.uniform(0.5, 3, size=100)

        nlpd = compute_log_score(y, mu, sigma**2)
        expected = -np.mean(norm.logpdf(y, loc=mu, scale=sigma))

        assert nlpd == pytest.approx(expected, rel=1e-10)

    def test_nlpd_rank_equivalent_to_dss(self) -> None:
        """NLPD ordering matches DSS ordering across model variants."""
        from growth.shared.metrics import compute_dawid_sebastiani, compute_log_score

        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, size=30)
        mu = rng.normal(0, 0.5, size=30)

        sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
        dss_vals = [compute_dawid_sebastiani(y, mu, np.ones(30) * s**2) for s in sigmas]
        nlpd_vals = [compute_log_score(y, mu, np.ones(30) * s**2) for s in sigmas]

        dss_order = np.argsort(dss_vals)
        nlpd_order = np.argsort(nlpd_vals)
        np.testing.assert_array_equal(dss_order, nlpd_order)


class TestPIT:
    """Tests for compute_pit."""

    def test_pit_uniform_for_calibrated(self) -> None:
        """PIT values from true model should pass KS uniformity test."""
        from growth.shared.metrics import compute_pit

        rng = np.random.default_rng(42)
        n = 10000
        sigma = 1.5
        mu = rng.normal(0, 3, size=n)
        y = mu + rng.normal(0, sigma, size=n)

        pit = compute_pit(y, mu, np.ones(n) * sigma)

        from scipy.stats import kstest

        _, p_value = kstest(pit, "uniform")
        assert p_value > 0.05

    def test_pit_range_01(self) -> None:
        """PIT values should be in [0, 1]."""
        from growth.shared.metrics import compute_pit

        rng = np.random.default_rng(42)
        y = rng.normal(0, 5, size=100)
        mu = np.zeros(100)
        sigma = np.ones(100)

        pit = compute_pit(y, mu, sigma)
        assert np.all(pit >= 0.0)
        assert np.all(pit <= 1.0)

    def test_pit_at_mean_is_half(self) -> None:
        """PIT(y=mu) = Phi(0) = 0.5."""
        from growth.shared.metrics import compute_pit

        y = np.array([1.0, 2.0, 3.0])
        mu = y.copy()
        sigma = np.ones(3) * 2.0

        pit = compute_pit(y, mu, sigma)
        np.testing.assert_allclose(pit, 0.5, atol=1e-10)


class TestPITHistogram:
    """Tests for compute_pit_histogram."""

    def test_histogram_bins(self) -> None:
        """Histogram should have correct number of bins."""
        from growth.shared.metrics import compute_pit_histogram

        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=1000)
        result = compute_pit_histogram(pit, n_bins=10)

        assert len(result["counts"]) == 10
        assert len(result["bin_edges"]) == 11

    def test_histogram_ks_for_uniform(self) -> None:
        """KS p-value should be high for uniform PIT values."""
        from growth.shared.metrics import compute_pit_histogram

        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=5000)
        result = compute_pit_histogram(pit)

        assert result["ks_p"] > 0.05

    def test_histogram_ks_for_biased(self) -> None:
        """KS p-value should be low for non-uniform PIT values."""
        from growth.shared.metrics import compute_pit_histogram

        rng = np.random.default_rng(42)
        pit = rng.beta(0.5, 0.5, size=1000)  # U-shaped = overconfident
        result = compute_pit_histogram(pit)

        assert result["ks_p"] < 0.05
