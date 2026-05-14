# tests/growth/test_conformal.py
"""Unit tests for the distribution-free conformal calibration core.

Synthetic-data tests with analytically known expected behaviour for the
calibrators in ``growth.shared.conformal``:

- Jackknife+ rank arithmetic and the signed/symmetric score distinction.
- Marginal coverage of split / normalised / CQR calibrators on exchangeable
  synthetic data converges to the nominal level.
- Normalised conformity collapses to constant width when ``sigma_hat`` carries
  no shape information (so it cannot do worse than split conformal).
- Beta-binomial coverage intervals at the boundaries.
"""

import numpy as np
import pytest

from growth.shared.conformal import (
    ConformalError,
    CQRCalibrator,
    NormalizedConformalCalibrator,
    SplitConformalCalibrator,
    _jackknife_plus_ranks,
    beta_binomial_coverage_ci,
    jackknife_plus_interval,
)

pytestmark = [pytest.mark.unit]


# --------------------------------------------------------------------------
# Jackknife+
# --------------------------------------------------------------------------
def test_jackknife_plus_ranks_standard_case() -> None:
    """At n=53, alpha=0.05 the ranks match floor/ceil of Barber's formula."""
    lo, hi = _jackknife_plus_ranks(53, 0.05)
    assert lo == int(np.floor(0.05 * 54)) == 2
    assert hi == int(np.ceil(0.95 * 54)) == 52


def test_jackknife_plus_ranks_clamped_at_small_n() -> None:
    """At tiny n the ranks clamp to [1, n], giving the conservative [min, max]."""
    lo, hi = _jackknife_plus_ranks(8, 0.05)
    assert lo == 1 and hi == 8


def test_jackknife_plus_symmetric_is_symmetric_about_pred() -> None:
    """With a single constant LOO prediction, the symmetric score is symmetric."""
    loo_pred = np.full(40, 3.0)
    res = np.array([0.2, -0.2, 0.5, -0.5] * 10)
    lo, hi = jackknife_plus_interval(loo_pred, res, alpha=0.05, score="symmetric")
    assert np.isclose((lo + hi) / 2.0, 3.0, atol=1e-9)


def test_jackknife_plus_signed_allows_asymmetry() -> None:
    """A skewed residual distribution yields an asymmetric signed interval."""
    rng = np.random.default_rng(0)
    loo_pred = np.full(200, 1.0)
    # Right-skewed residuals: under-prediction (positive residual) more extreme.
    res = rng.exponential(0.3, 200) - 0.1
    lo, hi = jackknife_plus_interval(loo_pred, res, alpha=0.05, score="signed")
    # Upper excursion from the centre should exceed the lower one.
    assert (hi - 1.0) > (1.0 - lo)


def test_jackknife_plus_widens_with_noise() -> None:
    """Larger residual scatter produces a wider jackknife+ interval."""
    rng = np.random.default_rng(1)
    loo_pred = np.full(300, 0.0)
    narrow = jackknife_plus_interval(loo_pred, rng.normal(0, 0.2, 300), score="signed")
    wide = jackknife_plus_interval(loo_pred, rng.normal(0, 1.0, 300), score="signed")
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_jackknife_plus_marginal_coverage_synthetic() -> None:
    """On exchangeable synthetic data, jackknife+ coverage is >= 1 - 2*alpha.

    Emulates the LOPO setting: each "calibration unit" contributes a
    leave-one-out prediction at the test point plus its own LOO residual,
    all drawn from the same exchangeable distribution as the test point.
    """
    rng = np.random.default_rng(2)
    alpha = 0.1
    n_cal = 80
    covered = []
    for _ in range(300):
        # LOO predictions at the test point: noisy estimates of the truth (0.0).
        loo_pred_at_test = rng.normal(0.0, 0.1, n_cal)
        # LOO residuals: exchangeable with the test residual.
        loo_res = rng.normal(0.0, 0.5, n_cal)
        y_test = rng.normal(0.0, 0.5)
        lo, hi = jackknife_plus_interval(loo_pred_at_test, loo_res, alpha=alpha, score="signed")
        covered.append(lo <= y_test <= hi)
    assert np.mean(covered) >= 1.0 - 2.0 * alpha


def test_jackknife_plus_shape_mismatch_raises() -> None:
    with pytest.raises(ConformalError):
        jackknife_plus_interval(np.zeros(5), np.zeros(4))


def test_jackknife_plus_unknown_score_raises() -> None:
    with pytest.raises(ConformalError):
        jackknife_plus_interval(np.zeros(5), np.zeros(5), score="bogus")


# --------------------------------------------------------------------------
# Split / normalised conformal (crepes-backed)
# --------------------------------------------------------------------------
def _linear_synthetic(rng: np.random.Generator, n: int, noise: float = 0.3):
    """Return ``(x, y, mu_hat, residuals)`` for ``y = 2 + 0.5 x + N(0, noise^2)``."""
    x = rng.uniform(0, 5, n)
    y = 2.0 + 0.5 * x + rng.normal(0, noise, n)
    mu_hat = 2.0 + 0.5 * x
    return x, y, mu_hat, y - mu_hat


def test_split_conformal_marginal_coverage() -> None:
    """Split-conformal coverage averages to the nominal level over reps."""
    covs = []
    for rep in range(40):
        rng = np.random.default_rng(rep)
        x, y, mu, res = _linear_synthetic(rng, 400)
        cal = SplitConformalCalibrator(confidence=0.9).fit(res[:250])
        iv = cal.predict_interval(mu[250:])
        covs.append(np.mean((y[250:] >= iv[:, 0]) & (y[250:] <= iv[:, 1])))
    assert abs(np.mean(covs) - 0.9) < 0.03


def test_normalized_conformity_marginal_coverage() -> None:
    """Normalised-conformity coverage averages to the nominal level over reps."""
    covs = []
    for rep in range(40):
        rng = np.random.default_rng(100 + rep)
        x, y, mu, res = _linear_synthetic(rng, 400)
        # Genuinely heteroscedastic difficulty estimate.
        sigma = 0.2 + 0.1 * x
        y = mu + rng.normal(0, 1.0, x.size) * sigma
        res = y - mu
        cal = NormalizedConformalCalibrator(confidence=0.9).fit(res[:250], sigma[:250])
        iv = cal.predict_interval(mu[250:], sigma[250:])
        covs.append(np.mean((y[250:] >= iv[:, 0]) & (y[250:] <= iv[:, 1])))
    assert abs(np.mean(covs) - 0.9) < 0.04


def test_normalized_collapses_to_constant_width_when_sigma_uninformative() -> None:
    """Constant ``sigma_hat`` => normalised conformity has constant width.

    This is the guarantee that normalised conformity cannot do worse than
    split conformal when the difficulty estimate carries no shape information.
    """
    rng = np.random.default_rng(7)
    x, y, mu, res = _linear_synthetic(rng, 300)
    sigma = np.full(300, 0.5)  # uninformative: same for every unit
    cal = NormalizedConformalCalibrator(confidence=0.95).fit(res[:200], sigma[:200])
    iv = cal.predict_interval(mu[200:], sigma[200:])
    widths = iv[:, 1] - iv[:, 0]
    assert np.allclose(widths, widths[0], atol=1e-9)


def test_y_min_y_max_clamps_degenerate_interval() -> None:
    """At a tiny calibration size crepes returns max-size intervals; bounds clamp them."""
    rng = np.random.default_rng(9)
    res = rng.normal(0, 0.3, 5)  # too few for a 95% interval
    cal = SplitConformalCalibrator(confidence=0.95).fit(res)
    unbounded = cal.predict_interval(np.array([1.0]))
    bounded = cal.predict_interval(np.array([1.0]), y_min=-3.0, y_max=4.0)
    assert not np.all(np.isfinite(unbounded))
    assert np.all(np.isfinite(bounded))
    assert bounded[0, 0] >= -3.0 - 1e-9 and bounded[0, 1] <= 4.0 + 1e-9


def test_split_conformal_empty_residuals_raises() -> None:
    with pytest.raises(ConformalError):
        SplitConformalCalibrator().fit(np.array([]))


# --------------------------------------------------------------------------
# CQR
# --------------------------------------------------------------------------
def test_cqr_marginal_coverage() -> None:
    """Conformalised quantile regression coverage is close to the nominal level."""
    covs = []
    for rep in range(30):
        rng = np.random.default_rng(200 + rep)
        x = rng.uniform(0, 5, 400)
        y = 2.0 + 0.5 * x + rng.normal(0, 0.2 + 0.1 * x)
        cal = CQRCalibrator(alpha=0.1, seed=rep).fit(x[:250].reshape(-1, 1), y[:250])
        iv = cal.predict_interval(x[250:].reshape(-1, 1))
        covs.append(np.mean((y[250:] >= iv[:, 0]) & (y[250:] <= iv[:, 1])))
    # CQR is known to be slightly anti-conservative at finite calibration sizes.
    assert 0.85 < np.mean(covs) < 0.97


def test_cqr_invalid_calib_fraction_raises() -> None:
    with pytest.raises(ConformalError):
        CQRCalibrator(calib_fraction=1.5)


def test_cqr_too_few_units_raises() -> None:
    cal = CQRCalibrator()
    with pytest.raises(ConformalError):
        cal.fit(np.zeros((3, 1)), np.zeros(3))


def test_cqr_predict_before_fit_raises() -> None:
    with pytest.raises(ConformalError):
        CQRCalibrator().predict_interval(np.zeros((2, 1)))


# --------------------------------------------------------------------------
# Beta-binomial coverage CI
# --------------------------------------------------------------------------
def test_beta_binomial_ci_contains_point_estimate() -> None:
    lo, hi = beta_binomial_coverage_ci(51, 54, confidence=0.95)
    assert lo < 51 / 54 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_beta_binomial_ci_boundaries() -> None:
    """All-covered / none-covered pin one endpoint to the boundary."""
    lo_full, hi_full = beta_binomial_coverage_ci(54, 54)
    assert hi_full == 1.0 and lo_full < 1.0
    lo_none, hi_none = beta_binomial_coverage_ci(0, 54)
    assert lo_none == 0.0 and hi_none > 0.0


def test_beta_binomial_ci_invalid_args_raise() -> None:
    with pytest.raises(ConformalError):
        beta_binomial_coverage_ci(10, 0)
    with pytest.raises(ConformalError):
        beta_binomial_coverage_ci(60, 54)
