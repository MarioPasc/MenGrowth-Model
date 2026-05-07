"""Tests for the empirical-shift σ²_v sampler."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stage1_volumetric.main_experiment.modules.sigma_v_generators import (
    build_tau_grid,
    compute_tau_endpoints,
    sample_beta_alpha,
    sample_shifted_empirical,
)

pytestmark = [pytest.mark.unit]


def _toy_log_emp(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bulk = rng.normal(loc=-7.0, scale=1.0, size=int(0.85 * n))
    tail = rng.normal(loc=0.0, scale=0.6, size=n - int(0.85 * n))
    return np.concatenate([bulk, tail])


def test_shape_and_positive():
    log_emp = _toy_log_emp()
    rng = np.random.default_rng(1)
    s = sample_shifted_empirical(0.0, 500, log_emp, rng, sigma_v_sq_floor=1e-12)
    assert s.shape == (500,)
    assert (s > 0).all()


def test_tau_zero_matches_empirical_in_distribution():
    log_emp = _toy_log_emp(n=500, seed=2)
    rng = np.random.default_rng(2)
    s = sample_shifted_empirical(0.0, 5000, log_emp, rng, sigma_v_sq_floor=1e-12)
    # Means in log-space should agree to within MC noise (~3 sigma / sqrt(n))
    diff = abs(np.mean(np.log(s)) - np.mean(log_emp))
    assert diff < 0.1


def test_tau_positive_shifts_up():
    log_emp = _toy_log_emp(n=500, seed=3)
    rng = np.random.default_rng(3)
    s_zero = sample_shifted_empirical(0.0, 5000, log_emp, rng, sigma_v_sq_floor=1e-12)
    rng = np.random.default_rng(3)
    s_plus = sample_shifted_empirical(2.0, 5000, log_emp, rng, sigma_v_sq_floor=1e-12)
    assert np.mean(np.log(s_plus)) - np.mean(np.log(s_zero)) == pytest.approx(2.0, abs=0.05)


def test_tau_negative_saturates_at_floor():
    log_emp = _toy_log_emp(n=500, seed=4)
    rng = np.random.default_rng(4)
    s = sample_shifted_empirical(-50.0, 1000, log_emp, rng, sigma_v_sq_floor=1e-3)
    assert np.allclose(s, 1e-3)


def test_tau_positive_saturates_at_ceil():
    log_emp = _toy_log_emp(n=500, seed=5)
    rng = np.random.default_rng(5)
    s = sample_shifted_empirical(
        +50.0, 1000, log_emp, rng, sigma_v_sq_floor=1e-3, sigma_v_sq_ceil=10.0
    )
    assert np.allclose(s, 10.0)


def test_endpoints_anchor_correctly():
    log_emp = _toy_log_emp(n=500, seed=6)
    tau_min, tau_max = compute_tau_endpoints(
        log_emp, sigma_v_sq_floor=1e-3, sigma_v_sq_ceil=50.0, safety_margin=2.0
    )
    p95 = float(np.percentile(log_emp, 95))
    p5 = float(np.percentile(log_emp, 5))
    assert tau_min < np.log(1e-3) - p95
    assert tau_max > np.log(50.0) - p5


def test_grid_includes_zero():
    grid = build_tau_grid(7, tau_min=-6.0, tau_max=8.0, include_zero=True)
    assert grid[0] == pytest.approx(-6.0)
    assert grid[-1] == pytest.approx(8.0)
    assert np.any(np.isclose(grid, 0.0))


def test_grid_sorted_and_unique():
    grid = build_tau_grid(11, tau_min=-5.0, tau_max=5.0)
    assert np.all(np.diff(grid) >= 0)
    assert len(np.unique(grid)) == len(grid)


def test_beta_alpha_dispatch():
    rng = np.random.default_rng(0)
    s = sample_beta_alpha(alpha=0.0, n=500, rng=rng, sigma_v_sq_max=1.5, steepness=9.0)
    assert s.shape == (500,)
    assert np.all(s >= 0) and np.all(s <= 1.5 + 1e-9)
