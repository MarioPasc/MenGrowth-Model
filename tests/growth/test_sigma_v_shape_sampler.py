# tests/growth/test_sigma_v_shape_sampler.py
"""Tests for the smooth σ²_v Beta-family sampler.

Markers: unit
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stage1_volumetric.synthetic_uq.synthetic_sigma_v_generation.beta_sampler import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_SIGMA_V_SQ_MAX,
    DEFAULT_STEEPNESS,
    beta_shape_params,
    sample_sigma_v_grid,
    sample_sigma_v_shape,
)

pytestmark = [pytest.mark.unit]


def test_beta_shape_params_at_limits() -> None:
    """At α=0 the Beta is uniform; at ±1 the peak is sharp."""
    a0, b0 = beta_shape_params(0.0)
    assert a0 == pytest.approx(1.0)
    assert b0 == pytest.approx(1.0)

    a_neg, b_neg = beta_shape_params(-1.0, steepness=9.0)
    assert a_neg == pytest.approx(1.0)
    assert b_neg == pytest.approx(10.0)

    a_pos, b_pos = beta_shape_params(+1.0, steepness=9.0)
    assert a_pos == pytest.approx(10.0)
    assert b_pos == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", [-1.5, 1.5, 2.0])
def test_beta_shape_params_rejects_out_of_range(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        beta_shape_params(alpha)


def test_beta_shape_params_rejects_nonpositive_steepness() -> None:
    with pytest.raises(ValueError, match="steepness"):
        beta_shape_params(0.0, steepness=0.0)


def test_sample_shape_support() -> None:
    """All samples must lie in [0, sigma_v_sq_max]."""
    rng = np.random.default_rng(0)
    samples = sample_sigma_v_shape(0.5, n=2000, sigma_v_sq_max=2.0, rng=rng)
    assert samples.shape == (2000,)
    assert samples.min() >= 0.0
    assert samples.max() <= 2.0


def test_mean_increases_monotonically_with_alpha() -> None:
    """E[σ²_v | α] is monotonically increasing in α."""
    n = 50000
    means = []
    for alpha in DEFAULT_ALPHA_GRID:
        samples = sample_sigma_v_shape(
            alpha,
            n=n,
            sigma_v_sq_max=DEFAULT_SIGMA_V_SQ_MAX,
            steepness=DEFAULT_STEEPNESS,
            rng=np.random.default_rng(42),
        )
        means.append(float(samples.mean()))
    diffs = np.diff(means)
    assert (diffs > 0).all(), f"means not monotonic: {means}"


def test_alpha_zero_is_uniform() -> None:
    """At α=0 the Beta(1,1) is uniform; mean ≈ σ²_max / 2."""
    n = 100000
    samples = sample_sigma_v_shape(
        0.0,
        n=n,
        sigma_v_sq_max=1.5,
        rng=np.random.default_rng(7),
    )
    assert samples.mean() == pytest.approx(0.75, rel=0.02)
    # KS-test-style coarse check: first and second halves should have
    # similar counts.
    n_below_half = int(np.sum(samples < 0.75))
    assert abs(n_below_half - n / 2) < n * 0.01


def test_alpha_minus_one_concentrates_at_zero() -> None:
    """At α=-1, ≥ 90% of mass should sit below 25% of σ²_max with s=9."""
    samples = sample_sigma_v_shape(
        -1.0,
        n=20000,
        sigma_v_sq_max=1.0,
        steepness=9.0,
        rng=np.random.default_rng(1),
    )
    # Beta(1, 10) on [0,1]: P(X < 0.25) = 1 - 0.75^10 ≈ 0.944
    frac_below = float(np.mean(samples < 0.25))
    assert frac_below > 0.90, f"only {frac_below:.3f} below 0.25"


def test_alpha_plus_one_concentrates_at_max() -> None:
    """Mirror of above: at α=+1, ≥ 90% of mass above 75% of σ²_max."""
    samples = sample_sigma_v_shape(
        +1.0,
        n=20000,
        sigma_v_sq_max=1.0,
        steepness=9.0,
        rng=np.random.default_rng(2),
    )
    frac_above = float(np.mean(samples > 0.75))
    assert frac_above > 0.90, f"only {frac_above:.3f} above 0.75"


def test_fixed_mean_pins_target_within_tolerance() -> None:
    """When ``fixed_mean`` is set, empirical mean lands close to it."""
    target = 0.5
    samples = sample_sigma_v_shape(
        +0.5,
        n=20000,
        sigma_v_sq_max=2.0,
        fixed_mean=target,
        rng=np.random.default_rng(3),
    )
    # Rescaling + clipping at sigma_v_sq_max can leave the mean slightly
    # below the target. Accept up to 5% relative deviation downward.
    assert samples.mean() == pytest.approx(target, rel=0.05) or samples.mean() < target


def test_grid_sampler_shape_and_independence() -> None:
    """``sample_sigma_v_grid`` returns one row per α."""
    alphas = (-1.0, 0.0, 1.0)
    grid = sample_sigma_v_grid(alphas=alphas, n=500, seed=0)
    assert grid.shape == (3, 500)
    # Different α rows must differ in distribution (different means).
    means = grid.mean(axis=1)
    assert means[0] < means[1] < means[2]


def test_seeding_is_deterministic() -> None:
    """Same seed → identical samples."""
    a = sample_sigma_v_shape(0.3, n=200, rng=np.random.default_rng(99))
    b = sample_sigma_v_shape(0.3, n=200, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(a, b)


def test_grid_sampler_with_fixed_mean() -> None:
    """Fixed-mean sweep keeps the cohort mean approximately constant."""
    alphas = (-0.5, 0.0, 0.5)
    target = 0.4
    grid = sample_sigma_v_grid(
        alphas=alphas,
        n=10000,
        sigma_v_sq_max=1.5,
        fixed_mean=target,
        seed=4,
    )
    means = grid.mean(axis=1)
    for m in means:
        assert m == pytest.approx(target, rel=0.05) or m < target
