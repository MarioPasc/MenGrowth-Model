"""σ²_v samplers for the main experiment τ-shift sweep.

The primary sweep is a one-parameter family $\\{P_\\tau\\}$ obtained by
bootstrapping the empirical post-QC log σ²_v vector and adding a global
log-space shift τ:

.. math::
   \\sigma^2_{v,k}(\\tau) = \\exp\\!\\bigl(L_k + \\tau\\bigr),
   \\qquad L_k \\stackrel{\\text{iid}}{\\sim} \\widehat{F}_{\\log\\sigma^2_v}^{\\text{post-QC}}.

This preserves the empirical distribution shape exactly. By construction:

* τ = 0 ⇒ exact empirical distribution (sanity-match cell, deterministic up
  to bootstrap variance).
* τ → −∞ ⇒ all σ²_v collapse to the numerical floor — saturated "ensemble
  confident on every scan".
* τ → +∞ ⇒ all σ²_v exceed the signal-variance ceiling — saturated
  "ensemble uncertain on every scan".

The Beta(α) sampler is retained as a secondary ablation only; the LogNormal
mixture sampler used in earlier iterations is retired.
"""

from __future__ import annotations

import numpy as np

from experiments.stage1_volumetric.synthetic_uq.synthetic_sigma_v_generation.beta_sampler import (
    sample_sigma_v_shape as _beta_sample,
)


def sample_shifted_empirical(
    tau: float,
    n: int,
    log_empirical_sigma_v_sq: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_v_sq_floor: float = 1e-12,
    sigma_v_sq_ceil: float | None = None,
) -> np.ndarray:
    """Draw n σ²_v values from the empirical-shape distribution shifted by τ.

    Args:
        tau: Log-space shift. ``tau=0`` reproduces the empirical distribution.
        n: Number of samples (= n_scans in the cohort).
        log_empirical_sigma_v_sq: 1-D array of post-QC ``log σ²_v`` values
            from the actual cohort. Bootstrap source.
        rng: NumPy generator for reproducibility.
        sigma_v_sq_floor: Lower clip applied after the shift. Should match
            ``cfg.uncertainty.floor_variance`` so the saturated-low extreme
            collapses to the same value LMEHetero@σ²_v=floor uses.
        sigma_v_sq_ceil: Optional upper clip. ``None`` disables.

    Returns:
        ``np.ndarray`` of shape ``(n,)``, σ²_v values strictly positive and
        respecting the floor/ceiling.
    """
    base = np.asarray(log_empirical_sigma_v_sq, dtype=np.float64).ravel()
    if base.size == 0:
        raise ValueError("log_empirical_sigma_v_sq must be non-empty")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    drawn_log = rng.choice(base, size=n, replace=True) + float(tau)
    drawn = np.exp(drawn_log)
    drawn = np.maximum(drawn, float(sigma_v_sq_floor))
    if sigma_v_sq_ceil is not None:
        drawn = np.minimum(drawn, float(sigma_v_sq_ceil))
    return drawn.astype(np.float64)


def compute_tau_endpoints(
    log_empirical_sigma_v_sq: np.ndarray,
    *,
    sigma_v_sq_floor: float,
    sigma_v_sq_ceil: float,
    safety_margin: float = 2.0,
) -> tuple[float, float]:
    """Anchor τ_min/τ_max to the noise floor and signal-variance ceiling.

    τ_min is chosen so the empirical p95 lands at or below the floor (then a
    safety margin is added for robustness). τ_max is chosen so the empirical
    p5 lands at or above the ceiling.

    Args:
        log_empirical_sigma_v_sq: Empirical log σ²_v values.
        sigma_v_sq_floor: Numerical floor (≈ floor_variance).
        sigma_v_sq_ceil: Variance-scale ceiling (≈ signal random-slope variance).
        safety_margin: Extra log-units beyond the saturation point.

    Returns:
        ``(tau_min, tau_max)`` such that ``sample_shifted_empirical`` at the
        endpoints saturates at the floor / ceiling respectively.
    """
    if sigma_v_sq_floor <= 0 or sigma_v_sq_ceil <= 0:
        raise ValueError("floor and ceiling must be positive")
    if sigma_v_sq_ceil <= sigma_v_sq_floor:
        raise ValueError("ceiling must exceed floor")

    p5 = float(np.percentile(log_empirical_sigma_v_sq, 5))
    p95 = float(np.percentile(log_empirical_sigma_v_sq, 95))

    tau_min = float(np.log(sigma_v_sq_floor) - p95 - safety_margin)
    tau_max = float(np.log(sigma_v_sq_ceil) - p5 + safety_margin)
    return tau_min, tau_max


def build_tau_grid(
    n_levels: int,
    tau_min: float,
    tau_max: float,
    *,
    include_zero: bool = True,
) -> np.ndarray:
    """Build a τ-grid that includes 0 (the empirical-match cell) by construction.

    Args:
        n_levels: Total number of τ values (must be ≥ 3 if include_zero).
        tau_min: Saturation-low endpoint.
        tau_max: Saturation-high endpoint.
        include_zero: Force τ=0 to appear exactly in the grid.

    Returns:
        Sorted ``np.ndarray`` of τ values.
    """
    if n_levels < 2:
        raise ValueError("n_levels must be ≥ 2")

    grid = np.linspace(tau_min, tau_max, n_levels)
    if include_zero and not np.any(np.isclose(grid, 0.0, atol=1e-9)):
        # Snap the closest grid point to exactly 0.
        idx = int(np.argmin(np.abs(grid)))
        grid = grid.copy()
        grid[idx] = 0.0
        grid = np.sort(grid)
    return grid


def sample_beta_alpha(
    alpha: float,
    n: int,
    rng: np.random.Generator,
    *,
    sigma_v_sq_max: float = 1.5,
    steepness: float = 9.0,
) -> np.ndarray:
    """Wrapper around the existing Beta(α) sampler for the ablation arm."""
    return _beta_sample(
        alpha=alpha,
        n=n,
        sigma_v_sq_max=sigma_v_sq_max,
        steepness=steepness,
        rng=rng,
    )
