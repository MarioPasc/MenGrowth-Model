"""Beta-family σ²_v sampler with a single shape knob α ∈ [-1, +1].

For each α we draw $u_k \\sim \\mathrm{Beta}(a(\\alpha), b(\\alpha))$ on
$[0, 1]$ and rescale to $\\sigma^2_{v,k} = \\sigma^2_\\max \\cdot u_k$.
The mapping (a(α), b(α)) is

    a(α) = 1 + s · max(α, 0),    b(α) = 1 + s · max(-α, 0),

so that

    α = -1  →  Beta(1, 1+s)        : peak at 0
    α =  0  →  Beta(1, 1)          : uniform on [0, σ²_max]
    α = +1  →  Beta(1+s, 1)        : peak at σ²_max

with steepness s controlling how sharp the peaks are at |α|=1. The
mean varies smoothly with α: at α=±1 it is at the support edges,
and at α=0 it is σ²_max/2.

This is the design specified in
``docs/deprecated/UQ_PRED/UQ_SMOOTH_SHIFT_STRESSTEST.md`` §1.

References:
    Gelman et al., "Bayesian Data Analysis" (3rd ed.), Ch. 3, 2014 —
    Beta family as a one-parameter flexible prior on bounded support.
"""

from __future__ import annotations

import numpy as np

# Default sweep grid: 9 levels symmetric around 0 with 0.25 spacing.
DEFAULT_ALPHA_GRID: tuple[float, ...] = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)

# σ²_v_max: empirical p95 of the M=20 LoRA ensemble σ²_v on MenGrowth.h5
# (≈ 1.5 in log-volume units²). Excludes the 6% segmentation-failure tail
# while covering the full plausible range of measurement noise.
DEFAULT_SIGMA_V_SQ_MAX: float = 1.5

# Steepness: at |α|=1, Beta(1, 1+s) has its 90th percentile at
# 1 - 0.1**(1/(1+s)), i.e. for s=9 ⇒ ~22% of σ²_max. Larger s ⇒ tighter
# peaks at the support edges.
DEFAULT_STEEPNESS: float = 9.0


def beta_shape_params(alpha: float, steepness: float = DEFAULT_STEEPNESS) -> tuple[float, float]:
    """Return Beta(a, b) shape parameters for the shape knob α.

    Args:
        alpha: Shape knob in [-1, +1]. Negative ⇒ peak near 0,
            zero ⇒ uniform, positive ⇒ peak near 1.
        steepness: Sharpness of the peaks at |α|=1. Must be > 0.

    Returns:
        Tuple ``(a, b)`` of Beta shape parameters.
    """
    if not -1.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [-1, 1], got {alpha}")
    if steepness <= 0.0:
        raise ValueError(f"steepness must be > 0, got {steepness}")

    a = 1.0 + steepness * max(alpha, 0.0)
    b = 1.0 + steepness * max(-alpha, 0.0)
    return float(a), float(b)


def sample_sigma_v_shape(
    alpha: float,
    n: int,
    *,
    sigma_v_sq_max: float = DEFAULT_SIGMA_V_SQ_MAX,
    steepness: float = DEFAULT_STEEPNESS,
    fixed_mean: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Draw `n` σ²_v samples from the Beta family at shape α.

    Args:
        alpha: Shape knob in [-1, +1].
        n: Number of samples (= n_scans in the cohort).
        sigma_v_sq_max: Upper bound of the σ²_v support.
        steepness: Sharpness of the peaks at |α|=1.
        fixed_mean: If set, rescale the samples so their empirical
            mean equals this value (within the [0, σ²_max] support;
            clipped at sigma_v_sq_max). Use ``None`` for the natural
            free-mean sweep.
        rng: ``numpy.random.Generator``, integer seed, or ``None``.

    Returns:
        Array of σ²_v values, shape ``(n,)``, dtype float64.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if sigma_v_sq_max <= 0.0:
        raise ValueError(f"sigma_v_sq_max must be > 0, got {sigma_v_sq_max}")

    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(rng)

    a, b = beta_shape_params(alpha, steepness)
    u = rng.beta(a, b, size=n)
    sigma_v_sq = sigma_v_sq_max * u

    if fixed_mean is not None:
        if fixed_mean <= 0.0:
            raise ValueError(f"fixed_mean must be > 0, got {fixed_mean}")
        if fixed_mean >= sigma_v_sq_max:
            raise ValueError(
                f"fixed_mean ({fixed_mean}) must be strictly less than "
                f"sigma_v_sq_max ({sigma_v_sq_max})."
            )
        cur = float(np.mean(sigma_v_sq))
        if cur > 0.0:
            sigma_v_sq = sigma_v_sq * (fixed_mean / cur)
            sigma_v_sq = np.minimum(sigma_v_sq, sigma_v_sq_max)

    return sigma_v_sq.astype(np.float64)


def sample_sigma_v_grid(
    alphas: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    n: int = 1000,
    *,
    sigma_v_sq_max: float = DEFAULT_SIGMA_V_SQ_MAX,
    steepness: float = DEFAULT_STEEPNESS,
    fixed_mean: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample σ²_v values for every α in a grid.

    Returns a 2-D array of shape ``(len(alphas), n)`` whose row k holds
    `n` independent draws from the Beta family at α=alphas[k].

    Args:
        alphas: Sequence of shape knobs in [-1, +1].
        n: Samples per α.
        sigma_v_sq_max: Upper bound of the support.
        steepness: Peak sharpness.
        fixed_mean: Optional mean pinning.
        seed: Master seed; per-α RNGs spawn deterministically from it.

    Returns:
        ``(len(alphas), n)`` float64 array.
    """
    master = np.random.default_rng(seed)
    streams = master.spawn(len(alphas))
    grid = np.empty((len(alphas), n), dtype=np.float64)
    for k, (a_k, rng_k) in enumerate(zip(alphas, streams, strict=True)):
        grid[k] = sample_sigma_v_shape(
            a_k,
            n,
            sigma_v_sq_max=sigma_v_sq_max,
            steepness=steepness,
            fixed_mean=fixed_mean,
            rng=rng_k,
        )
    return grid
