"""Smooth σ²_v shape-shift family for the Stage 1 UQ stress test.

Implements a Beta-distributed σ²_v generator parameterised by a single
shape knob α ∈ [-1, +1] that interpolates from "all mass at zero" (α=-1)
through uniform on [0, σ²_max] (α=0) to "all mass at high variance"
(α=+1). See ``docs/deprecated/UQ_PRED/UQ_SMOOTH_SHIFT_STRESSTEST.md``.
"""

from .beta_sampler import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_SIGMA_V_SQ_MAX,
    DEFAULT_STEEPNESS,
    beta_shape_params,
    sample_sigma_v_shape,
    sample_sigma_v_grid,
)

__all__ = [
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_SIGMA_V_SQ_MAX",
    "DEFAULT_STEEPNESS",
    "beta_shape_params",
    "sample_sigma_v_shape",
    "sample_sigma_v_grid",
]
