# src/growth/stages/stage2_severity/__init__.py
"""Stage 2: Latent severity model (SECONDARY).

Implements a nonlinear mixed-effects (NLME) model where a single latent
severity variable s_i ∈ [0, 1] per patient governs growth trajectories.
Formally connected to Item Response Theory and reduced Gompertz dynamics.

Spec: ``docs/stages/stage_2_severity_model.md``
"""

from growth.stages.stage2_severity.growth_functions import (
    ReducedGompertz,
    WeightedSigmoid,
)
from growth.stages.stage2_severity.quantile_transform import QuantileTransform
from growth.stages.stage2_severity.severity_model import SeverityModel

__all__ = [
    "QuantileTransform",
    "ReducedGompertz",
    "SeverityModel",
    "WeightedSigmoid",
]
