# src/growth/stages/stage2_severity/__init__.py
"""Stage 2: Latent severity model (SECONDARY).

Implements a nonlinear mixed-effects (NLME) model where a single latent
severity variable s_i in [0, 1] per patient governs growth trajectories.
Formally connected to Item Response Theory and reduced Gompertz dynamics.

Two estimation methods:
- **MLE** (Option A): Joint optimization via L-BFGS-B (``SeverityModel``)
- **Bayesian** (Option B): MCMC via numpyro (``BayesianSeverityModel``)

Both share growth functions, quantile transform, and severity regression head.

Spec: ``docs/stages/stage_2_severity_model.md``
"""

from growth.stages.stage2_severity.growth_functions import (
    GrowthFunction,
    GrowthFunctionRegistry,
    ReducedGompertz,
    WeightedSigmoid,
)
from growth.stages.stage2_severity.quantile_transform import (
    QuantileTransform,
    QuantileTransformResult,
)
from growth.stages.stage2_severity.severity_model import (
    SeverityFitResult,
    SeverityModel,
)
from growth.stages.stage2_severity.severity_regression import SeverityRegressionHead

# Bayesian model is optional (requires numpyro + jax)
try:
    from growth.stages.stage2_severity.bayesian_severity_model import (
        BayesianSeverityModel,
    )

    _BAYESIAN_AVAILABLE = True
except ImportError:
    _BAYESIAN_AVAILABLE = False

__all__ = [
    "GrowthFunction",
    "GrowthFunctionRegistry",
    "QuantileTransform",
    "QuantileTransformResult",
    "ReducedGompertz",
    "SeverityFitResult",
    "SeverityModel",
    "SeverityRegressionHead",
    "WeightedSigmoid",
]

if _BAYESIAN_AVAILABLE:
    __all__.append("BayesianSeverityModel")
