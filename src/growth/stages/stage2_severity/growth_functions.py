# src/growth/stages/stage2_severity/growth_functions.py
"""Parametric growth functions for the latent severity model.

Each function maps (severity s, time t) → growth quantile q, satisfying:
- g(s, 0) = 0 for all s (boundary condition)
- ∂g/∂t ≥ 0 (monotonically increasing in time)
- ∂g/∂s ≥ 0 (monotonically increasing in severity)

Three implementations per PLAN_OF_ACTION_v1.md §2.5:
1. ReducedGompertz (RECOMMENDED): 3 population params, strongest biological grounding
2. WeightedSigmoid: 3 params, simpler but approximate monotonicity
3. CMNNGrowthFunction: ~50 params, overparameterized at N=31 (ablation only)

References:
    - Vaghi et al. (2020) PLOS Computational Biology — reduced Gompertz
    - Runje & Shankaranarayana (2023) ICML — constrained monotonic NNs
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


class GrowthFunction(ABC):
    """Abstract base class for severity-conditioned growth functions.

    All growth functions take (s, t, params) and return predicted growth q.
    """

    @abstractmethod
    def __call__(self, s: np.ndarray, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Evaluate g(s, t; θ).

        Args:
            s: Severity values, shape ``[N]`` or scalar.
            t: Time values, shape ``[N]`` or scalar.
            params: Population parameters θ, shape ``[n_params]``.

        Returns:
            Predicted growth quantile, shape ``[N]``.
        """

    @abstractmethod
    def n_params(self) -> int:
        """Number of population-level parameters."""

    @abstractmethod
    def param_bounds(self) -> list[tuple[float, float]]:
        """Parameter bounds for optimization, list of (lower, upper)."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""


class ReducedGompertz(GrowthFunction):
    """Reduced Gompertz growth function (Vaghi et al. 2020).

    V(t) = V₀ · exp(α(s)/β · (1 - exp(-β·t)))
    where α(s) = α₀ + α₁·s (growth rate increases with severity).

    In quantile space (after normalization):
    g(s, t; α₀, α₁, β) = normalize(exp(α(s)/β · (1 - exp(-β·t))) - 1)

    The boundary condition g(s, 0) = 0 is satisfied exactly because
    exp(0) - 1 = 0.

    Population parameters: [α₀, α₁, β] (3 params).
    """

    def __call__(self, s: np.ndarray, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        alpha_0, alpha_1, beta = params[0], params[1], params[2]

        alpha = alpha_0 + alpha_1 * s
        # Gompertz growth relative to baseline
        growth = np.exp(alpha / (beta + 1e-8) * (1 - np.exp(-beta * t))) - 1
        # Clip to [0, 1] for quantile space
        return np.clip(growth, 0.0, 1.0)

    def n_params(self) -> int:
        return 3

    def param_bounds(self) -> list[tuple[float, float]]:
        return [
            (0.001, 5.0),  # α₀: baseline growth rate
            (0.001, 5.0),  # α₁: severity sensitivity
            (0.001, 5.0),  # β: deceleration rate
        ]

    def name(self) -> str:
        return "ReducedGompertz"


class WeightedSigmoid(GrowthFunction):
    """Weighted sigmoid growth function.

    g(s, t; w₁, w₂, b) = t · σ(w₁·s + w₂·t + b)

    where σ is the logistic sigmoid and w₁, w₂ > 0 (constrained positive).
    The multiplication by t ensures g(s, 0) = 0 exactly.

    Monotonicity in t is approximately (but not exactly) guaranteed.

    Population parameters: [w₁, w₂, b] (3 params).
    """

    def __call__(self, s: np.ndarray, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        w1, w2, b = params[0], params[1], params[2]

        sigmoid = 1.0 / (1.0 + np.exp(-(w1 * s + w2 * t + b)))
        return np.clip(t * sigmoid, 0.0, 1.0)

    def n_params(self) -> int:
        return 3

    def param_bounds(self) -> list[tuple[float, float]]:
        return [
            (0.01, 10.0),  # w₁: severity weight (positive)
            (0.01, 10.0),  # w₂: time weight (positive)
            (-5.0, 5.0),  # b: bias
        ]

    def name(self) -> str:
        return "WeightedSigmoid"


@dataclass
class GrowthFunctionRegistry:
    """Registry of available growth functions.

    Usage::

        fn = GrowthFunctionRegistry.get("gompertz_reduced")
    """

    _registry: dict[str, type[GrowthFunction]] = None

    @classmethod
    def get(cls, name: str) -> GrowthFunction:
        """Get a growth function by name.

        Args:
            name: One of ``"gompertz_reduced"``, ``"weighted_sigmoid"``.

        Returns:
            Instantiated GrowthFunction.

        Raises:
            ValueError: If name is unknown.
        """
        registry = {
            "gompertz_reduced": ReducedGompertz,
            "weighted_sigmoid": WeightedSigmoid,
        }
        if name not in registry:
            raise ValueError(f"Unknown growth function: {name}. Available: {list(registry.keys())}")
        return registry[name]()
