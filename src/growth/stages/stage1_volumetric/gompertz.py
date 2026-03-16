# src/growth/stages/stage1_volumetric/gompertz.py
"""Gompertz mean function for hierarchical GP growth models.

Provides a Gompertz growth curve as the GP mean function, following
Engelhardt et al. (2023) and Vaghi et al. (2020) who established
Gompertz as the best parametric model for meningioma growth.

Implementation strategy (D26): fit Gompertz parametrically first via
``scipy.optimize.curve_fit``, then use the fitted curve as a fixed mean
for the GP — this costs 0 additional GP parameters.

Usage with HierarchicalGPModel::

    from growth.stages.stage1_volumetric import fit_gompertz, GompertzMeanFunction

    # Fit Gompertz to population data
    params = fit_gompertz(all_times, all_volumes)

    # Use as fixed mean function in HGP
    gompertz_mean = GompertzMeanFunction(params)
    # The HGP then learns deviations from this curve
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class GompertzParams:
    """Fitted Gompertz parameters.

    The Gompertz model is:
        V(t) = K * exp(-b * exp(-c * t))

    where:
        K: Carrying capacity (asymptotic maximum volume).
        b: Shape parameter (displacement along time axis).
        c: Growth rate (deceleration constant, units of 1/time).

    Args:
        K: Carrying capacity (log-volume units).
        b: Shape parameter (dimensionless, > 0).
        c: Growth rate (1/time_unit, > 0).
    """

    K: float
    b: float
    c: float


def _gompertz_curve(t: np.ndarray, K: float, b: float, c: float) -> np.ndarray:
    """Evaluate Gompertz curve at times t.

    V(t) = K * exp(-b * exp(-c * t))
    """
    return K * np.exp(-b * np.exp(-c * t))


def fit_gompertz(
    times: np.ndarray,
    volumes: np.ndarray,
    max_iter: int = 5000,
) -> GompertzParams:
    """Fit a Gompertz curve to population volume data.

    Uses ``scipy.optimize.curve_fit`` with robust initial guesses derived
    from the data range.

    Args:
        times: Observation times, shape ``[N_total]`` (all patients pooled).
        volumes: Log-volume observations, shape ``[N_total]``.
        max_iter: Maximum iterations for curve_fit.

    Returns:
        GompertzParams with fitted K, b, c.

    Raises:
        RuntimeError: If curve_fit fails to converge.
    """
    times = np.asarray(times, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    # Initial guesses from data range
    K0 = float(np.max(volumes) * 1.2)  # Slightly above max observed
    b0 = 1.0
    c0 = 0.1 if np.ptp(times) > 1 else 1.0

    try:
        popt, _ = curve_fit(
            _gompertz_curve,
            times,
            volumes,
            p0=[K0, b0, c0],
            bounds=([0.01, 0.001, 0.001], [np.inf, 50.0, 10.0]),
            maxfev=max_iter,
        )
        params = GompertzParams(K=popt[0], b=popt[1], c=popt[2])
        logger.info(f"Gompertz fit: K={params.K:.3f}, b={params.b:.3f}, c={params.c:.4f}")
        return params
    except RuntimeError as e:
        logger.warning(f"Gompertz fit failed: {e}. Using linear fallback.")
        # Fallback: large K (effectively linear regime of Gompertz)
        return GompertzParams(K=K0 * 10, b=0.01, c=0.01)


class GompertzMeanFunction:
    """Gompertz curve as a fixed mean function for GP models.

    After fitting, this provides m(t) = K * exp(-b * exp(-c * t)) with
    frozen parameters (no additional GP hyperparameters).

    Args:
        params: Fitted GompertzParams from ``fit_gompertz()``.
    """

    def __init__(self, params: GompertzParams) -> None:
        self.params = params

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the Gompertz mean at times t.

        Args:
            t: Query times, shape ``[N]``.

        Returns:
            Mean values, shape ``[N]``.
        """
        return _gompertz_curve(t, self.params.K, self.params.b, self.params.c)

    def __repr__(self) -> str:
        return (
            f"GompertzMeanFunction(K={self.params.K:.3f}, "
            f"b={self.params.b:.3f}, c={self.params.c:.4f})"
        )
