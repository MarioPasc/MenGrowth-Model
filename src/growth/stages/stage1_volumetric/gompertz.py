# src/growth/stages/stage1_volumetric/gompertz.py
"""Gompertz mean function for hierarchical GP growth models.

Provides a Gompertz growth curve as the GP mean function, following
Engelhardt et al. (2023) and Vaghi et al. (2020) who established
Gompertz as the best parametric model for meningioma growth.

Implementation strategy (D26): fit Gompertz parametrically first via
``scipy.optimize.curve_fit``, then use the fitted curve as a fixed mean
for the GP — this costs 0 additional GP parameters.

Note: With ordinal time indices and pooled heterogeneous volumes,
Gompertz fitting is unreliable.  This is expected — Gompertz models
individual growth curves, not cross-sectional volume distributions.
The fallback returns a linear fit (OLS) as the population mean.

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

    If ``converged`` is False, the parameters represent a linear fallback
    fitted via OLS: m(t) = beta_0 + beta_1 * t, encoded as
    K = beta_0 + beta_1 * t_max (approximately).

    Args:
        K: Carrying capacity (log-volume units).
        b: Shape parameter (dimensionless, > 0).
        c: Growth rate (1/time_unit, > 0).
        converged: Whether Gompertz curve_fit converged.
    """

    K: float
    b: float
    c: float
    converged: bool = True


def _gompertz_curve(t: np.ndarray, K: float, b: float, c: float) -> np.ndarray:
    """Evaluate Gompertz curve at times t.

    V(t) = K * exp(-b * exp(-c * t))
    """
    return K * np.exp(-b * np.exp(-c * t))


def _linear_fallback(times: np.ndarray, volumes: np.ndarray) -> GompertzParams:
    """Fit OLS linear fallback and encode as near-linear Gompertz.

    Returns Gompertz parameters where the curve approximates a line
    through the data.  This is used when Gompertz fitting fails.
    """
    # OLS: y = beta_0 + beta_1 * t
    n = len(times)
    t_mean = np.mean(times)
    y_mean = np.mean(volumes)

    ss_tt = np.sum((times - t_mean) ** 2)
    if ss_tt > 1e-15:
        beta_1 = float(np.sum((times - t_mean) * (volumes - y_mean)) / ss_tt)
    else:
        beta_1 = 0.0
    beta_0 = float(y_mean - beta_1 * t_mean)

    # Encode as Gompertz with very small b (near-linear approximation):
    # K ~ max_value * 1.1, b ~ tiny, c ~ tiny
    # This makes _gompertz_curve(t) ≈ K * (1 - b + b*c*t) ≈ K - K*b + K*b*c*t
    # We set: K*(1-b) = beta_0, K*b*c = beta_1
    K = max(float(np.max(volumes)) * 1.1, abs(beta_0) + 1.0)
    b = max((K - beta_0) / K, 0.001)
    c = max(beta_1 / (K * b), 0.001) if K * b > 1e-10 else 0.001

    logger.info(
        f"Gompertz linear fallback: beta_0={beta_0:.3f}, beta_1={beta_1:.4f} "
        f"-> K={K:.3f}, b={b:.4f}, c={c:.4f}"
    )
    return GompertzParams(K=K, b=b, c=c, converged=False)


def fit_gompertz(
    times: np.ndarray,
    volumes: np.ndarray,
    max_iter: int = 5000,
) -> GompertzParams:
    """Fit a Gompertz curve to population volume data.

    Uses ``scipy.optimize.curve_fit`` with multiple restarts using
    different initial guesses.  Falls back to OLS linear fit when
    Gompertz fails to converge (common with ordinal time).

    Args:
        times: Observation times, shape ``[N_total]`` (all patients pooled).
        volumes: Log-volume observations, shape ``[N_total]``.
        max_iter: Maximum function evaluations per restart.

    Returns:
        GompertzParams with fitted K, b, c.  If ``converged=False``,
        the parameters encode a linear fallback.
    """
    times = np.asarray(times, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    y_max = float(np.max(volumes))
    y_min = float(np.min(volumes[volumes > 0])) if np.any(volumes > 0) else 0.1
    y_mean = float(np.mean(volumes))
    t_range = float(np.ptp(times))

    # Multiple initial guesses for robustness
    init_guesses = [
        (y_max * 1.2, 1.0, 0.5 / max(t_range, 1.0)),
        (y_max * 1.5, 0.5, 0.1),
        (y_max * 2.0, 2.0, 1.0 / max(t_range, 1.0)),
        (y_mean * 2.0, 0.3, 0.2),
    ]

    best_params: GompertzParams | None = None
    best_residual = float("inf")

    for K0, b0, c0 in init_guesses:
        try:
            popt, _ = curve_fit(
                _gompertz_curve,
                times,
                volumes,
                p0=[K0, b0, c0],
                bounds=([y_min * 0.5, 0.001, 0.001], [y_max * 5.0, 50.0, 10.0]),
                maxfev=max_iter,
            )
            residual = float(np.sum((_gompertz_curve(times, *popt) - volumes) ** 2))
            if residual < best_residual:
                best_residual = residual
                best_params = GompertzParams(K=popt[0], b=popt[1], c=popt[2])
        except (RuntimeError, ValueError):
            continue

    if best_params is not None:
        logger.info(
            f"Gompertz fit: K={best_params.K:.3f}, b={best_params.b:.3f}, "
            f"c={best_params.c:.4f} (residual={best_residual:.2f})"
        )
        return best_params

    logger.warning("Gompertz fit failed for all restarts. Using linear fallback.")
    return _linear_fallback(times, volumes)


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
        conv = "converged" if self.params.converged else "linear-fallback"
        return (
            f"GompertzMeanFunction(K={self.params.K:.3f}, "
            f"b={self.params.b:.3f}, c={self.params.c:.4f}, {conv})"
        )
