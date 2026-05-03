# src/growth/models/growth/_nlme_internals.py
"""Numerical engine for NLME models via Laplace approximation.

Pure functions for pack/unpack, inner/outer optimisation, Hessian
computation, and delta-method predictive variance. No GrowthModel
subclassing — consumed by nlme_analytical.py.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pack / unpack population parameters
# ---------------------------------------------------------------------------

_OMEGA_FLOOR = 1e-8


def pack_population_params(
    beta: np.ndarray,
    L_chol_vech: np.ndarray,
    log_sigma_sq: float,
) -> np.ndarray:
    """Pack population parameters into a single flat vector for L-BFGS-B.

    Layout: [beta (n_fe), vech(L_chol) (n_re*(n_re+1)/2), log_sigma_sq (1)].

    Args:
        beta: Fixed effects vector, shape (n_fe,).
        L_chol_vech: Lower-triangular Cholesky factor in vech form.
        log_sigma_sq: Log of residual variance.

    Returns:
        Flat parameter vector.
    """
    return np.concatenate([beta, L_chol_vech, [log_sigma_sq]])


def unpack_population_params(
    theta: np.ndarray,
    n_fe: int,
    n_re: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Unpack flat vector into (beta, Omega, sigma_sq).

    Args:
        theta: Packed parameter vector.
        n_fe: Number of fixed effects.
        n_re: Number of random effects.

    Returns:
        (beta, Omega, sigma_sq) where Omega = L @ L.T + floor*I.
    """
    beta = theta[:n_fe]
    n_chol = n_re * (n_re + 1) // 2
    L_vech = theta[n_fe : n_fe + n_chol]
    log_sigma_sq = theta[n_fe + n_chol]

    L = np.zeros((n_re, n_re))
    idx = 0
    for i in range(n_re):
        for j in range(i + 1):
            L[i, j] = L_vech[idx]
            idx += 1

    Omega = L @ L.T + _OMEGA_FLOOR * np.eye(n_re)
    sigma_sq = np.exp(log_sigma_sq)

    return beta, Omega, sigma_sq


def vech_from_matrix(L: np.ndarray) -> np.ndarray:
    """Extract lower-triangular elements in column-major (vech) order."""
    n = L.shape[0]
    elems = []
    for i in range(n):
        for j in range(i + 1):
            elems.append(L[i, j])
    return np.array(elems)


# ---------------------------------------------------------------------------
# Per-patient negative joint log-likelihood
# ---------------------------------------------------------------------------


def neg_joint_log_patient(
    u: np.ndarray,
    times: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    beta: np.ndarray,
    Omega_inv: np.ndarray,
    Omega_logdet: float,
    sigma_sq: float,
    re_indices: np.ndarray,
) -> float:
    """Negative joint log p(y_i, u_i) = -log p(y_i|u_i) - log p(u_i).

    Args:
        u: Random effects for this patient, shape (n_re,).
        times: Observation times, shape (n_i,).
        y: Observations (log-volume), shape (n_i,).
        model_fn: f(t, theta_i) -> predicted log-volume, shape (n_i,).
        beta: Fixed effects, shape (n_fe,).
        Omega_inv: Inverse of RE covariance, shape (n_re, n_re).
        Omega_logdet: Log determinant of Omega.
        sigma_sq: Residual variance.
        re_indices: Which elements of theta_i get RE, shape (n_re,).

    Returns:
        Scalar negative joint log-likelihood.
    """
    theta_i = beta.copy()
    theta_i[re_indices] += u

    y_hat = model_fn(times, theta_i)
    residuals = y - y_hat
    n_i = len(y)
    n_re = len(u)

    # -log p(y_i | u_i) = 0.5 * [n_i * log(2π σ²) + ||r||² / σ²]
    neg_log_lik = 0.5 * (n_i * np.log(2.0 * np.pi * sigma_sq) + np.sum(residuals**2) / sigma_sq)

    # -log p(u_i) = 0.5 * [n_re * log(2π) + logdet(Ω) + u^T Ω^{-1} u]
    neg_log_prior = 0.5 * (n_re * np.log(2.0 * np.pi) + Omega_logdet + u @ Omega_inv @ u)

    return neg_log_lik + neg_log_prior


# ---------------------------------------------------------------------------
# Inner optimisation: find MAP random effects û_i
# ---------------------------------------------------------------------------


def inner_optimise(
    times: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    beta: np.ndarray,
    Omega: np.ndarray,
    sigma_sq: float,
    re_indices: np.ndarray,
    u_init: np.ndarray | None = None,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    """Find MAP random effects û_i via L-BFGS-B.

    Args:
        times: Observation times for patient i.
        y: Observations for patient i.
        model_fn: Growth model function.
        beta: Current fixed effects.
        Omega: RE covariance matrix.
        sigma_sq: Current residual variance.
        re_indices: Indices of theta_i that receive random effects.
        u_init: Initial guess for u. Defaults to zeros.
        max_iter: Maximum L-BFGS-B iterations.

    Returns:
        (u_hat, neg_joint_value) at the optimum.
    """
    n_re = len(re_indices)
    if u_init is None:
        u_init = np.zeros(n_re)

    try:
        Omega_inv = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        Omega_inv = np.linalg.pinv(Omega)
    _, Omega_logdet = np.linalg.slogdet(Omega)

    def objective(u: np.ndarray) -> float:
        return neg_joint_log_patient(
            u, times, y, model_fn, beta, Omega_inv, Omega_logdet, sigma_sq, re_indices
        )

    result = minimize(
        objective,
        u_init,
        method="L-BFGS-B",
        jac="3-point",
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-6},
    )

    return result.x, result.fun


# ---------------------------------------------------------------------------
# Numerical Hessian (4-point central differences)
# ---------------------------------------------------------------------------


def numerical_hessian(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute Hessian via 4-point central differences.

    Uses the formula:
        H[i,j] = (f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)) / (4*h_i*h_j)

    Args:
        f: Scalar function of vector x.
        x: Point at which to evaluate the Hessian.
        eps: Base step size, scaled by max(|x[j]|, 1).

    Returns:
        Symmetric Hessian matrix, shape (n, n).
    """
    n = len(x)
    H = np.zeros((n, n))
    h = eps * np.maximum(np.abs(x), 1.0)

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += h[i]
            x_pp[j] += h[j]
            x_pm[i] += h[i]
            x_pm[j] -= h[j]
            x_mp[i] -= h[i]
            x_mp[j] += h[j]
            x_mm[i] -= h[i]
            x_mm[j] -= h[j]

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * h[i] * h[j])
            H[j, i] = H[i, j]

    return H


# ---------------------------------------------------------------------------
# Laplace log-marginal-likelihood for one patient
# ---------------------------------------------------------------------------


def laplace_log_marginal_patient(
    times: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    beta: np.ndarray,
    Omega: np.ndarray,
    sigma_sq: float,
    re_indices: np.ndarray,
    u_init: np.ndarray | None = None,
    max_iter: int = 200,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Laplace approximation to log p(y_i) for one patient.

    log p(y_i) ≈ -neg_joint(û_i) - 0.5 * log det(H_i / (2π))
              = -neg_joint(û_i) - 0.5 * (log det H_i - n_re * log(2π))

    Args:
        times: Observation times.
        y: Observations (log-volume).
        model_fn: Growth model function.
        beta: Fixed effects.
        Omega: RE covariance.
        sigma_sq: Residual variance.
        re_indices: Which theta_i elements receive RE.
        u_init: Warm-start for inner optimisation.
        max_iter: Max inner iterations.

    Returns:
        (log_p_i, u_hat_i, H_i) where H_i is the Hessian at the mode.
    """
    n_re = len(re_indices)

    u_hat, neg_joint_val = inner_optimise(
        times, y, model_fn, beta, Omega, sigma_sq, re_indices, u_init, max_iter
    )

    try:
        Omega_inv = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        Omega_inv = np.linalg.pinv(Omega)
    _, Omega_logdet = np.linalg.slogdet(Omega)

    def obj_u(u: np.ndarray) -> float:
        return neg_joint_log_patient(
            u, times, y, model_fn, beta, Omega_inv, Omega_logdet, sigma_sq, re_indices
        )

    H_i = numerical_hessian(obj_u, u_hat)

    # Regularise Hessian for numerical stability
    H_i = 0.5 * (H_i + H_i.T)
    eigvals = np.linalg.eigvalsh(H_i)
    if eigvals.min() < 1e-6:
        H_i += (1e-6 - eigvals.min()) * np.eye(n_re)

    sign, logdet_H = np.linalg.slogdet(H_i)
    if sign <= 0:
        logger.warning("Hessian not positive-definite; returning -inf for patient LML")
        return -np.inf, u_hat, H_i

    # log p(y_i) ≈ -neg_joint(û) - 0.5*(logdet(H) - n_re*log(2π))
    log_p_i = -neg_joint_val - 0.5 * (logdet_H - n_re * np.log(2.0 * np.pi))

    return log_p_i, u_hat, H_i


# ---------------------------------------------------------------------------
# Outer objective: total negative Laplace LML
# ---------------------------------------------------------------------------


def outer_objective(
    theta_packed: np.ndarray,
    patients_data: list[tuple[np.ndarray, np.ndarray]],
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_fe: int,
    n_re: int,
    re_indices: np.ndarray,
    u_cache: list[np.ndarray | None],
    max_inner_iter: int = 200,
) -> tuple[float, list[np.ndarray]]:
    """Negative total Laplace log-marginal-likelihood (outer objective).

    Args:
        theta_packed: Packed population parameters.
        patients_data: List of (times_i, y_i) per patient.
        model_fn: Growth model function.
        n_fe: Number of fixed effects.
        n_re: Number of random effects.
        re_indices: Which theta elements receive RE.
        u_cache: Warm-start u_hat per patient (updated in-place).
        max_inner_iter: Max iterations for inner optimisation.

    Returns:
        (neg_total_lml, updated_u_cache).
    """
    beta, Omega, sigma_sq = unpack_population_params(theta_packed, n_fe, n_re)

    total_lml = 0.0
    new_u_cache: list[np.ndarray] = []

    for i, (times_i, y_i) in enumerate(patients_data):
        u_init = u_cache[i] if u_cache[i] is not None else np.zeros(n_re)

        log_p_i, u_hat_i, _ = laplace_log_marginal_patient(
            times_i, y_i, model_fn, beta, Omega, sigma_sq, re_indices, u_init, max_inner_iter
        )

        total_lml += log_p_i
        new_u_cache.append(u_hat_i)

    if not np.isfinite(total_lml):
        total_lml = -1e15

    return -total_lml, new_u_cache


# ---------------------------------------------------------------------------
# Delta-method predictive variance
# ---------------------------------------------------------------------------


def delta_method_predictive_variance(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t_star: float,
    beta: np.ndarray,
    u_hat: np.ndarray,
    Var_u: np.ndarray,
    sigma_sq: float,
    re_indices: np.ndarray,
    eps: float = 1e-5,
) -> float:
    """Predictive variance via first-order delta method.

    Var(ŷ*) = g^T Var(u) g + σ²_pop

    where g = ∇_u f(t*; β + u)|_{u=û}, computed via central differences.

    Args:
        model_fn: f(t, theta_i) -> predicted values.
        t_star: Prediction time (scalar).
        beta: Fixed effects.
        u_hat: MAP random effects.
        Var_u: Posterior covariance of u, shape (n_re, n_re).
        sigma_sq: Population residual variance.
        re_indices: Indices into theta_i for random effects.
        eps: Step size for numerical gradient.

    Returns:
        Scalar predictive variance.
    """
    n_re = len(re_indices)
    t_arr = np.array([t_star])
    g = np.zeros(n_re)

    theta_base = beta.copy()
    theta_base[re_indices] += u_hat

    for j in range(n_re):
        u_plus = u_hat.copy()
        u_minus = u_hat.copy()
        h = eps * max(abs(u_hat[j]), 1.0)
        u_plus[j] += h
        u_minus[j] -= h

        theta_plus = beta.copy()
        theta_plus[re_indices] += u_plus
        theta_minus = beta.copy()
        theta_minus[re_indices] += u_minus

        f_plus = model_fn(t_arr, theta_plus)[0]
        f_minus = model_fn(t_arr, theta_minus)[0]
        g[j] = (f_plus - f_minus) / (2.0 * h)

    var_pred = g @ Var_u @ g + sigma_sq
    return max(var_pred, 0.0)
