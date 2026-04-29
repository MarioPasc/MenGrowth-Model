# src/growth/models/growth/_covariance_utils.py
"""Shared covariance-matrix utilities for heteroscedastic growth models.

Provides Cholesky-based solvers and kernel builders reused by
LMEHeteroGrowthModel, ScalarGPHetero, and HGPHeteroModel.
"""

import numpy as np


def build_omega(tau0_sq: float, tau1_sq: float, rho: float) -> np.ndarray:
    """Build 2x2 random-effects covariance matrix.

    Args:
        tau0_sq: Variance of random intercept.
        tau1_sq: Variance of random slope.
        rho: Correlation between random intercept and slope.

    Returns:
        Omega [2, 2] symmetric positive semi-definite.
    """
    tau0 = np.sqrt(tau0_sq)
    tau1 = np.sqrt(tau1_sq)
    off_diag = rho * tau0 * tau1
    return np.array([[tau0_sq, off_diag], [off_diag, tau1_sq]])


def build_Vi(
    t_i: np.ndarray,
    omega: np.ndarray,
    sigma_n_sq: float,
    sigma_v_sq_i: np.ndarray,
) -> np.ndarray:
    """Build per-patient marginal covariance V_i = Z_i Omega Z_i^T + R_i.

    Args:
        t_i: Observation times for patient i, shape [n_i].
        omega: Random-effects covariance [2, 2].
        sigma_n_sq: Biological residual variance (fitted).
        sigma_v_sq_i: Known measurement-error variances [n_i].

    Returns:
        V_i [n_i, n_i] symmetric positive definite.
    """
    n_i = len(t_i)
    Z_i = np.column_stack([np.ones(n_i), t_i])  # [n_i, 2]
    R_i = sigma_n_sq * np.eye(n_i) + np.diag(sigma_v_sq_i)
    V_i = Z_i @ omega @ Z_i.T + R_i
    return V_i


def chol_log_det(L: np.ndarray) -> float:
    """Compute log|V| = 2 * sum(log(diag(L))) from Cholesky factor L.

    Args:
        L: Lower-triangular Cholesky factor [n, n].

    Returns:
        Log-determinant of V = L @ L.T.
    """
    return float(2.0 * np.sum(np.log(np.diag(L))))


def gls_suffstat(
    Xi: np.ndarray,
    Vi: np.ndarray,
    yi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GLS sufficient statistics via Cholesky solve.

    Args:
        Xi: Design matrix [n_i, p].
        Vi: Marginal covariance [n_i, n_i].
        yi: Observations [n_i].

    Returns:
        (XtVinvX [p, p], XtVinvy [p]).
    """
    L = np.linalg.cholesky(Vi)
    # Solve L @ A = Xi => A = L^{-1} Xi
    A = np.linalg.solve(L, Xi)  # [n_i, p]
    # Solve L @ b = yi => b = L^{-1} yi
    b = np.linalg.solve(L, yi)  # [n_i]
    XtVinvX = A.T @ A  # [p, p]
    XtVinvy = A.T @ b  # [p]
    return XtVinvX, XtVinvy


def solve_cholesky(Vi: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve V_i @ x = rhs via Cholesky decomposition.

    Args:
        Vi: Symmetric positive definite matrix [n, n].
        rhs: Right-hand side [n] or [n, k].

    Returns:
        Solution x = V_i^{-1} @ rhs.
    """
    L = np.linalg.cholesky(Vi)
    z = np.linalg.solve(L, rhs)
    return np.linalg.solve(L.T, z)


def matern52_kernel(
    t1: np.ndarray,
    t2: np.ndarray,
    sf2: float,
    ell: float,
) -> np.ndarray:
    """Matern-5/2 kernel matrix K(t1, t2).

    k(r) = sf2 * (1 + sqrt(5)*r/ell + 5*r^2/(3*ell^2)) * exp(-sqrt(5)*r/ell)

    Args:
        t1: First set of times [n1].
        t2: Second set of times [n2].
        sf2: Signal variance.
        ell: Lengthscale.

    Returns:
        Kernel matrix [n1, n2].
    """
    t1 = np.asarray(t1).ravel()
    t2 = np.asarray(t2).ravel()
    r = np.abs(t1[:, np.newaxis] - t2[np.newaxis, :])
    sqrt5_r_ell = np.sqrt(5.0) * r / ell
    K = sf2 * (1.0 + sqrt5_r_ell + 5.0 * r**2 / (3.0 * ell**2)) * np.exp(-sqrt5_r_ell)
    return K


def condition_and_predict(
    K_train: np.ndarray,
    Sigma: np.ndarray,
    y_train: np.ndarray,
    K_star: np.ndarray,
    K_star_star: np.ndarray,
    sigma_v_star: np.ndarray,
    sigma_n_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form GP posterior with heteroscedastic noise.

    Args:
        K_train: Kernel matrix between training points [n_train, n_train].
        Sigma: Diagonal noise variances for training points [n_train].
        y_train: Training observations [n_train].
        K_star: Cross-kernel [n_pred, n_train].
        K_star_star: Prior variance at prediction points [n_pred].
        sigma_v_star: Measurement noise at prediction points [n_pred].
        sigma_n_sq: Biological noise variance.

    Returns:
        (mean [n_pred], variance [n_pred]) for the observable predictive distribution.
    """
    L = np.linalg.cholesky(K_train + np.diag(Sigma))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mean = K_star @ alpha
    v = np.linalg.solve(L, K_star.T)  # [n_train, n_pred]
    var_latent = K_star_star - np.einsum("ij,ij->j", v, v) + sigma_n_sq
    return mean, np.maximum(var_latent + sigma_v_star, 1e-10)
