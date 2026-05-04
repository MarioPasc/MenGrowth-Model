# src/growth/models/growth/nlme_analytical.py
"""Analytical NLME growth models via Laplace approximation.

Three nonlinear mixed-effects models (Exponential, Logistic, Gompertz)
fitted with population-level residual variance σ²_pop. These serve as
classical analytical baselines against which heteroscedastic models
(propagated segmentation uncertainty) are compared.

Mathematical framework: Engelhardt et al. (2023), NeuroImage: Clinical.
Laplace approximation: Pinheiro & Bates (2000), Mixed-Effects Models in S and S-PLUS.
"""

from __future__ import annotations

import logging
from abc import abstractmethod

import numpy as np
from scipy.optimize import minimize

from growth.shared.growth_models import (
    FitResult,
    GrowthModel,
    PatientTrajectory,
    PredictionResult,
)

from ._nlme_internals import (
    delta_method_predictive_variance,
    inner_optimise,
    numerical_hessian,
    outer_objective,
    pack_population_params,
    unpack_population_params,
    vech_from_matrix,
)

logger = logging.getLogger(__name__)


class AnalyticalNLMEModel(GrowthModel):
    """ABC for analytical NLME growth models via Laplace approximation.

    Subclasses define the growth ODE, parameterisation, and initial
    values. The fit/predict machinery is shared.

    Args:
        n_restarts: Number of random restarts for outer optimisation.
        max_iter: Maximum outer L-BFGS-B iterations.
        seed: Random seed for restarts.
        fallback_to_1re: If True, retry with 1 random effect on convergence failure.
    """

    def __init__(
        self,
        n_restarts: int = 3,
        max_iter: int = 500,
        seed: int = 42,
        fallback_to_1re: bool = True,
    ) -> None:
        self._n_restarts = n_restarts
        self._max_iter = max_iter
        self._seed = seed
        self._fallback_to_1re = fallback_to_1re

        # State set by fit()
        self._beta: np.ndarray | None = None
        self._Omega: np.ndarray | None = None
        self._sigma_sq: float | None = None
        self._u_hats: dict[str, np.ndarray] = {}
        self._fitted = False
        self._n_re_actual: int | None = None

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _model_fn(self, t: np.ndarray, theta_i: np.ndarray) -> np.ndarray:
        """Evaluate growth model on log-volume scale.

        Args:
            t: Times, shape (n,).
            theta_i: Patient-specific parameter vector (beta + u_i), shape (n_fe,).

        Returns:
            Predicted log-volume, shape (n,).
        """

    @abstractmethod
    def _init_beta(self, patients: list[PatientTrajectory]) -> np.ndarray:
        """Initialise fixed-effect vector from pooled data.

        Returns:
            Initial beta, shape (n_fe,).
        """

    @property
    @abstractmethod
    def n_fixed_effects(self) -> int:
        """Number of fixed-effect parameters."""

    @property
    @abstractmethod
    def n_random_effects(self) -> int:
        """Number of random-effect parameters (full model)."""

    @property
    @abstractmethod
    def _re_indices(self) -> np.ndarray:
        """Indices into theta_i that receive random effects."""

    @property
    def _re_indices_fallback(self) -> np.ndarray:
        """Indices for 1-RE fallback (intercept only)."""
        return np.array([0])

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit NLME via Laplace approximation with L-BFGS-B outer loop.

        Args:
            patients: Training patient trajectories.

        Returns:
            FitResult with Laplace log-marginal-likelihood.
        """
        patients_data = [
            (p.times.astype(np.float64), p.observations[:, 0].astype(np.float64)) for p in patients
        ]
        n_obs = sum(len(t) for t, _ in patients_data)

        beta_init = self._init_beta(patients)
        n_fe = self.n_fixed_effects
        n_re = self.n_random_effects
        re_idx = self._re_indices

        best_lml, best_params = self._run_outer_optimisation(
            patients_data, beta_init, n_fe, n_re, re_idx
        )

        if not np.isfinite(best_lml) and self._fallback_to_1re and n_re > 1:
            logger.warning(
                f"{self.name()}: full {n_re}-RE model failed to converge, "
                "falling back to 1-RE (intercept only)"
            )
            n_re_fb = 1
            re_idx_fb = self._re_indices_fallback
            best_lml, best_params = self._run_outer_optimisation(
                patients_data, beta_init, n_fe, n_re_fb, re_idx_fb
            )
            if np.isfinite(best_lml):
                n_re = n_re_fb
                re_idx = re_idx_fb

        self._beta, self._Omega, self._sigma_sq = unpack_population_params(best_params, n_fe, n_re)
        self._n_re_actual = n_re
        self._re_indices_actual = re_idx

        # Cache per-patient u_hats from final fit
        self._u_hats = {}
        for i, (times_i, y_i) in enumerate(patients_data):
            u_hat, _ = inner_optimise(
                times_i, y_i, self._model_fn, self._beta, self._Omega, self._sigma_sq, re_idx
            )
            self._u_hats[patients[i].patient_id] = u_hat

        self._fitted = True

        hyp = {f"beta_{j}": float(self._beta[j]) for j in range(n_fe)}
        hyp["sigma_sq"] = float(self._sigma_sq)
        hyp["n_random_effects"] = n_re
        for i in range(n_re):
            for j in range(i + 1):
                hyp[f"Omega_{i}{j}"] = float(self._Omega[i, j])

        return FitResult(
            log_marginal_likelihood=float(best_lml),
            hyperparameters=hyp,
            n_train_patients=len(patients),
            n_train_observations=n_obs,
        )

    def _run_outer_optimisation(
        self,
        patients_data: list[tuple[np.ndarray, np.ndarray]],
        beta_init: np.ndarray,
        n_fe: int,
        n_re: int,
        re_indices: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Run L-BFGS-B outer loop with random restarts.

        Returns:
            (best_lml, best_params_packed).
        """
        rng = np.random.default_rng(self._seed)

        best_neg_lml = np.inf
        best_params = None

        for restart in range(self._n_restarts):
            if restart == 0:
                beta = beta_init.copy()
                L_init = 0.1 * np.eye(n_re)
                log_sig_init = np.log(0.1)
            else:
                beta = beta_init + rng.normal(0, 0.2, size=n_fe)
                L_init = (0.05 + 0.1 * rng.random()) * np.eye(n_re)
                log_sig_init = np.log(0.05 + 0.15 * rng.random())

            L_vech = vech_from_matrix(L_init)
            theta0 = pack_population_params(beta, L_vech, log_sig_init)
            u_cache: list[np.ndarray | None] = [None] * len(patients_data)

            # Wrapper that updates u_cache across calls
            cache_holder = {"u": u_cache}

            def objective(theta: np.ndarray) -> float:
                neg_lml, new_cache = outer_objective(
                    theta,
                    patients_data,
                    self._model_fn,
                    n_fe,
                    n_re,
                    re_indices,
                    cache_holder["u"],
                    max_inner_iter=min(200, self._max_iter),
                )
                cache_holder["u"] = new_cache
                return neg_lml

            try:
                result = minimize(
                    objective,
                    theta0,
                    method="L-BFGS-B",
                    jac="3-point",
                    options={"maxiter": self._max_iter, "ftol": 1e-8, "gtol": 1e-5},
                )

                if result.fun < best_neg_lml:
                    best_neg_lml = result.fun
                    best_params = result.x.copy()

            except Exception as e:
                logger.debug(f"{self.name()} restart {restart} failed: {e}")
                continue

        if best_params is None:
            logger.warning(f"{self.name()}: all restarts failed")
            # Return a fallback with very bad LML
            L_vech = vech_from_matrix(0.1 * np.eye(n_re))
            best_params = pack_population_params(beta_init, L_vech, np.log(1.0))
            best_neg_lml = np.inf

        best_lml = -best_neg_lml if np.isfinite(best_neg_lml) else -np.inf
        return best_lml, best_params

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict with delta-method uncertainty from Laplace approximation.

        Args:
            patient: Patient trajectory (conditioning observations).
            t_pred: Times at which to predict, shape (n_pred,).
            n_condition: Number of initial observations to condition on.
                If None, uses all available observations.

        Returns:
            PredictionResult with mean, variance, and 95% CIs.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        t_pred = np.asarray(t_pred, dtype=np.float64).ravel()
        n_pred = len(t_pred)
        n_re = self._n_re_actual
        re_idx = self._re_indices_actual

        if n_condition is not None:
            t_cond = patient.times[:n_condition].astype(np.float64)
            y_cond = patient.observations[:n_condition, 0].astype(np.float64)
        else:
            t_cond = patient.times.astype(np.float64)
            y_cond = patient.observations[:, 0].astype(np.float64)

        # Warm-start from cached u_hat if available
        u_init = self._u_hats.get(patient.patient_id)

        # Find MAP u for conditioning data
        u_hat_star, _ = inner_optimise(
            t_cond, y_cond, self._model_fn, self._beta, self._Omega, self._sigma_sq, re_idx, u_init
        )

        # Posterior covariance from Hessian at mode
        try:
            Omega_inv = np.linalg.inv(self._Omega)
        except np.linalg.LinAlgError:
            Omega_inv = np.linalg.pinv(self._Omega)
        _, Omega_logdet = np.linalg.slogdet(self._Omega)

        from ._nlme_internals import neg_joint_log_patient

        def obj_u(u: np.ndarray) -> float:
            return neg_joint_log_patient(
                u,
                t_cond,
                y_cond,
                self._model_fn,
                self._beta,
                Omega_inv,
                Omega_logdet,
                self._sigma_sq,
                re_idx,
            )

        H_star = numerical_hessian(obj_u, u_hat_star)
        H_star = 0.5 * (H_star + H_star.T)

        # Regularise
        eigvals = np.linalg.eigvalsh(H_star)
        if eigvals.min() < 1e-6:
            H_star += (1e-6 - eigvals.min()) * np.eye(n_re)

        try:
            Var_u = np.linalg.inv(H_star)
            Var_u = 0.5 * (Var_u + Var_u.T)
        except np.linalg.LinAlgError:
            Var_u = np.linalg.pinv(H_star)
            Var_u = 0.5 * (Var_u + Var_u.T)

        # Predictive mean and variance
        theta_hat = self._beta.copy()
        theta_hat[re_idx] += u_hat_star

        pred_mean = self._model_fn(t_pred, theta_hat)
        pred_var = np.zeros(n_pred)

        for j in range(n_pred):
            pred_var[j] = delta_method_predictive_variance(
                self._model_fn,
                t_pred[j],
                self._beta,
                u_hat_star,
                Var_u,
                self._sigma_sq,
                re_idx,
            )

        std = np.sqrt(np.maximum(pred_var, 0.0))

        return PredictionResult(
            mean=pred_mean.reshape(-1, 1),
            variance=pred_var.reshape(-1, 1),
            lower_95=(pred_mean - 1.96 * std).reshape(-1, 1),
            upper_95=(pred_mean + 1.96 * std).reshape(-1, 1),
            metadata={
                "n_random_effects": n_re,
                "sigma_sq_pop": float(self._sigma_sq),
                "u_hat": u_hat_star.tolist(),
            },
        )


# ======================================================================
# Concrete models
# ======================================================================


class ExponentialNLME(AnalyticalNLMEModel):
    """Exponential growth NLME: log V(t) = log_V0 + a*t.

    Linear in log-space. Two fixed effects (log_V0, a), two random
    effects on both. Simplest parametric baseline.
    """

    @property
    def n_fixed_effects(self) -> int:
        return 2

    @property
    def n_random_effects(self) -> int:
        return 2

    @property
    def _re_indices(self) -> np.ndarray:
        return np.array([0, 1])

    def _model_fn(self, t: np.ndarray, theta_i: np.ndarray) -> np.ndarray:
        log_V0 = theta_i[0]
        a = theta_i[1]
        return log_V0 + a * t

    def _init_beta(self, patients: list[PatientTrajectory]) -> np.ndarray:
        all_t = np.concatenate([p.times for p in patients])
        all_y = np.concatenate([p.observations[:, 0] for p in patients])

        # Pooled OLS: y = beta0 + beta1 * t
        A = np.column_stack([np.ones_like(all_t), all_t])
        beta, _, _, _ = np.linalg.lstsq(A, all_y, rcond=None)
        return beta

    def name(self) -> str:
        return "NLME_Exponential"


class LogisticNLME(AnalyticalNLMEModel):
    """Logistic growth NLME on log-volume scale.

    V(t) = K / (1 + (K/V0 - 1)*exp(-a*t))
    log V(t) = log(K) - log(1 + (K/V0 - 1)*exp(-a*t))

    Three fixed effects: (log_V0, a, log_K). Random effects on
    (log_V0, a) only — K is population-shared since N is too small
    for per-patient carrying capacity.
    """

    @property
    def n_fixed_effects(self) -> int:
        return 3

    @property
    def n_random_effects(self) -> int:
        return 2

    @property
    def _re_indices(self) -> np.ndarray:
        return np.array([0, 1])

    def _model_fn(self, t: np.ndarray, theta_i: np.ndarray) -> np.ndarray:
        log_V0 = theta_i[0]
        a = theta_i[1]
        log_K = theta_i[2]

        K = np.exp(log_K)
        V0 = np.exp(log_V0)

        ratio = np.maximum(K / (V0 + 1e-15), 1.0 + 1e-10)
        V_t = K / (1.0 + (ratio - 1.0) * np.exp(-a * t))
        return np.log(np.maximum(V_t, 1e-15))

    def _init_beta(self, patients: list[PatientTrajectory]) -> np.ndarray:
        all_t = np.concatenate([p.times for p in patients])
        all_y = np.concatenate([p.observations[:, 0] for p in patients])

        # OLS for initial log_V0 and a
        A = np.column_stack([np.ones_like(all_t), all_t])
        beta_ols, _, _, _ = np.linalg.lstsq(A, all_y, rcond=None)

        # K = 2 * max observed volume (on original scale)
        max_log_vol = np.max(all_y)
        log_K = max_log_vol + np.log(2.0)

        return np.array([beta_ols[0], max(beta_ols[1], 0.01), log_K])

    def name(self) -> str:
        return "NLME_Logistic"


class GompertzNLME(AnalyticalNLMEModel):
    """Gompertz growth NLME, same parameterisation as gompertz.py.

    V(t) = K * exp(-b * exp(-c * t))
    log V(t) = log(K) - b * exp(-c * t)

    Three fixed effects: (log_K, log_b, log_c). Random effects on
    (log_K, log_b) — patient-level variation in asymptote and initial
    growth. c (decay rate) is population-shared.

    Warm-starts from fit_gompertz() on pooled data.
    """

    @property
    def n_fixed_effects(self) -> int:
        return 3

    @property
    def n_random_effects(self) -> int:
        return 2

    @property
    def _re_indices(self) -> np.ndarray:
        return np.array([0, 1])

    def _model_fn(self, t: np.ndarray, theta_i: np.ndarray) -> np.ndarray:
        log_K = theta_i[0]
        log_b = theta_i[1]
        log_c = theta_i[2]

        K = np.exp(log_K)
        b = np.exp(log_b)
        c = np.exp(log_c)

        V_t = K * np.exp(-b * np.exp(-c * t))
        return np.log(np.maximum(V_t, 1e-15))

    def _init_beta(self, patients: list[PatientTrajectory]) -> np.ndarray:
        all_t = np.concatenate([p.times for p in patients])
        all_y = np.concatenate([p.observations[:, 0] for p in patients])

        # Warm-start from existing Gompertz fitter (operates on original scale)
        try:
            from growth.stages.stage1_volumetric.gompertz import fit_gompertz

            volumes = np.exp(all_y)
            gp = fit_gompertz(all_t, volumes)

            if gp.converged and gp.K > 0 and gp.b > 0 and gp.c > 0:
                return np.array([np.log(gp.K), np.log(gp.b), np.log(gp.c)])
        except Exception:
            pass

        # Fallback: heuristic from OLS
        A = np.column_stack([np.ones_like(all_t), all_t])
        beta_ols, _, _, _ = np.linalg.lstsq(A, all_y, rcond=None)
        log_K = np.max(all_y) + 0.5
        log_b = np.log(max(log_K - beta_ols[0], 0.1))
        log_c = np.log(max(abs(beta_ols[1]), 0.01))

        return np.array([log_K, log_b, log_c])

    def name(self) -> str:
        return "NLME_Gompertz"
