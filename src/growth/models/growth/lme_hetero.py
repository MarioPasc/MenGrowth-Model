# src/growth/models/growth/lme_hetero.py
"""Heteroscedastic Linear Mixed-Effects growth model.

Custom REML implementation with additive known measurement-error variance:

    R_i = sigma_n^2 * I + diag(sigma_v_i)
    V_i = Z_i @ Omega @ Z_i^T + R_i

where sigma_v_i are read from PatientTrajectory.observation_variance
(known per-scan segmentation uncertainty from the LoRA ensemble).

References:
    Laird & Ware, "Random-Effects Models for Longitudinal Data", Biometrics 1982.
    Carroll et al., "Measurement Error in Nonlinear Models" (2nd ed.), Ch. 9, 2006.
    Pinheiro & Bates, "Mixed-Effects Models in S and S-PLUS", Ch. 5, 2000.
"""

import logging

import numpy as np
from scipy.optimize import minimize

from growth.exceptions import UncertaintyPropagationError

from ._covariance_utils import build_omega, build_Vi, chol_log_det, gls_suffstat, solve_cholesky
from .base import FitResult, GrowthModel, PatientTrajectory, PredictionResult
from .covariate_utils import collect_covariates, get_patient_covariate_vector

logger = logging.getLogger(__name__)


class LMEHeteroGrowthModel(GrowthModel):
    """LME with per-observation known measurement-error variance.

    Hyperparameters (sigma_n^2, tau_0^2, tau_1^2, rho) are optimised by
    L-BFGS-B over a log/atanh parameterisation; fixed effects are profiled
    out by GLS.

    Args:
        method: ``"reml"`` (default) or ``"ml"``.
        n_restarts: Number of L-BFGS-B restarts from random initialisations.
        max_iter: Maximum iterations per restart.
        seed: Random seed.
        use_covariates: Whether to extend the fixed-effect design with covariates.
        covariate_names: Ordered names of covariates.
        missing_strategy: How to handle missing covariates.
        floor_variance: Minimum allowed observation variance entry.
    """

    def __init__(
        self,
        method: str = "reml",
        n_restarts: int = 5,
        max_iter: int = 1000,
        seed: int = 42,
        use_covariates: bool = False,
        covariate_names: list[str] | None = None,
        missing_strategy: str = "skip",
        floor_variance: float = 1e-6,
    ) -> None:
        if method not in ("reml", "ml"):
            raise ValueError(f"method must be 'reml' or 'ml', got '{method}'")
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

        self.method = method
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.seed = seed
        self.use_covariates = use_covariates
        self.covariate_names = covariate_names or []
        self.missing_strategy = missing_strategy
        self.floor_variance = floor_variance

        self._beta: np.ndarray | None = None
        self._sigma_n_sq: float = 0.0
        self._omega: np.ndarray | None = None
        self._fitted: bool = False
        self._active_cov_names: list[str] = []
        self._cov_means: dict[str, float] = {}
        # GLS covariance of fixed-effect estimates: (sum_i X_i^T V_i^{-1} X_i)^{-1}.
        # Required for the fixed-effect-uncertainty term of the predictive variance
        # (uncertainty_propagation.tex, subsec:methods-end-to-end-uq).
        self._cov_beta: np.ndarray | None = None

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit via custom REML with heteroscedastic residuals.

        Args:
            patients: Training patient trajectories with observation_variance set.

        Returns:
            FitResult with optimised hyperparameters.
        """
        if len(patients) == 0:
            raise ValueError("Cannot fit with zero patients")

        for p in patients:
            if p.observation_variance is None:
                raise UncertaintyPropagationError(
                    f"LMEHetero requires observation_variance to be set for patient {p.patient_id}"
                )
            if p.n_timepoints < 2:
                raise ValueError(
                    f"LMEHetero requires n_i >= 2, patient {p.patient_id} has {p.n_timepoints}"
                )

        # Collect covariates if enabled
        cov_values: dict[str, np.ndarray] = {}
        self._active_cov_names = []
        if self.use_covariates and self.covariate_names:
            cov_values, self._active_cov_names, patients = collect_covariates(
                patients, self.covariate_names, self.missing_strategy
            )
            for i, cov_name in enumerate(self._active_cov_names):
                vals = [v[i] for v in cov_values.values()]
                self._cov_means[cov_name] = float(np.mean(vals)) if vals else 0.0

        n_total = sum(p.n_timepoints for p in patients)
        n_cov = len(self._active_cov_names)
        p_dim = 2 + n_cov  # intercept + slope + covariates

        # Build per-patient data structures
        patient_data = self._build_patient_data(patients, cov_values)

        rng = np.random.default_rng(self.seed)

        best_nll = np.inf
        best_theta = None
        best_beta = None

        for restart in range(self.n_restarts):
            theta0 = np.array(
                [
                    np.log(rng.uniform(1e-3, 1.0)),  # log(sigma_n^2)
                    np.log(rng.uniform(1e-2, 10.0)),  # log(tau0^2)
                    np.log(rng.uniform(1e-2, 10.0)),  # log(tau1^2)
                    np.arctanh(rng.uniform(-0.5, 0.5)),  # atanh(rho)
                ]
            )

            try:
                result = minimize(
                    self._neg_reml,
                    theta0,
                    args=(patient_data, p_dim),
                    method="L-BFGS-B",
                    options={"maxiter": self.max_iter, "ftol": 1e-10},
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_theta = result.x
                    best_beta = self._last_beta
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.debug(f"Restart {restart} failed: {e}")
                continue

        if best_theta is None:
            logger.warning("All random-slope restarts failed, trying intercept-only")
            best_nll, best_theta, best_beta = self._fit_intercept_only(patient_data, p_dim, rng)

        if best_theta is None:
            raise UncertaintyPropagationError("LMEHetero optimisation failed on all restarts")

        # Unpack best solution
        sigma_n_sq = np.exp(best_theta[0])
        tau0_sq = np.exp(best_theta[1])
        tau1_sq = np.exp(best_theta[2])
        rho = np.tanh(best_theta[3])

        self._sigma_n_sq = sigma_n_sq
        self._omega = build_omega(tau0_sq, tau1_sq, rho)
        self._beta = best_beta
        self._fitted = True

        # Recompute (sum_i X_i^T V_i^{-1} X_i) at the optimum and invert it
        # to obtain the GLS covariance of beta_hat. Used by predict().
        # Also compute the condition number of the largest V_i in the same loop.
        sum_XtVinvX_opt = np.zeros((p_dim, p_dim))
        max_cond = 0.0
        for Xi, Zi, yi, sv in patient_data:
            Vi = build_Vi(Zi[:, 1], self._omega, sigma_n_sq, sv)
            eigvals = np.linalg.eigvalsh(Vi)
            cond = float(eigvals[-1] / max(eigvals[0], 1e-15))
            max_cond = max(max_cond, cond)
            XtVinvX_i, _ = gls_suffstat(Xi, Vi, yi)
            sum_XtVinvX_opt += XtVinvX_i

        try:
            self._cov_beta = np.linalg.inv(sum_XtVinvX_opt)
        except np.linalg.LinAlgError:
            logger.warning("Singular GLS information matrix; cov_beta unavailable")
            self._cov_beta = None

        hypers = {
            "sigma_n_sq": sigma_n_sq,
            "tau0_sq": tau0_sq,
            "tau1_sq": tau1_sq,
            "rho": rho,
        }
        for i, cov_name in enumerate(self._active_cov_names):
            hypers[f"beta_{cov_name}"] = float(self._beta[2 + i])
        hypers["beta_0"] = float(self._beta[0])
        hypers["beta_1"] = float(self._beta[1])

        logger.info(
            f"LMEHetero fit: sigma_n^2={sigma_n_sq:.6f}, "
            f"tau0^2={tau0_sq:.4f}, tau1^2={tau1_sq:.4f}, rho={rho:.3f}, "
            f"beta=[{self._beta[0]:.4f}, {self._beta[1]:.4f}], "
            f"max_cond={max_cond:.1f}"
        )

        return FitResult(
            log_marginal_likelihood=-best_nll,
            hyperparameters=hypers,
            condition_number=max_cond,
            n_train_patients=len(patients),
            n_train_observations=n_total,
        )

    def _build_patient_data(
        self,
        patients: list[PatientTrajectory],
        cov_values: dict[str, np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Build per-patient (X_i, Z_i, y_i, sigma_v_i) tuples.

        Returns:
            List of (Xi [n_i, p], Zi [n_i, 2], yi [n_i], sigma_v_i [n_i]).
        """
        result = []
        for p in patients:
            n_i = p.n_timepoints
            t_i = p.times

            # Fixed-effects design: [1, t, cov1, cov2, ...]
            Xi = np.column_stack([np.ones(n_i), t_i])
            if self._active_cov_names and p.patient_id in cov_values:
                cov = cov_values[p.patient_id]
                cov_cols = np.tile(cov, (n_i, 1))
                Xi = np.column_stack([Xi, cov_cols])
            elif self._active_cov_names:
                Xi = np.column_stack([Xi, np.zeros((n_i, len(self._active_cov_names)))])

            # Random-effects design: [1, t] (intercept + slope only)
            Zi = np.column_stack([np.ones(n_i), t_i])

            yi = p.observations[:, 0]
            sigma_v_i = np.maximum(p.observation_variance, self.floor_variance)

            result.append((Xi, Zi, yi, sigma_v_i))

        return result

    def _neg_reml(
        self,
        theta: np.ndarray,
        patient_data: list[tuple],
        p_dim: int,
    ) -> float:
        """Negative REML criterion.

        Args:
            theta: Unconstrained parameters [log(sigma_n^2), log(tau0^2),
                log(tau1^2), atanh(rho)].
            patient_data: Per-patient data tuples.
            p_dim: Number of fixed-effect parameters.

        Returns:
            Negative restricted log-likelihood.
        """
        sigma_n_sq = np.exp(theta[0])
        tau0_sq = np.exp(theta[1])
        tau1_sq = np.exp(theta[2])
        rho = np.tanh(theta[3])

        omega = build_omega(tau0_sq, tau1_sq, rho)

        sum_XtVinvX = np.zeros((p_dim, p_dim))
        sum_XtVinvy = np.zeros(p_dim)
        sum_log_det = 0.0

        for Xi, Zi, yi, sv in patient_data:
            t_i = Zi[:, 1]
            Vi = build_Vi(t_i, omega, sigma_n_sq, sv)

            try:
                Li = np.linalg.cholesky(Vi)
            except np.linalg.LinAlgError:
                return 1e15

            sum_log_det += chol_log_det(Li)

            XtVinvX_i, XtVinvy_i = gls_suffstat(Xi, Vi, yi)
            sum_XtVinvX += XtVinvX_i
            sum_XtVinvy += XtVinvy_i

        # GLS fixed effects
        try:
            beta_hat = np.linalg.solve(sum_XtVinvX, sum_XtVinvy)
        except np.linalg.LinAlgError:
            return 1e15

        self._last_beta = beta_hat

        # Quadratic form sum_i (y_i - X_i beta)^T V_i^{-1} (y_i - X_i beta)
        quad_form = 0.0
        for Xi, Zi, yi, sv in patient_data:
            t_i = Zi[:, 1]
            Vi = build_Vi(t_i, omega, sigma_n_sq, sv)
            resid = yi - Xi @ beta_hat
            quad_form += float(resid @ solve_cholesky(Vi, resid))

        nll = 0.5 * (sum_log_det + quad_form)

        if self.method == "reml":
            sign, logdet_XtVinvX = np.linalg.slogdet(sum_XtVinvX)
            if sign > 0:
                nll += 0.5 * logdet_XtVinvX

        return float(nll)

    def _fit_intercept_only(
        self,
        patient_data: list[tuple],
        p_dim: int,
        rng: np.random.Generator,
    ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        """Fallback: random-intercept-only (tau1^2 = 0, rho = 0)."""
        best_nll = np.inf
        best_theta = None
        best_beta = None

        for _ in range(self.n_restarts):
            theta0 = np.array(
                [
                    np.log(rng.uniform(1e-3, 1.0)),
                    np.log(rng.uniform(1e-2, 10.0)),
                    np.log(1e-10),  # tau1^2 ~ 0
                    0.0,  # rho = 0
                ]
            )

            try:
                result = minimize(
                    self._neg_reml,
                    theta0,
                    args=(patient_data, p_dim),
                    method="L-BFGS-B",
                    bounds=[
                        (None, None),
                        (None, None),
                        (np.log(1e-12), np.log(1e-8)),
                        (-0.01, 0.01),
                    ],
                    options={"maxiter": self.max_iter, "ftol": 1e-10},
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_theta = result.x
                    best_beta = self._last_beta
            except (np.linalg.LinAlgError, ValueError):
                continue

        return best_nll, best_theta, best_beta

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict via BLUP with heteroscedastic conditioning.

        Args:
            patient: Patient trajectory with observation_variance.
            t_pred: Query times [n_pred].
            n_condition: Condition on first n_condition observations.

        Returns:
            PredictionResult with observable variance (regime b).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        t_pred = np.asarray(t_pred, dtype=np.float64)
        if t_pred.ndim == 0:
            t_pred = t_pred[np.newaxis]
        n_pred = len(t_pred)

        # Extract conditioning data
        if n_condition is not None:
            t_cond = patient.times[:n_condition]
            y_cond = patient.observations[:n_condition, 0]
            sv_cond = (
                np.maximum(patient.observation_variance[:n_condition], self.floor_variance)
                if patient.observation_variance is not None
                else np.zeros(n_condition)
            )
        else:
            t_cond = patient.times
            y_cond = patient.observations[:, 0]
            sv_cond = (
                np.maximum(patient.observation_variance, self.floor_variance)
                if patient.observation_variance is not None
                else np.zeros(len(t_cond))
            )

        n_cond = len(t_cond)

        # Held-out observation variance for prediction times (regime b)
        if patient.observation_variance is not None and n_condition is not None:
            sv_pred = np.maximum(
                patient.observation_variance[n_condition : n_condition + n_pred],
                self.floor_variance,
            )
            if len(sv_pred) < n_pred:
                sv_pred = np.pad(sv_pred, (0, n_pred - len(sv_pred)), constant_values=0.0)
        else:
            sv_pred = np.zeros(n_pred)

        # Covariate contribution
        cov_vec = get_patient_covariate_vector(patient, self._active_cov_names, self._cov_means)
        cov_contribution = 0.0
        if cov_vec is not None and len(self._active_cov_names) > 0:
            for k, cov_name in enumerate(self._active_cov_names):
                cov_contribution += self._beta[2 + k] * cov_vec[k]

        # Design matrices
        Z_cond = np.column_stack([np.ones(n_cond), t_cond])
        Z_pred = np.column_stack([np.ones(n_pred), t_pred])

        # Population mean
        pop_mean_cond = self._beta[0] + self._beta[1] * t_cond + cov_contribution
        pop_mean_pred = self._beta[0] + self._beta[1] * t_pred + cov_contribution

        # V_i and BLUP
        Vi = build_Vi(t_cond, self._omega, self._sigma_n_sq, sv_cond)
        residuals = y_cond - pop_mean_cond

        try:
            Vi_inv_resid = solve_cholesky(Vi, residuals)
            u_hat = self._omega @ Z_cond.T @ Vi_inv_resid
        except np.linalg.LinAlgError:
            logger.debug("Singular V_i, using population mean only")
            u_hat = np.zeros(2)

        # Posterior covariance of u_i
        try:
            Vi_inv_Z = solve_cholesky(Vi, Z_cond)
            cov_post = self._omega - self._omega @ Z_cond.T @ Vi_inv_Z @ self._omega
        except np.linalg.LinAlgError:
            cov_post = self._omega

        # Predictions
        mean = pop_mean_pred + Z_pred @ u_hat

        # Fixed-effect uncertainty term (thesis: subsec:methods-end-to-end-uq).
        # x* has the full FE design at t*: [1, t*, cov_1, ..., cov_K] in fit order.
        if self._cov_beta is not None and self._cov_beta.shape[0] >= 2:
            n_fe = self._cov_beta.shape[0]
            cov_tail = np.zeros(n_fe - 2)
            if cov_vec is not None and self._active_cov_names:
                k = min(len(cov_vec), n_fe - 2)
                cov_tail[:k] = cov_vec[:k]
            X_pred_full = np.column_stack([np.ones(n_pred), t_pred, np.tile(cov_tail, (n_pred, 1))])
            fe_var = np.einsum("ij,jk,ik->i", X_pred_full, self._cov_beta, X_pred_full)
            fe_var = np.maximum(fe_var, 0.0)
        else:
            fe_var = np.zeros(n_pred)

        latent_var = np.array(
            [float(Z_pred[j] @ cov_post @ Z_pred[j]) + self._sigma_n_sq for j in range(n_pred)]
        )
        latent_var = latent_var + fe_var
        latent_var = np.maximum(latent_var, 1e-10)

        observable_var = latent_var + sv_pred
        observable_var = np.maximum(observable_var, 1e-10)

        std = np.sqrt(observable_var)

        return PredictionResult(
            mean=mean,
            variance=observable_var,
            lower_95=mean - 1.96 * std,
            upper_95=mean + 1.96 * std,
            metadata={
                "latent_variance": latent_var.tolist(),
                "observable_variance": observable_var.tolist(),
            },
        )

    def get_fixed_effects(self) -> list[tuple[float, float]]:
        """Return (beta_0, beta_1) for the single dimension.

        Returns:
            List with one (intercept, slope) tuple.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return [(float(self._beta[0]), float(self._beta[1]))]

    def get_covariate_effects(self) -> list[dict[str, float]]:
        """Return per-dimension covariate fixed effects.

        Returns:
            List with one dict mapping covariate name to beta.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        cov_betas = {}
        for k, cov_name in enumerate(self._active_cov_names):
            cov_betas[cov_name] = float(self._beta[2 + k])
        return [cov_betas]

    def get_active_covariate_names(self) -> list[str]:
        """Return the list of covariate names actually used during fitting."""
        return self._active_cov_names

    def get_covariate_means(self) -> dict[str, float]:
        """Return training-set covariate means."""
        return dict(self._cov_means)

    def name(self) -> str:
        """Human-readable model name."""
        cov_str = f", cov={self._active_cov_names}" if self._active_cov_names else ""
        return f"LMEHetero(method={self.method}{cov_str})"
