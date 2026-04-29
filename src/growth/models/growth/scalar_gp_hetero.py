# src/growth/models/growth/scalar_gp_hetero.py
"""Heteroscedastic scalar Gaussian Process for 1-D growth prediction.

Uses sklearn GaussianProcessRegressor with per-observation noise via
alpha=array and WhiteKernel for biological noise. Per-patient conditioning
uses the closed-form GP posterior (Rasmussen & Williams 2006, §2.2).

References:
    Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006.
    Goldberg et al., *Regression with input-dependent noise*, NIPS 1998.
"""

import logging
from typing import Literal

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from growth.exceptions import UncertaintyPropagationError

from ._covariance_utils import condition_and_predict, matern52_kernel
from .base import FitResult, GrowthModel, PatientTrajectory, PredictionResult
from .covariate_utils import collect_covariates, get_patient_covariate_vector

logger = logging.getLogger(__name__)

VALID_MEAN_FUNCTIONS = ("linear", "constant", "zero")


class ScalarGPHetero(GrowthModel):
    """Heteroscedastic scalar GP with pooled hyperparameters.

    Fits a GP to all training patients' data with per-observation known noise
    from the segmentation ensemble. At prediction time, the GP is conditioned
    on the target patient's observations using the closed-form posterior.

    Args:
        mean_function: GP mean. ``"linear"`` fits m(t) = a*t + b via OLS,
            ``"constant"`` uses m(t) = c, ``"zero"`` uses m(t) = 0.
        n_restarts: Number of random restarts for hyperparameter optimization.
        max_iter: Maximum L-BFGS-B iterations per restart.
        lengthscale_bounds: (lower, upper) bounds for temporal lengthscale.
        signal_var_bounds: (lower, upper) bounds for signal variance.
        noise_var_bounds: (lower, upper) bounds for biological noise variance.
        seed: Random seed for reproducibility.
        floor_variance: Minimum allowed observation variance.
        use_covariates: Whether to use covariates for mean residualization.
        covariate_names: List of covariate names.
        missing_strategy: Strategy for missing covariates.
    """

    def __init__(
        self,
        mean_function: Literal["linear", "constant", "zero"] = "linear",
        n_restarts: int = 5,
        max_iter: int = 1000,
        lengthscale_bounds: tuple[float, float] = (0.1, 50.0),
        signal_var_bounds: tuple[float, float] = (0.001, 10.0),
        noise_var_bounds: tuple[float, float] = (1e-6, 5.0),
        seed: int = 42,
        floor_variance: float = 1e-6,
        use_covariates: bool = False,
        covariate_names: list[str] | None = None,
        missing_strategy: str = "skip",
    ) -> None:
        if mean_function not in VALID_MEAN_FUNCTIONS:
            raise ValueError(
                f"Invalid mean_function '{mean_function}'. Must be one of {VALID_MEAN_FUNCTIONS}"
            )

        self.mean_function = mean_function
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.lengthscale_bounds = lengthscale_bounds
        self.signal_var_bounds = signal_var_bounds
        self.noise_var_bounds = noise_var_bounds
        self.seed = seed
        self.floor_variance = floor_variance
        self.use_covariates = use_covariates
        self.covariate_names = covariate_names or []
        self.missing_strategy = missing_strategy

        # Populated by fit()
        self._sf2: float = 1.0
        self._ell: float = 1.0
        self._sigma_n_sq: float = 0.01
        self._mean_a: float = 0.0
        self._mean_b: float = 0.0
        self._mean_c: float = 0.0
        self._fitted: bool = False
        self._cov_alpha: float = 0.0
        self._cov_gammas: np.ndarray | None = None
        self._active_cov_names: list[str] = []
        self._cov_means: dict[str, float] = {}

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit shared hyperparameters by pooling all training data.

        Args:
            patients: Training patient trajectories (scalar, D=1) with
                observation_variance set.

        Returns:
            FitResult with optimized hyperparameters.
        """
        if len(patients) == 0:
            raise ValueError("Cannot fit with zero patients")

        for p in patients:
            if p.observation_variance is None:
                raise UncertaintyPropagationError(
                    f"ScalarGPHetero requires observation_variance for {p.patient_id}"
                )
            assert p.obs_dim == 1, (
                f"ScalarGPHetero requires obs_dim=1, got {p.obs_dim} for {p.patient_id}"
            )

        # Collect covariates
        cov_values: dict[str, np.ndarray] = {}
        self._active_cov_names = []
        if self.use_covariates and self.covariate_names:
            cov_values, self._active_cov_names, patients = collect_covariates(
                patients, self.covariate_names, self.missing_strategy
            )
            for i, cov_name in enumerate(self._active_cov_names):
                vals = [v[i] for v in cov_values.values()]
                self._cov_means[cov_name] = float(np.mean(vals)) if vals else 0.0

        # Pool data
        all_t, all_y, all_sv = [], [], []
        for p in patients:
            all_t.append(p.times)
            all_y.append(p.observations[:, 0])
            all_sv.append(np.maximum(p.observation_variance, self.floor_variance))

        X_pool = np.concatenate(all_t)[:, np.newaxis]
        Y_pool = np.concatenate(all_y)
        sv_pool = np.concatenate(all_sv)
        n_total = len(Y_pool)

        logger.info(
            f"ScalarGPHetero fitting: {len(patients)} patients, {n_total} obs, "
            f"covariates={self._active_cov_names}"
        )

        # Residualize for covariates
        if self._active_cov_names and cov_values:
            Y_pool = self._residualize(patients, Y_pool, cov_values)

        # Fit mean function and subtract
        Y_resid = self._fit_and_subtract_mean(X_pool[:, 0], Y_pool)

        # Build sklearn kernel
        kernel = ConstantKernel(1.0, constant_value_bounds=self.signal_var_bounds) * Matern(
            length_scale=1.0, nu=2.5, length_scale_bounds=self.lengthscale_bounds
        ) + WhiteKernel(noise_level=0.01, noise_level_bounds=self.noise_var_bounds)

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=sv_pool,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=self.n_restarts,
            random_state=self.seed,
            normalize_y=False,
        )
        gpr.fit(X_pool, Y_resid)

        # Extract fitted hyperparameters from kernel_
        k = gpr.kernel_
        self._sf2 = float(k.k1.k1.constant_value)
        self._ell = float(k.k1.k2.length_scale)
        self._sigma_n_sq = float(k.k2.noise_level)
        self._fitted = True

        # Condition number
        K = matern52_kernel(X_pool[:, 0], X_pool[:, 0], self._sf2, self._ell)
        diag_noise = self._sigma_n_sq + sv_pool
        K_noisy = K + np.diag(diag_noise)
        eigvals = np.linalg.eigvalsh(K_noisy)
        cond = float(eigvals[-1] / max(eigvals[0], 1e-15))

        lml = float(gpr.log_marginal_likelihood_value_)

        hypers: dict[str, float] = {
            "signal_variance": self._sf2,
            "lengthscale": self._ell,
            "noise_variance": self._sigma_n_sq,
        }
        if self.mean_function == "linear":
            hypers["mean_a"] = self._mean_a
            hypers["mean_b"] = self._mean_b
        elif self.mean_function == "constant":
            hypers["mean_c"] = self._mean_c
        if self._cov_gammas is not None:
            for i, cov_name in enumerate(self._active_cov_names):
                hypers[f"cov_gamma_{cov_name}"] = float(self._cov_gammas[i])
            hypers["cov_alpha"] = self._cov_alpha

        logger.info(
            f"ScalarGPHetero fit: LML={lml:.2f}, sf2={self._sf2:.4f}, "
            f"ell={self._ell:.3f}, sn2={self._sigma_n_sq:.6f}, cond={cond:.1f}"
        )

        return FitResult(
            log_marginal_likelihood=lml,
            hyperparameters=hypers,
            condition_number=cond,
            n_train_patients=len(patients),
            n_train_observations=n_total,
        )

    def _fit_and_subtract_mean(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit mean function and return residuals."""
        if self.mean_function == "linear":
            A = np.column_stack([t, np.ones_like(t)])
            params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            self._mean_a = float(params[0])
            self._mean_b = float(params[1])
            return y - (self._mean_a * t + self._mean_b)
        elif self.mean_function == "constant":
            self._mean_c = float(np.mean(y))
            return y - self._mean_c
        else:
            return y.copy()

    def _eval_mean(self, t: np.ndarray) -> np.ndarray:
        """Evaluate mean function at times t."""
        if self.mean_function == "linear":
            return self._mean_a * t + self._mean_b
        elif self.mean_function == "constant":
            return np.full_like(t, self._mean_c)
        else:
            return np.zeros_like(t)

    def _residualize(
        self,
        patients: list[PatientTrajectory],
        Y: np.ndarray,
        cov_values: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Two-stage residualization: regress out covariate effects."""
        n_cov = len(self._active_cov_names)
        cov_rows: list[np.ndarray] = []
        for p in patients:
            cov = cov_values.get(p.patient_id)
            if cov is not None:
                for _ in range(p.n_timepoints):
                    cov_rows.append(cov)
            else:
                for _ in range(p.n_timepoints):
                    cov_rows.append(np.zeros(n_cov))

        C = np.array(cov_rows)
        A = np.column_stack([np.ones(len(C)), C])
        params, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        self._cov_alpha = float(params[0])
        self._cov_gammas = params[1:]
        return Y - A @ params

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict via closed-form GP posterior with heteroscedastic noise.

        Args:
            patient: Patient trajectory (scalar, D=1).
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

        # Held-out variance for prediction times (regime b)
        if patient.observation_variance is not None and n_condition is not None:
            sv_pred = np.maximum(
                patient.observation_variance[n_condition : n_condition + n_pred],
                self.floor_variance,
            )
            if len(sv_pred) < n_pred:
                sv_pred = np.pad(sv_pred, (0, n_pred - len(sv_pred)), constant_values=0.0)
        else:
            sv_pred = np.zeros(n_pred)

        # Covariate offset
        cov_offset = 0.0
        if self._cov_gammas is not None and self._active_cov_names:
            cov_vec = get_patient_covariate_vector(patient, self._active_cov_names, self._cov_means)
            if cov_vec is not None:
                cov_offset = self._cov_alpha + float(self._cov_gammas @ cov_vec)

        # Residualize conditioning data
        y_resid = y_cond - cov_offset - self._eval_mean(t_cond)

        # Build kernel matrices
        K_train = matern52_kernel(t_cond, t_cond, self._sf2, self._ell)
        K_star = matern52_kernel(t_pred, t_cond, self._sf2, self._ell)
        K_star_star = np.full(n_pred, self._sf2)

        Sigma = self._sigma_n_sq + sv_cond

        mean_resid, var_obs = condition_and_predict(
            K_train, Sigma, y_resid, K_star, K_star_star, sv_pred, self._sigma_n_sq
        )

        # Add mean function and covariate offset back
        mean = mean_resid + self._eval_mean(t_pred) + cov_offset

        std = np.sqrt(np.maximum(var_obs, 0.0))
        latent_var = var_obs - sv_pred

        return PredictionResult(
            mean=mean,
            variance=var_obs,
            lower_95=mean - 1.96 * std,
            upper_95=mean + 1.96 * std,
            metadata={
                "latent_variance": np.maximum(latent_var, 1e-10).tolist(),
                "observable_variance": var_obs.tolist(),
            },
        )

    def name(self) -> str:
        """Human-readable model name."""
        return f"ScalarGPHetero(matern52, mean={self.mean_function})"
