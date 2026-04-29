# src/growth/models/growth/hgp_hetero.py
"""Heteroscedastic Hierarchical GP growth model.

Decomposes trajectory as y_i(t) = m(t) + g_i(t) + epsilon_i(t) where:
- m(t) is the population mean (from LMEHetero or weighted Gompertz)
- g_i(t) is the per-patient residual GP
- epsilon_i(t) ~ N(0, sigma_n^2 + sigma_v^2(t)) is heteroscedastic noise

References:
    Rasmussen & Williams, *Gaussian Processes for Machine Learning*, 2006.
    Schulam & Saria, *Individualizing Predictions of Disease Trajectories*, 2015.
"""

import logging
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit

from growth.exceptions import UncertaintyPropagationError

from ._covariance_utils import condition_and_predict, matern52_kernel
from .base import FitResult, GrowthModel, PatientTrajectory, PredictionResult
from .covariate_utils import get_patient_covariate_vector
from .lme_hetero import LMEHeteroGrowthModel

logger = logging.getLogger(__name__)

VALID_MEAN_FUNCTIONS = ("linear", "gompertz")


class HGPHeteroModel(GrowthModel):
    """Heteroscedastic HGP with population mean + residual GP.

    For the linear mean, uses LMEHeteroGrowthModel internally.
    For Gompertz, uses weighted curve_fit with sigma=sqrt(sigma_v).

    Args:
        mean_function: ``"linear"`` (LMEHetero) or ``"gompertz"`` (weighted curve_fit).
        n_restarts: Number of random restarts for GP hyperparameter optimization.
        max_iter: Maximum L-BFGS-B iterations per restart.
        lengthscale_bounds: (lower, upper) for temporal lengthscale.
        signal_var_bounds: (lower, upper) for signal variance.
        noise_var_bounds: (lower, upper) for noise variance.
        seed: Random seed.
        floor_variance: Minimum allowed observation variance.
        use_covariates: Whether to use covariates in the LME mean function.
        covariate_names: List of covariate names.
        missing_strategy: Strategy for missing covariates.
    """

    def __init__(
        self,
        mean_function: Literal["linear", "gompertz"] = "linear",
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

        # Internal state
        self._lme_hetero: LMEHeteroGrowthModel | None = None
        self._gompertz_params: tuple[float, float, float] | None = None
        self._sf2: float = 1.0
        self._ell: float = 1.0
        self._sigma_n_sq: float = 0.01
        self._active_cov_names: list[str] = []
        self._cov_means: dict[str, float] = {}
        self._fitted: bool = False

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit population mean + residual GP on heteroscedastic data.

        Args:
            patients: Training trajectories with observation_variance set.

        Returns:
            FitResult with combined hyperparameters.
        """
        if len(patients) == 0:
            raise ValueError("Cannot fit with zero patients")

        for p in patients:
            if p.observation_variance is None:
                raise UncertaintyPropagationError(
                    f"HGPHetero requires observation_variance for {p.patient_id}"
                )
            assert p.obs_dim == 1, (
                f"HGPHetero requires obs_dim=1, got {p.obs_dim} for {p.patient_id}"
            )

        n_total = sum(p.n_timepoints for p in patients)

        # Step 1: Fit population mean
        hypers: dict[str, float] = {}

        if self.mean_function == "gompertz":
            self._fit_gompertz_mean(patients)
            hypers["gompertz_K"] = self._gompertz_params[0]
            hypers["gompertz_b"] = self._gompertz_params[1]
            hypers["gompertz_c"] = self._gompertz_params[2]
            self._active_cov_names = []
            self._cov_means = {}
        else:
            self._lme_hetero = LMEHeteroGrowthModel(
                n_restarts=self.n_restarts,
                max_iter=self.max_iter,
                seed=self.seed,
                floor_variance=self.floor_variance,
                use_covariates=self.use_covariates,
                covariate_names=self.covariate_names,
                missing_strategy=self.missing_strategy,
            )
            lme_fit = self._lme_hetero.fit(patients)
            hypers.update({f"lme_{k}": v for k, v in lme_fit.hyperparameters.items()})
            self._active_cov_names = self._lme_hetero.get_active_covariate_names()
            self._cov_means = self._lme_hetero.get_covariate_means()

        # Step 2: Compute residuals
        all_t = []
        all_resid = []
        all_sv = []

        for p in patients:
            t_i = p.times
            y_i = p.observations[:, 0]
            sv_i = np.maximum(p.observation_variance, self.floor_variance)
            pop_mean = self._compute_pop_mean(t_i, p)
            all_t.append(t_i)
            all_resid.append(y_i - pop_mean)
            all_sv.append(sv_i)

        t_pool = np.concatenate(all_t)
        r_pool = np.concatenate(all_resid)
        sv_pool = np.concatenate(all_sv)

        logger.info(f"HGPHetero fitting residual GP: {n_total} obs, mean={self.mean_function}")

        # Step 3: Fit residual GP via sklearn
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

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
        gpr.fit(t_pool[:, np.newaxis], r_pool)

        k = gpr.kernel_
        self._sf2 = float(k.k1.k1.constant_value)
        self._ell = float(k.k1.k2.length_scale)
        self._sigma_n_sq = float(k.k2.noise_level)
        self._fitted = True

        hypers["gp_signal_variance"] = self._sf2
        hypers["gp_lengthscale"] = self._ell
        hypers["gp_noise_variance"] = self._sigma_n_sq
        hypers["mean_function"] = 0.0 if self.mean_function == "linear" else 1.0

        lml = float(gpr.log_marginal_likelihood_value_)

        logger.info(
            f"HGPHetero fit: sf2={self._sf2:.4f}, ell={self._ell:.3f}, "
            f"sn2={self._sigma_n_sq:.6f}, LML={lml:.2f}"
        )

        return FitResult(
            log_marginal_likelihood=lml,
            hyperparameters=hypers,
            condition_number=0.0,
            n_train_patients=len(patients),
            n_train_observations=n_total,
        )

    def _fit_gompertz_mean(self, patients: list[PatientTrajectory]) -> None:
        """Fit Gompertz mean with inverse-variance weighting."""
        all_t, all_y, all_sigma = [], [], []
        for p in patients:
            all_t.append(p.times)
            all_y.append(p.observations[:, 0])
            all_sigma.append(np.sqrt(np.maximum(p.observation_variance, self.floor_variance)))

        t = np.concatenate(all_t)
        y = np.concatenate(all_y)
        sigma = np.concatenate(all_sigma)

        def gompertz(t_val: np.ndarray, K: float, b: float, c: float) -> np.ndarray:
            return K * np.exp(-b * np.exp(-c * t_val))

        try:
            y_max = float(np.max(y)) * 1.2
            popt, _ = curve_fit(
                gompertz,
                t,
                y,
                p0=[y_max, 1.0, 0.1],
                sigma=sigma,
                absolute_sigma=True,
                maxfev=self.max_iter * 10,
                bounds=([0.01, 0.001, 0.001], [y_max * 5, 100.0, 10.0]),
            )
            self._gompertz_params = (float(popt[0]), float(popt[1]), float(popt[2]))
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Gompertz fit failed ({e}), using linear fallback")
            A = np.column_stack([np.ones_like(t), t])
            params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            self._gompertz_params = None
            self._lme_hetero = None
            self.mean_function = "linear"
            self._lme_hetero = LMEHeteroGrowthModel(
                n_restarts=self.n_restarts,
                max_iter=self.max_iter,
                seed=self.seed,
                floor_variance=self.floor_variance,
            )
            self._lme_hetero.fit([p for p in []])  # Will be refitted below

    def _compute_pop_mean(
        self, t: np.ndarray, patient: PatientTrajectory | None = None
    ) -> np.ndarray:
        """Compute population mean at times t."""
        if self.mean_function == "gompertz" and self._gompertz_params is not None:
            K, b, c = self._gompertz_params
            return K * np.exp(-b * np.exp(-c * t))

        # Linear mean via LMEHetero BLUP (population mean only, no random effects)
        if self._lme_hetero is not None and self._lme_hetero._fitted:
            beta = self._lme_hetero._beta
            mean = beta[0] + beta[1] * t

            if patient is not None and self._active_cov_names:
                cov_vec = get_patient_covariate_vector(
                    patient, self._active_cov_names, self._cov_means
                )
                if cov_vec is not None:
                    for k, _ in enumerate(self._active_cov_names):
                        mean = mean + beta[2 + k] * cov_vec[k]
            return mean

        return np.zeros_like(t)

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict by conditioning residual GP on patient residuals.

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

        # Held-out variance
        if patient.observation_variance is not None and n_condition is not None:
            sv_pred = np.maximum(
                patient.observation_variance[n_condition : n_condition + n_pred],
                self.floor_variance,
            )
            if len(sv_pred) < n_pred:
                sv_pred = np.pad(sv_pred, (0, n_pred - len(sv_pred)), constant_values=0.0)
        else:
            sv_pred = np.zeros(n_pred)

        # Population mean
        pop_cond = self._compute_pop_mean(t_cond, patient)
        pop_pred = self._compute_pop_mean(t_pred, patient)

        # Residuals at conditioning times
        r_cond = y_cond - pop_cond

        # Condition residual GP
        K_train = matern52_kernel(t_cond, t_cond, self._sf2, self._ell)
        K_star = matern52_kernel(t_pred, t_cond, self._sf2, self._ell)
        K_star_star = np.full(n_pred, self._sf2)

        Sigma = self._sigma_n_sq + sv_cond

        mean_resid, var_obs = condition_and_predict(
            K_train, Sigma, r_cond, K_star, K_star_star, sv_pred, self._sigma_n_sq
        )

        mean = mean_resid + pop_pred
        latent_var = var_obs - sv_pred

        std = np.sqrt(np.maximum(var_obs, 0.0))

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
        return f"HGPHetero(matern52, mean={self.mean_function})"
