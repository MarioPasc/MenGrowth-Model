# src/growth/models/growth/hgp_model.py
"""Hierarchical Gaussian Process (H-GP) growth model (Model B).

Per-dimension GP on observations in R^D with:
- Population mean from LME (linear) or Gompertz (nonlinear)
- Matern-5/2 temporal kernel (default; SE and Matern-3/2 as ablation A9)
- Hierarchical hyperparameter sharing across patients (empirical Bayes, D19)

Posterior conditioning provides calibrated uncertainty that grows with
extrapolation distance and reverts to the population mean.

The ``mean_function`` parameter selects the population mean:
- ``"linear"`` (default): m_d(t) = b0_d + b1_d * t from LME (D18)
- ``"gompertz"``: m(t) = K*exp(-b*exp(-c*t)) fitted via curve_fit on
  pooled training data.  Only supported for D=1 (scalar volume).
  See Vaghi et al. (2020), Engelhardt et al. (2023).

Works for:
  - D=1: scalar volume baseline (alongside A0)
  - D=24: z_vol partition (Module 5, Model B)

References:
    Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006.
    Schulam & Saria, *Individualizing Predictions of Disease Trajectories*, NeurIPS 2015.
    Vaghi et al., *Population modeling of tumor growth curves*, PLOS Comp. Biol., 2020.
"""

import logging
from typing import Literal

import GPy
import numpy as np

from .base import FitResult, GrowthModel, PatientTrajectory, PredictionResult
from .covariate_utils import get_patient_covariate_vector
from .lme_model import LMEGrowthModel

logger = logging.getLogger(__name__)

VALID_KERNELS = ("matern52", "matern32", "se", "rbf")
VALID_MEAN_FUNCTIONS = ("linear", "gompertz")


class HierarchicalGPModel(GrowthModel):
    """Per-dimension GP with population mean (LME linear or Gompertz).

    For each dimension d in {0, ..., D-1}:
        z_d(t) ~ GP(m_d(t), k_d(t, t'))
        m_d(t) = population mean (linear or Gompertz)
        k_d = s2_f * Matern_5/2(t, t'; l_d) + s2_n * delta(t, t')

    Kernel hyperparameters (s_f, l, s_n) are fitted per-dimension by pooling
    all patients' **residuals** (observations minus population mean).

    Args:
        kernel_type: Temporal kernel type.
        mean_function: ``"linear"`` (LME) or ``"gompertz"`` (curve_fit).
            Gompertz is only supported for D=1.
        n_restarts: Number of random restarts for hyperparameter optimization.
        max_iter: Maximum L-BFGS-B iterations per restart.
        lengthscale_bounds: (lower, upper) for temporal lengthscale.
        signal_var_bounds: (lower, upper) for signal variance.
        noise_var_bounds: (lower, upper) for noise variance.
        seed: Random seed.
        use_covariates: Whether to include covariates in the mean function.
        covariate_names: List of covariate names to use.
        missing_strategy: Strategy for missing covariates.
    """

    def __init__(
        self,
        kernel_type: Literal["matern52", "matern32", "se", "rbf"] = "matern52",
        mean_function: Literal["linear", "gompertz"] = "linear",
        n_restarts: int = 5,
        max_iter: int = 1000,
        lengthscale_bounds: tuple[float, float] = (0.1, 50.0),
        signal_var_bounds: tuple[float, float] = (0.001, 10.0),
        noise_var_bounds: tuple[float, float] = (1e-6, 5.0),
        seed: int = 42,
        use_covariates: bool = False,
        covariate_names: list[str] | None = None,
        missing_strategy: str = "skip",
    ) -> None:
        if kernel_type not in VALID_KERNELS:
            raise ValueError(f"Invalid kernel_type '{kernel_type}'. Must be one of {VALID_KERNELS}")
        if mean_function not in VALID_MEAN_FUNCTIONS:
            raise ValueError(
                f"Invalid mean_function '{mean_function}'. Must be one of {VALID_MEAN_FUNCTIONS}"
            )

        self.kernel_type = kernel_type
        self.mean_function = mean_function
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.lengthscale_bounds = lengthscale_bounds
        self.signal_var_bounds = signal_var_bounds
        self.noise_var_bounds = noise_var_bounds
        self.seed = seed
        self.use_covariates = use_covariates
        self.covariate_names = covariate_names or []
        self.missing_strategy = missing_strategy

        self._obs_dim: int = 0
        self._lme_effects: list[tuple[float, float]] = []  # (beta_0_d, beta_1_d)
        self._cov_effects: list[dict[str, float]] = []  # per-dim covariate betas
        self._active_cov_names: list[str] = []
        self._cov_means: dict[str, float] = {}
        self._gompertz_fn: object | None = None  # GompertzMeanFunction when active
        self._gpy_models: list[GPy.models.GPRegression] = []
        self._fitted: bool = False

    def _build_kernel(self) -> GPy.kern.Kern:
        """Construct GPy kernel."""
        if self.kernel_type == "matern52":
            return GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
        elif self.kernel_type == "matern32":
            return GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=1.0)
        elif self.kernel_type in ("se", "rbf"):
            return GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel_type}")

    def _apply_bounds(self, model: GPy.models.GPRegression) -> None:
        """Apply hyperparameter bounds."""
        model.kern.lengthscale.constrain_bounded(*self.lengthscale_bounds)
        model.kern.variance.constrain_bounded(*self.signal_var_bounds)
        model.Gaussian_noise.variance.constrain_bounded(*self.noise_var_bounds)

    def _fit_gompertz_mean(self, patients: list[PatientTrajectory]) -> None:
        """Fit Gompertz mean function on pooled training data.

        Only valid for D=1 (scalar volume).  Sets ``self._gompertz_fn``.
        """
        from growth.stages.stage1_volumetric.gompertz import (
            GompertzMeanFunction,
            fit_gompertz,
        )

        all_t = []
        all_y = []
        for p in patients:
            for j in range(p.n_timepoints):
                all_t.append(p.times[j])
                all_y.append(p.observations[j, 0])

        times = np.array(all_t)
        volumes = np.array(all_y)

        params = fit_gompertz(times, volumes, max_iter=self.max_iter)
        self._gompertz_fn = GompertzMeanFunction(params)
        logger.info(f"Gompertz mean fitted: {self._gompertz_fn}")

    def _compute_pop_mean(
        self, t: np.ndarray, d: int, cov_vec: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute population mean at times t for dimension d.

        Dispatches to LME linear or Gompertz based on ``self.mean_function``.

        Args:
            t: Times, shape ``[N]``.
            d: Dimension index.
            cov_vec: Covariate vector for the patient (or None).

        Returns:
            Population mean values, shape ``[N]``.
        """
        if self.mean_function == "gompertz" and self._gompertz_fn is not None:
            return self._gompertz_fn(t)

        # Default: LME linear mean
        beta_0, beta_1 = self._lme_effects[d]
        pop_mean = beta_0 + beta_1 * t

        if cov_vec is not None and d < len(self._cov_effects) and self._cov_effects[d]:
            for k, name in enumerate(self._active_cov_names):
                pop_mean = pop_mean + self._cov_effects[d].get(name, 0.0) * cov_vec[k]

        return pop_mean

    def fit(
        self,
        patients: list[PatientTrajectory],
        lme_model: LMEGrowthModel | None = None,
    ) -> FitResult:
        """Fit per-dimension GPs on residuals after subtracting population mean.

        If ``mean_function == "gompertz"`` and D=1, fits Gompertz via curve_fit.
        Otherwise uses LME fixed effects.

        If ``lme_model`` is not provided (for linear mean), an LME is fitted
        internally.

        Args:
            patients: Training patient trajectories.
            lme_model: Pre-fitted LME model. If None, one is fitted internally.
                Ignored when ``mean_function == "gompertz"``.

        Returns:
            FitResult with per-dimension hyperparameters.
        """
        if len(patients) == 0:
            raise ValueError("Cannot fit with zero patients")

        # GPy's optimize_restarts() uses the global numpy RNG; cannot use
        # np.random.default_rng() here without patching GPy internals.
        np.random.seed(self.seed)
        self._obs_dim = patients[0].obs_dim
        n_total = sum(p.n_timepoints for p in patients)

        # Step 1: Fit population mean function
        if self.mean_function == "gompertz":
            if self._obs_dim != 1:
                raise ValueError(f"Gompertz mean function only supports D=1, got D={self._obs_dim}")
            self._fit_gompertz_mean(patients)
            # Still need LME effects as fallback storage (unused for mean)
            self._lme_effects = [(0.0, 0.0)] * self._obs_dim
            self._cov_effects = [{}] * self._obs_dim
            self._active_cov_names = []
            self._cov_means = {}
        else:
            # Linear mean via LME
            if lme_model is not None and lme_model._fitted:
                self._lme_effects = lme_model.get_fixed_effects()
                self._cov_effects = lme_model.get_covariate_effects()
                self._active_cov_names = lme_model.get_active_covariate_names()
                self._cov_means = lme_model.get_covariate_means()
            else:
                logger.info("No pre-fitted LME provided; fitting internally")
                lme = LMEGrowthModel(
                    use_covariates=self.use_covariates,
                    covariate_names=self.covariate_names,
                    missing_strategy=self.missing_strategy,
                )
                lme.fit(patients)
                self._lme_effects = lme.get_fixed_effects()
                self._cov_effects = lme.get_covariate_effects()
                self._active_cov_names = lme.get_active_covariate_names()
                self._cov_means = lme.get_covariate_means()

            assert len(self._lme_effects) == self._obs_dim

        logger.info(
            f"H-GP fitting: {len(patients)} patients, {n_total} obs, "
            f"D={self._obs_dim}, kernel={self.kernel_type}, "
            f"mean={self.mean_function}"
        )

        # Step 2: Pool data and subtract population mean per dimension
        all_t: list[float] = []
        all_residuals: list[list[float]] = [[] for _ in range(self._obs_dim)]

        for p in patients:
            cov_vec = get_patient_covariate_vector(p, self._active_cov_names, self._cov_means)
            for j in range(p.n_timepoints):
                t = p.times[j]
                all_t.append(t)
                for d in range(self._obs_dim):
                    pop_mean = self._compute_pop_mean(np.array([t]), d, cov_vec)[0]
                    all_residuals[d].append(p.observations[j, d] - pop_mean)

        X_pool = np.array(all_t)[:, np.newaxis]  # [N, 1]

        # Step 3: Fit one GP per dimension on residuals
        self._gpy_models = []
        hypers: dict[str, float] = {}
        total_lml = 0.0

        for d in range(self._obs_dim):
            Y_d = np.array(all_residuals[d])[:, np.newaxis]  # [N, 1]
            kern = self._build_kernel()

            # Zero mean for residuals (population mean already subtracted)
            model = GPy.models.GPRegression(X_pool, Y_d, kernel=kern)
            self._apply_bounds(model)

            model.optimize_restarts(
                num_restarts=self.n_restarts,
                max_iters=self.max_iter,
                verbose=False,
                robust=True,
            )

            self._gpy_models.append(model)
            total_lml += float(model.log_likelihood())

            hypers[f"lengthscale_d{d}"] = float(model.kern.lengthscale)
            hypers[f"signal_var_d{d}"] = float(model.kern.variance)
            hypers[f"noise_var_d{d}"] = float(model.Gaussian_noise.variance)

        # Add mean function info to hyperparameters
        hypers["mean_function"] = 0.0 if self.mean_function == "linear" else 1.0
        if self.mean_function == "gompertz" and self._gompertz_fn is not None:
            hypers["gompertz_K"] = self._gompertz_fn.params.K
            hypers["gompertz_b"] = self._gompertz_fn.params.b
            hypers["gompertz_c"] = self._gompertz_fn.params.c

        self._fitted = True

        logger.info(
            f"H-GP fit: total LML={total_lml:.2f}, "
            f"mean l={np.mean([float(m.kern.lengthscale) for m in self._gpy_models]):.3f}"
        )

        return FitResult(
            log_marginal_likelihood=total_lml,
            hyperparameters=hypers,
            condition_number=0.0,
            n_train_patients=len(patients),
            n_train_observations=n_total,
        )

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict by conditioning per-dimension GPs on patient residuals.

        For each dimension d:
            1. Compute residuals: r = y_cond - m_d(t_cond)
            2. Condition GP_d on (t_cond, r)
            3. Posterior mean + m_d(t_pred) gives final prediction

        Args:
            patient: Patient trajectory.
            t_pred: Query times [n_pred].
            n_condition: Use first n_condition observations only.

        Returns:
            PredictionResult with mean, variance, and 95% CI.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        t_pred = np.asarray(t_pred, dtype=np.float64)
        if t_pred.ndim == 0:
            t_pred = t_pred[np.newaxis]

        if n_condition is not None:
            t_cond = patient.times[:n_condition]
            y_cond = patient.observations[:n_condition]
        else:
            t_cond = patient.times
            y_cond = patient.observations

        n_pred = len(t_pred)
        D = self._obs_dim

        # Get covariate vector for this patient
        cov_vec = get_patient_covariate_vector(patient, self._active_cov_names, self._cov_means)

        mean = np.zeros((n_pred, D))
        variance = np.zeros((n_pred, D))

        X_cond = t_cond[:, np.newaxis]
        X_pred = t_pred[:, np.newaxis]

        for d in range(D):
            # Population mean at conditioning and prediction times
            pop_cond = self._compute_pop_mean(t_cond, d, cov_vec)
            pop_pred = self._compute_pop_mean(t_pred, d, cov_vec)

            # Residuals at conditioning times
            r_cond = (y_cond[:, d] - pop_cond)[:, np.newaxis]

            # Build conditioned GP with fitted hyperparameters
            fitted_model = self._gpy_models[d]
            kern_copy = fitted_model.kern.copy()
            cond_model = GPy.models.GPRegression(X_cond, r_cond, kernel=kern_copy)
            cond_model.Gaussian_noise.variance = float(fitted_model.Gaussian_noise.variance)
            cond_model.Gaussian_noise.variance.fix()

            # Predict residuals
            mu_r, var_r = cond_model.predict(X_pred)

            # Add population mean back
            mean[:, d] = mu_r[:, 0] + pop_pred
            variance[:, d] = np.maximum(var_r[:, 0], 0.0)

        std = np.sqrt(variance)
        return PredictionResult(
            mean=mean,
            variance=variance,
            lower_95=mean - 1.96 * std,
            upper_95=mean + 1.96 * std,
        )

    def name(self) -> str:
        """Human-readable model name."""
        mean_str = self.mean_function
        return f"H-GP(D={self._obs_dim}, kernel={self.kernel_type}, mean={mean_str})"
