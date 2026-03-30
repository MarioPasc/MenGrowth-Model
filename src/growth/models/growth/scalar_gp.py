# src/growth/models/growth/scalar_gp.py
"""Hierarchical scalar Gaussian Process for 1-D growth prediction.

Wraps GPy to provide a pooled-fitting, per-patient-conditioning GP.
Shared kernel hyperparameters are estimated from all training patients
(empirical Bayes), then individual posteriors provide personalized
predictions for held-out patients.

References:
    Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006.
    Schulam & Saria, *Individualizing Predictions of Disease Trajectories*, NeurIPS 2015.
"""

import logging
from typing import Literal

import GPy
import numpy as np

from .base import FitResult, GrowthModel, PatientTrajectory, PredictionResult
from .covariate_utils import (
    collect_covariates,
    get_patient_covariate_vector,
)

logger = logging.getLogger(__name__)

VALID_KERNELS = ("matern52", "matern32", "se", "rbf")
VALID_MEAN_FUNCTIONS = ("linear", "constant", "zero")


class ScalarGP(GrowthModel):
    """Hierarchical scalar GP with pooled hyperparameters.

    Fits a single GP to all training patients' data concatenated, yielding
    shared kernel hyperparameters.  At prediction time, the GP is conditioned
    on the target patient's observations to produce a personalized posterior.

    Args:
        kernel_type: Temporal kernel. One of ``"matern52"``, ``"matern32"``,
            ``"se"`` / ``"rbf"``.
        mean_function: GP mean. ``"linear"`` fits ``m(t) = a*t + b``,
            ``"constant"`` fits ``m(t) = c``, ``"zero"`` uses ``m(t) = 0``.
        n_restarts: Number of random restarts for hyperparameter optimization.
        max_iter: Maximum L-BFGS-B iterations per restart.
        lengthscale_bounds: ``(lower, upper)`` bounds for the temporal lengthscale.
        signal_var_bounds: ``(lower, upper)`` bounds for the signal variance.
        noise_var_bounds: ``(lower, upper)`` bounds for the noise variance.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        kernel_type: Literal["matern52", "matern32", "se", "rbf"] = "matern52",
        mean_function: Literal["linear", "constant", "zero"] = "linear",
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

        # Populated by fit()
        self._gpy_model: GPy.models.GPRegression | None = None
        self._fit_result: FitResult | None = None
        # Covariate regression (two-stage residualization)
        self._cov_alpha: float = 0.0  # Intercept of linear regression
        self._cov_gammas: np.ndarray | None = None  # Covariate coefficients
        self._active_cov_names: list[str] = []
        self._cov_means: dict[str, float] = {}

    def _build_kernel(self) -> GPy.kern.Kern:
        """Construct the GPy kernel object."""
        if self.kernel_type == "matern52":
            kern = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
        elif self.kernel_type == "matern32":
            kern = GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=1.0)
        elif self.kernel_type in ("se", "rbf"):
            kern = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel_type}")
        return kern

    def _build_mean_function(self) -> object | None:
        """Construct the GPy mean function, or None for zero mean."""
        if self.mean_function == "linear":
            return GPy.mappings.Linear(input_dim=1, output_dim=1)
        elif self.mean_function == "constant":
            return GPy.mappings.Constant(input_dim=1, output_dim=1)
        elif self.mean_function == "zero":
            return None
        else:
            raise ValueError(f"Unsupported mean function: {self.mean_function}")

    def _apply_bounds(self, model: GPy.models.GPRegression) -> None:
        """Apply hyperparameter bounds to the GPy model."""
        lb_ls, ub_ls = self.lengthscale_bounds
        lb_sv, ub_sv = self.signal_var_bounds
        lb_nv, ub_nv = self.noise_var_bounds

        model.kern.lengthscale.constrain_bounded(lb_ls, ub_ls)
        model.kern.variance.constrain_bounded(lb_sv, ub_sv)
        model.Gaussian_noise.variance.constrain_bounded(lb_nv, ub_nv)

    def _pool_data(self, patients: list[PatientTrajectory]) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate all patient data into pooled arrays.

        Returns:
            (X, Y) where X is ``[N_total, 1]`` and Y is ``[N_total, 1]``.
        """
        all_t: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for p in patients:
            assert p.obs_dim == 1, (
                f"ScalarGP requires obs_dim=1, got {p.obs_dim} for {p.patient_id}"
            )
            all_t.append(p.times)
            all_y.append(p.observations[:, 0])
        X = np.concatenate(all_t)[:, np.newaxis]
        Y = np.concatenate(all_y)[:, np.newaxis]
        return X, Y

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit shared hyperparameters by pooling all training data.

        If covariates are enabled, uses a two-stage approach:
        1. Fit linear regression: y = alpha + sum(gamma_k * x_k) + r
        2. GP operates on residuals r

        Args:
            patients: Training patient trajectories (scalar observations, D=1).

        Returns:
            FitResult with optimized hyperparameters.
        """
        if len(patients) == 0:
            raise ValueError("Cannot fit with zero patients")

        # GPy's optimize_restarts() uses the global numpy RNG; cannot use
        # np.random.default_rng() here without patching GPy internals.
        np.random.seed(self.seed)

        # Collect covariates if enabled
        cov_values: dict[str, np.ndarray] = {}
        self._active_cov_names = []
        if self.use_covariates and self.covariate_names:
            cov_values, self._active_cov_names, patients = collect_covariates(
                patients, self.covariate_names, self.missing_strategy
            )
            for i, name in enumerate(self._active_cov_names):
                vals = [v[i] for v in cov_values.values()]
                self._cov_means[name] = float(np.mean(vals)) if vals else 0.0

        X, Y = self._pool_data(patients)
        n_total = X.shape[0]

        logger.info(
            f"ScalarGP fitting: {len(patients)} patients, {n_total} observations, "
            f"covariates={self._active_cov_names}"
        )

        # Stage 1: If covariates present, residualize Y
        if self._active_cov_names and cov_values:
            Y = self._residualize(patients, Y, cov_values)

        kern = self._build_kernel()
        mean_fn = self._build_mean_function()

        model = GPy.models.GPRegression(X, Y, kernel=kern, mean_function=mean_fn)
        self._apply_bounds(model)

        # Optimize with restarts
        model.optimize_restarts(
            num_restarts=self.n_restarts,
            max_iters=self.max_iter,
            verbose=False,
            robust=True,
        )

        self._gpy_model = model

        # Extract hyperparameters
        hypers: dict[str, float] = {
            "lengthscale": float(model.kern.lengthscale),
            "signal_variance": float(model.kern.variance),
            "noise_variance": float(model.Gaussian_noise.variance),
        }
        if mean_fn is not None:
            hypers["mean_params"] = model.mean_function.param_array.tolist()
        if self._cov_gammas is not None:
            for i, name in enumerate(self._active_cov_names):
                hypers[f"cov_gamma_{name}"] = float(self._cov_gammas[i])
            hypers["cov_alpha"] = self._cov_alpha

        # Condition number of the kernel matrix
        K = model.kern.K(X) + float(model.Gaussian_noise.variance) * np.eye(n_total)
        eigvals = np.linalg.eigvalsh(K)
        cond = float(eigvals[-1] / max(eigvals[0], 1e-15))

        lml = float(model.log_likelihood())

        self._fit_result = FitResult(
            log_marginal_likelihood=lml,
            hyperparameters=hypers,
            condition_number=cond,
            n_train_patients=len(patients),
            n_train_observations=n_total,
        )

        logger.info(
            f"ScalarGP fit: LML={lml:.2f}, ls={hypers['lengthscale']:.3f}, "
            f"sf2={hypers['signal_variance']:.4f}, sn2={hypers['noise_variance']:.6f}, "
            f"cond={cond:.1f}"
        )

        return self._fit_result

    def _residualize(
        self,
        patients: list[PatientTrajectory],
        Y: np.ndarray,
        cov_values: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Two-stage residualization: regress out covariate effects from Y.

        Fits y_ij = alpha + sum(gamma_k * x_ik) + r_ij via OLS on per-patient
        mean observations, then returns residuals r.

        Args:
            patients: Patient trajectories.
            Y: Pooled observations [N_total, 1].
            cov_values: Dict mapping patient_id to covariate array.

        Returns:
            Residualized Y [N_total, 1].
        """
        n_cov = len(self._active_cov_names)

        # Build per-observation covariate matrix
        cov_rows: list[np.ndarray] = []
        for p in patients:
            cov = cov_values.get(p.patient_id)
            if cov is not None:
                for _ in range(p.n_timepoints):
                    cov_rows.append(cov)
            else:
                for _ in range(p.n_timepoints):
                    cov_rows.append(np.zeros(n_cov))

        C = np.array(cov_rows)  # [N_total, n_cov]
        A = np.column_stack([np.ones(len(C)), C])  # [N_total, 1 + n_cov]

        # OLS: y = [1, C] @ [alpha, gamma] + r
        params, _, _, _ = np.linalg.lstsq(A, Y[:, 0], rcond=None)
        self._cov_alpha = float(params[0])
        self._cov_gammas = params[1:]

        # Compute residuals
        Y_pred = (A @ params)[:, np.newaxis]
        residuals = Y - Y_pred

        logger.info(
            f"ScalarGP residualization: alpha={self._cov_alpha:.4f}, "
            f"gammas={dict(zip(self._active_cov_names, self._cov_gammas.tolist()))}"
        )
        return residuals

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict at query times by conditioning on patient's observations.

        The fitted GP (with pooled hyperparameters) is conditioned on the
        patient's data to produce a personalized posterior. If covariates
        were used during fitting, observations are residualized before
        conditioning and the covariate contribution is added back.

        Args:
            patient: Patient trajectory (scalar observations, D=1).
            t_pred: Query times, shape ``[n_pred]``.
            n_condition: Condition on first ``n_condition`` observations only.
                If None, use all observations.

        Returns:
            PredictionResult with mean, variance, and 95% CI.
        """
        if self._gpy_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        t_pred = np.asarray(t_pred, dtype=np.float64)
        if t_pred.ndim == 0:
            t_pred = t_pred[np.newaxis]

        # Extract conditioning data
        if n_condition is not None:
            t_cond = patient.times[:n_condition]
            y_cond = patient.observations[:n_condition, 0]
        else:
            t_cond = patient.times
            y_cond = patient.observations[:, 0]

        # Compute covariate offset for this patient
        cov_offset = 0.0
        if self._cov_gammas is not None and self._active_cov_names:
            cov_vec = get_patient_covariate_vector(
                patient, self._active_cov_names, self._cov_means
            )
            if cov_vec is not None:
                cov_offset = self._cov_alpha + float(self._cov_gammas @ cov_vec)

        # Residualize conditioning data if covariates were used
        Y_cond = (y_cond - cov_offset)[:, np.newaxis]

        X_cond = t_cond[:, np.newaxis]
        X_pred = t_pred[:, np.newaxis]

        # Build a new GP with the same hyperparameters but conditioned on this patient
        kern_copy = self._gpy_model.kern.copy()
        mean_fn_copy = None
        if self._gpy_model.mean_function is not None:
            mean_fn_copy = self._gpy_model.mean_function.copy()

        cond_model = GPy.models.GPRegression(
            X_cond, Y_cond, kernel=kern_copy, mean_function=mean_fn_copy
        )
        # Fix noise variance to the fitted value
        cond_model.Gaussian_noise.variance = float(self._gpy_model.Gaussian_noise.variance)
        cond_model.Gaussian_noise.variance.fix()

        # Predict residuals
        mu, var = cond_model.predict(X_pred)

        # Add covariate offset back
        mu = mu + cov_offset

        # Ensure variance is non-negative
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)

        return PredictionResult(
            mean=mu,
            variance=var,
            lower_95=mu - 1.96 * std,
            upper_95=mu + 1.96 * std,
        )

    def name(self) -> str:
        """Human-readable model name."""
        return f"ScalarGP({self.kernel_type}, mean={self.mean_function})"
