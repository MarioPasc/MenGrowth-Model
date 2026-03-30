# src/growth/stages/stage2_severity/severity_model.py
"""Latent severity NLME growth model — Option A: MLE via L-BFGS-B.

Implements a nonlinear mixed-effects model where a single latent variable
s_i in [0, 1] governs each patient's growth trajectory:

    q_ij = g(s_i, t_ij; theta) + eps_ij

where q_ij is the growth quantile, s_i is the latent severity, t_ij is the
normalized time, and g is a monotonic growth function satisfying g(s, 0) = 0.

Severity is estimated jointly with population parameters theta via L-BFGS-B
during training (Approach A), and predicted from baseline features at test
time via a linear regression head (D25).

References:
    - Vaghi et al. (2020) PLOS Computational Biology
    - Proust-Lima et al. (2014, 2023) lcmm R package

Spec: ``docs/stages/stage_2_severity_model.md``
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from growth.shared.growth_models import (
    FitResult,
    GrowthModel,
    PatientTrajectory,
    PredictionResult,
)
from growth.stages.stage2_severity.growth_functions import (
    GrowthFunction,
    GrowthFunctionRegistry,
)
from growth.stages.stage2_severity.quantile_transform import QuantileTransform
from growth.stages.stage2_severity.severity_regression import SeverityRegressionHead

logger = logging.getLogger(__name__)


@dataclass
class SeverityFitResult:
    """Extended fit result with severity-specific diagnostics.

    Args:
        severities: Estimated severity per patient, dict mapping patient_id to s_i.
        population_params: Fitted population parameters theta.
        growth_function_name: Name of the growth function used.
        final_loss: Final optimization loss value.
        converged: Whether the optimizer converged.
    """

    severities: dict[str, float] = field(default_factory=dict)
    population_params: np.ndarray = field(default_factory=lambda: np.array([]))
    growth_function_name: str = ""
    final_loss: float = float("inf")
    converged: bool = False


class SeverityModel(GrowthModel):
    """Latent severity NLME model for growth prediction (MLE).

    Each patient is assigned a latent severity s_i in [0, 1] that, together
    with the shared population parameters theta, determines the growth
    trajectory via a monotonic growth function g(s, t; theta).

    At training time, s_i and theta are jointly optimized via L-BFGS-B.
    At test time, severity is estimated from baseline features using a
    ridge regression head (D25).

    Args:
        growth_function: Name of the growth function (``"gompertz_reduced"``
            or ``"weighted_sigmoid"``). Default: ``"gompertz_reduced"`` (D22).
        lambda_reg: L2 regularization on population parameters.
        n_restarts: Number of random restarts for optimization.
        max_iter: Maximum L-BFGS-B iterations per restart.
        severity_features: Feature names for the severity regression head.
            Default: ``["log_volume", "sphericity"]``.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        growth_function: str = "gompertz_reduced",
        lambda_reg: float = 0.01,
        n_restarts: int = 5,
        max_iter: int = 5000,
        severity_features: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self._growth_fn: GrowthFunction = GrowthFunctionRegistry.get(growth_function)
        self._lambda_reg = lambda_reg
        self._n_restarts = n_restarts
        self._max_iter = max_iter
        self._severity_features = severity_features or ["log_volume", "sphericity"]
        self._seed = seed

        # Fitted state
        self._qt: QuantileTransform | None = None
        self._severity_result: SeverityFitResult | None = None
        self._regression_head: SeverityRegressionHead | None = None
        self._residual_std: float = 0.1
        self._train_patient_ids: set[str] = set()

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit severity model via joint MLE.

        1. Compute growth values (delta log-volume from baseline)
        2. Fit quantile transform on non-baseline observations
        3. Jointly optimize theta and {s_i} via L-BFGS-B
        4. Compute residual std for uncertainty estimation
        5. Fit severity regression head from baseline features

        Args:
            patients: Training patient trajectories with D=1 observations.

        Returns:
            FitResult with optimization diagnostics.
        """
        rng = np.random.default_rng(self._seed)
        self._train_patient_ids = {p.patient_id for p in patients}

        # 1. Compute growth values relative to baseline
        all_times_nonbaseline: list[float] = []
        all_growths_nonbaseline: list[float] = []
        patient_data: list[tuple[np.ndarray, np.ndarray]] = []

        for p in patients:
            baseline = float(p.observations[0, 0])
            growth = p.observations[:, 0] - baseline
            elapsed = p.times - p.times[0]
            patient_data.append((elapsed, growth))

            # Exclude baseline (elapsed=0) from quantile transform fitting
            for j in range(len(elapsed)):
                if elapsed[j] > 0:
                    all_times_nonbaseline.append(elapsed[j])
                    all_growths_nonbaseline.append(growth[j])

        # 2. Fit quantile transform on non-baseline data only
        self._qt = QuantileTransform()
        if len(all_times_nonbaseline) > 0:
            self._qt.fit(
                np.array(all_times_nonbaseline),
                np.array(all_growths_nonbaseline),
            )
        else:
            # Degenerate case: all patients have only one timepoint
            self._qt.fit(np.array([1.0]), np.array([0.0]))

        # Transform each patient to quantile space
        # Handle baseline: t_q=0, q_g=0 at baseline; transform rest
        patient_quantiles: list[tuple[np.ndarray, np.ndarray]] = []
        for elapsed, growth in patient_data:
            n_tp = len(elapsed)
            t_q = np.zeros(n_tp)
            q_g = np.zeros(n_tp)

            mask = elapsed > 0
            if np.any(mask):
                result = self._qt.transform(elapsed[mask], growth[mask])
                t_q[mask] = result.t_quantile
                q_g[mask] = result.q_growth

            patient_quantiles.append((t_q, q_g))

        # 3. Joint optimization: theta + {s_i}
        n_theta = self._growth_fn.n_params()
        n_patients = len(patients)

        best_loss = float("inf")
        best_params: np.ndarray | None = None
        best_converged = False

        for restart in range(self._n_restarts):
            theta_bounds = self._growth_fn.param_bounds()
            theta_init = np.array([rng.uniform(lo, (lo + hi) / 2) for lo, hi in theta_bounds])
            s_init = rng.uniform(0.1, 0.9, size=n_patients)
            x0 = np.concatenate([theta_init, s_init])

            bounds = theta_bounds + [(0.01, 0.99)] * n_patients

            result = minimize(
                self._objective,
                x0,
                args=(patient_quantiles,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self._max_iter, "ftol": 1e-12},
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x.copy()
                best_converged = result.success

        assert best_params is not None, "No optimization succeeded"

        theta_opt = best_params[:n_theta]
        s_opt = best_params[n_theta:]

        self._severity_result = SeverityFitResult(
            severities={p.patient_id: float(s_opt[i]) for i, p in enumerate(patients)},
            population_params=theta_opt,
            growth_function_name=self._growth_fn.name(),
            final_loss=best_loss,
            converged=best_converged,
        )

        logger.info(
            f"Severity model fit: loss={best_loss:.6f}, converged={best_converged}, "
            f"theta={theta_opt}, severity range=[{s_opt.min():.3f}, {s_opt.max():.3f}], "
            f"severity std={s_opt.std():.3f}"
        )
        logger.info(
            f"Severity percentiles: "
            f"25%={np.percentile(s_opt, 25):.3f}, "
            f"median={np.median(s_opt):.3f}, "
            f"75%={np.percentile(s_opt, 75):.3f}"
        )

        # 4. Compute residual std from training data
        self._residual_std = self._compute_residual_std(patient_quantiles, theta_opt, s_opt)

        # 5. Fit severity regression head
        self._regression_head = SeverityRegressionHead(feature_names=self._severity_features)
        self._regression_head.fit(patients, s_opt)

        return FitResult(
            log_marginal_likelihood=-best_loss,
            hyperparameters={
                **{f"theta_{i}": float(v) for i, v in enumerate(theta_opt)},
                "severity_mean": float(s_opt.mean()),
                "severity_std": float(s_opt.std()),
                "residual_std": self._residual_std,
            },
            n_train_patients=n_patients,
            n_train_observations=sum(p.n_timepoints for p in patients),
        )

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict growth at query times.

        For held-out patients, severity is estimated from the regression head.
        For training patients, the fitted severity is used directly.

        Predictions are made in quantile space, then inverse-transformed
        to log-volume space via the fitted empirical CDF.

        Args:
            patient: Patient trajectory.
            t_pred: Query times, shape ``[n_pred]``.
            n_condition: Number of observations to condition on (unused;
                severity model prediction depends on s_i, not conditioning).

        Returns:
            PredictionResult in log-volume space (not quantile space).
        """
        assert self._severity_result is not None, "Call fit() first"
        assert self._qt is not None, "Call fit() first"

        t_pred = np.asarray(t_pred, dtype=np.float64)
        if t_pred.ndim == 0:
            t_pred = t_pred[np.newaxis]

        # Estimate severity: training patients use fitted, others use regression
        if patient.patient_id in self._severity_result.severities:
            s_i = self._severity_result.severities[patient.patient_id]
        else:
            s_i = self._regression_head.predict(patient) if self._regression_head else 0.5

        # Compute elapsed prediction times from baseline
        baseline = float(patient.observations[0, 0])
        elapsed_pred = t_pred - patient.times[0]

        # Transform prediction times to quantile space
        # For elapsed=0, t_q=0 directly
        t_q = np.zeros_like(elapsed_pred)
        mask = elapsed_pred > 0
        if np.any(mask):
            qt_result = self._qt.transform(elapsed_pred[mask], np.zeros(int(mask.sum())))
            t_q[mask] = qt_result.t_quantile

        # Evaluate growth function in quantile space
        theta = self._severity_result.population_params
        q_pred = self._growth_fn(np.full_like(t_q, s_i), t_q, theta)

        # Inverse quantile transform: q -> delta log-volume
        pred_growth = self._qt.inverse_growth(q_pred)
        pred_log_vol = baseline + pred_growth

        # Uncertainty from training residuals
        sigma = max(self._residual_std, 0.01)
        variance = np.full_like(pred_log_vol, sigma**2)
        lower = pred_log_vol - 1.96 * sigma
        upper = pred_log_vol + 1.96 * sigma

        return PredictionResult(
            mean=pred_log_vol,
            variance=variance,
            lower_95=lower,
            upper_95=upper,
        )

    def name(self) -> str:
        """Human-readable model name."""
        fn_name = self._growth_fn.name() if self._growth_fn else "Unknown"
        return f"SeverityModel({fn_name})"

    def _objective(
        self,
        x: np.ndarray,
        patient_quantiles: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Joint optimization objective: MSE + L2 regularization.

        Args:
            x: Concatenated [theta, s_1, ..., s_N].
            patient_quantiles: Per-patient (t_quantile, q_growth) tuples.

        Returns:
            Loss value.
        """
        n_theta = self._growth_fn.n_params()
        theta = x[:n_theta]
        severities = x[n_theta:]

        total_loss = 0.0
        n_total = 0

        for i, (t_q, q_actual) in enumerate(patient_quantiles):
            s_i = severities[i]
            q_pred = self._growth_fn(np.full_like(t_q, s_i), t_q, theta)
            total_loss += np.sum((q_actual - q_pred) ** 2)
            n_total += len(t_q)

        mse = total_loss / max(n_total, 1)
        reg = self._lambda_reg * np.sum(theta**2)

        return mse + reg

    def _compute_residual_std(
        self,
        patient_quantiles: list[tuple[np.ndarray, np.ndarray]],
        theta: np.ndarray,
        severities: np.ndarray,
    ) -> float:
        """Compute residual standard deviation in log-volume space.

        Evaluates the fitted model on training data and computes the
        std of (actual - predicted) in log-volume space (after inverse
        quantile transform).
        """
        residuals: list[float] = []

        for i, (t_q, q_actual) in enumerate(patient_quantiles):
            s_i = severities[i]
            q_pred = self._growth_fn(np.full_like(t_q, s_i), t_q, theta)

            # Convert both to growth space
            actual_growth = self._qt.inverse_growth(q_actual)
            pred_growth = self._qt.inverse_growth(q_pred)
            residuals.extend((actual_growth - pred_growth).tolist())

        if len(residuals) < 2:
            return 0.1

        return float(np.std(residuals))

    @property
    def fitted_severities(self) -> dict[str, float] | None:
        """Access fitted severity values after training."""
        if self._severity_result is None:
            return None
        return self._severity_result.severities

    @property
    def severity_fit_result(self) -> SeverityFitResult | None:
        """Access the full severity fit result."""
        return self._severity_result
