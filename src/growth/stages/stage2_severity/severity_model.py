# src/growth/stages/stage2_severity/severity_model.py
"""Latent severity NLME growth model (Stage 2).

Implements a nonlinear mixed-effects model where a single latent variable
s_i ∈ [0, 1] governs each patient's growth trajectory:

    q_ij = g(s_i, t_ij; θ) + ε_ij

where q_ij is the growth quantile, s_i is the latent severity, t_ij is the
normalized time, and g is a monotonic growth function satisfying g(s, 0) = 0.

The model is formally equivalent to an IRT 2PL model (D22, D23).
Severity is estimated jointly with population parameters θ via L-BFGS-B
during training, and predicted from baseline features at test time (D25).

References:
    - Vaghi et al. (2020) PLOS Computational Biology
    - Proust-Lima et al. (2014, 2023) lcmm R package
    - Runje & Shankaranarayana (2023) ICML — CMNN

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

logger = logging.getLogger(__name__)


@dataclass
class SeverityFitResult:
    """Extended fit result with severity-specific diagnostics.

    Args:
        severities: Estimated severity per patient, dict mapping patient_id → s_i.
        population_params: Fitted population parameters θ.
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
    """Latent severity NLME model for growth prediction.

    Each patient is assigned a latent severity s_i ∈ [0, 1] that, together
    with the shared population parameters θ, determines the growth trajectory
    via a monotonic growth function g(s, t; θ).

    At training time, s_i and θ are jointly optimized via L-BFGS-B.
    At test time, severity is estimated from baseline features using a
    simple linear regression head.

    Args:
        growth_function: Name of the growth function (``"gompertz_reduced"``
            or ``"weighted_sigmoid"``). Default: ``"gompertz_reduced"`` (D22).
        lambda_reg: L2 regularization on population parameters.
        n_restarts: Number of random restarts for optimization.
        max_iter: Maximum L-BFGS-B iterations per restart.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        growth_function: str = "gompertz_reduced",
        lambda_reg: float = 0.01,
        n_restarts: int = 5,
        max_iter: int = 5000,
        seed: int = 42,
    ) -> None:
        self._growth_fn: GrowthFunction = GrowthFunctionRegistry.get(growth_function)
        self._lambda_reg = lambda_reg
        self._n_restarts = n_restarts
        self._max_iter = max_iter
        self._seed = seed

        # Fitted state
        self._qt: QuantileTransform | None = None
        self._severity_result: SeverityFitResult | None = None
        self._severity_regression_weights: np.ndarray | None = None
        self._severity_regression_bias: float = 0.0
        self._train_patient_ids: list[str] = []

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit severity model via joint MLE.

        1. Compute growth values (Δ log-volume from baseline)
        2. Fit quantile transform on pooled training data
        3. Jointly optimize θ and {s_i} via L-BFGS-B
        4. Fit severity regression head from baseline features

        Args:
            patients: Training patient trajectories with D=1 observations.

        Returns:
            FitResult with optimization diagnostics.
        """
        rng = np.random.default_rng(self._seed)
        self._train_patient_ids = [p.patient_id for p in patients]

        # 1. Compute growth values relative to baseline
        all_times: list[float] = []
        all_growths: list[float] = []
        patient_data: list[tuple[np.ndarray, np.ndarray]] = []

        for p in patients:
            baseline = float(p.observations[0, 0])
            growth = p.observations[:, 0] - baseline  # ΔV in log-space
            elapsed = p.times - p.times[0]
            all_times.extend(elapsed.tolist())
            all_growths.extend(growth.tolist())
            patient_data.append((elapsed, growth))

        # 2. Fit quantile transform
        self._qt = QuantileTransform()
        self._qt.fit(np.array(all_times), np.array(all_growths))

        # Transform each patient to quantile space
        patient_quantiles: list[tuple[np.ndarray, np.ndarray]] = []
        for elapsed, growth in patient_data:
            result = self._qt.transform(elapsed, growth)
            patient_quantiles.append((result.t_quantile, result.q_growth))

        # 3. Joint optimization: θ + {s_i}
        n_theta = self._growth_fn.n_params()
        n_patients = len(patients)

        best_loss = float("inf")
        best_params = None

        for restart in range(self._n_restarts):
            # Initialize: random θ within bounds, uniform s_i
            theta_bounds = self._growth_fn.param_bounds()
            theta_init = np.array([rng.uniform(lo, (lo + hi) / 2) for lo, hi in theta_bounds])
            s_init = rng.uniform(0.1, 0.9, size=n_patients)
            x0 = np.concatenate([theta_init, s_init])

            # Bounds: θ bounds + [0, 1] for each s_i
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
                best_params = result.x
                converged = result.success

        # Extract best results
        theta_opt = best_params[:n_theta]
        s_opt = best_params[n_theta:]

        self._severity_result = SeverityFitResult(
            severities={p.patient_id: float(s_opt[i]) for i, p in enumerate(patients)},
            population_params=theta_opt,
            growth_function_name=self._growth_fn.name(),
            final_loss=best_loss,
            converged=converged,
        )

        logger.info(
            f"Severity model fit: loss={best_loss:.6f}, converged={converged}, "
            f"θ={theta_opt}, severity range=[{s_opt.min():.3f}, {s_opt.max():.3f}]"
        )

        # 4. Fit severity regression head from baseline features
        self._fit_severity_regression(patients, s_opt)

        return FitResult(
            log_marginal_likelihood=-best_loss,
            hyperparameters={f"theta_{i}": float(v) for i, v in enumerate(theta_opt)},
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

        Args:
            patient: Patient trajectory.
            t_pred: Query times, shape ``[n_pred]``.
            n_condition: Number of observations to condition on (unused for
                severity model; prediction depends on s_i, not on conditioning).

        Returns:
            PredictionResult in log-volume space (not quantile space).
        """
        assert self._severity_result is not None, "Call fit() first"
        assert self._qt is not None, "Call fit() first"

        # Estimate severity
        if patient.patient_id in self._severity_result.severities:
            s_i = self._severity_result.severities[patient.patient_id]
        else:
            s_i = self._estimate_severity_from_features(patient)

        # Predict in quantile space
        baseline = float(patient.observations[0, 0])
        elapsed_pred = t_pred - patient.times[0]

        # Transform prediction times to quantile space
        qt_result = self._qt.transform(elapsed_pred, np.zeros_like(elapsed_pred))
        t_q = qt_result.t_quantile

        # Evaluate growth function
        theta = self._severity_result.population_params
        q_pred = self._growth_fn(np.full_like(t_q, s_i), t_q, theta)

        # Map back to log-volume space (approximate inverse via linear scaling)
        # q ≈ 0 means no growth, q ≈ 1 means max growth in training set
        all_growths = np.array(list(self._qt._all_growths))
        growth_scale = np.percentile(all_growths, 95) if len(all_growths) > 0 else 1.0
        pred_growth = q_pred * max(growth_scale, 0.01)
        pred_log_vol = baseline + pred_growth

        # Uncertainty: simple estimate from residual variance
        sigma = max(np.std(all_growths) * 0.3, 0.01)
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
        fn_name = self._growth_fn.name() if self._growth_fn else "Unknown"
        return f"SeverityModel({fn_name})"

    def _objective(
        self,
        x: np.ndarray,
        patient_quantiles: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Joint optimization objective: MSE + L2 regularization.

        Args:
            x: Concatenated [θ, s_1, ..., s_N].
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

        # Normalize by number of observations
        mse = total_loss / max(n_total, 1)

        # L2 regularization on population params
        reg = self._lambda_reg * np.sum(theta**2)

        return mse + reg

    def _fit_severity_regression(
        self,
        patients: list[PatientTrajectory],
        severities: np.ndarray,
    ) -> None:
        """Fit a linear regression from baseline features to severity.

        Features: [log_volume_baseline, + any available covariates]

        Args:
            patients: Training patients.
            severities: Fitted severity values, shape ``[N]``.
        """
        # Build feature matrix: [log_vol_baseline, covariates...]
        features: list[np.ndarray] = []
        valid_indices: list[int] = []

        for i, p in enumerate(patients):
            feat = [float(p.observations[0, 0])]  # log_vol at baseline
            if p.covariates:
                for key in sorted(p.covariates.keys()):
                    feat.append(p.covariates[key])
            features.append(np.array(feat))
            valid_indices.append(i)

        if not features:
            self._severity_regression_weights = None
            return

        # Ensure all feature vectors have the same length
        max_len = max(len(f) for f in features)
        X = np.zeros((len(features), max_len))
        for i, f in enumerate(features):
            X[i, : len(f)] = f

        y = severities[valid_indices]

        # Simple least-squares regression with regularization
        # ŝ = X @ w + b
        X_aug = np.column_stack([X, np.ones(len(X))])
        reg_matrix = np.eye(X_aug.shape[1]) * 0.01
        reg_matrix[-1, -1] = 0  # Don't regularize bias

        try:
            w = np.linalg.solve(X_aug.T @ X_aug + reg_matrix, X_aug.T @ y)
            self._severity_regression_weights = w[:-1]
            self._severity_regression_bias = float(w[-1])
            logger.info(f"Severity regression fitted: {len(w) - 1} features")
        except np.linalg.LinAlgError:
            logger.warning("Severity regression failed; using mean severity as fallback")
            self._severity_regression_weights = None
            self._severity_regression_bias = float(np.mean(severities))

    def _estimate_severity_from_features(self, patient: PatientTrajectory) -> float:
        """Estimate severity for a new patient from baseline features (D25).

        Args:
            patient: Patient with at least one observation.

        Returns:
            Estimated severity in [0, 1].
        """
        if self._severity_regression_weights is None:
            return self._severity_regression_bias

        feat = [float(patient.observations[0, 0])]
        if patient.covariates:
            for key in sorted(patient.covariates.keys()):
                feat.append(patient.covariates[key])

        x = np.array(feat)
        # Pad or truncate to match regression weights
        n_w = len(self._severity_regression_weights)
        if len(x) < n_w:
            x = np.pad(x, (0, n_w - len(x)))
        elif len(x) > n_w:
            x = x[:n_w]

        s_hat = float(np.dot(self._severity_regression_weights, x) + self._severity_regression_bias)
        return np.clip(s_hat, 0.01, 0.99)

    @property
    def fitted_severities(self) -> dict[str, float] | None:
        """Access fitted severity values after training."""
        if self._severity_result is None:
            return None
        return self._severity_result.severities
