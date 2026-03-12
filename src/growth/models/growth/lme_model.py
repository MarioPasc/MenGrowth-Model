# src/growth/models/growth/lme_model.py
"""Linear Mixed-Effects (LME) baseline growth model (Model A).

Per-dimension LME on observations ∈ ℝᴰ:
    z_d(t) = (β₀_d + b₀ᵢ_d) + (β₁_d + b₁ᵢ_d) · t + ε

Fitted via REML (statsmodels.MixedLM). Patient-specific predictions via BLUP
with automatic shrinkage for patients with few observations (n_i = 2).

Works for:
  - D=1: scalar volume baseline (A0)
  - D=24: z_vol partition (Module 5, Model A)

References:
    Laird & Ware, "Random-Effects Models for Longitudinal Data", Biometrics 1982.
    Robinson, "That BLUP Is a Good Thing", Statistical Science 1991.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .base import FitResult, GrowthModel, PatientTrajectory, PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class LMEDimensionFit:
    """Fitted parameters for one dimension of the LME model."""

    beta_0: float  # Fixed intercept
    beta_1: float  # Fixed slope
    sigma_sq: float  # Residual variance
    omega: np.ndarray  # Random effects covariance [2, 2]
    converged: bool


class LMEGrowthModel(GrowthModel):
    """Per-dimension Linear Mixed-Effects model.

    Fits D independent LME models (one per observation dimension) using REML.
    Prediction uses BLUP (Best Linear Unbiased Predictor) for patient-specific
    random effects, with shrinkage toward the population mean.

    Args:
        method: Estimation method. ``"reml"`` (default) or ``"ml"``.
    """

    def __init__(self, method: str = "reml") -> None:
        self.method = method
        self._dim_fits: list[LMEDimensionFit] = []
        self._obs_dim: int = 0
        self._fitted: bool = False

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit D independent LME models via REML.

        Args:
            patients: Training patient trajectories.

        Returns:
            FitResult with fixed-effect parameters and diagnostics.
        """
        if len(patients) == 0:
            raise ValueError("Cannot fit with zero patients")

        self._obs_dim = patients[0].obs_dim
        n_total = sum(p.n_timepoints for p in patients)

        logger.info(
            f"LME fitting: {len(patients)} patients, {n_total} observations, D={self._obs_dim}"
        )

        # Build long-format dataframe
        rows: list[dict] = []
        for p in patients:
            for j in range(p.n_timepoints):
                row = {
                    "patient_id": p.patient_id,
                    "time": p.times[j],
                }
                for d in range(self._obs_dim):
                    row[f"y_{d}"] = p.observations[j, d]
                rows.append(row)
        df = pd.DataFrame(rows)

        self._dim_fits = []
        hypers: dict[str, float] = {}
        total_lml = 0.0
        n_converged = 0

        for d in range(self._obs_dim):
            dim_fit = self._fit_dimension(df, d)
            self._dim_fits.append(dim_fit)

            hypers[f"beta_0_d{d}"] = dim_fit.beta_0
            hypers[f"beta_1_d{d}"] = dim_fit.beta_1
            hypers[f"sigma_sq_d{d}"] = dim_fit.sigma_sq

            if dim_fit.converged:
                n_converged += 1

        self._fitted = True

        logger.info(
            f"LME fit: {n_converged}/{self._obs_dim} dims converged, "
            f"{len(patients)} patients, {n_total} obs"
        )

        return FitResult(
            log_marginal_likelihood=total_lml,
            hyperparameters=hypers,
            condition_number=0.0,
            n_train_patients=len(patients),
            n_train_observations=n_total,
        )

    def _fit_dimension(self, df: pd.DataFrame, d: int) -> LMEDimensionFit:
        """Fit a single-dimension LME via statsmodels.

        Model: y_d ~ time + (1 + time | patient_id)

        Args:
            df: Long-format dataframe with columns patient_id, time, y_d.
            d: Dimension index.

        Returns:
            LMEDimensionFit with fixed effects, variance, and convergence flag.
        """
        y_col = f"y_{d}"

        try:
            model = smf.mixedlm(
                f"{y_col} ~ time",
                data=df,
                groups=df["patient_id"],
                re_formula="~time",
            )
            result = model.fit(reml=(self.method == "reml"), method="lbfgs")

            beta_0 = float(result.fe_params["Intercept"])
            beta_1 = float(result.fe_params["time"])
            sigma_sq = float(result.scale)

            # Random effects covariance
            re_cov = np.array(result.cov_re)
            if re_cov.ndim == 0:
                re_cov = np.array([[float(re_cov)]])

            return LMEDimensionFit(
                beta_0=beta_0,
                beta_1=beta_1,
                sigma_sq=sigma_sq,
                omega=re_cov,
                converged=result.converged,
            )

        except Exception as e:
            logger.warning(f"LME dim {d} failed ({e}), falling back to random-intercept")
            return self._fit_dimension_intercept_only(df, d)

    def _fit_dimension_intercept_only(self, df: pd.DataFrame, d: int) -> LMEDimensionFit:
        """Fallback: random-intercept-only model (no random slope).

        Args:
            df: Long-format dataframe.
            d: Dimension index.

        Returns:
            LMEDimensionFit with 1x1 random effects covariance.
        """
        y_col = f"y_{d}"
        try:
            model = smf.mixedlm(
                f"{y_col} ~ time",
                data=df,
                groups=df["patient_id"],
            )
            result = model.fit(reml=(self.method == "reml"), method="lbfgs")

            beta_0 = float(result.fe_params["Intercept"])
            beta_1 = float(result.fe_params["time"])
            sigma_sq = float(result.scale)

            re_var = (
                float(result.cov_re.iloc[0, 0])
                if hasattr(result.cov_re, "iloc")
                else float(np.array(result.cov_re).flat[0])
            )
            omega = np.array([[re_var, 0.0], [0.0, 0.0]])

            return LMEDimensionFit(
                beta_0=beta_0,
                beta_1=beta_1,
                sigma_sq=sigma_sq,
                omega=omega,
                converged=result.converged,
            )
        except Exception as e:
            logger.warning(f"LME dim {d} intercept-only also failed ({e}), using OLS")
            return self._fit_dimension_ols(df, d)

    def _fit_dimension_ols(self, df: pd.DataFrame, d: int) -> LMEDimensionFit:
        """Last-resort fallback: simple OLS (no random effects)."""
        y_col = f"y_{d}"
        y = df[y_col].values
        t = df["time"].values

        # OLS: y = beta_0 + beta_1 * t
        A = np.column_stack([np.ones_like(t), t])
        beta, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
        sigma_sq = float(np.var(y - A @ beta))

        return LMEDimensionFit(
            beta_0=float(beta[0]),
            beta_1=float(beta[1]),
            sigma_sq=max(sigma_sq, 1e-10),
            omega=np.zeros((2, 2)),
            converged=True,
        )

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict via BLUP: population mean + shrunk random effects.

        Args:
            patient: Patient trajectory for conditioning.
            t_pred: Query times, shape ``[n_pred]``.
            n_condition: Condition on first ``n_condition`` observations.

        Returns:
            PredictionResult with mean, variance, and 95% CI.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        t_pred = np.asarray(t_pred, dtype=np.float64)
        if t_pred.ndim == 0:
            t_pred = t_pred[np.newaxis]
        n_pred = len(t_pred)
        D = self._obs_dim

        if n_condition is not None:
            t_cond = patient.times[:n_condition]
            y_cond = patient.observations[:n_condition]
        else:
            t_cond = patient.times
            y_cond = patient.observations

        mean = np.zeros((n_pred, D))
        variance = np.zeros((n_pred, D))

        for d in range(D):
            dim_fit = self._dim_fits[d]
            mu_d, var_d = self._predict_dimension(dim_fit, t_cond, y_cond[:, d], t_pred)
            mean[:, d] = mu_d
            variance[:, d] = var_d

        std = np.sqrt(np.maximum(variance, 0.0))
        return PredictionResult(
            mean=mean,
            variance=variance,
            lower_95=mean - 1.96 * std,
            upper_95=mean + 1.96 * std,
        )

    def _predict_dimension(
        self,
        dim_fit: LMEDimensionFit,
        t_cond: np.ndarray,
        y_cond: np.ndarray,
        t_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """BLUP prediction for a single dimension.

        Computes:
            b_hat = Omega @ Z^T @ V_inv @ (y - X @ beta)
            y_pred = X_pred @ beta + Z_pred @ b_hat

        where V = Z @ Omega @ Z^T + sigma^2 * I.

        Args:
            dim_fit: Fitted parameters for this dimension.
            t_cond: Conditioning times [n_cond].
            y_cond: Conditioning observations [n_cond].
            t_pred: Prediction times [n_pred].

        Returns:
            (mean [n_pred], variance [n_pred]).
        """
        n_cond = len(t_cond)
        beta = np.array([dim_fit.beta_0, dim_fit.beta_1])

        # Design matrices
        X_cond = np.column_stack([np.ones(n_cond), t_cond])  # [n_cond, 2]
        X_pred = np.column_stack([np.ones(len(t_pred)), t_pred])  # [n_pred, 2]
        Z_cond = X_cond  # Random effects on intercept + slope: same design
        Z_pred = X_pred

        # Population mean
        pop_mean_cond = X_cond @ beta
        pop_mean_pred = X_pred @ beta

        # Residuals from population mean
        residuals = y_cond - pop_mean_cond

        # Covariance of observations: V = Z @ Omega @ Z^T + sigma^2 * I
        Omega = dim_fit.omega
        V = Z_cond @ Omega @ Z_cond.T + dim_fit.sigma_sq * np.eye(n_cond)

        # BLUP random effects: b_hat = Omega @ Z^T @ V^{-1} @ residuals
        try:
            V_inv = np.linalg.solve(V, np.eye(n_cond))
            b_hat = Omega @ Z_cond.T @ V_inv @ residuals
        except np.linalg.LinAlgError:
            # Singular V: fall back to population mean
            logger.debug("Singular V matrix, using population mean only")
            b_hat = np.zeros(2)

        # Prediction: population mean + random effects contribution
        mean = pop_mean_pred + Z_pred @ b_hat

        # Prediction variance:
        # Var(y_pred) = sigma^2 + Z_pred @ (Omega - Omega @ Z^T @ V^{-1} @ Z @ Omega) @ Z_pred^T
        try:
            shrunk_cov = Omega - Omega @ Z_cond.T @ V_inv @ Z_cond @ Omega
            pred_var = np.array(
                [dim_fit.sigma_sq + Z_pred[i] @ shrunk_cov @ Z_pred[i] for i in range(len(t_pred))]
            )
        except Exception:
            pred_var = np.full(len(t_pred), dim_fit.sigma_sq)

        return mean, np.maximum(pred_var, 1e-10)

    def get_fixed_effects(self) -> list[tuple[float, float]]:
        """Return (β₀_d, β₁_d) for each dimension. Used by H-GP as mean function (D18).

        Returns:
            List of (intercept, slope) tuples, length D.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return [(f.beta_0, f.beta_1) for f in self._dim_fits]

    def name(self) -> str:
        """Human-readable model name."""
        return f"LME(D={self._obs_dim}, method={self.method})"
