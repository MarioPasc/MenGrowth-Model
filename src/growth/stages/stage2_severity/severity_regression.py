# src/growth/stages/stage2_severity/severity_regression.py
"""Severity regression head for test-time severity estimation (D25).

At test time, a held-out patient's severity must be predicted from baseline
features — not from the jointly-optimized severity (that would be data
leakage). This module implements a simple ridge regression from named
baseline features to severity in [0, 1].

Features available with 100% coverage on MenGrowth:
- ``"log_volume"``: log(V_baseline + 1) from observations[0, 0]
- ``"sphericity"``: from semantic/shape[:, 0] loaded as covariate

Features with partial coverage (use only when available):
- ``"age"``: from metadata/age
- ``"sex"``: from metadata/sex (encoded as 0/1)
"""

import logging

import numpy as np

from growth.shared.growth_models import PatientTrajectory

logger = logging.getLogger(__name__)


class SeverityRegressionHead:
    """Linear regression from baseline features to severity.

    Uses named features in a fixed order to avoid fragile pad/truncate
    logic. Applies clipping to ensure output in [0.01, 0.99].

    Args:
        feature_names: Ordered list of feature names to use.
        lambda_reg: Ridge regularization strength.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        lambda_reg: float = 0.01,
    ) -> None:
        self.feature_names = feature_names or ["log_volume"]
        self.lambda_reg = lambda_reg
        self._weights: np.ndarray | None = None
        self._bias: float = 0.5
        self._feature_means: np.ndarray | None = None
        self._fitted = False

    def extract_features(self, patient: PatientTrajectory) -> dict[str, float]:
        """Extract named features from a patient trajectory.

        ``log_volume`` is always extracted from ``observations[0, 0]``.
        Other features are looked up in ``patient.covariates``.

        Args:
            patient: Patient with at least one observation.

        Returns:
            Dict mapping feature name to value. Missing features have
            value ``None`` (caller decides how to handle).
        """
        features: dict[str, float] = {}

        for name in self.feature_names:
            if name == "log_volume":
                features[name] = float(patient.observations[0, 0])
            elif patient.covariates is not None and name in patient.covariates:
                features[name] = float(patient.covariates[name])
            else:
                features[name] = float("nan")

        return features

    def fit(
        self,
        patients: list[PatientTrajectory],
        severities: np.ndarray,
    ) -> None:
        """Fit ridge regression from baseline features to severity.

        Patients with any missing (NaN) feature are excluded from fitting.

        Args:
            patients: Training patients.
            severities: Fitted severity values, shape ``[N]``.
        """
        X_list: list[np.ndarray] = []
        y_list: list[float] = []

        for i, patient in enumerate(patients):
            feats = self.extract_features(patient)
            vec = np.array([feats[name] for name in self.feature_names])
            if np.any(np.isnan(vec)):
                continue
            X_list.append(vec)
            y_list.append(severities[i])

        if len(X_list) < 2:
            logger.warning(
                f"Only {len(X_list)} patients with complete features for "
                f"severity regression. Using mean severity as fallback."
            )
            self._weights = None
            self._bias = float(np.mean(severities))
            self._feature_means = np.zeros(len(self.feature_names))
            self._fitted = True
            return

        X = np.array(X_list)
        y = np.array(y_list)

        # Store per-feature means for NaN imputation at predict time
        self._feature_means = np.mean(X, axis=0)

        # Ridge regression: w = (X^T X + lambda I)^{-1} X^T y
        # Augmented with bias column
        X_aug = np.column_stack([X, np.ones(len(X))])
        n_features = X_aug.shape[1]
        reg_matrix = np.eye(n_features) * self.lambda_reg
        reg_matrix[-1, -1] = 0.0  # Don't regularize bias

        try:
            w = np.linalg.solve(X_aug.T @ X_aug + reg_matrix, X_aug.T @ y)
            self._weights = w[:-1]
            self._bias = float(w[-1])
        except np.linalg.LinAlgError:
            logger.warning("Ridge regression failed; using mean severity")
            self._weights = None
            self._bias = float(np.mean(severities))

        self._fitted = True
        logger.info(
            f"SeverityRegressionHead fitted: {len(self.feature_names)} features, "
            f"{len(X_list)} patients, weights={self._weights}, bias={self._bias:.4f}"
        )

    def predict(self, patient: PatientTrajectory) -> float:
        """Predict severity for a patient from baseline features.

        Args:
            patient: Patient with at least one observation.

        Returns:
            Estimated severity in [0.01, 0.99].
        """
        if not self._fitted:
            return 0.5

        if self._weights is None:
            return float(np.clip(self._bias, 0.01, 0.99))

        feats = self.extract_features(patient)
        vec = np.array([feats[name] for name in self.feature_names])

        # Impute NaN with training feature means (not 0, which would be far
        # from typical values — e.g. log_volume=0 means zero-volume tumor)
        nan_mask = np.isnan(vec)
        if np.any(nan_mask) and self._feature_means is not None:
            nan_names = [n for n, m in zip(self.feature_names, nan_mask) if m]
            logger.debug(f"Imputing NaN features with training means: {nan_names}")
            vec = np.where(nan_mask, self._feature_means, vec)

        s_hat = float(np.dot(self._weights, vec) + self._bias)
        return float(np.clip(s_hat, 0.01, 0.99))

    def predict_from_features(self, features: dict[str, float]) -> float:
        """Predict severity from a pre-extracted feature dict.

        Args:
            features: Dict mapping feature name to value.

        Returns:
            Estimated severity in [0.01, 0.99].
        """
        if not self._fitted or self._weights is None:
            return float(np.clip(self._bias, 0.01, 0.99))

        defaults = self._feature_means if self._feature_means is not None else np.zeros(len(self.feature_names))
        vec = np.array([features.get(name, defaults[i]) for i, name in enumerate(self.feature_names)])
        nan_mask = np.isnan(vec)
        if np.any(nan_mask):
            vec = np.where(nan_mask, defaults, vec)
        s_hat = float(np.dot(self._weights, vec) + self._bias)
        return float(np.clip(s_hat, 0.01, 0.99))
