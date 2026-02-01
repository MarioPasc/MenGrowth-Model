# src/growth/evaluation/enhanced_probes.py
"""Enhanced probe evaluation with MLP probes and target normalization.

This module extends the basic linear probing framework with:
1. MLP (nonlinear) probes to test if information exists in nonlinear form
2. Target normalization for stable training
3. Multi-scale feature support (concatenated features from multiple stages)
4. Better handling of low-variance features

The key insight is that negative R² from linear probes doesn't mean
information is absent - it may be encoded nonlinearly.

References:
    - Alain & Bengio, "Understanding intermediate layers using linear
      classifier probes." ICLR 2017 Workshop.
    - Chen et al., "A simple framework for contrastive learning of
      visual representations." ICML 2020.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class EnhancedProbeResults:
    """Results from enhanced probe evaluation.

    Attributes:
        r2_linear: R² from linear probe.
        r2_mlp: R² from MLP probe.
        r2_per_dim_linear: Per-dimension R² from linear probe.
        r2_per_dim_mlp: Per-dimension R² from MLP probe.
        mse_linear: MSE from linear probe.
        mse_mlp: MSE from MLP probe.
        predictions_linear: Linear probe predictions.
        predictions_mlp: MLP predictions.
        feature_variance: Per-dimension feature variance.
        target_variance: Per-dimension target variance.
    """
    r2_linear: float
    r2_mlp: float
    r2_per_dim_linear: np.ndarray
    r2_per_dim_mlp: np.ndarray
    mse_linear: float
    mse_mlp: float
    predictions_linear: np.ndarray
    predictions_mlp: np.ndarray
    feature_variance: np.ndarray = field(default_factory=lambda: np.array([]))
    target_variance: np.ndarray = field(default_factory=lambda: np.array([]))


class EnhancedLinearProbe:
    """Enhanced linear probe with target normalization.

    Key improvements over basic LinearProbe:
    1. Normalizes both features AND targets
    2. Handles low-variance features gracefully
    3. Returns denormalized predictions for interpretability

    Args:
        alpha: Ridge regularization strength.
        normalize_features: If True, standardize input features.
        normalize_targets: If True, standardize target variables.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        normalize_features: bool = True,
        normalize_targets: bool = True,
    ):
        self.alpha = alpha
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets

        self.model = Ridge(alpha=alpha)
        self.feature_scaler = StandardScaler() if normalize_features else None
        self.target_scaler = StandardScaler() if normalize_targets else None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnhancedLinearProbe":
        """Fit probe with optional target normalization.

        Args:
            X: Features [N, D].
            y: Targets [N, K].

        Returns:
            Self for chaining.
        """
        X_scaled = X
        y_scaled = y

        if self.feature_scaler is not None:
            X_scaled = self.feature_scaler.fit_transform(X)

        if self.target_scaler is not None:
            y_scaled = self.target_scaler.fit_transform(y)

        self.model.fit(X_scaled, y_scaled)
        self.fitted = True

        return self

    def predict(self, X: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """Predict with optional denormalization.

        Args:
            X: Features [N, D].
            denormalize: If True, return predictions in original scale.

        Returns:
            Predictions [N, K].
        """
        if not self.fitted:
            raise RuntimeError("Probe must be fitted first")

        X_scaled = X
        if self.feature_scaler is not None:
            X_scaled = self.feature_scaler.transform(X)

        y_pred = self.model.predict(X_scaled)

        if denormalize and self.target_scaler is not None:
            y_pred = self.target_scaler.inverse_transform(y_pred)

        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate probe on test data.

        Args:
            X: Features [N, D].
            y: Targets [N, K].

        Returns:
            Dict with 'r2', 'r2_per_dim', 'mse'.
        """
        predictions = self.predict(X, denormalize=True)

        r2 = r2_score(y, predictions)
        r2_per_dim = np.array([
            r2_score(y[:, i], predictions[:, i])
            for i in range(y.shape[1])
        ])
        mse = mean_squared_error(y, predictions)

        return {
            'r2': r2,
            'r2_per_dim': r2_per_dim,
            'mse': mse,
            'predictions': predictions,
        }


class MLPProbe:
    """Nonlinear MLP probe for testing nonlinear predictability.

    If linear probes show negative R² but MLP probes show positive R²,
    it indicates the information exists but is nonlinearly encoded.

    Args:
        hidden_sizes: Tuple of hidden layer sizes.
        alpha: L2 regularization strength.
        max_iter: Maximum training iterations.
        normalize_features: If True, standardize inputs.
        normalize_targets: If True, standardize targets.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (256, 128),
        alpha: float = 1e-4,
        max_iter: int = 500,
        normalize_features: bool = True,
        normalize_targets: bool = True,
        random_state: int = 42,
    ):
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        self.random_state = random_state

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        self.feature_scaler = StandardScaler() if normalize_features else None
        self.target_scaler = StandardScaler() if normalize_targets else None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPProbe":
        """Fit MLP probe.

        Args:
            X: Features [N, D].
            y: Targets [N, K].

        Returns:
            Self for chaining.
        """
        X_scaled = X
        y_scaled = y

        if self.feature_scaler is not None:
            X_scaled = self.feature_scaler.fit_transform(X)

        if self.target_scaler is not None:
            y_scaled = self.target_scaler.fit_transform(y)

        self.model.fit(X_scaled, y_scaled)
        self.fitted = True

        logger.debug(f"MLP probe converged in {self.model.n_iter_} iterations")
        return self

    def predict(self, X: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """Predict with MLP.

        Args:
            X: Features [N, D].
            denormalize: If True, return in original scale.

        Returns:
            Predictions [N, K].
        """
        if not self.fitted:
            raise RuntimeError("MLP probe must be fitted first")

        X_scaled = X
        if self.feature_scaler is not None:
            X_scaled = self.feature_scaler.transform(X)

        y_pred = self.model.predict(X_scaled)

        if denormalize and self.target_scaler is not None:
            y_pred = self.target_scaler.inverse_transform(y_pred)

        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate MLP probe.

        Args:
            X: Features [N, D].
            y: Targets [N, K].

        Returns:
            Dict with 'r2', 'r2_per_dim', 'mse'.
        """
        predictions = self.predict(X, denormalize=True)

        r2 = r2_score(y, predictions)
        r2_per_dim = np.array([
            r2_score(y[:, i], predictions[:, i])
            for i in range(y.shape[1])
        ])
        mse = mean_squared_error(y, predictions)

        return {
            'r2': r2,
            'r2_per_dim': r2_per_dim,
            'mse': mse,
            'predictions': predictions,
        }


class EnhancedSemanticProbes:
    """Enhanced semantic probes with linear + MLP evaluation.

    Trains both linear and MLP probes for each semantic feature type,
    providing a comprehensive view of how information is encoded.

    Args:
        input_dim: Feature dimension.
        alpha_linear: Ridge regularization for linear probes.
        alpha_mlp: L2 regularization for MLP probes.
        hidden_sizes: MLP hidden layer sizes.
        normalize_targets: If True, normalize target variables.

    Example:
        >>> probes = EnhancedSemanticProbes(input_dim=768)
        >>> probes.fit(X_train, targets_train)
        >>> results = probes.evaluate(X_test, targets_test)
        >>> print(f"Linear R²: {results['volume']['r2_linear']:.4f}")
        >>> print(f"MLP R²: {results['volume']['r2_mlp']:.4f}")
    """

    FEATURE_DIMS = {
        'volume': 4,
        'location': 3,
        'shape': 6,
    }

    def __init__(
        self,
        input_dim: int = 768,
        alpha_linear: float = 1.0,
        alpha_mlp: float = 1e-4,
        hidden_sizes: Tuple[int, ...] = (256, 128),
        normalize_targets: bool = True,
    ):
        self.input_dim = input_dim
        self.normalize_targets = normalize_targets

        # Create probe pairs for each feature type
        self.linear_probes = {}
        self.mlp_probes = {}

        for name in self.FEATURE_DIMS:
            self.linear_probes[name] = EnhancedLinearProbe(
                alpha=alpha_linear,
                normalize_features=True,
                normalize_targets=normalize_targets,
            )
            self.mlp_probes[name] = MLPProbe(
                hidden_sizes=hidden_sizes,
                alpha=alpha_mlp,
                normalize_features=True,
                normalize_targets=normalize_targets,
            )

    def fit(
        self,
        X: np.ndarray,
        targets: Dict[str, np.ndarray],
    ) -> "EnhancedSemanticProbes":
        """Fit all probes.

        Args:
            X: Features [N, D].
            targets: Dict with 'volume', 'location', 'shape' arrays.

        Returns:
            Self for chaining.
        """
        for name in self.FEATURE_DIMS:
            if name not in targets:
                raise ValueError(f"Missing target: {name}")

            logger.info(f"Fitting probes for {name}...")
            self.linear_probes[name].fit(X, targets[name])
            self.mlp_probes[name].fit(X, targets[name])

        return self

    def evaluate(
        self,
        X: np.ndarray,
        targets: Dict[str, np.ndarray],
    ) -> Dict[str, EnhancedProbeResults]:
        """Evaluate all probes.

        Args:
            X: Features [N, D].
            targets: Dict with 'volume', 'location', 'shape' arrays.

        Returns:
            Dict mapping feature names to EnhancedProbeResults.
        """
        results = {}

        for name in self.FEATURE_DIMS:
            y = targets[name]

            linear_res = self.linear_probes[name].evaluate(X, y)
            mlp_res = self.mlp_probes[name].evaluate(X, y)

            results[name] = EnhancedProbeResults(
                r2_linear=linear_res['r2'],
                r2_mlp=mlp_res['r2'],
                r2_per_dim_linear=linear_res['r2_per_dim'],
                r2_per_dim_mlp=mlp_res['r2_per_dim'],
                mse_linear=linear_res['mse'],
                mse_mlp=mlp_res['mse'],
                predictions_linear=linear_res['predictions'],
                predictions_mlp=mlp_res['predictions'],
                feature_variance=X.var(axis=0),
                target_variance=y.var(axis=0),
            )

        return results

    def get_summary(
        self,
        results: Dict[str, EnhancedProbeResults],
    ) -> Dict[str, float]:
        """Get summary metrics.

        Args:
            results: Results from evaluate().

        Returns:
            Dict with summary metrics.
        """
        summary = {}

        for name, res in results.items():
            summary[f'r2_{name}_linear'] = res.r2_linear
            summary[f'r2_{name}_mlp'] = res.r2_mlp
            summary[f'mse_{name}_linear'] = res.mse_linear
            summary[f'mse_{name}_mlp'] = res.mse_mlp

        # Averages
        summary['r2_mean_linear'] = np.mean([
            res.r2_linear for res in results.values()
        ])
        summary['r2_mean_mlp'] = np.mean([
            res.r2_mlp for res in results.values()
        ])

        # Nonlinearity gap: how much better MLP is than linear
        for name, res in results.items():
            gap = res.r2_mlp - res.r2_linear
            summary[f'nonlinearity_gap_{name}'] = gap

        return summary


def analyze_feature_quality(
    features: np.ndarray,
    targets: Dict[str, np.ndarray],
    alpha_linear: float = 1.0,
    normalize_targets: bool = True,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """Comprehensive feature quality analysis.

    Runs both linear and MLP probes and computes additional diagnostics.

    Args:
        features: Encoder features [N, D].
        targets: Semantic targets dict.
        alpha_linear: Ridge regularization.
        normalize_targets: If True, normalize targets.

    Returns:
        Dict with comprehensive quality metrics.
    """
    # Split data
    n = len(features)
    split_idx = int(0.8 * n)
    X_train, X_test = features[:split_idx], features[split_idx:]
    targets_train = {k: v[:split_idx] for k, v in targets.items()}
    targets_test = {k: v[split_idx:] for k, v in targets.items()}

    # Fit and evaluate probes
    probes = EnhancedSemanticProbes(
        input_dim=features.shape[1],
        alpha_linear=alpha_linear,
        normalize_targets=normalize_targets,
    )
    probes.fit(X_train, targets_train)
    results = probes.evaluate(X_test, targets_test)
    summary = probes.get_summary(results)

    # Feature diagnostics
    variance = features.var(axis=0)
    summary['feature_variance_mean'] = float(variance.mean())
    summary['feature_variance_min'] = float(variance.min())
    summary['feature_variance_max'] = float(variance.max())
    summary['num_low_variance_dims'] = int((variance < 0.01).sum())

    # Per-feature results
    summary['detailed_results'] = {
        name: {
            'r2_linear': res.r2_linear,
            'r2_mlp': res.r2_mlp,
            'r2_per_dim_linear': res.r2_per_dim_linear.tolist(),
            'r2_per_dim_mlp': res.r2_per_dim_mlp.tolist(),
        }
        for name, res in results.items()
    }

    return summary


def compute_multi_scale_features(
    features_dict: Dict[str, np.ndarray],
) -> np.ndarray:
    """Concatenate multi-scale features.

    Args:
        features_dict: Dict with 'layers2', 'layers3', 'layers4' features.

    Returns:
        Concatenated features [N, sum(dims)].
    """
    feature_list = []
    for key in ['layers2', 'layers3', 'layers4']:
        if key in features_dict:
            feature_list.append(features_dict[key])

    if not feature_list:
        raise ValueError("No features found in dict")

    return np.concatenate(feature_list, axis=1)
