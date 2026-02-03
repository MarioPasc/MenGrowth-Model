# src/growth/evaluation/latent_quality.py
"""Latent space quality metrics for evaluating encoder features.

This module provides tools to evaluate the quality of learned representations
through linear probing - a standard technique for assessing feature quality.

The primary metric is R² from linear probes trained to predict semantic
features (volume, location, shape) from encoder features. High R² indicates
that the encoder has learned features that are linearly predictive of
clinically meaningful attributes.

Key Components:
    - LinearProbe: Ridge regression probe for feature evaluation
    - R² metrics: Per-dimension and aggregate R² scores
    - Correlation analysis: Cross-partition correlation matrices
    - Distance correlation: Nonlinear independence measure
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ProbeResults:
    """Results from linear probe evaluation.

    Attributes:
        r2: Overall R² score.
        r2_per_dim: R² score per target dimension.
        mse: Mean squared error.
        predictions: Model predictions on test set.
        coefficients: Learned linear weights.
    """

    r2: float
    r2_per_dim: np.ndarray
    mse: float
    predictions: np.ndarray
    coefficients: np.ndarray


class LinearProbe:
    """Linear regression probe for evaluating feature quality.

    Trains a ridge regression model to predict semantic features from
    encoder features. The R² score indicates how well the encoder has
    learned the target semantics.

    Args:
        input_dim: Dimension of encoder features.
        output_dim: Dimension of target features.
        alpha: Ridge regularization strength.
        normalize: Whether to standardize inputs before fitting.

    Example:
        >>> probe = LinearProbe(input_dim=768, output_dim=4)
        >>> probe.fit(X_train, y_train)
        >>> results = probe.evaluate(X_test, y_test)
        >>> print(f"R² = {results.r2:.4f}")
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        alpha: float = 1.0,
        normalize: bool = True,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.normalize = normalize

        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler() if normalize else None
        self.fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "LinearProbe":
        """Fit linear probe on training data.

        Args:
            X: Encoder features [N, input_dim].
            y: Target semantic features [N, output_dim].

        Returns:
            Self for method chaining.
        """
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {X.shape[1]}"
            )
        if y.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected output_dim={self.output_dim}, got {y.shape[1]}"
            )

        # Normalize inputs
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        # Fit ridge regression
        self.model.fit(X, y)
        self.fitted = True

        logger.debug(f"LinearProbe fitted on {X.shape[0]} samples")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict semantic features from encoder features.

        Args:
            X: Encoder features [N, input_dim].

        Returns:
            Predicted semantic features [N, output_dim].
        """
        if not self.fitted:
            raise RuntimeError("LinearProbe must be fitted before prediction")

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ProbeResults:
        """Evaluate linear probe on test data.

        Args:
            X: Encoder features [N, input_dim].
            y: Target semantic features [N, output_dim].

        Returns:
            ProbeResults with R², MSE, and predictions.
        """
        predictions = self.predict(X)

        # Overall R²
        r2 = r2_score(y, predictions)

        # Per-dimension R²
        r2_per_dim = np.array([
            r2_score(y[:, i], predictions[:, i])
            for i in range(y.shape[1])
        ])

        # MSE
        mse = mean_squared_error(y, predictions)

        return ProbeResults(
            r2=r2,
            r2_per_dim=r2_per_dim,
            mse=mse,
            predictions=predictions,
            coefficients=self.model.coef_,
        )

    def get_coefficients(self) -> np.ndarray:
        """Get learned linear weights.

        Returns:
            Coefficient matrix [output_dim, input_dim].
        """
        if not self.fitted:
            raise RuntimeError("LinearProbe must be fitted first")
        return self.model.coef_


class SemanticProbes:
    """Collection of linear probes for all semantic features.

    Manages separate probes for volume, location, and shape features,
    providing a unified interface for training and evaluation.

    Args:
        input_dim: Dimension of encoder features.
        alpha: Ridge regularization strength for all probes.

    Example:
        >>> probes = SemanticProbes(input_dim=768)
        >>> probes.fit(X_train, {
        ...     "volume": y_vol_train,
        ...     "location": y_loc_train,
        ...     "shape": y_shape_train,
        ... })
        >>> results = probes.evaluate(X_test, {
        ...     "volume": y_vol_test,
        ...     "location": y_loc_test,
        ...     "shape": y_shape_test,
        ... })
    """

    # Standard dimensions for BraTS semantic features
    FEATURE_DIMS = {
        "volume": 4,   # total, NCR, ED, ET
        "location": 3,  # centroid x, y, z
        "shape": 3,    # sphericity, surface_area_log, solidity
    }

    def __init__(
        self,
        input_dim: int = 768,
        alpha: float = 1.0,
    ):
        self.input_dim = input_dim
        self.alpha = alpha

        self.probes = {
            name: LinearProbe(input_dim, dim, alpha=alpha)
            for name, dim in self.FEATURE_DIMS.items()
        }

    def fit(
        self,
        X: np.ndarray,
        targets: Dict[str, np.ndarray],
    ) -> "SemanticProbes":
        """Fit all probes on training data.

        Args:
            X: Encoder features [N, input_dim].
            targets: Dict mapping feature names to target arrays.

        Returns:
            Self for method chaining.
        """
        for name, probe in self.probes.items():
            if name not in targets:
                raise ValueError(f"Missing target: {name}")
            probe.fit(X, targets[name])

        logger.info(f"Fitted {len(self.probes)} semantic probes on {X.shape[0]} samples")
        return self

    def evaluate(
        self,
        X: np.ndarray,
        targets: Dict[str, np.ndarray],
    ) -> Dict[str, ProbeResults]:
        """Evaluate all probes on test data.

        Args:
            X: Encoder features [N, input_dim].
            targets: Dict mapping feature names to target arrays.

        Returns:
            Dict mapping feature names to ProbeResults.
        """
        results = {}
        for name, probe in self.probes.items():
            if name not in targets:
                raise ValueError(f"Missing target: {name}")
            results[name] = probe.evaluate(X, targets[name])

        return results

    def get_summary(self, results: Dict[str, ProbeResults]) -> Dict[str, float]:
        """Get summary R² scores from results.

        Args:
            results: Results from evaluate().

        Returns:
            Dict with R² for each feature type and overall.
        """
        summary = {
            f"r2_{name}": res.r2
            for name, res in results.items()
        }
        summary["r2_mean"] = np.mean([res.r2 for res in results.values()])
        return summary


def compute_r2_scores(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    test_split: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute R² scores using train/test split.

    Convenience function for quick R² evaluation.

    Args:
        X: Encoder features [N, D].
        y: Target features [N, K].
        alpha: Ridge regularization.
        test_split: Fraction of data for testing.
        random_state: Random seed for split.

    Returns:
        Dict with 'r2', 'r2_per_dim', 'mse'.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )

    probe = LinearProbe(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
        alpha=alpha,
    )
    probe.fit(X_train, y_train)
    results = probe.evaluate(X_test, y_test)

    return {
        "r2": results.r2,
        "r2_per_dim": results.r2_per_dim.tolist(),
        "mse": results.mse,
    }


def compute_cross_correlation(
    features: np.ndarray,
    partition_indices: Optional[Dict[str, Tuple[int, int]]] = None,
) -> np.ndarray:
    """Compute cross-correlation matrix between partitions.

    Args:
        features: Feature array [N, D].
        partition_indices: Dict mapping partition names to (start, end) indices.
            If None, computes full correlation matrix.

    Returns:
        Correlation matrix.
    """
    return np.corrcoef(features.T)


def compute_partition_correlation(
    features: np.ndarray,
    partition_indices: Dict[str, Tuple[int, int]],
) -> Dict[str, float]:
    """Compute mean absolute correlation between partitions.

    Args:
        features: Feature array [N, D].
        partition_indices: Dict mapping partition names to (start, end) indices.

    Returns:
        Dict with correlation for each partition pair.
    """
    corr_matrix = np.corrcoef(features.T)
    results = {}

    partition_names = list(partition_indices.keys())
    for i, name_i in enumerate(partition_names):
        for j, name_j in enumerate(partition_names[i + 1:], i + 1):
            start_i, end_i = partition_indices[name_i]
            start_j, end_j = partition_indices[name_j]

            # Extract cross-partition block
            block = corr_matrix[start_i:end_i, start_j:end_j]
            mean_abs_corr = np.abs(block).mean()

            results[f"{name_i}_{name_j}"] = mean_abs_corr

    return results


def distance_correlation(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """Compute distance correlation between two arrays.

    Distance correlation measures both linear and nonlinear dependence.
    dCor = 0 implies independence, dCor = 1 implies strong dependence.

    Args:
        X: First array [N, D1].
        Y: Second array [N, D2].

    Returns:
        Distance correlation in [0, 1].
    """
    n = X.shape[0]
    if n < 2:
        return 0.0

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    dX = squareform(pdist(X, "euclidean"))
    dY = squareform(pdist(Y, "euclidean"))

    # Double center
    dX = dX - dX.mean(axis=0, keepdims=True) - dX.mean(axis=1, keepdims=True) + dX.mean()
    dY = dY - dY.mean(axis=0, keepdims=True) - dY.mean(axis=1, keepdims=True) + dY.mean()

    # Compute dCov^2 and dVar^2
    dCov2 = (dX * dY).sum() / (n * n)
    dVarX2 = (dX * dX).sum() / (n * n)
    dVarY2 = (dY * dY).sum() / (n * n)

    # Compute dCor
    if dVarX2 * dVarY2 == 0:
        return 0.0

    dCor = np.sqrt(dCov2) / np.sqrt(np.sqrt(dVarX2) * np.sqrt(dVarY2))
    return float(np.clip(dCor, 0, 1))


def compute_dcor_matrix(
    features: np.ndarray,
    partition_indices: Dict[str, Tuple[int, int]],
    max_samples: int = 500,
) -> Dict[str, float]:
    """Compute distance correlation between partitions.

    Distance correlation is computationally expensive O(n²), so we
    optionally subsample for large datasets.

    Args:
        features: Feature array [N, D].
        partition_indices: Dict mapping partition names to (start, end) indices.
        max_samples: Maximum samples to use (for speed).

    Returns:
        Dict with dCor for each partition pair.
    """
    n = features.shape[0]

    # Subsample if needed
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        features = features[idx]

    results = {}
    partition_names = list(partition_indices.keys())

    for i, name_i in enumerate(partition_names):
        for j, name_j in enumerate(partition_names[i + 1:], i + 1):
            start_i, end_i = partition_indices[name_i]
            start_j, end_j = partition_indices[name_j]

            X = features[:, start_i:end_i]
            Y = features[:, start_j:end_j]

            dcor = distance_correlation(X, Y)
            results[f"dcor_{name_i}_{name_j}"] = dcor

    return results


def compute_variance_per_dim(features: np.ndarray) -> np.ndarray:
    """Compute variance per feature dimension.

    Low variance dimensions may indicate collapse.

    Args:
        features: Feature array [N, D].

    Returns:
        Variance per dimension [D].
    """
    return features.var(axis=0)


def evaluate_latent_quality(
    features: np.ndarray,
    semantic_targets: Dict[str, np.ndarray],
    partition_indices: Optional[Dict[str, Tuple[int, int]]] = None,
    alpha: float = 1.0,
    test_split: float = 0.2,
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """Comprehensive latent space quality evaluation.

    Args:
        features: Encoder features [N, D].
        semantic_targets: Dict with 'volume', 'location', 'shape' arrays.
        partition_indices: Optional partition boundaries for correlation analysis.
        alpha: Ridge regularization.
        test_split: Fraction for test set.

    Returns:
        Dict with:
            - r2_volume, r2_location, r2_shape: R² scores
            - cross_correlation: Partition correlations (if indices provided)
            - dcor: Distance correlations (if indices provided)
            - variance_per_dim: Per-dimension variance
    """
    from sklearn.model_selection import train_test_split

    # Split data
    indices = np.arange(len(features))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_split, random_state=42
    )

    X_train, X_test = features[train_idx], features[test_idx]

    results = {}

    # Linear probe R² scores
    probes = SemanticProbes(input_dim=features.shape[1], alpha=alpha)

    targets_train = {k: v[train_idx] for k, v in semantic_targets.items()}
    targets_test = {k: v[test_idx] for k, v in semantic_targets.items()}

    probes.fit(X_train, targets_train)
    probe_results = probes.evaluate(X_test, targets_test)

    for name, res in probe_results.items():
        results[f"r2_{name}"] = res.r2
        results[f"r2_{name}_per_dim"] = res.r2_per_dim.tolist()

    results["r2_mean"] = np.mean([res.r2 for res in probe_results.values()])

    # Correlation analysis (if partitions provided)
    if partition_indices is not None:
        results["partition_correlation"] = compute_partition_correlation(
            features, partition_indices
        )
        results["dcor"] = compute_dcor_matrix(
            features, partition_indices
        )

    # Variance analysis
    results["variance_per_dim"] = compute_variance_per_dim(features).tolist()
    results["variance_mean"] = float(np.mean(results["variance_per_dim"]))
    results["variance_min"] = float(np.min(results["variance_per_dim"]))

    return results
