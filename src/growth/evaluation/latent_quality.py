# src/growth/evaluation/latent_quality.py
"""Latent space quality metrics for evaluating encoder features.

This module provides tools to evaluate the quality of learned representations
through GP probing and domain-shift metrics.

Key Components:
    - GP probes (via gp_probes.py): GPProbe, GPSemanticProbes for feature evaluation
    - R² metrics: Per-dimension and aggregate R² scores
    - Domain shift: CKA, MMD, distance correlation, DCI
    - Correlation analysis: Cross-partition correlation matrices
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_r2_scores(
    X: np.ndarray,
    y: np.ndarray,
    test_split: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute R² scores using train/test split with GP-linear probe.

    Convenience function for quick R² evaluation.

    Args:
        X: Encoder features [N, D].
        y: Target features [N, K].
        test_split: Fraction of data for testing.
        random_state: Random seed for split.

    Returns:
        Dict with 'r2', 'r2_per_dim', 'mse'.
    """
    from sklearn.model_selection import train_test_split

    from .gp_probes import GPProbe

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )

    probe = GPProbe(kernel_type="linear", r2_ci_samples=0)
    probe.fit(X_train, y_train)
    results = probe.evaluate(X_test, y_test)

    return {
        "r2": results.r2,
        "r2_per_dim": results.r2_per_dim.tolist(),
        "mse": results.mse,
    }


def compute_cross_correlation(
    features: np.ndarray,
    partition_indices: dict[str, tuple[int, int]] | None = None,
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
    partition_indices: dict[str, tuple[int, int]],
) -> dict[str, float]:
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
        for j, name_j in enumerate(partition_names[i + 1 :], i + 1):
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
    partition_indices: dict[str, tuple[int, int]],
    max_samples: int = 500,
) -> dict[str, float]:
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
        for j, name_j in enumerate(partition_names[i + 1 :], i + 1):
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


def compute_effective_rank(features: np.ndarray) -> float:
    """Compute effective rank of feature matrix.

    Effective rank measures how many dimensions are actively used.
    Based on the entropy of normalized singular values.

    A higher effective rank indicates better use of the feature space.

    Args:
        features: Feature array [N, D].

    Returns:
        Effective rank in [1, min(N, D)].
    """
    # Center features
    features_centered = features - features.mean(axis=0)

    # SVD
    _, s, _ = np.linalg.svd(features_centered, full_matrices=False)

    # Normalize singular values to probabilities
    s_norm = s / s.sum()

    # Remove zeros to avoid log(0)
    s_norm = s_norm[s_norm > 1e-10]

    # Entropy
    entropy = -np.sum(s_norm * np.log(s_norm))

    # Effective rank = exp(entropy)
    return float(np.exp(entropy))


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "rbf",
    gamma: float | None = None,
) -> float:
    """Compute Maximum Mean Discrepancy between two distributions.

    MMD measures the distance between distributions in a reproducing kernel
    Hilbert space. Lower MMD indicates more similar distributions.

    Args:
        X: First sample [N1, D].
        Y: Second sample [N2, D].
        kernel: Kernel type ("rbf" or "linear").
        gamma: RBF kernel bandwidth. If None, uses median heuristic.

    Returns:
        MMD² value (unbiased estimator).
    """
    from scipy.spatial.distance import cdist

    n, m = len(X), len(Y)

    if kernel == "linear":
        # Linear kernel: K(x, y) = x^T y
        K_XX = X @ X.T
        K_YY = Y @ Y.T
        K_XY = X @ Y.T
    else:
        # RBF kernel with median heuristic for bandwidth
        if gamma is None:
            # Median heuristic
            XY = np.vstack([X, Y])
            dists = cdist(XY, XY, "sqeuclidean")
            median_dist = np.median(dists[dists > 0])
            gamma = 1.0 / median_dist if median_dist > 0 else 1.0

        K_XX = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
        K_YY = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
        K_XY = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))

    # Unbiased MMD² estimator
    # MMD² = E[K(X,X')] + E[K(Y,Y')] - 2*E[K(X,Y)]
    mmd2 = (
        (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
        + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
        - 2 * K_XY.mean()
    )

    return max(0.0, float(mmd2))  # Clamp to non-negative


def compute_cka(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """Centered Kernel Alignment between two feature matrices.

    CKA measures representation similarity between two sets of features.
    Uses linear kernel: CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F).

    Args:
        X: First feature matrix [N, D1].
        Y: Second feature matrix [N, D2].

    Returns:
        CKA score in [0, 1]. 1.0 means identical representations
        (up to linear transform), 0.0 means orthogonal.

    References:
        Kornblith et al. (2019). "Similarity of Neural Network Representations
        Revisited." ICML.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same number of samples: {X.shape[0]} vs {Y.shape[0]}")

    # Center both matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute HSIC terms with linear kernel
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_xy = np.linalg.norm(YtX, "fro") ** 2
    hsic_xx = np.linalg.norm(XtX, "fro")
    hsic_yy = np.linalg.norm(YtY, "fro")

    denom = hsic_xx * hsic_yy
    if denom == 0:
        return 0.0

    return float(np.clip(hsic_xy / denom, 0.0, 1.0))


def mmd_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_perm: int = 1000,
    kernel: str = "rbf",
    gamma: float | None = None,
) -> tuple[float, float]:
    """MMD with permutation test for statistical significance.

    Computes MMD² between X and Y, then estimates the p-value by
    repeatedly permuting the combined samples and recomputing MMD².

    Args:
        X: First sample [N1, D].
        Y: Second sample [N2, D].
        n_perm: Number of permutations for p-value estimation.
        kernel: Kernel type for MMD ("rbf" or "linear").
        gamma: RBF kernel bandwidth. If None, uses median heuristic.

    Returns:
        Tuple of (mmd_squared, p_value). p_value < 0.05 indicates
        statistically significant distributional difference.
    """
    observed_mmd = compute_mmd(X, Y, kernel=kernel, gamma=gamma)

    # Pool samples for permutation test
    combined = np.vstack([X, Y])
    n = len(X)
    total = len(combined)

    rng = np.random.RandomState(42)
    count_ge = 0

    for _ in range(n_perm):
        perm = rng.permutation(total)
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]
        perm_mmd = compute_mmd(X_perm, Y_perm, kernel=kernel, gamma=gamma)
        if perm_mmd >= observed_mmd:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perm + 1)  # +1 for continuity correction
    return float(observed_mmd), float(p_value)


def compute_domain_classifier_accuracy(
    source_features: np.ndarray,
    target_features: np.ndarray,
    n_splits: int = 5,
) -> float:
    """Train domain classifier and return accuracy.

    Lower accuracy indicates more domain-invariant features.
    Random chance = 0.5 for balanced binary classification.

    Args:
        source_features: Source domain features [N1, D].
        target_features: Target domain features [N2, D].
        n_splits: Number of cross-validation splits.

    Returns:
        Mean cross-validated accuracy in [0, 1].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    X = np.vstack([source_features, target_features])
    y = np.array([0] * len(source_features) + [1] * len(target_features))

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train domain classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=n_splits)

    return float(scores.mean())


def compute_proxy_a_distance(
    source_features: np.ndarray,
    target_features: np.ndarray,
) -> float:
    """Compute Proxy A-distance between domains.

    PAD = 2 * (1 - 2 * error), where error is domain classifier error.
    PAD ∈ [0, 2], with 0 meaning indistinguishable domains.

    Args:
        source_features: Source domain features.
        target_features: Target domain features.

    Returns:
        Proxy A-distance in [0, 2].
    """
    acc = compute_domain_classifier_accuracy(source_features, target_features)
    error = 1.0 - acc
    return 2.0 * (1.0 - 2.0 * error)


@dataclass
class DomainShiftMetrics:
    """Metrics for evaluating domain shift."""

    mmd: float
    domain_classifier_accuracy: float
    proxy_a_distance: float
    source_effective_rank: float
    target_effective_rank: float
    cka: float | None = None
    mmd_pvalue: float | None = None


def compute_domain_shift_metrics(
    source_features: np.ndarray,
    target_features: np.ndarray,
    max_samples: int = 500,
) -> DomainShiftMetrics:
    """Compute comprehensive domain shift metrics.

    Args:
        source_features: Source domain features [N1, D].
        target_features: Target domain features [N2, D].
        max_samples: Maximum samples per domain (for speed).

    Returns:
        DomainShiftMetrics with all metrics.
    """
    # Subsample if needed
    if len(source_features) > max_samples:
        idx = np.random.choice(len(source_features), max_samples, replace=False)
        source_features = source_features[idx]
    if len(target_features) > max_samples:
        idx = np.random.choice(len(target_features), max_samples, replace=False)
        target_features = target_features[idx]

    # Align to same size for CKA (requires equal N)
    n_min = min(len(source_features), len(target_features))
    source_aligned = source_features[:n_min]
    target_aligned = target_features[:n_min]

    mmd_val, mmd_pval = mmd_permutation_test(source_features, target_features, n_perm=100)

    return DomainShiftMetrics(
        mmd=mmd_val,
        domain_classifier_accuracy=compute_domain_classifier_accuracy(
            source_features, target_features
        ),
        proxy_a_distance=compute_proxy_a_distance(source_features, target_features),
        source_effective_rank=compute_effective_rank(source_features),
        target_effective_rank=compute_effective_rank(target_features),
        cka=compute_cka(source_aligned, target_aligned),
        mmd_pvalue=mmd_pval,
    )


@dataclass
class DCIResults:
    """Results from DCI disentanglement evaluation.

    Attributes:
        disentanglement: Weighted-average disentanglement D in [0, 1].
        completeness: Weighted-average completeness C in [0, 1].
        informativeness: Mean LASSO R² across factors.
        importance_matrix: LASSO importance matrix [n_factors, n_latent_dims].
        r2_per_factor: R² per target factor from LASSO.
    """

    disentanglement: float
    completeness: float
    informativeness: float
    importance_matrix: np.ndarray
    r2_per_factor: np.ndarray


def compute_dci(
    z: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 5000,
) -> DCIResults:
    """Compute DCI disentanglement, completeness, and informativeness.

    Implements Eastwood & Williams (2018): "A framework for the quantitative
    evaluation of disentangled representations."

    Algorithm:
        1. For each target factor, train LASSO from all latent dims.
        2. Extract importance matrix R[j, d] = |coef_j[d]|.
        3. Disentanglement D_d = 1 - H(R_norm[:, d]) / log(n_factors).
        4. Completeness C_j = 1 - H(R_norm[j, :]) / log(n_dims).
        5. Informativeness = mean R² of LASSO predictions.

    Args:
        z: Latent representations [N, D].
        targets: Target factors [N, K].
        alpha: LASSO regularization strength.
        max_iter: Maximum iterations for LASSO.

    Returns:
        DCIResults with D, C, informativeness, and importance matrix.
    """
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import cross_val_score

    n_samples, n_dims = z.shape
    n_factors = targets.shape[1]

    # Standardize inputs
    scaler_z = StandardScaler()
    z_scaled = scaler_z.fit_transform(z)

    scaler_t = StandardScaler()
    targets_scaled = scaler_t.fit_transform(targets)

    # Build importance matrix: train LASSO for each factor
    importance = np.zeros((n_factors, n_dims))
    r2_per_factor = np.zeros(n_factors)

    for j in range(n_factors):
        y = targets_scaled[:, j]

        # Skip constant targets
        if np.std(y) < 1e-10:
            continue

        # Fit full model for importance matrix (unchanged)
        lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
        lasso.fit(z_scaled, y)

        importance[j, :] = np.abs(lasso.coef_)

        # Cross-validated R² for informativeness (avoids inflation)
        n_cv = min(5, max(2, len(z_scaled)))
        cv_scores = cross_val_score(
            Lasso(alpha=alpha, max_iter=max_iter, random_state=42),
            z_scaled,
            y,
            cv=n_cv,
            scoring="r2",
        )
        r2_per_factor[j] = max(0.0, float(cv_scores.mean()))

    # Disentanglement: per latent dim, then weighted average
    # D_d = 1 - H(importance_norm[:, d]) / log(n_factors)
    col_sums = importance.sum(axis=0)  # [n_dims]
    active_dims = col_sums > 1e-10
    n_active = active_dims.sum()

    if n_active == 0 or n_factors < 2:
        return DCIResults(
            disentanglement=0.0,
            completeness=0.0,
            informativeness=float(r2_per_factor.mean()),
            importance_matrix=importance,
            r2_per_factor=r2_per_factor,
        )

    # Normalize columns for disentanglement
    importance_col_norm = importance[:, active_dims] / col_sums[active_dims][np.newaxis, :]
    # Clamp for numerical stability
    importance_col_norm = np.clip(importance_col_norm, 1e-10, 1.0)

    # Per-dim entropy
    log_n_factors = np.log(n_factors)
    H_d = -np.sum(importance_col_norm * np.log(importance_col_norm), axis=0) / log_n_factors
    D_d = 1.0 - H_d  # [n_active]

    # Weighted average by column importance
    weights_d = col_sums[active_dims] / col_sums[active_dims].sum()
    disentanglement = float(np.sum(weights_d * D_d))

    # Completeness: per factor, then weighted average
    # C_j = 1 - H(importance_norm[j, :]) / log(n_dims)
    row_sums = importance.sum(axis=1)  # [n_factors]
    active_factors = row_sums > 1e-10

    if active_factors.sum() == 0 or n_dims < 2:
        completeness = 0.0
    else:
        importance_row_norm = importance[active_factors, :] / row_sums[active_factors, np.newaxis]
        importance_row_norm = np.clip(importance_row_norm, 1e-10, 1.0)

        log_n_dims = np.log(n_dims)
        H_j = -np.sum(importance_row_norm * np.log(importance_row_norm), axis=1) / log_n_dims
        C_j = 1.0 - H_j

        weights_j = row_sums[active_factors] / row_sums[active_factors].sum()
        completeness = float(np.sum(weights_j * C_j))

    informativeness = float(r2_per_factor.mean())

    return DCIResults(
        disentanglement=disentanglement,
        completeness=completeness,
        informativeness=informativeness,
        importance_matrix=importance,
        r2_per_factor=r2_per_factor,
    )


def evaluate_latent_quality(
    features: np.ndarray,
    semantic_targets: dict[str, np.ndarray],
    partition_indices: dict[str, tuple[int, int]] | None = None,
    test_split: float = 0.2,
) -> dict[str, float | dict | np.ndarray]:
    """Comprehensive latent space quality evaluation.

    Args:
        features: Encoder features [N, D].
        semantic_targets: Dict with 'volume', 'location', 'shape' arrays.
        partition_indices: Optional partition boundaries for correlation analysis.
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
    train_idx, test_idx = train_test_split(indices, test_size=test_split, random_state=42)

    X_train, X_test = features[train_idx], features[test_idx]

    results = {}

    # GP-linear probe R² scores
    from .gp_probes import GPSemanticProbes

    targets_train = {k: v[train_idx] for k, v in semantic_targets.items()}
    targets_test = {k: v[test_idx] for k, v in semantic_targets.items()}

    probes = GPSemanticProbes(
        input_dim=features.shape[1],
        n_restarts=0,
        r2_ci_samples=0,
    )
    probes.fit(X_train, targets_train)
    gp_results = probes.evaluate(X_test, targets_test)

    for name, res in gp_results.linear.items():
        results[f"r2_{name}"] = res.r2
        results[f"r2_{name}_per_dim"] = res.r2_per_dim.tolist()

    results["r2_mean"] = np.mean([res.r2 for res in gp_results.linear.values()])

    # Correlation analysis (if partitions provided)
    if partition_indices is not None:
        results["partition_correlation"] = compute_partition_correlation(
            features, partition_indices
        )
        results["dcor"] = compute_dcor_matrix(features, partition_indices)

    # Variance analysis
    results["variance_per_dim"] = compute_variance_per_dim(features).tolist()
    results["variance_mean"] = float(np.mean(results["variance_per_dim"]))
    results["variance_min"] = float(np.min(results["variance_per_dim"]))

    return results
