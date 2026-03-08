# tests/growth/test_latent_quality.py
"""Tests for latent space quality metrics (domain-shift, correlation, variance).

Probe-specific tests are in test_gp_probes.py.
"""

import numpy as np
import pytest

from growth.evaluation.latent_quality import (
    compute_cka,
    compute_dcor_matrix,
    compute_partition_correlation,
    compute_r2_scores,
    compute_variance_per_dim,
    distance_correlation,
    evaluate_latent_quality,
    mmd_permutation_test,
)

pytestmark = [pytest.mark.evaluation, pytest.mark.unit]


class TestDistanceCorrelation:
    """Tests for distance correlation."""

    def test_independent_arrays(self):
        """Test low dCor for independent arrays."""
        np.random.seed(42)
        X = np.random.randn(500, 10)
        Y = np.random.randn(500, 10)

        dcor = distance_correlation(X, Y)

        # Should be relatively low for independent data
        assert 0 <= dcor <= 1
        assert dcor < 0.5

    def test_dependent_arrays(self):
        """Test high dCor for dependent arrays."""
        np.random.seed(42)
        X = np.random.randn(500, 10)
        Y = X + np.random.randn(500, 10) * 0.1

        dcor = distance_correlation(X, Y)

        assert dcor > 0.9

    def test_dCor_bounded(self):
        """Test dCor is in [0, 1]."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5)

        dcor = distance_correlation(X, Y)

        assert 0 <= dcor <= 1

    def test_small_sample_returns_zero(self):
        """Test small sample returns 0."""
        X = np.array([[1.0]])
        Y = np.array([[1.0]])

        dcor = distance_correlation(X, Y)

        assert dcor == 0.0


class TestComputePartitionCorrelation:
    """Tests for partition correlation."""

    def test_basic_partition_correlation(self):
        """Test computing partition correlations."""
        np.random.seed(42)
        features = np.random.randn(100, 30)
        partition_indices = {
            "a": (0, 10),
            "b": (10, 20),
            "c": (20, 30),
        }

        corrs = compute_partition_correlation(features, partition_indices)

        assert "a_b" in corrs
        assert "a_c" in corrs
        assert "b_c" in corrs
        assert all(0 <= v <= 1 for v in corrs.values())


class TestComputeDcorMatrix:
    """Tests for distance correlation matrix."""

    def test_basic_dcor_matrix(self):
        """Test computing dCor between partitions."""
        np.random.seed(42)
        features = np.random.randn(100, 30)
        partition_indices = {
            "a": (0, 10),
            "b": (10, 20),
            "c": (20, 30),
        }

        dcors = compute_dcor_matrix(features, partition_indices)

        assert "dcor_a_b" in dcors
        assert "dcor_a_c" in dcors
        assert "dcor_b_c" in dcors


class TestComputeVariancePerDim:
    """Tests for variance computation."""

    def test_variance_shape(self):
        """Test variance returns correct shape."""
        features = np.random.randn(100, 50)
        variance = compute_variance_per_dim(features)

        assert variance.shape == (50,)

    def test_variance_positive(self):
        """Test variance is positive."""
        features = np.random.randn(100, 50)
        variance = compute_variance_per_dim(features)

        assert (variance >= 0).all()


class TestComputeR2Scores:
    """Tests for convenience R² function."""

    def test_basic_r2_scores(self):
        """Test computing R² scores."""
        np.random.seed(42)
        n_samples = 500
        X = np.random.randn(n_samples, 64)
        W = np.random.randn(64, 4) * 0.1
        y = X @ W + np.random.randn(n_samples, 4) * 0.01

        scores = compute_r2_scores(X, y)

        assert "r2" in scores
        assert "r2_per_dim" in scores
        assert "mse" in scores
        assert scores["r2"] > 0.9


class TestEvaluateLatentQuality:
    """Tests for comprehensive evaluation function."""

    def test_basic_evaluation(self):
        """Test basic latent quality evaluation."""
        np.random.seed(42)
        features = np.random.randn(100, 32)
        semantic_targets = {
            "volume": np.random.randn(100, 4),
            "location": np.random.randn(100, 3),
            "shape": np.random.randn(100, 3),
        }

        quality = evaluate_latent_quality(features, semantic_targets)

        assert "r2_volume" in quality
        assert "r2_location" in quality
        assert "r2_shape" in quality
        assert "r2_mean" in quality
        assert "variance_per_dim" in quality

    def test_evaluation_with_partitions(self):
        """Test evaluation with partition analysis."""
        np.random.seed(42)
        features = np.random.randn(100, 32)
        semantic_targets = {
            "volume": np.random.randn(100, 4),
            "location": np.random.randn(100, 3),
            "shape": np.random.randn(100, 3),
        }
        partition_indices = {
            "vol": (0, 10),
            "loc": (10, 18),
            "shape": (18, 32),
        }

        quality = evaluate_latent_quality(features, semantic_targets, partition_indices)

        assert "partition_correlation" in quality
        assert "dcor" in quality


class TestComputeCKA:
    """Tests for Centered Kernel Alignment."""

    def test_compute_cka_identical(self):
        """CKA of identical matrices should be 1.0."""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        cka = compute_cka(X, X)

        assert abs(cka - 1.0) < 1e-6

    def test_compute_cka_orthogonal(self):
        """CKA of orthogonal matrices should be ~0.0."""
        np.random.seed(42)
        n = 200
        # Create orthogonal features via QR decomposition
        A = np.random.randn(n, 20)
        B = np.random.randn(n, 20)
        # Remove correlation: project B onto orthogonal complement of A
        Q, _ = np.linalg.qr(A)
        B_orth = B - Q @ (Q.T @ B)

        cka = compute_cka(A, B_orth)

        assert cka < 0.1

    def test_compute_cka_range(self):
        """CKA should always be in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            X = np.random.randn(50, 30)
            Y = np.random.randn(50, 20)

            cka = compute_cka(X, Y)

            assert 0.0 <= cka <= 1.0

    def test_compute_cka_shape_mismatch_raises(self):
        """CKA should raise if sample counts differ."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(60, 10)

        with pytest.raises(ValueError, match="same number of samples"):
            compute_cka(X, Y)


class TestMMDPermutationTest:
    """Tests for MMD permutation test."""

    def test_mmd_permutation_identical(self):
        """p-value should be > 0.05 for identical distributions."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        Y = np.random.randn(50, 20)

        mmd_val, p_value = mmd_permutation_test(X, Y, n_perm=200)

        assert 0 <= p_value <= 1
        assert p_value > 0.05  # Not significantly different

    def test_mmd_permutation_shifted(self):
        """p-value should be < 0.05 for shifted distributions."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        Y = np.random.randn(50, 20) + 3.0  # Large shift

        mmd_val, p_value = mmd_permutation_test(X, Y, n_perm=200)

        assert mmd_val > 0
        assert p_value < 0.05  # Significantly different

    def test_mmd_permutation_returns_tuple(self):
        """Should return (mmd_squared, p_value) tuple."""
        np.random.seed(42)
        X = np.random.randn(30, 10)
        Y = np.random.randn(30, 10)

        result = mmd_permutation_test(X, Y, n_perm=50)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
