# tests/growth/test_latent_quality.py
"""Tests for latent space quality metrics."""

import numpy as np
import pytest

from growth.evaluation.latent_quality import (
    LinearProbe,
    SemanticProbes,
    ProbeResults,
    compute_r2_scores,
    distance_correlation,
    compute_partition_correlation,
    compute_dcor_matrix,
    compute_variance_per_dim,
    evaluate_latent_quality,
)


class TestLinearProbe:
    """Tests for LinearProbe class."""

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        np.random.seed(42)
        X = np.random.randn(100, 64)
        y = np.random.randn(100, 4)

        probe = LinearProbe(input_dim=64, output_dim=4)
        probe.fit(X[:80], y[:80])
        predictions = probe.predict(X[80:])

        assert predictions.shape == (20, 4)

    def test_high_r2_for_linear_data(self):
        """Test high R² for linearly related data."""
        np.random.seed(42)
        n_samples = 500
        input_dim = 64
        output_dim = 4

        X = np.random.randn(n_samples, input_dim)
        W = np.random.randn(input_dim, output_dim) * 0.1
        y = X @ W + np.random.randn(n_samples, output_dim) * 0.01

        probe = LinearProbe(input_dim=input_dim, output_dim=output_dim)
        probe.fit(X[:400], y[:400])
        results = probe.evaluate(X[400:], y[400:])

        assert results.r2 > 0.95

    def test_low_r2_for_random_data(self):
        """Test low R² for unrelated data."""
        np.random.seed(42)
        X = np.random.randn(200, 32)
        y = np.random.randn(200, 4)

        probe = LinearProbe(input_dim=32, output_dim=4)
        probe.fit(X[:160], y[:160])
        results = probe.evaluate(X[160:], y[160:])

        assert results.r2 < 0.3

    def test_evaluate_returns_probe_results(self):
        """Test evaluate returns ProbeResults dataclass."""
        np.random.seed(42)
        X = np.random.randn(100, 32)
        y = np.random.randn(100, 4)

        probe = LinearProbe(input_dim=32, output_dim=4)
        probe.fit(X[:80], y[:80])
        results = probe.evaluate(X[80:], y[80:])

        assert isinstance(results, ProbeResults)
        assert isinstance(results.r2, float)
        assert results.r2_per_dim.shape == (4,)
        assert results.predictions.shape == (20, 4)
        assert results.coefficients.shape == (4, 32)

    def test_input_dim_validation(self):
        """Test input dimension validation."""
        probe = LinearProbe(input_dim=64, output_dim=4)

        with pytest.raises(ValueError, match="Expected input_dim"):
            probe.fit(np.random.randn(100, 32), np.random.randn(100, 4))

    def test_output_dim_validation(self):
        """Test output dimension validation."""
        probe = LinearProbe(input_dim=64, output_dim=4)

        with pytest.raises(ValueError, match="Expected output_dim"):
            probe.fit(np.random.randn(100, 64), np.random.randn(100, 8))

    def test_predict_before_fit_raises(self):
        """Test predicting before fitting raises error."""
        probe = LinearProbe(input_dim=64, output_dim=4)

        with pytest.raises(RuntimeError, match="must be fitted"):
            probe.predict(np.random.randn(10, 64))


class TestSemanticProbes:
    """Tests for SemanticProbes class."""

    def test_fit_and_evaluate(self):
        """Test fitting and evaluating all probes."""
        np.random.seed(42)
        X = np.random.randn(200, 768)
        targets = {
            "volume": np.random.randn(200, 4),
            "location": np.random.randn(200, 3),
            "shape": np.random.randn(200, 6),
        }

        probes = SemanticProbes(input_dim=768)
        probes.fit(X[:160], {k: v[:160] for k, v in targets.items()})
        results = probes.evaluate(X[160:], {k: v[160:] for k, v in targets.items()})

        assert "volume" in results
        assert "location" in results
        assert "shape" in results
        assert all(isinstance(r, ProbeResults) for r in results.values())

    def test_missing_target_raises(self):
        """Test missing target in fit raises error."""
        probes = SemanticProbes(input_dim=768)
        X = np.random.randn(100, 768)
        targets = {"volume": np.random.randn(100, 4)}  # Missing location, shape

        with pytest.raises(ValueError, match="Missing target"):
            probes.fit(X, targets)

    def test_get_summary(self):
        """Test get_summary returns R² dict."""
        np.random.seed(42)
        X = np.random.randn(200, 768)
        targets = {
            "volume": np.random.randn(200, 4),
            "location": np.random.randn(200, 3),
            "shape": np.random.randn(200, 6),
        }

        probes = SemanticProbes(input_dim=768)
        probes.fit(X[:160], {k: v[:160] for k, v in targets.items()})
        results = probes.evaluate(X[160:], {k: v[160:] for k, v in targets.items()})
        summary = probes.get_summary(results)

        assert "r2_volume" in summary
        assert "r2_location" in summary
        assert "r2_shape" in summary
        assert "r2_mean" in summary


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
            "shape": np.random.randn(100, 6),
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
            "shape": np.random.randn(100, 6),
        }
        partition_indices = {
            "vol": (0, 10),
            "loc": (10, 18),
            "shape": (18, 32),
        }

        quality = evaluate_latent_quality(
            features, semantic_targets, partition_indices
        )

        assert "partition_correlation" in quality
        assert "dcor" in quality
