# tests/growth/test_domain_divergence.py
"""Tests for domain divergence analysis engine (synthetic data, CPU only)."""

import numpy as np
import pytest

from experiments.uncertainty_segmentation.explainability.engine.domain_divergence import (
    STAGE_CHANNELS,
    StageDomainMetrics,
    apply_bh_correction,
    compute_cka_adaptation_drift,
    compute_cka_cross_stage,
    compute_drift_divergence_correlation,
    compute_fsd,
    compute_per_stage_domain_metrics,
)

pytestmark = [pytest.mark.evaluation, pytest.mark.unit]


# ---------------------------------------------------------------------------
# FSD tests
# ---------------------------------------------------------------------------


class TestComputeFSD:
    """Tests for Feature Statistics Divergence."""

    def test_identical_distributions(self):
        """FSD(X, X) should be approximately 0."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 16).astype(np.float32)
        fsd = compute_fsd(X, X)
        np.testing.assert_allclose(fsd, 0.0, atol=1e-6)

    def test_shifted_scales_with_delta(self):
        """FSD(X, X + delta) should increase monotonically with delta."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 16).astype(np.float32)

        deltas = [0.1, 0.5, 1.0, 3.0]
        fsd_values = [compute_fsd(X, X + d) for d in deltas]

        for i in range(len(fsd_values) - 1):
            assert fsd_values[i] < fsd_values[i + 1], (
                f"FSD should increase with delta: "
                f"FSD(delta={deltas[i]})={fsd_values[i]:.4f} >= "
                f"FSD(delta={deltas[i+1]})={fsd_values[i+1]:.4f}"
            )

    def test_dimension_invariant(self):
        """FSD is a mean over channels, so two feature dims with the same
        per-channel shift should yield similar FSD values."""
        rng = np.random.RandomState(42)
        delta = 2.0

        # 16-channel and 128-channel features with the same shift magnitude
        X_small = rng.randn(50, 16).astype(np.float32)
        X_big = rng.randn(50, 128).astype(np.float32)

        fsd_small = compute_fsd(X_small, X_small + delta)
        fsd_big = compute_fsd(X_big, X_big + delta)

        # Both should be close (same shift, same unit variance)
        np.testing.assert_allclose(fsd_small, fsd_big, rtol=0.3)

    def test_mismatched_channels_raises(self):
        """Mismatched channel dimensions should raise ValueError."""
        A = np.random.randn(10, 16).astype(np.float32)
        B = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(ValueError, match="Channel dimensions must match"):
            compute_fsd(A, B)


# ---------------------------------------------------------------------------
# Per-stage metrics smoke test
# ---------------------------------------------------------------------------


class TestPerStageMetrics:
    """Smoke test for the full per-stage metric battery."""

    def test_all_metrics_valid_ranges(self):
        """Random features should produce metrics in their valid ranges."""
        rng = np.random.RandomState(42)
        stages = (0, 1)
        gli = {0: rng.randn(20, 16).astype(np.float32),
               1: rng.randn(20, 32).astype(np.float32)}
        men = {0: rng.randn(20, 16).astype(np.float32),
               1: rng.randn(20, 32).astype(np.float32)}

        results = compute_per_stage_domain_metrics(
            gli, men, stages=stages,
            n_mmd_perm=100,      # fast
            n_bootstrap=100,     # fast
        )

        for s in stages:
            m = results[s]
            assert isinstance(m, StageDomainMetrics)
            # Domain accuracy in [0, 1]
            assert 0.0 <= m.domain_acc_linear <= 1.0
            assert 0.0 <= m.domain_acc_mlp <= 1.0
            # PAD in [-2, 2] (can be negative with random features)
            assert -2.0 <= m.pad <= 2.0
            # FSD >= 0
            assert m.fsd >= 0.0
            # MMD p-value in [0, 1]
            assert 0.0 <= m.mmd_p <= 1.0
            # CI ordering
            assert m.domain_acc_ci_lower <= m.domain_acc_ci_upper
            # Metadata
            assert m.n_gli == 20
            assert m.n_men == 20
            assert m.stage == s


# ---------------------------------------------------------------------------
# CKA tests
# ---------------------------------------------------------------------------


class TestCKACrossStage:
    """Tests for cross-stage CKA matrix computation."""

    def test_diagonal_is_one(self):
        """CKA(X, X) = 1.0 for every stage on the diagonal."""
        rng = np.random.RandomState(42)
        features = {
            0: rng.randn(30, 16).astype(np.float32),
            1: rng.randn(30, 32).astype(np.float32),
            2: rng.randn(30, 64).astype(np.float32),
        }
        matrix = compute_cka_cross_stage(features)

        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-5)

    def test_symmetry(self):
        """CKA matrix should be symmetric."""
        rng = np.random.RandomState(42)
        features = {
            0: rng.randn(30, 16).astype(np.float32),
            1: rng.randn(30, 32).astype(np.float32),
        }
        matrix = compute_cka_cross_stage(features)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)

    def test_mismatched_n_raises(self):
        """Mismatched sample counts should raise ValueError."""
        features = {
            0: np.random.randn(10, 16).astype(np.float32),
            1: np.random.randn(15, 32).astype(np.float32),
        }
        with pytest.raises(ValueError, match="same N"):
            compute_cka_cross_stage(features)


class TestCKAAdaptationDrift:
    """Tests for frozen-vs-adapted CKA drift computation."""

    def test_identity_drift_is_one(self):
        """CKA(frozen, frozen) = 1.0 per stage (no adaptation drift)."""
        rng = np.random.RandomState(42)
        features = {
            0: rng.randn(30, 16).astype(np.float32),
            1: rng.randn(30, 32).astype(np.float32),
            2: rng.randn(30, 64).astype(np.float32),
        }
        drift = compute_cka_adaptation_drift(features, features)

        for s in features:
            np.testing.assert_allclose(drift[s], 1.0, atol=1e-5)

    def test_perturbed_drift_is_below_one(self):
        """Adding noise to adapted features should lower CKA below 1.0."""
        rng = np.random.RandomState(42)
        frozen = {0: rng.randn(30, 16).astype(np.float32)}
        adapted = {0: frozen[0] + 2.0 * rng.randn(30, 16).astype(np.float32)}

        drift = compute_cka_adaptation_drift(frozen, adapted)
        assert drift[0] < 1.0


# ---------------------------------------------------------------------------
# Benjamini-Hochberg correction
# ---------------------------------------------------------------------------


class TestBHCorrection:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_known_p_values(self):
        """Verify BH adjustment against a textbook example.

        p-values: [0.001, 0.008, 0.039, 0.041, 0.20]
        m = 5.  BH adjusted:
            rank 1: min(0.001 * 5/1, next) = 0.005
            rank 2: min(0.008 * 5/2, next) = 0.020
            rank 3: min(0.039 * 5/3, next) = 0.065
            rank 4: min(0.041 * 5/4, next) = 0.05125
            rank 5: 0.20 * 5/5 = 0.20
        Step-up enforced monotonicity: [0.005, 0.020, 0.05125, 0.05125, 0.20]
        """
        raw_p = np.array([0.001, 0.008, 0.039, 0.041, 0.20])
        adjusted, significant = apply_bh_correction(raw_p, alpha=0.05)

        assert len(adjusted) == 5
        assert len(significant) == 5

        # Expected adjusted p-values
        expected = np.array([0.005, 0.020, 0.065, 0.065, 0.20])
        # Step-up monotonicity: rank 3 and 4 tie at min(0.065, 0.05125) = 0.05125
        # Actually: backward pass: adj[4]=0.20, adj[3]=min(0.041*5/4, 0.20)=0.05125
        # adj[2]=min(0.039*5/3, 0.05125)=0.05125, adj[1]=min(0.008*5/2, 0.05125)=0.020
        # adj[0]=min(0.001*5/1, 0.020)=0.005
        expected_correct = np.array([0.005, 0.020, 0.05125, 0.05125, 0.20])
        np.testing.assert_allclose(adjusted, expected_correct, atol=1e-10)

        # Significance at alpha=0.05: first two are < 0.05
        assert significant[0] is True or significant[0] == True  # noqa: E712
        assert significant[1] is True or significant[1] == True  # noqa: E712
        assert not significant[2]
        assert not significant[3]
        assert not significant[4]

    def test_all_significant(self):
        """All p-values below alpha should remain significant."""
        raw_p = np.array([0.001, 0.002, 0.003])
        adjusted, significant = apply_bh_correction(raw_p, alpha=0.05)

        assert all(significant)
        assert all(adjusted <= 0.05)

    def test_all_nonsignificant(self):
        """Large p-values should remain non-significant."""
        raw_p = np.array([0.5, 0.6, 0.9])
        adjusted, significant = apply_bh_correction(raw_p, alpha=0.05)

        assert not any(significant)


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------


class TestDriftDivergenceCorrelation:
    """Tests for Spearman rank correlation between divergence and drift."""

    def test_perfect_monotone(self):
        """Perfect monotone relationship should give rho = 1.0."""
        # Stages with increasing divergence
        domain_acc = {0: 0.55, 1: 0.65, 2: 0.75, 3: 0.85, 4: 0.95}
        # CKA drift decreases (1 - CKA increases) monotonically
        cka_drift = {0: 0.95, 1: 0.85, 2: 0.75, 3: 0.65, 4: 0.55}

        rho, p = compute_drift_divergence_correlation(domain_acc, cka_drift)
        np.testing.assert_allclose(rho, 1.0, atol=1e-10)

    def test_too_few_stages_returns_default(self):
        """Fewer than 3 shared stages should return rho=0, p=1."""
        domain_acc = {0: 0.6, 1: 0.7}
        cka_drift = {0: 0.9, 1: 0.8}

        rho, p = compute_drift_divergence_correlation(domain_acc, cka_drift)
        assert rho == 0.0
        assert p == 1.0

    def test_partial_stage_overlap(self):
        """Only shared stages should be used for correlation."""
        # Stages 1, 2, 3 shared; 0 and 4 only in one dict
        domain_acc = {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8}
        cka_drift = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.5}

        rho, p = compute_drift_divergence_correlation(domain_acc, cka_drift)
        # Shared: 1,2,3 → divergence [0.6,0.7,0.8], drift (1-CKA) [0.1,0.2,0.3]
        # Perfect monotone → rho = 1.0
        np.testing.assert_allclose(rho, 1.0, atol=1e-10)
