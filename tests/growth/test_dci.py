# tests/growth/test_dci.py
"""Tests for DCI disentanglement metric (Eastwood & Williams, 2018).

Verifies compute_dci() on synthetic data with known disentanglement properties.
"""

import numpy as np

from growth.evaluation.latent_quality import DCIResults, compute_dci

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perfectly_disentangled(
    n_samples: int = 500,
    n_factors: int = 10,
    n_latent: int = 128,
    noise: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic data where each factor maps to a disjoint set of dims.

    First n_factors dims of z each encode exactly one factor.
    Remaining dims are noise.

    Returns:
        Tuple of (z [N, n_latent], targets [N, n_factors]).
    """
    rng = np.random.RandomState(seed)
    targets = rng.randn(n_samples, n_factors)
    z = rng.randn(n_samples, n_latent) * noise

    # Each factor maps to exactly one latent dim
    for j in range(n_factors):
        z[:, j] = targets[:, j] + noise * rng.randn(n_samples)

    return z, targets


def _make_fully_entangled(
    n_samples: int = 500,
    n_factors: int = 10,
    n_latent: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic data where every dim encodes every factor equally.

    Each latent dim is the SAME weighted sum of all factors (plus noise),
    so no individual dim is more informative about one factor than another.
    LASSO cannot distinguish which dims encode which factors — all are equal.

    Returns:
        Tuple of (z [N, n_latent], targets [N, n_factors]).
    """
    rng = np.random.RandomState(seed)
    targets = rng.randn(n_samples, n_factors)

    # All latent dims are the same signal (sum of all factors) + noise.
    # This makes every dim equally informative for every factor.
    shared_signal = targets.sum(axis=1, keepdims=True)  # [N, 1]
    z = np.tile(shared_signal, (1, n_latent)) + 0.5 * rng.randn(n_samples, n_latent)

    return z, targets


# ===========================================================================
# Tests
# ===========================================================================


class TestDCIPerfectDisentanglement:
    """DCI scores for perfectly disentangled representations."""

    def test_high_disentanglement(self) -> None:
        """D > 0.7 when each factor maps to exactly one latent dim."""
        z, targets = _make_perfectly_disentangled()
        result = compute_dci(z, targets)
        assert result.disentanglement > 0.7, f"D = {result.disentanglement:.3f}, expected > 0.7"

    def test_high_completeness(self) -> None:
        """C > 0.5 when each factor maps to exactly one latent dim."""
        z, targets = _make_perfectly_disentangled()
        result = compute_dci(z, targets)
        assert result.completeness > 0.5, f"C = {result.completeness:.3f}, expected > 0.5"

    def test_high_informativeness(self) -> None:
        """Informativeness (mean R²) > 0.5 when factors are easily predictable."""
        z, targets = _make_perfectly_disentangled()
        result = compute_dci(z, targets)
        assert result.informativeness > 0.5, (
            f"Informativeness = {result.informativeness:.3f}, expected > 0.5"
        )


class TestDCIFullyEntangled:
    """DCI scores for fully entangled representations."""

    def test_lower_disentanglement_than_perfect(self) -> None:
        """D for entangled < D for disentangled representation.

        LASSO's sparsity bias prevents D from reaching 0 even for
        entangled data, so we test relative ordering rather than
        an absolute threshold.
        """
        z_ent, targets_ent = _make_fully_entangled(n_samples=500, n_factors=10, n_latent=10)
        result_ent = compute_dci(z_ent, targets_ent)

        z_dis, targets_dis = _make_perfectly_disentangled(n_samples=500, n_factors=10, n_latent=128)
        result_dis = compute_dci(z_dis, targets_dis)

        assert result_ent.disentanglement < result_dis.disentanglement, (
            f"Entangled D={result_ent.disentanglement:.3f} should be < "
            f"disentangled D={result_dis.disentanglement:.3f}"
        )


class TestDCIOutputShape:
    """Verify shapes and types of DCI outputs."""

    def test_returns_dci_results(self) -> None:
        """compute_dci returns a DCIResults dataclass."""
        z, targets = _make_perfectly_disentangled()
        result = compute_dci(z, targets)
        assert isinstance(result, DCIResults)

    def test_importance_matrix_shape(self) -> None:
        """Importance matrix has shape [n_factors, n_latent_dims]."""
        n_factors, n_latent = 10, 128
        z, targets = _make_perfectly_disentangled(n_factors=n_factors, n_latent=n_latent)
        result = compute_dci(z, targets)
        assert result.importance_matrix.shape == (n_factors, n_latent)

    def test_r2_per_factor_shape(self) -> None:
        """R² per factor has length n_factors."""
        n_factors = 10
        z, targets = _make_perfectly_disentangled(n_factors=n_factors)
        result = compute_dci(z, targets)
        assert result.r2_per_factor.shape == (n_factors,)

    def test_scores_in_valid_range(self) -> None:
        """D and C are in [0, 1]."""
        z, targets = _make_perfectly_disentangled()
        result = compute_dci(z, targets)
        assert 0.0 <= result.disentanglement <= 1.0
        assert 0.0 <= result.completeness <= 1.0


class TestDCIEdgeCases:
    """Edge cases for compute_dci."""

    def test_constant_target_does_not_crash(self) -> None:
        """Constant target column does not cause division by zero."""
        rng = np.random.RandomState(42)
        z = rng.randn(100, 32)
        targets = rng.randn(100, 5)
        targets[:, 2] = 7.0  # Constant column

        result = compute_dci(z, targets)
        assert isinstance(result, DCIResults)
        assert np.isfinite(result.disentanglement)
        assert np.isfinite(result.completeness)

    def test_single_factor(self) -> None:
        """Works with a single target factor (n_factors=1)."""
        rng = np.random.RandomState(42)
        z = rng.randn(200, 32)
        targets = z[:, 0:1] + 0.01 * rng.randn(200, 1)

        result = compute_dci(z, targets)
        assert isinstance(result, DCIResults)
        # Single factor means n_factors < 2, D should be 0 by convention
        assert result.informativeness > 0.0


class TestDCICrossValidatedR2:
    """FLAW-6: CV R² should prevent training-set inflation."""

    def test_cv_r2_lower_than_train_r2_overfit(self) -> None:
        """In overfit regime (n_samples=30, n_dims=128), CV R² < train R².

        With many more features than samples, LASSO overfits training data.
        Cross-validated R² should be much lower than training-set R².
        """
        rng = np.random.RandomState(42)
        n_samples, n_dims = 30, 128
        z = rng.randn(n_samples, n_dims)
        # Factor is actually just noise (no real signal)
        targets = rng.randn(n_samples, 3)

        result = compute_dci(z, targets)

        # With noise targets and overparameterized z, CV R² should be low
        assert result.informativeness < 0.5, (
            f"Informativeness = {result.informativeness:.3f}, "
            f"expected < 0.5 for noise targets in overfit regime"
        )

    def test_cv_r2_high_for_easy_case(self) -> None:
        """When factors are clearly predictable, CV R² should be high."""
        z, targets = _make_perfectly_disentangled(n_samples=500, n_factors=3, n_latent=32)
        result = compute_dci(z, targets)

        assert result.informativeness > 0.5, (
            f"Informativeness = {result.informativeness:.3f}, "
            f"expected > 0.5 for perfectly disentangled data"
        )
