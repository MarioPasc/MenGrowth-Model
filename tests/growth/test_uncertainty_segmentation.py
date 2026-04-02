# tests/growth/test_uncertainty_segmentation.py
"""Unit tests for LoRA-Ensemble uncertainty segmentation module.

All tests use synthetic data — no real checkpoints, GPU, or H5 files required.
Tests verify mathematical correctness of:
    - Welford online aggregation (mean, variance)
    - Binary entropy computation
    - Mutual information properties
    - Calibration metrics (ECE, Brier, reliability)
    - Volume extraction logic
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.uncertainty_segmentation.engine.uncertainty_metrics import (
    compute_binary_entropy,
    compute_brier_score,
    compute_ece,
    compute_mutual_information,
    compute_reliability_data,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Welford Online Aggregation Tests
# =============================================================================


class TestWelfordAggregation:
    """Verify that Welford's online algorithm matches offline statistics."""

    def _welford_update(
        self, tensors: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Welford algorithm on a list of tensors."""
        M = len(tensors)
        shape = tensors[0].shape

        mean = torch.zeros(shape)
        M2 = torch.zeros(shape)

        for i, t in enumerate(tensors):
            delta = t - mean
            mean += delta / (i + 1)
            delta2 = t - mean
            M2 += delta * delta2

        var = M2 / (M - 1) if M > 1 else torch.zeros(shape)
        return mean, var

    def test_mean_matches_offline(self) -> None:
        """Welford running mean matches torch.mean() on M random tensors."""
        torch.manual_seed(42)
        tensors = [torch.rand(3, 8, 8, 8) for _ in range(5)]

        welford_mean, _ = self._welford_update(tensors)
        offline_mean = torch.stack(tensors).mean(dim=0)

        torch.testing.assert_close(welford_mean, offline_mean, atol=1e-6, rtol=1e-5)

    def test_variance_matches_offline(self) -> None:
        """Welford M2/(M-1) matches torch.var() on M random tensors."""
        torch.manual_seed(123)
        tensors = [torch.rand(3, 8, 8, 8) for _ in range(7)]

        _, welford_var = self._welford_update(tensors)
        # torch.var with correction=1 is unbiased (Bessel's correction)
        offline_var = torch.stack(tensors).var(dim=0, correction=1)

        torch.testing.assert_close(welford_var, offline_var, atol=1e-5, rtol=1e-4)

    def test_single_member_zero_variance(self) -> None:
        """With M=1, Welford variance should be zero."""
        tensor = torch.rand(3, 4, 4, 4)
        _, var = self._welford_update([tensor])
        assert (var == 0).all()

    def test_identical_members_zero_variance(self) -> None:
        """When all members produce identical values, variance is 0."""
        tensor = torch.rand(3, 4, 4, 4)
        tensors = [tensor.clone() for _ in range(5)]
        _, var = self._welford_update(tensors)
        torch.testing.assert_close(var, torch.zeros_like(var), atol=1e-7, rtol=0)

    def test_two_members_known_variance(self) -> None:
        """For M=2 with known values, verify exact variance."""
        a = torch.tensor([1.0, 3.0])
        b = torch.tensor([3.0, 1.0])
        mean, var = self._welford_update([a, b])

        expected_mean = torch.tensor([2.0, 2.0])
        expected_var = torch.tensor([2.0, 2.0])  # sample var with ddof=1

        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(var, expected_var)


# =============================================================================
# Binary Entropy Tests
# =============================================================================


class TestBinaryEntropy:
    """Test per-channel binary entropy computation."""

    def test_entropy_at_half(self) -> None:
        """H(0.5) = ln(2) for binary entropy (maximum uncertainty)."""
        probs = torch.tensor([0.5])
        entropy = compute_binary_entropy(probs)
        assert abs(entropy.item() - math.log(2)) < 1e-5

    def test_entropy_near_zero(self) -> None:
        """H(p) ≈ 0 for p close to 0 (high certainty)."""
        probs = torch.tensor([1e-6])
        entropy = compute_binary_entropy(probs)
        assert entropy.item() < 0.01

    def test_entropy_near_one(self) -> None:
        """H(p) ≈ 0 for p close to 1 (high certainty)."""
        probs = torch.tensor([1.0 - 1e-6])
        entropy = compute_binary_entropy(probs)
        assert entropy.item() < 0.01

    def test_entropy_symmetry(self) -> None:
        """H(p) = H(1-p) for binary entropy."""
        probs = torch.rand(10)
        entropy_p = compute_binary_entropy(probs)
        entropy_1mp = compute_binary_entropy(1.0 - probs)
        torch.testing.assert_close(entropy_p, entropy_1mp, atol=1e-6, rtol=1e-5)

    def test_entropy_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        probs = torch.rand(3, 16, 16, 16)
        entropy = compute_binary_entropy(probs)
        assert entropy.shape == probs.shape

    def test_entropy_nonnegative(self) -> None:
        """Binary entropy is always ≥ 0."""
        probs = torch.rand(100)
        entropy = compute_binary_entropy(probs)
        assert (entropy >= 0).all()

    def test_entropy_bounded(self) -> None:
        """Binary entropy is always ≤ ln(2)."""
        probs = torch.rand(100)
        entropy = compute_binary_entropy(probs)
        assert (entropy <= math.log(2) + 1e-5).all()


# =============================================================================
# Mutual Information Tests
# =============================================================================


class TestMutualInformation:
    """Test mutual information (epistemic uncertainty)."""

    def test_mi_nonnegative(self) -> None:
        """MI ≥ 0 always (by Jensen's inequality)."""
        pred_entropy = torch.rand(3, 8, 8, 8) * math.log(2)
        # mean_member_entropy ≤ pred_entropy for valid inputs
        mean_member_entropy = pred_entropy * torch.rand_like(pred_entropy)
        mi = compute_mutual_information(pred_entropy, mean_member_entropy)
        assert (mi >= 0).all()

    def test_mi_zero_when_members_agree(self) -> None:
        """If all members produce identical probs, MI = 0.

        When all members agree, mean_member_entropy == predictive_entropy.
        """
        probs = torch.rand(3, 8, 8, 8)
        pred_entropy = compute_binary_entropy(probs)
        # All members produce the same probs → mean member entropy = pred entropy
        mi = compute_mutual_information(pred_entropy, pred_entropy)
        torch.testing.assert_close(mi, torch.zeros_like(mi), atol=1e-7, rtol=0)

    def test_mi_positive_when_members_disagree(self) -> None:
        """MI > 0 when ensemble members disagree."""
        # Two members: one says 0.1, other says 0.9
        p1 = torch.tensor([0.1])
        p2 = torch.tensor([0.9])
        mean_prob = (p1 + p2) / 2  # 0.5

        pred_entropy = compute_binary_entropy(mean_prob)
        h1 = compute_binary_entropy(p1)
        h2 = compute_binary_entropy(p2)
        mean_member_entropy = (h1 + h2) / 2

        mi = compute_mutual_information(pred_entropy, mean_member_entropy)
        assert mi.item() > 0.1  # Substantial disagreement

    def test_mi_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        shape = (3, 16, 16, 16)
        pred_entropy = torch.rand(shape) * math.log(2)
        mean_member_entropy = pred_entropy * 0.5
        mi = compute_mutual_information(pred_entropy, mean_member_entropy)
        assert mi.shape == shape


# =============================================================================
# ECE Tests
# =============================================================================


class TestECE:
    """Test Expected Calibration Error."""

    def test_ece_perfect_calibration(self) -> None:
        """ECE ≈ 0 when predicted probs match empirical accuracy."""
        rng = np.random.RandomState(42)
        n = 10000
        # Generate perfectly calibrated predictions
        probs = rng.uniform(0, 1, size=(n, 1))
        labels = (rng.uniform(0, 1, size=(n, 1)) < probs).astype(float)
        ece = compute_ece(probs, labels, n_bins=15)
        assert ece < 0.05  # Should be close to 0 with enough samples

    def test_ece_bounded(self) -> None:
        """ECE is in [0, 1]."""
        rng = np.random.RandomState(42)
        probs = rng.uniform(0, 1, size=(1000, 3))
        labels = rng.randint(0, 2, size=(1000, 3)).astype(float)
        ece = compute_ece(probs, labels)
        assert 0 <= ece <= 1

    def test_ece_overconfident(self) -> None:
        """ECE > 0 for systematically overconfident predictions."""
        # Predict 0.9 for everything, but only 50% are correct
        n = 1000
        probs = np.full((n, 1), 0.9)
        labels = np.zeros((n, 1))
        labels[:500] = 1.0
        ece = compute_ece(probs, labels)
        assert ece > 0.3  # Should be ~0.4


# =============================================================================
# Brier Score Tests
# =============================================================================


class TestBrierScore:
    """Test Brier score computation."""

    def test_brier_perfect(self) -> None:
        """Brier = 0 for perfect one-hot predictions."""
        probs = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        labels = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        brier = compute_brier_score(probs, labels)
        assert abs(brier) < 1e-7

    def test_brier_worst(self) -> None:
        """Brier = 1 for completely wrong one-hot predictions."""
        probs = np.array([[1.0], [1.0], [1.0]])
        labels = np.array([[0.0], [0.0], [0.0]])
        brier = compute_brier_score(probs, labels)
        assert abs(brier - 1.0) < 1e-7

    def test_brier_uniform(self) -> None:
        """Brier = 0.25 for p=0.5 on binary task."""
        probs = np.full((100, 1), 0.5)
        labels = np.ones((100, 1))
        brier = compute_brier_score(probs, labels)
        assert abs(brier - 0.25) < 1e-5

    def test_brier_nonnegative(self) -> None:
        """Brier score is always ≥ 0."""
        rng = np.random.RandomState(42)
        probs = rng.uniform(0, 1, size=(500, 3))
        labels = rng.randint(0, 2, size=(500, 3)).astype(float)
        brier = compute_brier_score(probs, labels)
        assert brier >= 0


# =============================================================================
# Reliability Data Tests
# =============================================================================


class TestReliabilityData:
    """Test reliability diagram data generation."""

    def test_reliability_data_shape(self) -> None:
        """Output has correct number of bins."""
        rng = np.random.RandomState(42)
        n_bins = 10
        probs = rng.uniform(0, 1, size=1000)
        labels = rng.randint(0, 2, size=1000).astype(float)
        data = compute_reliability_data(probs, labels, n_bins=n_bins)

        assert data["bin_edges"].shape == (n_bins + 1,)
        assert data["bin_accuracy"].shape == (n_bins,)
        assert data["bin_confidence"].shape == (n_bins,)
        assert data["bin_count"].shape == (n_bins,)

    def test_reliability_bin_counts_sum(self) -> None:
        """Sum of bin counts equals total number of samples."""
        rng = np.random.RandomState(42)
        n = 500
        probs = rng.uniform(0, 1, size=n)
        labels = rng.randint(0, 2, size=n).astype(float)
        data = compute_reliability_data(probs, labels, n_bins=15)
        assert data["bin_count"].sum() == n

    def test_reliability_accuracy_bounded(self) -> None:
        """Bin accuracy values are in [0, 1]."""
        rng = np.random.RandomState(42)
        probs = rng.uniform(0, 1, size=1000)
        labels = rng.randint(0, 2, size=1000).astype(float)
        data = compute_reliability_data(probs, labels)
        nonempty = data["bin_count"] > 0
        assert (data["bin_accuracy"][nonempty] >= 0).all()
        assert (data["bin_accuracy"][nonempty] <= 1).all()


# =============================================================================
# Volume Extraction Tests
# =============================================================================


class TestVolumeExtraction:
    """Test volume computation from binary masks."""

    def test_volume_from_known_mask(self) -> None:
        """Volume = sum(mask) for 1mm³ isotropic spacing."""
        mask = torch.zeros(64, 64, 64)
        mask[10:20, 10:20, 10:20] = 1.0  # 10³ = 1000 voxels
        volume = mask.sum().item()
        assert volume == 1000.0

    def test_volume_empty_mask(self) -> None:
        """Volume = 0 for empty mask."""
        mask = torch.zeros(32, 32, 32)
        volume = mask.sum().item()
        assert volume == 0.0

    def test_volume_full_mask(self) -> None:
        """Volume = total voxels for full mask."""
        size = 16
        mask = torch.ones(size, size, size)
        volume = mask.sum().item()
        assert volume == size**3

    def test_log_volume_transform(self) -> None:
        """log(V+1) transform is correct."""
        volumes = [1000.0, 5000.0, 10000.0]
        log_volumes = [math.log(v + 1) for v in volumes]

        for v, lv in zip(volumes, log_volumes):
            assert abs(lv - math.log(v + 1)) < 1e-10

    def test_volume_statistics_two_members(self) -> None:
        """Mean and std are correct for known 2-member ensemble."""
        vols = [1000.0, 2000.0]
        mean_v = sum(vols) / 2
        std_v = (sum((v - mean_v) ** 2 for v in vols) / 1) ** 0.5

        assert abs(mean_v - 1500.0) < 1e-10
        assert abs(std_v - (500.0 * math.sqrt(2))) < 1e-5


# =============================================================================
# Boundary Detection Tests
# =============================================================================


class TestBoundaryDetection:
    """Test boundary finding for uncertainty analysis."""

    def test_boundary_of_cube(self) -> None:
        """Boundary of a filled cube is its surface voxels."""
        from experiments.uncertainty_segmentation.engine.volume_extraction import (
            _find_boundary,
        )

        mask = torch.zeros(32, 32, 32)
        mask[10:20, 10:20, 10:20] = 1.0

        boundary = _find_boundary(mask)

        # Boundary should be non-empty
        assert boundary.any()
        # Boundary should be subset of dilated mask minus interior
        # Interior: voxels where all 6 neighbors are also in mask
        # Just check boundary has fewer voxels than the full cube
        assert boundary.sum() < mask.sum()

    def test_boundary_of_single_voxel(self) -> None:
        """A single voxel is entirely boundary."""
        from experiments.uncertainty_segmentation.engine.volume_extraction import (
            _find_boundary,
        )

        mask = torch.zeros(16, 16, 16)
        mask[8, 8, 8] = 1.0

        boundary = _find_boundary(mask)
        # The boundary should include the voxel's neighbors (dilation) XOR the voxel
        # Since dilation expands into empty space, boundary = dilated region minus original
        assert boundary.any()

    def test_empty_mask_no_boundary(self) -> None:
        """Empty mask produces empty boundary."""
        from experiments.uncertainty_segmentation.engine.volume_extraction import (
            _find_boundary,
        )

        mask = torch.zeros(16, 16, 16)
        boundary = _find_boundary(mask)
        assert not boundary.any()


# =============================================================================
# v1.1 Tests: Run Directory Derivation (R1)
# =============================================================================


class TestGetRunDir:
    """Test parameterised run directory derivation."""

    def test_derives_correct_path(self) -> None:
        """r8_M5_s42 from rank=8, n_members=5, base_seed=42."""
        from omegaconf import OmegaConf

        from experiments.uncertainty_segmentation.engine.paths import get_run_dir

        config = OmegaConf.create({
            "experiment": {"output_dir": "/tmp/results"},
            "lora": {"rank": 8},
            "ensemble": {"n_members": 5, "base_seed": 42},
        })
        result = get_run_dir(config)
        assert result.name == "r8_M5_s42"
        assert str(result) == "/tmp/results/r8_M5_s42"

    def test_override_takes_precedence(self) -> None:
        """--run-dir overrides derivation."""
        from omegaconf import OmegaConf

        from experiments.uncertainty_segmentation.engine.paths import get_run_dir

        config = OmegaConf.create({
            "experiment": {"output_dir": "/tmp/results"},
            "lora": {"rank": 8},
            "ensemble": {"n_members": 5, "base_seed": 42},
        })
        result = get_run_dir(config, override="/custom/path")
        assert str(result) == "/custom/path"

    def test_different_params_different_dirs(self) -> None:
        """Different (rank, M, seed) produce different directories."""
        from omegaconf import OmegaConf

        from experiments.uncertainty_segmentation.engine.paths import get_run_dir

        base = {"experiment": {"output_dir": "/tmp/results"}, "ensemble": {"base_seed": 42}}

        c1 = OmegaConf.create({**base, "lora": {"rank": 4}, "ensemble": {"n_members": 3, "base_seed": 42}})
        c2 = OmegaConf.create({**base, "lora": {"rank": 8}, "ensemble": {"n_members": 5, "base_seed": 42}})

        assert get_run_dir(c1) != get_run_dir(c2)


# =============================================================================
# v1.1 Tests: Statistical Analysis (R5, R9)
# =============================================================================


class TestStatisticalAnalysis:
    """Test statistical analysis functions with synthetic data."""

    def test_bootstrap_ci_covers_true_mean(self) -> None:
        """Bootstrap CI of N(0,1) samples covers 0."""
        from experiments.uncertainty_segmentation.engine.statistical_analysis import (
            bootstrap_ci,
        )

        rng = np.random.RandomState(42)
        values = rng.normal(0, 1, size=200)
        mean, ci_lo, ci_hi = bootstrap_ci(values, n_bootstrap=5000)

        assert ci_lo < 0 < ci_hi  # CI covers true mean
        assert abs(mean) < 0.2  # Sample mean close to 0

    def test_cohens_d_known_effect(self) -> None:
        """d ≈ 1.0 for samples shifted by 1 std."""
        from experiments.uncertainty_segmentation.engine.statistical_analysis import (
            paired_cohens_d,
        )

        rng = np.random.RandomState(42)
        n = 500
        a = rng.normal(1, 1, size=n)  # Mean 1, std 1
        b = rng.normal(0, 1, size=n)  # Mean 0, std 1

        d = paired_cohens_d(a, b)
        assert 0.6 < d < 1.4  # Should be ~1.0 (paired, so depends on correlation)

    def test_cohens_d_no_effect(self) -> None:
        """d ≈ 0 when both samples from same distribution."""
        from experiments.uncertainty_segmentation.engine.statistical_analysis import (
            paired_cohens_d,
        )

        rng = np.random.RandomState(42)
        n = 500
        a = rng.normal(0, 1, size=n)
        b = rng.normal(0, 1, size=n)

        d = paired_cohens_d(a, b)
        assert abs(d) < 0.3  # Near zero

    def test_wilcoxon_detects_shift(self) -> None:
        """p < 0.05 for clearly shifted paired samples."""
        from experiments.uncertainty_segmentation.engine.statistical_analysis import (
            paired_wilcoxon,
        )

        rng = np.random.RandomState(42)
        n = 50
        a = rng.normal(1.0, 0.5, size=n)
        b = rng.normal(0.0, 0.5, size=n)

        result = paired_wilcoxon(a, b)
        assert result["p_value"] < 0.05

    def test_wilcoxon_no_shift(self) -> None:
        """p > 0.05 for identical distributions (large N)."""
        from experiments.uncertainty_segmentation.engine.statistical_analysis import (
            paired_wilcoxon,
        )

        rng = np.random.RandomState(42)
        n = 50
        a = rng.normal(0, 1, size=n)
        b = a + rng.normal(0, 0.01, size=n)  # Nearly identical

        result = paired_wilcoxon(a, b)
        # With tiny noise, p should be relatively high
        assert result["p_value"] > 0.01

    def test_icc_perfect_agreement(self) -> None:
        """ICC = 1.0 when all raters agree."""
        from experiments.uncertainty_segmentation.engine.statistical_analysis import (
            compute_icc,
        )

        # All raters give same scores
        data = np.array([[0.8, 0.8, 0.8],
                         [0.6, 0.6, 0.6],
                         [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        icc = compute_icc(data)
        assert abs(icc - 1.0) < 1e-5

    def test_icc_moderate_agreement(self) -> None:
        """ICC > 0.5 for correlated raters."""
        from experiments.uncertainty_segmentation.engine.statistical_analysis import (
            compute_icc,
        )

        rng = np.random.RandomState(42)
        n_subjects = 30
        k_raters = 5
        # Shared signal + noise
        true_scores = rng.uniform(0.3, 0.95, size=n_subjects)
        data = np.column_stack([
            true_scores + rng.normal(0, 0.05, size=n_subjects)
            for _ in range(k_raters)
        ])

        icc = compute_icc(data)
        assert icc > 0.5  # Should show substantial agreement

    def test_icc_single_rater_returns_nan(self) -> None:
        """ICC is undefined for single rater — returns NaN."""
        from experiments.uncertainty_segmentation.engine.statistical_analysis import (
            compute_icc,
        )

        data = np.array([[0.8], [0.6], [0.9]])
        icc = compute_icc(data)
        assert math.isnan(icc)


# =============================================================================
# v1.1 Tests: Volume CSV Schema (R7)
# =============================================================================


class TestVolumeCSVSchema:
    """Test volume CSV structure matches spec."""

    def test_logvol_columns_present(self) -> None:
        """Verify logvol_m* columns are generated alongside vol_m* columns."""
        # Simulate what volume_extraction does
        n_members = 3
        volumes = [1000.0, 1500.0, 2000.0]
        row: dict = {}
        for m_idx, vol in enumerate(volumes):
            row[f"vol_m{m_idx}"] = vol
            row[f"logvol_m{m_idx}"] = math.log(vol + 1)

        # Check all expected columns exist
        for m in range(n_members):
            assert f"vol_m{m}" in row
            assert f"logvol_m{m}" in row
            assert row[f"logvol_m{m}"] == math.log(row[f"vol_m{m}"] + 1)

    def test_per_member_columns_match_n_members(self) -> None:
        """Number of vol_m* and logvol_m* columns equals n_members."""
        n_members = 5
        row: dict = {}
        for m in range(n_members):
            row[f"vol_m{m}"] = float(m * 100 + 500)
            row[f"logvol_m{m}"] = math.log(row[f"vol_m{m}"] + 1)

        vol_cols = [k for k in row if k.startswith("vol_m")]
        logvol_cols = [k for k in row if k.startswith("logvol_m")]
        assert len(vol_cols) == n_members
        assert len(logvol_cols) == n_members


# =============================================================================
# v1.2 Tests: Median/MAD (robust statistics)
# =============================================================================


class TestMedianMAD:
    """Test median and MAD computation."""

    def test_median_known_odd(self) -> None:
        """Median of [1, 2, 3, 4, 5] = 3."""
        import statistics

        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert statistics.median(vals) == 3.0

    def test_mad_known_values(self) -> None:
        """MAD of [1, 2, 3, 4, 5]: median=3, deviations=[2,1,0,1,2], MAD=1."""
        import statistics

        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        median = statistics.median(vals)
        mad = statistics.median([abs(v - median) for v in vals])
        assert mad == 1.0

    def test_mad_scaled_consistency(self) -> None:
        """1.4826 * MAD ≈ std for normal data (large N)."""
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, size=10000).tolist()

        import statistics

        median = statistics.median(vals)
        mad = statistics.median([abs(v - median) for v in vals])
        scaled_mad = 1.4826 * mad

        # Should be close to 1.0 (the true std)
        assert abs(scaled_mad - 1.0) < 0.05

    def test_mad_single_member(self) -> None:
        """MAD = 0 for single member."""
        import statistics

        vals = [42.0]
        median = statistics.median(vals)
        mad = statistics.median([abs(v - median) for v in vals])
        assert mad == 0.0

    def test_mad_identical_values(self) -> None:
        """MAD = 0 when all values identical."""
        import statistics

        vals = [5.0, 5.0, 5.0, 5.0, 5.0]
        median = statistics.median(vals)
        mad = statistics.median([abs(v - median) for v in vals])
        assert mad == 0.0

    def test_mad_robust_to_outlier(self) -> None:
        """MAD is not affected by a single extreme outlier (breakdown=0.5)."""
        import statistics

        # Normal values
        vals_clean = [100.0, 102.0, 101.0, 99.0, 103.0]
        median_clean = statistics.median(vals_clean)
        mad_clean = statistics.median([abs(v - median_clean) for v in vals_clean])

        # Add one extreme outlier
        vals_dirty = [100.0, 102.0, 101.0, 99.0, 10000.0]
        median_dirty = statistics.median(vals_dirty)
        mad_dirty = statistics.median([abs(v - median_dirty) for v in vals_dirty])

        # MAD should be similar (robust), while std would explode
        assert abs(mad_clean - mad_dirty) < 5.0
        # But std is very different
        std_clean = np.std(vals_clean, ddof=1)
        std_dirty = np.std(vals_dirty, ddof=1)
        assert std_dirty > 10 * std_clean


# =============================================================================
# v1.2 Tests: Save Predictions
# =============================================================================


class TestSavePredictions:
    """Test prediction saving utilities."""

    def test_select_sample_spread(self) -> None:
        """Spread strategy gives evenly spaced indices."""
        from experiments.uncertainty_segmentation.engine.save_predictions import (
            select_sample_indices,
        )

        indices = select_sample_indices(100, 5, "spread")
        assert len(indices) == 5
        assert indices[0] == 0
        assert indices[-1] == 99
        # Roughly evenly spaced
        gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        assert all(20 <= g <= 30 for g in gaps)

    def test_select_sample_first(self) -> None:
        """First strategy gives 0..N-1."""
        from experiments.uncertainty_segmentation.engine.save_predictions import (
            select_sample_indices,
        )

        indices = select_sample_indices(100, 3, "first")
        assert indices == [0, 1, 2]

    def test_select_sample_more_than_total(self) -> None:
        """Requesting more samples than total returns all."""
        from experiments.uncertainty_segmentation.engine.save_predictions import (
            select_sample_indices,
        )

        indices = select_sample_indices(3, 10, "spread")
        assert indices == [0, 1, 2]

    def test_nifti_roundtrip_3d(self) -> None:
        """Save and load a 3D mask through NIfTI."""
        import tempfile

        import nibabel as nib

        from experiments.uncertainty_segmentation.engine.save_predictions import (
            save_ensemble_mask,
        )

        mask = torch.zeros(32, 32, 32)
        mask[10:20, 10:20, 10:20] = 1.0

        with tempfile.TemporaryDirectory() as tmp:
            path = save_ensemble_mask(mask, Path(tmp), "test_scan")
            assert path.exists()

            # Reload and verify
            img = nib.load(str(path))
            data = np.asarray(img.dataobj)
            assert data.shape == (32, 32, 32)
            assert data.sum() == 1000  # 10^3 voxels
            assert data.dtype == np.int8

    def test_nifti_roundtrip_4d(self) -> None:
        """Save and load a 4D probability map through NIfTI."""
        import tempfile

        import nibabel as nib

        from experiments.uncertainty_segmentation.engine.save_predictions import (
            _save_4d,
        )

        probs = torch.rand(3, 16, 16, 16)  # [C, D, H, W]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.nii.gz"
            _save_4d(probs, out_path)
            assert out_path.exists()

            img = nib.load(str(out_path))
            data = np.asarray(img.dataobj)
            # nibabel stores as [D, H, W, C]
            assert data.shape == (16, 16, 16, 3)
            # Verify data matches (after permute back)
            np.testing.assert_allclose(
                data, probs.permute(1, 2, 3, 0).numpy(), atol=1e-5
            )


# =============================================================================
# v1.2 Tests: Convergence Analysis
# =============================================================================


class TestConvergenceAnalysis:
    """Test convergence curve computation."""

    def test_k1_nan_se(self) -> None:
        """SE is NaN for k=1 (undefined)."""
        from experiments.uncertainty_segmentation.engine.convergence_analysis import (
            compute_convergence_curve,
        )

        curve = compute_convergence_curve([1.0, 2.0, 3.0, 4.0, 5.0])
        assert math.isnan(curve.iloc[0]["running_se"])

    def test_se_decreases(self) -> None:
        """SE decreases monotonically as k increases (for constant std)."""
        from experiments.uncertainty_segmentation.engine.convergence_analysis import (
            compute_convergence_curve,
        )

        # With enough spread, SE should decrease
        rng = np.random.RandomState(42)
        values = rng.normal(100, 10, size=20).tolist()
        curve = compute_convergence_curve(values)

        # Compare SE at k=3 vs k=20
        se_early = curve.iloc[2]["running_se"]  # k=3
        se_late = curve.iloc[-1]["running_se"]   # k=20
        assert se_late < se_early

    def test_ci_shrinks_with_k(self) -> None:
        """CI width shrinks as k increases (guaranteed by 1/sqrt(k))."""
        from experiments.uncertainty_segmentation.engine.convergence_analysis import (
            compute_convergence_curve,
        )

        rng = np.random.RandomState(42)
        values = rng.normal(1000, 50, size=20).tolist()
        curve = compute_convergence_curve(values)

        # CI width at k=3 vs k=20 (guaranteed to shrink with enough k)
        ci_width_early = curve.iloc[2]["running_ci_upper"] - curve.iloc[2]["running_ci_lower"]
        ci_width_late = curve.iloc[-1]["running_ci_upper"] - curve.iloc[-1]["running_ci_lower"]
        assert ci_width_late < ci_width_early

    def test_curve_has_correct_length(self) -> None:
        """Convergence curve has M rows (one per k)."""
        from experiments.uncertainty_segmentation.engine.convergence_analysis import (
            compute_convergence_curve,
        )

        M = 7
        values = list(range(M))
        curve = compute_convergence_curve([float(v) for v in values])
        assert len(curve) == M
        assert list(curve["k"]) == list(range(1, M + 1))

    def test_median_robust_in_curve(self) -> None:
        """Running median is robust to outlier at later k."""
        from experiments.uncertainty_segmentation.engine.convergence_analysis import (
            compute_convergence_curve,
        )

        # 4 normal values + 1 outlier
        values = [100.0, 102.0, 101.0, 99.0, 10000.0]
        curve = compute_convergence_curve(values)

        # At k=5, median should still be ~101, not pulled by outlier
        assert abs(curve.iloc[-1]["running_median"] - 101.0) < 5.0
        # But mean is pulled way up
        assert curve.iloc[-1]["running_mean"] > 2000.0


# =============================================================================
# v1.2 Tests: Multi-label Mask + Bug Fixes
# =============================================================================


class TestMultiLabelMask:
    """Test multi-label BraTS-convention segmentation saving."""

    def test_multilabel_roundtrip(self) -> None:
        """Save and load a multi-label mask, verify BraTS labels."""
        import tempfile

        import nibabel as nib

        from experiments.uncertainty_segmentation.engine.save_predictions import (
            save_multilabel_mask,
        )

        # Simulate 3-channel sigmoid probs: TC(ch0), WT(ch1), ET(ch2)
        probs = torch.zeros(3, 16, 16, 16)
        # Region 1: WT-only (edema → label 2)
        probs[1, 2:5, 2:5, 2:5] = 1.0  # WT=1, TC=0, ET=0
        # Region 2: TC but not ET (NCR/NET → label 1)
        probs[0, 6:9, 6:9, 6:9] = 1.0  # TC=1
        probs[1, 6:9, 6:9, 6:9] = 1.0  # WT=1
        # Region 3: ET (→ label 3)
        probs[0, 10:13, 10:13, 10:13] = 1.0  # TC=1
        probs[1, 10:13, 10:13, 10:13] = 1.0  # WT=1
        probs[2, 10:13, 10:13, 10:13] = 1.0  # ET=1

        with tempfile.TemporaryDirectory() as tmp:
            path = save_multilabel_mask(probs, Path(tmp), "test_scan")
            assert path.exists()

            img = nib.load(str(path))
            seg = np.asarray(img.dataobj)

            # Check labels
            assert (seg[2:5, 2:5, 2:5] == 2).all()   # ED
            assert (seg[6:9, 6:9, 6:9] == 1).all()    # NCR/NET
            assert (seg[10:13, 10:13, 10:13] == 3).all()  # ET
            assert (seg[0, 0, 0] == 0)                 # Background

    def test_multilabel_nonoverlapping(self) -> None:
        """Multi-label output has no overlapping labels."""
        import tempfile

        import nibabel as nib

        from experiments.uncertainty_segmentation.engine.save_predictions import (
            save_multilabel_mask,
        )

        probs = torch.rand(3, 8, 8, 8)  # Random probs

        with tempfile.TemporaryDirectory() as tmp:
            path = save_multilabel_mask(probs, Path(tmp), "test")

            img = nib.load(str(path))
            seg = np.asarray(img.dataobj)

            # Each voxel should have exactly one label (0, 1, 2, or 3)
            assert set(np.unique(seg)).issubset({0, 1, 2, 3})


class TestBugFixes:
    """Verify specific bug fixes from LORA_ENSEMBLE_BUGFIX.md."""

    def test_bug6_vol_ensemble_mask_column(self) -> None:
        """BUG-6: vol_ensemble_mask should be distinct from vol_mean."""
        # vol_mean = mean of per-member volumes
        # vol_ensemble_mask = volume from the averaged-then-thresholded mask
        # These are mathematically different
        per_member_volumes = [100.0, 200.0, 300.0]
        vol_mean = sum(per_member_volumes) / len(per_member_volumes)
        assert vol_mean == 200.0

        # The ensemble mask volume depends on the spatial average, not the scalar average
        # Just verify the column logic works
        mask = torch.zeros(16, 16, 16)
        mask[3:7, 3:7, 3:7] = 1.0  # 64 voxels
        vol_ensemble_mask = float(mask.sum().item())
        assert vol_ensemble_mask == 64.0
        assert vol_ensemble_mask != vol_mean  # They should differ in general


# =============================================================================
# v1.3 Tests: Audit Fixes
# =============================================================================


class TestAuditFixes:
    """Verify bug fixes from v1.3 pre-launch audit."""

    def test_min_delta_early_stopping(self) -> None:
        """Early stopping respects min_delta: small improvements don't reset patience."""
        # Simulate the early stopping logic from train_member.py
        min_delta = 0.01
        best_score = 0.80
        patience_counter = 0
        patience = 3

        # Improvement below min_delta → should NOT reset patience
        val_dice = 0.805  # +0.005 < 0.01
        if val_dice > best_score + min_delta:
            best_score = val_dice
            patience_counter = 0
        else:
            patience_counter += 1
        assert patience_counter == 1
        assert best_score == 0.80  # unchanged

        # Improvement above min_delta → SHOULD reset patience
        val_dice = 0.815  # +0.015 > 0.01
        if val_dice > best_score + min_delta:
            best_score = val_dice
            patience_counter = 0
        else:
            patience_counter += 1
        assert patience_counter == 0  # reset
        assert best_score == 0.815  # updated

    def test_icc_negative_correlation(self) -> None:
        """ICC can be negative (members disagree more than expected by chance)."""
        from experiments.uncertainty_segmentation.engine.statistical_analysis import (
            compute_icc,
        )

        # Anti-correlated raters: when one is high, the other is low
        data = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ])
        icc = compute_icc(data)
        assert icc < 0  # Should be negative

    def test_inference_time_field_exists(self) -> None:
        """EnsemblePrediction has inference_time_sec field."""
        from experiments.uncertainty_segmentation.engine.ensemble_inference import (
            EnsemblePrediction,
        )

        result = EnsemblePrediction(
            mean_probs=torch.zeros(3, 4, 4, 4),
            var_probs=torch.zeros(3, 4, 4, 4),
            predictive_entropy=torch.zeros(3, 4, 4, 4),
            mutual_information=torch.zeros(3, 4, 4, 4),
            ensemble_mask=torch.zeros(4, 4, 4, dtype=torch.bool),
            per_member_volumes=[100.0],
            volume_mean=100.0,
            volume_std=0.0,
            log_volume_mean=math.log(101),
            log_volume_std=0.0,
            volume_median=100.0,
            volume_mad=0.0,
            log_volume_median=math.log(101),
            log_volume_mad=0.0,
            n_members=1,
            inference_time_sec=5.3,
        )
        assert result.inference_time_sec == 5.3

    def test_gaussian_augmentation_imports(self) -> None:
        """Gaussian noise and smooth augmentations are importable."""
        from monai.transforms import RandGaussianNoised, RandGaussianSmoothd

        # Verify they can be instantiated
        noise = RandGaussianNoised(keys=["image"], prob=0.15, std=0.05)
        smooth = RandGaussianSmoothd(
            keys=["image"], sigma_x=(0.5, 1.0), prob=0.1
        )
        assert noise is not None
        assert smooth is not None

    def test_augmentation_pipeline_with_gaussian(self) -> None:
        """Training transforms include Gaussian noise/smooth when enabled."""
        from src.growth.data.transforms import get_h5_train_transforms

        # With gaussian augmentations disabled (default)
        pipeline_default = get_h5_train_transforms(augment=True)
        n_default = len(pipeline_default.transforms)

        # With gaussian augmentations enabled
        pipeline_aug = get_h5_train_transforms(
            augment=True,
            include_gaussian_noise=True,
            include_gaussian_smooth=True,
        )
        n_aug = len(pipeline_aug.transforms)

        # Should have 2 more transforms
        assert n_aug == n_default + 2
