"""Unit tests for TSI (Tumor Selectivity Index) analysis.

Tests core TSI computation on synthetic data without requiring
real models or GPU.
"""

import numpy as np
import pytest
import scipy.stats
import torch

from experiments.uncertainty_segmentation.explainability.engine.tsi import (
    STAGE_META,
    ScanTSIResult,
    TSIResult,
    compute_tsi_single_scan,
    compute_tsi_single_stage,
    downsample_mask,
)

pytestmark = [pytest.mark.unit, pytest.mark.experiment]


class TestDownsampleMask:
    """Tests for nearest-neighbor mask downsampling."""

    def test_preserves_binary(self) -> None:
        """Downsampled mask should only contain 0s and 1s."""
        mask = torch.zeros(64, 64, 64)
        mask[20:40, 20:40, 20:40] = 1.0
        ds = downsample_mask(mask, (16, 16, 16))
        unique = torch.unique(ds)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_output_shape(self) -> None:
        """Output should match target spatial dims."""
        mask = torch.ones(96, 96, 96)
        ds = downsample_mask(mask, (12, 12, 12))
        assert ds.shape == (12, 12, 12)

    def test_all_ones_stays_all_ones(self) -> None:
        """All-ones mask should stay all-ones after downsampling."""
        mask = torch.ones(48, 48, 48)
        ds = downsample_mask(mask, (6, 6, 6))
        assert ds.sum().item() == 6 * 6 * 6

    def test_all_zeros_stays_all_zeros(self) -> None:
        """All-zeros mask should stay all-zeros after downsampling."""
        mask = torch.zeros(48, 48, 48)
        ds = downsample_mask(mask, (6, 6, 6))
        assert ds.sum().item() == 0


class TestTSIConstant:
    """Tests for TSI computation edge cases."""

    def test_constant_activation_gives_tsi_one(self) -> None:
        """TSI of spatially uniform activation should be 1.0 for all channels."""
        h = torch.ones(1, 10, 8, 8, 8)
        mask = torch.zeros(8, 8, 8)
        mask[:4, :, :] = 1.0  # Half tumor, half non-tumor

        result = compute_tsi_single_stage(h, mask, thresholds=[1.5, 2.0])

        np.testing.assert_allclose(result.tsi_per_channel, 1.0, atol=1e-6)
        assert abs(result.mean_tsi - 1.0) < 1e-6

    def test_tumor_selective_activation_gives_high_tsi(self) -> None:
        """Activation concentrated inside tumor should give TSI >> 1."""
        h = torch.zeros(1, 5, 8, 8, 8)
        mask = torch.zeros(8, 8, 8)
        mask[:4, :, :] = 1.0

        # Make activation strong inside tumor, weak outside
        h[:, :, :4, :, :] = 10.0
        h[:, :, 4:, :, :] = 1.0

        result = compute_tsi_single_stage(h, mask, thresholds=[1.5, 2.0])

        assert result.mean_tsi > 5.0
        assert result.frac_above[1.5] == 1.0
        assert result.frac_above[2.0] == 1.0

    def test_tumor_suppressed_gives_low_tsi(self) -> None:
        """Activation concentrated outside tumor should give TSI < 1."""
        h = torch.zeros(1, 5, 8, 8, 8)
        mask = torch.zeros(8, 8, 8)
        mask[:4, :, :] = 1.0

        # Activation weak inside tumor, strong outside
        h[:, :, :4, :, :] = 1.0
        h[:, :, 4:, :, :] = 10.0

        result = compute_tsi_single_stage(h, mask, thresholds=[1.5, 2.0])

        assert result.mean_tsi < 0.5
        assert result.frac_above[1.5] == 0.0


class TestWilcoxon:
    """Tests for Wilcoxon signed-rank test in TSI."""

    def test_all_ones_gives_high_pvalue(self) -> None:
        """TSI = 1.0 for all channels should give p ≈ 1.0 (no evidence against H0)."""
        h = torch.ones(1, 20, 8, 8, 8)
        mask = torch.zeros(8, 8, 8)
        mask[:4, :, :] = 1.0

        result = compute_tsi_single_stage(h, mask, thresholds=[1.5])

        # p should be high (no evidence against H0: median = 1)
        assert result.wilcoxon_p >= 0.5

    def test_high_tsi_gives_low_pvalue(self) -> None:
        """If most channels are tumor-selective, Wilcoxon should reject H0."""
        rng = np.random.RandomState(42)
        C = 50

        h = torch.zeros(1, C, 16, 16, 16)
        mask = torch.zeros(16, 16, 16)
        mask[:8, :, :] = 1.0

        # Make channels tumor-selective with varying strength
        for c in range(C):
            strength = 2.0 + rng.rand() * 3.0
            h[0, c, :8, :, :] = strength
            h[0, c, 8:, :, :] = 1.0

        result = compute_tsi_single_stage(h, mask, thresholds=[1.5])

        assert result.wilcoxon_p < 0.01


class TestTopKChannels:
    """Tests for top-K channel identification."""

    def test_top_k_are_highest_tsi(self) -> None:
        """Top-K channels should have the highest TSI values."""
        C = 20
        h = torch.ones(1, C, 8, 8, 8)
        mask = torch.zeros(8, 8, 8)
        mask[:4, :, :] = 1.0

        # Make channels 17, 18, 19 highly tumor-selective
        h[0, 17, :4, :, :] = 5.0
        h[0, 18, :4, :, :] = 8.0
        h[0, 19, :4, :, :] = 10.0

        result = compute_tsi_single_stage(h, mask, thresholds=[1.5], top_k=3)

        assert set(result.top_k_channels) == {17, 18, 19}


class TestEmptyMask:
    """Tests for edge cases with empty masks."""

    def test_no_tumor_voxels_returns_nan(self) -> None:
        """If the mask has no tumor voxels, TSI should be NaN."""
        h = torch.ones(1, 10, 8, 8, 8)
        mask = torch.zeros(8, 8, 8)  # No tumor

        result = compute_tsi_single_stage(h, mask, thresholds=[1.5])

        assert np.all(np.isnan(result.tsi_per_channel))
        assert np.isnan(result.mean_tsi)

    def test_all_tumor_voxels_returns_nan(self) -> None:
        """If the mask is all tumor, TSI should be NaN (no non-tumor reference)."""
        h = torch.ones(1, 10, 8, 8, 8)
        mask = torch.ones(8, 8, 8)  # All tumor

        result = compute_tsi_single_stage(h, mask, thresholds=[1.5])

        assert np.all(np.isnan(result.tsi_per_channel))


class TestComputeSingleScan:
    """Tests for full scan-level TSI computation."""

    def test_returns_five_stages(self) -> None:
        """Should return TSI results for all 5 stages."""
        # Synthetic hidden states matching the architecture
        hidden_states = [
            torch.randn(1, 48, 48, 48, 48),
            torch.randn(1, 96, 24, 24, 24),
            torch.randn(1, 192, 12, 12, 12),
            torch.randn(1, 384, 6, 6, 6),
            torch.randn(1, 768, 3, 3, 3),
        ]
        gt_mask = torch.zeros(96, 96, 96)
        gt_mask[30:60, 30:60, 30:60] = 1.0

        result = compute_tsi_single_scan(
            hidden_states, gt_mask,
            scan_id="test_scan",
            condition="frozen",
        )

        assert len(result.stages) == 5
        assert result.scan_id == "test_scan"
        assert result.condition == "frozen"
        for s in range(5):
            assert result.stages[s].stage == s
            assert result.stages[s].n_channels == STAGE_META[s]["channels"]

    def test_return_maps_only_when_requested(self) -> None:
        """Activation maps should only be stored when return_maps=True."""
        hidden_states = [
            torch.randn(1, 48, 48, 48, 48),
            torch.randn(1, 96, 24, 24, 24),
            torch.randn(1, 192, 12, 12, 12),
            torch.randn(1, 384, 6, 6, 6),
            torch.randn(1, 768, 3, 3, 3),
        ]
        gt_mask = torch.zeros(96, 96, 96)
        gt_mask[30:60, 30:60, 30:60] = 1.0

        result_no_maps = compute_tsi_single_scan(
            hidden_states, gt_mask, "scan", "frozen", return_maps=False,
        )
        result_maps = compute_tsi_single_scan(
            hidden_states, gt_mask, "scan", "frozen", return_maps=True,
        )

        for sr in result_no_maps.stages:
            assert sr.mean_activation_map is None
            assert sr.top_channels_map is None

        for sr in result_maps.stages:
            assert sr.mean_activation_map is not None
            assert sr.top_channels_map is not None


class TestFracAboveThresholds:
    """Tests for fraction-above-threshold computation."""

    def test_mixed_channels(self) -> None:
        """Frac above threshold should be correct for a mix of selective/general."""
        C = 10
        h = torch.ones(1, C, 8, 8, 8)
        mask = torch.zeros(8, 8, 8)
        mask[:4, :, :] = 1.0

        # Make 3 of 10 channels highly selective (TSI >> 1.5)
        for c in range(3):
            h[0, c, :4, :, :] = 5.0

        result = compute_tsi_single_stage(h, mask, thresholds=[1.5, 2.0])

        assert abs(result.frac_above[1.5] - 0.3) < 0.01  # 3/10
        assert abs(result.frac_above[2.0] - 0.3) < 0.01
