"""Unit tests for the brain-mask helpers and brain-mask-aware TSI."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.uncertainty_segmentation.explainability.engine.brain_mask import (
    brain_mask_coverage,
    derive_brain_mask,
)
from experiments.uncertainty_segmentation.explainability.engine.tsi import (
    compute_tsi_single_stage,
)

pytestmark = [pytest.mark.experiment, pytest.mark.unit]


# ---------------------------------------------------------------------------
# derive_brain_mask
# ---------------------------------------------------------------------------


class TestDeriveBrainMask:

    def test_strict_zero_rule(self) -> None:
        """Zero voxels (skull-stripped background) become mask = 0."""
        img = torch.zeros(4, 8, 8, 8)
        img[:, 2:6, 2:6, 2:6] = 1.5
        mask = derive_brain_mask(img)
        assert mask.shape == (8, 8, 8)
        assert mask[2:6, 2:6, 2:6].min().item() == 1.0
        assert mask.sum().item() == 4 ** 3

    def test_union_across_modalities(self) -> None:
        """The mask is the union of non-zero voxels across all modalities."""
        img = torch.zeros(4, 4, 4, 4)
        # Each modality contributes a disjoint corner.
        img[0, 0, 0, 0] = 1.0
        img[1, 0, 0, 1] = 1.0
        img[2, 0, 1, 0] = 1.0
        img[3, 1, 0, 0] = 1.0
        mask = derive_brain_mask(img)
        assert mask.sum().item() == 4

    def test_threshold_fallback_uses_first_channel_only(self) -> None:
        img = torch.zeros(4, 4, 4, 4)
        img[0] = 0.05
        img[1:] = 1.0
        mask_strict = derive_brain_mask(img)
        mask_thresh = derive_brain_mask(img, threshold=0.1)
        assert mask_strict.sum().item() == 4 ** 3
        # First channel below threshold ⇒ no voxel in mask.
        assert mask_thresh.sum().item() == 0

    def test_5d_input_with_singleton_batch(self) -> None:
        img = torch.zeros(1, 4, 4, 4, 4)
        img[0, 0] = 1.0
        mask = derive_brain_mask(img)
        assert mask.shape == (4, 4, 4)

    def test_5d_input_with_batch_rejected(self) -> None:
        img = torch.zeros(2, 4, 4, 4, 4)
        with pytest.raises(ValueError, match="single scan"):
            derive_brain_mask(img)


class TestBrainMaskCoverage:

    def test_all_zeros(self) -> None:
        assert brain_mask_coverage(torch.zeros(8, 8, 8)) == 0.0

    def test_all_ones(self) -> None:
        assert brain_mask_coverage(torch.ones(8, 8, 8)) == 1.0

    def test_half_brain(self) -> None:
        m = torch.zeros(2, 4, 4)
        m[0] = 1.0
        assert brain_mask_coverage(m) == 0.5


# ---------------------------------------------------------------------------
# Brain-masked TSI: parity with legacy when brain_mask = ones
# ---------------------------------------------------------------------------


class TestBrainMaskedTSI:

    def test_parity_with_legacy_when_full_brain(self) -> None:
        """``brain_mask = ones`` ⇒ identical to ``brain_mask=None``."""
        torch.manual_seed(0)
        h = torch.randn(1, 5, 8, 8, 8).abs()
        mask = torch.zeros(8, 8, 8)
        mask[:4] = 1.0
        brain = torch.ones(8, 8, 8)
        out_legacy = compute_tsi_single_stage(h, mask, thresholds=[1.5])
        out_brain = compute_tsi_single_stage(
            h, mask, thresholds=[1.5], brain_mask_ds=brain,
        )
        np.testing.assert_allclose(
            out_brain.tsi_per_channel,
            out_legacy.tsi_per_channel,
            rtol=1e-6,
        )

    def test_brain_mask_excludes_background(self) -> None:
        """When the original (legacy) denominator counted background voxels with
        small activation, the brain-masked TSI is NOT lower (the background
        was diluting the per-voxel mean *down*, so removing it can either
        increase or decrease the denominator)."""
        torch.manual_seed(1)
        c = 3
        h = torch.zeros(1, c, 8, 8, 8)
        # Tumor voxels: high activation.
        h[..., :4, :, :] = 5.0
        # Brain (non-tumor) voxels: medium activation.
        h[..., 4:6, :, :] = 1.0
        # "Skull / air" voxels: zero activation.
        # (already zero by default)

        mask = torch.zeros(8, 8, 8)
        mask[:4] = 1.0
        brain = torch.zeros(8, 8, 8)
        brain[:6] = 1.0  # rows 0-5 are brain, rows 6-7 are skull/air

        legacy = compute_tsi_single_stage(h, mask, thresholds=[1.5])
        brain_tsi = compute_tsi_single_stage(
            h, mask, thresholds=[1.5], brain_mask_ds=brain,
        )
        # Legacy denom = mean over rows 4-7 = mean of [1,1,0,0]/4 = 0.5.
        # Brain denom  = mean over rows 4-5 = mean of [1,1]/2  = 1.0.
        # → legacy TSI = 5.0 / 0.5 = 10.0; brain TSI = 5.0 / 1.0 = 5.0.
        np.testing.assert_allclose(legacy.tsi_per_channel, 10.0, atol=1e-5)
        np.testing.assert_allclose(brain_tsi.tsi_per_channel, 5.0, atol=1e-5)
        # Brain-masked TSI is LOWER here because background dilution had
        # *inflated* the legacy ratio.
        assert brain_tsi.mean_tsi < legacy.mean_tsi
