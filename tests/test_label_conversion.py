#!/usr/bin/env python
"""Unit tests for domain-aware label conversion.

Tests the _convert_single_domain function and the domain parameter in
SegmentationLoss3Ch and DiceMetric3Ch for GLI, MEN, and mixed batches.
"""

import pytest
import torch

from growth.losses.segmentation import (
    DiceMetric3Ch,
    SegmentationLoss3Ch,
    _convert_single_domain,
)

pytestmark = [pytest.mark.phase0, pytest.mark.unit]

# ---------------------------------------------------------------------------
# _convert_single_domain
# ---------------------------------------------------------------------------


class TestConvertSingleDomain:
    """Tests for the module-level _convert_single_domain function."""

    def test_gli_labels(self) -> None:
        """GLI: 1=NETC, 2=SNFH, 3=ET, 4=RC (BraTS-hierarchical target).

        TC = (1|3|4), WT = (seg>0), ET = (3)
        """
        seg = torch.tensor([[[[0, 1, 2, 3, 4]]]])  # [1, 1, 1, 5]
        seg = seg.squeeze(1)  # [1, 1, 5]
        result = _convert_single_domain(seg, "GLI")  # [1, 3, 1, 5]

        tc = result[0, 0, 0]
        wt = result[0, 1, 0]
        et = result[0, 2, 0]

        # bg=0: all zero
        assert tc[0] == 0 and wt[0] == 0 and et[0] == 0
        # NETC=1: TC=1, WT=1, ET=0
        assert tc[1] == 1 and wt[1] == 1 and et[1] == 0
        # SNFH=2: TC=0, WT=1 (WT includes edema), ET=0
        assert tc[2] == 0 and wt[2] == 1 and et[2] == 0
        # ET=3: TC=1, WT=1, ET=1
        assert tc[3] == 1 and wt[3] == 1 and et[3] == 1
        # RC=4: TC=1, WT=1, ET=0
        assert tc[4] == 1 and wt[4] == 1 and et[4] == 0

    def test_men_labels(self) -> None:
        """MEN: 1=NETC, 2=SNFH, 3=ET (BraTS-hierarchical target).

            TC = (1|3), WT = (seg>0), ET = (3)

        Downstream clinical regions (``WT_meningioma ⊥ ED_edema``) are
        derived from these hierarchical channels via
        ``growth.inference.postprocess.derive_disjoint_regions``, not
        by altering the training target.
        """
        seg = torch.tensor([[[[0, 1, 2, 3]]]])  # [1, 1, 1, 4]
        seg = seg.squeeze(1)  # [1, 1, 4]
        result = _convert_single_domain(seg, "MEN")  # [1, 3, 1, 4]

        tc = result[0, 0, 0]
        wt = result[0, 1, 0]
        et = result[0, 2, 0]

        # bg=0: all zero
        assert tc[0] == 0 and wt[0] == 0 and et[0] == 0
        # NETC=1: TC=1, WT=1, ET=0
        assert tc[1] == 1 and wt[1] == 1 and et[1] == 0
        # SNFH=2: TC=0, WT=1 (includes edema), ET=0
        assert tc[2] == 0 and wt[2] == 1 and et[2] == 0
        # ET=3: TC=1, WT=1, ET=1
        assert tc[3] == 1 and wt[3] == 1 and et[3] == 1

        # BraTS-hierarchical invariants: ET ⊂ TC ⊂ WT.
        assert ((et.bool() & tc.bool()) == et.bool()).all()
        assert ((tc.bool() & wt.bool()) == tc.bool()).all()

    def test_all_background(self) -> None:
        """All background labels should produce all-zero masks."""
        seg = torch.zeros(2, 4, 4, 4, dtype=torch.long)
        for domain in ("MEN", "GLI"):
            result = _convert_single_domain(seg, domain)
            assert result.sum() == 0

    def test_output_shape(self) -> None:
        """Output shape should be [B, 3, D, H, W]."""
        seg = torch.randint(0, 4, (3, 8, 8, 8))
        result = _convert_single_domain(seg, "MEN")
        assert result.shape == (3, 3, 8, 8, 8)


# ---------------------------------------------------------------------------
# SegmentationLoss3Ch with domain
# ---------------------------------------------------------------------------


class TestSegmentationLoss3ChDomain:
    """Tests for domain-aware SegmentationLoss3Ch."""

    def setup_method(self) -> None:
        self.loss_fn = SegmentationLoss3Ch()

    def test_gli_default_backward_compat(self) -> None:
        """domain=None should use GLI conversion (backward compatible)."""
        pred = torch.randn(2, 3, 8, 8, 8)
        seg = torch.randint(0, 4, (2, 1, 8, 8, 8))
        loss = self.loss_fn(pred, seg)
        assert torch.isfinite(loss)

    def test_men_single_domain(self) -> None:
        """domain='MEN' should produce finite loss."""
        pred = torch.randn(2, 3, 8, 8, 8)
        seg = torch.randint(0, 4, (2, 1, 8, 8, 8))
        loss = self.loss_fn(pred, seg, domain="MEN")
        assert torch.isfinite(loss)

    def test_mixed_batch(self) -> None:
        """Mixed batch with per-sample domains should produce finite loss."""
        pred = torch.randn(4, 3, 8, 8, 8)
        seg = torch.randint(0, 4, (4, 1, 8, 8, 8))
        domains = ["MEN", "GLI", "MEN", "GLI"]
        loss = self.loss_fn(pred, seg, domain=domains)
        assert torch.isfinite(loss)

    def test_gli_with_label4(self) -> None:
        """GLI label 4 (RC) should not cause errors."""
        pred = torch.randn(1, 3, 8, 8, 8)
        seg = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
        seg[0, 0, :2, :2, :2] = 4  # Resection cavity
        loss = self.loss_fn(pred, seg, domain="GLI")
        assert torch.isfinite(loss)

    def test_all_background_loss_finite(self) -> None:
        """All-background segs should produce finite loss."""
        pred = torch.randn(2, 3, 8, 8, 8)
        seg = torch.zeros(2, 1, 8, 8, 8, dtype=torch.long)
        for domain in ("MEN", "GLI", None):
            loss = self.loss_fn(pred, seg, domain=domain)
            assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# DiceMetric3Ch with domain
# ---------------------------------------------------------------------------


class TestDiceMetric3ChDomain:
    """Tests for domain-aware DiceMetric3Ch."""

    def setup_method(self) -> None:
        self.metric = DiceMetric3Ch()

    def test_perfect_men_prediction(self) -> None:
        """Perfect prediction should give Dice=1.0 for MEN."""
        # Create a segmentation and matching prediction
        seg = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
        seg[0, 0, 2:6, 2:6, 2:6] = 1  # ET
        seg[0, 0, 1:7, 1:7, 1:7] = torch.where(
            seg[0, 0, 1:7, 1:7, 1:7] == 0,
            torch.tensor(2),  # NET
            seg[0, 0, 1:7, 1:7, 1:7],
        )

        # Convert to get target masks
        target_3ch = _convert_single_domain(seg.squeeze(1), "MEN")

        # Create perfect logits (large positive where mask=1, large negative where mask=0)
        pred = target_3ch * 20.0 - 10.0  # sigmoid(10)≈1, sigmoid(-10)≈0

        dice = self.metric(pred, seg, domain="MEN")
        assert dice.shape == (1, 3)
        # All channels should be ~1.0
        assert dice.min() > 0.95

    def test_mixed_batch_dice(self) -> None:
        """Mixed batch dice computation should work."""
        pred = torch.randn(4, 3, 8, 8, 8)
        seg = torch.randint(0, 4, (4, 1, 8, 8, 8))
        domains = ["MEN", "GLI", "GLI", "MEN"]
        dice = self.metric(pred, seg, domain=domains)
        assert dice.shape == (4, 3)
        assert torch.all(dice >= 0) and torch.all(dice <= 1)

    def test_backward_compat(self) -> None:
        """domain=None should work (GLI default)."""
        pred = torch.randn(2, 3, 8, 8, 8)
        seg = torch.randint(0, 4, (2, 1, 8, 8, 8))
        dice = self.metric(pred, seg)
        assert dice.shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
