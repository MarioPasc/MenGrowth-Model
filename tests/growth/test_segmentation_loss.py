# tests/growth/test_segmentation_loss.py
"""Tests for segmentation loss functions."""

import pytest
import torch

from growth.losses.segmentation import (
    DeepSupervisionLoss,
    DiceMetric,
    SegmentationLoss,
    create_segmentation_loss,
)

pytestmark = [pytest.mark.phase1, pytest.mark.unit]


class TestSegmentationLoss:
    """Tests for SegmentationLoss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = SegmentationLoss()
        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.randint(0, 3, (2, 1, 16, 16, 16))

        loss = loss_fn(pred, target)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_loss_decreases_for_correct_predictions(self):
        """Test loss is lower when predictions match targets."""
        loss_fn = SegmentationLoss()

        # Create target
        target = torch.randint(0, 3, (1, 1, 16, 16, 16))

        # Random predictions
        pred_random = torch.randn(1, 3, 16, 16, 16)
        loss_random = loss_fn(pred_random, target)

        # Predictions that match target (one-hot style)
        pred_correct = torch.zeros(1, 3, 16, 16, 16)
        for c in range(3):
            pred_correct[0, c, target[0, 0] == c] = 10.0  # High logit
        loss_correct = loss_fn(pred_correct, target)

        assert loss_correct < loss_random

    def test_different_lambda_weights(self):
        """Test different lambda weights produce different losses."""
        pred = torch.randn(1, 3, 16, 16, 16)
        target = torch.randint(0, 3, (1, 1, 16, 16, 16))

        loss_equal = SegmentationLoss(lambda_dice=1.0, lambda_ce=1.0)
        loss_dice_heavy = SegmentationLoss(lambda_dice=2.0, lambda_ce=0.5)

        l1 = loss_equal(pred, target)
        l2 = loss_dice_heavy(pred, target)

        # Just verify they're different
        assert l1.item() != l2.item()

    def test_batch_size_invariance(self):
        """Test loss scales reasonably with batch size."""
        loss_fn = SegmentationLoss()

        torch.manual_seed(42)
        pred = torch.randn(1, 3, 16, 16, 16)
        target = torch.randint(0, 3, (1, 1, 16, 16, 16))
        loss_single = loss_fn(pred, target)

        # Same data batched
        pred_batch = pred.repeat(4, 1, 1, 1, 1)
        target_batch = target.repeat(4, 1, 1, 1, 1)
        loss_batch = loss_fn(pred_batch, target_batch)

        # With mean reduction, should be similar
        assert abs(loss_single.item() - loss_batch.item()) < 0.1


class TestDeepSupervisionLoss:
    """Tests for DeepSupervisionLoss."""

    def test_single_tensor_input(self):
        """Test with single tensor (no deep supervision)."""
        loss_fn = DeepSupervisionLoss()
        pred = torch.randn(1, 3, 96, 96, 96)
        target = torch.randint(0, 3, (1, 1, 96, 96, 96))

        loss = loss_fn(pred, target)

        assert loss.dim() == 0

    def test_tuple_input(self):
        """Test with tuple input (deep supervision)."""
        loss_fn = DeepSupervisionLoss()
        pred_main = torch.randn(1, 3, 96, 96, 96)
        pred_ds = [
            torch.randn(1, 3, 6, 6, 6),
            torch.randn(1, 3, 12, 12, 12),
            torch.randn(1, 3, 24, 24, 24),
        ]
        target = torch.randint(0, 3, (1, 1, 96, 96, 96))

        loss = loss_fn((pred_main, pred_ds), target)

        assert loss.dim() == 0

    def test_ds_loss_higher_than_main_only(self):
        """Test DS loss is higher due to additional terms."""
        loss_fn = DeepSupervisionLoss()
        pred_main = torch.randn(1, 3, 96, 96, 96)
        pred_ds = [
            torch.randn(1, 3, 6, 6, 6),
            torch.randn(1, 3, 12, 12, 12),
            torch.randn(1, 3, 24, 24, 24),
        ]
        target = torch.randint(0, 3, (1, 1, 96, 96, 96))

        loss_main_only = loss_fn(pred_main, target)
        loss_with_ds = loss_fn((pred_main, pred_ds), target)

        # DS loss should be higher (main + weighted DS terms)
        assert loss_with_ds.item() > loss_main_only.item()

    def test_custom_weights(self):
        """Test custom DS weights."""
        loss_fn = DeepSupervisionLoss(weights=[1.0, 0.1, 0.1, 0.1])
        pred_main = torch.randn(1, 3, 96, 96, 96)
        pred_ds = [
            torch.randn(1, 3, 6, 6, 6),
            torch.randn(1, 3, 12, 12, 12),
            torch.randn(1, 3, 24, 24, 24),
        ]
        target = torch.randint(0, 3, (1, 1, 96, 96, 96))

        loss = loss_fn((pred_main, pred_ds), target)

        assert loss.dim() == 0


class TestDiceMetric:
    """Tests for DiceMetric."""

    def test_output_shape(self):
        """Test output shape without background."""
        metric = DiceMetric(include_background=False)
        pred = torch.randn(1, 3, 16, 16, 16)
        target = torch.randint(0, 3, (1, 1, 16, 16, 16))

        dice = metric(pred, target)

        # 3 classes - 1 background = 2 classes
        assert dice.shape == (2,)

    def test_output_shape_with_background(self):
        """Test output shape with background."""
        metric = DiceMetric(include_background=True)
        pred = torch.randn(1, 3, 16, 16, 16)
        target = torch.randint(0, 3, (1, 1, 16, 16, 16))

        dice = metric(pred, target)

        assert dice.shape == (3,)

    def test_dice_bounded(self):
        """Test Dice scores are in [0, 1]."""
        metric = DiceMetric()
        pred = torch.randn(1, 3, 16, 16, 16)
        target = torch.randint(0, 3, (1, 1, 16, 16, 16))

        dice = metric(pred, target)

        assert (dice >= 0).all()
        assert (dice <= 1).all()

    def test_perfect_prediction(self):
        """Test Dice is 1.0 for perfect predictions."""
        metric = DiceMetric(include_background=True)

        # Create target with all classes present
        target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
        target[0, 0, :4, :, :] = 0
        target[0, 0, 4:6, :, :] = 1
        target[0, 0, 6:, :, :] = 2

        # Create perfect predictions
        pred = torch.zeros(1, 3, 8, 8, 8)
        pred[0, 0, :4, :, :] = 10.0  # Background
        pred[0, 1, 4:6, :, :] = 10.0  # Class 1
        pred[0, 2, 6:, :, :] = 10.0  # Class 2

        dice = metric(pred, target)

        # All Dice scores should be 1.0
        assert torch.allclose(dice, torch.ones(3), atol=1e-5)

    def test_mean_reduction(self):
        """Test mean reduction."""
        metric = DiceMetric(reduction="mean")
        pred = torch.randn(1, 3, 16, 16, 16)
        target = torch.randint(0, 3, (1, 1, 16, 16, 16))

        dice = metric(pred, target)

        assert dice.dim() == 0


class TestCreateSegmentationLoss:
    """Tests for create_segmentation_loss factory."""

    def test_without_deep_supervision(self):
        """Test creating loss without DS."""
        loss_fn = create_segmentation_loss()
        pred = torch.randn(1, 3, 16, 16, 16)
        target = torch.randint(0, 3, (1, 1, 16, 16, 16))

        loss = loss_fn(pred, target)

        assert loss.dim() == 0

    def test_with_deep_supervision(self):
        """Test creating loss with DS."""
        loss_fn = create_segmentation_loss(deep_supervision=True)
        pred_main = torch.randn(1, 3, 96, 96, 96)
        pred_ds = [
            torch.randn(1, 3, 6, 6, 6),
            torch.randn(1, 3, 12, 12, 12),
            torch.randn(1, 3, 24, 24, 24),
        ]
        target = torch.randint(0, 3, (1, 1, 96, 96, 96))

        loss = loss_fn((pred_main, pred_ds), target)

        assert loss.dim() == 0

    def test_custom_lambda_weights(self):
        """Test custom lambda weights."""
        loss_fn = create_segmentation_loss(lambda_dice=2.0, lambda_ce=0.5)

        assert isinstance(loss_fn, SegmentationLoss)
        assert loss_fn.lambda_dice == 2.0
        assert loss_fn.lambda_ce == 0.5


# =============================================================================
# Label-convention tests for the 3-channel TC/WT/ET converter (BraTS-MEN ↔ BSF)
#
# These guard against a regression of the MEN convention: BraTS-MEN labels are
# {1=NETC, 2=SNFH (edema), 3=ET (enhancing tumor — main meningioma mass)}, and
# the BSF 3-channel sigmoid output expects TC=NETC∪ET, WT=all-tumor, ET=label 3.
# A previous version of the code had this swapped (ET = label 1) which silently
# broke LoRA training on meningioma data.
# =============================================================================


class TestLabelConversionMEN:
    """Verify BraTS-MEN label conversion produces correct TC/WT/ET targets.

    BSF effectively only models 2 tissues for meningioma (SNFH + ET); NETC is
    **merged into ET** during the BraTS→BSF translation (NETC is part of the
    solid meningioma mass that BSF was never trained to distinguish from ET).
    The TC sigmoid channel has no analogue in this 2-label space and is left
    empty (target = zeros) — the model is taught to suppress it.
    """

    def test_men_branch_labels_match_BSF_two_label_translation(self):
        """For MEN: TC=empty, WT=(1|2|3), ET=(1|3) — NETC merged into ET."""
        from growth.losses.segmentation import _convert_single_domain

        # Synthesize a tensor [B=1, D=1, H=1, W=5] containing one of each label.
        target = torch.tensor([[[[0, 1, 2, 3, 3]]]], dtype=torch.long).squeeze(1)

        masks = _convert_single_domain(target, "MEN")  # [1, 3, 1, 1, 5]

        # TC = empty (BSF has no tumor-core concept for meningioma)
        assert masks[0, 0].sum().item() == 0, "MEN TC must be entirely empty"
        # WT = {1, 2, 3} = all tumor = positions 1, 2, 3, 4 → 4 voxels
        assert masks[0, 1].sum().item() == 4, "MEN WT must be all tumor (1|2|3)"
        # ET = {1, 3} = merged_ET = positions 1, 3, 4 → 3 voxels
        assert masks[0, 2].sum().item() == 3, "MEN ET must be NETC ∪ ET (merged)"

        # NETC voxels must appear in BOTH WT and ET (merged into ET).
        wt_mask = masks[0, 1].flatten().bool()
        et_mask = masks[0, 2].flatten().bool()
        assert wt_mask[1].item(), "WT must include NETC (label 1) — it is tumor"
        assert et_mask[1].item(), "ET must include NETC (label 1) — merged into ET"
        # SNFH (label 2) is in WT but NOT in ET (edema is not part of the solid mass).
        assert wt_mask[2].item(), "WT must include SNFH (label 2)"
        assert not et_mask[2].item(), "ET must not include SNFH (edema)"

    def test_gli_branch_unchanged(self):
        """Regression: GLI branch retains TC=(1|3|4), WT=(>0), ET=(3)."""
        from growth.losses.segmentation import _convert_single_domain

        target = torch.tensor([[[[0, 1, 2, 3, 4]]]], dtype=torch.long).squeeze(1)
        masks = _convert_single_domain(target, "GLI")

        assert masks[0, 0].sum().item() == 3  # TC = {1, 3, 4}
        assert masks[0, 1].sum().item() == 4  # WT = {1, 2, 3, 4}
        assert masks[0, 2].sum().item() == 1  # ET = {3} only

    def test_dice_metric_3ch_men_recognises_merged_ET(self):
        """Perfect prediction of (NETC ∪ ET) voxels must yield ET Dice ≈ 1.0 on MEN."""
        from growth.losses.segmentation import DiceMetric3Ch

        # GT: a small spatial volume with label 3 (ET), label 1 (NETC), label 2 (SNFH).
        D = H = W = 4
        target = torch.zeros(1, 1, D, H, W, dtype=torch.long)
        target[0, 0, 1:3, 1:3, 1:3] = 3  # 8 ET voxels
        target[0, 0, 0, 0, 0] = 1  # 1 NETC voxel — must be predicted as part of ET
        target[0, 0, 0, 0, 1] = 2  # 1 SNFH voxel (in WT only)

        # Predict ET (channel 2) over both ET region AND the NETC voxel (merged target);
        # WT (channel 1) covers everything tumor; TC (channel 0) stays empty.
        logits = -10.0 * torch.ones(1, 3, D, H, W)
        logits[0, 2, 1:3, 1:3, 1:3] = 10.0  # ET part of merged target
        logits[0, 2, 0, 0, 0] = 10.0  # NETC voxel — also predicted as ET
        logits[0, 1, 1:3, 1:3, 1:3] = 10.0  # WT covers ET region
        logits[0, 1, 0, 0, 0] = 10.0  # WT covers NETC voxel
        logits[0, 1, 0, 0, 1] = 10.0  # WT covers SNFH voxel

        metric = DiceMetric3Ch()
        dice = metric(logits, target, domain="MEN")  # [B=1, 3]

        assert dice.shape == (1, 3)
        # ET (channel 2) ≈ 1.0 — we covered NETC ∪ ET perfectly
        assert dice[0, 2].item() > 0.99, f"Expected ET Dice ≈ 1.0, got {dice[0, 2].item():.3f}"
        # WT (channel 1) ≈ 1.0 — we covered all tumor
        assert dice[0, 1].item() > 0.99, f"Expected WT Dice ≈ 1.0, got {dice[0, 1].item():.3f}"
        # TC (channel 0) — empty target + empty pred → DiceMetric3Ch returns 1.0 (union==0 path)
        assert dice[0, 0].item() > 0.99, (
            f"Expected TC Dice ≈ 1.0 (empty/empty), got {dice[0, 0].item():.3f}"
        )

    def test_men_tc_channel_is_always_empty(self):
        """TC target must be all zeros for MEN regardless of input labels."""
        from growth.losses.segmentation import _convert_single_domain

        target = torch.tensor([0, 1, 2, 3, 3, 1, 2, 3], dtype=torch.long).reshape(1, 8)
        masks = _convert_single_domain(target, "MEN")
        assert masks[0, 0].sum().item() == 0, "TC must be empty for any MEN target"

    def test_men_merge_regression_caught(self):
        """Verify NETC (label 1) is merged into ET, not dropped, not standalone."""
        from growth.losses.segmentation import _convert_single_domain

        # All-label-1 target — must be marked as ET (merged) AND as WT.
        target = torch.ones(1, 1, 4, 4, 4, dtype=torch.long).squeeze(1)
        masks = _convert_single_domain(target, "MEN")
        n_vox = 4 * 4 * 4
        assert masks[0, 0].sum().item() == 0, "TC must remain empty even if input is all NETC"
        assert masks[0, 1].sum().item() == n_vox, "WT must include every NETC voxel (it is tumor)"
        assert masks[0, 2].sum().item() == n_vox, (
            "ET must include every NETC voxel (merged into ET)"
        )

        # All-label-3 target — every voxel is ET (and ET ⊂ WT).
        target = (3 * torch.ones(1, 1, 4, 4, 4, dtype=torch.long)).squeeze(1)
        masks = _convert_single_domain(target, "MEN")
        assert masks[0, 2].sum().item() == n_vox, "ET channel must cover every label-3 voxel"
        assert masks[0, 1].sum().item() == n_vox, "WT channel must cover every label-3 voxel"

        # All-label-2 target — SNFH is in WT but NOT in ET (edema is not part of solid mass).
        target = (2 * torch.ones(1, 1, 4, 4, 4, dtype=torch.long)).squeeze(1)
        masks = _convert_single_domain(target, "MEN")
        assert masks[0, 1].sum().item() == n_vox, "WT channel must cover SNFH (edema)"
        assert masks[0, 2].sum().item() == 0, "ET channel must NOT contain SNFH (edema)"
