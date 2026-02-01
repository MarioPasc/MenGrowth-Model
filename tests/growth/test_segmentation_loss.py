# tests/growth/test_segmentation_loss.py
"""Tests for segmentation loss functions."""

import pytest
import torch

from growth.losses.segmentation import (
    SegmentationLoss,
    DeepSupervisionLoss,
    DiceMetric,
    create_segmentation_loss,
)


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
