# src/growth/losses/segmentation.py
"""Segmentation losses for Phase 1 LoRA adaptation.

Implements Dice loss + Cross-Entropy for meningioma segmentation using
MONAI's robust loss implementations. Standard BraTS-style multi-class
segmentation objective with support for deep supervision.

BraTS Segmentation Labels:
    - 0: Background
    - 1: NCR (Necrotic Core)
    - 2: ED (Peritumoral Edema)
    - 3: ET (Enhancing Tumor)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss, FocalLoss

logger = logging.getLogger(__name__)


class SegmentationLoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for BraTS-style segmentation.

    This loss combines:
    - Dice loss: Good for class imbalance, focuses on overlap
    - Cross-Entropy: Provides strong gradients, especially early in training

    Args:
        include_background: Whether to include background class in Dice.
        softmax: Apply softmax to predictions before computing loss.
        to_onehot_y: Convert targets to one-hot encoding.
        lambda_dice: Weight for Dice loss component.
        lambda_ce: Weight for Cross-Entropy loss component.
        reduction: Reduction method ('mean', 'sum', 'none').
        smooth_nr: Dice numerator smoothing constant.
        smooth_dr: Dice denominator smoothing constant.

    Example:
        >>> loss_fn = SegmentationLoss()
        >>> pred = torch.randn(1, 3, 96, 96, 96)  # logits
        >>> target = torch.randint(0, 3, (1, 1, 96, 96, 96))  # labels
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        include_background: bool = False,
        softmax: bool = True,
        to_onehot_y: bool = True,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ):
        super().__init__()

        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

        # MONAI's combined Dice + CE loss
        self.dice_ce = DiceCELoss(
            include_background=include_background,
            softmax=softmax,
            to_onehot_y=to_onehot_y,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined Dice + CE loss.

        Args:
            pred: Predicted logits [B, C, D, H, W].
            target: Ground truth labels [B, 1, D, H, W] (integers) or
                    [B, C, D, H, W] (one-hot encoded).

        Returns:
            Combined loss scalar.
        """
        return self.dice_ce(pred, target)


class DeepSupervisionLoss(nn.Module):
    """Segmentation loss with deep supervision support.

    Combines losses from main output and intermediate outputs at
    different resolutions, with decreasing weights for deeper outputs.

    Args:
        base_loss: Base loss function to use at each level.
        weights: Weights for [main, ds1, ds2, ds3] outputs.
            If None, uses default [1.0, 0.5, 0.25, 0.125].

    Example:
        >>> loss_fn = DeepSupervisionLoss()
        >>> pred_main = torch.randn(1, 3, 96, 96, 96)
        >>> pred_ds = [
        ...     torch.randn(1, 3, 6, 6, 6),
        ...     torch.randn(1, 3, 12, 12, 12),
        ...     torch.randn(1, 3, 24, 24, 24),
        ... ]
        >>> target = torch.randint(0, 3, (1, 1, 96, 96, 96))
        >>> loss = loss_fn((pred_main, pred_ds), target)
    """

    def __init__(
        self,
        base_loss: Optional[nn.Module] = None,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.base_loss = base_loss or SegmentationLoss()
        self.weights = weights or [1.0, 0.5, 0.25, 0.125]

    def forward(
        self,
        pred: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss with optional deep supervision.

        Args:
            pred: Either a single prediction tensor [B, C, D, H, W],
                  or a tuple of (main_pred, [ds_preds...]).
            target: Ground truth labels [B, 1, D, H, W].

        Returns:
            Combined loss scalar.
        """
        # Handle single prediction (no deep supervision)
        if isinstance(pred, torch.Tensor):
            return self.base_loss(pred, target)

        # Handle deep supervision tuple
        main_pred, ds_preds = pred

        # Main loss
        total_loss = self.weights[0] * self.base_loss(main_pred, target)

        # Deep supervision losses (resize target to match prediction size)
        for i, ds_pred in enumerate(ds_preds):
            if i + 1 >= len(self.weights):
                break

            # Resize target to match DS prediction spatial dimensions
            target_resized = F.interpolate(
                target.float(),
                size=ds_pred.shape[2:],
                mode="nearest",
            ).long()

            total_loss += self.weights[i + 1] * self.base_loss(ds_pred, target_resized)

        return total_loss


class DiceMetric(nn.Module):
    """Dice score metric for evaluation (not training).

    Computes Dice score per class for BraTS-style evaluation.

    Args:
        include_background: Whether to include background class.
        reduction: How to reduce class-wise Dice ('mean', 'none').

    Example:
        >>> metric = DiceMetric()
        >>> pred = torch.randn(1, 3, 96, 96, 96)
        >>> target = torch.randint(0, 3, (1, 1, 96, 96, 96))
        >>> dice_scores = metric(pred, target)
        >>> dice_scores.shape
        torch.Size([3])  # Dice for each class
    """

    def __init__(
        self,
        include_background: bool = False,
        reduction: str = "none",
    ):
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice scores per class.

        Args:
            pred: Predicted logits [B, C, D, H, W].
            target: Ground truth labels [B, 1, D, H, W].

        Returns:
            Dice scores per class [C] or scalar if reduction='mean'.
        """
        # Get number of classes
        num_classes = pred.shape[1]

        # Convert predictions to one-hot
        pred_softmax = F.softmax(pred, dim=1)
        pred_argmax = pred_softmax.argmax(dim=1, keepdim=True)
        pred_onehot = F.one_hot(pred_argmax.squeeze(1), num_classes)
        pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).float()

        # Convert target to one-hot (must be LongTensor for one_hot)
        target_onehot = F.one_hot(target.squeeze(1).long(), num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        # Compute Dice per class
        dice_scores = []
        start_class = 0 if self.include_background else 1

        for c in range(start_class, num_classes):
            pred_c = pred_onehot[:, c]
            target_c = target_onehot[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union == 0:
                # Both empty: perfect match
                dice = torch.tensor(1.0, device=pred.device)
            else:
                dice = (2.0 * intersection) / (union + 1e-8)

            dice_scores.append(dice)

        dice_tensor = torch.stack(dice_scores)

        if self.reduction == "mean":
            return dice_tensor.mean()
        return dice_tensor


def create_segmentation_loss(
    lambda_dice: float = 1.0,
    lambda_ce: float = 1.0,
    deep_supervision: bool = False,
    ds_weights: Optional[List[float]] = None,
) -> nn.Module:
    """Factory function to create segmentation loss.

    Args:
        lambda_dice: Weight for Dice loss.
        lambda_ce: Weight for Cross-Entropy loss.
        deep_supervision: Enable deep supervision.
        ds_weights: Weights for deep supervision outputs.

    Returns:
        Configured loss module.

    Example:
        >>> loss_fn = create_segmentation_loss(deep_supervision=True)
        >>> # Use with model that returns (main_pred, ds_preds)
    """
    base_loss = SegmentationLoss(
        lambda_dice=lambda_dice,
        lambda_ce=lambda_ce,
    )

    if deep_supervision:
        return DeepSupervisionLoss(
            base_loss=base_loss,
            weights=ds_weights,
        )

    return base_loss


# =============================================================================
# 3-Channel Approach (Sigmoid per channel, no explicit background)
# =============================================================================
# BrainSegFounder was trained with 3 output channels using sigmoid activation
# per channel. This allows using the FULL pretrained decoder including output
# layer, which is critical for frozen baseline comparisons.
#
# Channel mapping:
#   - Channel 0: NCR (Necrotic Core) - corresponds to label == 1
#   - Channel 1: ED (Peritumoral Edema) - corresponds to label == 2
#   - Channel 2: ET (Enhancing Tumor) - corresponds to label == 3
#
# Background is implicit (where all channels < threshold).
# =============================================================================


class SegmentationLoss3Ch(nn.Module):
    """Segmentation loss for 3-channel sigmoid outputs.

    Combines Dice loss and Binary Cross-Entropy for each channel independently.
    Converts integer labels to 3 binary masks internally.

    Args:
        lambda_dice: Weight for Dice loss component.
        lambda_bce: Weight for BCE loss component.
        smooth: Smoothing constant for Dice computation.

    Example:
        >>> loss_fn = SegmentationLoss3Ch()
        >>> pred = torch.randn(2, 3, 96, 96, 96)  # logits (pre-sigmoid)
        >>> target = torch.randint(0, 4, (2, 1, 96, 96, 96))  # integer labels
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.smooth = smooth

    def _convert_target(self, target: torch.Tensor) -> torch.Tensor:
        """Convert integer labels [B, 1, D, H, W] to binary masks [B, 3, D, H, W].

        Args:
            target: Integer labels with values 0 (background), 1 (NCR), 2 (ED), 3 (ET).

        Returns:
            Binary masks for each of the 3 tumor classes.
        """
        # Squeeze channel dim if present
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)  # [B, D, H, W]

        # Create binary masks for each class
        ncr = (target == 1).float()  # [B, D, H, W]
        ed = (target == 2).float()
        et = (target == 3).float()

        # Stack to [B, 3, D, H, W]
        return torch.stack([ncr, ed, et], dim=1)

    def _dice_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice loss per channel and average.

        Args:
            pred: Sigmoid probabilities [B, 3, D, H, W].
            target: Binary masks [B, 3, D, H, W].

        Returns:
            Average Dice loss across channels.
        """
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)  # [B, 3, N]
        target_flat = target.view(target.shape[0], target.shape[1], -1)

        # Compute Dice per channel
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        # Average across channels and batch
        return dice_loss.mean()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined Dice + BCE loss.

        Args:
            pred: Predicted logits [B, 3, D, H, W] (pre-sigmoid).
            target: Ground truth labels [B, 1, D, H, W] (integers 0-3).

        Returns:
            Combined loss scalar.
        """
        # Convert integer labels to binary masks
        target_3ch = self._convert_target(target)

        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)

        # Dice loss
        dice_loss = self._dice_loss(pred_prob, target_3ch)

        # BCE loss (using logits for numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target_3ch, reduction="mean"
        )

        return self.lambda_dice * dice_loss + self.lambda_bce * bce_loss


class DiceMetric3Ch(nn.Module):
    """Dice score metric for 3-channel sigmoid outputs.

    Computes Dice score per channel for evaluation.

    Args:
        threshold: Threshold for converting probabilities to binary predictions.
        reduction: How to reduce channel-wise Dice ('mean', 'none').

    Example:
        >>> metric = DiceMetric3Ch()
        >>> pred = torch.randn(2, 3, 96, 96, 96)  # logits
        >>> target = torch.randint(0, 4, (2, 1, 96, 96, 96))  # integer labels
        >>> dice_scores = metric(pred, target)
        >>> dice_scores.shape
        torch.Size([3])  # Dice for NCR, ED, ET
    """

    def __init__(
        self,
        threshold: float = 0.5,
        reduction: str = "none",
    ):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

    def _convert_target(self, target: torch.Tensor) -> torch.Tensor:
        """Convert integer labels to binary masks."""
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        ncr = (target == 1).float()
        ed = (target == 2).float()
        et = (target == 3).float()

        return torch.stack([ncr, ed, et], dim=1)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice scores per channel.

        Args:
            pred: Predicted logits [B, 3, D, H, W].
            target: Ground truth labels [B, 1, D, H, W].

        Returns:
            Dice scores per channel [3] or scalar if reduction='mean'.
        """
        # Convert target to binary masks
        target_3ch = self._convert_target(target)

        # Apply sigmoid and threshold
        pred_prob = torch.sigmoid(pred)
        pred_binary = (pred_prob > self.threshold).float()

        # Compute Dice per channel
        dice_scores = []
        for c in range(3):
            pred_c = pred_binary[:, c]
            target_c = target_3ch[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union == 0:
                # Both empty: perfect match
                dice = torch.tensor(1.0, device=pred.device)
            else:
                dice = (2.0 * intersection) / (union + 1e-8)

            dice_scores.append(dice)

        dice_tensor = torch.stack(dice_scores)

        if self.reduction == "mean":
            return dice_tensor.mean()
        return dice_tensor
