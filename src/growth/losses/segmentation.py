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

        # Convert target to one-hot
        target_onehot = F.one_hot(target.squeeze(1), num_classes)
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
