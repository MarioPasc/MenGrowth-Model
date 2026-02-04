# src/growth/evaluation/segmentation_metrics.py
"""Segmentation evaluation metrics and utilities.

Provides generic segmentation model evaluation capabilities:
- Multi-class Dice score computation
- Per-class metrics tracking
- Batch evaluation with DataLoader

Example:
    >>> from growth.evaluation.segmentation_metrics import SegmentationEvaluator
    >>> evaluator = SegmentationEvaluator(device="cuda")
    >>> results = evaluator.evaluate(model, dataloader)
    >>> print(f"Mean Dice: {results['dice_mean']:.4f}")
"""

import logging
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SegmentationEvaluator:
    """Generic segmentation model evaluator.

    Evaluates segmentation models on datasets, computing per-class and
    mean Dice scores.

    Args:
        dice_metric: Custom Dice metric function. If None, uses internal implementation.
        device: Device to run evaluation on.
        num_classes: Number of segmentation classes (excluding background).
        class_names: Optional names for classes (for logging).
        include_background: Whether to include background in Dice computation.

    Example:
        >>> evaluator = SegmentationEvaluator(device="cuda", num_classes=3)
        >>> model = load_my_model()
        >>> dataloader = create_dataloader(test_ids)
        >>> results = evaluator.evaluate(model, dataloader, desc="Testing")
        >>> print(f"Dice: {results['dice_mean']:.4f}")
    """

    def __init__(
        self,
        dice_metric: Optional[Callable] = None,
        device: str = "cuda",
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
        include_background: bool = False,
    ):
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.include_background = include_background

        if dice_metric is not None:
            self.dice_metric = dice_metric
        else:
            # Try to import from growth.losses.segmentation
            try:
                from growth.losses.segmentation import DiceMetric3Ch

                self.dice_metric = DiceMetric3Ch()
            except ImportError:
                # Fallback to internal implementation
                self.dice_metric = self._default_dice_metric

    def _default_dice_metric(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-class Dice scores.

        Args:
            pred: Predictions [B, C, D, H, W] (logits or softmax).
            target: Ground truth [B, 1, D, H, W] or [B, D, H, W] (integer labels).

        Returns:
            Dice scores [B, num_classes].
        """
        # Convert logits to predictions
        if pred.shape[1] > 1:
            pred_classes = pred.argmax(dim=1)
        else:
            pred_classes = (pred > 0.5).squeeze(1).long()

        # Squeeze target if needed
        if target.dim() == 5:
            target = target.squeeze(1)

        batch_size = pred.shape[0]
        dice_scores = []

        for b in range(batch_size):
            class_dices = []
            for c in range(self.num_classes):
                # Class index (1-indexed if background is class 0)
                class_idx = c + 1 if not self.include_background else c

                pred_mask = (pred_classes[b] == class_idx).float()
                target_mask = (target[b] == class_idx).float()

                intersection = (pred_mask * target_mask).sum()
                union = pred_mask.sum() + target_mask.sum()

                if union > 0:
                    dice = 2 * intersection / union
                else:
                    dice = torch.tensor(1.0, device=pred.device)  # Empty class = 1.0

                class_dices.append(dice)

            dice_scores.append(torch.stack(class_dices))

        return torch.stack(dice_scores)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        desc: str = "Evaluating",
        image_key: str = "image",
        seg_key: str = "seg",
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.

        Args:
            model: PyTorch model to evaluate.
            dataloader: DataLoader for evaluation dataset.
            desc: Description for progress bar.
            image_key: Key for input images in batch dict.
            seg_key: Key for segmentation masks in batch dict.

        Returns:
            Dict with metrics:
            - 'dice_mean': Mean Dice across all classes
            - 'dice_{class_name}': Per-class Dice scores
            - 'dice_std': Standard deviation of per-sample mean Dice
            - 'num_samples': Number of samples evaluated
        """
        model.eval()
        all_dice_scores = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch[image_key].to(self.device)
            segs = batch[seg_key].to(self.device)

            # Forward pass - handle different model interfaces
            if hasattr(model, "forward_with_semantics"):
                outputs = model.forward_with_semantics(images)
                pred = outputs["logits"]
            else:
                pred = model(images)

            # Compute Dice per class
            dice_scores = self.dice_metric(pred, segs)
            all_dice_scores.append(dice_scores.cpu())

        # Aggregate results
        dice_tensor = torch.cat(all_dice_scores, dim=0)  # [N, num_classes]
        dice_mean_per_sample = dice_tensor.mean(dim=1)  # [N]

        results = {
            "dice_mean": float(dice_tensor.mean()),
            "dice_std": float(dice_mean_per_sample.std()),
            "num_samples": len(dice_tensor),
        }

        # Add per-class metrics
        for i, name in enumerate(self.class_names):
            results[f"dice_{name}"] = float(dice_tensor[:, i].mean())

        return results

    @torch.no_grad()
    def evaluate_single(
        self,
        model: nn.Module,
        image: torch.Tensor,
        seg: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate model on a single sample.

        Args:
            model: PyTorch model to evaluate.
            image: Input image [1, C, D, H, W] or [C, D, H, W].
            seg: Segmentation mask [1, 1, D, H, W] or [D, H, W].

        Returns:
            Dict with per-class and mean Dice scores.
        """
        model.eval()

        # Add batch dimension if needed
        if image.dim() == 4:
            image = image.unsqueeze(0)
        if seg.dim() == 3:
            seg = seg.unsqueeze(0).unsqueeze(0)
        elif seg.dim() == 4:
            seg = seg.unsqueeze(0)

        image = image.to(self.device)
        seg = seg.to(self.device)

        # Forward pass
        if hasattr(model, "forward_with_semantics"):
            outputs = model.forward_with_semantics(image)
            pred = outputs["logits"]
        else:
            pred = model(image)

        # Compute Dice
        dice_scores = self.dice_metric(pred, seg)  # [1, num_classes]

        results = {"dice_mean": float(dice_scores.mean())}
        for i, name in enumerate(self.class_names):
            results[f"dice_{name}"] = float(dice_scores[0, i])

        return results


def compute_dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Compute Dice coefficient between prediction and target.

    Args:
        pred: Binary prediction tensor.
        target: Binary target tensor.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice coefficient (scalar tensor).
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + smooth) / (union + smooth)


def compute_per_class_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
) -> Dict[int, float]:
    """Compute Dice coefficient for each class.

    Args:
        pred: Prediction tensor [D, H, W] (integer labels).
        target: Target tensor [D, H, W] (integer labels).
        num_classes: Number of classes (excluding background).
        include_background: Whether to include background (class 0).

    Returns:
        Dict mapping class index to Dice score.
    """
    start_class = 0 if include_background else 1
    end_class = num_classes if include_background else num_classes + 1

    results = {}
    for c in range(start_class, end_class):
        pred_mask = (pred == c).float()
        target_mask = (target == c).float()

        dice = compute_dice_coefficient(pred_mask, target_mask)
        results[c] = float(dice)

    return results


__all__ = [
    "SegmentationEvaluator",
    "compute_dice_coefficient",
    "compute_per_class_dice",
]
