# src/growth/losses/semantic.py
"""Semantic regression losses for SDP training.

MSE losses for volume, location, and shape prediction from latent partitions.
Ensures informativeness of the disentangled representation.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SemanticRegressionLoss(nn.Module):
    """Weighted per-partition MSE loss for semantic target prediction.

    Each partition's loss is normalized by its target dimensionality
    (1/k_p factor) and weighted by a per-partition lambda.

    Loss per partition: L_p = lambda_p * (1/k_p) * ||pred_p - target_p||^2

    Args:
        lambda_vol: Weight for volume loss.
        lambda_loc: Weight for location loss.
        lambda_shape: Weight for shape loss.

    Example:
        >>> loss_fn = SemanticRegressionLoss()
        >>> preds = {"vol": torch.randn(8, 4), "loc": torch.randn(8, 3),
        ...          "shape": torch.randn(8, 3)}
        >>> targets = {"vol": torch.randn(8, 4), "loc": torch.randn(8, 3),
        ...            "shape": torch.randn(8, 3)}
        >>> total, details = loss_fn(preds, targets)
    """

    def __init__(
        self,
        lambda_vol: float = 20.0,
        lambda_loc: float = 12.0,
        lambda_shape: float = 15.0,
    ) -> None:
        super().__init__()
        self.lambdas = {"vol": lambda_vol, "loc": lambda_loc, "shape": lambda_shape}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted semantic regression loss.

        Args:
            predictions: Dict of predicted targets {name: [B, k_p]}.
            targets: Dict of ground truth targets {name: [B, k_p]}.

        Returns:
            Tuple of (total weighted loss, dict of unweighted per-partition losses).
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        details = {}

        for name in ["vol", "loc", "shape"]:
            pred = predictions[name]
            target = targets[name]

            # Dimension-normalized MSE: (1/k_p) * ||pred - target||^2
            k_p = pred.shape[1]
            mse = torch.mean((pred - target) ** 2) / k_p
            details[f"mse_{name}"] = mse.detach()

            total_loss = total_loss + self.lambdas[name] * mse

        return total_loss, details
