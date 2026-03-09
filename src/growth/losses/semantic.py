# src/growth/losses/semantic.py
"""Semantic regression losses for SDP training.

MSE loss for volume prediction from latent partition.
Methodology Revision R1: location and shape losses removed.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SemanticRegressionLoss(nn.Module):
    """Weighted MSE loss for volume target prediction.

    Loss: L_vol = lambda_vol * MSE(pred_vol, target_vol)

    Args:
        lambda_vol: Weight for volume loss.

    Example:
        >>> loss_fn = SemanticRegressionLoss()
        >>> preds = {"vol": torch.randn(8, 1)}
        >>> targets = {"vol": torch.randn(8, 1)}
        >>> total, details = loss_fn(preds, targets)
    """

    def __init__(
        self,
        lambda_vol: float = 25.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.lambdas = {"vol": lambda_vol}

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

        for name in ["vol"]:
            pred = predictions[name]
            target = targets[name]

            # MSE: mean over B × k_p elements (torch.mean already normalizes)
            mse = torch.mean((pred - target) ** 2)
            details[f"mse_{name}"] = mse.detach()

            total_loss = total_loss + self.lambdas[name] * mse

        return total_loss, details
