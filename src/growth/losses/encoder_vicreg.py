# src/growth/losses/encoder_vicreg.py
"""VICReg-style regularization for encoder features.

Applies variance + covariance regularization directly on the 768-dim
GAP-pooled encoder10 features during LoRA training. This fights
dimensional collapse at the encoder level (root cause of low SDP ceilings).

Unlike the SDP-level vicreg.py which operates on 128-dim post-projection
latents, this operates on the raw 768-dim encoder features.

Follows Bardes et al., ICLR 2022, but without the invariance term
(we have no augmented views — segmentation is the primary task).

L_vicreg = lambda_var * L_var + lambda_cov * L_cov

L_var = (1/D) sum_d max(0, gamma - sqrt(Var(h_d) + eps))^2
L_cov = (1/D^2) sum_{d != d'} Cov(h_d, h_d')^2
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class EncoderVICRegLoss(nn.Module):
    """VICReg variance + covariance loss for encoder features.

    Designed for 768-dim GAP-pooled encoder10 outputs. Always active
    (no warmup needed — it's unsupervised).

    Args:
        lambda_var: Weight for variance hinge loss.
        lambda_cov: Weight for covariance penalty.
        gamma: Target minimum std per dimension.

    Example:
        >>> loss_fn = EncoderVICRegLoss(lambda_var=5.0, lambda_cov=1.0)
        >>> features = torch.randn(8, 768)
        >>> loss, components = loss_fn(features)
    """

    def __init__(
        self,
        lambda_var: float = 5.0,
        lambda_cov: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute encoder VICReg loss.

        Args:
            features: GAP-pooled encoder features [B, D] (e.g. [B, 768]).

        Returns:
            Tuple of (total_loss, component_dict) where component_dict
            has 'var_loss' and 'cov_loss' scalar values.
        """
        assert features.ndim == 2, f"Expected 2D tensor, got {features.ndim}D"

        batch_size, feat_dim = features.shape

        # VICReg requires batch_size >= 2 for meaningful variance/covariance
        if batch_size < 2:
            zero = torch.tensor(0.0, device=features.device, requires_grad=True)
            return zero, {"vicreg_var_loss": 0.0, "vicreg_cov_loss": 0.0, "vicreg_total": 0.0}

        # Variance hinge loss: keep each dimension alive
        # L_var = (1/D) sum_d max(0, gamma - sqrt(Var(h_d) + eps))^2
        eps = 1e-4
        std = torch.sqrt(features.var(dim=0) + eps)  # [D]
        var_loss = torch.clamp(self.gamma - std, min=0.0).pow(2).mean()

        # Covariance loss: decorrelate dimensions
        # L_cov = (1/D^2) sum_{d != d'} Cov(h_d, h_d')^2
        features_centered = features - features.mean(dim=0)
        cov_matrix = (features_centered.T @ features_centered) / (batch_size - 1)
        # Zero diagonal (we only penalize off-diagonal)
        cov_off_diag = cov_matrix - torch.diag(cov_matrix.diag())
        cov_loss = (cov_off_diag.pow(2)).sum() / (feat_dim * feat_dim)

        total_loss = self.lambda_var * var_loss + self.lambda_cov * cov_loss

        components = {
            "vicreg_var_loss": var_loss.item(),
            "vicreg_cov_loss": cov_loss.item(),
            "vicreg_total": total_loss.item(),
        }

        return total_loss, components
