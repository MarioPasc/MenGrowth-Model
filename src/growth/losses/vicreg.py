# src/growth/losses/vicreg.py
"""VICReg-style regularization losses.

Implements:
- Cross-partition covariance loss (linear decorrelation)
- Variance hinge loss (collapse prevention)

Adapted from Bardes et al., ICLR 2022.
"""

import logging
from itertools import combinations

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CovarianceLoss(nn.Module):
    """Cross-partition covariance penalty.

    Penalizes covariance BETWEEN different partition pairs only.
    Does NOT penalize within-partition correlation (the semantic heads
    handle intra-partition structure).

    For each pair (i, j), computes the cross-covariance matrix and
    returns the mean of squared off-diagonal elements.

    Example:
        >>> loss_fn = CovarianceLoss()
        >>> partitions = {"vol": torch.randn(100, 24),
        ...               "loc": torch.randn(100, 8),
        ...               "shape": torch.randn(100, 12)}
        >>> loss = loss_fn(partitions)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        partitions: dict[str, torch.Tensor],
        partition_names: list[str] = ("vol", "loc", "shape"),
    ) -> torch.Tensor:
        """Compute cross-partition covariance loss.

        Args:
            partitions: Dict of partition tensors {name: [B, dim]}.
            partition_names: Names of partitions to penalize.

        Returns:
            Mean squared cross-covariance across all partition pairs.
        """
        device = next(iter(partitions.values())).device
        total_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for name_i, name_j in combinations(partition_names, 2):
            z_i = partitions[name_i]  # [B, d_i]
            z_j = partitions[name_j]  # [B, d_j]

            # Center
            z_i_centered = z_i - z_i.mean(dim=0)
            z_j_centered = z_j - z_j.mean(dim=0)

            batch_size = z_i.shape[0]

            # Cross-covariance matrix: [d_i, d_j]
            cross_cov = (z_i_centered.T @ z_j_centered) / (batch_size - 1)

            # Mean squared elements
            total_loss = total_loss + (cross_cov**2).mean()
            n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss


class VarianceHingeLoss(nn.Module):
    """Variance hinge loss for collapse prevention.

    Encourages each latent dimension to have standard deviation >= gamma.
    Applied to the FULL latent vector (including residual).

    L_var = mean(max(0, gamma - std(z_d))) over all dims d.

    Args:
        gamma: Target minimum standard deviation per dimension.

    Example:
        >>> loss_fn = VarianceHingeLoss(gamma=1.0)
        >>> z = torch.randn(100, 128)
        >>> loss = loss_fn(z)  # Should be ~0 if z is standard normal
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute variance hinge loss.

        Args:
            z: Full latent vector [B, D].

        Returns:
            Mean hinge loss across all dimensions.
        """
        # Per-dimension standard deviation with numerical stability (ε=1e-4)
        # Matches spec: sqrt(Var(z_j) + ε) from methodology_refined.md §3.3
        eps = 1e-4
        std = torch.sqrt(z.var(dim=0) + eps)  # [D]

        # Hinge: penalize dimensions with std < gamma
        hinge = torch.clamp(self.gamma - std, min=0.0)

        return hinge.mean()
