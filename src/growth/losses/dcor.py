# src/growth/losses/dcor.py
"""Distance correlation loss for nonlinear independence.

Implements dCor(z_i, z_j) between latent partitions.
Captures all forms of statistical dependence (Szekely et al., 2007).

Pure PyTorch implementation — no external dcor library needed.
O(N^2) complexity, trivial for N~800 (full-batch SDP training).
"""

import logging
from itertools import combinations

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix.

    Args:
        x: Input tensor [N, D].

    Returns:
        Distance matrix [N, N].
    """
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
    x_sq = (x**2).sum(dim=1)  # [N]
    dist_sq = x_sq.unsqueeze(1) + x_sq.unsqueeze(0) - 2.0 * (x @ x.T)
    # Clamp for numerical stability before sqrt
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.sqrt(dist_sq + 1e-10)


def _double_center(d: torch.Tensor) -> torch.Tensor:
    """Double-center a distance matrix.

    A_{ij} = d_{ij} - mean_row_i - mean_col_j + grand_mean

    Args:
        d: Distance matrix [N, N].

    Returns:
        Double-centered matrix [N, N].
    """
    row_mean = d.mean(dim=1, keepdim=True)
    col_mean = d.mean(dim=0, keepdim=True)
    grand_mean = d.mean()
    return d - row_mean - col_mean + grand_mean


def distance_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute distance correlation between two multivariate samples.

    Uses the V-statistic formulation (Szekely et al., 2007):
        dCor^2(X, Y) = dCov^2(X, Y) / sqrt(dVar^2(X) * dVar^2(Y))
        dCor(X, Y) = sqrt(dCor^2(X, Y))

    where dCov^2 = (1/n^2) * sum A_ij * B_ij with double-centered distances.

    Args:
        x: First sample [N, D1].
        y: Second sample [N, D2].
        eps: Small constant for numerical stability.

    Returns:
        Distance correlation scalar in [0, 1].
    """
    assert x.shape[0] == y.shape[0], "Samples must have same batch size"

    # Pairwise distances
    dx = _pairwise_distances(x)
    dy = _pairwise_distances(y)

    # Double centering
    a = _double_center(dx)
    b = _double_center(dy)

    # Distance covariance and variances (squared)
    n = x.shape[0]
    dcov2 = (a * b).sum() / (n * n)
    dvar_x2 = (a * a).sum() / (n * n)
    dvar_y2 = (b * b).sum() / (n * n)

    # Clamp non-negative
    dcov2 = torch.clamp(dcov2, min=0.0)
    dvar_x2 = torch.clamp(dvar_x2, min=0.0)
    dvar_y2 = torch.clamp(dvar_y2, min=0.0)

    # dCor^2 = dCov^2 / sqrt(dVar_X^2 * dVar_Y^2)
    denom = torch.sqrt(dvar_x2 * dvar_y2)
    if denom < eps:
        return torch.tensor(0.0, device=x.device, requires_grad=x.requires_grad)

    dcor_sq = dcov2 / (denom + eps)

    # dCor = sqrt(dCor^2), clamped to [0, 1]
    dcor_sq = torch.clamp(dcor_sq, 0.0, 1.0)
    return torch.sqrt(dcor_sq)


class DistanceCorrelationLoss(nn.Module):
    """Distance correlation loss between supervised latent partitions.

    Computes mean dCor across all C(k, 2) pairs of supervised partitions.

    Args:
        partition_names: Names of partitions to compute dCor between.

    Example:
        >>> loss_fn = DistanceCorrelationLoss()
        >>> partitions = {"vol": torch.randn(100, 24),
        ...               "loc": torch.randn(100, 8),
        ...               "shape": torch.randn(100, 12)}
        >>> mean_dcor, details = loss_fn(partitions)
    """

    def __init__(
        self,
        partition_names: list[str] = ("vol", "loc", "shape"),
    ) -> None:
        super().__init__()
        self.partition_names = list(partition_names)

    def forward(
        self,
        partitions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute mean distance correlation across partition pairs.

        Args:
            partitions: Dict of partition tensors {name: [B, dim]}.

        Returns:
            Tuple of (mean dCor, dict of per-pair dCor values).
        """
        device = next(iter(partitions.values())).device
        total_dcor = torch.tensor(0.0, device=device)
        details = {}
        n_pairs = 0

        for name_i, name_j in combinations(self.partition_names, 2):
            dcor = distance_correlation(partitions[name_i], partitions[name_j])
            pair_key = f"dcor_{name_i}_{name_j}"
            details[pair_key] = dcor.detach()
            total_dcor = total_dcor + dcor
            n_pairs += 1

        if n_pairs > 0:
            total_dcor = total_dcor / n_pairs

        return total_dcor, details
