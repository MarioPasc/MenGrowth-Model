# src/growth/models/projection/semantic_heads.py
"""Semantic prediction heads for SDP.

Lightweight linear projections that map latent partitions to semantic targets:
- pi_vol: z_vol (24) -> volumes (4)
- pi_loc: z_loc (8) -> centroid (3)
- pi_shape: z_shape (12) -> shape features (3: sphericity, surface_area_log, solidity)
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SemanticHeads(nn.Module):
    """Linear projection heads for semantic target prediction.

    Each head is a single linear layer mapping from a latent partition
    to its corresponding semantic target space. Kept linear so that
    the latent partitions themselves must encode useful structure.

    Args:
        vol_in: Input dimension for volume head.
        vol_out: Output dimension for volume head.
        loc_in: Input dimension for location head.
        loc_out: Output dimension for location head.
        shape_in: Input dimension for shape head.
        shape_out: Output dimension for shape head (3 default, 6 for aspect ratios).

    Example:
        >>> heads = SemanticHeads()
        >>> partitions = {"vol": torch.randn(8, 24), "loc": torch.randn(8, 8),
        ...               "shape": torch.randn(8, 12)}
        >>> preds = heads(partitions)
        >>> preds["vol"].shape
        torch.Size([8, 4])
    """

    def __init__(
        self,
        vol_in: int = 24,
        vol_out: int = 4,
        loc_in: int = 8,
        loc_out: int = 3,
        shape_in: int = 12,
        shape_out: int = 3,
    ) -> None:
        super().__init__()

        self.vol_head = nn.Linear(vol_in, vol_out)
        self.loc_head = nn.Linear(loc_in, loc_out)
        self.shape_head = nn.Linear(shape_in, shape_out)

        logger.info(
            f"SemanticHeads: vol({vol_in}->{vol_out}), "
            f"loc({loc_in}->{loc_out}), shape({shape_in}->{shape_out})"
        )

    def forward(self, partitions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict semantic targets from latent partitions.

        Args:
            partitions: Dict with at least "vol", "loc", "shape" tensors.

        Returns:
            Dict with predicted targets for each supervised partition.
        """
        return {
            "vol": self.vol_head(partitions["vol"]),
            "loc": self.loc_head(partitions["loc"]),
            "shape": self.shape_head(partitions["shape"]),
        }
