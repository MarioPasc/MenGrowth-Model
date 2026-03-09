# src/growth/models/projection/semantic_heads.py
"""Semantic prediction heads for SDP.

Lightweight linear projection mapping the volume latent partition to
the whole-tumor volume target: pi_vol: z_vol (32) -> log(V_WT + 1) (1).

Methodology Revision R1: location and shape heads removed.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SemanticHeads(nn.Module):
    """Linear projection head for volume target prediction.

    A single linear layer mapping from the volume latent partition
    to the whole-tumor volume. Kept linear so that the latent partition
    itself must encode useful structure.

    Args:
        vol_in: Input dimension for volume head.
        vol_out: Output dimension for volume head (1: log V_WT).

    Example:
        >>> heads = SemanticHeads()
        >>> partitions = {"vol": torch.randn(8, 32)}
        >>> preds = heads(partitions)
        >>> preds["vol"].shape
        torch.Size([8, 1])
    """

    def __init__(
        self,
        vol_in: int = 32,
        vol_out: int = 1,
    ) -> None:
        super().__init__()

        self.vol_head = nn.Linear(vol_in, vol_out)

        logger.info(
            f"SemanticHeads: vol({vol_in}->{vol_out})"
        )

    def forward(self, partitions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict semantic targets from latent partitions.

        Args:
            partitions: Dict with at least "vol" tensor.

        Returns:
            Dict with predicted volume target.
        """
        return {
            "vol": self.vol_head(partitions["vol"]),
        }
