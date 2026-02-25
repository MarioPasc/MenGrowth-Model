# src/growth/models/projection/sdp.py
"""Supervised Disentangled Projection (SDP) network.

2-layer MLP with spectral normalization: 768 -> 512 -> 128.
Maps foundation model features to partitioned latent space.
"""

import logging

import torch
from torch import nn

from .partition import LatentPartition
from .semantic_heads import SemanticHeads

logger = logging.getLogger(__name__)


class SDP(nn.Module):
    """2-layer spectrally-normalized MLP for latent projection.

    Architecture:
        LayerNorm(768) -> SN(Linear(768, 512)) -> GELU -> Dropout(0.1)
        -> SN(Linear(512, 128)) -> z [B, 128]

    Spectral normalization on ALL linear layers bounds the Lipschitz
    constant, stabilizing training and preventing representation collapse.

    Args:
        in_dim: Input feature dimension (encoder output).
        hidden_dim: Hidden layer dimension.
        out_dim: Output latent space dimension.
        dropout: Dropout probability after activation.

    Example:
        >>> sdp = SDP()
        >>> h = torch.randn(800, 768)
        >>> z = sdp(h)
        >>> z.shape
        torch.Size([800, 128])
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, out_dim))

        logger.info(
            f"SDP initialized: {in_dim} -> {hidden_dim} -> {out_dim}, "
            f"dropout={dropout}, spectral_norm=all"
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project encoder features to latent space.

        Args:
            h: Encoder features of shape [B, in_dim].

        Returns:
            Latent vector of shape [B, out_dim].
        """
        assert h.dim() == 2, f"Expected 2D input, got {h.dim()}D"
        assert h.shape[1] == self.in_dim, f"Expected h.shape[1]={self.in_dim}, got {h.shape[1]}"

        z = self.norm(h)
        z = self.fc1(z)
        z = self.act(z)
        z = self.drop(z)
        z = self.fc2(z)

        assert z.shape[1] == self.out_dim, (
            f"Output shape mismatch: expected {self.out_dim}, got {z.shape[1]}"
        )
        return z


class SDPWithHeads(nn.Module):
    """SDP network bundled with partition splitting and semantic heads.

    Combines the projection MLP, latent partition, and semantic prediction
    heads into a single forward pass.

    Args:
        sdp: SDP projection network.
        partition: Latent space partition specification.
        heads: Semantic prediction heads.

    Example:
        >>> model = SDPWithHeads.from_config()
        >>> h = torch.randn(800, 768)
        >>> z, parts, preds = model(h)
        >>> z.shape
        torch.Size([800, 128])
        >>> preds["vol"].shape
        torch.Size([800, 4])
    """

    def __init__(
        self,
        sdp: SDP,
        partition: LatentPartition,
        heads: SemanticHeads,
    ) -> None:
        super().__init__()
        self.sdp = sdp
        self.partition = partition
        self.heads = heads

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Full forward pass: project, partition, predict.

        Args:
            h: Encoder features [B, 768].

        Returns:
            Tuple of:
                - z: Full latent vector [B, 128]
                - partitions: Dict of partition tensors {name: [B, dim]}
                - predictions: Dict of semantic predictions {name: [B, target_dim]}
        """
        z = self.sdp(h)
        partitions = self.partition.split(z)
        predictions = self.heads(partitions)
        return z, partitions, predictions

    @classmethod
    def from_config(
        cls,
        in_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout: float = 0.1,
        vol_dim: int = 24,
        loc_dim: int = 8,
        shape_dim: int = 12,
        residual_dim: int = 84,
        n_vol: int = 4,
        n_loc: int = 3,
        n_shape: int = 3,
    ) -> "SDPWithHeads":
        """Construct from hyperparameters.

        Args:
            in_dim: Encoder feature dimension.
            hidden_dim: SDP hidden layer dimension.
            out_dim: Total latent dimension.
            dropout: SDP dropout rate.
            vol_dim: Latent dims for volume partition.
            loc_dim: Latent dims for location partition.
            shape_dim: Latent dims for shape partition.
            residual_dim: Latent dims for residual partition.
            n_vol: Volume target dimensionality.
            n_loc: Location target dimensionality.
            n_shape: Shape target dimensionality.

        Returns:
            Configured SDPWithHeads instance.
        """
        sdp = SDP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
        partition = LatentPartition.from_config(
            vol_dim=vol_dim,
            loc_dim=loc_dim,
            shape_dim=shape_dim,
            residual_dim=residual_dim,
            n_vol=n_vol,
            n_loc=n_loc,
            n_shape=n_shape,
        )
        heads = SemanticHeads(
            vol_in=vol_dim,
            vol_out=n_vol,
            loc_in=loc_dim,
            loc_out=n_loc,
            shape_in=shape_dim,
            shape_out=n_shape,
        )
        return cls(sdp=sdp, partition=partition, heads=heads)
