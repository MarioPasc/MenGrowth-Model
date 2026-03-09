# src/growth/models/projection/partition.py
"""Latent space partitioning for Supervised Disentangled Projection.

Defines the partition schema that splits a 128-dim latent vector into
volume and residual subspaces. Methodology Revision R1: location and
shape partitions removed — location is temporally static (GP covariate),
shape R² <= 0.11 (scientifically unjustified).
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PartitionSpec:
    """Specification for a single latent partition.

    Args:
        name: Partition identifier (e.g., "vol", "loc", "shape", "residual").
        start: Start index (inclusive) in the latent vector.
        end: End index (exclusive) in the latent vector.
        target_dim: Dimensionality of the regression target. None for residual.
    """

    name: str
    start: int
    end: int
    target_dim: int | None = None

    @property
    def dim(self) -> int:
        """Number of latent dimensions in this partition."""
        return self.end - self.start


# Default partition layout: 128 dims total
# Methodology Revision R1: vol + residual only
DEFAULT_PARTITIONS: dict[str, PartitionSpec] = {
    "vol": PartitionSpec(name="vol", start=0, end=32, target_dim=1),
    "residual": PartitionSpec(name="residual", start=32, end=128, target_dim=None),
}

# Partitions with supervised regression targets
SUPERVISED_PARTITIONS: list[str] = ["vol"]


class LatentPartition:
    """Splits a latent vector into named partitions.

    Args:
        partitions: Dict mapping partition names to PartitionSpec.
            Defaults to DEFAULT_PARTITIONS.

    Raises:
        ValueError: If partitions are not contiguous or have gaps/overlaps.

    Example:
        >>> lp = LatentPartition()
        >>> z = torch.randn(8, 128)
        >>> parts = lp.split(z)
        >>> parts["vol"].shape
        torch.Size([8, 24])
    """

    def __init__(self, partitions: dict[str, PartitionSpec] | None = None) -> None:
        self.partitions = partitions or DEFAULT_PARTITIONS
        self._validate()

    def _validate(self) -> None:
        """Validate partition contiguity and non-overlap."""
        sorted_parts = sorted(self.partitions.values(), key=lambda p: p.start)

        for i, part in enumerate(sorted_parts):
            if part.end <= part.start:
                raise ValueError(
                    f"Partition '{part.name}' has non-positive width: [{part.start}, {part.end})"
                )
            if i > 0:
                prev = sorted_parts[i - 1]
                if part.start != prev.end:
                    raise ValueError(
                        f"Gap or overlap between '{prev.name}' (end={prev.end}) "
                        f"and '{part.name}' (start={part.start})"
                    )

        if sorted_parts[0].start != 0:
            raise ValueError(f"First partition '{sorted_parts[0].name}' does not start at 0")

    def split(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split latent vector into named partitions.

        Args:
            z: Latent tensor of shape [B, total_dim].

        Returns:
            Dict mapping partition names to sliced tensors.
        """
        assert z.dim() == 2, f"Expected 2D tensor, got {z.dim()}D"
        assert z.shape[1] == self.total_dim, (
            f"Expected z.shape[1]={self.total_dim}, got {z.shape[1]}"
        )

        return {name: z[:, spec.start : spec.end] for name, spec in self.partitions.items()}

    @property
    def total_dim(self) -> int:
        """Total dimensionality of the latent space."""
        return max(spec.end for spec in self.partitions.values())

    @property
    def indices(self) -> dict[str, tuple[int, int]]:
        """Index ranges for each partition. Compatible with latent_quality.py."""
        return {name: (spec.start, spec.end) for name, spec in self.partitions.items()}

    def get_supervised_partitions(self) -> dict[str, PartitionSpec]:
        """Return only partitions that have regression targets."""
        return {name: spec for name, spec in self.partitions.items() if spec.target_dim is not None}

    @classmethod
    def from_config(
        cls,
        vol_dim: int = 32,
        residual_dim: int = 96,
        n_vol: int = 1,
        **kwargs,
    ) -> "LatentPartition":
        """Construct from dimension sizes (for ablations).

        Args:
            vol_dim: Latent dims for volume partition.
            residual_dim: Latent dims for residual partition.
            n_vol: Target dimensionality for volume.
            **kwargs: Ignored (backward compat for old configs with loc/shape).

        Returns:
            Configured LatentPartition instance.
        """
        res_start = vol_dim
        res_end = res_start + residual_dim

        partitions = {
            "vol": PartitionSpec("vol", 0, res_start, n_vol),
            "residual": PartitionSpec("residual", res_start, res_end, None),
        }
        return cls(partitions)
