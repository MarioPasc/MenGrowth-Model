# src/growth/models/projection/__init__.py
"""Supervised Disentangled Projection (SDP) components.

The core innovation: lightweight MLP mapping encoder features to disentangled latent space.
"""

from .partition import (
    DEFAULT_PARTITIONS,
    SUPERVISED_PARTITIONS,
    LatentPartition,
    PartitionSpec,
)
from .sdp import SDP, SDPWithHeads
from .semantic_heads import SemanticHeads

__all__ = [
    "SDP",
    "SDPWithHeads",
    "LatentPartition",
    "PartitionSpec",
    "SemanticHeads",
    "DEFAULT_PARTITIONS",
    "SUPERVISED_PARTITIONS",
]
