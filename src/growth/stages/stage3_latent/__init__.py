# src/growth/stages/stage3_latent/__init__.py
"""Stage 3: Representation learning pipeline (TERTIARY).

Facade re-exporting encoder, SDP projection, and latent-space growth models.
This is the old primary pipeline, now the last complexity level to try.

Components:
- Encoder: BrainSegFounder SwinUNETR (frozen or LoRA-adapted)
- Projection: Supervised Disentangled Projection (SDP) 768→128 dims
- Growth models: Same as Stage 1 (ScalarGP, LME, HGP) but on latent features

Spec: ``docs/stages/stage_3_representation_learning.md``
Reference module specs in ``docs/growth-related/claude_files_BSGNeuralODE/``
"""

# Encoder components
from growth.models.encoder.feature_extractor import FeatureExtractor
from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_swin_encoder

# Projection
from growth.models.projection.partition import LatentPartition, PartitionSpec
from growth.models.projection.sdp import SDP, SDPWithHeads
from growth.models.projection.semantic_heads import SemanticHeads

__all__ = [
    # Encoder
    "FeatureExtractor",
    "LoRASwinViT",
    "load_swin_encoder",
    # Projection
    "LatentPartition",
    "PartitionSpec",
    "SDP",
    "SDPWithHeads",
    "SemanticHeads",
]
