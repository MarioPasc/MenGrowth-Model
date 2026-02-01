# src/growth/models/segmentation/__init__.py
"""
Segmentation models for Phase 1 encoder adaptation.

Components:
- SegmentationHead: Lightweight decoder (~2M params)
- OriginalDecoderWrapper: Full SwinUNETR decoder (~30M params)
- AuxiliarySemanticHeads: Auxiliary prediction heads for enhanced training
"""

from .seg_head import SegmentationHead, LoRASegmentationModel
from .original_decoder import (
    OriginalDecoderWrapper,
    OriginalDecoderSegmentationModel,
    LoRAOriginalDecoderModel,
    load_original_decoder_model,
)
from .semantic_heads import (
    SemanticHead,
    AuxiliarySemanticHeads,
    AuxiliarySemanticLoss,
    MultiScaleSemanticHeads,
)

__all__ = [
    # Lightweight decoder
    "SegmentationHead",
    "LoRASegmentationModel",
    # Original decoder
    "OriginalDecoderWrapper",
    "OriginalDecoderSegmentationModel",
    "LoRAOriginalDecoderModel",
    "load_original_decoder_model",
    # Semantic heads
    "SemanticHead",
    "AuxiliarySemanticHeads",
    "AuxiliarySemanticLoss",
    "MultiScaleSemanticHeads",
]
