# src/growth/models/__init__.py
"""
Model architectures for the growth forecasting pipeline.

Contains encoder, projection, segmentation, and ODE model definitions.

Components:
- encoder/: SwinViT encoder loading and LoRA adapters
- segmentation/: Segmentation heads and decoder models
- factory: Unified model creation utilities
"""

from .factory import (
    FrozenSwinUNETR,
    CompletelyFrozenModel,
    FrozenEncoderWithDecoder,
    create_swinunetr_model,
    get_model_config,
)

__all__ = [
    # Factory classes
    "FrozenSwinUNETR",
    "CompletelyFrozenModel",
    "FrozenEncoderWithDecoder",
    # Factory functions
    "create_swinunetr_model",
    "get_model_config",
]
