#!/usr/bin/env python
# experiments/lora_ablation/model_factory.py
"""Unified model factory for LoRA ablation experiment.

This module provides a single entry point for creating models with either:
- Lightweight decoder (SegmentationHead, ~2M params) - original v1 approach
- Original decoder (SwinUNETR decoder, ~30M params) - v2 approach with pretrained weights

The decoder type is controlled by the `decoder_type` config parameter:
- "lightweight": Uses custom SegmentationHead (v1 behavior)
- "original": Uses full SwinUNETR decoder with pretrained weights (v2 behavior, recommended)

Usage:
    from experiments.lora_ablation.model_factory import create_ablation_model

    model = create_ablation_model(
        condition_config={"lora_rank": 8, "lora_alpha": 16},
        training_config={"decoder_type": "original", "freeze_decoder": False},
        checkpoint_path="/path/to/checkpoint.pt",
        device="cuda",
    )
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_swin_encoder, load_full_swinunetr
from growth.models.segmentation.seg_head import SegmentationHead, LoRASegmentationModel
from growth.models.segmentation.original_decoder import (
    LoRAOriginalDecoderModel,
    OriginalDecoderSegmentationModel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Lightweight Decoder Models (v1 approach)
# =============================================================================

class BaselineSegmentationModel(nn.Module):
    """Frozen encoder with trainable lightweight segmentation head.

    Used for baseline condition (no LoRA) with lightweight decoder.
    This is the v1 approach with ~2M trainable parameters in the decoder.

    Note: BraTS has 4 classes (0=background, 1=NCR, 2=ED, 3=ET),
    so out_channels must be 4 for proper one-hot encoding in the loss.
    """

    def __init__(self, encoder: nn.Module, out_channels: int = 4):
        super().__init__()
        self.encoder = encoder
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.decoder = SegmentationHead(out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden_states = self.encoder.swinViT(x, self.encoder.normalize)
        return self.decoder(hidden_states)

    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get encoder hidden states (for feature extraction)."""
        with torch.no_grad():
            return self.encoder.swinViT(x, self.encoder.normalize)

    def get_trainable_param_count(self) -> dict:
        return {
            "encoder": 0,
            "decoder": self.decoder.get_param_count(),
            "total": self.decoder.get_param_count(),
        }


# =============================================================================
# Original Decoder Models (v2 approach)
# =============================================================================

class BaselineOriginalDecoderModel(nn.Module):
    """Frozen encoder with original SwinUNETR decoder (pretrained weights).

    Used for baseline condition with the full decoder capacity.
    This is the v2 approach with ~30M trainable parameters in the decoder.

    IMPORTANT: Uses load_full_swinunetr to load pretrained decoder weights.
    """

    def __init__(self, full_model: nn.Module, out_channels: int = 4):
        super().__init__()

        # full_model already has pretrained decoder weights from load_full_swinunetr
        # Wrap in OriginalDecoderSegmentationModel
        self.model = OriginalDecoderSegmentationModel(
            encoder=full_model,
            freeze_decoder=False,  # Train decoder (fine-tune on meningiomas)
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.model.get_hidden_states(x)

    def get_trainable_param_count(self) -> dict:
        return self.model.get_trainable_param_count()


# =============================================================================
# Model Factory
# =============================================================================

def create_ablation_model(
    condition_config: dict,
    training_config: dict,
    checkpoint_path: str,
    device: str = "cuda",
) -> nn.Module:
    """Create model based on decoder_type and condition configuration.

    This is the unified factory function that routes model creation based on
    the decoder_type parameter, supporting both v1 (lightweight) and v2 (original)
    decoder architectures.

    Args:
        condition_config: Condition config dict with 'lora_rank' and 'lora_alpha' keys.
        training_config: Training config with 'decoder_type', 'freeze_decoder',
                        'use_semantic_heads' keys.
        checkpoint_path: Path to encoder/model checkpoint.
        device: Device to load model to.

    Returns:
        nn.Module: Either lightweight or original decoder model, with or without LoRA.

    Configuration options:
        decoder_type: "lightweight" | "original" (default: "original")
            - "lightweight": Uses SegmentationHead (~2M params), v1 behavior
            - "original": Uses SwinUNETR decoder (~30M params), v2 behavior

        freeze_decoder: bool (default: False)
            - Only applies to decoder_type="original"
            - If True, decoder weights are frozen

        use_semantic_heads: bool (default: False)
            - Only applies to decoder_type="original" with LoRA
            - If True, adds auxiliary semantic prediction heads
    """
    decoder_type = training_config.get("decoder_type", "original")
    lora_rank = condition_config.get("lora_rank")

    if decoder_type == "lightweight":
        return _create_lightweight_model(
            condition_config=condition_config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    elif decoder_type == "original":
        return _create_original_decoder_model(
            condition_config=condition_config,
            training_config=training_config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown decoder_type: {decoder_type}. "
            f"Must be 'lightweight' or 'original'."
        )


def _create_lightweight_model(
    condition_config: dict,
    checkpoint_path: str,
    device: str,
) -> nn.Module:
    """Create model with lightweight SegmentationHead decoder (v1 approach).

    Args:
        condition_config: Condition config dict with 'lora_rank' key.
        checkpoint_path: Path to encoder checkpoint.
        device: Device to load model to.

    Returns:
        Either BaselineSegmentationModel or LoRASegmentationModel.
    """
    lora_rank = condition_config.get("lora_rank")

    if lora_rank is None:
        # Baseline: frozen encoder + lightweight decoder
        logger.info("Creating baseline model with LIGHTWEIGHT decoder")
        encoder = load_swin_encoder(
            checkpoint_path,
            freeze=True,
            device=device,
        )
        model = BaselineSegmentationModel(encoder)
    else:
        # LoRA: frozen encoder + LoRA adapters + lightweight decoder
        lora_alpha = condition_config.get("lora_alpha", lora_rank * 2)
        logger.info(
            f"Creating LoRA model with LIGHTWEIGHT decoder "
            f"(rank={lora_rank}, alpha={lora_alpha})"
        )

        # Load base encoder and create LoRA wrapper
        from growth.models.encoder.lora_adapter import create_lora_encoder

        lora_encoder = create_lora_encoder(
            checkpoint_path,
            rank=lora_rank,
            alpha=lora_alpha,
            device=device,
        )
        model = LoRASegmentationModel(lora_encoder)

    model = model.to(device)
    return model


def _create_original_decoder_model(
    condition_config: dict,
    training_config: dict,
    checkpoint_path: str,
    device: str,
) -> nn.Module:
    """Create model with original SwinUNETR decoder (v2 approach).

    IMPORTANT: Uses load_full_swinunetr() to load ALL pretrained weights
    including the decoder, which is essential for good performance (~0.85 Dice).

    Args:
        condition_config: Condition config dict.
        training_config: Training config with freeze_decoder, use_semantic_heads.
        checkpoint_path: Path to BrainSegFounder checkpoint.
        device: Device to load model to.

    Returns:
        Model with original decoder architecture and pretrained weights.
    """
    lora_rank = condition_config.get("lora_rank")
    freeze_decoder = training_config.get("freeze_decoder", False)
    use_semantic_heads = training_config.get("use_semantic_heads", False)

    if lora_rank is None:
        # Baseline: frozen encoder + trainable original decoder (pretrained)
        logger.info("Creating baseline model with ORIGINAL decoder (pretrained weights)")
        full_model = load_full_swinunetr(
            checkpoint_path,
            freeze_encoder=True,   # Freeze swinViT
            freeze_decoder=False,  # Train decoder (fine-tune for meningiomas)
            out_channels=4,
            device=device,
        )
        model = BaselineOriginalDecoderModel(full_model, out_channels=4)
    else:
        # LoRA: frozen encoder + LoRA adapters + pretrained original decoder
        lora_alpha = condition_config.get("lora_alpha", lora_rank * 2)
        logger.info(
            f"Creating LoRA model with ORIGINAL decoder "
            f"(rank={lora_rank}, pretrained weights)"
        )

        # Load full model with pretrained decoder weights
        full_model = load_full_swinunetr(
            checkpoint_path,
            freeze_encoder=True,   # Will be unfrozen for LoRA layers
            freeze_decoder=freeze_decoder,
            out_channels=4,
            device=device,
        )

        # Wrap with LoRA adapters
        lora_encoder = LoRASwinViT(
            full_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=0.1,
            target_stages=[3, 4],
        )

        model = LoRAOriginalDecoderModel(
            lora_encoder=lora_encoder,
            freeze_decoder=freeze_decoder,
            out_channels=4,
            use_semantic_heads=use_semantic_heads,
        )

    model = model.to(device)
    return model


def get_condition_config(config: dict, condition_name: str) -> dict:
    """Get configuration for a specific condition.

    Args:
        config: Full experiment configuration.
        condition_name: Name of condition to find.

    Returns:
        Condition configuration dict.

    Raises:
        ValueError: If condition not found.
    """
    for cond in config["conditions"]:
        if cond["name"] == condition_name:
            return cond
    raise ValueError(
        f"Unknown condition: {condition_name}. "
        f"Available: {[c['name'] for c in config['conditions']]}"
    )
