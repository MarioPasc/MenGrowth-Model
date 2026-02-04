# src/growth/models/factory.py
"""Model factory for creating segmentation models.

Provides unified model creation for SwinUNETR-based segmentation with:
- Frozen encoder variants
- LoRA/DoRA adapter variants
- Original vs lightweight decoder options

Example:
    >>> from growth.models.factory import create_swinunetr_model
    >>> model = create_swinunetr_model(
    ...     checkpoint_path="/path/to/checkpoint.pt",
    ...     lora_rank=8,
    ...     decoder_type="original",
    ...     device="cuda",
    ... )
"""

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FrozenSwinUNETR(nn.Module):
    """Completely frozen SwinUNETR model.

    All parameters (encoder + decoder) are frozen and the model is locked
    in eval mode. Use for inference/evaluation only.

    Args:
        model: Full SwinUNETR model with pretrained weights.
        out_channels: Number of segmentation classes.

    Example:
        >>> from growth.models.encoder.swin_loader import load_full_swinunetr
        >>> full_model = load_full_swinunetr(ckpt_path, freeze_encoder=True,
        ...                                   freeze_decoder=True)
        >>> frozen = FrozenSwinUNETR(full_model)
        >>> assert frozen.get_trainable_param_count()["total"] == 0
    """

    def __init__(self, model: nn.Module, out_channels: int = 3):
        super().__init__()
        self.model = model
        self.out_channels = out_channels

        # Freeze ALL parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Set to eval mode permanently
        self.model.eval()
        self.eval()

        logger.info("FrozenSwinUNETR: ALL parameters frozen, eval mode locked")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen model."""
        return self.model(x)

    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get encoder hidden states (for feature extraction)."""
        with torch.no_grad():
            return self.model.swinViT(x, self.model.normalize)

    def get_trainable_param_count(self) -> Dict[str, int]:
        """Returns zero for all components (completely frozen)."""
        return {
            "encoder": 0,
            "decoder": 0,
            "total": 0,
        }

    def train(self, mode: bool = True) -> "FrozenSwinUNETR":
        """Override train to keep model in eval mode."""
        return super().train(False)


# Backward compatibility alias
CompletelyFrozenModel = FrozenSwinUNETR


class FrozenEncoderWithDecoder(nn.Module):
    """Frozen encoder with trainable decoder.

    The encoder (SwinViT) is frozen while the decoder can be trained.
    Supports both original SwinUNETR decoder and lightweight SegmentationHead.

    Args:
        encoder: SwinViT encoder (will be frozen).
        decoder: Decoder module (original or lightweight).
        out_channels: Number of segmentation classes.

    Example:
        >>> encoder = load_swin_encoder(ckpt_path, freeze=True)
        >>> decoder = SegmentationHead(out_channels=3)
        >>> model = FrozenEncoderWithDecoder(encoder, decoder)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        out_channels: int = 3,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out_channels = out_channels

        # Ensure encoder is frozen
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with frozen encoder."""
        with torch.no_grad():
            hidden_states = self.encoder.swinViT(x, self.encoder.normalize)
        return self.decoder(hidden_states)

    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get encoder hidden states (for feature extraction)."""
        with torch.no_grad():
            return self.encoder.swinViT(x, self.encoder.normalize)

    def get_trainable_param_count(self) -> Dict[str, int]:
        """Count trainable parameters by component."""
        encoder_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        decoder_params = sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        )
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "total": encoder_params + decoder_params,
        }


def create_swinunetr_model(
    checkpoint_path: str,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    decoder_type: str = "original",
    freeze_decoder: bool = False,
    freeze_all: bool = False,
    use_dora: bool = False,
    use_semantic_heads: bool = False,
    out_channels: int = 3,
    device: str = "cuda",
) -> nn.Module:
    """Create SwinUNETR-based segmentation model.

    Unified factory function that supports:
    - Completely frozen model (freeze_all=True)
    - Frozen encoder + trainable decoder (lora_rank=None)
    - LoRA/DoRA adapted encoder + decoder (lora_rank > 0)

    Args:
        checkpoint_path: Path to BrainSegFounder checkpoint.
        lora_rank: LoRA rank. If None, no LoRA adapters are added.
        lora_alpha: LoRA alpha. Defaults to 2 * lora_rank.
        decoder_type: "original" or "lightweight".
            - "original": Full SwinUNETR decoder with pretrained weights.
            - "lightweight": Smaller SegmentationHead decoder.
        freeze_decoder: Whether to freeze decoder weights.
        freeze_all: If True, freeze everything (encoder + decoder).
        use_dora: Use DoRA instead of LoRA.
        use_semantic_heads: Add auxiliary semantic prediction heads.
        out_channels: Number of output classes.
        device: Device to load model to.

    Returns:
        Configured segmentation model.

    Example:
        >>> # Baseline with frozen encoder, trainable original decoder
        >>> model = create_swinunetr_model(
        ...     checkpoint_path="/path/to/ckpt.pt",
        ...     decoder_type="original",
        ... )

        >>> # LoRA-adapted encoder with original decoder
        >>> model = create_swinunetr_model(
        ...     checkpoint_path="/path/to/ckpt.pt",
        ...     lora_rank=8,
        ...     decoder_type="original",
        ... )

        >>> # Completely frozen (inference only)
        >>> model = create_swinunetr_model(
        ...     checkpoint_path="/path/to/ckpt.pt",
        ...     freeze_all=True,
        ... )
    """
    # Import here to avoid circular dependencies
    from growth.models.encoder.swin_loader import load_swin_encoder, load_full_swinunetr

    # Special case: completely frozen model
    if freeze_all:
        logger.info("Creating COMPLETELY FROZEN model (inference only)")
        full_model = load_full_swinunetr(
            checkpoint_path,
            freeze_encoder=True,
            freeze_decoder=True,
            out_channels=out_channels,
            device=device,
        )
        model = FrozenSwinUNETR(full_model, out_channels=out_channels)
        return model.to(device)

    # Set lora_alpha default
    if lora_rank is not None and lora_alpha is None:
        lora_alpha = lora_rank * 2

    if decoder_type == "lightweight":
        return _create_lightweight_model(
            checkpoint_path=checkpoint_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_dora=use_dora,
            out_channels=out_channels,
            device=device,
        )
    elif decoder_type == "original":
        return _create_original_decoder_model(
            checkpoint_path=checkpoint_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_decoder=freeze_decoder,
            use_dora=use_dora,
            use_semantic_heads=use_semantic_heads,
            out_channels=out_channels,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown decoder_type: {decoder_type}. "
            f"Must be 'lightweight' or 'original'."
        )


def _create_lightweight_model(
    checkpoint_path: str,
    lora_rank: Optional[int],
    lora_alpha: Optional[int],
    use_dora: bool,
    out_channels: int,
    device: str,
) -> nn.Module:
    """Create model with lightweight SegmentationHead decoder."""
    from growth.models.encoder.swin_loader import load_swin_encoder
    from growth.models.segmentation.seg_head import SegmentationHead, LoRASegmentationModel

    if lora_rank is None:
        # Baseline: frozen encoder + lightweight decoder
        logger.info("Creating baseline model with LIGHTWEIGHT decoder")
        encoder = load_swin_encoder(
            checkpoint_path,
            freeze=True,
            device=device,
        )
        decoder = SegmentationHead(out_channels=out_channels)
        model = FrozenEncoderWithDecoder(encoder, decoder, out_channels)
    else:
        # LoRA: frozen encoder + LoRA adapters + lightweight decoder
        adapter_type = "DoRA" if use_dora else "LoRA"
        logger.info(
            f"Creating {adapter_type} model with LIGHTWEIGHT decoder "
            f"(rank={lora_rank}, alpha={lora_alpha})"
        )

        from growth.models.encoder.lora_adapter import create_lora_encoder

        lora_encoder = create_lora_encoder(
            checkpoint_path,
            rank=lora_rank,
            alpha=lora_alpha,
            device=device,
        )
        model = LoRASegmentationModel(lora_encoder)

    return model.to(device)


def _create_original_decoder_model(
    checkpoint_path: str,
    lora_rank: Optional[int],
    lora_alpha: Optional[int],
    freeze_decoder: bool,
    use_dora: bool,
    use_semantic_heads: bool,
    out_channels: int,
    device: str,
) -> nn.Module:
    """Create model with original SwinUNETR decoder."""
    from growth.models.encoder.swin_loader import load_full_swinunetr
    from growth.models.encoder.lora_adapter import LoRASwinViT
    from growth.models.segmentation.original_decoder import (
        LoRAOriginalDecoderModel,
        OriginalDecoderSegmentationModel,
    )

    if lora_rank is None:
        # Baseline: frozen encoder + trainable original decoder
        logger.info(
            f"Creating baseline model with ORIGINAL decoder "
            f"(pretrained weights, semantic_heads={use_semantic_heads})"
        )
        full_model = load_full_swinunetr(
            checkpoint_path,
            freeze_encoder=True,
            freeze_decoder=freeze_decoder,
            out_channels=out_channels,
            device=device,
        )

        model = OriginalDecoderSegmentationModel(
            encoder=full_model,
            freeze_decoder=freeze_decoder,
            out_channels=out_channels,
        )

        # Add semantic heads if requested
        if use_semantic_heads:
            from growth.models.segmentation.semantic_heads import AuxiliarySemanticHeads

            model.semantic_heads = AuxiliarySemanticHeads(
                input_dim=768,
                volume_dim=4,
                location_dim=3,
                shape_dim=3,
            )
            logger.info("Baseline model: semantic heads enabled")

    else:
        # LoRA/DoRA: adapted encoder + pretrained original decoder
        adapter_type = "DoRA" if use_dora else "LoRA"
        logger.info(
            f"Creating {adapter_type} model with ORIGINAL decoder "
            f"(rank={lora_rank}, pretrained weights)"
        )

        full_model = load_full_swinunetr(
            checkpoint_path,
            freeze_encoder=True,  # Will be partially unfrozen for LoRA layers
            freeze_decoder=freeze_decoder,
            out_channels=out_channels,
            device=device,
        )

        # Wrap with LoRA/DoRA adapters
        lora_encoder = LoRASwinViT(
            full_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=0.1,
            target_stages=[3, 4],
            use_dora=use_dora,
        )

        model = LoRAOriginalDecoderModel(
            lora_encoder=lora_encoder,
            freeze_decoder=freeze_decoder,
            out_channels=out_channels,
            use_semantic_heads=use_semantic_heads,
        )

    return model.to(device)


def get_model_config(config: dict, model_name: str) -> dict:
    """Get configuration for a specific model/condition.

    Args:
        config: Full experiment configuration.
        model_name: Name of model/condition to find.

    Returns:
        Model configuration dict.

    Raises:
        ValueError: If model/condition not found.
    """
    # Try 'conditions' key first (ablation experiments)
    if "conditions" in config:
        for cond in config["conditions"]:
            if cond.get("name") == model_name:
                return cond

    # Try 'models' key (general experiments)
    if "models" in config:
        for model in config["models"]:
            if model.get("name") == model_name:
                return model

    available = []
    if "conditions" in config:
        available.extend([c["name"] for c in config["conditions"]])
    if "models" in config:
        available.extend([m["name"] for m in config["models"]])

    raise ValueError(
        f"Unknown model/condition: {model_name}. " f"Available: {available}"
    )


__all__ = [
    # Model classes
    "FrozenSwinUNETR",
    "CompletelyFrozenModel",  # Backward compatibility alias
    "FrozenEncoderWithDecoder",
    # Factory functions
    "create_swinunetr_model",
    "get_model_config",
]
