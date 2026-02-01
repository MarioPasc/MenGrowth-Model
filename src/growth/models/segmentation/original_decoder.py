# src/growth/models/segmentation/original_decoder.py
"""Original SwinUNETR decoder wrapper for Phase 1 LoRA adaptation.

This module provides access to the original MONAI SwinUNETR decoder architecture,
which has significantly more capacity (~30M params) than the lightweight decoder
(~2M params). Using the original decoder provides:

1. Stronger gradient signal to encoder via skip connections at all scales
2. Better segmentation performance (Dice ~0.85 vs ~0.56)
3. More effective LoRA adaptation due to richer supervision

References:
    - Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic
      Segmentation of Brain Tumors in MRI Images." BrainLes 2021.
    - Cox et al. "BrainFounder: Towards Brain Foundation Models for
      Neuroimage Analysis." Medical Image Analysis, 2024.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import SwinUNETR
except ImportError:
    raise ImportError("MONAI is required. Install with: pip install monai>=1.3.0")

logger = logging.getLogger(__name__)


class OriginalDecoderWrapper(nn.Module):
    """Wrapper to extract and use the original SwinUNETR decoder.

    This class wraps a full SwinUNETR model but only exposes the decoder
    components, allowing the encoder to be replaced with a LoRA-adapted version.

    Args:
        base_model: Full SwinUNETR model with pre-trained weights.
        freeze_decoder: If True, freeze all decoder parameters.

    Attributes:
        encoder_blocks: List of encoder processing blocks (encoder1-4, encoder10).
        decoder_blocks: List of decoder blocks (decoder1-5).
        out: Final output convolution.
    """

    def __init__(
        self,
        base_model: SwinUNETR,
        freeze_decoder: bool = False,
    ):
        super().__init__()

        # Store reference to full model components
        # Encoder processing blocks (process swinViT outputs)
        self.encoder1 = base_model.encoder1
        self.encoder2 = base_model.encoder2
        self.encoder3 = base_model.encoder3
        self.encoder4 = base_model.encoder4
        self.encoder10 = base_model.encoder10

        # Decoder blocks with skip connections
        self.decoder5 = base_model.decoder5
        self.decoder4 = base_model.decoder4
        self.decoder3 = base_model.decoder3
        self.decoder2 = base_model.decoder2
        self.decoder1 = base_model.decoder1

        # Final output
        self.out = base_model.out

        if freeze_decoder:
            self._freeze_decoder()
            logger.info("Decoder parameters frozen")

        # Log parameter counts
        dec_params = self._count_decoder_params()
        logger.info(f"Original decoder initialized: {dec_params/1e6:.2f}M params")

    def _freeze_decoder(self):
        """Freeze all decoder parameters."""
        decoder_modules = [
            self.decoder5, self.decoder4, self.decoder3,
            self.decoder2, self.decoder1, self.out
        ]
        for module in decoder_modules:
            for param in module.parameters():
                param.requires_grad = False

    def _count_decoder_params(self) -> int:
        """Count decoder parameters (excluding encoder blocks)."""
        decoder_modules = [
            self.decoder5, self.decoder4, self.decoder3,
            self.decoder2, self.decoder1, self.out
        ]
        return sum(
            p.numel() for m in decoder_modules
            for p in m.parameters()
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through original decoder.

        Args:
            hidden_states: List of 5 tensors from SwinViT:
                - hidden_states[0]: [B, 48, 48, 48, 48] after patch_embed
                - hidden_states[1]: [B, 96, 24, 24, 24] after layers1
                - hidden_states[2]: [B, 192, 12, 12, 12] after layers2
                - hidden_states[3]: [B, 384, 6, 6, 6] after layers3
                - hidden_states[4]: [B, 768, 3, 3, 3] after layers4

        Returns:
            Segmentation logits [B, out_channels, 96, 96, 96].
        """
        # Process encoder outputs through encoder blocks
        enc0 = self.encoder1(hidden_states[0])  # [B, 48, 48, 48, 48]
        enc1 = self.encoder2(hidden_states[1])  # [B, 96, 24, 24, 24]
        enc2 = self.encoder3(hidden_states[2])  # [B, 192, 12, 12, 12]
        enc3 = self.encoder4(hidden_states[3])  # [B, 384, 6, 6, 6]

        # Bottleneck
        dec4 = self.encoder10(hidden_states[4])  # [B, 768, 3, 3, 3]

        # Decoder with skip connections
        dec3 = self.decoder5(dec4, enc3)  # [B, 384, 6, 6, 6]
        dec2 = self.decoder4(dec3, enc2)  # [B, 192, 12, 12, 12]
        dec1 = self.decoder3(dec2, enc1)  # [B, 96, 24, 24, 24]
        dec0 = self.decoder2(dec1, enc0)  # [B, 48, 48, 48, 48]

        # Final upsampling and output
        out = self.decoder1(dec0)  # [B, 48, 96, 96, 96]
        logits = self.out(out)     # [B, out_channels, 96, 96, 96]

        return logits

    def get_bottleneck_features(
        self,
        hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """Extract 768-dim bottleneck features.

        Args:
            hidden_states: List of hidden states from swinViT.

        Returns:
            Bottleneck features [B, 768] after GAP.
        """
        dec4 = self.encoder10(hidden_states[4])
        features = F.adaptive_avg_pool3d(dec4, 1).flatten(1)
        return features

    def get_param_count(self) -> int:
        """Get total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OriginalDecoderSegmentationModel(nn.Module):
    """Full segmentation model with original SwinUNETR decoder.

    Combines LoRA-adapted (or frozen) encoder with the original MONAI
    decoder for maximum segmentation quality and gradient flow.

    Args:
        encoder: SwinUNETR model (used only for swinViT and normalize).
        freeze_decoder: If True, freeze decoder weights.
        out_channels: Number of segmentation classes (4 for BraTS).

    Example:
        >>> encoder = load_swin_encoder(ckpt_path, freeze=False)
        >>> model = OriginalDecoderSegmentationModel(encoder)
        >>> x = torch.randn(1, 4, 96, 96, 96)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 4, 96, 96, 96])
    """

    def __init__(
        self,
        encoder: SwinUNETR,
        freeze_decoder: bool = False,
        out_channels: int = 4,
    ):
        super().__init__()

        # Store the full encoder (we'll use its swinViT)
        self.encoder = encoder

        # Extract original decoder from the same model
        self.decoder = OriginalDecoderWrapper(encoder, freeze_decoder=freeze_decoder)

        # If output channels differ from model, replace out layer
        if out_channels != encoder.out.conv.conv.out_channels:
            in_channels = encoder.out.conv.conv.in_channels
            self.decoder.out = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
            )
            logger.info(f"Replaced output layer: {in_channels} -> {out_channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder.

        Args:
            x: Input tensor [B, 4, 96, 96, 96].

        Returns:
            Segmentation logits [B, out_channels, 96, 96, 96].
        """
        # Get hidden states from swinViT
        hidden_states = self.encoder.swinViT(x, self.encoder.normalize)

        # Decode
        return self.decoder(hidden_states)

    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get encoder hidden states for feature extraction."""
        return self.encoder.swinViT(x, self.encoder.normalize)

    def get_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get 768-dim bottleneck features."""
        hidden_states = self.get_hidden_states(x)
        return self.decoder.get_bottleneck_features(hidden_states)

    def get_trainable_param_count(self) -> Dict[str, int]:
        """Count trainable parameters by component."""
        enc_trainable = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        dec_trainable = self.decoder.get_param_count()
        return {
            "encoder": enc_trainable,
            "decoder": dec_trainable,
            "total": enc_trainable + dec_trainable,
        }


class LoRAOriginalDecoderModel(nn.Module):
    """LoRA-adapted encoder with original SwinUNETR decoder.

    This is the recommended model for Phase 1 LoRA adaptation, combining:
    1. LoRA adapters on encoder stages 3-4
    2. Original MONAI decoder with skip connections
    3. Optional auxiliary semantic prediction heads

    Args:
        lora_encoder: LoRASwinViT instance with adapters.
        freeze_decoder: If True, freeze decoder weights.
        out_channels: Number of segmentation classes.
        use_semantic_heads: If True, add auxiliary semantic prediction heads.

    Example:
        >>> from growth.models.encoder.lora_adapter import create_lora_encoder
        >>> lora_enc = create_lora_encoder(ckpt_path, rank=8)
        >>> model = LoRAOriginalDecoderModel(lora_enc, use_semantic_heads=True)
    """

    def __init__(
        self,
        lora_encoder: nn.Module,
        freeze_decoder: bool = False,
        out_channels: int = 4,
        use_semantic_heads: bool = False,
    ):
        super().__init__()

        self.lora_encoder = lora_encoder
        self.use_semantic_heads = use_semantic_heads

        # Get the underlying SwinUNETR model from PEFT wrapper
        base_model = lora_encoder.model

        # Create decoder wrapper
        self.decoder = OriginalDecoderWrapper(base_model, freeze_decoder=freeze_decoder)

        # Replace output layer if needed
        if out_channels != base_model.out.conv.conv.out_channels:
            in_channels = base_model.out.conv.conv.in_channels
            self.decoder.out = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
            )

        # Optional semantic prediction heads
        if use_semantic_heads:
            from .semantic_heads import AuxiliarySemanticHeads
            self.semantic_heads = AuxiliarySemanticHeads(
                input_dim=768,
                volume_dim=4,
                location_dim=3,
                shape_dim=6,
            )
        else:
            self.semantic_heads = None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through LoRA encoder and original decoder.

        Args:
            x: Input tensor [B, 4, 96, 96, 96].
            return_features: If True, also return bottleneck features.

        Returns:
            If return_features=False: Segmentation logits [B, 4, 96, 96, 96].
            If return_features=True: (logits, features [B, 768]).
        """
        # Get hidden states through LoRA-adapted encoder
        hidden_states = self.lora_encoder.get_hidden_states(x)

        # Decode
        logits = self.decoder(hidden_states)

        if return_features:
            features = self.decoder.get_bottleneck_features(hidden_states)
            return logits, features

        return logits

    def forward_with_semantics(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with semantic predictions.

        Args:
            x: Input tensor [B, 4, 96, 96, 96].

        Returns:
            Dict with:
                - 'logits': Segmentation logits [B, 4, 96, 96, 96]
                - 'features': Bottleneck features [B, 768]
                - 'pred_volume': Volume predictions [B, 4] (if semantic_heads)
                - 'pred_location': Location predictions [B, 3] (if semantic_heads)
                - 'pred_shape': Shape predictions [B, 6] (if semantic_heads)
        """
        # Get hidden states
        hidden_states = self.lora_encoder.get_hidden_states(x)

        # Decode
        logits = self.decoder(hidden_states)
        features = self.decoder.get_bottleneck_features(hidden_states)

        result = {
            'logits': logits,
            'features': features,
        }

        # Semantic predictions
        if self.semantic_heads is not None:
            semantic_preds = self.semantic_heads(features)
            result.update(semantic_preds)

        return result

    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get encoder hidden states."""
        return self.lora_encoder.get_hidden_states(x)

    def get_encoder_params(self):
        """Get encoder (LoRA) parameters for separate optimizer group."""
        return [p for p in self.lora_encoder.model.parameters() if p.requires_grad]

    def get_decoder_params(self):
        """Get decoder parameters for separate optimizer group."""
        params = list(self.decoder.parameters())
        if self.semantic_heads is not None:
            params.extend(self.semantic_heads.parameters())
        return [p for p in params if p.requires_grad]

    def get_trainable_param_count(self) -> Dict[str, int]:
        """Count trainable parameters by component."""
        enc_trainable = sum(
            p.numel() for p in self.lora_encoder.model.parameters() if p.requires_grad
        )
        dec_trainable = sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        )
        sem_trainable = 0
        if self.semantic_heads is not None:
            sem_trainable = sum(
                p.numel() for p in self.semantic_heads.parameters() if p.requires_grad
            )
        return {
            "encoder_lora": enc_trainable,
            "decoder": dec_trainable,
            "semantic_heads": sem_trainable,
            "total": enc_trainable + dec_trainable + sem_trainable,
        }


def load_original_decoder_model(
    checkpoint_path: str,
    freeze_encoder: bool = True,
    freeze_decoder: bool = False,
    out_channels: int = 4,
    device: str = "cuda",
) -> OriginalDecoderSegmentationModel:
    """Load model with original decoder from checkpoint.

    Args:
        checkpoint_path: Path to BrainSegFounder checkpoint.
        freeze_encoder: If True, freeze encoder weights.
        freeze_decoder: If True, freeze decoder weights.
        out_channels: Number of output classes.
        device: Device to load model to.

    Returns:
        OriginalDecoderSegmentationModel instance.
    """
    from growth.models.encoder.swin_loader import load_swin_encoder

    # Load full model (including decoder weights)
    encoder = load_swin_encoder(
        checkpoint_path,
        freeze=freeze_encoder,
        device=device,
    )

    model = OriginalDecoderSegmentationModel(
        encoder,
        freeze_decoder=freeze_decoder,
        out_channels=out_channels,
    )

    return model.to(device)
