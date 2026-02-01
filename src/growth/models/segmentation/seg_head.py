# src/growth/models/segmentation/seg_head.py
"""Lightweight segmentation head for Phase 1 LoRA adaptation.

This module provides a lightweight decoder for BraTS-style segmentation that
attaches to the SwinViT encoder. It's used during Phase 1 to fine-tune the
encoder with LoRA adapters and is discarded after training.

The decoder is deliberately simple (~2M params) to:
1. Keep the focus on encoder adaptation (not decoder learning)
2. Minimize overfitting during the limited LoRA training phase
3. Allow fast training with limited data

Architecture:
    Takes multi-scale features from SwinViT stages and progressively
    upsamples using transposed convolutions with skip connections.

    SwinViT outputs (for 96^3 input):
        Stage 0: [B, 48, 48, 48, 48]   <- patch embedding
        Stage 1: [B, 96, 24, 24, 24]
        Stage 2: [B, 192, 12, 12, 12]
        Stage 3: [B, 384, 6, 6, 6]
        Stage 4: [B, 768, 3, 3, 3]
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Basic 3D convolution block with normalization and activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        norm: Normalization type ('instance', 'batch', or None).
        activation: Activation type ('relu', 'leaky_relu', 'gelu', or None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: str = "instance",
        activation: str = "leaky_relu",
    ):
        super().__init__()

        padding = kernel_size // 2

        layers = [
            nn.Conv3d(
                in_channels, out_channels, kernel_size, padding=padding, bias=norm is None
            )
        ]

        if norm == "instance":
            layers.append(nn.InstanceNorm3d(out_channels, affine=True))
        elif norm == "batch":
            layers.append(nn.BatchNorm3d(out_channels))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.01, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Upsampling block with transposed convolution and optional skip connection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        skip_channels: Number of channels from skip connection (0 if no skip).
        scale_factor: Upsampling factor (2 for doubling spatial dims).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        scale_factor: int = 2,
    ):
        super().__init__()

        self.skip_channels = skip_channels

        # Transposed convolution for upsampling
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=scale_factor,
            stride=scale_factor,
        )

        # Convolution after concatenation with skip
        conv_in_channels = out_channels + skip_channels
        self.conv = ConvBlock(conv_in_channels, out_channels)

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional skip connection.

        Args:
            x: Input tensor [B, C, D, H, W].
            skip: Skip connection tensor [B, C_skip, D', H', W'].

        Returns:
            Upsampled tensor [B, C_out, D*2, H*2, W*2].
        """
        x = self.upsample(x)

        if skip is not None:
            # Handle size mismatches from odd dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class SegmentationHead(nn.Module):
    """Lightweight segmentation decoder for BraTS-style 3-class segmentation.

    Takes multi-scale features from SwinViT encoder and produces segmentation
    masks for NCR (1), ED (2), and ET (3) classes.

    Args:
        encoder_channels: Channel counts from encoder stages [stage0, ..., stage4].
            Default matches SwinUNETR with feature_size=48.
        out_channels: Number of output segmentation classes.
        use_deep_supervision: If True, return intermediate outputs for deep supervision.

    Example:
        >>> head = SegmentationHead()
        >>> # hidden_states from SwinViT: list of 5 tensors
        >>> hidden_states = [
        ...     torch.randn(1, 48, 48, 48, 48),   # stage 0
        ...     torch.randn(1, 96, 24, 24, 24),   # stage 1
        ...     torch.randn(1, 192, 12, 12, 12),  # stage 2
        ...     torch.randn(1, 384, 6, 6, 6),     # stage 3
        ...     torch.randn(1, 768, 3, 3, 3),     # stage 4
        ... ]
        >>> out = head(hidden_states)
        >>> out.shape
        torch.Size([1, 3, 96, 96, 96])
    """

    def __init__(
        self,
        encoder_channels: Tuple[int, ...] = (48, 96, 192, 384, 768),
        out_channels: int = 3,
        use_deep_supervision: bool = False,
    ):
        super().__init__()

        self.encoder_channels = encoder_channels
        self.out_channels = out_channels
        self.use_deep_supervision = use_deep_supervision

        # Decoder channels (progressively reduce)
        dec_channels = [256, 128, 64, 32]

        # Bottleneck: 768 -> 256
        self.bottleneck = ConvBlock(encoder_channels[4], dec_channels[0])

        # Upsampling blocks (from deepest to shallowest)
        # 256 + 384 (skip from stage 3) -> 128, 3x3x3 -> 6x6x6
        self.up4 = UpsampleBlock(dec_channels[0], dec_channels[1], encoder_channels[3])
        # 128 + 192 (skip from stage 2) -> 64, 6x6x6 -> 12x12x12
        self.up3 = UpsampleBlock(dec_channels[1], dec_channels[2], encoder_channels[2])
        # 64 + 96 (skip from stage 1) -> 32, 12x12x12 -> 24x24x24
        self.up2 = UpsampleBlock(dec_channels[2], dec_channels[3], encoder_channels[1])
        # 32 + 48 (skip from stage 0) -> 32, 24x24x24 -> 48x48x48
        self.up1 = UpsampleBlock(dec_channels[3], dec_channels[3], encoder_channels[0])
        # Final upsample: 32 -> 32, 48x48x48 -> 96x96x96
        self.up0 = nn.ConvTranspose3d(dec_channels[3], dec_channels[3], kernel_size=2, stride=2)

        # Output head
        self.out_conv = nn.Conv3d(dec_channels[3], out_channels, kernel_size=1)

        # Deep supervision heads (if enabled)
        if use_deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(dec_channels[1], out_channels, kernel_size=1),  # 6x6x6
                nn.Conv3d(dec_channels[2], out_channels, kernel_size=1),  # 12x12x12
                nn.Conv3d(dec_channels[3], out_channels, kernel_size=1),  # 24x24x24
            ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, hidden_states: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through segmentation decoder.

        Args:
            hidden_states: List of 5 tensors from SwinViT stages:
                - hidden_states[0]: [B, 48, 48, 48, 48] (stage 0)
                - hidden_states[1]: [B, 96, 24, 24, 24] (stage 1)
                - hidden_states[2]: [B, 192, 12, 12, 12] (stage 2)
                - hidden_states[3]: [B, 384, 6, 6, 6] (stage 3)
                - hidden_states[4]: [B, 768, 3, 3, 3] (stage 4)

        Returns:
            Segmentation logits [B, out_channels, 96, 96, 96].
            If use_deep_supervision=True, returns tuple of (main_output, deep_outputs).
        """
        if len(hidden_states) != 5:
            raise ValueError(f"Expected 5 hidden states, got {len(hidden_states)}")

        # Unpack hidden states
        enc0, enc1, enc2, enc3, enc4 = hidden_states

        # Bottleneck
        x = self.bottleneck(enc4)  # [B, 256, 3, 3, 3]

        # Decoder with skip connections
        x = self.up4(x, enc3)  # [B, 128, 6, 6, 6]
        if self.use_deep_supervision:
            ds4 = self.ds_heads[0](x)

        x = self.up3(x, enc2)  # [B, 64, 12, 12, 12]
        if self.use_deep_supervision:
            ds3 = self.ds_heads[1](x)

        x = self.up2(x, enc1)  # [B, 32, 24, 24, 24]
        if self.use_deep_supervision:
            ds2 = self.ds_heads[2](x)

        x = self.up1(x, enc0)  # [B, 32, 48, 48, 48]
        x = self.up0(x)  # [B, 32, 96, 96, 96]

        # Output
        out = self.out_conv(x)  # [B, 3, 96, 96, 96]

        if self.use_deep_supervision:
            return out, [ds4, ds3, ds2]
        return out

    def get_param_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoRASegmentationModel(nn.Module):
    """Complete model combining LoRA-adapted encoder with segmentation head.

    This is a convenience wrapper for Phase 1 training that combines:
    1. LoRASwinViT encoder (from lora_adapter.py)
    2. SegmentationHead decoder

    Args:
        lora_encoder: LoRASwinViT instance.
        out_channels: Number of segmentation classes.
        use_deep_supervision: Enable deep supervision.

    Example:
        >>> from growth.models.encoder.lora_adapter import create_lora_encoder
        >>> lora_enc = create_lora_encoder(checkpoint_path, rank=8)
        >>> model = LoRASegmentationModel(lora_enc)
        >>> x = torch.randn(1, 4, 96, 96, 96)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 3, 96, 96, 96])
    """

    def __init__(
        self,
        lora_encoder: nn.Module,
        out_channels: int = 3,
        use_deep_supervision: bool = False,
    ):
        super().__init__()

        self.encoder = lora_encoder
        self.decoder = SegmentationHead(
            out_channels=out_channels,
            use_deep_supervision=use_deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder.

        Args:
            x: Input tensor [B, 4, 96, 96, 96].

        Returns:
            Segmentation logits [B, out_channels, 96, 96, 96].
        """
        hidden_states = self.encoder.get_hidden_states(x)
        return self.decoder(hidden_states)

    def get_encoder_params(self):
        """Get encoder (LoRA) parameters for separate optimizer group."""
        return self.encoder.model.parameters()

    def get_decoder_params(self):
        """Get decoder parameters for separate optimizer group."""
        return self.decoder.parameters()

    def get_trainable_param_count(self) -> dict:
        """Count trainable parameters by component."""
        enc_trainable = sum(
            p.numel() for p in self.encoder.model.parameters() if p.requires_grad
        )
        dec_trainable = self.decoder.get_param_count()
        return {
            "encoder_lora": enc_trainable,
            "decoder": dec_trainable,
            "total": enc_trainable + dec_trainable,
        }
