"""
This module implements a 3D convolutional decoder, suitable for VAEs.
It supports both traditional transposed convolutions and the more modern
"resize-conv" upsampling method to prevent checkerboard artifacts.
"""

import torch
import torch.nn as nn
from typing import Tuple

from .basic import get_activation, get_norm


class Decoder3D(nn.Module):
    """3D transposed-convolution or resize-convolution decoder for VAEs.

    A standard VAE decoder that takes a latent vector `z` and decodes it
    into a 3D volume. It is designed to be roughly symmetric to the `Encoder3D`.

    Architecture:
        - Linear: z_dim -> (base_filters*8) * 4^3
        - Reshape to [B, base_filters*8, 4, 4, 4]
        - 5x upsample blocks to reach 128x128x128
        - Final conv to output channels

    Reference on upsampling methods:
        Odena et al., "Deconvolution and Checkerboard Artifacts" (2016).
        https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(
        self,
        z_dim: int = 128,
        output_channels: int = 4,
        base_filters: int = 32,
        num_groups: int = 8,
        init_method: str = "kaiming",
        activation: str = "relu",
        norm_type: str = "group",
        upsample_mode: str = "resize_conv",
    ):
        """Initialize Decoder3D.

        Args:
            z_dim: Latent space dimensionality.
            output_channels: Number of output channels (e.g., 4 for MRI modalities).
            base_filters: Base number of filters, defines model capacity.
            num_groups: Number of groups for GroupNorm.
            init_method: Weight initialization method.
            activation: Activation function name.
            norm_type: Normalization type name.
            upsample_mode: Upsampling method. Options:
                - "resize_conv": Upsample -> Conv3d (avoids checkerboard artifacts).
                - "deconv": ConvTranspose3d (legacy).
        """
        super().__init__()

        self.base_filters = base_filters
        self.initial_size = 4  # Start from 4^3 spatial size
        self.init_method = init_method
        self.activation_name = activation
        self.norm_type = norm_type
        self.num_groups = num_groups
        self.upsample_mode = upsample_mode

        # Project latent to initial volume
        initial_channels = base_filters * 8
        self.fc = nn.Linear(z_dim, initial_channels * self.initial_size ** 3)

        # Upsample blocks: 4->8->16->32->64->128
        self.up1 = self._make_upsample_block(
            base_filters * 8, base_filters * 8
        )  # 4 -> 8
        self.up2 = self._make_upsample_block(
            base_filters * 8, base_filters * 4
        )  # 8 -> 16
        self.up3 = self._make_upsample_block(
            base_filters * 4, base_filters * 2
        )  # 16 -> 32
        self.up4 = self._make_upsample_block(
            base_filters * 2, base_filters
        )  # 32 -> 64
        self.up5 = self._make_upsample_block(
            base_filters, base_filters
        )  # 64 -> 128

        # Final convolution to output channels
        self.final_conv = nn.Conv3d(
            base_filters, output_channels, kernel_size=3, padding=1
        )
        # Add a final activation function for the output
        self.final_act = nn.Tanh()


        # Initialize weights
        self._initialize_weights()

    def _make_upsample_block(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Sequential:
        """Create an upsample block."""
        if self.upsample_mode == "resize_conv":
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                get_norm(self.norm_type, out_channels, self.num_groups),
                get_activation(self.activation_name),
            )
        elif self.upsample_mode == "deconv":
            return nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels, out_channels, kernel_size=4,
                    stride=2, padding=1, bias=False
                ),
                get_norm(self.norm_type, out_channels, self.num_groups),
                get_activation(self.activation_name),
            )
        else:
            raise ValueError(f"Invalid upsample_mode: {self.upsample_mode}")

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                if self.init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif self.init_method == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif self.init_method == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector [B, z_dim].

        Returns:
            x_hat: Reconstructed output [B, output_channels, 128, 128, 128].
        """
        batch_size = z.size(0)

        # Project and reshape
        x = self.fc(z)
        x = x.view(batch_size, self.base_filters * 8,
                   self.initial_size, self.initial_size, self.initial_size)

        # Upsample stages
        x = self.up1(x)  # 4 -> 8
        x = self.up2(x)  # 8 -> 16
        x = self.up3(x)  # 16 -> 32
        x = self.up4(x)  # 32 -> 64
        x = self.up5(x)  # 64 -> 128

        # Final conv
        x = self.final_conv(x)
        x_hat = self.final_act(x)


        return x_hat