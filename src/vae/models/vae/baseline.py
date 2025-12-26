"""Baseline 3D VAE model for multi-modal MRI.

This module implements a 3D Variational Autoencoder with:
- 3D ResNet encoder with GroupNorm (suitable for small batch sizes)
- Diagonal Gaussian posterior q(z|x) = N(mu, diag(exp(logvar)))
- Symmetric transposed-convolution decoder
- Reparameterization trick for gradient flow

Input: [B, 4, 128, 128, 128] (4 MRI modalities)
Output: x_hat [B, 4, 128, 128, 128], mu [B, z_dim], logvar [B, z_dim]
"""

import torch
import torch.nn as nn
from typing import Tuple


class BasicBlock3d(nn.Module):
    """3D Residual block with GroupNorm.

    Implements: y = F(x, {W_i}) + shortcut(x)
    where F is two 3x3x3 convolutions with GroupNorm and ReLU.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_groups: int = 8,
        downsample: nn.Module = None,
    ):
        """Initialize BasicBlock3d.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for first convolution (for downsampling).
            num_groups: Number of groups for GroupNorm.
            downsample: Optional downsample module for skip connection.
        """
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Encoder3D(nn.Module):
    """3D ResNet encoder for VAE.

    Architecture (for 128^3 input):
        - Initial conv: 128 -> 64 (stride 2)
        - Stage 1: 64^3, channels=base_filters
        - Stage 2: 32^3, channels=base_filters*2
        - Stage 3: 16^3, channels=base_filters*4
        - Stage 4: 8^3, channels=base_filters*8
        - AdaptiveAvgPool -> flatten
        - Linear heads for mu and logvar
    """

    def __init__(
        self,
        input_channels: int = 4,
        base_filters: int = 32,
        z_dim: int = 128,
        num_groups: int = 8,
    ):
        """Initialize Encoder3D.

        Args:
            input_channels: Number of input channels (4 for MRI modalities).
            base_filters: Base number of filters (doubled at each stage).
            z_dim: Latent space dimensionality.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()

        self.in_channels = base_filters
        self.num_groups = num_groups

        # Initial convolution: reduce spatial by 2
        self.conv1 = nn.Conv3d(
            input_channels, base_filters, kernel_size=3,
            stride=2, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups, base_filters)
        self.relu = nn.ReLU(inplace=True)

        # ResNet stages (each stage has 2 blocks)
        self.layer1 = self._make_layer(base_filters, 2, stride=1)      # 64^3
        self.layer2 = self._make_layer(base_filters * 2, 2, stride=2)  # 32^3
        self.layer3 = self._make_layer(base_filters * 4, 2, stride=2)  # 16^3
        self.layer4 = self._make_layer(base_filters * 8, 2, stride=2)  # 8^3

        # Global pooling and latent heads
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        final_channels = base_filters * 8
        self.fc_mu = nn.Linear(final_channels, z_dim)
        self.fc_logvar = nn.Linear(final_channels, z_dim)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.GroupNorm(self.num_groups, out_channels),
            )

        layers = []
        layers.append(BasicBlock3d(
            self.in_channels, out_channels, stride,
            self.num_groups, downsample
        ))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock3d(
                out_channels, out_channels,
                num_groups=self.num_groups
            ))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            mu: Mean of posterior [B, z_dim].
            logvar: Log-variance of posterior [B, z_dim].
        """
        # Initial conv
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        # ResNet stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder3D(nn.Module):
    """3D transposed-convolution decoder for VAE.

    Symmetric architecture to encoder:
        - Linear: z_dim -> (base_filters*8) * 4^3
        - Reshape to [B, base_filters*8, 4, 4, 4]
        - 4x upsample blocks using ConvTranspose3d
        - Final conv to output channels
    """

    def __init__(
        self,
        z_dim: int = 128,
        output_channels: int = 4,
        base_filters: int = 32,
        num_groups: int = 8,
    ):
        """Initialize Decoder3D.

        Args:
            z_dim: Latent space dimensionality.
            output_channels: Number of output channels (4 for MRI modalities).
            base_filters: Base number of filters.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()

        self.base_filters = base_filters
        self.initial_size = 4  # Start from 4^3 spatial size

        # Project latent to initial volume
        initial_channels = base_filters * 8
        self.fc = nn.Linear(z_dim, initial_channels * self.initial_size ** 3)

        # Upsample blocks: 4->8->16->32->64->128
        self.up1 = self._make_upsample_block(
            base_filters * 8, base_filters * 8, num_groups
        )  # 4 -> 8
        self.up2 = self._make_upsample_block(
            base_filters * 8, base_filters * 4, num_groups
        )  # 8 -> 16
        self.up3 = self._make_upsample_block(
            base_filters * 4, base_filters * 2, num_groups
        )  # 16 -> 32
        self.up4 = self._make_upsample_block(
            base_filters * 2, base_filters, num_groups
        )  # 32 -> 64
        self.up5 = self._make_upsample_block(
            base_filters, base_filters, num_groups
        )  # 64 -> 128

        # Final convolution to output channels
        self.final_conv = nn.Conv3d(
            base_filters, output_channels, kernel_size=3, padding=1
        )

        # Initialize weights
        self._initialize_weights()

    def _make_upsample_block(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
    ) -> nn.Sequential:
        """Create an upsample block with transposed convolution."""
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        x_hat = self.final_conv(x)

        return x_hat


class BaselineVAE(nn.Module):
    """Baseline 3D Variational Autoencoder.

    Combines encoder and decoder with reparameterization trick for
    end-to-end training via backpropagation.

    Forward signature:
        forward(x) -> (x_hat, mu, logvar)

    Where:
        - x: Input [B, 4, 128, 128, 128]
        - x_hat: Reconstruction [B, 4, 128, 128, 128]
        - mu: Posterior mean [B, z_dim]
        - logvar: Posterior log-variance [B, z_dim]
    """

    def __init__(
        self,
        input_channels: int = 4,
        z_dim: int = 128,
        base_filters: int = 32,
        num_groups: int = 8,
    ):
        """Initialize BaselineVAE.

        Args:
            input_channels: Number of input channels.
            z_dim: Latent space dimensionality.
            base_filters: Base number of filters for encoder/decoder.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()

        self.z_dim = z_dim

        self.encoder = Encoder3D(
            input_channels=input_channels,
            base_filters=base_filters,
            z_dim=z_dim,
            num_groups=num_groups,
        )

        self.decoder = Decoder3D(
            z_dim=z_dim,
            output_channels=input_channels,
            base_filters=base_filters,
            num_groups=num_groups,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for sampling from posterior.

        z = mu + eps * exp(0.5 * logvar), where eps ~ N(0, I)

        Args:
            mu: Mean of posterior [B, z_dim].
            logvar: Log-variance of posterior [B, z_dim].

        Returns:
            z: Sampled latent vector [B, z_dim].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            mu: Posterior mean [B, z_dim].
            logvar: Posterior log-variance [B, z_dim].
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector [B, z_dim].

        Returns:
            x_hat: Reconstruction [B, C, D, H, W].
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            x_hat: Reconstruction [B, C, D, H, W].
            mu: Posterior mean [B, z_dim].
            logvar: Posterior log-variance [B, z_dim].
            z: Sampled latent vector [B, z_dim].
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z
