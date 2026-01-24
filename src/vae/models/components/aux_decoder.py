"""Lightweight auxiliary decoder for z_residual -> low-resolution reconstruction.

Reconstructs a 64^3 version of the input using only residual latent dims.
This prevents residual collapse by providing gradient signal that forces
z_residual to encode useful information about the input.

Design choices:
- Target resolution 64^3 (not 160^3): captures low-frequency anatomy
  without competing with the main SBD decoder on fine tumor features.
- base_filters=16 (vs 32 for main decoder): ~4x fewer parameters (~2M total).
- No spectral norm (not needed for auxiliary reconstruction task).
- GroupNorm + ReLU for stable training with small batch sizes.
"""

import torch
import torch.nn as nn


class _UpBlock3D(nn.Module):
    """Upsample 2x + Conv3d + GroupNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 4):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class AuxDecoder3D(nn.Module):
    """Lightweight decoder: z_residual [B, residual_dim] -> [B, C, 64, 64, 64].

    Architecture:
        FC -> Reshape [B, 128, 4, 4, 4]
        UpBlock: 4^3 -> 8^3  (128 -> 64 channels)
        UpBlock: 8^3 -> 16^3 (64 -> 32 channels)
        UpBlock: 16^3 -> 32^3 (32 -> 16 channels)
        UpBlock: 32^3 -> 64^3 (16 -> 16 channels)
        Conv3d: 16 -> output_channels

    Total parameters: ~2M (vs ~15M for main SBD decoder)
    """

    def __init__(
        self,
        z_dim: int = 84,
        output_channels: int = 4,
        base_filters: int = 16,
        num_groups: int = 4,
    ):
        """Initialize auxiliary decoder.

        Args:
            z_dim: Dimension of z_residual input
            output_channels: Number of output channels (4 for MRI modalities)
            base_filters: Base filter count (kept small for lightweight decoder)
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()

        self.z_dim = z_dim
        self.initial_channels = base_filters * 8  # 128
        self.initial_size = 4

        # FC to spatial representation
        spatial_dim = self.initial_channels * (self.initial_size ** 3)
        self.fc = nn.Linear(z_dim, spatial_dim)
        self.fc_norm = nn.GroupNorm(
            num_groups=min(num_groups, self.initial_channels),
            num_channels=self.initial_channels,
        )
        self.fc_act = nn.ReLU(inplace=True)

        # Upsample blocks: 4^3 -> 8^3 -> 16^3 -> 32^3 -> 64^3
        self.up1 = _UpBlock3D(base_filters * 8, base_filters * 4, num_groups)  # 128->64
        self.up2 = _UpBlock3D(base_filters * 4, base_filters * 2, num_groups)  # 64->32
        self.up3 = _UpBlock3D(base_filters * 2, base_filters, num_groups)      # 32->16
        self.up4 = _UpBlock3D(base_filters, base_filters, num_groups)          # 16->16

        # Final conv to output channels
        self.final_conv = nn.Conv3d(base_filters, output_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier normal for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z_residual: torch.Tensor) -> torch.Tensor:
        """Decode from residual latent dims to low-resolution reconstruction.

        Args:
            z_residual: Residual latent vector [B, z_dim]

        Returns:
            Low-resolution reconstruction [B, output_channels, 64, 64, 64]
        """
        # FC to spatial
        h = self.fc(z_residual)
        h = h.view(-1, self.initial_channels, self.initial_size, self.initial_size, self.initial_size)
        h = self.fc_norm(h)
        h = self.fc_act(h)

        # Upsample
        h = self.up1(h)   # [B, 64, 8, 8, 8]
        h = self.up2(h)   # [B, 32, 16, 16, 16]
        h = self.up3(h)   # [B, 16, 32, 32, 32]
        h = self.up4(h)   # [B, 16, 64, 64, 64]

        # Final projection
        return self.final_conv(h)  # [B, C, 64, 64, 64]
