"""Spatial Broadcast Decoder (SBD) for disentangled 3D VAE.

The SBD provides explicit coordinate information to the decoder, removing
the need for the latent vector z to encode positional information. This
architectural bias encourages z to become translation-invariant, effectively
disentangling position from content.

Reference:
    Watters et al. "Spatial Broadcast Decoder: A Simple Architecture for
    Learning Disentangled Representations in VAEs" (2019)
"""

from typing import Tuple

import torch
import torch.nn as nn

from .basic import maybe_spectral_norm


class SpatialBroadcastDecoder(nn.Module):
    """Spatial Broadcast Decoder for 3D volumes.

    Given latent vector z [B, z_dim], produces decoder input by:
    1. Broadcasting z to spatial grid of fixed resolution
    2. Concatenating fixed coordinate grids (depth, height, width)
    3. Upsampling through convolutional stages to target resolution

    The coordinate grids are registered as buffers and normalized to [-1, 1].
    """

    def __init__(
        self,
        z_dim: int = 128,
        output_channels: int = 4,
        base_filters: int = 32,
        grid_size: Tuple[int, int, int] = (8, 8, 8),
        num_groups: int = 8,
        upsample_mode: str = "resize_conv",
        use_spectral_norm: bool = False,
    ):
        """Initialize SpatialBroadcastDecoder.

        Args:
            z_dim: Dimensionality of latent vector.
            output_channels: Number of output channels (4 for MRI modalities).
            base_filters: Base number of filters for decoder.
            grid_size: Spatial resolution for broadcast grid (D, H, W).
            num_groups: Number of groups for GroupNorm.
            upsample_mode: Upsampling method. Options:
                - "resize_conv": Upsample → Conv3d (fixes checkerboard artifacts)
                - "deconv": ConvTranspose3d (legacy mode)
            use_spectral_norm: If True, apply spectral normalization to conv layers
                               for Lipschitz continuity (required for Neural ODE).
        """
        super().__init__()

        self.z_dim = z_dim
        self.output_channels = output_channels
        self.base_filters = base_filters
        self.grid_size = grid_size
        self.num_groups = num_groups
        self.upsample_mode = upsample_mode
        self.use_spectral_norm = use_spectral_norm

        # Register coordinate grids as buffer (not trainable)
        coords = self._create_coordinate_grid(grid_size)
        self.register_buffer("coords", coords)  # Shape: [1, 3, D, H, W]

        # Input channels: z_dim + 3 coordinate channels
        input_channels = z_dim + 3  # 128 + 3 = 131

        # Initial projection from broadcast input to base channels
        # 131 -> base_filters * 8 = 256 (for base_filters=32)
        initial_channels = base_filters * 8
        self.initial_conv = nn.Sequential(
            maybe_spectral_norm(
                nn.Conv3d(input_channels, initial_channels, kernel_size=1, bias=False),
                use_spectral_norm=use_spectral_norm,
            ),
            nn.GroupNorm(num_groups, initial_channels),
            nn.ReLU(inplace=True),
        )

        # Upsample stages: 8 -> 16 -> 32 -> 64 -> 128
        # Channel schedule: 256 -> 128 -> 64 -> 32 -> 16
        self.up1 = self._make_upsample_block(
            base_filters * 8, base_filters * 4, num_groups
        )  # 8 -> 16, 256 -> 128
        self.up2 = self._make_upsample_block(
            base_filters * 4, base_filters * 2, num_groups
        )  # 16 -> 32, 128 -> 64
        self.up3 = self._make_upsample_block(
            base_filters * 2, base_filters, num_groups
        )  # 32 -> 64, 64 -> 32
        self.up4 = self._make_upsample_block(
            base_filters, max(base_filters // 2, 8), num_groups
        )  # 64 -> 128, 32 -> 16

        # Final convolution to output channels
        final_in_channels = max(base_filters // 2, 8)
        self.final_conv = maybe_spectral_norm(
            nn.Conv3d(final_in_channels, output_channels, kernel_size=3, padding=1),
            use_spectral_norm=use_spectral_norm,
        )
        # Final activation to bound output to [-1, 1], matching Decoder3D behavior
        self.final_act = nn.Tanh()

        # Initialize weights
        self._initialize_weights()

    def _create_coordinate_grid(
        self, grid_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Create normalized coordinate grids for the broadcast.

        Creates 3 coordinate channels (D, H, W) normalized to [-1, 1].
        Uses indexing='ij' for meshgrid to align with tensor dimensions.

        Args:
            grid_size: Spatial dimensions (D, H, W).

        Returns:
            Coordinate tensor of shape [1, 3, D, H, W].
        """
        d_size, h_size, w_size = grid_size

        # Create 1D coordinate arrays in [-1, 1]
        d_coords = torch.linspace(-1, 1, d_size)
        h_coords = torch.linspace(-1, 1, h_size)
        w_coords = torch.linspace(-1, 1, w_size)

        # Create 3D meshgrid with indexing='ij'
        # This ensures axis order matches tensor dimensions (D, H, W)
        grid_d, grid_h, grid_w = torch.meshgrid(
            d_coords, h_coords, w_coords, indexing='ij'
        )

        # Stack in order [depth, height, width]
        coords = torch.stack([grid_d, grid_h, grid_w], dim=0)  # [3, D, H, W]

        # Add batch dimension
        coords = coords.unsqueeze(0)  # [1, 3, D, H, W]

        return coords

    def _make_upsample_block(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
    ) -> nn.Sequential:
        """Create an upsampling block.

        Supports two modes:
        - "resize_conv": Upsample + Conv3d (eliminates checkerboard artifacts)
        - "deconv": ConvTranspose3d (legacy mode)

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_groups: Number of groups for GroupNorm.

        Returns:
            Sequential block for 2x upsampling.

        Reference:
            Odena et al. "Deconvolution and Checkerboard Artifacts" (2016)
            https://distill.pub/2016/deconv-checkerboard/
        """
        if self.upsample_mode == "resize_conv":
            # Resize-conv: fixes checkerboard artifacts
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                maybe_spectral_norm(
                    nn.Conv3d(
                        in_channels, out_channels, kernel_size=3,
                        padding=1, bias=False
                    ),
                    use_spectral_norm=self.use_spectral_norm,
                ),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True),
            )
        elif self.upsample_mode == "deconv":
            # Legacy ConvTranspose3d
            return nn.Sequential(
                maybe_spectral_norm(
                    nn.ConvTranspose3d(
                        in_channels, out_channels, kernel_size=4,
                        stride=2, padding=1, bias=False
                    ),
                    use_spectral_norm=self.use_spectral_norm,
                ),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(
                f"Invalid upsample_mode: {self.upsample_mode}. "
                "Must be 'resize_conv' or 'deconv'."
            )

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def broadcast_and_concat(self, z: torch.Tensor) -> torch.Tensor:
        """Broadcast latent vector and concatenate with coordinates.

        Args:
            z: Latent vector [B, z_dim].

        Returns:
            Decoder input tensor [B, z_dim + 3, D, H, W].
        """
        batch_size = z.size(0)
        d, h, w = self.grid_size

        # Broadcast z to spatial grid: [B, z_dim] -> [B, z_dim, D, H, W]
        z_tiled = z.view(batch_size, self.z_dim, 1, 1, 1).expand(-1, -1, d, h, w)

        # Expand coords buffer to batch size: [1, 3, D, H, W] -> [B, 3, D, H, W]
        coords_expanded = self.coords.expand(batch_size, -1, -1, -1, -1)

        # Concatenate: [B, z_dim + 3, D, H, W]
        decoder_input = torch.cat([z_tiled, coords_expanded], dim=1)

        return decoder_input

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector [B, z_dim].

        Returns:
            x_hat: Reconstructed output [B, output_channels, 128, 128, 128].
        """
        # Broadcast and concat with coordinates
        x = self.broadcast_and_concat(z)  # [B, 131, 8, 8, 8]

        # Initial projection
        x = self.initial_conv(x)  # [B, 256, 8, 8, 8]

        # Upsample stages
        x = self.up1(x)  # [B, 128, 16, 16, 16]
        x = self.up2(x)  # [B, 64, 32, 32, 32]
        x = self.up3(x)  # [B, 32, 64, 64, 64]
        x = self.up4(x)  # [B, 16, 128, 128, 128]

        # Final conv to output channels with Tanh activation
        x = self.final_conv(x)  # [B, 4, 128, 128, 128]
        x_hat = self.final_act(x)  # Bound output to [-1, 1]

        return x_hat
