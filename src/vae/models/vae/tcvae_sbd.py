"""β-TCVAE with Spatial Broadcast Decoder for Exp2.

This module implements the TCVAE+SBD architecture:
- Same 3D ResNet encoder as Exp1 (with GroupNorm)
- Spatial Broadcast Decoder (SBD) replacing the standard decoder
- Support for gradient checkpointing for memory efficiency

The SBD provides explicit coordinate information to the decoder,
disentangling position from content in the latent space.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .baseline import Encoder3D
from ..components import SpatialBroadcastDecoder


class TCVAESBD(nn.Module):
    """β-TCVAE with Spatial Broadcast Decoder.

    Architecture:
    - Encoder: 3D ResNet with GroupNorm (same as Exp1)
    - Decoder: Spatial Broadcast Decoder with coordinate grids
    - Latent: diagonal Gaussian with reparameterization

    Forward signature:
        forward(x) -> (x_hat, mu, logvar, z)
    """

    def __init__(
        self,
        input_channels: int = 4,
        z_dim: int = 128,
        base_filters: int = 32,
        num_groups: int = 8,
        sbd_grid_size: Tuple[int, int, int] = (8, 8, 8),
        sbd_upsample_mode: str = "resize_conv",
        posterior_logvar_min: float = -6.0,
        gradient_checkpointing: bool = False,
    ):
        """Initialize TCVAESBD.

        Args:
            input_channels: Number of input channels (4 for MRI modalities).
            z_dim: Latent space dimensionality.
            base_filters: Base number of filters for encoder/decoder.
            num_groups: Number of groups for GroupNorm.
            sbd_grid_size: Spatial resolution for SBD broadcast grid.
            sbd_upsample_mode: SBD upsampling method ("resize_conv" or "deconv").
            posterior_logvar_min: Minimum log-variance for posterior (prevents underflow).
                Default: -6.0 (exp(-6) ≈ 0.0025 variance).
            gradient_checkpointing: If True, use gradient checkpointing
                on encoder/decoder blocks to reduce memory.
        """
        super().__init__()

        self.z_dim = z_dim
        self.posterior_logvar_min = posterior_logvar_min
        self.gradient_checkpointing = gradient_checkpointing

        # Encoder: same as Exp1 baseline
        self.encoder = Encoder3D(
            input_channels=input_channels,
            base_filters=base_filters,
            z_dim=z_dim,
            num_groups=num_groups,
        )

        # Decoder: Spatial Broadcast Decoder
        self.decoder = SpatialBroadcastDecoder(
            z_dim=z_dim,
            output_channels=input_channels,
            base_filters=base_filters,
            grid_size=sbd_grid_size,
            num_groups=num_groups,
            upsample_mode=sbd_upsample_mode,
        )

        # Store checkpointing flag
        self._setup_checkpointing()

    def _setup_checkpointing(self):
        """Setup gradient checkpointing if enabled."""
        if self.gradient_checkpointing:
            # Mark encoder layers for checkpointing
            self._checkpoint_encoder_layers = [
                self.encoder.layer1,
                self.encoder.layer2,
                self.encoder.layer3,
                self.encoder.layer4,
            ]
            # Mark decoder upsample blocks for checkpointing
            self._checkpoint_decoder_layers = [
                self.decoder.up1,
                self.decoder.up2,
                self.decoder.up3,
                self.decoder.up4,
            ]

    def _encoder_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder forward pass with optional checkpointing.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            mu: Posterior mean [B, z_dim].
            logvar: Posterior log-variance [B, z_dim].
        """
        if self.gradient_checkpointing and self.training:
            # Initial conv (not checkpointed, small)
            x = self.encoder.conv1(x)
            x = self.encoder.gn1(x)
            x = self.encoder.relu(x)

            # Checkpointed residual layers
            for layer in self._checkpoint_encoder_layers:
                x = checkpoint(layer, x, use_reentrant=False)

            # Global pool and heads
            x = self.encoder.avgpool(x)
            x = torch.flatten(x, 1)
            mu = self.encoder.fc_mu(x)
            logvar = self.encoder.fc_logvar(x)
        else:
            mu, logvar = self.encoder(x)

        return mu, logvar

    def _decoder_forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder forward pass with optional checkpointing.

        Args:
            z: Latent vector [B, z_dim].

        Returns:
            x_hat: Reconstruction [B, C, D, H, W].
        """
        if self.gradient_checkpointing and self.training:
            # Broadcast and concat (not checkpointed)
            x = self.decoder.broadcast_and_concat(z)
            x = self.decoder.initial_conv(x)

            # Checkpointed upsample blocks
            for layer in self._checkpoint_decoder_layers:
                x = checkpoint(layer, x, use_reentrant=False)

            # Final conv (not checkpointed)
            x_hat = self.decoder.final_conv(x)
        else:
            x_hat = self.decoder(z)

        return x_hat

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterization trick for sampling from posterior with variance floor.

        z = mu + eps * exp(0.5 * logvar), where eps ~ N(0, I)

        CRITICAL: Applies variance floor BEFORE sampling to ensure numerical stability.
        The clamped logvar is returned for consistent KL computation in the loss.

        Args:
            mu: Mean of posterior [B, z_dim].
            logvar: Log-variance of posterior [B, z_dim] (unclamped).

        Returns:
            z: Sampled latent vector [B, z_dim].
            logvar_clamped: Clamped log-variance [B, z_dim] (for loss computation).
        """
        # Apply variance floor BEFORE sampling
        logvar_clamped = torch.clamp(logvar, min=self.posterior_logvar_min)

        # Sample using clamped variance
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Return both z and clamped logvar for consistent loss computation
        return z, logvar_clamped

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            mu: Posterior mean [B, z_dim].
            logvar: Posterior log-variance [B, z_dim].
        """
        return self._encoder_forward(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector [B, z_dim].

        Returns:
            x_hat: Reconstruction [B, C, D, H, W].
        """
        return self._decoder_forward(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through TCVAE+SBD.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            x_hat: Reconstruction [B, C, D, H, W].
            mu: Posterior mean [B, z_dim].
            logvar: Posterior log-variance [B, z_dim] (clamped for stability).
            z: Sampled latent vector [B, z_dim].
        """
        mu, logvar_raw = self.encode(x)
        z, logvar = self.reparameterize(mu, logvar_raw)  # Get clamped logvar
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z  # Return clamped logvar for loss
