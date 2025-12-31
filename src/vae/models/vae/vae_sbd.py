"""3D VAE with configurable decoder (standard or SBD) for Exp2.

This module implements a flexible VAE architecture:
- 3D ResNet encoder (with GroupNorm)
- Configurable decoder: Spatial Broadcast Decoder (SBD) or standard transposed-conv
- Support for gradient checkpointing for memory efficiency

The SBD provides explicit coordinate information to the decoder,
potentially disentangling position from content in the latent space.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .vae import Encoder3D
from ..components import SpatialBroadcastDecoder


class VAESBD(nn.Module):
    """3D VAE with Spatial Broadcast Decoder.

    Architecture:
    - Encoder: 3D ResNet with GroupNorm (same as Exp1)
    - Decoder: Spatial Broadcast Decoder with coordinate grids
    - Latent: diagonal Gaussian with reparameterization

    Forward signature:
        forward(x) -> (x_hat, mu, logvar, z)

    Note: This model specifically uses SBD. For standard decoder, use BaselineVAE.
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
        dropout: float = 0.0,
        use_residual: bool = True,
        init_method: str = "kaiming",
        activation: str = "relu",
        norm_type: str = "group",
    ):
        """Initialize VAESBD.

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
            dropout: Dropout probability for encoder.
            use_residual: Whether to use residual connections in encoder blocks.
            init_method: Weight initialization method.
            activation: Activation function name.
            norm_type: Normalization type name.
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
            dropout=dropout,
            use_residual=use_residual,
            init_method=init_method,
            activation=activation,
            norm_type=norm_type,
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
            x = self.encoder.activation(x)

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
        """Reparameterization trick (logvar already clamped in encode()).

        z = mu + eps * exp(0.5 * logvar), where eps ~ N(0, I)

        NOTE: logvar is ALREADY CLAMPED in encode() to ensure numerical stability
        across all uses (KL, DIP covariance, sampling).

        Args:
            mu: Mean of posterior [B, z_dim].
            logvar: Log-variance of posterior [B, z_dim], ALREADY CLAMPED in encode().

        Returns:
            z: Sampled latent vector [B, z_dim].
            logvar: Passed through unchanged (for API compatibility).
        """
        # Sample using variance (already clamped in encode())
        std = torch.exp(0.5 * logvar)  # No clamp needed - already done in encode()
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Return both z and logvar (for API compatibility)
        return z, logvar

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Returns clamped logvar to ensure numerical stability across
        all uses: reparameterization, KL computation, DIP covariance.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            mu: Posterior mean [B, z_dim].
            logvar: Posterior log-variance [B, z_dim], CLAMPED at source.
        """
        mu, logvar = self._encoder_forward(x)

        # CRITICAL: Clamp at source before any downstream use
        # This ensures KL divergence and DIP covariance also use clamped values
        logvar_clamped = torch.clamp(logvar, min=self.posterior_logvar_min)

        return mu, logvar_clamped

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
        """Forward pass through VAE+SBD.

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
