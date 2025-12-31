"""
This module implements a 3D VAE with a Spatial Broadcast Decoder (SBD).
It combines a ResNet-style encoder with the SBD to encourage disentangled
representations, particularly of content vs. position.
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..components import SpatialBroadcastDecoder, Encoder3D


class VAESBD(nn.Module):
    """3D VAE with a Spatial Broadcast Decoder.

    This architecture uses an SBD to provide explicit coordinate information
    to the decoder, which encourages the latent space to become invariant to
    spatial transformations, a key goal for disentanglement.

    Reference:
        Watters et al., "Spatial Broadcast Decoder" (2019).
    """

    def __init__(
        self,
        input_channels: int = 4,
        z_dim: int = 128,
        base_filters: int = 32,
        blocks_per_layer: List[int] = (2, 2, 2, 2),
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
        pre_activation: bool = False,
    ):
        """Initialize VAESBD.

        Args:
            input_channels: Number of input channels (e.g., 4 for MRI modalities).
            z_dim: Latent space dimensionality.
            base_filters: Base number of filters for encoder/decoder.
            blocks_per_layer: List of block counts for each encoder stage.
            num_groups: Number of groups for GroupNorm.
            sbd_grid_size: Spatial resolution for SBD broadcast grid.
            sbd_upsample_mode: SBD upsampling method ("resize_conv" or "deconv").
            posterior_logvar_min: Minimum log-variance for posterior.
            gradient_checkpointing: If True, use gradient checkpointing to save memory.
            dropout: Dropout probability for encoder.
            use_residual: Whether to use residual connections in encoder blocks.
            init_method: Weight initialization method.
            activation: Activation function name.
            norm_type: Normalization type name.
            pre_activation: Use pre-activation layout in encoder ResNet blocks.
        """
        super().__init__()

        self.z_dim = z_dim
        self.posterior_logvar_min = posterior_logvar_min
        self.gradient_checkpointing = gradient_checkpointing

        self.encoder = Encoder3D(
            input_channels=input_channels,
            base_filters=base_filters,
            z_dim=z_dim,
            blocks_per_layer=blocks_per_layer,
            num_groups=num_groups,
            dropout=dropout,
            use_residual=use_residual,
            init_method=init_method,
            activation=activation,
            norm_type=norm_type,
            pre_activation=pre_activation,
        )

        self.decoder = SpatialBroadcastDecoder(
            z_dim=z_dim,
            output_channels=input_channels,
            base_filters=base_filters,
            grid_size=sbd_grid_size,
            num_groups=num_groups,
            upsample_mode=sbd_upsample_mode,
        )

        self._setup_checkpointing()

    def _setup_checkpointing(self):
        """Mark layers for gradient checkpointing if enabled."""
        if self.gradient_checkpointing:
            self.encoder.layer1.is_checkpointed = True
            self.encoder.layer2.is_checkpointed = True
            self.encoder.layer3.is_checkpointed = True
            self.encoder.layer4.is_checkpointed = True
            self.decoder.up1.is_checkpointed = True
            self.decoder.up2.is_checkpointed = True
            self.decoder.up3.is_checkpointed = True
            self.decoder.up4.is_checkpointed = True

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick.

        Args:
            mu: Mean of the posterior distribution.
            logvar: Log-variance of the posterior distribution (already clamped).

        Returns:
            A sample from the latent distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            mu: Posterior mean [B, z_dim].
            logvar: Clamped posterior log-variance [B, z_dim].
        """
        if self.gradient_checkpointing and self.training:
            # Custom forward with checkpointing
            # Initial conv
            x = self.encoder.conv1(x)
            x = self.encoder.gn1(x)
            x = self.encoder.activation(x)
            # Checkpointed layers
            x = checkpoint(self.encoder.layer1, x, use_reentrant=False)
            x = checkpoint(self.encoder.layer2, x, use_reentrant=False)
            x = checkpoint(self.encoder.layer3, x, use_reentrant=False)
            x = checkpoint(self.encoder.layer4, x, use_reentrant=False)
            # Final part
            x = self.encoder.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.encoder.dropout(x)
            mu = self.encoder.fc_mu(x)
            logvar = self.encoder.fc_logvar(x)
        else:
            # Standard forward
            mu, logvar = self.encoder(x)

        # CRITICAL: Clamp at the source before any downstream use.
        logvar_clamped = torch.clamp(logvar, min=self.posterior_logvar_min)
        return mu, logvar_clamped

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector [B, z_dim].

        Returns:
            x_hat: Reconstructed output [B, C, D, H, W].
        """
        if self.gradient_checkpointing and self.training:
            # Custom forward with checkpointing
            x = self.decoder.broadcast_and_concat(z)
            x = self.decoder.initial_conv(x)
            # Checkpointed layers
            x = checkpoint(self.decoder.up1, x, use_reentrant=False)
            x = checkpoint(self.decoder.up2, x, use_reentrant=False)
            x = checkpoint(self.decoder.up3, x, use_reentrant=False)
            x = checkpoint(self.decoder.up4, x, use_reentrant=False)
            # Final part
            x = self.decoder.final_conv(x)
            x_hat = self.decoder.final_act(x)
        else:
            # Standard forward
            x_hat = self.decoder(z)

        return x_hat

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the complete VAE.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            A tuple containing:
            - x_hat: Reconstructed output [B, C, D, H, W].
            - mu: Posterior mean [B, z_dim].
            - logvar: Clamped posterior log-variance [B, z_dim].
            - z: Sampled latent vector [B, z_dim].
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z
