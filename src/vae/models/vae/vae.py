"""
This module implements a baseline 3D Variational Autoencoder, combining a
ResNet-style encoder and a convolutional decoder into an end-to-end model.
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from ..components import Encoder3D, Decoder3D


class BaselineVAE(nn.Module):
    """Baseline 3D Variational Autoencoder.

    This class brings together the encoder and decoder components to form a complete
    VAE model. It handles the reparameterization trick for end-to-end training.
    The architecture is highly configurable, supporting different depths,
    normalization layers, and upsampling methods.
    """

    def __init__(
        self,
        input_channels: int = 4,
        z_dim: int = 128,
        base_filters: int = 32,
        num_groups: int = 8,
        blocks_per_layer: List[int] = (2, 2, 2, 2),
        posterior_logvar_min: float = -6.0,
        dropout: float = 0.0,
        use_residual: bool = True,
        init_method: str = "kaiming",
        activation: str = "relu",
        norm_type: str = "group",
        pre_activation: bool = False,
        upsample_mode: str = "resize_conv",
        gradient_checkpointing: bool = False,
    ):
        """Initialize BaselineVAE.

        Args:
            input_channels: Number of input channels.
            z_dim: Latent space dimensionality.
            base_filters: Base number of filters for encoder/decoder.
            num_groups: Number of groups for GroupNorm.
            blocks_per_layer: List of block counts for each encoder stage.
            posterior_logvar_min: Minimum value for log-variance.
            dropout: Dropout probability for encoder.
            use_residual: Whether to use residual connections in encoder blocks.
            init_method: Weight initialization method.
            activation: Activation function name.
            norm_type: Normalization type name.
            pre_activation: Use pre-activation layout in encoder ResNet blocks.
            upsample_mode: Upsampling method for decoder ("resize_conv" or "deconv").
            gradient_checkpointing: If True, enables gradient checkpointing for memory
                efficiency. Not implemented in BaselineVAE (no-op), included for API
                compatibility with VAESBD and model_factory.
        """
        super().__init__()

        self.z_dim = z_dim
        self.posterior_logvar_min = posterior_logvar_min
        # Note: gradient_checkpointing is accepted for API compatibility but not
        # implemented in BaselineVAE. See VAESBD for actual implementation.
        self.gradient_checkpointing = gradient_checkpointing

        self.encoder = Encoder3D(
            input_channels=input_channels,
            base_filters=base_filters,
            z_dim=z_dim,
            num_groups=num_groups,
            blocks_per_layer=blocks_per_layer,
            dropout=dropout,
            use_residual=use_residual,
            init_method=init_method,
            activation=activation,
            norm_type=norm_type,
            pre_activation=pre_activation,
        )

        self.decoder = Decoder3D(
            z_dim=z_dim,
            output_channels=input_channels,
            base_filters=base_filters,
            num_groups=num_groups,
            init_method=init_method,
            activation=activation,
            norm_type=norm_type,
            upsample_mode=upsample_mode,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick.

        z = mu + eps * exp(0.5 * logvar), where eps ~ N(0, I)

        Args:
            mu: Mean of posterior [B, z_dim].
            logvar: Log-variance of posterior [B, z_dim], already clamped in encode().

        Returns:
            z: Sampled latent vector [B, z_dim].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        This method is the single source of truth for the posterior. It returns
        a clamped log-variance to ensure numerical stability in all downstream
        computations (KL divergence, sampling, DIP-VAE covariance).

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            mu: Posterior mean [B, z_dim].
            logvar: Posterior log-variance [B, z_dim], CLAMPED at the source.
        """
        mu, logvar = self.encoder(x)

        # CRITICAL: Clamp at the source before any downstream use.
        logvar_clamped = torch.clamp(logvar, min=self.posterior_logvar_min)

        return mu, logvar_clamped

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
