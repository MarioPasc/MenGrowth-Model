"""Semi-Supervised VAE with partitioned latent space.

This model extends VAESBD with semantic projection heads for supervised
disentanglement. The latent space is partitioned into:
- Supervised dimensions: Mapped to semantic features via projection heads
- Residual dimensions: Standard VAE prior with optional TC regularization

The encoder and decoder are shared, but different loss terms apply to
different latent subsets:
- z_vol, z_loc, z_shape: Regression loss to extracted semantic features
- z_residual: KL + TC loss for unsupervised factors

Architecture:
    Input → Encoder → (mu, logvar) → Reparameterize → z
                                                       ↓
                        Semantic Heads ←── z[0:52] ←──┤
                                                       ↓
                                                  z[52:128] → TC Loss
                                                       ↓
                                                    Decoder → Reconstruction

References:
- Kingma et al., "Semi-Supervised Learning with Deep Generative Models", NeurIPS 2014
- Locatello et al., "Disentangling Factors of Variation Using Few Labels", ICLR 2020
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .vae_sbd import VAESBD
from ..components.encoder import Encoder3D
from ..components.decoder import Decoder3D
from ..components.sbd import SpatialBroadcastDecoder


class SemanticProjectionHead(nn.Module):
    """MLP head for projecting latent dimensions to semantic features.

    Maps a subset of latent dimensions to target semantic features.
    Uses a small MLP for non-linear capacity.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Initialize projection head.

        Args:
            input_dim: Number of input latent dimensions
            output_dim: Number of output semantic features
            hidden_dim: Hidden layer size (default: 2 * input_dim)
            dropout: Dropout probability
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 2 * input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z_subset: torch.Tensor) -> torch.Tensor:
        """Project latent subset to semantic features.

        Args:
            z_subset: Latent dimensions [B, input_dim]

        Returns:
            Predicted semantic features [B, output_dim]
        """
        return self.mlp(z_subset)


class SemiVAE(nn.Module):
    """Semi-Supervised VAE with Spatial Broadcast Decoder.

    Extends VAESBD with semantic projection heads for supervised
    disentanglement of specific latent dimensions.
    """

    def __init__(
        self,
        # Standard VAE parameters
        input_channels: int = 4,
        z_dim: int = 128,
        base_filters: int = 32,
        num_groups: int = 8,
        blocks_per_layer: Tuple[int, ...] = (2, 2, 2, 2),
        upsample_mode: str = "resize_conv",
        # SBD parameters
        use_sbd: bool = True,
        sbd_grid_size: Tuple[int, int, int] = (8, 8, 8),
        sbd_upsample_mode: str = "resize_conv",
        # Semi-supervised parameters
        latent_partitioning: Optional[Dict] = None,
        # Stability
        posterior_logvar_min: float = -6.0,
        gradient_checkpointing: bool = False,
        # Lipschitz continuity for Neural ODE
        use_spectral_norm: bool = False,
    ):
        """Initialize SemiVAE.

        Args:
            input_channels: Number of input channels (4 for MRI)
            z_dim: Total latent dimensionality
            base_filters: Base filter count for encoder/decoder
            num_groups: Groups for GroupNorm
            blocks_per_layer: Blocks per encoder stage
            upsample_mode: Decoder upsampling mode
            use_sbd: Whether to use Spatial Broadcast Decoder
            sbd_grid_size: SBD initial grid size
            sbd_upsample_mode: SBD upsampling mode
            latent_partitioning: Configuration for latent space partitioning
            posterior_logvar_min: Minimum logvar (stability)
            gradient_checkpointing: Enable gradient checkpointing
            use_spectral_norm: If True, apply spectral normalization to encoder
                               and decoder conv layers for Lipschitz continuity.
                               Required for stable Neural ODE training.
                               NOTE: NOT applied to encoder fc_mu/fc_logvar heads.
        """
        super().__init__()

        self.z_dim = z_dim
        self.posterior_logvar_min = posterior_logvar_min
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sbd = use_sbd
        self.use_spectral_norm = use_spectral_norm

        # Build encoder (spectral norm on conv layers, NOT on fc_mu/fc_logvar)
        self.encoder = Encoder3D(
            input_channels=input_channels,
            base_filters=base_filters,
            z_dim=z_dim,
            num_groups=num_groups,
            blocks_per_layer=blocks_per_layer,
            use_spectral_norm=use_spectral_norm,
        )

        # Build decoder (spectral norm on all conv layers)
        if use_sbd:
            self.decoder = SpatialBroadcastDecoder(
                z_dim=z_dim,
                output_channels=input_channels,
                base_filters=base_filters,
                grid_size=sbd_grid_size,
                upsample_mode=sbd_upsample_mode,
                use_spectral_norm=use_spectral_norm,
            )
        else:
            self.decoder = Decoder3D(
                z_dim=z_dim,
                output_channels=input_channels,
                base_filters=base_filters,
                upsample_mode=upsample_mode,
            )

        # Parse latent partitioning configuration
        self.partitioning = self._parse_partitioning(latent_partitioning)

        # Build semantic projection heads
        self.semantic_heads = nn.ModuleDict()
        for name, config in self.partitioning.items():
            if config["supervision"] == "regression":
                head = SemanticProjectionHead(
                    input_dim=config["dim"],
                    output_dim=len(config["target_features"]),
                )
                self.semantic_heads[name] = head

    def _parse_partitioning(
        self, config: Optional[Dict]
    ) -> Dict[str, Dict]:
        """Parse latent partitioning configuration.

        Args:
            config: Partitioning config from YAML

        Returns:
            Parsed partitioning dictionary
        """
        if config is None or not config.get("enabled", False):
            # Default: no partitioning, all dimensions unsupervised
            return {
                "z_residual": {
                    "start_idx": 0,
                    "end_idx": self.z_dim,
                    "dim": self.z_dim,
                    "supervision": "none",
                    "target_features": [],
                }
            }

        partitioning = {}
        for name in ["z_vol", "z_loc", "z_shape", "z_residual"]:
            if name in config:
                part = config[name]
                partitioning[name] = {
                    "start_idx": part["start_idx"],
                    "end_idx": part["start_idx"] + part["dim"],
                    "dim": part["dim"],
                    "supervision": part.get("supervision", "none"),
                    "target_features": part.get("target_features", []),
                }

        return partitioning

    def get_latent_subset(
        self, z: torch.Tensor, name: str
    ) -> torch.Tensor:
        """Extract a subset of latent dimensions.

        Args:
            z: Full latent vector [B, z_dim]
            name: Partition name (e.g., "z_vol", "z_residual")

        Returns:
            Latent subset [B, partition_dim]
        """
        if name not in self.partitioning:
            raise ValueError(f"Unknown partition: {name}")

        config = self.partitioning[name]
        return z[:, config["start_idx"]:config["end_idx"]]

    def predict_semantic_features(
        self, mu: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict semantic features from posterior mean.

        Uses the semantic projection heads to predict features
        from the supervised latent dimensions.

        Args:
            mu: Posterior mean [B, z_dim]

        Returns:
            Dictionary of predicted features per partition
        """
        predictions = {}
        for name, head in self.semantic_heads.items():
            z_subset = self.get_latent_subset(mu, name)
            predictions[name] = head(z_subset)
        return predictions

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to posterior parameters.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            mu: Posterior mean [B, z_dim]
            logvar: Posterior log-variance [B, z_dim], clamped
        """
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=self.posterior_logvar_min)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction.

        Args:
            z: Latent vector [B, z_dim]

        Returns:
            Reconstruction [B, C, D, H, W]
        """
        return self.decoder(z)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from posterior using reparameterization trick.

        Args:
            mu: Posterior mean [B, z_dim]
            logvar: Posterior log-variance [B, z_dim]

        Returns:
            Sampled latent [B, z_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Full forward pass.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            x_hat: Reconstruction [B, C, D, H, W]
            mu: Posterior mean [B, z_dim]
            logvar: Posterior log-variance [B, z_dim]
            z: Sampled latent [B, z_dim]
            semantic_preds: Dictionary of semantic predictions
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        # Predict semantic features from mu (deterministic)
        semantic_preds = self.predict_semantic_features(mu)

        return x_hat, mu, logvar, z, semantic_preds

    def get_partition_info(self) -> Dict[str, Dict]:
        """Get information about latent partitioning.

        Returns:
            Dictionary with partition details
        """
        return self.partitioning.copy()

    def get_residual_indices(self) -> Tuple[int, int]:
        """Get start and end indices for residual (unsupervised) dimensions.

        Returns:
            (start_idx, end_idx) for residual dimensions
        """
        if "z_residual" in self.partitioning:
            config = self.partitioning["z_residual"]
            return config["start_idx"], config["end_idx"]
        else:
            # Fallback: all dimensions are residual
            return 0, self.z_dim

    def get_supervised_indices(self) -> List[Tuple[str, int, int]]:
        """Get indices for all supervised partitions.

        Returns:
            List of (name, start_idx, end_idx) tuples
        """
        supervised = []
        for name, config in self.partitioning.items():
            if config["supervision"] == "regression":
                supervised.append((name, config["start_idx"], config["end_idx"]))
        return supervised
