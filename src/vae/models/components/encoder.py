"""
This module implements a 3D ResNet-style convolutional encoder, suitable for VAEs.
It is configurable in depth and choice of residual block layout.
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from .basic import BasicBlock3d, get_activation, get_norm

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
        blocks_per_layer: List[int] = (2, 2, 2, 2),
        dropout: float = 0.0,
        use_residual: bool = True,
        init_method: str = "kaiming",
        activation: str = "relu",
        norm_type: str = "group",
        pre_activation: bool = False,
    ):
        """Initialize Encoder3D.

        Args:
            input_channels: Number of input channels (4 for MRI modalities).
            base_filters: Base number of filters (doubled at each stage).
            z_dim: Latent space dimensionality.
            num_groups: Number of groups for GroupNorm.
            blocks_per_layer: List of integers specifying the number of residual
                              blocks in each of the 4 stages.
            dropout: Dropout probability for the final layer.
            use_residual: Whether to use residual connections in blocks.
            init_method: Weight initialization method ('kaiming', 'xavier', 'orthogonal').
            activation: Activation function name.
            norm_type: Normalization type name.
            pre_activation: If True, use pre-activation layout in residual blocks.
        """
        super().__init__()

        if len(blocks_per_layer) != 4:
            raise ValueError("`blocks_per_layer` must have 4 elements.")

        self.in_channels = base_filters
        self.num_groups = num_groups
        self.use_residual = use_residual
        self.init_method = init_method
        self.activation_name = activation
        self.norm_type = norm_type
        self.pre_activation = pre_activation

        # Initial convolution: reduce spatial by 2
        self.conv1 = nn.Conv3d(
            input_channels, base_filters, kernel_size=3,
            stride=2, padding=1, bias=False
        )
        self.gn1 = get_norm(norm_type, base_filters, num_groups)
        self.activation = get_activation(activation)

        # ResNet stages
        self.layer1 = self._make_layer(base_filters, blocks_per_layer[0], stride=1)      # 64^3
        self.layer2 = self._make_layer(base_filters * 2, blocks_per_layer[1], stride=2)  # 32^3
        self.layer3 = self._make_layer(base_filters * 4, blocks_per_layer[2], stride=2)  # 16^3
        self.layer4 = self._make_layer(base_filters * 8, blocks_per_layer[3], stride=2)  # 8^3

        # Global pooling and latent heads
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        final_channels = base_filters * 8
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
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
            # For the shortcut connection, we use a 1x1 conv to match dimensions
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                get_norm(self.norm_type, out_channels, self.num_groups),
            )

        layers = []
        layers.append(BasicBlock3d(
            self.in_channels, out_channels, stride,
            self.num_groups, downsample,
            use_residual=self.use_residual,
            activation=self.activation_name,
            norm_type=self.norm_type,
            pre_activation=self.pre_activation,
        ))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock3d(
                out_channels, out_channels,
                num_groups=self.num_groups,
                use_residual=self.use_residual,
                activation=self.activation_name,
                norm_type=self.norm_type,
                pre_activation=self.pre_activation,
            ))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif self.init_method == "orthogonal":
                    nn.init.orthogonal_(m.weight)
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
        x = self.activation(x)

        # ResNet stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        # Latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

