"""
This module implements a 3D Residual Basic Block, a fundamental component
for building deep convolutional neural networks. It includes a configurable
pre-activation variant for improved gradient flow.
"""

import torch
import torch.nn as nn
from typing import Optional

def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif name.lower() == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif name.lower() in ["silu", "swish"]:
        return nn.SiLU(inplace=True)
    elif name.lower() == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


def get_norm(name: str, channels: int, num_groups: int = 8) -> nn.Module:
    """Get normalization layer by name."""
    if name.lower() == "group":
        return nn.GroupNorm(num_groups, channels)
    elif name.lower() == "batch":
        return nn.BatchNorm3d(channels)
    elif name.lower() == "instance":
        return nn.InstanceNorm3d(channels)
    elif name.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {name}")


class BasicBlock3d(nn.Module):
    """3D Residual block with configurable Norm, Activation, and layout.

    Implements a standard "post-activation" residual block or a "pre-activation"
    block, which can improve gradient flow and regularization.

    Layouts:
    - Post-activation (pre_activation=False):
        x -> [CONV -> NORM -> ACT] -> [CONV -> NORM] -> ADD -> ACT -> out
    - Pre-activation (pre_activation=True):
        x -> [NORM -> ACT -> CONV] -> [NORM -> ACT -> CONV] -> ADD -> out

    Reference:
        He et al., "Identity Mappings in Deep Residual Networks" (2016).
        https://arxiv.org/abs/1603.05027
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_groups: int = 8,
        downsample: Optional[nn.Module] = None,
        use_residual: bool = True,
        activation: str = "relu",
        norm_type: str = "group",
        pre_activation: bool = False,
    ):
        """Initialize BasicBlock3d.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for first convolution (for downsampling).
            num_groups: Number of groups for GroupNorm.
            downsample: Optional downsample module for shortcut connection.
            use_residual: Whether to use residual skip connection.
            activation: Activation function name.
            norm_type: Normalization type name.
            pre_activation: If True, use pre-activation layout (NORM->ACT->CONV).
                            Default: False.
        """
        super().__init__()
        self.use_residual = use_residual
        self.pre_activation = pre_activation

        if self.pre_activation:
            # Pre-activation: NORM -> ACT -> CONV
            self.norm1 = get_norm(norm_type, in_channels, num_groups)
            self.act1 = get_activation(activation)
            self.conv1 = nn.Conv3d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=1, bias=False
            )
            self.norm2 = get_norm(norm_type, out_channels, num_groups)
            self.act2 = get_activation(activation)
            self.conv2 = nn.Conv3d(
                out_channels, out_channels, kernel_size=3,
                stride=1, padding=1, bias=False
            )
        else:
            # Post-activation: CONV -> NORM -> ACT
            self.conv1 = nn.Conv3d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=1, bias=False
            )
            self.norm1 = get_norm(norm_type, out_channels, num_groups)
            self.act1 = get_activation(activation)
            self.conv2 = nn.Conv3d(
                out_channels, out_channels, kernel_size=3,
                stride=1, padding=1, bias=False
            )
            self.norm2 = get_norm(norm_type, out_channels, num_groups)
            self.final_act = get_activation(activation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.pre_activation:
            out = self.norm1(x)
            out = self.act1(out)
            out = self.conv1(out)

            out = self.norm2(out)
            out = self.act2(out)
            out = self.conv2(out)

            if self.use_residual:
                out += identity
        else:
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.act1(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.use_residual:
                out += identity
            
            out = self.final_act(out)

        return out

