"""Model factory for VAE experiments.

This module separates model instantiation logic from training logic,
making it easy to switch between different model variants (e.g., with/without SBD).
"""

from typing import Tuple
from omegaconf import DictConfig
import torch.nn as nn

from vae.models import BaselineVAE, VAESBD


def create_vae_model(cfg: DictConfig) -> nn.Module:
    """Create VAE model from configuration.

    This factory handles model instantiation for all VAE variants:
    - Baseline VAE (standard transposed-conv decoder)
    - VAE with Spatial Broadcast Decoder (SBD)

    The choice is controlled by cfg.model.use_sbd flag.

    Args:
        cfg: OmegaConf configuration object with model parameters.

    Returns:
        Initialized VAE model (BaselineVAE or VAESBD).
    """
    # Common parameters
    input_channels = cfg.model.input_channels
    z_dim = cfg.model.z_dim
    base_filters = cfg.model.base_filters
    dropout = cfg.model.get("dropout", 0.0)
    use_residual = cfg.model.get("use_residual", True)
    init_method = cfg.model.get("init_method", "kaiming")
    activation = cfg.model.get("activation", "relu")
    num_groups = cfg.model.get("num_groups", 8)
    norm_type = cfg.model.get("norm", "group").lower()  # Support "GROUP" or "group"

    # Get architectural parameters with defaults for backward compatibility
    arch_cfg = cfg.model.get("architecture", {})
    pre_activation = arch_cfg.get("pre_activation", False)
    blocks_per_layer = arch_cfg.get("blocks_per_layer", [2, 2, 2, 2])
    upsample_mode = arch_cfg.get("upsample_mode", "resize_conv")

    # Get training parameters
    posterior_logvar_min = cfg.train.get("posterior_logvar_min", -6.0)
    gradient_checkpointing = cfg.train.get("gradient_checkpointing", False)

    # Check if SBD should be used
    use_sbd = cfg.model.get("use_sbd", True)  # Default to True for backward compat

    if use_sbd:
        # Build VAE with Spatial Broadcast Decoder
        sbd_grid_size = tuple(cfg.model.get("sbd_grid_size", [8, 8, 8]))
        sbd_upsample_mode = cfg.model.get("sbd_upsample_mode", "resize_conv")

        model = VAESBD(
            input_channels=input_channels,
            z_dim=z_dim,
            base_filters=base_filters,
            num_groups=num_groups,
            blocks_per_layer=blocks_per_layer,
            sbd_grid_size=sbd_grid_size,
            sbd_upsample_mode=sbd_upsample_mode,
            posterior_logvar_min=posterior_logvar_min,
            gradient_checkpointing=gradient_checkpointing,
            dropout=dropout,
            use_residual=use_residual,
            init_method=init_method,
            activation=activation,
            norm_type=norm_type,
            pre_activation=pre_activation,
        )
    else:
        # Build standard baseline VAE (transposed-conv decoder)
        model = BaselineVAE(
            input_channels=input_channels,
            z_dim=z_dim,
            base_filters=base_filters,
            num_groups=num_groups,
            blocks_per_layer=blocks_per_layer,
            gradient_checkpointing=gradient_checkpointing,
            posterior_logvar_min=posterior_logvar_min,
            dropout=dropout,
            use_residual=use_residual,
            init_method=init_method,
            activation=activation,
            norm_type=norm_type,
            pre_activation=pre_activation,
            upsample_mode=upsample_mode,
        )

    return model


def get_model_signature(model: nn.Module) -> str:
    """Get forward signature of model for validation.

    Args:
        model: VAE model instance.

    Returns:
        Signature string: "(x_hat, mu, logvar, z)" for all VAE models.

    Note: Both BaselineVAE and VAESBD now return 4 values for consistency.
    """
    if isinstance(model, (BaselineVAE, VAESBD)):
        return "(x_hat, mu, logvar, z)"
    else:
        return "unknown"
