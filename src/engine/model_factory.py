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

    Example config:
        model:
          variant: "dipvae_sbd"  # for routing to DIPVAELitModule
          use_sbd: true          # NEW: controls decoder type
          z_dim: 128
          input_channels: 4
          base_filters: 32
          norm: "GROUP"
          # SBD-specific (only used if use_sbd=true)
          sbd_grid_size: [8, 8, 8]
          sbd_upsample_mode: "resize_conv"
    """
    # Common parameters
    input_channels = cfg.model.input_channels
    z_dim = cfg.model.z_dim
    base_filters = cfg.model.base_filters

    # Determine num_groups from norm config
    num_groups = 8  # default
    if cfg.model.norm == "GROUP":
        num_groups = 8

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
            sbd_grid_size=sbd_grid_size,
            sbd_upsample_mode=sbd_upsample_mode,
            posterior_logvar_min=posterior_logvar_min,
            gradient_checkpointing=gradient_checkpointing,
        )
    else:
        # Build standard baseline VAE (transposed-conv decoder)
        model = BaselineVAE(
            input_channels=input_channels,
            z_dim=z_dim,
            base_filters=base_filters,
            num_groups=num_groups,
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
