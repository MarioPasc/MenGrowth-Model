"""VAE models module."""

from .vae import BaselineVAE
from .vae_sbd import VAESBD

# Backward compatibility alias
TCVAESBD = VAESBD

__all__ = ["BaselineVAE", "VAESBD", "TCVAESBD"]
