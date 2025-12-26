"""VAE models module."""

from .baseline import BaselineVAE
from .vae_sbd import VAESBD

# Backward compatibility alias
TCVAESBD = VAESBD

__all__ = ["BaselineVAE", "VAESBD", "TCVAESBD"]
