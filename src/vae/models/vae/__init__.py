"""VAE models module."""

from .vae import BaselineVAE
from .vae_sbd import VAESBD
from .semivae import SemiVAE, SemanticProjectionHead

# Backward compatibility alias
TCVAESBD = VAESBD

__all__ = ["BaselineVAE", "VAESBD", "TCVAESBD", "SemiVAE", "SemanticProjectionHead"]
