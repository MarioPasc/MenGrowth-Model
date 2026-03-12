# src/growth/models/growth/__init__.py
"""Growth prediction models for tumor trajectory forecasting.

Implements a three-model GP hierarchy (LME -> H-GP -> PA-MOGP) operating on
the SDP's disentangled latent space, evaluated under LOPO-CV.

Also provides a scalar GP baseline for direct volumetric prediction.
"""

from .base import FitResult, GrowthModel, PatientTrajectory, PredictionResult
from .hgp_model import HierarchicalGPModel
from .lme_model import LMEGrowthModel
from .scalar_gp import ScalarGP

__all__ = [
    "FitResult",
    "GrowthModel",
    "HierarchicalGPModel",
    "LMEGrowthModel",
    "PatientTrajectory",
    "PredictionResult",
    "ScalarGP",
]
