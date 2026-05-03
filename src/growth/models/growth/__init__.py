# src/growth/models/growth/__init__.py
"""Growth prediction models for tumor trajectory forecasting.

Implements a three-model GP hierarchy (LME -> H-GP -> PA-MOGP) operating on
the SDP's disentangled latent space, evaluated under LOPO-CV.

Also provides a scalar GP baseline for direct volumetric prediction.
"""

from .base import FitResult, GrowthModel, PatientTrajectory, PredictionResult
from .hgp_hetero import HGPHeteroModel
from .hgp_model import HierarchicalGPModel
from .lme_hetero import LMEHeteroGrowthModel
from .lme_model import LMEGrowthModel
from .nlme_analytical import ExponentialNLME, GompertzNLME, LogisticNLME
from .scalar_gp import ScalarGP
from .scalar_gp_hetero import ScalarGPHetero

__all__ = [
    "ExponentialNLME",
    "FitResult",
    "GompertzNLME",
    "GrowthModel",
    "HGPHeteroModel",
    "HierarchicalGPModel",
    "LMEGrowthModel",
    "LMEHeteroGrowthModel",
    "LogisticNLME",
    "PatientTrajectory",
    "PredictionResult",
    "ScalarGP",
    "ScalarGPHetero",
]
