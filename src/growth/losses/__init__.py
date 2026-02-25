# src/growth/losses/__init__.py
"""Loss functions for the growth forecasting pipeline.

Contains losses for segmentation (Phase 1), SDP training (Phase 2),
and Neural ODE training (Phase 4).
"""

from .dcor import DistanceCorrelationLoss, distance_correlation
from .sdp_loss import CurriculumSchedule, SDPLoss
from .semantic import SemanticRegressionLoss
from .vicreg import CovarianceLoss, VarianceHingeLoss

__all__ = [
    "SemanticRegressionLoss",
    "CovarianceLoss",
    "VarianceHingeLoss",
    "DistanceCorrelationLoss",
    "distance_correlation",
    "SDPLoss",
    "CurriculumSchedule",
]
