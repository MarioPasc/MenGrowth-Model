# src/growth/losses/__init__.py
"""Loss functions for the growth forecasting pipeline.

Contains losses for segmentation (Phase 1) and SDP training (Phase 2).
Phase 4 (Growth Prediction) uses closed-form GP/LME estimation, not neural losses.
"""

from .dcor import DistanceCorrelationLoss, distance_correlation
from .encoder_vicreg import EncoderVICRegLoss
from .sdp_loss import CurriculumSchedule, SDPLoss
from .semantic import SemanticRegressionLoss
from .vicreg import CovarianceLoss, VarianceHingeLoss

__all__ = [
    "SemanticRegressionLoss",
    "CovarianceLoss",
    "VarianceHingeLoss",
    "EncoderVICRegLoss",
    "DistanceCorrelationLoss",
    "distance_correlation",
    "SDPLoss",
    "CurriculumSchedule",
]
