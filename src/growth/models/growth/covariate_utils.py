# src/growth/models/growth/covariate_utils.py
"""Backward-compatible re-exports.

Canonical location: ``growth.shared.covariate_utils``.
"""

from growth.shared.covariate_utils import (
    VALID_MISSING_STRATEGIES,
    collect_covariates,
    get_patient_covariate_vector,
)

__all__ = ["VALID_MISSING_STRATEGIES", "collect_covariates", "get_patient_covariate_vector"]
