# src/growth/shared/__init__.py
"""Cross-stage shared infrastructure for the MenGrowth growth prediction framework.

This package contains components used by all three stages of the complexity ladder:
- Stage 1 (Volumetric Baseline)
- Stage 2 (Latent Severity Model)
- Stage 3 (Representation Learning)

Key components:
- ``GrowthModel`` ABC and data structures (PatientTrajectory, FitResult, PredictionResult)
- ``LOPOEvaluator`` for Leave-One-Patient-Out cross-validation
- Metric functions (R², MAE, RMSE, calibration)
- Covariate handling utilities
- Trajectory I/O (load/save JSON)
- Bootstrap CIs and permutation tests
"""

from growth.shared.bootstrap import (
    BootstrapResult,
    PermutationTestResult,
    bootstrap_metric,
    paired_permutation_test,
)
from growth.shared.covariate_utils import (
    collect_covariates,
    get_patient_covariate_vector,
)
from growth.shared.growth_models import (
    FitResult,
    GrowthModel,
    PatientTrajectory,
    PredictionResult,
)
from growth.shared.lopo import (
    LOPOEvaluator,
    LOPOFoldResult,
    LOPOResults,
)
from growth.shared.metrics import (
    compute_calibration,
    compute_mae,
    compute_mape,
    compute_r2,
    compute_rmse,
)
from growth.shared.trajectory_io import (
    load_trajectories,
    save_trajectories,
    trajectories_to_dataframe,
)

__all__ = [
    # Growth model ABC + data structures
    "FitResult",
    "GrowthModel",
    "PatientTrajectory",
    "PredictionResult",
    # LOPO-CV
    "LOPOEvaluator",
    "LOPOFoldResult",
    "LOPOResults",
    # Metrics
    "compute_calibration",
    "compute_mae",
    "compute_mape",
    "compute_r2",
    "compute_rmse",
    # Covariates
    "collect_covariates",
    "get_patient_covariate_vector",
    # Trajectory I/O
    "load_trajectories",
    "save_trajectories",
    "trajectories_to_dataframe",
    # Bootstrap
    "BootstrapResult",
    "PermutationTestResult",
    "bootstrap_metric",
    "paired_permutation_test",
]
