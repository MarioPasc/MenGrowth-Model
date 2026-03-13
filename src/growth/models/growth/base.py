# src/growth/models/growth/base.py
"""Abstract base class and shared data structures for growth prediction models.

All growth models (ScalarGP, LME, H-GP, PA-MOGP) share the same interface:
- fit(patients) -> train on a list of PatientTrajectory objects
- predict(patient, t_pred, n_condition) -> return PredictionResult at given times

The ``observations`` array is ``[n_i, D]`` so the interface works for scalar
(D=1), per-dimension (D=24), and multi-output (D=44) models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PatientTrajectory:
    """Single patient's longitudinal observation data.

    Args:
        patient_id: Unique patient identifier.
        times: Observation times, shape ``[n_i]``. Units depend on config
            (ordinal indices or months from first scan).
        observations: Observed values, shape ``[n_i, D]``. D=1 for scalar
            volume, D=24 for volume partition, D=44 for active subspace.
        covariates: Optional static covariates for this patient (e.g.
            ``{"centroid_x": 0.5, "centroid_y": 0.3, "age": 65}``).
            Used by growth models as fixed effects (LME) or mean-function
            terms (GP). ``None`` means no covariates available.
    """

    patient_id: str
    times: np.ndarray
    observations: np.ndarray
    covariates: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        self.observations = np.asarray(self.observations, dtype=np.float64)
        if self.observations.ndim == 1:
            self.observations = self.observations[:, np.newaxis]
        assert self.times.ndim == 1, f"times must be 1-D, got shape {self.times.shape}"
        assert self.observations.ndim == 2, (
            f"observations must be 2-D, got shape {self.observations.shape}"
        )
        assert len(self.times) == len(self.observations), (
            f"times ({len(self.times)}) and observations ({len(self.observations)}) "
            f"length mismatch for patient {self.patient_id}"
        )

    @property
    def n_timepoints(self) -> int:
        """Number of observation timepoints."""
        return len(self.times)

    @property
    def obs_dim(self) -> int:
        """Observation dimensionality (D)."""
        return self.observations.shape[1]


@dataclass
class FitResult:
    """Result of fitting a growth model to training data.

    Args:
        log_marginal_likelihood: Optimized log-marginal-likelihood (or REML criterion).
        hyperparameters: Dict of fitted hyperparameter names to values.
        condition_number: Condition number of the kernel/covariance matrix.
        n_train_patients: Number of patients used for fitting.
        n_train_observations: Total number of observation timepoints used.
    """

    log_marginal_likelihood: float
    hyperparameters: dict[str, float] = field(default_factory=dict)
    condition_number: float = 0.0
    n_train_patients: int = 0
    n_train_observations: int = 0


@dataclass
class PredictionResult:
    """Predictive distribution at query times.

    All arrays have shape ``[n_pred, D]`` where D is the observation dimension.

    Args:
        mean: Posterior predictive mean.
        variance: Posterior predictive variance (diagonal).
        lower_95: Lower bound of 95% credible interval.
        upper_95: Upper bound of 95% credible interval.
    """

    mean: np.ndarray
    variance: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mean", "variance", "lower_95", "upper_95"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            setattr(self, name, arr)


class GrowthModel(ABC):
    """Abstract base class for growth prediction models.

    Subclasses must implement ``fit``, ``predict``, and ``name``.
    """

    @abstractmethod
    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit model on a list of patient trajectories.

        Args:
            patients: Training patient trajectories (all patients pooled).

        Returns:
            FitResult with optimized hyperparameters and diagnostics.
        """

    @abstractmethod
    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict at query times, optionally conditioning on a subset of observations.

        Args:
            patient: Patient trajectory (observations used for conditioning).
            t_pred: Query times, shape ``[n_pred]``.
            n_condition: If given, condition on only the first ``n_condition``
                observations. If None, condition on all observations.

        Returns:
            PredictionResult with posterior mean, variance, and 95% CI.
        """

    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
