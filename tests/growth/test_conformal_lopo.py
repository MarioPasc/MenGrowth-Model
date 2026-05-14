# tests/growth/test_conformal_lopo.py
"""Unit tests for the nested conformal LOPO evaluator.

Synthetic-data tests for ``growth.evaluation.conformal_lopo``:

- ``evaluate`` produces a finite interval for every requested calibration
  layer and every patient.
- Results round-trip through ``to_dict`` / ``from_dict``.
- On a deliberately over-confident base model, the distribution-free layers
  recover coverage that the parametric layer loses.
- Misuse (too few patients, unknown layer) raises ``ConformalLOPOError``.
"""

import numpy as np
import pytest

from growth.evaluation.conformal_lopo import (
    ALL_LAYERS,
    ConformalLOPOError,
    ConformalLOPOEvaluator,
    ConformalLOPOResults,
    default_cqr_features,
)
from growth.shared.growth_models import (
    FitResult,
    GrowthModel,
    PatientTrajectory,
    PredictionResult,
)

pytestmark = [pytest.mark.evaluation, pytest.mark.unit]


def _make_patients(
    n_patients: int = 22, seed: int = 0, with_variance: bool = True
) -> list[PatientTrajectory]:
    """Synthetic log-linear trajectories with optional per-scan measurement variance."""
    rng = np.random.default_rng(seed)
    patients: list[PatientTrajectory] = []
    for pid in range(n_patients):
        n = int(rng.integers(3, 5))
        times = np.sort(rng.choice(np.arange(0, 8), size=n, replace=False)).astype(float)
        b0 = 2.0 + rng.normal(0, 0.4)
        b1 = 0.3 + rng.normal(0, 0.08)
        y = b0 + b1 * times + rng.normal(0, 0.2, n)
        obs_var = np.maximum(rng.gamma(2.0, 0.02, n), 1e-6) if with_variance else None
        patients.append(
            PatientTrajectory(
                patient_id=f"P{pid:02d}",
                times=times,
                observations=y,
                observation_variance=obs_var,
            )
        )
    return patients


class _OverconfidentModel(GrowthModel):
    """A deliberately mis-calibrated model: correct mean, far-too-tight variance.

    Used to show that the distribution-free calibration layers restore coverage
    that the parametric (native Gaussian) layer loses.
    """

    def __init__(self, variance_shrink: float = 1e-4) -> None:
        self.variance_shrink = variance_shrink
        self._b0 = 0.0
        self._b1 = 0.0

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        t = np.concatenate([p.times for p in patients])
        y = np.concatenate([p.observations[:, 0] for p in patients])
        design = np.column_stack([np.ones_like(t), t])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        self._b0, self._b1 = float(beta[0]), float(beta[1])
        return FitResult(log_marginal_likelihood=0.0, n_train_patients=len(patients))

    def predict(
        self, patient: PatientTrajectory, t_pred: np.ndarray, n_condition: int | None = None
    ) -> PredictionResult:
        t_pred = np.atleast_1d(np.asarray(t_pred, dtype=np.float64))
        mean = self._b0 + self._b1 * t_pred
        var = np.full_like(mean, self.variance_shrink)
        std = np.sqrt(var)
        return PredictionResult(
            mean=mean, variance=var, lower_95=mean - 1.96 * std, upper_95=mean + 1.96 * std
        )

    def name(self) -> str:
        return "OverconfidentModel"


def test_evaluate_produces_all_layers_finite() -> None:
    """Every requested calibration layer yields a finite interval for every fold."""
    patients = _make_patients(n_patients=22, seed=1)
    evaluator = ConformalLOPOEvaluator(alpha=0.05, seed=3, y_min=-2.0, y_max=12.0)
    results = evaluator.evaluate(_OverconfidentModel, patients)
    assert len(results.fold_results) == 22
    assert not results.failed_folds
    for fold in results.fold_results:
        assert set(fold.intervals) == set(ALL_LAYERS)
        for layer, (lo, hi) in fold.intervals.items():
            assert np.isfinite(lo) and np.isfinite(hi), layer
            assert lo <= hi, layer


def test_aggregate_metrics_keys_and_finiteness() -> None:
    patients = _make_patients(n_patients=22, seed=2)
    evaluator = ConformalLOPOEvaluator(seed=3, y_min=-2.0, y_max=12.0)
    metrics = evaluator.evaluate(_OverconfidentModel, patients).aggregate_metrics()
    assert "r2_log" in metrics
    for layer in ALL_LAYERS:
        for suffix in ("is_95", "coverage_95", "mean_width"):
            assert np.isfinite(metrics[f"{layer}/{suffix}"]), f"{layer}/{suffix}"
        assert (
            metrics[f"{layer}/coverage_95_ci_low"]
            <= metrics[f"{layer}/coverage_95"]
            <= metrics[f"{layer}/coverage_95_ci_high"]
        )


def test_conformal_layers_restore_coverage_lost_by_parametric() -> None:
    """On an over-confident base model the conformal layers out-cover the parametric one.

    The base model's mean is unbiased but its variance is shrunk by 1e-4, so
    its native intervals are far too tight. Distribution-free calibration does
    not depend on the (wrong) variance, so it must recover coverage.
    """
    patients = _make_patients(n_patients=24, seed=5)
    evaluator = ConformalLOPOEvaluator(alpha=0.05, seed=7, y_min=-3.0, y_max=13.0)
    metrics = evaluator.evaluate(_OverconfidentModel, patients).aggregate_metrics()
    parametric_cov = metrics["parametric/coverage_95"]
    assert parametric_cov < 0.5  # the model is genuinely over-confident
    for layer in ("jackknife_plus", "cqr_norm", "cqr_proper"):
        assert metrics[f"{layer}/coverage_95"] > parametric_cov
    # the headline distribution-free layer should reach a respectable coverage
    assert metrics["jackknife_plus/coverage_95"] >= 0.75


def test_layers_subset_is_respected() -> None:
    patients = _make_patients(n_patients=20, seed=4)
    evaluator = ConformalLOPOEvaluator(
        layers=("parametric", "jackknife_plus"), seed=1, y_min=-2.0, y_max=12.0
    )
    results = evaluator.evaluate(_OverconfidentModel, patients)
    assert results.layers == ["parametric", "jackknife_plus"]
    for fold in results.fold_results:
        assert set(fold.intervals) == {"parametric", "jackknife_plus"}


def test_serialization_round_trip() -> None:
    patients = _make_patients(n_patients=20, seed=6)
    evaluator = ConformalLOPOEvaluator(seed=2, y_min=-2.0, y_max=12.0)
    results = evaluator.evaluate(_OverconfidentModel, patients)
    restored = ConformalLOPOResults.from_dict(results.to_dict())
    assert restored.model_name == results.model_name
    assert restored.aggregate_metrics().keys() == results.aggregate_metrics().keys()
    np.testing.assert_allclose(
        [restored.aggregate_metrics()[k] for k in sorted(restored.aggregate_metrics())],
        [results.aggregate_metrics()[k] for k in sorted(results.aggregate_metrics())],
        rtol=1e-9,
    )


def test_per_patient_table_shape() -> None:
    patients = _make_patients(n_patients=20, seed=8)
    evaluator = ConformalLOPOEvaluator(seed=1, y_min=-2.0, y_max=12.0)
    results = evaluator.evaluate(_OverconfidentModel, patients)
    rows = results.per_patient_table()
    assert len(rows) == len(results.fold_results) * len(results.layers)
    expected_cols = {
        "patient_id",
        "layer",
        "actual",
        "lower",
        "upper",
        "width",
        "covered",
        "interval_score",
        "sigma_v_sq_target",
    }
    assert expected_cols <= set(rows[0])


def test_too_few_patients_raises() -> None:
    evaluator = ConformalLOPOEvaluator()
    with pytest.raises(ConformalLOPOError):
        evaluator.evaluate(_OverconfidentModel, _make_patients(n_patients=2))


def test_unknown_layer_raises() -> None:
    with pytest.raises(ConformalLOPOError):
        ConformalLOPOEvaluator(layers=("parametric", "not_a_layer"))


def test_default_cqr_features_shape_and_content() -> None:
    patient = PatientTrajectory(
        patient_id="X",
        times=np.array([0.0, 2.0, 5.0]),
        observations=np.array([1.0, 1.5, 2.2]),
    )
    feats = default_cqr_features(patient)
    assert feats.shape == (3,)
    # [t_target, y_last_observed, delta_t]
    np.testing.assert_allclose(feats, [5.0, 1.5, 3.0])


def test_default_cqr_features_requires_two_timepoints() -> None:
    patient = PatientTrajectory(patient_id="X", times=np.array([0.0]), observations=np.array([1.0]))
    with pytest.raises(ConformalLOPOError):
        default_cqr_features(patient)
