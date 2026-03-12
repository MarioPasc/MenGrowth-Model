# tests/growth/test_lme_model.py
"""Tests for the LME growth model."""

import numpy as np
import pytest

from growth.models.growth.base import PatientTrajectory
from growth.models.growth.lme_model import LMEGrowthModel

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_linear_patients(
    n_patients: int = 10,
    obs_dim: int = 1,
    slope: float = 0.5,
    intercept: float = 2.0,
    noise_std: float = 0.1,
    min_tp: int = 3,
    max_tp: int = 5,
    seed: int = 42,
) -> list[PatientTrajectory]:
    """Generate patients with noisy linear trajectories and random effects."""
    rng = np.random.RandomState(seed)
    patients = []
    for i in range(n_patients):
        n_tp = rng.randint(min_tp, max_tp + 1)
        times = np.sort(rng.uniform(0, 10, size=n_tp))
        obs = np.zeros((n_tp, obs_dim))
        for d in range(obs_dim):
            b0 = rng.normal(0, 0.3)
            b1 = rng.normal(0, 0.05)
            obs[:, d] = (intercept + b0) + (slope + b1) * times + rng.normal(0, noise_std, n_tp)
        patients.append(PatientTrajectory(patient_id=f"P{i:03d}", times=times, observations=obs))
    return patients


@pytest.fixture
def patients_1d() -> list[PatientTrajectory]:
    return _make_linear_patients(n_patients=10, obs_dim=1, seed=42)


@pytest.fixture
def patients_3d() -> list[PatientTrajectory]:
    return _make_linear_patients(n_patients=10, obs_dim=3, seed=42)


@pytest.fixture
def fitted_lme_1d(patients_1d: list[PatientTrajectory]) -> LMEGrowthModel:
    lme = LMEGrowthModel()
    lme.fit(patients_1d)
    return lme


# ---------------------------------------------------------------------------
# TestLMEFit
# ---------------------------------------------------------------------------


class TestLMEFit:
    def test_fit_1d(self, patients_1d: list[PatientTrajectory]) -> None:
        lme = LMEGrowthModel()
        result = lme.fit(patients_1d)
        assert result.n_train_patients == len(patients_1d)
        assert result.n_train_observations > 0
        assert "beta_0_d0" in result.hyperparameters
        assert "beta_1_d0" in result.hyperparameters

    def test_fit_3d(self, patients_3d: list[PatientTrajectory]) -> None:
        lme = LMEGrowthModel()
        result = lme.fit(patients_3d)
        for d in range(3):
            assert f"beta_0_d{d}" in result.hyperparameters
            assert f"beta_1_d{d}" in result.hyperparameters

    def test_fit_recovers_trend(self, patients_1d: list[PatientTrajectory]) -> None:
        """Fixed effects should approximate the true slope=0.5, intercept=2.0."""
        lme = LMEGrowthModel()
        result = lme.fit(patients_1d)
        beta_0 = result.hyperparameters["beta_0_d0"]
        beta_1 = result.hyperparameters["beta_1_d0"]
        assert abs(beta_0 - 2.0) < 1.0, f"Intercept {beta_0} too far from 2.0"
        assert abs(beta_1 - 0.5) < 0.5, f"Slope {beta_1} too far from 0.5"

    def test_fit_empty_raises(self) -> None:
        lme = LMEGrowthModel()
        with pytest.raises(ValueError, match="zero patients"):
            lme.fit([])

    def test_get_fixed_effects(self, fitted_lme_1d: LMEGrowthModel) -> None:
        effects = fitted_lme_1d.get_fixed_effects()
        assert len(effects) == 1
        beta_0, beta_1 = effects[0]
        assert np.isfinite(beta_0)
        assert np.isfinite(beta_1)

    def test_get_fixed_effects_not_fitted_raises(self) -> None:
        lme = LMEGrowthModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            lme.get_fixed_effects()


# ---------------------------------------------------------------------------
# TestLMEPredict
# ---------------------------------------------------------------------------


class TestLMEPredict:
    def test_predict_not_fitted_raises(self) -> None:
        lme = LMEGrowthModel()
        patient = PatientTrajectory("P001", np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))
        with pytest.raises(RuntimeError, match="not fitted"):
            lme.predict(patient, np.array([1.5]))

    def test_predict_shapes_1d(
        self, fitted_lme_1d: LMEGrowthModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        patient = patients_1d[0]
        t_pred = np.array([0.0, 5.0, 10.0])
        pred = fitted_lme_1d.predict(patient, t_pred)
        assert pred.mean.shape == (3, 1)
        assert pred.variance.shape == (3, 1)
        assert pred.lower_95.shape == (3, 1)
        assert pred.upper_95.shape == (3, 1)

    def test_predict_shapes_3d(self, patients_3d: list[PatientTrajectory]) -> None:
        lme = LMEGrowthModel()
        lme.fit(patients_3d)
        t_pred = np.array([0.0, 5.0])
        pred = lme.predict(patients_3d[0], t_pred)
        assert pred.mean.shape == (2, 3)
        assert pred.variance.shape == (2, 3)

    def test_predict_interpolation(
        self, fitted_lme_1d: LMEGrowthModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        """Predictions at observed times should be close to observations."""
        patient = patients_1d[0]
        pred = fitted_lme_1d.predict(patient, patient.times)
        residuals = np.abs(pred.mean[:, 0] - patient.observations[:, 0])
        assert np.all(residuals < 2.0), f"Max residual {residuals.max():.3f}"

    def test_predict_n_condition(
        self, fitted_lme_1d: LMEGrowthModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        """Predictions with fewer conditioning points should still be finite."""
        patient = patients_1d[0]
        t_pred = np.array([patient.times[-1]])
        pred_all = fitted_lme_1d.predict(patient, t_pred)
        pred_one = fitted_lme_1d.predict(patient, t_pred, n_condition=1)
        assert np.all(np.isfinite(pred_all.mean))
        assert np.all(np.isfinite(pred_one.mean))

    def test_predict_lower_below_upper(
        self, fitted_lme_1d: LMEGrowthModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        patient = patients_1d[0]
        t_pred = np.linspace(0, 15, 20)
        pred = fitted_lme_1d.predict(patient, t_pred)
        assert np.all(pred.lower_95 <= pred.upper_95)

    def test_predict_variance_positive(
        self, fitted_lme_1d: LMEGrowthModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        patient = patients_1d[0]
        t_pred = np.linspace(0, 15, 20)
        pred = fitted_lme_1d.predict(patient, t_pred)
        assert np.all(pred.variance > 0)


# ---------------------------------------------------------------------------
# TestLMEWithLOPO
# ---------------------------------------------------------------------------


class TestLMEWithLOPO:
    def test_lme_runs_in_lopo(self, patients_1d: list[PatientTrajectory]) -> None:
        """LME should work within LOPO evaluator."""
        from growth.evaluation.lopo_evaluator import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(LMEGrowthModel, patients_1d)
        assert len(results.fold_results) + len(results.failed_folds) == len(patients_1d)
        # At least some folds should succeed
        assert len(results.fold_results) > len(patients_1d) // 2
