# tests/growth/test_hgp_model.py
"""Tests for the Hierarchical GP growth model."""

import numpy as np
import pytest

from growth.models.growth.base import PatientTrajectory
from growth.models.growth.hgp_model import HierarchicalGPModel
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
def fitted_hgp_1d(patients_1d: list[PatientTrajectory]) -> HierarchicalGPModel:
    hgp = HierarchicalGPModel(n_restarts=2, max_iter=200)
    hgp.fit(patients_1d)
    return hgp


# ---------------------------------------------------------------------------
# TestHGPConstruction
# ---------------------------------------------------------------------------


class TestHGPConstruction:
    def test_default_kernel(self) -> None:
        hgp = HierarchicalGPModel()
        assert hgp.kernel_type == "matern52"

    def test_invalid_kernel_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid kernel_type"):
            HierarchicalGPModel(kernel_type="invalid")


# ---------------------------------------------------------------------------
# TestHGPFit
# ---------------------------------------------------------------------------


class TestHGPFit:
    def test_fit_1d_auto_lme(self, patients_1d: list[PatientTrajectory]) -> None:
        """Fit H-GP with internal LME estimation."""
        hgp = HierarchicalGPModel(n_restarts=2, max_iter=200)
        result = hgp.fit(patients_1d)
        assert np.isfinite(result.log_marginal_likelihood)
        assert result.n_train_patients == len(patients_1d)
        assert "lengthscale_d0" in result.hyperparameters
        assert "signal_var_d0" in result.hyperparameters
        assert "noise_var_d0" in result.hyperparameters

    def test_fit_with_prefit_lme(self, patients_1d: list[PatientTrajectory]) -> None:
        """Fit H-GP using a pre-fitted LME."""
        lme = LMEGrowthModel()
        lme.fit(patients_1d)

        hgp = HierarchicalGPModel(n_restarts=2, max_iter=200)
        result = hgp.fit(patients_1d, lme_model=lme)
        assert np.isfinite(result.log_marginal_likelihood)

    def test_fit_3d(self, patients_3d: list[PatientTrajectory]) -> None:
        hgp = HierarchicalGPModel(n_restarts=2, max_iter=200)
        result = hgp.fit(patients_3d)
        for d in range(3):
            assert f"lengthscale_d{d}" in result.hyperparameters

    def test_fit_positive_hyperparams(self, patients_1d: list[PatientTrajectory]) -> None:
        hgp = HierarchicalGPModel(n_restarts=2, max_iter=200)
        result = hgp.fit(patients_1d)
        assert result.hyperparameters["lengthscale_d0"] > 0
        assert result.hyperparameters["signal_var_d0"] > 0
        assert result.hyperparameters["noise_var_d0"] > 0

    def test_fit_empty_raises(self) -> None:
        hgp = HierarchicalGPModel()
        with pytest.raises(ValueError, match="zero patients"):
            hgp.fit([])


# ---------------------------------------------------------------------------
# TestHGPPredict
# ---------------------------------------------------------------------------


class TestHGPPredict:
    def test_predict_not_fitted_raises(self) -> None:
        hgp = HierarchicalGPModel()
        patient = PatientTrajectory("P001", np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))
        with pytest.raises(RuntimeError, match="not fitted"):
            hgp.predict(patient, np.array([1.5]))

    def test_predict_shapes_1d(
        self, fitted_hgp_1d: HierarchicalGPModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        patient = patients_1d[0]
        t_pred = np.array([0.0, 5.0, 10.0])
        pred = fitted_hgp_1d.predict(patient, t_pred)
        assert pred.mean.shape == (3, 1)
        assert pred.variance.shape == (3, 1)

    def test_predict_shapes_3d(self, patients_3d: list[PatientTrajectory]) -> None:
        hgp = HierarchicalGPModel(n_restarts=2, max_iter=200)
        hgp.fit(patients_3d)
        t_pred = np.array([0.0, 5.0])
        pred = hgp.predict(patients_3d[0], t_pred)
        assert pred.mean.shape == (2, 3)
        assert pred.variance.shape == (2, 3)

    def test_predict_interpolation(
        self, fitted_hgp_1d: HierarchicalGPModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        patient = patients_1d[0]
        pred = fitted_hgp_1d.predict(patient, patient.times)
        residuals = np.abs(pred.mean[:, 0] - patient.observations[:, 0])
        assert np.all(residuals < 1.5), f"Max residual {residuals.max():.3f}"

    def test_predict_variance_increases_with_extrapolation(
        self, fitted_hgp_1d: HierarchicalGPModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        patient = patients_1d[0]
        t_obs = patient.times
        t_extrap = np.array([t_obs.max() + 5.0, t_obs.max() + 10.0])

        pred_obs = fitted_hgp_1d.predict(patient, t_obs)
        pred_extrap = fitted_hgp_1d.predict(patient, t_extrap)

        mean_var_obs = np.mean(pred_obs.variance)
        mean_var_extrap = np.mean(pred_extrap.variance)
        assert mean_var_extrap > mean_var_obs

    def test_predict_n_condition(
        self, fitted_hgp_1d: HierarchicalGPModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        patient = patients_1d[0]
        t_pred = np.array([patient.times[-1]])
        pred_all = fitted_hgp_1d.predict(patient, t_pred)
        pred_one = fitted_hgp_1d.predict(patient, t_pred, n_condition=1)
        assert np.all(np.isfinite(pred_all.mean))
        assert np.all(np.isfinite(pred_one.mean))
        # Fewer conditioning points -> more variance
        assert pred_one.variance[0, 0] >= pred_all.variance[0, 0] - 1e-8

    def test_predict_lower_below_upper(
        self, fitted_hgp_1d: HierarchicalGPModel, patients_1d: list[PatientTrajectory]
    ) -> None:
        patient = patients_1d[0]
        t_pred = np.linspace(0, 15, 20)
        pred = fitted_hgp_1d.predict(patient, t_pred)
        assert np.all(pred.lower_95 <= pred.upper_95)


# ---------------------------------------------------------------------------
# TestHGPWithLOPO
# ---------------------------------------------------------------------------


class TestHGPWithLOPO:
    def test_hgp_runs_in_lopo(self, patients_1d: list[PatientTrajectory]) -> None:
        """H-GP should work within LOPO evaluator."""
        from growth.evaluation.lopo_evaluator import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(HierarchicalGPModel, patients_1d, n_restarts=1, max_iter=100)
        assert len(results.fold_results) + len(results.failed_folds) == len(patients_1d)
        assert len(results.fold_results) > len(patients_1d) // 2


# ---------------------------------------------------------------------------
# TestHGPvsScalarGP
# ---------------------------------------------------------------------------


class TestHGPvsScalarGP:
    def test_hgp_1d_comparable_to_scalar_gp(self, patients_1d: list[PatientTrajectory]) -> None:
        """On 1-D data, H-GP and ScalarGP should give qualitatively similar results."""
        from growth.models.growth.scalar_gp import ScalarGP

        sgp = ScalarGP(n_restarts=2, max_iter=200, mean_function="linear")
        sgp.fit(patients_1d)

        hgp = HierarchicalGPModel(n_restarts=2, max_iter=200)
        hgp.fit(patients_1d)

        patient = patients_1d[0]
        t_pred = np.linspace(0, 10, 20)

        pred_sgp = sgp.predict(patient, t_pred)
        pred_hgp = hgp.predict(patient, t_pred)

        # Both should produce finite predictions in the same ballpark
        assert np.all(np.isfinite(pred_sgp.mean))
        assert np.all(np.isfinite(pred_hgp.mean))
        # Correlation between the two predictions should be high
        corr = np.corrcoef(pred_sgp.mean[:, 0], pred_hgp.mean[:, 0])[0, 1]
        assert corr > 0.5, f"SGP vs HGP correlation={corr:.3f}"
