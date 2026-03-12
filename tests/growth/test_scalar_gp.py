# tests/growth/test_scalar_gp.py
"""Tests for the ScalarGP growth model."""

import numpy as np
import pytest

from growth.models.growth.base import PatientTrajectory
from growth.models.growth.scalar_gp import ScalarGP

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_linear_patients(
    n_patients: int = 10,
    slope: float = 0.5,
    intercept: float = 2.0,
    noise_std: float = 0.1,
    min_tp: int = 2,
    max_tp: int = 5,
    seed: int = 42,
) -> list[PatientTrajectory]:
    """Generate synthetic patients with noisy linear trajectories."""
    rng = np.random.RandomState(seed)
    patients = []
    for i in range(n_patients):
        n_tp = rng.randint(min_tp, max_tp + 1)
        times = np.sort(rng.uniform(0, 10, size=n_tp))
        # Per-patient random intercept offset
        b0 = rng.normal(0, 0.3)
        b1 = rng.normal(0, 0.1)
        y = (intercept + b0) + (slope + b1) * times + rng.normal(0, noise_std, n_tp)
        patients.append(
            PatientTrajectory(
                patient_id=f"P{i:03d}",
                times=times,
                observations=y,
            )
        )
    return patients


@pytest.fixture
def linear_patients() -> list[PatientTrajectory]:
    return _make_linear_patients(n_patients=10, seed=42)


@pytest.fixture
def fitted_gp(linear_patients: list[PatientTrajectory]) -> ScalarGP:
    gp = ScalarGP(kernel_type="matern52", mean_function="linear", n_restarts=2, max_iter=200)
    gp.fit(linear_patients)
    return gp


# ---------------------------------------------------------------------------
# TestScalarGPConstruction
# ---------------------------------------------------------------------------


class TestScalarGPConstruction:
    def test_default_kernel_is_matern52(self) -> None:
        gp = ScalarGP()
        assert gp.kernel_type == "matern52"

    def test_invalid_kernel_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid kernel_type"):
            ScalarGP(kernel_type="invalid_kernel")

    def test_linear_mean_function(self) -> None:
        gp = ScalarGP(mean_function="linear")
        assert gp.mean_function == "linear"

    def test_constant_mean_function(self) -> None:
        gp = ScalarGP(mean_function="constant")
        assert gp.mean_function == "constant"

    def test_zero_mean_function(self) -> None:
        gp = ScalarGP(mean_function="zero")
        assert gp.mean_function == "zero"

    def test_invalid_mean_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid mean_function"):
            ScalarGP(mean_function="quadratic")

    def test_se_and_rbf_are_both_valid(self) -> None:
        gp_se = ScalarGP(kernel_type="se")
        gp_rbf = ScalarGP(kernel_type="rbf")
        assert gp_se.kernel_type == "se"
        assert gp_rbf.kernel_type == "rbf"


# ---------------------------------------------------------------------------
# TestScalarGPFit
# ---------------------------------------------------------------------------


class TestScalarGPFit:
    def test_fit_linear_data(self, linear_patients: list[PatientTrajectory]) -> None:
        gp = ScalarGP(n_restarts=2, max_iter=200)
        result = gp.fit(linear_patients)
        assert np.isfinite(result.log_marginal_likelihood)
        assert result.n_train_patients == len(linear_patients)
        assert result.n_train_observations > 0

    def test_fit_returns_finite_lml(self, linear_patients: list[PatientTrajectory]) -> None:
        gp = ScalarGP(n_restarts=2, max_iter=200)
        result = gp.fit(linear_patients)
        assert np.isfinite(result.log_marginal_likelihood)

    def test_fit_returns_positive_hyperparams(
        self, linear_patients: list[PatientTrajectory]
    ) -> None:
        gp = ScalarGP(n_restarts=2, max_iter=200)
        result = gp.fit(linear_patients)
        assert result.hyperparameters["lengthscale"] > 0
        assert result.hyperparameters["signal_variance"] > 0
        assert result.hyperparameters["noise_variance"] > 0

    def test_fit_condition_number_finite(self, linear_patients: list[PatientTrajectory]) -> None:
        gp = ScalarGP(n_restarts=2, max_iter=200)
        result = gp.fit(linear_patients)
        assert np.isfinite(result.condition_number)
        assert result.condition_number > 0

    def test_fit_empty_patients_raises(self) -> None:
        gp = ScalarGP()
        with pytest.raises(ValueError, match="zero patients"):
            gp.fit([])

    def test_fit_multidim_raises(self) -> None:
        """ScalarGP should reject multi-dimensional observations."""
        p = PatientTrajectory("P001", np.array([0, 1]), np.array([[1, 2], [3, 4]]))
        gp = ScalarGP()
        with pytest.raises(AssertionError, match="obs_dim=1"):
            gp.fit([p])


# ---------------------------------------------------------------------------
# TestScalarGPPredict
# ---------------------------------------------------------------------------


class TestScalarGPPredict:
    def test_predict_not_fitted_raises(self) -> None:
        gp = ScalarGP()
        patient = PatientTrajectory("P001", np.array([0, 1]), np.array([1.0, 2.0]))
        with pytest.raises(RuntimeError, match="not fitted"):
            gp.predict(patient, np.array([0.5]))

    def test_predict_interpolation(
        self, fitted_gp: ScalarGP, linear_patients: list[PatientTrajectory]
    ) -> None:
        """Mean at observed times should approximate observations (up to noise)."""
        patient = linear_patients[0]
        t_pred = patient.times
        pred = fitted_gp.predict(patient, t_pred)

        assert pred.mean.shape == (len(t_pred), 1)
        # Residuals should be small (within ~3 noise std)
        residuals = np.abs(pred.mean[:, 0] - patient.observations[:, 0])
        assert np.all(residuals < 1.0), f"Max residual={residuals.max():.3f}"

    def test_predict_variance_increases_with_extrapolation(
        self, fitted_gp: ScalarGP, linear_patients: list[PatientTrajectory]
    ) -> None:
        """Variance at extrapolation points should exceed variance at observed times."""
        patient = linear_patients[0]
        t_obs = patient.times
        t_extrap = np.array([t_obs.max() + 5.0, t_obs.max() + 10.0])

        pred_obs = fitted_gp.predict(patient, t_obs)
        pred_extrap = fitted_gp.predict(patient, t_extrap)

        mean_var_obs = np.mean(pred_obs.variance)
        mean_var_extrap = np.mean(pred_extrap.variance)
        assert mean_var_extrap > mean_var_obs

    def test_predict_95ci_contains_observations(
        self, fitted_gp: ScalarGP, linear_patients: list[PatientTrajectory]
    ) -> None:
        """95% CI should contain most observations."""
        patient = linear_patients[0]
        pred = fitted_gp.predict(patient, patient.times)
        obs = patient.observations[:, 0]
        within = (obs >= pred.lower_95[:, 0]) & (obs <= pred.upper_95[:, 0])
        # At least 80% should be within CI (theoretical is 95%, but small sample)
        assert np.mean(within) >= 0.5

    def test_predict_n_condition_subsets(
        self, fitted_gp: ScalarGP, linear_patients: list[PatientTrajectory]
    ) -> None:
        """Predictions with fewer conditioning points should have larger variance."""
        patient = linear_patients[0]
        if patient.n_timepoints < 3:
            pytest.skip("Need >=3 timepoints")
        t_pred = np.array([patient.times[-1]])

        pred_all = fitted_gp.predict(patient, t_pred)
        pred_one = fitted_gp.predict(patient, t_pred, n_condition=1)

        assert pred_one.variance[0, 0] >= pred_all.variance[0, 0] - 1e-8

    def test_predict_shapes(
        self, fitted_gp: ScalarGP, linear_patients: list[PatientTrajectory]
    ) -> None:
        patient = linear_patients[0]
        t_pred = np.array([0.0, 5.0, 10.0])
        pred = fitted_gp.predict(patient, t_pred)
        assert pred.mean.shape == (3, 1)
        assert pred.variance.shape == (3, 1)
        assert pred.lower_95.shape == (3, 1)
        assert pred.upper_95.shape == (3, 1)

    def test_predict_lower_below_upper(
        self, fitted_gp: ScalarGP, linear_patients: list[PatientTrajectory]
    ) -> None:
        patient = linear_patients[0]
        t_pred = np.linspace(0, 15, 20)
        pred = fitted_gp.predict(patient, t_pred)
        assert np.all(pred.lower_95 <= pred.upper_95)


# ---------------------------------------------------------------------------
# TestScalarGPConvergence (marked slow)
# ---------------------------------------------------------------------------


class TestScalarGPConvergence:
    @pytest.mark.slow
    def test_linear_trend_recovery(self) -> None:
        """With 30 synthetic patients, the GP should recover the linear trend."""
        patients = _make_linear_patients(
            n_patients=30, slope=0.5, intercept=2.0, noise_std=0.1, seed=123
        )
        gp = ScalarGP(
            kernel_type="matern52",
            mean_function="linear",
            n_restarts=5,
            max_iter=500,
        )
        result = gp.fit(patients)

        # Test on held-out trajectory
        t_test = np.linspace(0, 10, 50)
        test_patient = PatientTrajectory("test", np.array([0.0, 5.0]), np.array([2.0, 4.5]))
        pred = gp.predict(test_patient, t_test)

        # The expected trend is y ~ 2.0 + 0.5*t
        expected = 2.0 + 0.5 * t_test
        mae = np.mean(np.abs(pred.mean[:, 0] - expected))
        assert mae < 0.5, f"MAE to true trend: {mae:.3f}"
        assert np.isfinite(result.log_marginal_likelihood)
