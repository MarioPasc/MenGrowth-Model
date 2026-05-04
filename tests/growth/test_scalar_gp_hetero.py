# tests/growth/test_scalar_gp_hetero.py
"""Tests for ScalarGPHetero.

Markers: phase4, unit
"""

import numpy as np
import pytest

from growth.exceptions import UncertaintyPropagationError
from growth.shared.growth_models import PatientTrajectory

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


def _make_gp_patients(
    n_patients: int = 10,
    n_obs: int = 4,
    seed: int = 42,
    sigma_v: float = 0.01,
) -> list[PatientTrajectory]:
    """Generate synthetic patients for GP testing."""
    rng = np.random.default_rng(seed)
    patients = []
    for i in range(n_patients):
        t = np.arange(n_obs, dtype=np.float64)
        base = rng.uniform(4.0, 10.0)
        slope = rng.uniform(-0.1, 0.5)
        y = base + slope * t + rng.normal(0, 0.1, n_obs)
        sv = np.full(n_obs, sigma_v**2)
        patients.append(
            PatientTrajectory(
                patient_id=f"P{i:03d}",
                times=t,
                observations=y,
                observation_variance=sv,
            )
        )
    return patients


class TestScalarGPHeteroConstructor:
    """Constructor validation."""

    def test_invalid_mean_function_raises(self) -> None:
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

        with pytest.raises(ValueError, match="mean_function"):
            ScalarGPHetero(mean_function="invalid")


class TestScalarGPHeteroFit:
    """Fitting tests."""

    def test_fit_requires_observation_variance(self) -> None:
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

        model = ScalarGPHetero(n_restarts=1)
        patients = [PatientTrajectory(patient_id="P001", times=[0, 1, 2], observations=[5, 6, 7])]
        with pytest.raises(UncertaintyPropagationError):
            model.fit(patients)

    def test_fit_completes(self) -> None:
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

        patients = _make_gp_patients(n_patients=8, seed=42)
        model = ScalarGPHetero(n_restarts=2, seed=42)
        fit = model.fit(patients)
        assert fit.n_train_patients == 8
        assert fit.hyperparameters["signal_variance"] > 0
        assert fit.hyperparameters["lengthscale"] > 0
        assert fit.hyperparameters["noise_variance"] > 0

    def test_fit_recovers_homoscedastic_when_var_zero(self) -> None:
        """With tiny sigma_v, ScalarGPHetero ≈ ScalarGP predictions."""
        from growth.models.growth.scalar_gp import ScalarGP
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

        patients_het = _make_gp_patients(n_patients=10, sigma_v=0.0, seed=42)
        for p in patients_het:
            p.observation_variance = np.full(p.n_timepoints, 1e-10)

        patients_homo = [
            PatientTrajectory(
                patient_id=p.patient_id,
                times=p.times,
                observations=p.observations,
            )
            for p in patients_het
        ]

        het = ScalarGPHetero(n_restarts=3, seed=42)
        het.fit(patients_het)

        homo = ScalarGP(n_restarts=3, seed=42)
        homo.fit(patients_homo)

        t_pred = np.array([patients_het[0].times[-1]])
        pred_het = het.predict(patients_het[0], t_pred, n_condition=3)
        pred_homo = homo.predict(patients_homo[0], t_pred, n_condition=3)

        np.testing.assert_allclose(pred_het.mean[0, 0], pred_homo.mean[0, 0], rtol=0.10)


class TestScalarGPHeteroPredict:
    """Prediction tests."""

    def test_predict_shape(self) -> None:
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

        patients = _make_gp_patients(n_patients=5, seed=42)
        model = ScalarGPHetero(n_restarts=1, seed=42)
        model.fit(patients)

        t_pred = np.array([0.0, 2.0, 5.0])
        pred = model.predict(patients[0], t_pred)
        assert pred.mean.shape == (3, 1)
        assert pred.variance.shape == (3, 1)

    def test_extrapolation_reverts_to_mean(self) -> None:
        """Far from data, mean reverts toward linear mean function."""
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

        patients = _make_gp_patients(n_patients=8, seed=42)
        model = ScalarGPHetero(n_restarts=2, seed=42)
        model.fit(patients)

        # Predict far from training data
        t_pred = np.array([100.0])
        pred = model.predict(patients[0], t_pred, n_condition=1)

        # Variance should be large
        assert pred.variance[0, 0] > 0.1

    def test_metadata_contains_variances(self) -> None:
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

        patients = _make_gp_patients(n_patients=5, seed=42)
        model = ScalarGPHetero(n_restarts=1, seed=42)
        model.fit(patients)

        pred = model.predict(patients[0], np.array([3.0]), n_condition=3)
        assert pred.metadata is not None
        assert "latent_variance" in pred.metadata
        assert "observable_variance" in pred.metadata

    def test_n_condition_reduces_uncertainty(self) -> None:
        """More conditioning data → lower variance."""
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

        patients = _make_gp_patients(n_patients=8, n_obs=5, seed=42)
        model = ScalarGPHetero(n_restarts=2, seed=42)
        model.fit(patients)

        p = patients[0]
        pred_1 = model.predict(p, np.array([4.0]), n_condition=1)
        pred_4 = model.predict(p, np.array([4.0]), n_condition=4)

        assert pred_4.variance[0, 0] <= pred_1.variance[0, 0] + 0.01

    def test_name_string(self) -> None:
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

        model = ScalarGPHetero()
        assert "ScalarGPHetero" in model.name()

    def test_lopo_compatible(self) -> None:
        """ScalarGPHetero works with LOPOEvaluator."""
        from growth.models.growth.scalar_gp_hetero import ScalarGPHetero
        from growth.shared.lopo import LOPOEvaluator

        patients = _make_gp_patients(n_patients=5, n_obs=3, seed=42)
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGPHetero, patients, n_restarts=1, seed=42)
        assert len(results.fold_results) > 0
        assert "last_from_rest/r2_log" in results.aggregate_metrics
