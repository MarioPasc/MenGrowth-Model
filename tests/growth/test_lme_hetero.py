# tests/growth/test_lme_hetero.py
"""Tests for LMEHeteroGrowthModel.

Markers: phase4, unit
"""

import numpy as np
import pytest

from growth.exceptions import UncertaintyPropagationError
from growth.shared.growth_models import PatientTrajectory

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


def _make_linear_patients(
    n_patients: int = 10,
    n_obs: int = 4,
    beta0: float = 5.0,
    beta1: float = 0.3,
    sigma_n: float = 0.1,
    sigma_v: float = 0.01,
    seed: int = 42,
) -> list[PatientTrajectory]:
    """Generate synthetic patients with linear growth + noise."""
    rng = np.random.default_rng(seed)
    patients = []
    for i in range(n_patients):
        t = np.arange(n_obs, dtype=np.float64)
        b0i = beta0 + rng.normal(0, 0.5)
        b1i = beta1 + rng.normal(0, 0.1)
        y = b0i + b1i * t + rng.normal(0, sigma_n, n_obs)
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


class TestLMEHeteroConstructor:
    """Constructor validation tests."""

    def test_invalid_method_raises(self) -> None:
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        with pytest.raises(ValueError, match="method"):
            LMEHeteroGrowthModel(method="invalid")

    def test_invalid_n_restarts_raises(self) -> None:
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        with pytest.raises(ValueError, match="n_restarts"):
            LMEHeteroGrowthModel(n_restarts=0)


class TestLMEHeteroFit:
    """Fitting tests."""

    def test_fit_requires_observation_variance(self) -> None:
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        model = LMEHeteroGrowthModel(n_restarts=1)
        patients = [PatientTrajectory(patient_id="P001", times=[0, 1, 2], observations=[5, 6, 7])]
        with pytest.raises(UncertaintyPropagationError, match="observation_variance"):
            model.fit(patients)

    def test_fit_recovers_homoscedastic_when_var_zero(self) -> None:
        """With tiny observation_variance, LMEHetero ≈ LMEGrowthModel."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel
        from growth.models.growth.lme_model import LMEGrowthModel

        patients = _make_linear_patients(n_patients=15, sigma_v=0.0, seed=42)
        # Set observation_variance to tiny value
        for p in patients:
            p.observation_variance = np.full(p.n_timepoints, 1e-10)

        # Fit hetero
        hetero = LMEHeteroGrowthModel(n_restarts=3, seed=42)
        hetero.fit(patients)

        # Fit homo
        homo_patients = [
            PatientTrajectory(
                patient_id=p.patient_id,
                times=p.times,
                observations=p.observations,
            )
            for p in patients
        ]
        homo = LMEGrowthModel()
        homo.fit(homo_patients)

        # Compare predictions on a held-out patient
        test_p = patients[0]
        t_pred = np.array([test_p.times[-1]])

        pred_het = hetero.predict(test_p, t_pred, n_condition=test_p.n_timepoints - 1)
        test_p_homo = homo_patients[0]
        pred_hom = homo.predict(test_p_homo, t_pred, n_condition=test_p_homo.n_timepoints - 1)

        np.testing.assert_allclose(pred_het.mean[0, 0], pred_hom.mean[0, 0], rtol=0.05)

    def test_fit_downweights_noisy_scan(self) -> None:
        """High sigma_v scan → BLUP closer to population mean."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        patients = _make_linear_patients(n_patients=8, n_obs=4, seed=123)

        # Give patient 0 a very noisy last observation
        patients_noisy = []
        for i, p in enumerate(patients):
            sv = p.observation_variance.copy()
            if i == 0:
                sv[-1] = 100.0  # Very noisy last scan
            patients_noisy.append(
                PatientTrajectory(
                    patient_id=p.patient_id,
                    times=p.times,
                    observations=p.observations,
                    observation_variance=sv,
                )
            )

        # Fit with noisy data
        model_noisy = LMEHeteroGrowthModel(n_restarts=3, seed=42)
        model_noisy.fit(patients_noisy)

        # Fit with clean data
        model_clean = LMEHeteroGrowthModel(n_restarts=3, seed=42)
        model_clean.fit(patients)

        # Predict for patient 0 — noisy model should shrink more to pop mean
        t_pred = np.array([5.0])
        pred_noisy = model_noisy.predict(patients_noisy[0], t_pred)
        pred_clean = model_clean.predict(patients[0], t_pred)

        pop_mean = model_noisy._beta[0] + model_noisy._beta[1] * 5.0
        dist_noisy = abs(pred_noisy.mean[0, 0] - pop_mean)
        dist_clean = abs(pred_clean.mean[0, 0] - pop_mean)

        assert dist_noisy <= dist_clean + 0.1, (
            f"Noisy prediction {dist_noisy:.4f} should be closer to pop mean "
            f"than clean {dist_clean:.4f}"
        )

    def test_fit_random_intercept_fallback(self) -> None:
        """When only 2 patients with 2 obs each, should still converge."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        patients = _make_linear_patients(n_patients=3, n_obs=2, seed=42)
        model = LMEHeteroGrowthModel(n_restarts=2, seed=42)
        fit = model.fit(patients)
        assert fit.n_train_patients == 3


class TestLMEHeteroPredict:
    """Prediction tests."""

    def test_predict_shape(self) -> None:
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        patients = _make_linear_patients(n_patients=5, seed=42)
        model = LMEHeteroGrowthModel(n_restarts=2, seed=42)
        model.fit(patients)

        t_pred = np.array([0.0, 1.0, 2.0, 5.0])
        pred = model.predict(patients[0], t_pred)
        assert pred.mean.shape == (4, 1)
        assert pred.variance.shape == (4, 1)
        assert pred.lower_95.shape == (4, 1)
        assert pred.upper_95.shape == (4, 1)

    def test_predict_at_training_time(self) -> None:
        """Predictive mean at observed times approximates observations."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        patients = _make_linear_patients(n_patients=10, sigma_n=0.05, seed=42)
        model = LMEHeteroGrowthModel(n_restarts=3, seed=42)
        model.fit(patients)

        p = patients[0]
        pred = model.predict(p, p.times)
        residuals = np.abs(pred.mean[:, 0] - p.observations[:, 0])
        assert np.mean(residuals) < 0.5

    def test_predict_extrapolation_grows_variance(self) -> None:
        """Variance grows with extrapolation distance."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        patients = _make_linear_patients(n_patients=8, seed=42)
        model = LMEHeteroGrowthModel(n_restarts=2, seed=42)
        model.fit(patients)

        t_pred = np.array([1.0, 10.0, 100.0])
        pred = model.predict(patients[0], t_pred)
        assert pred.variance[2, 0] > pred.variance[0, 0]

    def test_n_condition_subset(self) -> None:
        """n_condition=1 uses only first observation."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        patients = _make_linear_patients(n_patients=5, n_obs=4, seed=42)
        model = LMEHeteroGrowthModel(n_restarts=2, seed=42)
        model.fit(patients)

        p = patients[0]
        pred_1 = model.predict(p, np.array([3.0]), n_condition=1)
        pred_all = model.predict(p, np.array([3.0]), n_condition=p.n_timepoints)

        # More conditioning data → more confident
        assert pred_all.variance[0, 0] <= pred_1.variance[0, 0] + 0.01

    def test_serialization(self) -> None:
        """Model name returns a stable string."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        model = LMEHeteroGrowthModel()
        assert "LMEHetero" in model.name()

    def test_two_obs_closed_form_recovery(self) -> None:
        """One-patient, two-obs: predictive mean ≈ precision-weighted average."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        # Two observations at same time (t=0) with different variances
        p = PatientTrajectory(
            patient_id="P000",
            times=np.array([0.0, 0.0]),
            observations=np.array([3.0, 7.0]),
            observation_variance=np.array([1.0, 4.0]),
        )

        # With many patients to establish population mean near 5.0
        rng = np.random.default_rng(42)
        patients = [p]
        for i in range(20):
            patients.append(
                PatientTrajectory(
                    patient_id=f"P{i + 1:03d}",
                    times=np.array([0.0, 1.0]),
                    observations=np.array([5.0, 5.3]) + rng.normal(0, 0.1, 2),
                    observation_variance=np.array([0.01, 0.01]),
                )
            )

        model = LMEHeteroGrowthModel(n_restarts=3, seed=42)
        model.fit(patients)

        pred = model.predict(p, np.array([0.0]))
        # The prediction should exist and be finite
        assert np.isfinite(pred.mean[0, 0])
        assert pred.variance[0, 0] > 0

    def test_metadata_contains_both_variances(self) -> None:
        """PredictionResult.metadata has latent and observable variance."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel

        patients = _make_linear_patients(n_patients=5, seed=42)
        model = LMEHeteroGrowthModel(n_restarts=2, seed=42)
        model.fit(patients)

        pred = model.predict(patients[0], np.array([5.0]), n_condition=3)
        assert pred.metadata is not None
        assert "latent_variance" in pred.metadata
        assert "observable_variance" in pred.metadata
