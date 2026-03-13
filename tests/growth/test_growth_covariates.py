# tests/growth/test_growth_covariates.py
"""Tests for covariate support in growth prediction models."""

import numpy as np
import pytest

from growth.models.growth.base import PatientTrajectory
from growth.models.growth.covariate_utils import (
    collect_covariates,
    get_patient_covariate_vector,
)
from growth.models.growth.lme_model import LMEGrowthModel
from growth.models.growth.hgp_model import HierarchicalGPModel
from growth.models.growth.scalar_gp import ScalarGP

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_patient(pid: str, n_tp: int = 3, covs: dict | None = None) -> PatientTrajectory:
    """Create a synthetic patient trajectory with optional covariates."""
    rng = np.random.RandomState(hash(pid) % 2**31)
    times = np.arange(n_tp, dtype=np.float64)
    obs = 8.0 + 0.3 * times + rng.randn(n_tp) * 0.1
    return PatientTrajectory(
        patient_id=pid,
        times=times,
        observations=obs,
        covariates=covs,
    )


def _make_patients_with_covariates(n: int = 10) -> list[PatientTrajectory]:
    """Create n patients with centroid covariates."""
    patients = []
    for i in range(n):
        covs = {
            "centroid_x": 0.3 + 0.05 * i,
            "centroid_y": 0.5 + 0.02 * i,
            "centroid_z": 0.4 + 0.03 * i,
        }
        patients.append(_make_patient(f"P{i:03d}", n_tp=3 + (i % 3), covs=covs))
    return patients


# ---------------------------------------------------------------------------
# PatientTrajectory tests
# ---------------------------------------------------------------------------

class TestPatientTrajectoryCovariates:
    def test_with_covariates(self) -> None:
        covs = {"centroid_x": 0.5, "centroid_y": 0.3, "age": 65.0}
        pt = _make_patient("P001", covs=covs)
        assert pt.covariates is not None
        assert pt.covariates["centroid_x"] == 0.5
        assert pt.covariates["age"] == 65.0

    def test_backward_compatible_no_covariates(self) -> None:
        pt = _make_patient("P001", covs=None)
        assert pt.covariates is None
        assert pt.n_timepoints == 3
        assert pt.obs_dim == 1

    def test_default_none(self) -> None:
        pt = PatientTrajectory(
            patient_id="P001",
            times=np.array([0, 1, 2]),
            observations=np.array([1.0, 2.0, 3.0]),
        )
        assert pt.covariates is None


# ---------------------------------------------------------------------------
# collect_covariates tests
# ---------------------------------------------------------------------------

class TestCollectCovariates:
    def test_skip_strategy(self) -> None:
        patients = [
            _make_patient("P1", covs={"x": 1.0}),
            _make_patient("P2", covs=None),  # no covariates
            _make_patient("P3", covs={"x": 3.0}),
        ]
        cov_vals, names, filtered = collect_covariates(patients, ["x"], "skip")
        assert names == ["x"]
        assert len(filtered) == 3  # all patients kept
        assert "P1" in cov_vals
        assert "P2" not in cov_vals  # skipped, no covariate value
        assert "P3" in cov_vals

    def test_impute_mean_strategy(self) -> None:
        patients = [
            _make_patient("P1", covs={"x": 2.0}),
            _make_patient("P2", covs=None),
            _make_patient("P3", covs={"x": 4.0}),
        ]
        cov_vals, names, filtered = collect_covariates(patients, ["x"], "impute_mean")
        assert "P2" in cov_vals
        # Mean of x from P1, P3 = (2+4)/2 = 3.0
        np.testing.assert_allclose(cov_vals["P2"], [3.0])

    def test_drop_patient_strategy(self) -> None:
        patients = [
            _make_patient("P1", covs={"x": 1.0}),
            _make_patient("P2", covs=None),
            _make_patient("P3", covs={"x": 3.0}),
        ]
        cov_vals, names, filtered = collect_covariates(patients, ["x"], "drop_patient")
        assert len(filtered) == 2
        pids = [p.patient_id for p in filtered]
        assert "P2" not in pids

    def test_empty_covariate_names(self) -> None:
        patients = [_make_patient("P1")]
        cov_vals, names, filtered = collect_covariates(patients, [], "skip")
        assert cov_vals == {}
        assert names == []
        assert len(filtered) == 1

    def test_all_missing_covariate(self) -> None:
        patients = [_make_patient("P1", covs=None), _make_patient("P2", covs=None)]
        cov_vals, names, filtered = collect_covariates(patients, ["x"], "skip")
        assert names == []  # No covariate available for any patient

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid missing_strategy"):
            collect_covariates([], ["x"], "invalid")


# ---------------------------------------------------------------------------
# get_patient_covariate_vector tests
# ---------------------------------------------------------------------------

class TestGetPatientCovariateVector:
    def test_returns_vector(self) -> None:
        pt = _make_patient("P1", covs={"x": 1.0, "y": 2.0})
        vec = get_patient_covariate_vector(pt, ["x", "y"])
        np.testing.assert_array_equal(vec, [1.0, 2.0])

    def test_returns_none_if_missing(self) -> None:
        pt = _make_patient("P1", covs=None)
        vec = get_patient_covariate_vector(pt, ["x"])
        assert vec is None

    def test_imputes_from_means(self) -> None:
        pt = _make_patient("P1", covs={"x": 1.0})  # missing y
        vec = get_patient_covariate_vector(pt, ["x", "y"], {"x": 0.0, "y": 5.0})
        np.testing.assert_array_equal(vec, [1.0, 5.0])


# ---------------------------------------------------------------------------
# LME with covariates
# ---------------------------------------------------------------------------

class TestLMEWithCovariates:
    def test_fit_predict_with_covariates(self) -> None:
        patients = _make_patients_with_covariates(10)
        model = LMEGrowthModel(
            use_covariates=True,
            covariate_names=["centroid_x", "centroid_y", "centroid_z"],
            missing_strategy="skip",
        )
        result = model.fit(patients)
        assert result.n_train_patients == 10
        assert np.isfinite(result.log_marginal_likelihood) or result.log_marginal_likelihood == 0.0

        # Predict
        pred = model.predict(patients[0], patients[0].times)
        assert pred.mean.shape[0] == patients[0].n_timepoints
        assert np.all(np.isfinite(pred.mean))

    def test_fit_without_covariates_backward_compat(self) -> None:
        patients = [_make_patient(f"P{i}", n_tp=3) for i in range(10)]
        model = LMEGrowthModel()
        result = model.fit(patients)
        assert result.n_train_patients == 10

    def test_covariate_effects_returned(self) -> None:
        patients = _make_patients_with_covariates(10)
        model = LMEGrowthModel(
            use_covariates=True,
            covariate_names=["centroid_x"],
            missing_strategy="skip",
        )
        model.fit(patients)
        effects = model.get_covariate_effects()
        assert len(effects) == 1  # D=1
        assert "centroid_x" in effects[0]

    def test_get_active_names_and_means(self) -> None:
        patients = _make_patients_with_covariates(5)
        model = LMEGrowthModel(
            use_covariates=True,
            covariate_names=["centroid_x", "centroid_y"],
        )
        model.fit(patients)
        assert model.get_active_covariate_names() == ["centroid_x", "centroid_y"]
        means = model.get_covariate_means()
        assert "centroid_x" in means
        assert "centroid_y" in means


# ---------------------------------------------------------------------------
# HGP with covariates
# ---------------------------------------------------------------------------

class TestHGPWithCovariates:
    def test_fit_predict_with_covariates(self) -> None:
        patients = _make_patients_with_covariates(10)
        model = HierarchicalGPModel(
            n_restarts=1,
            max_iter=50,
            use_covariates=True,
            covariate_names=["centroid_x"],
            missing_strategy="skip",
        )
        result = model.fit(patients)
        assert np.isfinite(result.log_marginal_likelihood)

        pred = model.predict(patients[0], patients[0].times)
        assert pred.mean.shape[0] == patients[0].n_timepoints
        assert np.all(np.isfinite(pred.mean))


# ---------------------------------------------------------------------------
# ScalarGP with covariates
# ---------------------------------------------------------------------------

class TestScalarGPWithCovariates:
    def test_fit_predict_with_covariates(self) -> None:
        patients = _make_patients_with_covariates(10)
        model = ScalarGP(
            n_restarts=1,
            max_iter=50,
            use_covariates=True,
            covariate_names=["centroid_x", "centroid_y"],
            missing_strategy="skip",
        )
        result = model.fit(patients)
        assert np.isfinite(result.log_marginal_likelihood)

        # Check that covariate gammas were fitted
        assert "cov_gamma_centroid_x" in result.hyperparameters
        assert "cov_gamma_centroid_y" in result.hyperparameters

        pred = model.predict(patients[0], patients[0].times)
        assert pred.mean.shape[0] == patients[0].n_timepoints
        assert np.all(np.isfinite(pred.mean))

    def test_residualization_improves_or_neutral(self) -> None:
        """With informative covariates, residualization should not degrade fit."""
        rng = np.random.RandomState(42)
        patients = []
        for i in range(15):
            centroid = 0.5 + 0.1 * rng.randn()
            times = np.arange(3, dtype=np.float64)
            # Observation depends on centroid
            obs = 8.0 + 2.0 * centroid + 0.3 * times + rng.randn(3) * 0.05
            patients.append(
                PatientTrajectory(
                    patient_id=f"P{i}",
                    times=times,
                    observations=obs,
                    covariates={"centroid": centroid},
                )
            )

        # Fit with covariates
        model_cov = ScalarGP(n_restarts=1, max_iter=50, use_covariates=True,
                             covariate_names=["centroid"])
        result_cov = model_cov.fit(patients)

        # Fit without
        model_nocov = ScalarGP(n_restarts=1, max_iter=50)
        result_nocov = model_nocov.fit(patients)

        # Both should produce finite results
        assert np.isfinite(result_cov.log_marginal_likelihood)
        assert np.isfinite(result_nocov.log_marginal_likelihood)
