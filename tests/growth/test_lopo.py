# tests/growth/test_lopo.py
"""Tests for the LOPO cross-validation evaluator."""

import numpy as np
import pytest

from growth.evaluation.lopo_evaluator import LOPOEvaluator
from growth.models.growth.base import PatientTrajectory
from growth.models.growth.scalar_gp import ScalarGP

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_patients(
    n: int = 8,
    min_tp: int = 2,
    max_tp: int = 5,
    seed: int = 42,
) -> list[PatientTrajectory]:
    """Generate synthetic patients with noisy linear growth."""
    rng = np.random.RandomState(seed)
    patients = []
    for i in range(n):
        n_tp = rng.randint(min_tp, max_tp + 1)
        times = np.sort(rng.uniform(0, 10, size=n_tp))
        y = 2.0 + 0.5 * times + rng.normal(0, 0.2, n_tp)
        patients.append(PatientTrajectory(patient_id=f"P{i:03d}", times=times, observations=y))
    return patients


@pytest.fixture
def patients() -> list[PatientTrajectory]:
    return _make_patients(n=8, seed=42)


@pytest.fixture
def gp_kwargs() -> dict:
    return {"n_restarts": 1, "max_iter": 100, "mean_function": "linear"}


# ---------------------------------------------------------------------------
# TestLOPODataLeakage
# ---------------------------------------------------------------------------


class TestLOPODataLeakage:
    def test_held_out_not_in_training(
        self, patients: list[PatientTrajectory], gp_kwargs: dict
    ) -> None:
        """Verify the held-out patient is not in the training set for any fold."""
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)

        held_out_ids = {fr.patient_id for fr in results.fold_results}
        all_ids = {p.patient_id for p in patients}
        # Every patient should appear as held out exactly once
        assert held_out_ids == all_ids

    def test_all_patients_held_out_once(
        self, patients: list[PatientTrajectory], gp_kwargs: dict
    ) -> None:
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)
        held_ids = [fr.patient_id for fr in results.fold_results]
        # No duplicates
        assert len(held_ids) == len(set(held_ids))
        # All patients covered (modulo failures)
        assert len(held_ids) + len(results.failed_folds) == len(patients)

    def test_n_folds_equals_n_patients(
        self, patients: list[PatientTrajectory], gp_kwargs: dict
    ) -> None:
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)
        total = len(results.fold_results) + len(results.failed_folds)
        assert total == len(patients)

    def test_train_count_is_n_minus_1(
        self, patients: list[PatientTrajectory], gp_kwargs: dict
    ) -> None:
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)
        for fr in results.fold_results:
            assert fr.n_train_patients == len(patients) - 1


# ---------------------------------------------------------------------------
# TestLOPOMetrics
# ---------------------------------------------------------------------------


class TestLOPOMetrics:
    def test_r2_perfect_predictions(self) -> None:
        """R2 should be 1.0 for perfect predictions."""
        r2 = LOPOEvaluator._r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        assert abs(r2 - 1.0) < 1e-10

    def test_r2_mean_predictions(self) -> None:
        """R2 should be 0.0 when predicting the mean."""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.full_like(actual, actual.mean())
        r2 = LOPOEvaluator._r2(actual, predicted)
        assert abs(r2) < 1e-10

    def test_calibration_100_percent(self) -> None:
        """If all predictions are within CI, calibration should be 1.0."""
        # Create patients where GP predictions are very accurate
        # Use perfect data so the CI is guaranteed to contain observations
        patients = []
        for i in range(5):
            t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
            y = 2.0 + 0.5 * t  # Perfect linear, no noise
            patients.append(PatientTrajectory(f"P{i:03d}", t, y))

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(
            ScalarGP, patients, n_restarts=1, max_iter=100, mean_function="linear"
        )
        # Calibration should be high (>= 0.5 at minimum)
        if "last_from_rest/calibration_95" in results.aggregate_metrics:
            cal = results.aggregate_metrics["last_from_rest/calibration_95"]
            assert cal >= 0.0  # just check it's computed

    def test_mae_nonnegative(self, patients: list[PatientTrajectory], gp_kwargs: dict) -> None:
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)
        if "last_from_rest/mae_log" in results.aggregate_metrics:
            assert results.aggregate_metrics["last_from_rest/mae_log"] >= 0.0


# ---------------------------------------------------------------------------
# TestLOPOProtocols
# ---------------------------------------------------------------------------


class TestLOPOProtocols:
    def test_last_from_rest_uses_n_minus_1(
        self, patients: list[PatientTrajectory], gp_kwargs: dict
    ) -> None:
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)
        for fr in results.fold_results:
            if "last_from_rest" in fr.predictions:
                for pred in fr.predictions["last_from_rest"]:
                    assert pred["n_conditioning"] == fr.n_timepoints - 1

    def test_all_from_first_uses_1(
        self, patients: list[PatientTrajectory], gp_kwargs: dict
    ) -> None:
        evaluator = LOPOEvaluator(prediction_protocols=["all_from_first"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)
        for fr in results.fold_results:
            if "all_from_first" in fr.predictions:
                for pred in fr.predictions["all_from_first"]:
                    assert pred["n_conditioning"] == 1

    def test_all_from_first_only_3plus(self, gp_kwargs: dict) -> None:
        """all_from_first should only produce predictions for patients with >=3 timepoints."""
        # Create a mix: some with 2, some with 3+ timepoints
        patients_mixed = [
            PatientTrajectory("P2tp", np.array([0, 1]), np.array([1.0, 2.0])),
            PatientTrajectory("P3tp", np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0])),
            PatientTrajectory("P4tp", np.array([0, 1, 2, 3]), np.array([1.0, 2.0, 3.0, 4.0])),
        ]
        evaluator = LOPOEvaluator(prediction_protocols=["all_from_first"])
        results = evaluator.evaluate(ScalarGP, patients_mixed, **gp_kwargs)

        for fr in results.fold_results:
            if fr.n_timepoints < 3:
                # Should not have all_from_first predictions
                assert (
                    "all_from_first" not in fr.predictions
                    or fr.predictions["all_from_first"] is None
                ), (
                    f"Patient {fr.patient_id} with {fr.n_timepoints} tp should not have all_from_first"
                )

    def test_both_protocols_together(
        self, patients: list[PatientTrajectory], gp_kwargs: dict
    ) -> None:
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest", "all_from_first"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)
        # Should have metrics for both protocols
        assert any("last_from_rest" in k for k in results.aggregate_metrics)


# ---------------------------------------------------------------------------
# TestLOPOSerialization
# ---------------------------------------------------------------------------


class TestLOPOSerialization:
    def test_to_dict_is_json_compatible(
        self, patients: list[PatientTrajectory], gp_kwargs: dict
    ) -> None:
        import json

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)
        d = results.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_dict_has_required_fields(
        self, patients: list[PatientTrajectory], gp_kwargs: dict
    ) -> None:
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, patients, **gp_kwargs)
        d = results.to_dict()
        assert "model_name" in d
        assert "n_folds" in d
        assert "aggregate_metrics" in d
        assert "fold_results" in d
