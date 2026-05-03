# tests/growth/test_nlme_analytical.py
"""Tests for analytical NLME growth models (Exponential, Logistic, Gompertz)."""

import numpy as np
import pytest

from growth.shared.growth_models import PatientTrajectory

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_exponential_patients(
    n_patients: int = 15,
    n_timepoints: int = 4,
    seed: int = 42,
) -> list[PatientTrajectory]:
    """Generate synthetic patients with exponential growth + noise.

    Ground truth: log V(t) = log_V0_i + a_i * t
    with log_V0 ~ N(2.0, 0.1²), a ~ N(0.3, 0.05²), eps ~ N(0, 0.05²).
    """
    rng = np.random.default_rng(seed)
    patients = []
    for i in range(n_patients):
        log_V0 = 2.0 + rng.normal(0, 0.1)
        a = 0.3 + rng.normal(0, 0.05)
        times = np.linspace(0, 3, n_timepoints)
        y = log_V0 + a * times + rng.normal(0, 0.05, size=n_timepoints)
        patients.append(
            PatientTrajectory(
                patient_id=f"EXP-{i:03d}",
                times=times,
                observations=y.reshape(-1, 1),
            )
        )
    return patients


def _make_logistic_patients(
    n_patients: int = 15,
    n_timepoints: int = 5,
    seed: int = 42,
) -> list[PatientTrajectory]:
    """Generate synthetic patients with logistic growth + noise."""
    rng = np.random.default_rng(seed)
    patients = []
    K = 50.0
    for i in range(n_patients):
        V0 = np.exp(2.0 + rng.normal(0, 0.15))
        a = 0.5 + rng.normal(0, 0.1)
        times = np.linspace(0, 5, n_timepoints)
        V_t = K / (1.0 + (K / V0 - 1.0) * np.exp(-a * times))
        y = np.log(V_t) + rng.normal(0, 0.05, size=n_timepoints)
        patients.append(
            PatientTrajectory(
                patient_id=f"LOG-{i:03d}",
                times=times,
                observations=y.reshape(-1, 1),
            )
        )
    return patients


def _make_gompertz_patients(
    n_patients: int = 15,
    n_timepoints: int = 5,
    seed: int = 42,
) -> list[PatientTrajectory]:
    """Generate synthetic patients with Gompertz growth V(t) = K*exp(-b*exp(-c*t))."""
    rng = np.random.default_rng(seed)
    patients = []
    c_pop = 0.5
    for i in range(n_patients):
        K = np.exp(3.5 + rng.normal(0, 0.1))
        b = np.exp(0.5 + rng.normal(0, 0.1))
        times = np.linspace(0, 5, n_timepoints)
        V_t = K * np.exp(-b * np.exp(-c_pop * times))
        y = np.log(np.maximum(V_t, 1e-10)) + rng.normal(0, 0.05, size=n_timepoints)
        patients.append(
            PatientTrajectory(
                patient_id=f"GOM-{i:03d}",
                times=times,
                observations=y.reshape(-1, 1),
            )
        )
    return patients


def _make_degenerate_patients(
    n_patients: int = 10,
    seed: int = 42,
) -> list[PatientTrajectory]:
    """Patients with only 2 timepoints each (stress test for convergence)."""
    rng = np.random.default_rng(seed)
    patients = []
    for i in range(n_patients):
        times = np.array([0.0, 1.0])
        y = np.array([2.0, 2.0 + rng.normal(0.3, 0.1)])
        patients.append(
            PatientTrajectory(
                patient_id=f"DEG-{i:03d}",
                times=times,
                observations=y.reshape(-1, 1),
            )
        )
    return patients


# ---------------------------------------------------------------------------
# ExponentialNLME tests
# ---------------------------------------------------------------------------


class TestExponentialNLMEConstruction:
    def test_default_params(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        model = ExponentialNLME()
        assert model.n_fixed_effects == 2
        assert model.n_random_effects == 2

    def test_name(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        model = ExponentialNLME()
        assert model.name() == "NLME_Exponential"


class TestExponentialNLMEFit:
    def test_fit_returns_fitresult(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        patients = _make_exponential_patients()
        model = ExponentialNLME(n_restarts=2, max_iter=200)
        result = model.fit(patients)

        assert np.isfinite(result.log_marginal_likelihood)
        assert result.n_train_patients == 15
        assert "beta_0" in result.hyperparameters
        assert "beta_1" in result.hyperparameters
        assert "sigma_sq" in result.hyperparameters

    def test_fitted_beta_close_to_truth(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        patients = _make_exponential_patients(n_patients=20, seed=123)
        model = ExponentialNLME(n_restarts=3, max_iter=300)
        result = model.fit(patients)

        # Truth: beta_0 ≈ 2.0, beta_1 ≈ 0.3
        np.testing.assert_allclose(result.hyperparameters["beta_0"], 2.0, atol=0.5)
        np.testing.assert_allclose(result.hyperparameters["beta_1"], 0.3, atol=0.3)

    def test_sigma_sq_positive(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        patients = _make_exponential_patients()
        model = ExponentialNLME(n_restarts=2, max_iter=200)
        result = model.fit(patients)

        assert result.hyperparameters["sigma_sq"] > 0


class TestExponentialNLMEPredict:
    def test_predict_shape(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        patients = _make_exponential_patients()
        model = ExponentialNLME(n_restarts=2, max_iter=200)
        model.fit(patients)

        t_pred = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pred = model.predict(patients[0], t_pred)

        assert pred.mean.shape == (5, 1)
        assert pred.variance.shape == (5, 1)
        assert pred.lower_95.shape == (5, 1)
        assert pred.upper_95.shape == (5, 1)

    def test_variance_positive(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        patients = _make_exponential_patients()
        model = ExponentialNLME(n_restarts=2, max_iter=200)
        model.fit(patients)

        t_pred = np.array([0.0, 1.0, 2.0, 3.0])
        pred = model.predict(patients[0], t_pred)

        assert np.all(pred.variance > 0)

    def test_ci_contains_mean(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        patients = _make_exponential_patients()
        model = ExponentialNLME(n_restarts=2, max_iter=200)
        model.fit(patients)

        t_pred = np.array([0.0, 1.0, 2.0])
        pred = model.predict(patients[0], t_pred)

        assert np.all(pred.lower_95 <= pred.mean)
        assert np.all(pred.mean <= pred.upper_95)

    def test_n_condition(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        patients = _make_exponential_patients()
        model = ExponentialNLME(n_restarts=2, max_iter=200)
        model.fit(patients)

        t_pred = np.array([2.0, 3.0])
        pred = model.predict(patients[0], t_pred, n_condition=2)

        assert pred.mean.shape == (2, 1)
        assert np.all(np.isfinite(pred.mean))

    def test_predict_without_fit_raises(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        model = ExponentialNLME()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(
                _make_exponential_patients()[0],
                np.array([1.0]),
            )


# ---------------------------------------------------------------------------
# LogisticNLME tests
# ---------------------------------------------------------------------------


class TestLogisticNLMEFit:
    def test_fit_on_logistic_data(self) -> None:
        from growth.models.growth.nlme_analytical import LogisticNLME

        patients = _make_logistic_patients()
        model = LogisticNLME(n_restarts=2, max_iter=300)
        result = model.fit(patients)

        assert np.isfinite(result.log_marginal_likelihood)
        assert result.hyperparameters["n_random_effects"] == 2

    def test_predict_finite(self) -> None:
        from growth.models.growth.nlme_analytical import LogisticNLME

        patients = _make_logistic_patients()
        model = LogisticNLME(n_restarts=2, max_iter=300)
        model.fit(patients)

        t_pred = np.array([0.0, 2.0, 4.0])
        pred = model.predict(patients[0], t_pred, n_condition=2)

        assert np.all(np.isfinite(pred.mean))
        assert np.all(pred.variance > 0)


# ---------------------------------------------------------------------------
# GompertzNLME tests
# ---------------------------------------------------------------------------


class TestGompertzNLMEFit:
    def test_fit_on_gompertz_data(self) -> None:
        from growth.models.growth.nlme_analytical import GompertzNLME

        patients = _make_gompertz_patients()
        model = GompertzNLME(n_restarts=2, max_iter=300)
        result = model.fit(patients)

        assert np.isfinite(result.log_marginal_likelihood)

    def test_predict_finite(self) -> None:
        from growth.models.growth.nlme_analytical import GompertzNLME

        patients = _make_gompertz_patients()
        model = GompertzNLME(n_restarts=2, max_iter=300)
        model.fit(patients)

        t_pred = np.array([0.0, 2.0, 4.0])
        pred = model.predict(patients[0], t_pred, n_condition=2)

        assert np.all(np.isfinite(pred.mean))
        assert np.all(pred.variance > 0)

    def test_name(self) -> None:
        from growth.models.growth.nlme_analytical import GompertzNLME

        model = GompertzNLME()
        assert model.name() == "NLME_Gompertz"


# ---------------------------------------------------------------------------
# Fallback and edge cases
# ---------------------------------------------------------------------------


class TestNLMEFallback:
    def test_fallback_to_1re_on_degenerate_data(self) -> None:
        from growth.models.growth.nlme_analytical import ExponentialNLME

        patients = _make_degenerate_patients()
        model = ExponentialNLME(n_restarts=2, max_iter=200, fallback_to_1re=True)
        result = model.fit(patients)

        # Should not crash — may or may not fallback depending on data
        assert result.n_train_patients == 10
        assert np.isfinite(result.hyperparameters["sigma_sq"])


class TestNLMESmoke:
    @pytest.mark.integration
    def test_lopo_single_fold(self) -> None:
        """Smoke test: ExponentialNLME through LOPOEvaluator (1 fold)."""
        from growth.models.growth.nlme_analytical import ExponentialNLME
        from growth.shared.lopo import LOPOEvaluator

        patients = _make_exponential_patients(n_patients=8, n_timepoints=4)
        evaluator = LOPOEvaluator()
        results = evaluator.evaluate(ExponentialNLME, patients, n_restarts=1, max_iter=200)

        assert len(results.failed_folds) == 0 or len(results.fold_results) >= 5
        if results.fold_results:
            r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("nan"))
            assert np.isfinite(r2)
