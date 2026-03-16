# tests/growth/test_stage2_bayesian.py
"""Tests for Stage 2 Bayesian Severity Model (Option B: numpyro MCMC).

Uses minimal MCMC chains (n_warmup=50, n_samples=100, n_chains=1) for
fast testing (~5-10s per test). Not for convergence assessment — just
verifying the pipeline works end-to-end.

Markers: stage2, unit
"""

import numpy as np
import pytest

from growth.shared.growth_models import PatientTrajectory

# Skip all tests if numpyro not available
numpyro = pytest.importorskip("numpyro", reason="numpyro not installed")

pytestmark = [pytest.mark.stage2, pytest.mark.unit]

# Minimal MCMC settings for fast tests
_FAST_MCMC = {"n_warmup": 50, "n_samples": 100, "n_chains": 1}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_severity_patients(
    n_patients: int = 10,
    seed: int = 42,
) -> tuple[list[PatientTrajectory], np.ndarray]:
    """Generate synthetic patients with known severity signal."""
    rng = np.random.default_rng(seed)
    true_severities = np.linspace(0.15, 0.85, n_patients)
    rng.shuffle(true_severities)

    patients: list[PatientTrajectory] = []
    for i in range(n_patients):
        s_i = true_severities[i]
        n_tp = rng.integers(2, 5)
        baseline = rng.uniform(6.0, 10.0)
        times = np.arange(n_tp, dtype=np.float64)

        alpha = 0.3 + 1.0 * s_i
        growth = np.exp(alpha / 0.8 * (1 - np.exp(-0.8 * times))) - 1
        growth += rng.normal(0, 0.03, size=n_tp)
        growth[0] = 0.0
        obs = baseline + growth

        patients.append(
            PatientTrajectory(
                patient_id=f"BSev-{i:03d}",
                times=times,
                observations=obs,
                covariates={"sphericity": rng.uniform(0.5, 1.0)},
            )
        )

    return patients, true_severities


@pytest.fixture
def bayes_patients():
    """Synthetic patients for Bayesian tests."""
    return _make_severity_patients(n_patients=10, seed=42)


# ---------------------------------------------------------------------------
# Tests: Basic Bayesian Fit
# ---------------------------------------------------------------------------


class TestBayesianSeverityModel:
    """Core tests for BayesianSeverityModel."""

    def test_fit_returns_finite_result(self, bayes_patients) -> None:
        """MCMC fit produces finite FitResult."""
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        patients, _ = bayes_patients
        model = BayesianSeverityModel(**_FAST_MCMC, seed=42)
        result = model.fit(patients)

        assert result.n_train_patients == len(patients)
        assert "severity_mean" in result.hyperparameters
        assert "severity_std" in result.hyperparameters

    def test_posterior_samples_have_correct_keys(self, bayes_patients) -> None:
        """Posterior contains expected parameter names."""
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        patients, _ = bayes_patients
        model = BayesianSeverityModel(**_FAST_MCMC, seed=42)
        model.fit(patients)

        assert model.posterior_samples is not None
        assert "s" in model.posterior_samples
        assert "alpha_0" in model.posterior_samples
        assert "alpha_1" in model.posterior_samples
        assert "beta" in model.posterior_samples
        assert "sigma" in model.posterior_samples

    def test_severity_posterior_in_01(self, bayes_patients) -> None:
        """All posterior severity samples are in [0, 1]."""
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        patients, _ = bayes_patients
        model = BayesianSeverityModel(**_FAST_MCMC, seed=42)
        model.fit(patients)

        s_samples = model.posterior_samples["s"]
        assert np.all(s_samples >= 0.0)
        assert np.all(s_samples <= 1.0)

    def test_severity_summary(self, bayes_patients) -> None:
        """severity_summary returns per-patient mean, std, CI."""
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        patients, _ = bayes_patients
        model = BayesianSeverityModel(**_FAST_MCMC, seed=42)
        model.fit(patients)

        summary = model.severity_summary
        assert summary is not None
        assert len(summary) == len(patients)

        for pid, stats in summary.items():
            assert "mean" in stats
            assert "std" in stats
            assert "ci_lower" in stats
            assert "ci_upper" in stats
            assert 0 <= stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"] <= 1


# ---------------------------------------------------------------------------
# Tests: Prediction
# ---------------------------------------------------------------------------


class TestBayesianPredict:
    """Tests for BayesianSeverityModel.predict()."""

    def test_predict_shapes(self, bayes_patients) -> None:
        """Prediction output shapes match [n_pred, 1]."""
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        patients, _ = bayes_patients
        model = BayesianSeverityModel(**_FAST_MCMC, seed=42)
        model.fit(patients)

        pred = model.predict(patients[0], np.array([0.0, 1.0, 2.0]))
        assert pred.mean.shape == (3, 1)
        assert pred.variance.shape == (3, 1)

    def test_predict_variance_from_posterior(self, bayes_patients) -> None:
        """Variance is non-zero (comes from posterior spread)."""
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        patients, _ = bayes_patients
        model = BayesianSeverityModel(**_FAST_MCMC, seed=42)
        model.fit(patients)

        pred = model.predict(patients[0], np.array([1.0, 2.0]))
        assert np.all(pred.variance > 0), "Bayesian variance should be positive"

    def test_ci_ordering(self, bayes_patients) -> None:
        """lower_95 <= mean <= upper_95."""
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        patients, _ = bayes_patients
        model = BayesianSeverityModel(**_FAST_MCMC, seed=42)
        model.fit(patients)

        pred = model.predict(patients[0], np.array([0.0, 1.0, 2.0]))
        assert np.all(pred.lower_95 <= pred.mean + 1e-10)
        assert np.all(pred.mean <= pred.upper_95 + 1e-10)

    def test_held_out_patient_uses_regression(self, bayes_patients) -> None:
        """Unknown patients use regression head."""
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        patients, _ = bayes_patients
        model = BayesianSeverityModel(**_FAST_MCMC, seed=42)
        model.fit(patients)

        unknown = PatientTrajectory(
            patient_id="Unknown-999",
            times=np.array([0.0, 1.0]),
            observations=np.array([8.0, 8.5]),
            covariates={"sphericity": 0.8},
        )

        pred = model.predict(unknown, np.array([0.0, 1.0, 2.0]))
        assert np.all(np.isfinite(pred.mean))


# ---------------------------------------------------------------------------
# Tests: JAX/NumPy Consistency
# ---------------------------------------------------------------------------


class TestJaxNumpyConsistency:
    """Verify JAX growth functions match NumPy versions."""

    def test_gompertz_consistency(self) -> None:
        """JAX and NumPy Gompertz produce same results."""
        import jax.numpy as jnp

        from growth.stages.stage2_severity.bayesian_severity_model import (
            _gompertz_jax,
        )
        from growth.stages.stage2_severity.growth_functions import ReducedGompertz

        np_fn = ReducedGompertz()
        s = np.array([0.1, 0.5, 0.9])
        t = np.array([0.0, 0.3, 0.7])
        params = np.array([0.5, 1.0, 0.8])

        np_result = np_fn(s, t, params)
        jax_result = np.array(_gompertz_jax(jnp.array(s), jnp.array(t), 0.5, 1.0, 0.8))

        np.testing.assert_allclose(np_result, jax_result, atol=1e-6)

    def test_sigmoid_consistency(self) -> None:
        """JAX and NumPy sigmoid produce same results."""
        import jax.numpy as jnp

        from growth.stages.stage2_severity.bayesian_severity_model import (
            _sigmoid_jax,
        )
        from growth.stages.stage2_severity.growth_functions import WeightedSigmoid

        np_fn = WeightedSigmoid()
        s = np.array([0.1, 0.5, 0.9])
        t = np.array([0.0, 0.3, 0.7])
        params = np.array([2.0, 2.0, 0.0])

        np_result = np_fn(s, t, params)
        jax_result = np.array(_sigmoid_jax(jnp.array(s), jnp.array(t), 2.0, 2.0, 0.0))

        np.testing.assert_allclose(np_result, jax_result, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: LOPO Integration
# ---------------------------------------------------------------------------


class TestBayesianLOPO:
    """LOPO-CV integration test for Bayesian model."""

    def test_lopo_integration(self, bayes_patients) -> None:
        """Full LOPO-CV completes with finite metrics."""
        from growth.shared.lopo import LOPOEvaluator
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        patients, _ = bayes_patients
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(
            BayesianSeverityModel,
            patients,
            **_FAST_MCMC,
            seed=42,
        )

        r2 = results.aggregate_metrics.get("last_from_rest/r2_log")
        assert r2 is not None
        assert np.isfinite(r2)
        assert len(results.failed_folds) == 0

    def test_model_name(self) -> None:
        """Model name includes growth function."""
        from growth.stages.stage2_severity.bayesian_severity_model import (
            BayesianSeverityModel,
        )

        model = BayesianSeverityModel(**_FAST_MCMC)
        assert "gompertz_reduced" in model.name()
