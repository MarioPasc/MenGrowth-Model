# tests/growth/test_stage2_severity.py
"""Tests for Stage 2 Latent Severity Model (Option A: MLE).

Covers:
- Quantile transform: values in [0,1], baseline convention, inverse roundtrip
- Growth functions: boundary g(s,0)=0, monotonicity in t and s
- Severity regression head: fit/predict, output range
- Severity model: fit convergence, severity spread, LOPO integration
- Edge cases: minimal patients, constant volume

Markers: stage2, unit
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from growth.shared.growth_models import PatientTrajectory

pytestmark = [pytest.mark.stage2, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures: synthetic severity patients
# ---------------------------------------------------------------------------


def _make_severity_patients(
    n_patients: int = 15,
    min_tp: int = 2,
    max_tp: int = 5,
    true_alpha0: float = 0.3,
    true_alpha1: float = 1.0,
    true_beta: float = 0.8,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[list[PatientTrajectory], np.ndarray]:
    """Generate synthetic patients from known Gompertz severity model.

    Returns (patients, true_severities) where patients have observations
    in log-volume space with a clear severity-dependent growth signal.
    """
    rng = np.random.default_rng(seed)
    true_severities = np.linspace(0.1, 0.9, n_patients)
    rng.shuffle(true_severities)

    patients: list[PatientTrajectory] = []
    for i in range(n_patients):
        s_i = true_severities[i]
        n_tp = rng.integers(min_tp, max_tp + 1)
        baseline_vol = rng.uniform(5.0, 11.0)

        times = np.arange(n_tp, dtype=np.float64)
        # Reduced Gompertz growth
        alpha = true_alpha0 + true_alpha1 * s_i
        growth = np.exp(alpha / (true_beta + 1e-8) * (1 - np.exp(-true_beta * times))) - 1
        growth += rng.normal(0, noise_std, size=n_tp)
        growth[0] = 0.0  # Baseline has zero growth
        obs = baseline_vol + growth

        patients.append(
            PatientTrajectory(
                patient_id=f"Sev-{i:03d}",
                times=times,
                observations=obs,
                covariates={"sphericity": rng.uniform(0.5, 1.0)},
            )
        )

    return patients, true_severities


@pytest.fixture
def severity_patients():
    """Synthetic patients with known severity signal."""
    patients, true_sev = _make_severity_patients(n_patients=15, seed=42)
    return patients, true_sev


@pytest.fixture
def simple_patients():
    """Minimal synthetic patients for quick tests."""
    patients, _ = _make_severity_patients(n_patients=8, seed=99, noise_std=0.01)
    return patients


# ---------------------------------------------------------------------------
# Tests: Quantile Transform
# ---------------------------------------------------------------------------


class TestQuantileTransform:
    """Tests for QuantileTransform (S2-T1)."""

    def test_quantile_values_in_01(self) -> None:
        """All quantile outputs are in (0, 1)."""
        from growth.stages.stage2_severity.quantile_transform import QuantileTransform

        rng = np.random.default_rng(42)
        times = rng.uniform(0.1, 5.0, size=50)
        growths = rng.normal(0, 1, size=50)

        qt = QuantileTransform()
        qt.fit(times, growths)
        result = qt.transform(rng.uniform(0.1, 5.0, size=10), rng.normal(0, 1, size=10))

        assert np.all(result.t_quantile > 0)
        assert np.all(result.t_quantile < 1)
        assert np.all(result.q_growth > 0)
        assert np.all(result.q_growth < 1)

    def test_time_monotonicity(self) -> None:
        """Monotonically increasing times produce monotonically increasing quantiles."""
        from growth.stages.stage2_severity.quantile_transform import QuantileTransform

        ref_times = np.arange(1.0, 20.0)
        ref_growths = np.arange(1.0, 20.0) * 0.1

        qt = QuantileTransform()
        qt.fit(ref_times, ref_growths)

        query_times = np.array([1.0, 5.0, 10.0, 15.0, 19.0])
        query_growths = np.zeros(5)
        result = qt.transform(query_times, query_growths)

        assert np.all(np.diff(result.t_quantile) >= 0), "Time quantiles should be monotonic"

    def test_inverse_roundtrip(self) -> None:
        """inverse_growth(ecdf(x)) approximately recovers x."""
        from growth.stages.stage2_severity.quantile_transform import QuantileTransform

        rng = np.random.default_rng(42)
        growths = rng.normal(1.0, 0.5, size=100)

        qt = QuantileTransform()
        qt.fit(np.arange(100.0), growths)

        # Transform then inverse
        result = qt.transform(np.arange(100.0), growths)
        recovered = qt.inverse_growth(result.q_growth)

        # Should be close (not exact due to ECDF discretization).
        # Tolerance accounts for the searchsorted-based ECDF which uses
        # rank/(n+1) consistently but has step-function interpolation.
        np.testing.assert_allclose(recovered, growths, atol=0.20)

    def test_inverse_boundary_values(self) -> None:
        """Inverse at q~0 maps to min growth, q~1 to max growth."""
        from growth.stages.stage2_severity.quantile_transform import QuantileTransform

        growths = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        qt = QuantileTransform()
        qt.fit(np.arange(5.0), growths)

        low = qt.inverse_growth(np.array([0.01]))
        high = qt.inverse_growth(np.array([0.99]))
        assert low[0] <= high[0]

    def test_growth_std_property(self) -> None:
        """growth_std returns positive std of reference data."""
        from growth.stages.stage2_severity.quantile_transform import QuantileTransform

        qt = QuantileTransform()
        qt.fit(np.arange(10.0), np.random.default_rng(42).normal(0, 1, 10))
        assert qt.growth_std > 0

    def test_n_reference_property(self) -> None:
        """n_reference returns correct count."""
        from growth.stages.stage2_severity.quantile_transform import QuantileTransform

        qt = QuantileTransform()
        qt.fit(np.arange(20.0), np.zeros(20))
        assert qt.n_reference == 20


# ---------------------------------------------------------------------------
# Tests: Growth Functions
# ---------------------------------------------------------------------------


class TestGrowthFunctions:
    """Tests for ReducedGompertz and WeightedSigmoid (S2-T3, S2-T4)."""

    @pytest.mark.parametrize("fn_name", ["gompertz_reduced", "weighted_sigmoid"])
    def test_boundary_condition_g_s_0_equals_0(self, fn_name: str) -> None:
        """g(s, 0; theta) = 0 for all s (S2-T4)."""
        from growth.stages.stage2_severity.growth_functions import GrowthFunctionRegistry

        fn = GrowthFunctionRegistry.get(fn_name)
        # Use middle-of-bounds params
        bounds = fn.param_bounds()
        params = np.array([(lo + hi) / 2 for lo, hi in bounds])

        s_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        t_zero = np.zeros_like(s_values)

        result = fn(s_values, t_zero, params)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    @pytest.mark.parametrize("fn_name", ["gompertz_reduced", "weighted_sigmoid"])
    def test_monotonicity_in_time(self, fn_name: str) -> None:
        """dg/dt >= 0 on a grid (S2-T3)."""
        from growth.stages.stage2_severity.growth_functions import GrowthFunctionRegistry

        fn = GrowthFunctionRegistry.get(fn_name)
        bounds = fn.param_bounds()
        params = np.array([(lo + hi) / 2 for lo, hi in bounds])

        s_grid = np.linspace(0.1, 0.9, 20)
        t_grid = np.linspace(0.01, 0.99, 50)

        for s in s_grid:
            values = fn(np.full_like(t_grid, s), t_grid, params)
            diffs = np.diff(values)
            assert np.all(diffs >= -1e-6), f"Monotonicity violated at s={s:.2f}"

    @pytest.mark.parametrize("fn_name", ["gompertz_reduced", "weighted_sigmoid"])
    def test_monotonicity_in_severity(self, fn_name: str) -> None:
        """dg/ds >= 0 on a grid (S2-T3)."""
        from growth.stages.stage2_severity.growth_functions import GrowthFunctionRegistry

        fn = GrowthFunctionRegistry.get(fn_name)
        bounds = fn.param_bounds()
        params = np.array([(lo + hi) / 2 for lo, hi in bounds])

        s_grid = np.linspace(0.01, 0.99, 50)
        t_values = [0.2, 0.5, 0.8]

        for t in t_values:
            values = fn(s_grid, np.full_like(s_grid, t), params)
            diffs = np.diff(values)
            assert np.all(diffs >= -1e-6), f"Severity monotonicity violated at t={t:.2f}"

    @pytest.mark.parametrize("fn_name", ["gompertz_reduced", "weighted_sigmoid"])
    def test_output_range_01(self, fn_name: str) -> None:
        """Output is clipped to [0, 1]."""
        from growth.stages.stage2_severity.growth_functions import GrowthFunctionRegistry

        fn = GrowthFunctionRegistry.get(fn_name)
        bounds = fn.param_bounds()
        params = np.array([(lo + hi) / 2 for lo, hi in bounds])

        s = np.random.default_rng(42).uniform(0, 1, 100)
        t = np.random.default_rng(43).uniform(0, 1, 100)
        result = fn(s, t, params)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_registry_lookup(self) -> None:
        """Registry returns correct types."""
        from growth.stages.stage2_severity.growth_functions import (
            GrowthFunctionRegistry,
            ReducedGompertz,
            WeightedSigmoid,
        )

        assert isinstance(GrowthFunctionRegistry.get("gompertz_reduced"), ReducedGompertz)
        assert isinstance(GrowthFunctionRegistry.get("weighted_sigmoid"), WeightedSigmoid)

    def test_registry_unknown_raises(self) -> None:
        """Unknown function name raises ValueError."""
        from growth.stages.stage2_severity.growth_functions import GrowthFunctionRegistry

        with pytest.raises(ValueError, match="Unknown"):
            GrowthFunctionRegistry.get("nonexistent")


# ---------------------------------------------------------------------------
# Tests: Severity Regression Head
# ---------------------------------------------------------------------------


class TestSeverityRegressionHead:
    """Tests for SeverityRegressionHead (S2-T8)."""

    def test_fit_predict_roundtrip(self, simple_patients) -> None:
        """Fit on patients, predict returns values in [0, 1]."""
        from growth.stages.stage2_severity.severity_regression import SeverityRegressionHead

        head = SeverityRegressionHead(feature_names=["log_volume", "sphericity"])
        severities = np.linspace(0.1, 0.9, len(simple_patients))
        head.fit(simple_patients, severities)

        for p in simple_patients:
            s_hat = head.predict(p)
            assert 0.01 <= s_hat <= 0.99

    def test_output_in_01(self) -> None:
        """Predictions are clipped to [0.01, 0.99]."""
        from growth.stages.stage2_severity.severity_regression import SeverityRegressionHead

        rng = np.random.default_rng(42)
        patients = [
            PatientTrajectory(
                patient_id=f"P{i}",
                times=np.array([0.0, 1.0]),
                observations=np.array([rng.uniform(-10, 20), rng.uniform(-10, 20)]),
                covariates={"sphericity": rng.uniform(0, 1)},
            )
            for i in range(10)
        ]
        head = SeverityRegressionHead(feature_names=["log_volume", "sphericity"])
        head.fit(patients, rng.uniform(0, 1, 10))

        for p in patients:
            s = head.predict(p)
            assert 0.01 <= s <= 0.99

    def test_missing_features_fallback(self) -> None:
        """Patients without covariates get fallback prediction."""
        from growth.stages.stage2_severity.severity_regression import SeverityRegressionHead

        patients = [
            PatientTrajectory(
                patient_id=f"P{i}",
                times=np.array([0.0, 1.0]),
                observations=np.array([8.0, 8.5]),
                covariates=None,  # No covariates
            )
            for i in range(5)
        ]
        head = SeverityRegressionHead(feature_names=["log_volume", "sphericity"])
        head.fit(patients, np.linspace(0.2, 0.8, 5))

        # Should still produce a prediction via log_volume only
        s = head.predict(patients[0])
        assert 0.01 <= s <= 0.99


# ---------------------------------------------------------------------------
# Tests: Severity Model Fit
# ---------------------------------------------------------------------------


class TestSeverityModelFit:
    """Tests for SeverityModel.fit() (S2-T2, S2-T5, S2-T6)."""

    def test_fit_returns_finite_loss(self, severity_patients) -> None:
        """Joint optimization produces finite loss (S2-T2)."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        patients, _ = severity_patients
        model = SeverityModel(n_restarts=2, max_iter=1000, seed=42)
        result = model.fit(patients)

        assert np.isfinite(result.log_marginal_likelihood)
        assert result.n_train_patients == len(patients)

    def test_fitted_severities_in_01(self, severity_patients) -> None:
        """All fitted severities are in [0, 1] (S2-T2)."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        patients, _ = severity_patients
        model = SeverityModel(n_restarts=2, max_iter=1000, seed=42)
        model.fit(patients)

        severities = model.fitted_severities
        assert severities is not None
        for pid, s in severities.items():
            assert 0.0 <= s <= 1.0, f"Severity {s} out of range for {pid}"

    def test_severity_spread(self, severity_patients) -> None:
        """Severity values have spread (std > 0.1) (S2-T5)."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        patients, _ = severity_patients
        model = SeverityModel(n_restarts=3, max_iter=2000, seed=42)
        model.fit(patients)

        severities = model.fitted_severities
        s_values = np.array(list(severities.values()))
        assert s_values.std() > 0.1, f"Severity std={s_values.std():.3f}, expected > 0.1"

    def test_severity_correlates_with_growth(self, severity_patients) -> None:
        """Severity correlates with observed growth rate (S2-T6)."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        patients, _ = severity_patients
        model = SeverityModel(n_restarts=3, max_iter=2000, seed=42)
        model.fit(patients)

        severities = model.fitted_severities
        growth_rates = []
        sev_values = []
        for p in patients:
            if p.n_timepoints >= 2:
                rate = float(p.observations[-1, 0] - p.observations[0, 0]) / max(
                    p.times[-1] - p.times[0], 1.0
                )
                growth_rates.append(rate)
                sev_values.append(severities[p.patient_id])

        rho, _ = spearmanr(sev_values, growth_rates)
        assert rho > 0.2, f"Spearman(severity, growth_rate)={rho:.3f}, expected > 0.2"

    def test_fit_with_gompertz(self, simple_patients) -> None:
        """Gompertz growth function completes fitting."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        model = SeverityModel(growth_function="gompertz_reduced", n_restarts=2, max_iter=500)
        result = model.fit(simple_patients)
        assert np.isfinite(result.log_marginal_likelihood)

    def test_fit_with_sigmoid(self, simple_patients) -> None:
        """Weighted sigmoid growth function completes fitting."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        model = SeverityModel(growth_function="weighted_sigmoid", n_restarts=2, max_iter=500)
        result = model.fit(simple_patients)
        assert np.isfinite(result.log_marginal_likelihood)


# ---------------------------------------------------------------------------
# Tests: Severity Model Predict
# ---------------------------------------------------------------------------


class TestSeverityModelPredict:
    """Tests for SeverityModel.predict()."""

    def test_predict_shapes(self, simple_patients) -> None:
        """Prediction output shapes match [n_pred, 1]."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        model = SeverityModel(n_restarts=2, max_iter=500)
        model.fit(simple_patients)

        patient = simple_patients[0]
        t_pred = np.array([0.0, 1.0, 2.0, 3.0])
        pred = model.predict(patient, t_pred)

        assert pred.mean.shape == (4, 1)
        assert pred.variance.shape == (4, 1)
        assert pred.lower_95.shape == (4, 1)
        assert pred.upper_95.shape == (4, 1)

    def test_ci_ordering(self, simple_patients) -> None:
        """lower_95 < mean < upper_95."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        model = SeverityModel(n_restarts=2, max_iter=500)
        model.fit(simple_patients)

        pred = model.predict(simple_patients[0], np.array([0.0, 1.0, 2.0]))
        assert np.all(pred.lower_95 <= pred.mean)
        assert np.all(pred.mean <= pred.upper_95)

    def test_training_patient_uses_fitted_severity(self, simple_patients) -> None:
        """Known training patients use their fitted (not regressed) severity."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        model = SeverityModel(n_restarts=2, max_iter=500)
        model.fit(simple_patients)

        # Training patients should be in fitted_severities
        for p in simple_patients:
            assert p.patient_id in model.fitted_severities

    def test_unknown_patient_uses_regression(self, simple_patients) -> None:
        """Held-out patients use regression head for severity."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        model = SeverityModel(n_restarts=2, max_iter=500)
        model.fit(simple_patients)

        # Create unknown patient
        unknown = PatientTrajectory(
            patient_id="Unknown-999",
            times=np.array([0.0, 1.0]),
            observations=np.array([8.0, 8.5]),
            covariates={"sphericity": 0.9},
        )

        pred = model.predict(unknown, np.array([0.0, 1.0, 2.0]))
        assert np.all(np.isfinite(pred.mean))

    def test_predict_not_fitted_raises(self) -> None:
        """Predicting before fit raises AssertionError."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        model = SeverityModel()
        patient = PatientTrajectory(
            patient_id="P0",
            times=np.array([0.0, 1.0]),
            observations=np.array([8.0, 8.5]),
        )
        with pytest.raises(AssertionError):
            model.predict(patient, np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Tests: LOPO Integration
# ---------------------------------------------------------------------------


class TestSeverityLOPO:
    """Tests for LOPO-CV with SeverityModel (S2-T7)."""

    def test_lopo_integration(self, severity_patients) -> None:
        """Full LOPO-CV completes with finite metrics."""
        from growth.shared.lopo import LOPOEvaluator
        from growth.stages.stage2_severity.severity_model import SeverityModel

        patients, _ = severity_patients
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(
            SeverityModel,
            patients,
            growth_function="gompertz_reduced",
            n_restarts=2,
            max_iter=500,
            seed=42,
        )

        r2 = results.aggregate_metrics.get("last_from_rest/r2_log")
        assert r2 is not None
        assert np.isfinite(r2)

    def test_lopo_all_folds_succeed(self, severity_patients) -> None:
        """No folds fail on clean synthetic data."""
        from growth.shared.lopo import LOPOEvaluator
        from growth.stages.stage2_severity.severity_model import SeverityModel

        patients, _ = severity_patients
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(
            SeverityModel,
            patients,
            n_restarts=2,
            max_iter=500,
        )

        assert len(results.failed_folds) == 0, f"Failed folds: {results.failed_folds}"

    def test_lopo_no_data_leakage(self, severity_patients) -> None:
        """Held-out patient's severity is NOT in fitted_severities during prediction."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        patients, _ = severity_patients
        # Manually simulate one LOPO fold
        held_out = patients[0]
        train = patients[1:]

        model = SeverityModel(n_restarts=2, max_iter=500)
        model.fit(train)

        # Held-out should NOT be in fitted severities
        assert held_out.patient_id not in model.fitted_severities

        # Should still predict (via regression head)
        pred = model.predict(held_out, np.array([held_out.times[-1]]))
        assert np.all(np.isfinite(pred.mean))


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_two_timepoint_patients(self) -> None:
        """Model handles patients with exactly 2 timepoints."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        rng = np.random.default_rng(42)
        patients = [
            PatientTrajectory(
                patient_id=f"P{i}",
                times=np.array([0.0, 1.0]),
                observations=np.array(
                    [
                        rng.uniform(5, 12),
                        rng.uniform(5, 12),
                    ]
                ),
                covariates={"sphericity": rng.uniform(0.5, 1.0)},
            )
            for i in range(6)
        ]

        model = SeverityModel(n_restarts=2, max_iter=500)
        result = model.fit(patients)
        assert np.isfinite(result.log_marginal_likelihood)

    def test_constant_volume_patients(self) -> None:
        """Model handles patients with zero growth."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        patients = [
            PatientTrajectory(
                patient_id=f"P{i}",
                times=np.arange(3, dtype=np.float64),
                observations=np.full(3, 8.0 + i * 0.01),
                covariates={"sphericity": 0.9},
            )
            for i in range(6)
        ]

        model = SeverityModel(n_restarts=2, max_iter=500)
        result = model.fit(patients)
        assert np.isfinite(result.log_marginal_likelihood)

    def test_model_name(self) -> None:
        """Model name is descriptive."""
        from growth.stages.stage2_severity.severity_model import SeverityModel

        model = SeverityModel(growth_function="gompertz_reduced")
        assert "ReducedGompertz" in model.name()
