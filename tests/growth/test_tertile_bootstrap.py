# tests/growth/test_tertile_bootstrap.py
"""Tests for per-tertile paired-bootstrap p-values.

Markers: unit
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from experiments.stage1_volumetric.stats.tertile_bootstrap import (
    _extract_aligned,
    paired_bootstrap_tertile,
)
from growth.shared.lopo import (
    LOPOFoldResult,
    LOPOResults,
)
from growth.shared.growth_models import FitResult

pytestmark = [pytest.mark.unit]


def _make_results(
    n_patients: int,
    sigma_v_target: np.ndarray,
    pred_mean: np.ndarray,
    actual: np.ndarray,
    pred_var: np.ndarray,
    name: str,
) -> LOPOResults:
    """Construct a synthetic LOPOResults for testing."""
    fold_results: list[LOPOFoldResult] = []
    for i in range(n_patients):
        sigma = float(np.sqrt(max(pred_var[i], 1e-15)))
        lo = float(pred_mean[i] - 1.96 * sigma)
        hi = float(pred_mean[i] + 1.96 * sigma)
        fr = LOPOFoldResult(
            patient_id=f"P{i:03d}",
            n_timepoints=3,
            n_train_patients=n_patients - 1,
            n_train_observations=(n_patients - 1) * 3,
            fit_result=FitResult(
                log_marginal_likelihood=0.0,
                hyperparameters={},
                condition_number=1.0,
            ),
            predictions={
                "last_from_rest": [
                    {
                        "time": 1.0,
                        "pred_mean": float(pred_mean[i]),
                        "pred_var": float(pred_var[i]),
                        "actual": float(actual[i]),
                        "lower_95": lo,
                        "upper_95": hi,
                        "n_conditioning": 2,
                        "sigma_v_sq_target": float(sigma_v_target[i]),
                    }
                ]
            },
        )
        fold_results.append(fr)
    return LOPOResults(model_name=name, fold_results=fold_results, aggregate_metrics={})


def test_extract_aligned_returns_paired_arrays() -> None:
    """``_extract_aligned`` returns aligned arrays of expected shape."""
    rng = np.random.default_rng(0)
    n = 30
    sv = rng.exponential(0.1, size=n)
    pm = rng.normal(8.0, 1.0, size=n)
    pa = pm + rng.normal(0.0, 0.5, size=n)
    pv = rng.uniform(0.5, 1.5, size=n)
    res = _make_results(n, sv, pm, pa, pv, "TestModel")

    arrays = _extract_aligned(res)
    assert arrays is not None
    assert arrays["pids"].shape == (n,)
    assert arrays["y_true"].shape == (n,)
    assert arrays["y_pred"].shape == (n,)
    assert arrays["pred_var"].shape == (n,)
    assert arrays["lower"].shape == (n,)
    assert arrays["upper"].shape == (n,)
    assert arrays["sigma_v_sq"].shape == (n,)


def test_paired_bootstrap_returns_three_metrics_three_tertiles() -> None:
    """Returned dict has r2_log, is_95, coverage_95 each with 3 tertiles."""
    rng = np.random.default_rng(1)
    n = 60
    sv = rng.exponential(0.1, size=n)
    pm = rng.normal(8.0, 1.0, size=n)
    pa = pm + rng.normal(0.0, 0.5, size=n)
    pv = rng.uniform(0.5, 1.5, size=n)
    res_a = _make_results(n, sv, pm, pa, pv, "A")
    # Hetero: same predictions but slightly wider variance scaled by σ²_v.
    pv_b = pv + sv  # propagation: hetero adds σ²_v to predictive variance
    res_b = _make_results(n, sv, pm, pa, pv_b, "B")

    arr_a = _extract_aligned(res_a)
    arr_b = _extract_aligned(res_b)
    edges = (float(np.quantile(sv, 1 / 3)), float(np.quantile(sv, 2 / 3)))

    out = paired_bootstrap_tertile(arr_a, arr_b, edges=edges, n_bootstrap=200, seed=0)
    assert set(out.keys()) == {"r2_log", "is_95", "coverage_95"}
    for metric, results in out.items():
        assert len(results) == 3
        for r in results:
            assert r.tertile in {"low", "mid", "high"}


def test_p_values_are_in_unit_interval() -> None:
    """All returned p-values must be in [0, 1] or NaN."""
    rng = np.random.default_rng(2)
    n = 50
    sv = rng.exponential(0.05, size=n)
    pm = rng.normal(8.0, 1.0, size=n)
    pa = pm + rng.normal(0.0, 0.6, size=n)
    pv = rng.uniform(0.3, 1.0, size=n)
    res_a = _make_results(n, sv, pm, pa, pv, "A")
    pv_b = pv + sv
    res_b = _make_results(n, sv, pm, pa, pv_b, "B")
    arr_a = _extract_aligned(res_a)
    arr_b = _extract_aligned(res_b)
    edges = (float(np.quantile(sv, 1 / 3)), float(np.quantile(sv, 2 / 3)))

    out = paired_bootstrap_tertile(arr_a, arr_b, edges=edges, n_bootstrap=200, seed=0)
    for results in out.values():
        for r in results:
            if not np.isnan(r.p_value):
                assert 0.0 <= r.p_value <= 1.0


def test_widening_intervals_yields_negative_delta_is_on_high_tertile() -> None:
    """Adding σ²_v to pred_var widens intervals → IS may improve when
    the homo intervals were systematically too narrow on noisy targets.

    Build a scenario where homo under-covers the high tertile (large
    residual, narrow interval) and hetero widens correctly. ΔIS should
    be negative (hetero better) on the high tertile.
    """
    rng = np.random.default_rng(7)
    n = 90

    sv = np.concatenate(
        [
            rng.uniform(0.0, 0.001, size=n // 3),  # low
            rng.uniform(0.001, 0.01, size=n // 3),  # mid
            rng.uniform(0.5, 2.0, size=n - 2 * (n // 3)),  # high
        ]
    )
    rng.shuffle(sv)

    pm = rng.normal(8.0, 1.0, size=n)
    # Residuals scale with σ²_v: high-σ²_v scans have larger errors.
    residuals = rng.normal(0.0, np.sqrt(0.3 + 2.0 * sv))
    pa = pm + residuals
    # Homo: constant pred_var, intentionally too small at high σ²_v.
    pv_homo = np.full(n, 0.5)
    # Hetero: adds σ²_v, so widens at high σ²_v.
    pv_het = pv_homo + sv

    res_a = _make_results(n, sv, pm, pa, pv_homo, "A_homo")
    res_b = _make_results(n, sv, pm, pa, pv_het, "B_het")
    arr_a = _extract_aligned(res_a)
    arr_b = _extract_aligned(res_b)
    edges = (float(np.quantile(sv, 1 / 3)), float(np.quantile(sv, 2 / 3)))

    out = paired_bootstrap_tertile(arr_a, arr_b, edges=edges, n_bootstrap=500, seed=0)
    high_is = next(r for r in out["is_95"] if r.tertile == "high")
    assert high_is.delta < 0, f"hetero should reduce IS on high tertile, got {high_is.delta}"
    high_cov = next(r for r in out["coverage_95"] if r.tertile == "high")
    # Hetero should rescue coverage on the high tertile.
    assert high_cov.delta >= 0


def test_tertile_below_minimum_returns_nan_block() -> None:
    """A tertile with n<2 returns NaN for the Δ block (no test possible)."""
    rng = np.random.default_rng(3)
    n = 5  # so each tertile has roughly 1-2 patients
    sv = rng.exponential(0.05, size=n)
    pm = rng.normal(8.0, 1.0, size=n)
    pa = pm + rng.normal(0.0, 0.5, size=n)
    pv = rng.uniform(0.5, 1.0, size=n)
    res_a = _make_results(n, sv, pm, pa, pv, "A")
    res_b = _make_results(n, sv, pm, pa, pv, "B")
    arr_a = _extract_aligned(res_a)
    arr_b = _extract_aligned(res_b)
    # Force degenerate tertile cuts by using extreme quantiles.
    edges = (sv.max() - 1e-9, sv.max() + 1e-9)

    out = paired_bootstrap_tertile(arr_a, arr_b, edges=edges, n_bootstrap=50, seed=0)
    for r in out["is_95"]:
        if r.n < 2:
            assert np.isnan(r.delta)
            assert np.isnan(r.p_value)


def test_too_few_common_patients_raises() -> None:
    """`paired_bootstrap_tertile` requires ≥ 3 paired patients."""
    rng = np.random.default_rng(4)
    sv = rng.exponential(0.05, size=2)
    pm = rng.normal(8.0, 1.0, size=2)
    pa = pm + rng.normal(0.0, 0.5, size=2)
    pv = rng.uniform(0.5, 1.0, size=2)
    res_a = _make_results(2, sv, pm, pa, pv, "A")
    res_b = _make_results(2, sv, pm, pa, pv, "B")
    arr_a = _extract_aligned(res_a)
    arr_b = _extract_aligned(res_b)
    with pytest.raises(ValueError, match="common patients"):
        paired_bootstrap_tertile(arr_a, arr_b, edges=(0.0, 1.0), n_bootstrap=50, seed=0)
