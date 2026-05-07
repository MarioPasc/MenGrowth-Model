# tests/growth/test_conditional_calibration_r2.py
"""Sanity check that ``compute_conditional_calibration`` reports R²_log
per tertile and at the overall level.

Markers: unit
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stage1_volumetric.stats.conditional_calibration import (
    compute_conditional_calibration,
)
from growth.shared.growth_models import FitResult
from growth.shared.lopo import LOPOFoldResult, LOPOResults

pytestmark = [pytest.mark.unit]


def _build_results(n: int, seed: int = 0) -> LOPOResults:
    rng = np.random.default_rng(seed)
    sv = rng.exponential(0.05, size=n)
    pm = rng.normal(8.0, 1.0, size=n)
    pa = pm + rng.normal(0.0, 0.5, size=n)
    pv = rng.uniform(0.5, 1.5, size=n)
    fold_results: list[LOPOFoldResult] = []
    for i in range(n):
        sigma = float(np.sqrt(pv[i]))
        fold_results.append(
            LOPOFoldResult(
                patient_id=f"P{i:03d}",
                n_timepoints=3,
                n_train_patients=n - 1,
                n_train_observations=(n - 1) * 3,
                fit_result=FitResult(log_marginal_likelihood=0.0, hyperparameters={}, condition_number=1.0),
                predictions={
                    "last_from_rest": [
                        {
                            "time": 1.0,
                            "pred_mean": float(pm[i]),
                            "pred_var": float(pv[i]),
                            "actual": float(pa[i]),
                            "lower_95": float(pm[i] - 1.96 * sigma),
                            "upper_95": float(pm[i] + 1.96 * sigma),
                            "n_conditioning": 2,
                            "sigma_v_sq_target": float(sv[i]),
                        }
                    ]
                },
            )
        )
    return LOPOResults(model_name="Test", fold_results=fold_results, aggregate_metrics={})


def test_per_tertile_r2_present_and_finite() -> None:
    """Each tertile (with n>=2) should have a finite r2_log entry."""
    res = _build_results(n=60, seed=42)
    out = compute_conditional_calibration(res)
    assert out is not None
    for tname, tdata in out["tertiles"].items():
        if tdata.get("n", 0) >= 2:
            assert "r2_log" in tdata, f"{tname} missing r2_log"
            assert np.isfinite(tdata["r2_log"]), f"{tname} r2_log is NaN"
    assert "r2_log" in out["overall"]
    assert np.isfinite(out["overall"]["r2_log"])


def test_overall_r2_matches_global_compute() -> None:
    """The 'overall' R² in the conditional payload matches the standard formula."""
    res = _build_results(n=40, seed=7)
    out = compute_conditional_calibration(res)
    assert out is not None
    # Recompute the global R² from the raw data.
    yt = np.array([fr.predictions["last_from_rest"][0]["actual"] for fr in res.fold_results])
    yp = np.array([fr.predictions["last_from_rest"][0]["pred_mean"] for fr in res.fold_results])
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    expected = 1.0 - ss_res / ss_tot
    assert out["overall"]["r2_log"] == pytest.approx(expected, rel=1e-6)
