# experiments/stage1_volumetric/stats/calibration.py
"""Calibration metrics (DSS, NLPD, PIT) for probabilistic growth predictions."""

from __future__ import annotations

import numpy as np

from growth.shared.lopo import LOPOResults
from growth.shared.metrics import (
    compute_dawid_sebastiani,
    compute_log_score,
    compute_pit,
)

from .comparisons import extract_lopo_predictions


def compute_calibration_metrics(
    results: LOPOResults,
    protocol: str = "last_from_rest",
) -> dict[str, float | np.ndarray]:
    """Compute DSS, NLPD, PIT values, and KS test from LOPO results.

    Args:
        results: LOPO-CV results for one model.
        protocol: Prediction protocol to use.

    Returns:
        Dict with 'dss', 'nlpd', 'pit_ks_stat', 'pit_ks_p', and
        'pit_values'. Empty dict if no predictions available.
    """
    _, y_true, y_pred, pred_var, _, _ = extract_lopo_predictions(results, protocol)
    if len(y_true) == 0:
        return {}

    sigma_sq = np.maximum(pred_var, 1e-15)
    sigma = np.sqrt(sigma_sq)

    dss = compute_dawid_sebastiani(y_true, y_pred, sigma_sq)
    nlpd = compute_log_score(y_true, y_pred, sigma_sq)
    pit_vals = compute_pit(y_true, y_pred, sigma)

    from scipy.stats import kstest

    ks_stat, ks_p = kstest(pit_vals, "uniform")

    return {
        "dss": dss,
        "nlpd": nlpd,
        "pit_ks_stat": float(ks_stat),
        "pit_ks_p": float(ks_p),
        "pit_values": pit_vals,
    }
