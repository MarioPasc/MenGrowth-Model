"""Statistical testing for the main experiment.

Battery (committed up front in :mod:`README`):

1. Paired bootstrap of ΔIS@95, ΔR², Δcov@95 (marginal + per σ²_v tertile),
   B=10,000, BCa CI, two-sided percentile p-value. Wraps the existing
   :func:`paired_bootstrap_tertile` from ``stats.tertile_bootstrap``.

2. Per-patient Wilcoxon signed-rank + Cohen's d on absolute residuals or
   per-patient IS contributions.

3. Spearman ρ between the sweep variable (π for the primary sweep) and the
   median across-seed ΔIS@95, testing monotonicity of the propagation effect.

4. Benjamini–Hochberg FDR correction over the family of p-values from
   bootstrap and Wilcoxon. Two FDR families are kept separate: marginal vs
   per-tertile, since they target different hypotheses.

References
----------
Efron & Tibshirani, *An Introduction to the Bootstrap*, Ch. 13, 1993.
Benjamini & Hochberg, JRSS-B 57(1) 1995 — multiple-testing FDR.
Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 1988 — d.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from experiments.stage1_volumetric.stats.tertile_bootstrap import (
    _extract_aligned,
    paired_bootstrap_tertile,
)
from growth.shared.lopo import LOPOResults
from growth.shared.metrics import compute_interval_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap (per cell, per tertile, per metric)
# ---------------------------------------------------------------------------


def run_bootstrap_for_pair(
    results_a: LOPOResults,
    results_b: LOPOResults,
    edges: tuple[float, float],
    *,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
    protocol: str = "last_from_rest",
) -> dict[str, list[dict[str, Any]]]:
    """Run paired bootstrap (marginal + per tertile) for one transition pair."""
    arr_a = _extract_aligned(results_a, protocol=protocol)
    arr_b = _extract_aligned(results_b, protocol=protocol)
    if arr_a is None or arr_b is None:
        raise RuntimeError("Could not extract aligned arrays from LOPO results")

    per_tertile = paired_bootstrap_tertile(
        arr_a,
        arr_b,
        edges=edges,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    out = {metric: [asdict(r) for r in entries] for metric, entries in per_tertile.items()}

    # Marginal: bootstrap over all common patients (no tertile mask).
    marginal = _marginal_bootstrap(
        arr_a,
        arr_b,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    for metric, entry in marginal.items():
        out.setdefault(metric, []).append(entry)
    return out


def _marginal_bootstrap(
    arr_a: dict[str, np.ndarray],
    arr_b: dict[str, np.ndarray],
    *,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, Any]]:
    from growth.shared.metrics import compute_coverage_at_levels, compute_r2

    pid_to_idx_a = {pid: i for i, pid in enumerate(arr_a["pids"])}
    pid_to_idx_b = {pid: i for i, pid in enumerate(arr_b["pids"])}
    common = sorted(set(pid_to_idx_a) & set(pid_to_idx_b))
    if len(common) < 3:
        raise RuntimeError("Need ≥3 common patients for marginal bootstrap")

    sel_a = np.asarray([pid_to_idx_a[p] for p in common])
    sel_b = np.asarray([pid_to_idx_b[p] for p in common])

    yt = arr_a["y_true"][sel_a]
    yp_a = arr_a["y_pred"][sel_a]
    yp_b = arr_b["y_pred"][sel_b]
    pv_a = arr_a["pred_var"][sel_a]
    pv_b = arr_b["pred_var"][sel_b]
    lo_a = arr_a["lower"][sel_a]
    lo_b = arr_b["lower"][sel_b]
    hi_a = arr_a["upper"][sel_a]
    hi_b = arr_b["upper"][sel_b]

    rng = np.random.default_rng(seed)
    n = len(common)
    boots = {"r2": np.empty(n_bootstrap), "is": np.empty(n_bootstrap), "cov": np.empty(n_bootstrap)}

    def _r2(y, yp):
        return compute_r2(y, yp) if len(y) >= 2 else float("nan")

    def _cov(y, yp, pv):
        s = np.sqrt(np.maximum(pv, 1e-15))
        return float(compute_coverage_at_levels(y, yp, s, levels=(0.95,))[0.95])

    def _is(y, lo, hi):
        return float(compute_interval_score(y, lo, hi, alpha=0.05))

    obs = {
        "r2_log": (_r2(yt, yp_a), _r2(yt, yp_b)),
        "is_95": (_is(yt, lo_a, hi_a), _is(yt, lo_b, hi_b)),
        "coverage_95": (_cov(yt, yp_a, pv_a), _cov(yt, yp_b, pv_b)),
    }

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots["r2"][b] = _r2(yt[idx], yp_b[idx]) - _r2(yt[idx], yp_a[idx])
        boots["is"][b] = _is(yt[idx], lo_b[idx], hi_b[idx]) - _is(yt[idx], lo_a[idx], hi_a[idx])
        boots["cov"][b] = _cov(yt[idx], yp_b[idx], pv_b[idx]) - _cov(yt[idx], yp_a[idx], pv_a[idx])

    alpha = 1.0 - confidence_level
    out: dict[str, dict[str, Any]] = {}
    for metric_key, key_out in (("r2", "r2_log"), ("is", "is_95"), ("cov", "coverage_95")):
        delta_obs = obs[key_out][1] - obs[key_out][0]
        samples = boots[metric_key]
        ci_lo = float(np.percentile(samples, 100 * alpha / 2))
        ci_hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        if np.isnan(delta_obs) or delta_obs == 0:
            p = float("nan")
        else:
            if delta_obs > 0:
                one_sided = float(np.mean(samples <= 0))
            else:
                one_sided = float(np.mean(samples >= 0))
            p = min(2.0 * one_sided, 1.0)
        out[key_out] = {
            "metric": key_out,
            "tertile": "marginal",
            "n": int(n),
            "value_a": float(obs[key_out][0]),
            "value_b": float(obs[key_out][1]),
            "delta": float(delta_obs),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": float(p),
            "n_bootstrap": int(n_bootstrap),
            "confidence_level": float(confidence_level),
        }
    return out


# ---------------------------------------------------------------------------
# Wilcoxon + Cohen's d on per-patient IS@95
# ---------------------------------------------------------------------------


def _per_patient_is(arr: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient IS@95 contribution (one scan per patient under last_from_rest)."""
    pids = np.asarray(arr["pids"])
    y = arr["y_true"]
    lo = arr["lower"]
    hi = arr["upper"]
    width = hi - lo
    alpha = 0.05
    is_per_obs = width + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    return pids, is_per_obs


def wilcoxon_cohen_d(
    results_a: LOPOResults,
    results_b: LOPOResults,
    *,
    protocol: str = "last_from_rest",
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank and Cohen's d on per-patient ΔIS@95."""
    arr_a = _extract_aligned(results_a, protocol=protocol)
    arr_b = _extract_aligned(results_b, protocol=protocol)

    pid_a, is_a = _per_patient_is(arr_a)
    pid_b, is_b = _per_patient_is(arr_b)

    # Pair by patient_id.
    pid_to_a = {p: v for p, v in zip(pid_a, is_a, strict=True)}
    pid_to_b = {p: v for p, v in zip(pid_b, is_b, strict=True)}
    common = sorted(set(pid_to_a) & set(pid_to_b))
    if len(common) < 3:
        return {"n": len(common), "wilcoxon_p": float("nan"), "cohens_d": float("nan")}

    delta = np.asarray([pid_to_b[p] - pid_to_a[p] for p in common], dtype=np.float64)

    if np.allclose(delta, 0.0):
        return {
            "n": int(len(common)),
            "wilcoxon_stat": 0.0,
            "wilcoxon_p": 1.0,
            "cohens_d": 0.0,
            "delta_mean": 0.0,
            "delta_std": 0.0,
        }

    try:
        stat, p = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        wilc_stat = float(stat)
        wilc_p = float(p)
    except ValueError:
        wilc_stat = float("nan")
        wilc_p = float("nan")

    sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
    d = float(np.mean(delta)) / sd if sd > 0 else float("nan")
    return {
        "n": int(len(common)),
        "wilcoxon_stat": wilc_stat,
        "wilcoxon_p": wilc_p,
        "cohens_d": d,
        "delta_mean": float(np.mean(delta)),
        "delta_std": sd,
    }


# ---------------------------------------------------------------------------
# Spearman trend across the sweep
# ---------------------------------------------------------------------------


def spearman_across_sweep(
    df: pd.DataFrame,
    *,
    metric: str,
    scope: str,
    tertile: str,
    transition_label: str,
    family: str = "lognormal_mixture",
) -> dict[str, float]:
    """Median across-seed delta vs sweep level → Spearman ρ.

    The dataframe must already contain a ``transition`` column with the
    delta of ``metric`` between the two arms of ``transition_label``.
    """
    sub = df[
        (df["family"] == family)
        & (df["scope"] == scope)
        & (df["tertile"] == tertile)
        & (df["metric"] == metric)
        & (df["transition"] == transition_label)
    ]
    if sub.empty:
        return {"rho": float("nan"), "p": float("nan"), "n_levels": 0}
    grouped = sub.groupby("level_value", as_index=False)["delta"].median()
    if len(grouped) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n_levels": int(len(grouped))}
    rho, p = spearmanr(grouped["level_value"], grouped["delta"])
    return {"rho": float(rho), "p": float(p), "n_levels": int(len(grouped))}


# ---------------------------------------------------------------------------
# Benjamini–Hochberg FDR
# ---------------------------------------------------------------------------


def bh_fdr(p_values: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg correction.

    Args:
        p_values: 1-D array of raw p-values (NaNs are treated as 1.0).
        q: Target FDR.

    Returns:
        ``(rejected, p_adjusted)`` of the same shape as ``p_values``.
    """
    p = np.asarray(p_values, dtype=np.float64).copy()
    finite = np.isfinite(p)
    p[~finite] = 1.0

    m = int(len(p))
    order = np.argsort(p)
    ranked = p[order]

    # BH-adjusted: p_adj_i = min over j>=i of m * p_j / rank_j, then cumulative min from the back.
    adj_sorted = ranked * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    p_adj = np.empty_like(adj_sorted)
    p_adj[order] = adj_sorted

    rejected = p_adj <= q
    rejected[~finite] = False
    return rejected, p_adj


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def write_results(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            payload,
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    logger.info("Wrote %s", path)
