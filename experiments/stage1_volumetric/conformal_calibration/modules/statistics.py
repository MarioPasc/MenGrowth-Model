"""Statistical testing for the conformal calibration experiment.

Battery:

1. Paired BCa bootstrap of ΔIS@95, Δcoverage@95, Δmean_width across seeds
   (marginal + per σ²_v tertile), B configurable (default 10,000).
2. Paired Wilcoxon signed-rank + Cohen's d on per-patient IS@95 contributions.
3. Benjamini–Hochberg FDR correction over three comparison families:
   - calibration_lift: within each base model, each conformal layer vs parametric.
   - model_lift: within parametric, lme_hetero & ensemble_bma vs lme_homo.
   - headline: jackknife_plus@lme_homo vs parametric@lme_homo;
               cqr_norm@lme_hetero vs parametric@lme_hetero.

References
----------
Barber et al., Ann. Stat. 49(1):486-507, 2021 — Jackknife+.
Efron & Tibshirani, *Bootstrap*, Ch. 13, 1993 — BCa.
Benjamini & Hochberg, JRSS-B 57(1), 1995 — BH-FDR.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BCa bootstrap helpers
# ---------------------------------------------------------------------------


def _bca_ci(
    boot_samples: np.ndarray,
    observed: float,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """BCa confidence interval for a bootstrap distribution.

    Parameters
    ----------
    boot_samples : np.ndarray
        Bootstrap replicates of the statistic (delta), shape ``[B]``.
    observed : float
        Point estimate (delta on original data).
    confidence_level : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    tuple[float, float]
        ``(ci_lower, ci_upper)``.
    """
    from scipy.stats import norm as _norm

    b = len(boot_samples)
    alpha = 1.0 - confidence_level

    def _percentile_ci() -> tuple[float, float]:
        """Plain percentile interval — the degenerate-case fallback."""
        return (
            float(np.percentile(boot_samples, 100.0 * alpha / 2.0)),
            float(np.percentile(boot_samples, 100.0 * (1.0 - alpha / 2.0))),
        )

    # Bias correction. Clamp the proportion away from {0, 1} so z0 stays finite
    # (a degenerate subset can put every replicate on one side of `observed`).
    prop_less = float(np.mean(boot_samples < observed))
    prop_less = min(max(prop_less, 1.0 / (b + 1)), 1.0 - 1.0 / (b + 1))
    z0 = float(_norm.ppf(prop_less))

    # Acceleration: jackknife of the bootstrap mean.
    jack = np.array([np.mean(np.delete(boot_samples, i)) for i in range(min(b, 200))])
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)

    if den < 1e-15 or not np.isfinite(z0):
        # Degenerate acceleration / bias term: fall back to the percentile CI.
        return _percentile_ci()
    a_hat = num / den

    def adjusted_alpha(p: float) -> float:
        z = _norm.ppf(p)
        return float(_norm.cdf(z0 + (z0 + z) / (1.0 - a_hat * (z0 + z))))

    lo_q = adjusted_alpha(alpha / 2.0)
    hi_q = adjusted_alpha(1.0 - alpha / 2.0)
    if not (np.isfinite(lo_q) and np.isfinite(hi_q)):
        return _percentile_ci()
    lo_q = float(np.clip(lo_q, 0.0, 1.0))
    hi_q = float(np.clip(hi_q, 0.0, 1.0))
    return float(np.percentile(boot_samples, 100.0 * lo_q)), float(
        np.percentile(boot_samples, 100.0 * hi_q)
    )


def _two_sided_p(boot_samples: np.ndarray, observed: float) -> float:
    """Two-sided bootstrap p-value for H0: delta = 0."""
    if np.isnan(observed) or (observed == 0.0 and np.allclose(boot_samples, 0.0)):
        return float("nan")
    if observed > 0:
        one_sided = float(np.mean(boot_samples <= 0.0))
    else:
        one_sided = float(np.mean(boot_samples >= 0.0))
    return float(min(2.0 * one_sided, 1.0))


# ---------------------------------------------------------------------------
# Per-patient IS@95 extraction
# ---------------------------------------------------------------------------


def _per_patient_is(fold_results: list[dict], layer: str, alpha: float = 0.05) -> dict[str, float]:
    """Extract per-patient IS@95 for one layer from fold result dicts.

    Args:
        fold_results: List of fold dicts from ``ConformalLOPOResults.to_dict()``.
        layer: Calibration layer key.
        alpha: Target miscoverage.

    Returns:
        Dict mapping patient_id to IS@95 for that layer.
    """
    out: dict[str, float] = {}
    for fr in fold_results:
        pid = fr["patient_id"]
        ivs = fr.get("intervals", {})
        if layer not in ivs:
            continue
        lo, hi = ivs[layer]
        y = float(fr["actual"])
        width = hi - lo
        penalty = (2.0 / alpha) * max(lo - y, 0.0) + (2.0 / alpha) * max(y - hi, 0.0)
        out[pid] = float(width + penalty)
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap for one (layer_a, layer_b) comparison
# ---------------------------------------------------------------------------


def bootstrap_pair(
    results_a: dict,
    results_b: dict,
    layer_a: str,
    layer_b: str,
    cuts: tuple[float, float],
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 12345,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired BCa bootstrap for ΔIS@95, Δcoverage@95, Δmean_width.

    Compares layer ``layer_b`` against layer ``layer_a`` (positive delta means
    layer_b is worse for IS@95 / width, better for coverage).

    Args:
        results_a: Dict from ``ConformalLOPOResults.to_dict()`` for arm A.
        results_b: Dict from ``ConformalLOPOResults.to_dict()`` for arm B.
            Must be from the same (base model, seed) run.
        layer_a: Reference calibration layer in arm A.
        layer_b: Comparison calibration layer in arm B.
        cuts: (q33, q66) tertile edges on the σ²_v distribution.
        n_bootstrap: Number of bootstrap replicates.
        confidence_level: BCa confidence level.
        seed: RNG seed.
        alpha: Miscoverage for IS computation.

    Returns:
        Dict with marginal and per-tertile bootstrap results for each metric.
    """
    folds_a = {fr["patient_id"]: fr for fr in results_a.get("fold_results", [])}
    folds_b = {fr["patient_id"]: fr for fr in results_b.get("fold_results", [])}
    common_pids = sorted(set(folds_a) & set(folds_b))
    if len(common_pids) < 3:
        return {"error": "fewer than 3 common patients"}

    def _extract(folds: dict[str, dict], layer: str, pid: str) -> tuple[float, float, float, float]:
        fr = folds[pid]
        ivs = fr.get("intervals", {})
        if layer not in ivs:
            return float("nan"), float("nan"), float("nan"), float("nan")
        lo, hi = ivs[layer]
        y = float(fr["actual"])
        sv2 = float(fr.get("sigma_v_sq_target", float("nan")))
        width = hi - lo
        penalty = (2.0 / alpha) * max(lo - y, 0.0) + (2.0 / alpha) * max(y - hi, 0.0)
        is_val = width + penalty
        covered = float(y >= lo and y <= hi)
        return is_val, covered, width, sv2

    is_a = np.array([_extract(folds_a, layer_a, p)[0] for p in common_pids])
    is_b = np.array([_extract(folds_b, layer_b, p)[0] for p in common_pids])
    cov_a = np.array([_extract(folds_a, layer_a, p)[1] for p in common_pids])
    cov_b = np.array([_extract(folds_b, layer_b, p)[1] for p in common_pids])
    w_a = np.array([_extract(folds_a, layer_a, p)[2] for p in common_pids])
    w_b = np.array([_extract(folds_b, layer_b, p)[2] for p in common_pids])
    sv2 = np.array([_extract(folds_a, layer_a, p)[3] for p in common_pids])

    rng = np.random.default_rng(seed)
    n = len(common_pids)
    out: dict[str, Any] = {}

    def _run_scope(mask: np.ndarray, scope_name: str) -> None:
        nm = int(mask.sum())
        if nm < 2:
            return
        d_is = is_b[mask] - is_a[mask]
        d_cov = cov_b[mask] - cov_a[mask]
        d_w = w_b[mask] - w_a[mask]
        obs_is = float(np.mean(d_is))
        obs_cov = float(np.mean(d_cov))
        obs_w = float(np.mean(d_w))

        boots_is = np.empty(n_bootstrap)
        boots_cov = np.empty(n_bootstrap)
        boots_w = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, nm, size=nm)
            boots_is[b] = float(np.mean(d_is[idx]))
            boots_cov[b] = float(np.mean(d_cov[idx]))
            boots_w[b] = float(np.mean(d_w[idx]))

        for metric_key, obs_val, boots in (
            ("is_95", obs_is, boots_is),
            ("coverage_95", obs_cov, boots_cov),
            ("mean_width", obs_w, boots_w),
        ):
            ci_lo, ci_hi = _bca_ci(boots, obs_val, confidence_level)
            p = _two_sided_p(boots, obs_val)
            out.setdefault(metric_key, []).append(
                {
                    "scope": scope_name,
                    "n": nm,
                    "delta": obs_val,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "p_value": p,
                    "n_bootstrap": n_bootstrap,
                    "confidence_level": confidence_level,
                }
            )

    # Marginal (all patients)
    _run_scope(np.ones(n, dtype=bool), "marginal")

    # Tertile
    q33, q66 = cuts
    valid = np.isfinite(sv2)
    _run_scope(valid & (sv2 <= q33), "tertile_low")
    _run_scope(valid & (sv2 > q33) & (sv2 <= q66), "tertile_mid")
    _run_scope(valid & (sv2 > q66), "tertile_high")

    return out


# ---------------------------------------------------------------------------
# Wilcoxon + Cohen's d
# ---------------------------------------------------------------------------


def wilcoxon_cohen_d(
    results: dict,
    layer_a: str,
    layer_b: str,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank + Cohen's d on per-patient ΔIS@95.

    Args:
        results: Dict from a single ``ConformalLOPOResults.to_dict()`` (both
            layers must be in the same run).
        layer_a: Reference layer.
        layer_b: Comparison layer.
        alpha: Miscoverage for IS computation.

    Returns:
        Dict with ``n``, ``wilcoxon_stat``, ``wilcoxon_p``, ``cohens_d``,
        ``delta_mean``, ``delta_std``.
    """
    is_a = _per_patient_is(results.get("fold_results", []), layer_a, alpha=alpha)
    is_b = _per_patient_is(results.get("fold_results", []), layer_b, alpha=alpha)
    common = sorted(set(is_a) & set(is_b))
    if len(common) < 3:
        return {"n": len(common), "wilcoxon_p": float("nan"), "cohens_d": float("nan")}

    delta = np.array([is_b[p] - is_a[p] for p in common], dtype=np.float64)
    if np.allclose(delta, 0.0):
        return {
            "n": len(common),
            "wilcoxon_stat": 0.0,
            "wilcoxon_p": 1.0,
            "cohens_d": 0.0,
            "delta_mean": 0.0,
            "delta_std": 0.0,
        }
    try:
        stat, p = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        w_stat, w_p = float(stat), float(p)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
    d = float(np.mean(delta)) / sd if sd > 0 else float("nan")
    return {
        "n": len(common),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "cohens_d": d,
        "delta_mean": float(np.mean(delta)),
        "delta_std": sd,
    }


# ---------------------------------------------------------------------------
# Cohen's d (paired)
# ---------------------------------------------------------------------------


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired differences ``b - a``.

    Args:
        a: Array of values for condition A.
        b: Array of values for condition B (same order as ``a``).

    Returns:
        Cohen's d (float, or nan if std == 0 or too few observations).
    """
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if len(delta) < 2:
        return float("nan")
    sd = float(np.std(delta, ddof=1))
    return float(np.mean(delta)) / sd if sd > 0 else float("nan")


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------


def bh_fdr(p_values: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR correction.

    Args:
        p_values: 1-D array of raw p-values (NaNs treated as 1.0).
        q: Target FDR.

    Returns:
        ``(rejected, p_adjusted)`` arrays of the same shape as ``p_values``.
    """
    p = np.asarray(p_values, dtype=np.float64).copy()
    finite = np.isfinite(p)
    p[~finite] = 1.0

    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    adj_sorted = ranked * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    p_adj = np.empty_like(adj_sorted)
    p_adj[order] = adj_sorted

    rejected = p_adj <= q
    rejected[~finite] = False
    return rejected, p_adj


# ---------------------------------------------------------------------------
# Multi-family statistics driver
# ---------------------------------------------------------------------------


def run_statistics(
    results_by_task: dict[str, dict],
    cohort_sigma_v_sq_flat: np.ndarray,
    cfg: dict,
) -> dict[str, Any]:
    """Compute all comparison families for the conformal experiment.

    Args:
        results_by_task: Dict mapping ``"{base_model}/seed_{NNN}/{layer}"`` keys
            to loaded ``ConformalLOPOResults.to_dict()`` dicts. In practice the
            aggregator pre-loads all per-task JSON files and passes them here.
        cohort_sigma_v_sq_flat: Full cohort σ²_v vector for tertile cuts.
        cfg: Full experiment config dict.

    Returns:
        Dict with ``bootstrap``, ``wilcoxon``, ``bh_fdr`` sub-dicts.
    """
    stat_cfg = cfg.get("statistics", {})
    boot_cfg = stat_cfg.get("bootstrap", {})
    n_bootstrap = int(boot_cfg.get("n_samples", 10000))
    confidence_level = float(boot_cfg.get("confidence_level", 0.95))
    boot_seed = int(boot_cfg.get("seed", 12345))
    bh_q = float(stat_cfg.get("bh_fdr_q", 0.05))
    alpha = float(cfg.get("conformal", {}).get("alpha", 0.05))
    conf_fam_cfg = stat_cfg.get("comparison_families", {})

    q33 = float(np.quantile(cohort_sigma_v_sq_flat, 1.0 / 3.0))
    q66 = float(np.quantile(cohort_sigma_v_sq_flat, 2.0 / 3.0))
    cuts = (q33, q66)

    bootstrap_rows: list[dict[str, Any]] = []
    wilcoxon_rows: list[dict[str, Any]] = []

    # Each results_by_task entry is keyed by "{base_model}/seed_{NNN}".
    for task_key, task_results in results_by_task.items():
        base_model, seed_str = task_key.rsplit("/", 1)
        try:
            seed = int(seed_str.split("_")[-1])
        except (ValueError, IndexError):
            seed = -1

        layers_present = list(task_results.get("layers", []))
        if "parametric" not in layers_present:
            continue

        # calibration_lift: each conformal layer vs parametric (within same model/seed)
        if conf_fam_cfg.get("calibration_lift", True):
            for layer_b in layers_present:
                if layer_b == "parametric":
                    continue
                br = bootstrap_pair(
                    task_results,
                    task_results,
                    "parametric",
                    layer_b,
                    cuts,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    seed=boot_seed + seed,
                    alpha=alpha,
                )
                for metric, entries in br.items():
                    if isinstance(entries, list):
                        for e in entries:
                            bootstrap_rows.append(
                                {
                                    "family": "calibration_lift",
                                    "base_model": base_model,
                                    "layer_a": "parametric",
                                    "layer_b": layer_b,
                                    "seed": seed,
                                    **e,
                                }
                            )
                if stat_cfg.get("wilcoxon", True):
                    wd = wilcoxon_cohen_d(task_results, "parametric", layer_b, alpha=alpha)
                    wilcoxon_rows.append(
                        {
                            "family": "calibration_lift",
                            "base_model": base_model,
                            "layer_a": "parametric",
                            "layer_b": layer_b,
                            "seed": seed,
                            **wd,
                        }
                    )

    # BH-FDR per family
    p_vals = np.array(
        [r["p_value"] for r in bootstrap_rows if "p_value" in r],
        dtype=np.float64,
    )
    rej, p_adj = bh_fdr(p_vals, q=bh_q) if len(p_vals) > 0 else (np.array([]), np.array([]))
    for i, row in enumerate(r for r in bootstrap_rows if "p_value" in r):
        row["p_adj_bh"] = float(p_adj[i]) if i < len(p_adj) else float("nan")
        row["rejected_bh"] = bool(rej[i]) if i < len(rej) else False

    return {
        "bootstrap": bootstrap_rows,
        "wilcoxon": wilcoxon_rows,
        "bh_fdr": {
            "q": bh_q,
            "n_tests": int(len(p_vals)),
            "n_rejected": int(np.sum(rej)) if len(rej) > 0 else 0,
        },
    }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def write_results(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON payload, converting numpy scalars.

    Args:
        payload: JSON-serialisable dict.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            payload,
            fh,
            indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    logger.info("Wrote %s", path)
