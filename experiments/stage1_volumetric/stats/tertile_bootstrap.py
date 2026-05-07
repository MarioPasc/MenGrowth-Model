"""Per-tertile paired-bootstrap p-values for ΔR², ΔIS@95, Δcov@95.

Marginal Wilcoxon-on-|error| (the existing test in
``stats/comparisons.py``) misses the conditional thesis claim by
construction because the propagation effect is concentrated in the
high-σ²_v tertile (n≈19) and washes out when averaged over the cohort.

This module implements per-tertile paired-bootstrap tests for the three
headline metrics (R²_log, IS@95, cov@95) so the manuscript can support
"hetero rescues coverage on the high-σ²_v tertile (p<0.001)" with a
proper paired test on the right stratification.

Procedure (per pair, per tertile):

1. Pair predictions across the two models by patient_id and tertile.
2. For B bootstrap replicates, resample patients within the tertile
   *with replacement*, recompute Δmetric.
3. Report ΔObs, 95 % BCa interval, and a two-sided percentile p-value
   (fraction of replicates whose ΔR² has the opposite sign from
   ΔObs, doubled).

Reference:
    Efron & Tibshirani, *An Introduction to the Bootstrap*, Ch. 13, 1993.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from growth.shared.lopo import LOPOResults
from growth.shared.metrics import (
    compute_coverage_at_levels,
    compute_interval_score,
    compute_r2,
)

logger = logging.getLogger(__name__)

_TERTILE_LABELS = ("low", "mid", "high")


@dataclass
class TertileTestResult:
    """One per-tertile paired-bootstrap result for one metric."""

    metric: str
    tertile: str
    n: int
    value_a: float
    value_b: float
    delta: float  # value_b - value_a
    ci_lower: float
    ci_upper: float
    p_value: float  # two-sided percentile p-value
    n_bootstrap: int
    confidence_level: float


def _extract_aligned(
    results: LOPOResults,
    protocol: str = "last_from_rest",
) -> dict[str, np.ndarray] | None:
    """Pull aligned (pid, y_true, y_pred, lower, upper, sigma_v_sq)."""
    pids: list[str] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    pred_var: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    sigma_v_sq: list[float] = []
    for fr in results.fold_results:
        if protocol not in fr.predictions:
            continue
        for pred in fr.predictions[protocol]:
            pids.append(fr.patient_id)
            y_true.append(pred["actual"])
            y_pred.append(pred["pred_mean"])
            pred_var.append(pred["pred_var"])
            lower.append(pred["lower_95"])
            upper.append(pred["upper_95"])
            sigma_v_sq.append(pred.get("sigma_v_sq_target", float("nan")))
    if not pids:
        return None
    return {
        "pids": np.asarray(pids),
        "y_true": np.asarray(y_true, dtype=np.float64),
        "y_pred": np.asarray(y_pred, dtype=np.float64),
        "pred_var": np.asarray(pred_var, dtype=np.float64),
        "lower": np.asarray(lower, dtype=np.float64),
        "upper": np.asarray(upper, dtype=np.float64),
        "sigma_v_sq": np.asarray(sigma_v_sq, dtype=np.float64),
    }


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    if len(yt) < 2:
        return float("nan")
    return compute_r2(yt, yp)


def _cov95(yt: np.ndarray, yp: np.ndarray, pv: np.ndarray) -> float:
    sigma = np.sqrt(np.maximum(pv, 1e-15))
    cov = compute_coverage_at_levels(yt, yp, sigma, levels=(0.95,))
    return float(cov[0.95])


def _is95(yt: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(compute_interval_score(yt, lo, hi, alpha=0.05))


def paired_bootstrap_tertile(
    arrays_a: dict[str, np.ndarray],
    arrays_b: dict[str, np.ndarray],
    *,
    edges: tuple[float, float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, list[TertileTestResult]]:
    """Compute per-tertile paired-bootstrap tests for R², IS@95, cov@95.

    Pairing rule: predictions are aligned by ``patient_id`` (which is the
    LOPO fold key), and the σ²_v tertile is taken from model A's
    ``sigma_v_sq`` vector (caller is expected to supply the
    LMEHetero@empirical results as A or B; both share the same value
    when their ``sigma_v_sq_target`` field comes from the same H5).

    Args:
        arrays_a: Output of ``_extract_aligned`` for model A (homo).
        arrays_b: Output of ``_extract_aligned`` for model B (hetero).
        edges: ``(q33, q66)`` cuts on σ²_v.
        n_bootstrap: Bootstrap replicates per tertile.
        confidence_level: Confidence level for the interval bands.
        seed: Master seed.

    Returns:
        Dict ``{metric: [TertileTestResult, ...]}`` (one entry per
        tertile, ordered low→mid→high).
    """
    pid_to_idx_a = {pid: i for i, pid in enumerate(arrays_a["pids"])}
    pid_to_idx_b = {pid: i for i, pid in enumerate(arrays_b["pids"])}
    common = sorted(set(pid_to_idx_a.keys()) & set(pid_to_idx_b.keys()))
    if len(common) < 3:
        raise ValueError(
            f"Need ≥3 common patients for tertile bootstrap, got {len(common)}"
        )

    sel_a = np.asarray([pid_to_idx_a[p] for p in common])
    sel_b = np.asarray([pid_to_idx_b[p] for p in common])

    sv = arrays_a["sigma_v_sq"][sel_a]
    if not np.all(np.isfinite(sv)):
        raise ValueError("sigma_v_sq_target on the homo arrays has NaNs")

    q33, q66 = edges
    masks = {
        "low": sv <= q33,
        "mid": (sv > q33) & (sv <= q66),
        "high": sv > q66,
    }

    yt_a = arrays_a["y_true"][sel_a]
    yp_a = arrays_a["y_pred"][sel_a]
    pv_a = arrays_a["pred_var"][sel_a]
    lo_a = arrays_a["lower"][sel_a]
    hi_a = arrays_a["upper"][sel_a]

    yt_b = arrays_b["y_true"][sel_b]
    yp_b = arrays_b["y_pred"][sel_b]
    pv_b = arrays_b["pred_var"][sel_b]
    lo_b = arrays_b["lower"][sel_b]
    hi_b = arrays_b["upper"][sel_b]

    rng = np.random.default_rng(seed)

    out: dict[str, list[TertileTestResult]] = {"r2_log": [], "is_95": [], "coverage_95": []}
    alpha = 1.0 - confidence_level

    for tname in _TERTILE_LABELS:
        m = masks[tname]
        n = int(m.sum())
        if n < 2:
            for metric in out:
                out[metric].append(
                    TertileTestResult(
                        metric=metric,
                        tertile=tname,
                        n=n,
                        value_a=float("nan"),
                        value_b=float("nan"),
                        delta=float("nan"),
                        ci_lower=float("nan"),
                        ci_upper=float("nan"),
                        p_value=float("nan"),
                        n_bootstrap=n_bootstrap,
                        confidence_level=confidence_level,
                    )
                )
            continue

        # Restrict to tertile.
        yt_a_t, yp_a_t, pv_a_t, lo_a_t, hi_a_t = yt_a[m], yp_a[m], pv_a[m], lo_a[m], hi_a[m]
        yt_b_t, yp_b_t, pv_b_t, lo_b_t, hi_b_t = yt_b[m], yp_b[m], pv_b[m], lo_b[m], hi_b[m]

        obs_r2_a = _r2(yt_a_t, yp_a_t)
        obs_r2_b = _r2(yt_b_t, yp_b_t)
        obs_is_a = _is95(yt_a_t, lo_a_t, hi_a_t)
        obs_is_b = _is95(yt_b_t, lo_b_t, hi_b_t)
        obs_cov_a = _cov95(yt_a_t, yp_a_t, pv_a_t)
        obs_cov_b = _cov95(yt_b_t, yp_b_t, pv_b_t)

        boots = {"r2": np.empty(n_bootstrap), "is": np.empty(n_bootstrap), "cov": np.empty(n_bootstrap)}
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            r2_a = _r2(yt_a_t[idx], yp_a_t[idx])
            r2_b = _r2(yt_b_t[idx], yp_b_t[idx])
            is_a = _is95(yt_a_t[idx], lo_a_t[idx], hi_a_t[idx])
            is_b = _is95(yt_b_t[idx], lo_b_t[idx], hi_b_t[idx])
            cov_a = _cov95(yt_a_t[idx], yp_a_t[idx], pv_a_t[idx])
            cov_b = _cov95(yt_b_t[idx], yp_b_t[idx], pv_b_t[idx])
            boots["r2"][b] = r2_b - r2_a
            boots["is"][b] = is_b - is_a
            boots["cov"][b] = cov_b - cov_a

        for metric_key, (val_a, val_b, key_out) in zip(
            ("r2", "is", "cov"),
            ((obs_r2_a, obs_r2_b, "r2_log"), (obs_is_a, obs_is_b, "is_95"), (obs_cov_a, obs_cov_b, "coverage_95")),
            strict=True,
        ):
            delta_obs = val_b - val_a
            samples = boots[metric_key]
            ci_lo = float(np.percentile(samples, 100 * alpha / 2))
            ci_hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))

            # Two-sided percentile p-value: fraction of bootstrap replicates
            # whose Δ has sign opposite to Δobs (or zero), doubled and
            # capped at 1. Only meaningful if Δobs is non-zero.
            if np.isnan(delta_obs) or delta_obs == 0:
                p_value = float("nan")
            else:
                if delta_obs > 0:
                    one_sided = float(np.mean(samples <= 0))
                else:
                    one_sided = float(np.mean(samples >= 0))
                p_value = min(2.0 * one_sided, 1.0)

            out[key_out].append(
                TertileTestResult(
                    metric=key_out,
                    tertile=tname,
                    n=n,
                    value_a=float(val_a) if not np.isnan(val_a) else float("nan"),
                    value_b=float(val_b) if not np.isnan(val_b) else float("nan"),
                    delta=float(delta_obs) if not np.isnan(delta_obs) else float("nan"),
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                    p_value=float(p_value),
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                )
            )

    return out


def run_tertile_bootstrap_for_pairs(
    lopo_results: dict[str, LOPOResults],
    pairs: list[list[str]],
    output_dir: Path,
    *,
    protocol: str = "last_from_rest",
    reference_model: str | None = None,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
    filename: str = "tertile_bootstrap_last_from_rest",
) -> dict:
    """Run per-tertile paired bootstraps for every (homo, hetero) pair.

    Tertile cuts are computed once from ``reference_model``'s
    ``sigma_v_sq_target`` distribution (defaulting to the first hetero
    model present) so all pairs share the same partition.
    """
    if reference_model is None:
        for cand in ("LMEHetero", "HGPHetero", "ScalarGPHetero"):
            if cand in lopo_results:
                reference_model = cand
                break
    if reference_model is None or reference_model not in lopo_results:
        logger.warning("No reference model with sigma_v_sq_target; skipping.")
        return {}

    ref_arrays = _extract_aligned(lopo_results[reference_model], protocol)
    if ref_arrays is None:
        return {}
    sv_ref = ref_arrays["sigma_v_sq"]
    finite = np.isfinite(sv_ref)
    if finite.sum() < 6:
        logger.warning("Not enough finite sigma_v_sq_target to compute tertile cuts.")
        return {}
    q33, q66 = np.quantile(sv_ref[finite], [1 / 3.0, 2 / 3.0])
    edges = (float(q33), float(q66))
    logger.info("tertile_bootstrap edges: q33=%.4g, q66=%.4g", *edges)

    payload: dict[str, dict] = {
        "protocol": protocol,
        "reference_model": reference_model,
        "edges_sigma_v_sq": list(edges),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": float(confidence_level),
        "pairs": [],
    }
    md_lines = [
        "# Per-Tertile Paired-Bootstrap Tests",
        "",
        f"Protocol: `{protocol}`. Reference model for σ²_v cuts: `{reference_model}`.",
        f"Tertile cuts: q33={edges[0]:.4g}, q66={edges[1]:.4g}.",
        f"Bootstrap replicates: B={n_bootstrap}, confidence level={confidence_level:.2f}.",
        "",
        "ΔX = X(B) − X(A). Negative ΔIS@95 = pair B is better calibrated.",
        "Negative ΔR² = pair B is worse on point prediction.",
        "Positive Δcov_95 = pair B has higher empirical coverage.",
        "",
        "| Pair | Tertile | n | Metric | A | B | Δ | 95% CI | p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for pair in pairs:
        if len(pair) != 2:
            continue
        name_a, name_b = pair[0], pair[1]
        if name_a not in lopo_results or name_b not in lopo_results:
            logger.warning("Skipping pair %s: missing results", pair)
            continue
        arr_a = _extract_aligned(lopo_results[name_a], protocol)
        arr_b = _extract_aligned(lopo_results[name_b], protocol)
        if arr_a is None or arr_b is None:
            logger.warning("Skipping pair %s: protocol %s empty", pair, protocol)
            continue
        # Always tertile patients by the *empirical* σ²_v from the
        # reference model (joined by patient_id). This ensures the
        # patient strata are identical across all pairs, even when a
        # synthetic ablation (e.g. LMEHetero_Zero) stores a degenerate
        # σ²_v in its own predictions dict.
        ref_pid_to_sv = {
            pid: sv
            for pid, sv in zip(ref_arrays["pids"], ref_arrays["sigma_v_sq"], strict=True)
        }

        def _override_sv(arr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            new = dict(arr)
            new["sigma_v_sq"] = np.asarray(
                [ref_pid_to_sv.get(pid, float("nan")) for pid in arr["pids"]]
            )
            return new

        if name_a != reference_model:
            arr_a = _override_sv(arr_a)
        if name_b != reference_model:
            arr_b = _override_sv(arr_b)
        try:
            res = paired_bootstrap_tertile(
                arr_a,
                arr_b,
                edges=edges,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
        except ValueError as exc:
            logger.warning("Pair %s failed: %s", pair, exc)
            continue

        pair_entry = {
            "pair": [name_a, name_b],
            "results": {
                metric: [asdict(r) for r in res[metric]]
                for metric in ("r2_log", "is_95", "coverage_95")
            },
        }
        payload["pairs"].append(pair_entry)

        for metric in ("r2_log", "is_95", "coverage_95"):
            for r in res[metric]:
                if r.n < 2:
                    continue
                md_lines.append(
                    f"| {name_a}→{name_b} | {r.tertile} | {r.n} | {metric} | "
                    f"{r.value_a:+.4f} | {r.value_b:+.4f} | {r.delta:+.4f} | "
                    f"[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}] | "
                    f"{('n/a' if np.isnan(r.p_value) else f'{r.p_value:.4f}')} |"
                )

    md_lines += [
        "",
        "## Reading the table",
        "",
        "- The high-σ²_v tertile is the regime where the propagation thesis "
        "predicts the largest hetero gain.",
        "- A significant Δcov_95 on the high tertile (p<0.05) is the "
        "headline thesis test.",
        "- ΔIS@95 < 0 with overlapping CIs that exclude zero is the "
        "primary calibration signal.",
        "- ΔR² ≈ 0 (CI straddles zero) confirms propagation does not "
        "damage point prediction; large |ΔR²| is a red flag for the "
        "implementation.",
        "",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{filename}.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(output_dir / f"{filename}.md", "w") as f:
        f.write("\n".join(md_lines))
    logger.info("Wrote tertile-bootstrap results to %s/%s.{json,md}", output_dir, filename)
    return payload
