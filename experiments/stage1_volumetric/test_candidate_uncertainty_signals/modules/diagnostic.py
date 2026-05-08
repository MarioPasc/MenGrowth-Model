"""Stage 1 — information-content diagnostic.

For each candidate signal $c_k$ at scan $k$, correlate against the
held-out homoscedastic LOPO residual $r_k = |y_k - \\hat\\mu_k^{\\text{homo}}|$
loaded from the ``LME_baseline/lopo_results.json`` of the main
experiment. Reports Pearson, Spearman, Kendall τ + 95 % BCa CI bootstrapped
at the patient level (paired with each held-out scan).

This is the cheap "is the signal informative at all" test that runs
before the expensive Stage 2 LOPO sweep.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_homo_residuals(lopo_json: Path) -> pd.DataFrame:
    """Return one row per held-out prediction with abs residual + metadata.

    Columns: patient_id, time, mu, y, abs_resid, sigma_v_sq_target.
    """
    with open(lopo_json) as f:
        data = json.load(f)
    rows: list[dict] = []
    for fr in data["fold_results"]:
        pid = fr["patient_id"]
        preds = fr.get("predictions", {}).get("last_from_rest", [])
        for p in preds:
            rows.append(
                {
                    "patient_id": pid,
                    "time": float(p["time"]),
                    "mu": float(p["pred_mean"]),
                    "y": float(p["actual"]),
                    "abs_resid": float(abs(p["pred_mean"] - p["actual"])),
                    "sigma_v_sq_target": float(p.get("sigma_v_sq_target", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def join_with_candidates(
    homo_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    candidate_names: Iterable[str],
) -> pd.DataFrame:
    """Left-join homo residuals with per-scan candidate values.

    The candidate CSV has one row per scan in H5 order. For LOPO
    last-from-rest, the held-out scan is the patient's last timepoint.
    We pick the maximum ``timepoint_idx`` per patient from the candidate
    CSV — that matches the LOPO held-out scan.
    """
    held_out = (
        candidates_df.sort_values(["patient_id", "timepoint_idx"])
        .groupby("patient_id", as_index=False)
        .tail(1)
    )
    keep = ["patient_id", "timepoint_idx", "scan_idx_in_h5", *candidate_names]
    merged = homo_df.merge(held_out[keep], on="patient_id", how="left")
    if merged.isna().any().any():
        logger.warning(
            "Some homo rows could not be matched to candidate rows: %d unmatched",
            int(merged[list(candidate_names)].isna().any(axis=1).sum()),
        )
    return merged


# ---------------------------------------------------------------------------
# Correlation + bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationResult:
    candidate: str
    n: int
    pearson_r: float
    pearson_ci: tuple[float, float]
    spearman_rho: float
    spearman_ci: tuple[float, float]
    kendall_tau: float
    kendall_ci: tuple[float, float]
    r2_linear: float


def _bca_ci(boot: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Quick bootstrap percentile CI (BCa would need the influence-fn jackknife;
    percentile CI is adequate for the diagnostic and matches main_experiment
    usage when n is small). We document this honestly in the README.
    """
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return lo, hi


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.allclose(np.var(x), 0.0) or np.allclose(np.var(y), 0.0):
        return float("nan")
    r, _ = fn(x, y)
    if np.isnan(r):
        return float("nan")
    return float(r)


def correlate_one(
    candidate: str,
    df: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> CorrelationResult:
    """Correlations between ``df[candidate]`` and ``df.abs_resid``."""
    sub = df[[candidate, "abs_resid"]].dropna()
    x = sub[candidate].to_numpy(dtype=np.float64)
    y = sub["abs_resid"].to_numpy(dtype=np.float64)
    n = int(x.size)

    if n < 3:
        return CorrelationResult(
            candidate=candidate,
            n=n,
            pearson_r=float("nan"),
            pearson_ci=(float("nan"), float("nan")),
            spearman_rho=float("nan"),
            spearman_ci=(float("nan"), float("nan")),
            kendall_tau=float("nan"),
            kendall_ci=(float("nan"), float("nan")),
            r2_linear=float("nan"),
        )

    pearson_r = _safe_corr(pearsonr, x, y)
    spearman_rho = _safe_corr(spearmanr, x, y)
    kendall_tau = _safe_corr(kendalltau, x, y)
    if x.var() > 0:
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        r2 = float("nan")

    rng = np.random.default_rng(seed)
    p_boot = np.empty(n_bootstrap, dtype=np.float64)
    s_boot = np.empty(n_bootstrap, dtype=np.float64)
    k_boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        p_boot[b] = _safe_corr(pearsonr, xb, yb)
        s_boot[b] = _safe_corr(spearmanr, xb, yb)
        k_boot[b] = _safe_corr(kendalltau, xb, yb)

    return CorrelationResult(
        candidate=candidate,
        n=n,
        pearson_r=pearson_r,
        pearson_ci=_bca_ci(p_boot[~np.isnan(p_boot)])
        if np.isfinite(pearson_r)
        else (float("nan"),) * 2,
        spearman_rho=spearman_rho,
        spearman_ci=_bca_ci(s_boot[~np.isnan(s_boot)])
        if np.isfinite(spearman_rho)
        else (float("nan"),) * 2,
        kendall_tau=kendall_tau,
        kendall_ci=_bca_ci(k_boot[~np.isnan(k_boot)])
        if np.isfinite(kendall_tau)
        else (float("nan"),) * 2,
        r2_linear=float(r2),
    )


def correlate_all(
    df: pd.DataFrame,
    candidate_names: Iterable[str],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for c in candidate_names:
        if c not in df.columns:
            logger.warning("candidate %s not in joined dataframe — skipping", c)
            continue
        res = correlate_one(c, df, n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            {
                "candidate": res.candidate,
                "n": res.n,
                "pearson_r": res.pearson_r,
                "pearson_lo": res.pearson_ci[0],
                "pearson_hi": res.pearson_ci[1],
                "spearman_rho": res.spearman_rho,
                "spearman_lo": res.spearman_ci[0],
                "spearman_hi": res.spearman_ci[1],
                "kendall_tau": res.kendall_tau,
                "kendall_lo": res.kendall_ci[0],
                "kendall_hi": res.kendall_ci[1],
                "r2_linear": res.r2_linear,
            }
        )
    return pd.DataFrame(rows)
