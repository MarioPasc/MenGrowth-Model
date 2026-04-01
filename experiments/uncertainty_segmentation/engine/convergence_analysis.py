# experiments/uncertainty_segmentation/engine/convergence_analysis.py
"""Convergence analysis for LoRA ensemble.

Computes running statistics (mean, SE, median, MAD) as a function of
ensemble size k = 1..M to empirically verify the Law of Large Numbers
convergence rate: SE(V̄_M) = σ_V / √M.

Pure numpy/pandas — no GPU or model loading required. Operates on the
per-member volumes already stored in the volume CSV.

Reference:
    Efron & Tibshirani (1993). An Introduction to the Bootstrap.
"""

import logging
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)


def compute_convergence_curve(
    per_member_values: list[float],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute convergence curve over ensemble size k = 1..M.

    For each k, uses the first k values to compute running statistics.
    This verifies LLN convergence empirically.

    Args:
        per_member_values: List of M scalar values (volumes, Dice, etc.)
            in member order (seed order).
        alpha: Significance level for CIs (default: 0.05 → 95% CI).

    Returns:
        DataFrame with columns: k, running_mean, running_se,
        running_ci_lower, running_ci_upper, running_median,
        running_mad, running_mad_scaled.
    """
    M = len(per_member_values)
    rows: list[dict] = []

    for k in range(1, M + 1):
        subset = per_member_values[:k]
        mean_k = sum(subset) / k

        if k >= 2:
            std_k = (sum((v - mean_k) ** 2 for v in subset) / (k - 1)) ** 0.5
            se_k = std_k / math.sqrt(k)
            # t-distribution CI (exact for small k)
            t_crit = t_dist.ppf(1 - alpha / 2, df=k - 1)
            ci_lower = mean_k - t_crit * se_k
            ci_upper = mean_k + t_crit * se_k
        else:
            std_k = float("nan")
            se_k = float("nan")
            ci_lower = float("nan")
            ci_upper = float("nan")

        median_k = float(statistics.median(subset))
        mad_k = float(statistics.median([abs(v - median_k) for v in subset]))

        rows.append({
            "k": k,
            "running_mean": mean_k,
            "running_std": std_k,
            "running_se": se_k,
            "running_ci_lower": ci_lower,
            "running_ci_upper": ci_upper,
            "running_median": median_k,
            "running_mad": mad_k,
            "running_mad_scaled": 1.4826 * mad_k,
        })

    return pd.DataFrame(rows)


def compute_convergence_summary(
    volume_csv_path: Path | str,
) -> pd.DataFrame:
    """Compute mean convergence curve across all scans.

    Reads per-member volume columns (vol_m0, vol_m1, ...) from a volume CSV,
    computes a convergence curve per scan, then averages across scans.

    Args:
        volume_csv_path: Path to volume CSV with vol_m* columns.

    Returns:
        DataFrame with columns: k, mean_running_se, mean_running_mad_scaled,
        std_running_se, n_scans.
    """
    df = pd.read_csv(volume_csv_path)

    # Find per-member volume columns
    vol_cols = sorted(
        [c for c in df.columns if c.startswith("vol_m") and c[5:].isdigit()],
        key=lambda c: int(c[5:]),
    )
    M = len(vol_cols)
    if M < 2:
        logger.warning(f"Only {M} member columns found; need ≥ 2 for convergence")
        return pd.DataFrame()

    logger.info(f"Computing convergence curves for {len(df)} scans, M={M}")

    all_curves: list[pd.DataFrame] = []
    for _, row in df.iterrows():
        volumes = [row[c] for c in vol_cols]
        curve = compute_convergence_curve(volumes)
        all_curves.append(curve)

    # Stack and aggregate (filter k >= 2: SE undefined for k=1)
    stacked = pd.concat(all_curves, ignore_index=True)
    stacked = stacked[stacked["k"] >= 2]
    summary = stacked.groupby("k").agg(
        mean_running_se=("running_se", "mean"),
        std_running_se=("running_se", "std"),
        mean_running_mad_scaled=("running_mad_scaled", "mean"),
        std_running_mad_scaled=("running_mad_scaled", "std"),
        mean_running_mean=("running_mean", "mean"),
        mean_running_median=("running_median", "mean"),
    ).reset_index()

    summary["n_scans"] = len(df)

    if not summary.empty:
        logger.info(
            f"Convergence: SE at k={int(summary['k'].iloc[0])} → k={int(summary['k'].iloc[-1])}: "
            f"{summary['mean_running_se'].iloc[0]:.1f} → "
            f"{summary['mean_running_se'].iloc[-1]:.1f}"
        )

    return summary
