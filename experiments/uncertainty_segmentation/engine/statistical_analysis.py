# experiments/uncertainty_segmentation/engine/statistical_analysis.py
"""Statistical analysis for LoRA-Ensemble evaluation.

Computes:
    - Bootstrap 95% CIs on Dice (BCa or percentile method)
    - Paired Wilcoxon signed-rank test (ensemble vs. baseline)
    - Cohen's d (paired effect size)
    - ICC(3,1) for inter-member agreement

References:
    Efron & Tibshirani (1993). An Introduction to the Bootstrap.
    Shrout & Fleiss (1979). Intraclass Correlations: Uses in Assessing Rater
        Reliability. Psychological Bulletin.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Uses the percentile method. For small N, BCa would be better but
    the percentile method is simpler and sufficient for N_test > 50.

    Args:
        values: 1D array of per-subject values.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (default: 0.05 for 95% CI).
        seed: Random seed.

    Returns:
        (mean, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        means[b] = values[idx].mean()

    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lower), float(upper)


# =============================================================================
# Paired Statistical Tests
# =============================================================================


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute paired Cohen's d (effect size for paired samples).

    d = mean(a - b) / std(a - b)

    Interpretation:
        |d| < 0.2: negligible
        0.2 - 0.5: small
        0.5 - 0.8: medium
        > 0.8: large

    Args:
        a: Per-subject values for method A.
        b: Per-subject values for method B.

    Returns:
        Cohen's d (positive if A > B).
    """
    diff = a - b
    std = diff.std(ddof=1)
    if std < 1e-10:
        return 0.0
    return float(diff.mean() / std)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Run Wilcoxon signed-rank test on paired differences.

    H0: median(a - b) = 0.

    Args:
        a: Per-subject values for method A.
        b: Per-subject values for method B.

    Returns:
        Dict with 'statistic', 'p_value', 'effect_size_r'.
    """
    diff = a - b
    # Filter zero differences (Wilcoxon can't handle them)
    nonzero = np.abs(diff) > 1e-10
    if nonzero.sum() < 3:
        logger.warning("Too few non-zero differences for Wilcoxon test")
        return {"statistic": float("nan"), "p_value": 1.0, "effect_size_r": 0.0}

    result = stats.wilcoxon(diff[nonzero], alternative="two-sided")
    n = nonzero.sum()
    # Effect size r = Z / sqrt(N)
    # Approximate Z from p-value
    z = stats.norm.ppf(1 - result.pvalue / 2) if result.pvalue < 1.0 else 0.0
    r = z / np.sqrt(n)

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size_r": float(r),
    }


# =============================================================================
# Inter-member Agreement
# =============================================================================


def compute_icc(
    data: np.ndarray,
) -> float:
    """Compute ICC(3,1) — two-way mixed, single measures, consistency.

    ICC(3,1) = (MS_R - MS_E) / (MS_R + (k-1) * MS_E)

    where:
        MS_R = mean square for rows (subjects)
        MS_E = mean square error (residual)
        k = number of raters (members)

    Args:
        data: Array [n_subjects, k_raters] of ratings.

    Returns:
        ICC(3,1) value in [-1, 1]. Values > 0.75 indicate excellent agreement.
    """
    n, k = data.shape
    if k < 2:
        return 1.0

    # Grand mean
    grand_mean = data.mean()

    # Row means (subjects)
    row_means = data.mean(axis=1)

    # Column means (raters)
    col_means = data.mean(axis=0)

    # Sum of squares
    SS_total = np.sum((data - grand_mean) ** 2)
    SS_rows = k * np.sum((row_means - grand_mean) ** 2)
    SS_cols = n * np.sum((col_means - grand_mean) ** 2)
    SS_error = SS_total - SS_rows - SS_cols

    # Mean squares
    df_rows = n - 1
    df_error = (n - 1) * (k - 1)

    if df_rows == 0 or df_error == 0:
        return float("nan")

    MS_R = SS_rows / df_rows
    MS_E = SS_error / df_error

    # ICC(3,1)
    denom = MS_R + (k - 1) * MS_E
    if abs(denom) < 1e-10:
        return float("nan")

    icc = (MS_R - MS_E) / denom
    return float(icc)


# =============================================================================
# Full Statistical Summary
# =============================================================================


def compute_statistical_summary(
    per_member_dice: pd.DataFrame,
    ensemble_dice: pd.DataFrame,
    baseline_dice: pd.DataFrame,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute all statistical tests and return structured summary.

    Args:
        per_member_dice: Per-member × per-subject Dice (from evaluate_per_member).
        ensemble_dice: Ensemble Dice per subject (from evaluate_ensemble_per_subject).
        baseline_dice: Frozen BSF Dice per subject (from evaluate_baseline).
        n_bootstrap: Bootstrap iterations.
        alpha: Significance level.

    Returns:
        Dict matching the statistical_summary.json schema.
    """
    result: dict[str, Any] = {}

    # Align subjects between ensemble and baseline
    common_scans = set(ensemble_dice["scan_id"]) & set(baseline_dice["scan_id"])
    if len(common_scans) == 0:
        logger.error("No common scans between ensemble and baseline")
        return {"error": "no_common_scans"}

    ens = ensemble_dice[ensemble_dice["scan_id"].isin(common_scans)].sort_values("scan_id")
    bas = baseline_dice[baseline_dice["scan_id"].isin(common_scans)].sort_values("scan_id")

    # --- Ensemble vs Baseline ---
    ens_vs_bas = {}
    for ch in ["dice_tc", "dice_wt", "dice_et"]:
        a = ens[ch].values
        b = bas[ch].values

        ens_mean, ens_ci_lo, ens_ci_hi = bootstrap_ci(a, n_bootstrap, alpha)
        bas_mean, bas_ci_lo, bas_ci_hi = bootstrap_ci(b, n_bootstrap, alpha)

        # Delta CI via bootstrap
        delta = a - b
        delta_mean, delta_ci_lo, delta_ci_hi = bootstrap_ci(delta, n_bootstrap, alpha)

        wilcoxon = paired_wilcoxon(a, b)
        d = paired_cohens_d(a, b)

        ens_vs_bas[ch.replace("dice_", "")] = {
            "ensemble_mean": ens_mean,
            "ensemble_ci95": [ens_ci_lo, ens_ci_hi],
            "baseline_mean": bas_mean,
            "baseline_ci95": [bas_ci_lo, bas_ci_hi],
            "delta": delta_mean,
            "ci_95_lower": delta_ci_lo,
            "ci_95_upper": delta_ci_hi,
            "p_value_wilcoxon": wilcoxon["p_value"],
            "cohens_d": d,
        }

    result["ensemble_vs_baseline"] = ens_vs_bas

    # --- Ensemble vs Best Member ---
    member_ids = sorted(per_member_dice["member_id"].unique())
    member_mean_wt = {}
    for m in member_ids:
        m_dice = per_member_dice[per_member_dice["member_id"] == m]
        member_mean_wt[m] = m_dice["dice_wt"].mean()

    best_member_id = max(member_mean_wt, key=member_mean_wt.get)
    best_member_dice = (
        per_member_dice[per_member_dice["member_id"] == best_member_id]
        .sort_values("scan_id")
    )

    # Align
    common_ens_best = set(ens["scan_id"]) & set(best_member_dice["scan_id"])
    ens_aligned = ens[ens["scan_id"].isin(common_ens_best)].sort_values("scan_id")
    best_aligned = best_member_dice[best_member_dice["scan_id"].isin(common_ens_best)].sort_values("scan_id")

    ens_vs_best = {"best_member_id": int(best_member_id)}
    for ch in ["dice_tc", "dice_wt", "dice_et"]:
        a = ens_aligned[ch].values
        b = best_aligned[ch].values

        delta = a - b
        delta_mean, delta_ci_lo, delta_ci_hi = bootstrap_ci(delta, n_bootstrap, alpha)
        wilcoxon = paired_wilcoxon(a, b)
        d = paired_cohens_d(a, b)

        ens_vs_best[ch.replace("dice_", "")] = {
            "ensemble_mean": float(a.mean()),
            "best_member_mean": float(b.mean()),
            "delta": delta_mean,
            "ci_95_lower": delta_ci_lo,
            "ci_95_upper": delta_ci_hi,
            "p_value_wilcoxon": wilcoxon["p_value"],
            "cohens_d": d,
        }

    result["ensemble_vs_best_member"] = ens_vs_best

    # --- Inter-member Agreement ---
    agreement: dict[str, Any] = {}
    for ch in ["dice_wt", "dice_tc", "dice_et"]:
        # Build [n_subjects, k_members] matrix
        pivot = per_member_dice.pivot_table(
            index="scan_id", columns="member_id", values=ch
        ).dropna()

        if pivot.shape[1] >= 2:
            icc = compute_icc(pivot.values)
            agreement[f"icc_{ch.replace('dice_', '')}"] = icc

    # Mean pairwise Dice between members (on WT)
    pivot_wt = per_member_dice.pivot_table(
        index="scan_id", columns="member_id", values="dice_wt"
    ).dropna()
    if pivot_wt.shape[1] >= 2:
        # Mean correlation between members
        corr = np.corrcoef(pivot_wt.values.T)
        # Mask diagonal
        mask = ~np.eye(corr.shape[0], dtype=bool)
        agreement["mean_pairwise_correlation_wt"] = float(corr[mask].mean())

    result["inter_member_agreement"] = agreement

    # --- Per-member Summary ---
    per_member_summary = []
    for m in member_ids:
        m_df = per_member_dice[per_member_dice["member_id"] == m]
        for ch in ["dice_wt"]:
            vals = m_df[ch].values
            mean_val, ci_lo, ci_hi = bootstrap_ci(vals, n_bootstrap, alpha)
            per_member_summary.append({
                "member_id": int(m),
                f"{ch}_mean": mean_val,
                f"{ch}_std": float(vals.std()),
                f"{ch}_ci95": [ci_lo, ci_hi],
            })

    result["per_member_summary"] = per_member_summary

    return result
