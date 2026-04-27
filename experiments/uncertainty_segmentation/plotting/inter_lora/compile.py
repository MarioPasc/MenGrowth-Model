"""Build compiled_metrics.parquet — the single source of truth for all plots."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from experiments.uncertainty_segmentation.plotting.inter_lora.io_layer import RankRun

logger = logging.getLogger(__name__)

LABELS: list[str] = ["tc", "wt", "et"]
DICE_COLS: dict[str, str] = {
    "tc": "dice_tc",
    "wt": "dice_wt",
    "et": "dice_et",
}


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for j, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - j)
        corrected = min(corrected, 1.0)
        cummax = max(cummax, corrected)
        adjusted[orig_idx] = cummax
    return adjusted


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Cohen's d: mean(x-y) / std(x-y)."""
    diff = x - y
    sd = diff.std(ddof=1)
    if sd < 1e-15:
        return 0.0
    return float(diff.mean() / sd)


def _extract_row(
    run: RankRun,
    label: str,
    baseline_dice_col: pd.Series,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Extract metrics for one rank x label combination."""
    ss = run.statistical_summary
    col = DICE_COLS[label]

    evb = ss.get("ensemble_vs_baseline", {}).get(label, {})
    ensemble_mean = evb.get("ensemble_mean", np.nan)
    ci95 = evb.get("ensemble_ci95", [np.nan, np.nan])
    baseline_mean = evb.get("baseline_mean", np.nan)

    ens_dice = run.ensemble_dice[col].values
    bas_dice = baseline_dice_col.values

    delta = float(np.nanmean(ens_dice - bas_dice))

    boot_deltas = np.empty(n_boot)
    n = len(ens_dice)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[b] = np.mean(ens_dice[idx] - bas_dice[idx])
    delta_ci_lo = float(np.percentile(boot_deltas, 2.5))
    delta_ci_hi = float(np.percentile(boot_deltas, 97.5))

    try:
        _, p_raw = stats.wilcoxon(ens_dice, bas_dice, alternative="two-sided")
    except ValueError:
        p_raw = 1.0

    d = _cohens_d(ens_dice, bas_dice)

    calib = run.calibration
    ece = calib.get("ece", np.nan)
    brier = calib.get("brier_score", np.nan)

    taxonomy = run.epistemic_taxonomy
    calib_tax = taxonomy.get("calibration", {})
    cov95_deficit = calib_tax.get("coverage_deficit_95", np.nan)

    bias_dom = taxonomy.get("taxonomy", {}).get("estimation_bias", {}).get("bias_dominance", {})
    pct_bd = bias_dom.get("pct_scans_k_star_eq_1", 0.0) + bias_dom.get(
        "pct_scans_k_star_exceeds_M", 0.0
    )

    ima = ss.get("inter_member_agreement", {})
    icc = ima.get(f"icc_{label}", np.nan)

    return {
        "rank": run.rank,
        "label": label.upper(),
        "dice_mean": ensemble_mean,
        "dice_ci_lo": ci95[0] if len(ci95) >= 2 else np.nan,
        "dice_ci_hi": ci95[1] if len(ci95) >= 2 else np.nan,
        "delta_vs_baseline": delta,
        "delta_ci_lo": delta_ci_lo,
        "delta_ci_hi": delta_ci_hi,
        "p_wilcoxon_raw": p_raw,
        "p_wilcoxon_holm": np.nan,
        "cohens_d": d,
        "ece": ece,
        "brier": brier,
        "cov95_deficit": cov95_deficit,
        "pct_bias_dominated": pct_bd,
        "icc": icc,
    }


def _extract_baseline_row(
    run: RankRun,
    label: str,
) -> dict:
    """Extract baseline metrics (rank=0 pseudo-row)."""
    col = DICE_COLS[label]
    bas = run.baseline_dice[col].values
    mean_val = float(np.nanmean(bas))
    se = float(np.nanstd(bas, ddof=1) / np.sqrt(len(bas)))

    return {
        "rank": 0,
        "label": label.upper(),
        "dice_mean": mean_val,
        "dice_ci_lo": mean_val - 1.96 * se,
        "dice_ci_hi": mean_val + 1.96 * se,
        "delta_vs_baseline": 0.0,
        "delta_ci_lo": 0.0,
        "delta_ci_hi": 0.0,
        "p_wilcoxon_raw": 1.0,
        "p_wilcoxon_holm": 1.0,
        "cohens_d": 0.0,
        "ece": np.nan,
        "brier": np.nan,
        "cov95_deficit": np.nan,
        "pct_bias_dominated": np.nan,
        "icc": np.nan,
    }


def _extract_mean_row(rows: list[dict], rank: int) -> dict:
    """Compute the 'mean' label row by averaging across TC/WT/ET."""
    avg = {}
    numeric_keys = [
        "dice_mean",
        "dice_ci_lo",
        "dice_ci_hi",
        "delta_vs_baseline",
        "delta_ci_lo",
        "delta_ci_hi",
        "cohens_d",
        "icc",
    ]
    for key in numeric_keys:
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        avg[key] = float(np.mean(vals)) if vals else np.nan

    first = rows[0] if rows else {}
    return {
        "rank": rank,
        "label": "mean",
        "dice_mean": avg.get("dice_mean", np.nan),
        "dice_ci_lo": avg.get("dice_ci_lo", np.nan),
        "dice_ci_hi": avg.get("dice_ci_hi", np.nan),
        "delta_vs_baseline": avg.get("delta_vs_baseline", np.nan),
        "delta_ci_lo": avg.get("delta_ci_lo", np.nan),
        "delta_ci_hi": avg.get("delta_ci_hi", np.nan),
        "p_wilcoxon_raw": np.nan,
        "p_wilcoxon_holm": np.nan,
        "cohens_d": avg.get("cohens_d", np.nan),
        "ece": first.get("ece", np.nan),
        "brier": first.get("brier", np.nan),
        "cov95_deficit": first.get("cov95_deficit", np.nan),
        "pct_bias_dominated": first.get("pct_bias_dominated", np.nan),
        "icc": avg.get("icc", np.nan),
    }


def build_compiled_metrics(
    runs: list[RankRun],
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build compiled_metrics DataFrame (long-form, one row per rank x label).

    Args:
        runs: List of RankRun objects sorted by rank.
        n_boot: Number of bootstrap resamples for delta CIs.
        seed: Random seed.

    Returns:
        DataFrame with schema per spec §6.
    """
    rng = np.random.default_rng(seed)
    ref_baseline = runs[0].baseline_dice.set_index("scan_id")

    all_rows: list[dict] = []

    for label in LABELS:
        all_rows.append(_extract_baseline_row(runs[0], label))

    baseline_label_rows = all_rows[:3]
    all_rows.append(_extract_mean_row(baseline_label_rows, rank=0))

    for run in runs:
        ens_indexed = run.ensemble_dice.set_index("scan_id")
        common_ids = ref_baseline.index.intersection(ens_indexed.index)

        label_rows = []
        for label in LABELS:
            col = DICE_COLS[label]
            bas_col = ref_baseline.loc[common_ids, col]
            row = _extract_row(run, label, bas_col, n_boot, rng)
            label_rows.append(row)
            all_rows.append(row)

        all_rows.append(_extract_mean_row(label_rows, rank=run.rank))

    # Holm-Bonferroni correction across all (rank, label) cells (excluding baseline)
    non_baseline = [r for r in all_rows if r["rank"] > 0 and r["label"] != "mean"]
    raw_ps = [r["p_wilcoxon_raw"] for r in non_baseline]
    if raw_ps:
        adjusted = _holm_bonferroni(raw_ps)
        for r, p_adj in zip(non_baseline, adjusted):
            r["p_wilcoxon_holm"] = p_adj

    df = pd.DataFrame(all_rows)
    logger.info(
        "Compiled metrics: %d rows (%d ranks incl. baseline, %d labels + mean)",
        len(df),
        df["rank"].nunique(),
        len(LABELS),
    )
    return df


def validate_compiled_metrics(df: pd.DataFrame, n_ranks: int) -> None:
    """Validate compiled_metrics DataFrame.

    Args:
        df: The compiled metrics DataFrame.
        n_ranks: Number of non-baseline ranks.

    Raises:
        ValueError: If validation fails.
    """
    expected_rows = (n_ranks + 1) * (len(LABELS) + 1)
    if len(df) != expected_rows:
        msg = f"Expected {expected_rows} rows, got {len(df)}"
        raise ValueError(msg)

    core_cols = ["dice_mean", "ece", "brier", "pct_bias_dominated"]
    non_baseline = df[df["rank"] > 0]
    per_label = non_baseline[non_baseline["label"] != "mean"]

    for col in core_cols:
        if col in per_label.columns and per_label[col].isna().any():
            n_nan = per_label[col].isna().sum()
            msg = f"Found {n_nan} NaN(s) in {col} for non-baseline per-label rows"
            raise ValueError(msg)

    logger.info("Compiled metrics validation passed")
