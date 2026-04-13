"""Epistemic uncertainty taxonomy diagnostics.

Following the framework of Jiménez, Jürgens & Waegeman (2026,
arXiv:2505.23506v3), a LoRA/Deep Ensemble captures only the *procedural*
component of epistemic uncertainty (variance across random seeds) and
misses data uncertainty, estimation bias, and distributional shift. This
module computes the empirical diagnostics recommended by that paper —
bias vs procedural-std scatter data and calibration-coverage tables —
and caches them as CSVs/JSON inside each run's ``evaluation/`` directory
so the downstream plotting layer can re-read cached results instead of
recomputing.

Typical usage::

    from experiments.uncertainty_segmentation.plotting.epistemic_metrics import (
        run_for_rank, run_cross_rank,
    )
    bias_df, calib_df, k_star_df, taxonomy = run_for_rank(Path("results/r8_M20_s42"))
    cross_df = run_cross_rank(Path("results/r8_M20_s42"))

Outputs (per-rank, inside ``{run_dir}/evaluation/``):

* ``bias_diagnostics.csv``            — per-scan bias, std, ratios.
* ``calibration_coverage.csv``        — nominal vs empirical coverage table.
* ``bias_dominance_threshold.csv``    — per-scan k* = ceil((sigma/|bias|)^2).
* ``epistemic_taxonomy.json``         — five-source taxonomy summary.

Outputs (cross-rank, inside ``{run_dir.parent}/epistemic_summary/``):

* ``cross_rank_epistemic_summary.csv`` — one row per rank.
* ``cross_rank_taxonomy.json``         — taxonomy per rank, bundled.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Module-level constants (no magic numbers inside functions).
EPS: float = 1e-8
DEFAULT_NOMINAL_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
RUN_DIR_RE = re.compile(r"^r(?P<rank>\d+)_M(?P<members>\d+)_s(?P<seed>\d+)$")

# Bias-dominance threshold k* is unbounded when |bias| → 0. Cap at this
# sentinel to keep CSVs finite and aggregates well-defined; a separate
# boolean flag (``k_star_saturated``) marks the affected rows so downstream
# consumers can filter them if needed. Median aggregates are robust to
# saturation; mean aggregates are not — prefer median.
K_STAR_MAX: int = 10_000

# Column names referenced across the module.
COL_SCAN_ID = "scan_id"
COL_MEMBER_ID = "member_id"
COL_VOLUME_PRED = "volume_pred"
COL_VOLUME_GT = "volume_gt"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_is_valid(cache_path: Path, *input_paths: Path) -> bool:
    """Return True if ``cache_path`` exists and is newer than every input.

    Args:
        cache_path: Path to the cached output file.
        *input_paths: One or more raw-input paths.

    Returns:
        True if the cache can safely be reused.
    """
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for inp in input_paths:
        if not inp.exists():
            return False
        if inp.stat().st_mtime > cache_mtime:
            return False
    return True


# ---------------------------------------------------------------------------
# Per-scan diagnostics
# ---------------------------------------------------------------------------


def compute_bias_diagnostics(
    per_member_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-scan bias, procedural std, and bias-to-std ratio.

    The procedural-std estimate is the inter-member standard deviation of
    predicted volumes. Bias is the signed difference between the ensemble
    mean volume and the ground-truth volume. Metrics are reported on both
    the raw-volume scale and the ``log(V + 1)`` scale used by the
    downstream LME.

    Args:
        per_member_df: Output of ``per_member_test_dice.csv`` — must
            contain ``scan_id``, ``member_id``, ``volume_pred``.
        ensemble_df: Output of ``ensemble_test_dice.csv`` — must contain
            ``scan_id`` and ``volume_gt``.

    Returns:
        DataFrame with one row per scan, columns ``scan_id``,
        ``n_members``, ``volume_gt``, ``volume_ensemble_mean``,
        ``volume_ensemble_std``, ``bias``, ``abs_bias``,
        ``bias_to_std_ratio``, plus their ``logvol_*`` counterparts.

    Raises:
        KeyError: If required columns are missing from the inputs.
    """
    required_member_cols = {COL_SCAN_ID, COL_MEMBER_ID, COL_VOLUME_PRED}
    missing = required_member_cols - set(per_member_df.columns)
    if missing:
        raise KeyError(f"per_member_df missing columns: {missing}")
    required_ensemble_cols = {COL_SCAN_ID, COL_VOLUME_GT}
    missing = required_ensemble_cols - set(ensemble_df.columns)
    if missing:
        raise KeyError(f"ensemble_df missing columns: {missing}")

    grouped = per_member_df.groupby(COL_SCAN_ID)[COL_VOLUME_PRED]
    agg = grouped.agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(
        columns={
            "mean": "volume_ensemble_mean",
            "std": "volume_ensemble_std",
            "count": "n_members",
        }
    )

    gt = ensemble_df[[COL_SCAN_ID, COL_VOLUME_GT]].drop_duplicates(COL_SCAN_ID)
    merged = agg.merge(gt, on=COL_SCAN_ID, how="inner")

    # Raw-volume metrics
    merged["bias"] = merged["volume_ensemble_mean"] - merged[COL_VOLUME_GT]
    merged["abs_bias"] = merged["bias"].abs()
    merged["bias_to_std_ratio"] = merged["abs_bias"] / (
        merged["volume_ensemble_std"] + EPS
    )

    # Log-volume metrics (log(V + 1) per spec; downstream LME works in log
    # space). We project each member's raw volume through log(V+1) then
    # take mean/std — this is different from log(ensemble_mean), and it is
    # the version the spec actually describes.
    per_member_log = per_member_df.copy()
    per_member_log["logvol_pred"] = np.log1p(per_member_log[COL_VOLUME_PRED])
    log_agg = per_member_log.groupby(COL_SCAN_ID)["logvol_pred"].agg(
        logvol_ensemble_mean="mean", logvol_ensemble_std="std"
    ).reset_index()

    merged = merged.merge(log_agg, on=COL_SCAN_ID, how="left")
    merged["logvol_gt"] = np.log1p(merged[COL_VOLUME_GT])
    merged["logvol_bias"] = merged["logvol_ensemble_mean"] - merged["logvol_gt"]
    merged["logvol_abs_bias"] = merged["logvol_bias"].abs()
    merged["logvol_bias_to_std_ratio"] = merged["logvol_abs_bias"] / (
        merged["logvol_ensemble_std"] + EPS
    )

    cols = [
        COL_SCAN_ID, "n_members",
        COL_VOLUME_GT, "volume_ensemble_mean", "volume_ensemble_std",
        "bias", "abs_bias", "bias_to_std_ratio",
        "logvol_gt", "logvol_ensemble_mean", "logvol_ensemble_std",
        "logvol_bias", "logvol_abs_bias", "logvol_bias_to_std_ratio",
    ]
    return merged[cols].sort_values(COL_SCAN_ID).reset_index(drop=True)


def compute_bias_dominance_threshold(
    bias_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the per-scan bias-dominance threshold ``k*``.

    For an i.i.d. ensemble, the standard error of the mean scales as
    ``sigma / sqrt(k)`` while bias is independent of ``k``. The smallest
    ``k`` at which SE drops below ``|bias|`` is::

        k* = ceil((sigma / |bias|) ** 2)

    Scans with ``k* <= M`` are already bias-dominated at the sampled
    ensemble size: the calibration failure is structural, not a
    consequence of under-sampling the seed distribution.

    Edge cases:

    * ``|bias| == 0`` (lucky scan) → ``k*`` is undefined; we cap it at
      :data:`K_STAR_MAX` and set ``k_star_saturated = True``. The median
      aggregate ignores this by construction; the mean should not be used.
    * ``sigma == 0`` (collapsed ensemble — all members predict the same
      volume, e.g., a rank so small the adapters are nearly identical) →
      ``k*`` is 0 but uninformative. We flag ``degenerate_ensemble = True``
      so downstream summary aggregators can exclude the row.

    Args:
        bias_df: Output of :func:`compute_bias_diagnostics` (must contain
            ``scan_id``, ``n_members``, ``volume_ensemble_std``,
            ``abs_bias``, ``logvol_ensemble_std``, ``logvol_abs_bias``).

    Returns:
        One row per scan, columns ``scan_id``, ``n_members_actual``,
        ``k_star_raw``, ``k_star_logvol``, ``k_star_saturated``,
        ``degenerate_ensemble``, ``k_star_exceeds_M``.
    """
    required = {
        COL_SCAN_ID, "n_members",
        "volume_ensemble_std", "abs_bias",
        "logvol_ensemble_std", "logvol_abs_bias",
    }
    missing = required - set(bias_df.columns)
    if missing:
        raise KeyError(f"bias_df missing columns: {missing}")

    def _k_star(sigma: pd.Series, abs_bias: pd.Series) -> pd.Series:
        """Compute k* = ceil((sigma / |bias|)^2), capping the |bias|==0 case."""
        sigma = sigma.astype(float)
        abs_bias = abs_bias.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(abs_bias > 0.0, sigma / abs_bias, np.inf)
            k_inf = np.ceil(np.square(ratio))
        # Finite cap. np.inf stays inf through np.ceil(np.square(inf)).
        k_capped = np.minimum(k_inf, float(K_STAR_MAX))
        return pd.Series(k_capped.astype(int), index=sigma.index)

    k_star_raw = _k_star(bias_df["volume_ensemble_std"], bias_df["abs_bias"])
    k_star_log = _k_star(bias_df["logvol_ensemble_std"], bias_df["logvol_abs_bias"])

    # |bias|==0 triggers saturation on either scale — the primary narrative
    # uses the log-volume bias, so anchor the saturation flag there.
    k_star_saturated = bias_df["logvol_abs_bias"].astype(float) == 0.0

    # Degenerate if EITHER scale collapsed. In practice they co-vary, but
    # being explicit avoids surprising exclusions.
    degenerate = (
        (bias_df["volume_ensemble_std"].astype(float) == 0.0)
        | (bias_df["logvol_ensemble_std"].astype(float) == 0.0)
    )

    n_members_actual = bias_df["n_members"].astype(int)
    exceeds = k_star_log > n_members_actual

    out = pd.DataFrame({
        COL_SCAN_ID: bias_df[COL_SCAN_ID].values,
        "n_members_actual": n_members_actual.values,
        "k_star_raw": k_star_raw.values,
        "k_star_logvol": k_star_log.values,
        "k_star_saturated": k_star_saturated.values,
        "degenerate_ensemble": degenerate.values,
        "k_star_exceeds_M": exceeds.values,
    })
    return out.sort_values(COL_SCAN_ID).reset_index(drop=True)


def compute_calibration_coverage(
    per_member_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    nominal_levels: Iterable[float] = DEFAULT_NOMINAL_LEVELS,
) -> pd.DataFrame:
    """Empirical coverage of ensemble-derived CIs at several nominal levels.

    The CI uses the Student-t critical value with ``df = M − 1`` rather
    than the Gaussian, which matters for the M=20 ensembles (spec
    §2.4.2). Coverage is measured against ``volume_gt`` on the log-volume
    scale.

    Args:
        per_member_df: ``per_member_test_dice.csv`` dataframe.
        ensemble_df: ``ensemble_test_dice.csv`` dataframe.
        nominal_levels: Nominal coverage levels to evaluate.

    Returns:
        DataFrame with columns ``nominal_level``, ``t_multiplier``,
        ``n_scans``, ``n_covered``, ``empirical_coverage``,
        ``coverage_deficit``.
    """
    bias_df = compute_bias_diagnostics(per_member_df, ensemble_df)

    # df for t-distribution. Use min member-count across scans minus 1 —
    # in practice this equals M−1 when every scan has all members.
    m_min = int(bias_df["n_members"].min())
    df_t = max(m_min - 1, 1)

    rows: list[dict[str, float | int]] = []
    for level in nominal_levels:
        alpha = 1.0 - float(level)
        t_mult = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_t))
        half_width = t_mult * bias_df["logvol_ensemble_std"]
        lo = bias_df["logvol_ensemble_mean"] - half_width
        hi = bias_df["logvol_ensemble_mean"] + half_width
        covered = (bias_df["logvol_gt"] >= lo) & (bias_df["logvol_gt"] <= hi)
        n_scans = int(len(bias_df))
        n_covered = int(covered.sum())
        empirical = n_covered / n_scans if n_scans > 0 else float("nan")
        rows.append({
            "nominal_level": float(level),
            "t_multiplier": t_mult,
            "n_scans": n_scans,
            "n_covered": n_covered,
            "empirical_coverage": empirical,
            "coverage_deficit": float(level) - empirical,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rank-level summary + taxonomy dict
# ---------------------------------------------------------------------------


def compute_rank_summary(
    bias_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    rank: int,
    k_star_df: pd.DataFrame | None = None,
) -> dict[str, float | int]:
    """Condense per-scan diagnostics into a single summary row.

    Args:
        bias_df: Output of :func:`compute_bias_diagnostics`.
        calibration_df: Output of :func:`compute_calibration_coverage`.
        rank: LoRA rank for this configuration.
        k_star_df: Output of :func:`compute_bias_dominance_threshold`.
            If provided, the summary includes ``median_k_star_logvol`` —
            the Proposal-1 narrative anchor — plus ``pct_scans_k_star_eq_1``
            and ``pct_scans_k_star_exceeds_M``. Aggregates exclude rows
            flagged as ``degenerate_ensemble`` and use the median (robust
            to saturation).

    Returns:
        Dict with one float/int per summary statistic — suitable for a
        single row of a cross-rank DataFrame.
    """
    pct_bias_gt_std = float(
        (bias_df["logvol_abs_bias"] > bias_df["logvol_ensemble_std"]).mean()
    )

    def _coverage(level: float) -> float:
        match = calibration_df[np.isclose(calibration_df["nominal_level"], level)]
        if len(match) == 0:
            return float("nan")
        return float(match["empirical_coverage"].iloc[0])

    q25, q75 = bias_df["logvol_ensemble_std"].quantile([0.25, 0.75])
    summary: dict[str, float | int] = {
        "rank": int(rank),
        "n_scans": int(len(bias_df)),
        "median_logvol_std": float(bias_df["logvol_ensemble_std"].median()),
        "mean_logvol_std": float(bias_df["logvol_ensemble_std"].mean()),
        "iqr_logvol_std": float(q75 - q25),
        "median_logvol_std_q25": float(q25),
        "median_logvol_std_q75": float(q75),
        "median_abs_bias_logvol": float(bias_df["logvol_abs_bias"].median()),
        "mean_abs_bias_logvol": float(bias_df["logvol_abs_bias"].mean()),
        "median_abs_bias_q25": float(bias_df["logvol_abs_bias"].quantile(0.25)),
        "median_abs_bias_q75": float(bias_df["logvol_abs_bias"].quantile(0.75)),
        "pct_scans_bias_gt_std": pct_bias_gt_std,
        "coverage_50": _coverage(0.50),
        "coverage_80": _coverage(0.80),
        "coverage_90": _coverage(0.90),
        "coverage_95": _coverage(0.95),
        "coverage_deficit_95": 0.95 - _coverage(0.95),
    }

    if k_star_df is not None and len(k_star_df) > 0:
        # Exclude degenerate rows (sigma == 0) from k* aggregates — the
        # value is 0 but not meaningful. Median is robust to the
        # saturation cap (|bias|==0 rows).
        usable = k_star_df.loc[~k_star_df["degenerate_ensemble"].astype(bool)]
        n_k = int(len(usable))
        if n_k > 0:
            summary["n_scans_k_star"] = n_k
            summary["median_k_star_logvol"] = float(usable["k_star_logvol"].median())
            summary["median_k_star_raw"] = float(usable["k_star_raw"].median())
            summary["pct_scans_k_star_eq_1"] = float(
                (usable["k_star_logvol"] <= 1).mean()
            )
            summary["pct_scans_k_star_exceeds_M"] = float(
                usable["k_star_exceeds_M"].astype(bool).mean()
            )
            summary["pct_scans_k_star_saturated"] = float(
                usable["k_star_saturated"].astype(bool).mean()
            )
            summary["pct_scans_degenerate_ensemble"] = float(
                k_star_df["degenerate_ensemble"].astype(bool).mean()
            )
        else:
            # Every scan was degenerate — emit NaNs so the column exists.
            for key in (
                "median_k_star_logvol", "median_k_star_raw",
                "pct_scans_k_star_eq_1", "pct_scans_k_star_exceeds_M",
                "pct_scans_k_star_saturated",
            ):
                summary[key] = float("nan")
            summary["n_scans_k_star"] = 0
            summary["pct_scans_degenerate_ensemble"] = 1.0

    return summary


def build_taxonomy_dict(
    bias_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    rank: int,
    n_members: int,
    seed: int,
    k_star_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Assemble the five-source taxonomy dict (spec §2.4.4).

    Returns a JSON-serializable dict documenting which epistemic
    components the ensemble measures, diagnoses, or ignores. If
    ``k_star_df`` is provided, the estimation-bias entry gains the
    Proposal-1 bias-dominance metrics.
    """
    summary = compute_rank_summary(bias_df, calibration_df, rank, k_star_df=k_star_df)

    estimation_bias: dict[str, Any] = {
        "status": "diagnosed",
        "median_abs_bias_logvol": summary["median_abs_bias_logvol"],
        "mean_abs_bias_logvol": summary["mean_abs_bias_logvol"],
        "pct_scans_bias_gt_std": summary["pct_scans_bias_gt_std"],
        "note": (
            "Sources: LoRA low-rank constraint (rank="
            f"{int(rank)}), frozen encoder bias, finite BraTS-MEN"
            " training set."
        ),
    }
    if k_star_df is not None and "median_k_star_logvol" in summary:
        estimation_bias["bias_dominance"] = {
            "median_k_star_logvol": summary["median_k_star_logvol"],
            "pct_scans_k_star_eq_1": summary["pct_scans_k_star_eq_1"],
            "pct_scans_k_star_exceeds_M": summary["pct_scans_k_star_exceeds_M"],
            "pct_scans_degenerate_ensemble": summary.get(
                "pct_scans_degenerate_ensemble", 0.0
            ),
            "n_members_sampled": int(n_members),
            "note": (
                "k* = ceil((sigma / |bias|)^2) on the log-volume scale."
                " Scans with k* <= M are already bias-dominated at the"
                " sampled ensemble size, so additional members cannot"
                " improve calibration."
            ),
        }
    return {
        "config": {"rank": int(rank), "n_members": int(n_members), "seed": int(seed)},
        "taxonomy": {
            "approximation_error": {
                "status": "not_quantifiable",
                "note": (
                    "Requires oracle access to true p(y|x); universal"
                    " approximation implies this term vanishes in theory"
                    " (Hornik et al., 1989) but not in practice with finite"
                    " capacity."
                ),
            },
            "estimation_bias": estimation_bias,
            "procedural_uncertainty": {
                "status": "measured",
                "median_logvol_std": summary["median_logvol_std"],
                "mean_logvol_std": summary["mean_logvol_std"],
                "iqr_logvol_std": summary["iqr_logvol_std"],
                "note": (
                    "Ensemble volume std across M members with different"
                    " random seeds on the same dataset."
                ),
            },
            "data_uncertainty": {
                "status": "partially_captured",
                "note": (
                    "LOPO-CV varies the MenGrowth training set across"
                    " folds, capturing some data uncertainty at the growth"
                    " model level. Not propagated per-fold as"
                    " observation-level sigma_{v,k}. BraTS-MEN training set"
                    " is fixed across all ensemble members."
                ),
            },
            "distributional_uncertainty": {
                "status": "mitigated",
                "note": (
                    "BraTS-MEN (multi-site, standardized) -> Andalusian"
                    " cohort. ComBat harmonization applied; residual shift"
                    " not quantified at the segmentation level."
                ),
            },
        },
        "calibration": {
            "coverage_50": summary["coverage_50"],
            "coverage_80": summary["coverage_80"],
            "coverage_90": summary["coverage_90"],
            "coverage_95": summary["coverage_95"],
            "coverage_deficit_95": summary["coverage_deficit_95"],
        },
        "recommendation": (
            "Interpret sigma_{v,k} as a lower bound on total"
            " segmentation-derived epistemic uncertainty."
        ),
    }


# ---------------------------------------------------------------------------
# Run-dir inspection helpers
# ---------------------------------------------------------------------------


def _parse_run_dir(run_dir: Path) -> dict[str, int] | None:
    """Parse ``r{R}_M{M}_s{S}`` into integer fields; None if no match."""
    m = RUN_DIR_RE.match(run_dir.name)
    if not m:
        return None
    return {k: int(v) for k, v in m.groupdict().items()}


def _n_members_from_df(per_member_df: pd.DataFrame) -> int:
    """Infer the ensemble size from the per-member dice table."""
    return int(per_member_df[COL_MEMBER_ID].nunique())


# ---------------------------------------------------------------------------
# Public orchestration API
# ---------------------------------------------------------------------------


def run_for_rank(
    run_dir: Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute per-rank epistemic diagnostics and cache them to disk.

    Args:
        run_dir: Path to a single-rank run directory (e.g., ``r8_M20_s42``).
        force: If True, ignore cache and recompute everything.

    Returns:
        ``(bias_diagnostics_df, calibration_coverage_df,
        bias_dominance_threshold_df, taxonomy_dict)``.

    Raises:
        FileNotFoundError: If required raw CSVs are missing.
    """
    eval_dir = run_dir / "evaluation"
    per_member_path = eval_dir / "per_member_test_dice.csv"
    ensemble_path = eval_dir / "ensemble_test_dice.csv"
    if not per_member_path.exists() or not ensemble_path.exists():
        raise FileNotFoundError(
            f"Missing raw CSVs under {eval_dir}: need"
            " per_member_test_dice.csv + ensemble_test_dice.csv."
        )

    bias_cache = eval_dir / "bias_diagnostics.csv"
    calib_cache = eval_dir / "calibration_coverage.csv"
    k_star_cache = eval_dir / "bias_dominance_threshold.csv"
    taxonomy_cache = eval_dir / "epistemic_taxonomy.json"

    caches_valid = (
        not force
        and _cache_is_valid(bias_cache, per_member_path, ensemble_path)
        and _cache_is_valid(calib_cache, per_member_path, ensemble_path)
        and _cache_is_valid(k_star_cache, per_member_path, ensemble_path)
        and _cache_is_valid(taxonomy_cache, per_member_path, ensemble_path)
    )
    if caches_valid:
        logger.info("[epistemic] %s: using cached outputs", run_dir.name)
        bias_df = pd.read_csv(bias_cache)
        calib_df = pd.read_csv(calib_cache)
        k_star_df = pd.read_csv(k_star_cache)
        with open(taxonomy_cache) as f:
            taxonomy = json.load(f)
        return bias_df, calib_df, k_star_df, taxonomy

    logger.info("[epistemic] %s: computing from raw CSVs", run_dir.name)
    per_member_df = pd.read_csv(per_member_path)
    ensemble_df = pd.read_csv(ensemble_path)

    bias_df = compute_bias_diagnostics(per_member_df, ensemble_df)
    k_star_df = compute_bias_dominance_threshold(bias_df)
    calib_df = compute_calibration_coverage(per_member_df, ensemble_df)

    parsed = _parse_run_dir(run_dir) or {}
    rank = parsed.get("rank", -1)
    seed = parsed.get("seed", -1)
    n_members = _n_members_from_df(per_member_df)

    taxonomy = build_taxonomy_dict(
        bias_df, calib_df,
        rank=rank, n_members=n_members, seed=seed,
        k_star_df=k_star_df,
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    bias_df.to_csv(bias_cache, index=False)
    calib_df.to_csv(calib_cache, index=False)
    k_star_df.to_csv(k_star_cache, index=False)
    with open(taxonomy_cache, "w") as f:
        json.dump(taxonomy, f, indent=2)

    logger.info(
        "[epistemic] %s: saved bias_diagnostics(%d rows),"
        " calibration_coverage(%d rows), bias_dominance_threshold(%d rows),"
        " epistemic_taxonomy.json",
        run_dir.name, len(bias_df), len(calib_df), len(k_star_df),
    )
    return bias_df, calib_df, k_star_df, taxonomy


def _find_sibling_ranks(run_dir: Path) -> list[Path]:
    """Return sibling run dirs sharing parent and matching the naming scheme."""
    parent = run_dir.parent
    if not parent.exists():
        return []
    siblings = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and RUN_DIR_RE.match(d.name)
    )
    return siblings


def run_cross_rank(
    run_dir: Path,
    *,
    force: bool = False,
) -> pd.DataFrame | None:
    """Aggregate per-rank diagnostics across sibling runs in ``run_dir.parent``.

    Writes a cross-rank summary CSV + JSON into
    ``{run_dir.parent}/epistemic_summary/``. If fewer than two sibling
    runs are present, returns ``None`` and does not write anything.

    Args:
        run_dir: Any rank's run directory; siblings are discovered from
            its parent.
        force: If True, force recompute of each per-rank cache.

    Returns:
        The cross-rank summary DataFrame (one row per rank) or ``None``.
    """
    siblings = _find_sibling_ranks(run_dir)
    if len(siblings) < 2:
        logger.warning(
            "[epistemic] cross-rank aggregation skipped: found %d sibling"
            " rank dir(s) under %s (need >=2).",
            len(siblings), run_dir.parent,
        )
        return None

    per_rank_rows: list[dict[str, Any]] = []
    per_rank_taxonomies: dict[str, Any] = {}
    for sib in siblings:
        parsed = _parse_run_dir(sib)
        if parsed is None:
            continue
        try:
            bias_df, calib_df, k_star_df, taxonomy = run_for_rank(sib, force=force)
        except FileNotFoundError as exc:
            logger.warning("[epistemic] skipping %s: %s", sib.name, exc)
            continue
        summary = compute_rank_summary(
            bias_df, calib_df, rank=parsed["rank"], k_star_df=k_star_df,
        )
        per_rank_rows.append(summary)
        per_rank_taxonomies[f"r{parsed['rank']}"] = taxonomy

    if len(per_rank_rows) < 2:
        logger.warning(
            "[epistemic] cross-rank aggregation skipped: only %d usable"
            " ranks found", len(per_rank_rows),
        )
        return None

    cross_df = pd.DataFrame(per_rank_rows).sort_values("rank").reset_index(drop=True)

    out_dir = run_dir.parent / "epistemic_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    cross_df.to_csv(out_dir / "cross_rank_epistemic_summary.csv", index=False)
    with open(out_dir / "cross_rank_taxonomy.json", "w") as f:
        json.dump(per_rank_taxonomies, f, indent=2)

    logger.info(
        "[epistemic] cross-rank summary written to %s (%d ranks)",
        out_dir, len(cross_df),
    )
    return cross_df
