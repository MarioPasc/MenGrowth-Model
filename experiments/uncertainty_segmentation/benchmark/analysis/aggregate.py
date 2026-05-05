"""Pairwise statistics + best/median/worst case selection.

Reads ``cache/per_case_metrics.parquet`` and emits:
    - ``cache/pairwise_<metric>_<label>.parquet``  (one per metric × label)
    - ``cache/best_median_worst.json``            (TC ranking per metric)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .io import DEFAULT_ANALYSIS_ROOT
from .metrics import HIGHER_IS_BETTER, LABELS, METRICS

logger = logging.getLogger(__name__)


def _wide_matrix(
    df: pd.DataFrame,
    metric: str,
    label: str,
    model_order: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return an (n_cases, n_models) array; per-pair NaNs handled downstream."""
    sub = df[df["label"] == label][["model", "case_id", metric]]
    pivot = sub.pivot_table(index="case_id", columns="model", values=metric, aggfunc="first")
    pivot = pivot.reindex(columns=model_order)
    pivot = pivot.dropna(how="any")
    return pivot.to_numpy(dtype=float), pivot.index.tolist()


def pairwise_stats(values: np.ndarray, model_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise paired Wilcoxon (two-sided) and Cohen's d on differences.

    Args:
        values: (n_cases, n_models) array, no NaNs.
        model_order: column labels.

    Returns:
        (p_matrix, d_matrix), both (n_models, n_models). ``d_matrix`` is
        anti-symmetric so it can encode the sign of the comparison
        (i.e. ``d_matrix[i, j] > 0`` means model ``i`` beats model ``j``).
    """
    n = len(model_order)
    p = np.ones((n, n), dtype=float)
    d = np.zeros((n, n), dtype=float)
    if values.size == 0 or n < 2:
        return p, d
    for i in range(n):
        for j in range(i + 1, n):
            x = values[:, i]
            y = values[:, j]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue
            xv, yv = x[mask], y[mask]
            try:
                _, pv = stats.wilcoxon(xv, yv, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                pv = 1.0
            p[i, j] = pv
            p[j, i] = pv
            diff = xv - yv
            sd = float(diff.std(ddof=1))
            d_val = float(diff.mean() / sd) if sd > 1e-15 else 0.0
            d[i, j] = d_val
            d[j, i] = -d_val
    return p, d


def aggregate_pairwise(
    df: pd.DataFrame,
    model_order: list[str],
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
) -> dict[str, dict[str, dict]]:
    """Compute and persist pairwise stats for every (metric, label).

    Returns a nested dict ``{metric: {label: {p: ..., d: ..., models: [...], n_cases}}}``.
    """
    cache = analysis_root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, dict]] = {}
    for metric in METRICS:
        out[metric] = {}
        for label in LABELS:
            values, cases = _wide_matrix(df, metric=metric, label=label, model_order=model_order)
            p, d = pairwise_stats(values, model_order)
            df_p = pd.DataFrame(p, index=model_order, columns=model_order)
            df_d = pd.DataFrame(d, index=model_order, columns=model_order)
            df_p.to_csv(cache / f"pairwise_p_{metric}_{label}.csv")
            df_d.to_csv(cache / f"pairwise_d_{metric}_{label}.csv")
            out[metric][label] = {
                "p": df_p,
                "d": df_d,
                "models": list(model_order),
                "n_cases": len(cases),
                "cases": cases,
            }
    logger.info(
        "aggregate: wrote pairwise tables for %d metrics × %d labels under %s",
        len(METRICS),
        len(LABELS),
        cache,
    )
    return out


def select_best_median_worst(
    df: pd.DataFrame,
    model_order: list[str],
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    rank_label: str = "TC",
) -> dict[str, dict[str, str]]:
    """For each metric, pick best/median/worst case by mean across models on ``rank_label``.

    Stored as JSON at ``cache/best_median_worst.json``::

        {metric: {"best": case_id, "median": case_id, "worst": case_id, "score": ...}}

    HD95 inverts the ordering (lower is better).
    """
    cache = analysis_root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, str]] = {}
    sub = df[df["label"] == rank_label]
    for metric in METRICS:
        pivot = sub.pivot_table(index="case_id", columns="model", values=metric, aggfunc="first")
        pivot = pivot.reindex(columns=model_order).dropna(how="any")
        if pivot.empty:
            logger.warning("best/median/worst: no usable cases for metric=%s label=%s", metric, rank_label)
            out[metric] = {}
            continue
        scores = pivot.mean(axis=1, skipna=True).dropna()
        higher = HIGHER_IS_BETTER[metric]
        sorted_scores = scores.sort_values(ascending=not higher)
        best_case = str(sorted_scores.index[0])
        worst_case = str(sorted_scores.index[-1])
        median_case = str(sorted_scores.index[len(sorted_scores) // 2])
        out[metric] = {
            "best": best_case,
            "median": median_case,
            "worst": worst_case,
            "best_score": float(sorted_scores.iloc[0]),
            "median_score": float(sorted_scores.iloc[len(sorted_scores) // 2]),
            "worst_score": float(sorted_scores.iloc[-1]),
            "rank_label": rank_label,
            "higher_is_better": higher,
        }
    (cache / "best_median_worst.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    logger.info("aggregate: best/median/worst on label=%s saved to cache/", rank_label)
    return out


def model_order_from_df(df: pd.DataFrame) -> list[str]:
    """Return canonical model order: external models alphabetical, ``Ours`` last."""
    seen = sorted(df["model"].unique())
    others = [m for m in seen if m != "Ours"]
    if "Ours" in seen:
        return others + ["Ours"]
    return others
