# experiments/stage1_volumetric/stats/conditional_calibration.py
"""Calibration metrics stratified by per-target segmentation variance.

Marginal coverage averaged across the cohort hides the fact that the
LoRA-ensemble measurement variance ``sigma_v^2`` is strongly bimodal.
For most scans ``sigma_v^2`` is essentially zero so propagation has
nothing to do, while for a small subset it is large and is exactly
where propagation should help. This module groups predictions into
``sigma_v^2`` tertiles (per protocol) and reports coverage / CRPS /
interval score / sharpness within each, so the manuscript can support
a *conditional* calibration claim instead of an averaged one.

See ``docs/UQ_CALIBRATION_STORY.md`` §6.1 for the rationale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from growth.shared.lopo import LOPOResults
from growth.shared.metrics import (
    compute_coverage_at_levels,
    compute_crps_gaussian,
    compute_interval_score,
)

logger = logging.getLogger(__name__)

_TERTILE_LABELS = ("low", "mid", "high")


def _extract_arrays(
    results: LOPOResults,
    protocol: str,
) -> dict[str, np.ndarray] | None:
    """Pull aligned (pid, y_true, y_pred, pred_var, sigma_v_sq) arrays.

    Returns ``None`` if no usable predictions for this protocol or if
    ``sigma_v_sq_target`` is missing.
    """
    pids: list[str] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    pred_var: list[float] = []
    sigma_v_sq: list[float] = []

    for fr in results.fold_results:
        if protocol not in fr.predictions:
            continue
        for pd in fr.predictions[protocol]:
            if "sigma_v_sq_target" not in pd:
                return None
            pids.append(fr.patient_id)
            y_true.append(pd["actual"])
            y_pred.append(pd["pred_mean"])
            pred_var.append(pd["pred_var"])
            sigma_v_sq.append(pd["sigma_v_sq_target"])

    if not pids:
        return None

    return {
        "pids": np.asarray(pids),
        "y_true": np.asarray(y_true, dtype=np.float64),
        "y_pred": np.asarray(y_pred, dtype=np.float64),
        "pred_var": np.asarray(pred_var, dtype=np.float64),
        "sigma_v_sq": np.asarray(sigma_v_sq, dtype=np.float64),
    }


def _tertile_assignments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each entry to a tertile based on its rank.

    Uses ``np.quantile`` with [1/3, 2/3] cut points. Returns
    ``(labels, edges)`` where ``labels`` is in {0, 1, 2} and ``edges``
    is the (q33, q66) tuple. Ties at the boundary go to the lower
    tertile to keep the partition stable.
    """
    finite = np.isfinite(values)
    if finite.sum() < 3:
        # Degenerate: not enough points for tertiles
        labels = np.full(len(values), -1, dtype=np.int8)
        return labels, np.array([np.nan, np.nan])

    q33, q66 = np.quantile(values[finite], [1 / 3.0, 2 / 3.0])
    labels = np.full(len(values), -1, dtype=np.int8)
    labels[finite & (values <= q33)] = 0
    labels[finite & (values > q33) & (values <= q66)] = 1
    labels[finite & (values > q66)] = 2
    return labels, np.array([q33, q66])


def _bin_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_var: np.ndarray,
    sigma_v_sq: np.ndarray,
) -> dict[str, float]:
    """Compute the calibration battery for one tertile."""
    if len(y_true) == 0:
        return {"n": 0}

    sigma = np.sqrt(np.maximum(pred_var, 0.0))
    coverage = compute_coverage_at_levels(y_true, y_pred, sigma)
    crps = compute_crps_gaussian(y_true, y_pred, sigma)

    z = float(np.abs(scipy_stats.norm.ppf(0.025)))
    is_lower = y_pred - z * sigma
    is_upper = y_pred + z * sigma
    is95 = compute_interval_score(y_true, is_lower, is_upper, 0.05)

    abs_err = np.abs(y_true - y_pred)

    return {
        "n": int(len(y_true)),
        "sigma_v_sq_mean": float(np.nanmean(sigma_v_sq)),
        "sigma_v_sq_median": float(np.nanmedian(sigma_v_sq)),
        "abs_err_mean": float(np.mean(abs_err)),
        "abs_err_median": float(np.median(abs_err)),
        "ci_width_mean": float(np.mean(2 * z * sigma)),
        "coverage_50": float(coverage.get(0.50, np.nan)),
        "coverage_80": float(coverage.get(0.80, np.nan)),
        "coverage_90": float(coverage.get(0.90, np.nan)),
        "coverage_95": float(coverage.get(0.95, np.nan)),
        "crps": float(crps),
        "interval_score_95": float(is95),
    }


def compute_conditional_calibration(
    results: LOPOResults,
    *,
    protocol: str = "last_from_rest",
    edges: np.ndarray | None = None,
) -> dict | None:
    """Compute per-tertile calibration for one model.

    Args:
        results: LOPO-CV results for one model.
        protocol: Prediction protocol to evaluate.
        edges: Optional (q33, q66) tertile cut points to *reuse* across
            models (so tertile membership is identical across models and
            paired comparisons are meaningful). When ``None`` the cut
            points are computed from this model's own ``sigma_v_sq``
            distribution.

    Returns:
        Dict with the per-tertile metrics, the cut points used, and the
        overall (un-stratified) battery for reference. ``None`` if the
        protocol has no predictions or sigma_v_sq_target is unavailable
        (legacy results predating the loader change).
    """
    arrays = _extract_arrays(results, protocol)
    if arrays is None:
        return None

    sv = arrays["sigma_v_sq"]

    if edges is None:
        labels, used_edges = _tertile_assignments(sv)
    else:
        used_edges = np.asarray(edges, dtype=np.float64)
        labels = np.full(len(sv), -1, dtype=np.int8)
        labels[np.isfinite(sv) & (sv <= used_edges[0])] = 0
        labels[np.isfinite(sv) & (sv > used_edges[0]) & (sv <= used_edges[1])] = 1
        labels[np.isfinite(sv) & (sv > used_edges[1])] = 2

    tertiles: dict[str, dict] = {}
    for k, name in enumerate(_TERTILE_LABELS):
        sel = labels == k
        tertiles[name] = _bin_metrics(
            arrays["y_true"][sel],
            arrays["y_pred"][sel],
            arrays["pred_var"][sel],
            sv[sel],
        )

    overall = _bin_metrics(
        arrays["y_true"],
        arrays["y_pred"],
        arrays["pred_var"],
        sv,
    )

    return {
        "protocol": protocol,
        "edges_sigma_v_sq": [float(used_edges[0]), float(used_edges[1])],
        "n_total": int(len(sv)),
        "tertiles": tertiles,
        "overall": overall,
    }


def compute_shared_edges(
    lopo_results: dict[str, LOPOResults],
    protocol: str = "last_from_rest",
    reference_model: str | None = None,
) -> np.ndarray | None:
    """Return (q33, q66) cut points shared across models.

    Uses ``reference_model`` if given, else the first model whose
    predictions carry ``sigma_v_sq_target``. This guarantees identical
    tertile membership across hetero/homo families since they use the
    same trajectories.
    """
    candidates = [reference_model] if reference_model else list(lopo_results.keys())
    for name in candidates:
        if name is None or name not in lopo_results:
            continue
        arrays = _extract_arrays(lopo_results[name], protocol)
        if arrays is None:
            continue
        _, edges = _tertile_assignments(arrays["sigma_v_sq"])
        if np.all(np.isfinite(edges)):
            logger.info(
                f"Shared sigma_v^2 tertile edges from {name} ({protocol}): "
                f"q33={edges[0]:.4g}, q66={edges[1]:.4g}"
            )
            return edges
    return None


def write_conditional_table(
    per_model: dict[str, dict],
    output_dir: Path,
    *,
    protocol: str = "last_from_rest",
    filename_prefix: str = "conditional_calibration",
) -> None:
    """Write per-tertile calibration as JSON + Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": protocol,
        "models": per_model,
    }
    with open(output_dir / f"{filename_prefix}.json", "w") as f:
        json.dump(payload, f, indent=2)

    lines: list[str] = [
        f"# Conditional Calibration by sigma_v^2 Tertile ({protocol})",
        "",
        "Each row reports the calibration battery for one model on one",
        "tertile of the per-target segmentation variance sigma_v^2.",
        "Tertile membership is shared across models so the comparison is",
        "paired by patient.",
        "",
    ]

    edges_seen: list[float] = []
    for entry in per_model.values():
        if entry and "edges_sigma_v_sq" in entry:
            edges_seen = entry["edges_sigma_v_sq"]
            break
    if edges_seen:
        lines.extend(
            [
                f"- sigma_v^2 cut points: q33 = {edges_seen[0]:.4g}, "
                f"q66 = {edges_seen[1]:.4g}",
                "",
            ]
        )

    header = (
        "| Model | Tertile | n | sv2 mean | CI width | cov@95 | cov@90 | "
        "cov@80 | CRPS | IS@95 |"
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines.append(header)
    lines.append(sep)

    for model_name, entry in per_model.items():
        if entry is None:
            lines.append(
                f"| {model_name} | (no sigma_v_sq_target available) | | | | | | | | |"
            )
            continue
        for tname in _TERTILE_LABELS:
            t = entry["tertiles"][tname]
            if t.get("n", 0) == 0:
                continue
            lines.append(
                f"| {model_name} | {tname} | {t['n']} | "
                f"{t['sigma_v_sq_mean']:.4g} | {t['ci_width_mean']:.3f} | "
                f"{t['coverage_95']:.3f} | {t['coverage_90']:.3f} | "
                f"{t['coverage_80']:.3f} | {t['crps']:.4f} | "
                f"{t['interval_score_95']:.3f} |"
            )
        ov = entry["overall"]
        lines.append(
            f"| {model_name} | OVERALL | {ov['n']} | {ov['sigma_v_sq_mean']:.4g} | "
            f"{ov['ci_width_mean']:.3f} | {ov['coverage_95']:.3f} | "
            f"{ov['coverage_90']:.3f} | {ov['coverage_80']:.3f} | "
            f"{ov['crps']:.4f} | {ov['interval_score_95']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **low**: sigma_v^2 in the bottom third (well-segmented scans).",
            "  Hetero ~ homo expected here; if hetero is *narrower* and",
            "  under-covers, it confirms REML absorbed the mean sigma_v^2",
            "  into a smaller sigma_n^2 (see UQ_CALIBRATION_STORY.md §5).",
            "- **mid**: hetero begins to widen relative to homo.",
            "- **high**: hetero should widen substantially and either match",
            "  or exceed homo coverage; lower interval score is the win.",
            "",
        ]
    )

    with open(output_dir / f"{filename_prefix}.md", "w") as f:
        f.write("\n".join(lines))


def run_conditional_calibration(
    lopo_results: dict[str, LOPOResults],
    output_dir: Path,
    *,
    protocols: tuple[str, ...] = ("last_from_rest", "all_from_first"),
    reference_model: str | None = None,
) -> dict[str, dict[str, dict | None]]:
    """Compute and persist conditional calibration for all models.

    Args:
        lopo_results: All model LOPOResults.
        output_dir: Stage output directory; per-protocol files are
            written under it.
        protocols: Protocols to stratify.
        reference_model: Model to use to compute the shared
            sigma_v^2 tertile edges. Defaults to the first hetero
            model present.

    Returns:
        Dict ``{protocol: {model_name: per-model entry or None}}``.
    """
    if reference_model is None:
        for cand in ("LMEHetero", "HGPHetero", "ScalarGPHetero"):
            if cand in lopo_results:
                reference_model = cand
                break

    out: dict[str, dict[str, dict | None]] = {}
    for proto in protocols:
        edges = compute_shared_edges(lopo_results, protocol=proto, reference_model=reference_model)
        if edges is None:
            logger.warning(
                f"Conditional calibration ({proto}): no model has "
                f"sigma_v_sq_target; skipping."
            )
            continue
        per_model: dict[str, dict | None] = {}
        for name, results in lopo_results.items():
            per_model[name] = compute_conditional_calibration(
                results, protocol=proto, edges=edges
            )
        write_conditional_table(
            per_model,
            output_dir,
            protocol=proto,
            filename_prefix=f"conditional_calibration_{proto}",
        )
        out[proto] = per_model
    return out
