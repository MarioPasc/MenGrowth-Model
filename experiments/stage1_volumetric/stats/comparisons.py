# experiments/stage1_volumetric/stats/comparisons.py
"""Paired comparisons between growth model uncertainty approaches."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from growth.shared.bootstrap import paired_permutation_test
from growth.shared.lopo import LOPOResults
from growth.shared.metrics import (
    compute_coverage_at_levels,
    compute_crps_gaussian,
    compute_r2,
)

logger = logging.getLogger(__name__)


def extract_lopo_predictions(
    results: LOPOResults,
    protocol: str = "last_from_rest",
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned per-patient predictions from LOPO results.

    Args:
        results: LOPO-CV results for one model.
        protocol: Prediction protocol to extract.

    Returns:
        (patient_ids, y_true, y_pred, pred_var) arrays.
    """
    pids, y_true, y_pred, pred_var = [], [], [], []
    for fr in results.fold_results:
        if protocol not in fr.predictions:
            continue
        for pd in fr.predictions[protocol]:
            pids.append(fr.patient_id)
            y_true.append(pd["actual"])
            y_pred.append(pd["pred_mean"])
            pred_var.append(pd["pred_var"])
    return pids, np.array(y_true), np.array(y_pred), np.array(pred_var)


def run_paired_comparisons(
    lopo_results: dict[str, LOPOResults],
    pairs: list[list[str]],
    n_permutations: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """Run paired comparisons between two model families.

    For each pair, aligns predictions by patient ID and computes
    deltas in R², CRPS, and 95% coverage with a permutation test
    on absolute errors.

    Args:
        lopo_results: Dict mapping model name to LOPOResults.
        pairs: List of [model_a, model_b] pairs to compare.
        n_permutations: Number of permutations for significance test.
        seed: Random seed for permutation test.

    Returns:
        List of comparison dicts with deltas and p-values.
    """
    comparisons = []

    for pair in pairs:
        if len(pair) != 2:
            continue
        name_a, name_b = pair[0], pair[1]

        if name_a not in lopo_results or name_b not in lopo_results:
            logger.warning(f"Skipping pair {pair}: missing results")
            continue

        pids_a, yt_a, yp_a, var_a = extract_lopo_predictions(lopo_results[name_a])
        pids_b, yt_b, yp_b, var_b = extract_lopo_predictions(lopo_results[name_b])

        if len(yt_a) == 0 or len(yt_b) == 0:
            continue

        common_pids = sorted(set(pids_a) & set(pids_b))
        if len(common_pids) < 3:
            logger.warning(f"Pair {pair}: only {len(common_pids)} common patients, skipping")
            continue

        idx_a = {pid: i for i, pid in enumerate(pids_a)}
        idx_b = {pid: i for i, pid in enumerate(pids_b)}
        sel_a = np.array([idx_a[p] for p in common_pids])
        sel_b = np.array([idx_b[p] for p in common_pids])

        yt_a, yp_a, v_a = yt_a[sel_a], yp_a[sel_a], var_a[sel_a]
        yt_b, yp_b, v_b = yt_b[sel_b], yp_b[sel_b], var_b[sel_b]

        r2_a = compute_r2(yt_a, yp_a)
        r2_b = compute_r2(yt_b, yp_b)

        sigma_a = np.sqrt(np.maximum(v_a, 0.0))
        sigma_b = np.sqrt(np.maximum(v_b, 0.0))
        crps_a = compute_crps_gaussian(yt_a, yp_a, sigma_a)
        crps_b = compute_crps_gaussian(yt_b, yp_b, sigma_b)

        cov_a = compute_coverage_at_levels(yt_a, yp_a, sigma_a, (0.95,))
        cov_b = compute_coverage_at_levels(yt_b, yp_b, sigma_b, (0.95,))

        errors_a = np.abs(yt_a - yp_a)
        errors_b = np.abs(yt_b - yp_b)
        perm_result = paired_permutation_test(
            errors_a,
            errors_b,
            n_permutations=n_permutations,
            seed=seed,
        )

        comp = {
            "pair": [name_a, name_b],
            "r2_homo": r2_a,
            "r2_hetero": r2_b,
            "delta_r2": r2_b - r2_a,
            "crps_homo": crps_a,
            "crps_hetero": crps_b,
            "delta_crps": crps_b - crps_a,
            "coverage_95_homo": cov_a[0.95],
            "coverage_95_hetero": cov_b[0.95],
            "delta_coverage_95": cov_b[0.95] - cov_a[0.95],
            "p_value_errors": perm_result.p_value,
        }
        comparisons.append(comp)

        logger.info(
            f"Pair {name_a} -> {name_b}: "
            f"dR2={comp['delta_r2']:+.4f}, dCRPS={comp['delta_crps']:+.4f}, "
            f"dCov95={comp['delta_coverage_95']:+.3f}, p={perm_result.p_value:.4f}"
        )

    return comparisons


def write_comparison_table(
    comparisons: list[dict],
    output_dir: Path,
    filename_prefix: str = "comparison_homo_vs_hetero",
    title: str = "Homoscedastic vs Heteroscedastic Comparison",
) -> None:
    """Write comparison table as JSON and Markdown.

    Args:
        comparisons: List of comparison dicts from run_paired_comparisons.
        output_dir: Directory to write files.
        filename_prefix: Base filename (without extension).
        title: Markdown heading.
    """
    with open(output_dir / f"{filename_prefix}.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    lines = [
        f"# {title}",
        "",
        "| Pair | dR2 | dCRPS | dCov_95 | p-value |",
        "|------|-----|-------|---------|---------|",
    ]
    for c in comparisons:
        pair_str = f"{c['pair'][0]} -> {c['pair'][1]}"
        lines.append(
            f"| {pair_str} | {c['delta_r2']:+.4f} | "
            f"{c['delta_crps']:+.4f} | {c['delta_coverage_95']:+.3f} | "
            f"{c['p_value_errors']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- dR2 ~ 0: uncertainty propagation preserves point accuracy",
            "- dCRPS < 0: second model is better calibrated (lower is better)",
            "- dCov_95 > 0: second model intervals achieve better coverage",
            "",
        ]
    )

    with open(output_dir / f"{filename_prefix}.md", "w") as f:
        f.write("\n".join(lines))
