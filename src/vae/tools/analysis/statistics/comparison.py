"""Multi-run comparison logic.

Compares metrics across multiple experiment runs and performs
statistical tests to identify significant differences.
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from ..schemas import (
    AnalysisSummary,
    ComparisonSummary,
    StatisticalTestResults,
)
from .summary import generate_summary
from .statistical_tests import (
    mann_whitney_u_test,
    bootstrap_confidence_interval,
    levene_variance_test,
)

logger = logging.getLogger(__name__)


def compare_runs(
    run_data: Dict[str, Dict[str, Any]],
    include_statistical_tests: bool = True,
) -> ComparisonSummary:
    """Compare multiple experiment runs.

    Args:
        run_data: Dictionary mapping run_id to loaded experiment data
        include_statistical_tests: Whether to run statistical tests

    Returns:
        ComparisonSummary with per-run summaries and comparisons
    """
    comparison = ComparisonSummary()
    comparison.run_ids = list(run_data.keys())

    # Generate summary for each run
    for run_id, data in run_data.items():
        try:
            summary = generate_summary(data)
            comparison.summaries[run_id] = summary
        except Exception as e:
            logger.error(f"Failed to generate summary for {run_id}: {e}")

    if not comparison.summaries:
        logger.warning("No valid summaries generated")
        return comparison

    # Find best runs
    summaries = comparison.summaries

    # Best by volume R²
    best_vol_r2 = max(
        summaries.items(),
        key=lambda x: x[1].performance.vol_r2,
    )
    comparison.best_run_by_vol_r2 = best_vol_r2[0]

    # Best by ODE readiness
    best_ode = max(
        summaries.items(),
        key=lambda x: x[1].ode_utility.ode_readiness,
    )
    comparison.best_run_by_ode_readiness = best_ode[0]

    # Best overall (considering multiple factors)
    comparison.best_run_overall = _select_best_overall(summaries)

    # Statistical tests
    if include_statistical_tests and len(summaries) >= 2:
        comparison.test_results = _run_comparison_tests(run_data, summaries)

    return comparison


def _select_best_overall(summaries: Dict[str, AnalysisSummary]) -> str:
    """Select best run considering multiple criteria.

    Uses a weighted scoring system:
    - ODE readiness: 40%
    - Volume R²: 30%
    - No collapse: 20%
    - Factor independence: 10%

    Args:
        summaries: Dictionary of run summaries

    Returns:
        run_id of best overall run
    """
    scores = {}

    for run_id, summary in summaries.items():
        score = 0.0

        # ODE readiness (40%)
        score += 0.40 * summary.ode_utility.ode_readiness

        # Volume R² (30%)
        score += 0.30 * summary.performance.vol_r2

        # No collapse penalty (20%)
        if not summary.collapse.residual_collapsed:
            score += 0.20
        if not summary.collapse.decoder_bypassed:
            score += 0.05  # Bonus for no decoder bypass

        # Factor independence (10%)
        score += 0.10 * summary.ode_utility.independence_score

        scores[run_id] = score

    return max(scores.items(), key=lambda x: x[1])[0]


def _run_comparison_tests(
    run_data: Dict[str, Dict[str, Any]],
    summaries: Dict[str, AnalysisSummary],
) -> List[StatisticalTestResults]:
    """Run statistical tests comparing runs.

    Args:
        run_data: Raw loaded data for each run
        summaries: Generated summaries for each run

    Returns:
        List of StatisticalTestResults
    """
    results = []
    run_ids = list(summaries.keys())

    # Only do pairwise tests if exactly 2 runs
    if len(run_ids) == 2:
        run1_id, run2_id = run_ids

        # Compare AU counts over epochs
        au1 = _get_au_history(run_data.get(run1_id, {}))
        au2 = _get_au_history(run_data.get(run2_id, {}))

        if len(au1) > 0 and len(au2) > 0:
            au_test = mann_whitney_u_test(au1, au2)
            au_test.notes = f"AU counts: {run1_id} vs {run2_id}. " + au_test.notes
            results.append(au_test)

    # Bootstrap CI for final metrics (all runs)
    for run_id, summary in summaries.items():
        # Volume R² CI
        vol_r2_data = _get_metric_history(run_data.get(run_id, {}), "vol_r2")
        if len(vol_r2_data) >= 10:
            # Use last 20% of training for stability estimate
            n_recent = max(5, len(vol_r2_data) // 5)
            recent_data = vol_r2_data[-n_recent:]
            ci_result = bootstrap_confidence_interval(recent_data)
            ci_result.notes = f"Vol R² CI for {run_id} (last {n_recent} epochs). " + ci_result.notes
            results.append(ci_result)

    # Variance homogeneity across partitions within each run
    for run_id, data in run_data.items():
        partition_vars = _get_partition_variances(data)
        if len(partition_vars) >= 2:
            var_test = levene_variance_test(*partition_vars.values())
            var_test.notes = f"Partition variance homogeneity for {run_id}. " + var_test.notes
            results.append(var_test)

    return results


def _get_au_history(data: Dict[str, Any]) -> np.ndarray:
    """Extract AU count history from run data."""
    au_df = data.get("au_history")
    if au_df is None or au_df.empty:
        return np.array([])

    if "au_count" in au_df.columns:
        return au_df["au_count"].dropna().values
    return np.array([])


def _get_metric_history(data: Dict[str, Any], metric_name: str) -> np.ndarray:
    """Extract a metric's history over epochs."""
    semantic_df = data.get("semantic_quality")
    if semantic_df is None or semantic_df.empty:
        return np.array([])

    # Map metric names to columns
    col_map = {
        "vol_r2": "r2_vol",
        "loc_r2": "r2_loc",
        "shape_r2": "r2_shape",
    }
    col = col_map.get(metric_name, metric_name)

    if col in semantic_df.columns:
        return semantic_df[col].dropna().values
    return np.array([])


def _get_partition_variances(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract variance history for each partition."""
    partition_df = data.get("partition_stats")
    if partition_df is None or partition_df.empty:
        return {}

    variances = {}
    for partition in ["z_vol", "z_loc", "z_shape", "z_residual"]:
        part_data = partition_df[partition_df["partition"] == partition]
        if not part_data.empty and "mu_var_mean" in part_data.columns:
            variances[partition] = part_data["mu_var_mean"].dropna().values

    return variances


def create_comparison_dataframe(
    summaries: Dict[str, AnalysisSummary],
) -> pd.DataFrame:
    """Create a DataFrame comparing key metrics across runs.

    Args:
        summaries: Dictionary of run summaries

    Returns:
        DataFrame with runs as rows and metrics as columns
    """
    records = []

    for run_id, summary in summaries.items():
        record = {
            "run_id": run_id,
            "vol_r2": summary.performance.vol_r2,
            "loc_r2": summary.performance.loc_r2,
            "shape_r2": summary.performance.shape_r2,
            "recon_mse": summary.performance.recon_mse,
            "ssim_mean": summary.performance.ssim_mean,
            "au_frac_residual": summary.collapse.au_frac_residual,
            "residual_collapsed": summary.collapse.residual_collapsed,
            "decoder_bypassed": summary.collapse.decoder_bypassed,
            "max_cross_corr": summary.ode_utility.max_cross_corr,
            "independence_score": summary.ode_utility.independence_score,
            "ode_readiness": summary.ode_utility.ode_readiness,
            "grade": summary.overall_grade.value,
            "loss_final": summary.trends.loss_final,
            "loss_converged": summary.trends.loss_converged,
        }
        records.append(record)

    return pd.DataFrame(records).set_index("run_id")
