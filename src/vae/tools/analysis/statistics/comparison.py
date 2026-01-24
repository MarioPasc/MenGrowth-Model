"""Multi-run comparison logic.

Compares metrics across multiple experiment runs and performs
statistical tests to identify significant differences.
"""

import logging
from itertools import combinations
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
    cohens_d,
    cliff_delta,
    benjamini_hochberg,
    compute_stability_cv,
)

logger = logging.getLogger(__name__)

# Metrics tested in pairwise comparisons
COMPARISON_METRICS = [
    "vol_r2",
    "loc_r2",
    "shape_r2",
    "max_cross_corr",
    "au_frac_residual",
    "ode_readiness_expanded",
]

# Convergence thresholds: (metric_name, target_value, direction)
CONVERGENCE_THRESHOLDS = {
    "vol_r2": (0.85, ">="),
    "loc_r2": (0.90, ">="),
    "shape_r2": (0.35, ">="),
    "max_cross_corr": (0.30, "<"),
    "residual_au_frac": (0.10, ">="),
}


def compare_runs(
    run_data: Dict[str, Dict[str, Any]],
    include_statistical_tests: bool = True,
    name_map: Optional[Dict[str, str]] = None,
) -> ComparisonSummary:
    """Compare multiple experiment runs.

    Args:
        run_data: Dictionary mapping run_id to loaded experiment data
        include_statistical_tests: Whether to run statistical tests
        name_map: Optional mapping from run_id to display name

    Returns:
        ComparisonSummary with per-run summaries and comparisons
    """
    if name_map is None:
        name_map = {k: k for k in run_data.keys()}

    comparison = ComparisonSummary()
    comparison.run_ids = list(run_data.keys())
    comparison.name_map = name_map

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

    summaries = comparison.summaries

    # Find best runs
    best_vol_r2 = max(
        summaries.items(),
        key=lambda x: x[1].performance.vol_r2,
    )
    comparison.best_run_by_vol_r2 = best_vol_r2[0]

    best_ode = max(
        summaries.items(),
        key=lambda x: x[1].ode_utility.ode_readiness_expanded,
    )
    comparison.best_run_by_ode_readiness = best_ode[0]

    # Best overall (expanded scoring)
    comparison.best_run_overall = _select_best_overall(summaries)

    # Convergence epochs
    comparison.convergence_epochs = _compute_convergence_epochs(run_data)

    # Stability metrics
    comparison.stability_metrics = _compute_all_stability(run_data)

    # Statistical tests
    if include_statistical_tests and len(summaries) >= 2:
        comparison.test_results = _run_comparison_tests(run_data, summaries, name_map)

        # Bootstrap CIs
        comparison.confidence_intervals = _compute_all_bootstrap_cis(run_data)

    return comparison


def _select_best_overall(summaries: Dict[str, AnalysisSummary]) -> str:
    """Select best run using expanded ODE readiness scoring.

    Uses the expanded scoring formula:
    - 40% Volume R²
    - 20% Location R²
    - 15% Shape R² (clamped >= 0)
    - 15% Independence score
    - 10% Residual health

    Falls back to convergence speed as tiebreaker.

    Args:
        summaries: Dictionary of run summaries

    Returns:
        run_id of best overall run
    """
    scores = {}

    for run_id, summary in summaries.items():
        score = summary.ode_utility.ode_readiness_expanded
        # Tiebreaker: prefer non-collapsed residual
        if not summary.collapse.residual_collapsed:
            score += 0.001
        if not summary.collapse.decoder_bypassed:
            score += 0.0005

        scores[run_id] = score

    return max(scores.items(), key=lambda x: x[1])[0]


def _run_comparison_tests(
    run_data: Dict[str, Dict[str, Any]],
    summaries: Dict[str, AnalysisSummary],
    name_map: Dict[str, str],
) -> List[StatisticalTestResults]:
    """Run statistical tests comparing runs with FDR correction.

    Performs pairwise Mann-Whitney U tests for ALL pairs across
    all comparison metrics. Applies Benjamini-Hochberg correction.

    Args:
        run_data: Raw loaded data for each run
        summaries: Generated summaries for each run
        name_map: Run name mapping

    Returns:
        List of StatisticalTestResults with FDR-corrected p-values
    """
    results = []
    run_ids = list(summaries.keys())

    # Pairwise tests for ALL pairs
    pairwise_results = []
    for run_i, run_j in combinations(run_ids, 2):
        name_i = name_map.get(run_i, run_i)
        name_j = name_map.get(run_j, run_j)

        for metric_name in COMPARISON_METRICS:
            data_i = _get_metric_history(run_data.get(run_i, {}), metric_name)
            data_j = _get_metric_history(run_data.get(run_j, {}), metric_name)

            if len(data_i) < 10 or len(data_j) < 10:
                continue

            # Use last 20% of training for stability comparison
            n_i = max(5, len(data_i) // 5)
            n_j = max(5, len(data_j) // 5)
            tail_i = data_i[-n_i:]
            tail_j = data_j[-n_j:]

            # Mann-Whitney U test
            mw_test = mann_whitney_u_test(tail_i, tail_j)
            mw_test.notes = f"{metric_name}: {name_i} vs {name_j}. " + mw_test.notes

            # Effect sizes
            cd = cohens_d(tail_i, tail_j)
            cl = cliff_delta(tail_i, tail_j)

            mw_test.effect_size = cd.effect_size
            mw_test.effect_interpretation = cd.effect_interpretation

            pairwise_results.append(mw_test)

    # Apply Benjamini-Hochberg FDR correction
    if pairwise_results:
        raw_p_values = [t.p_value for t in pairwise_results]
        fdr_results = benjamini_hochberg(raw_p_values, alpha=0.05)

        for test_result, (adj_p, is_sig) in zip(pairwise_results, fdr_results):
            test_result.p_adjusted = adj_p
            test_result.significant = is_sig

    results.extend(pairwise_results)

    # Bootstrap CI for all key metrics (all runs)
    for run_id in run_ids:
        name = name_map.get(run_id, run_id)
        for metric_name in ["vol_r2", "loc_r2", "shape_r2", "max_cross_corr",
                            "au_frac_residual", "ode_readiness_expanded"]:
            metric_data = _get_metric_history(run_data.get(run_id, {}), metric_name)
            if len(metric_data) >= 10:
                n_recent = max(5, len(metric_data) // 5)
                recent_data = metric_data[-n_recent:]
                ci_result = bootstrap_confidence_interval(recent_data)
                ci_result.notes = (
                    f"{metric_name} CI for {name} "
                    f"(last {n_recent} epochs). " + ci_result.notes
                )
                results.append(ci_result)

    # Variance homogeneity across partitions within each run
    for run_id, data in run_data.items():
        name = name_map.get(run_id, run_id)
        partition_vars = _get_partition_variances(data)
        if len(partition_vars) >= 2:
            var_test = levene_variance_test(*partition_vars.values())
            var_test.notes = f"Partition variance homogeneity for {name}. " + var_test.notes
            results.append(var_test)

    return results


def _compute_convergence_epochs(
    run_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Compute first epoch each threshold is met for each run.

    Args:
        run_data: Dictionary mapping run_id to loaded data

    Returns:
        {run_id: {threshold_name: first_epoch or -1}}
    """
    result = {}

    for run_id, data in run_data.items():
        run_convergence = {}

        for threshold_name, (target, direction) in CONVERGENCE_THRESHOLDS.items():
            metric_history = _get_metric_history_with_epochs(data, threshold_name)

            if len(metric_history) == 0:
                run_convergence[threshold_name] = -1
                continue

            # Find first epoch meeting threshold
            first_epoch = -1
            for epoch, value in metric_history:
                if direction == ">=" and value >= target:
                    first_epoch = int(epoch)
                    break
                elif direction == "<" and value < target:
                    first_epoch = int(epoch)
                    break

            run_convergence[threshold_name] = first_epoch

        result[run_id] = run_convergence

    return result


def _compute_all_stability(
    run_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute stability CV for all metrics across all runs.

    Args:
        run_data: Dictionary mapping run_id to loaded data

    Returns:
        {run_id: {metric_name: cv_value}}
    """
    result = {}

    for run_id, data in run_data.items():
        run_stability = {}

        for metric_name in ["vol_r2", "loc_r2", "shape_r2", "max_cross_corr",
                            "ode_readiness_expanded"]:
            values = _get_metric_history(data, metric_name)
            if len(values) >= 10:
                cv = compute_stability_cv(values, tail_fraction=0.20)
                run_stability[f"{metric_name}_cv"] = cv

        # Also compute loss stability
        metrics_df = data.get("metrics")
        if metrics_df is not None and not metrics_df.empty:
            for col in ["val_epoch/loss", "val_epoch/recon"]:
                if col in metrics_df.columns:
                    values = metrics_df[col].dropna().values
                    if len(values) >= 10:
                        cv = compute_stability_cv(values, tail_fraction=0.20)
                        clean_name = col.replace("/", "_").replace("val_epoch_", "")
                        run_stability[f"{clean_name}_cv"] = cv

        result[run_id] = run_stability

    return result


def _compute_all_bootstrap_cis(
    run_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, tuple]]:
    """Compute bootstrap CIs for key metrics in each run.

    Args:
        run_data: Dictionary mapping run_id to loaded data

    Returns:
        {run_id: {metric_name: (lower, upper)}}
    """
    result = {}

    for run_id, data in run_data.items():
        run_cis = {}

        for metric_name in ["vol_r2", "loc_r2", "shape_r2", "max_cross_corr",
                            "au_frac_residual", "ode_readiness_expanded"]:
            values = _get_metric_history(data, metric_name)
            if len(values) >= 10:
                n_recent = max(5, len(values) // 5)
                recent = values[-n_recent:]
                ci = bootstrap_confidence_interval(recent)
                if ci.confidence_interval is not None:
                    run_cis[metric_name] = ci.confidence_interval

        result[run_id] = run_cis

    return result


def compare_convergence_rates(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, int]]:
    """Compute convergence rates for comparison output.

    Public interface for pipeline.py.

    Args:
        run_data: Dictionary mapping run_id to loaded data
        name_map: Optional run name mapping

    Returns:
        {run_id: {threshold_name: first_epoch or -1}}
    """
    return _compute_convergence_epochs(run_data)


def compute_config_diff(
    run_data: Dict[str, Dict[str, Any]],
    name_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare configs across runs and return differing parameters.

    Flattens nested YAML configs and identifies parameters that
    differ between any two runs.

    Args:
        run_data: Dictionary mapping run_id to loaded data
        name_map: Optional run name mapping

    Returns:
        {param_path: {display_name: value}} for differing params only
    """
    if name_map is None:
        name_map = {k: k for k in run_data.keys()}

    def flatten_dict(d, prefix=""):
        """Flatten nested dict with dot-separated keys."""
        items = {}
        if not isinstance(d, dict):
            return items
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(flatten_dict(value, full_key))
            elif isinstance(value, (list, tuple)):
                items[full_key] = str(value)
            else:
                items[full_key] = value
        return items

    # Flatten all configs
    flat_configs = {}
    for run_id, data in run_data.items():
        config = data.get("config", {})
        display_name = name_map.get(run_id, run_id)
        flat_configs[display_name] = flatten_dict(config)

    if len(flat_configs) < 2:
        return {}

    # Get all unique keys
    all_keys = set()
    for fc in flat_configs.values():
        all_keys.update(fc.keys())

    # Find keys with differing values
    diff = {}
    display_names = list(flat_configs.keys())

    for key in sorted(all_keys):
        values = []
        for name in display_names:
            values.append(flat_configs[name].get(key, "<missing>"))

        # Check if values differ
        unique_values = set(str(v) for v in values)
        if len(unique_values) > 1:
            # Skip internal/path-based keys that always differ
            skip_keys = ["data.root_dir", "data.cache_dir", "output.run_dir",
                         "output.base_dir", "wandb.run_id", "wandb.name"]
            if any(key.startswith(sk) or key == sk for sk in skip_keys):
                continue

            diff[key] = {name: flat_configs[name].get(key, "<missing>")
                         for name in display_names}

    return diff


def _get_au_history(data: Dict[str, Any]) -> np.ndarray:
    """Extract AU count history from run data."""
    au_df = data.get("au_history")
    if au_df is None or au_df.empty:
        return np.array([])

    if "au_count" in au_df.columns:
        return au_df["au_count"].dropna().values
    return np.array([])


def _get_metric_history(data: Dict[str, Any], metric_name: str) -> np.ndarray:
    """Extract a metric's history over epochs.

    Handles multiple data sources (semantic_quality, cross_correlation,
    au_history, metrics).
    """
    # Direct column mappings for semantic_quality.csv (long format)
    semantic_df = data.get("semantic_quality")
    if semantic_df is not None and not semantic_df.empty:
        if metric_name in ("vol_r2", "loc_r2", "shape_r2"):
            # Handle long format with "partition" column
            if "partition" in semantic_df.columns:
                partition_map = {"vol_r2": "z_vol", "loc_r2": "z_loc", "shape_r2": "z_shape"}
                partition = partition_map.get(metric_name)
                if partition:
                    part_data = semantic_df[semantic_df["partition"] == partition]
                    if not part_data.empty and "r2" in part_data.columns:
                        return part_data["r2"].dropna().values
            else:
                # Wide format
                col_map = {"vol_r2": "r2_vol", "loc_r2": "r2_loc", "shape_r2": "r2_shape"}
                col = col_map.get(metric_name, metric_name)
                if col in semantic_df.columns:
                    return semantic_df[col].dropna().values

    # Cross-correlation metrics
    if metric_name == "max_cross_corr":
        cross_df = data.get("cross_correlation")
        if cross_df is not None and not cross_df.empty:
            if "partition_i" in cross_df.columns:
                # Long format: compute max per epoch
                epochs = sorted(cross_df["epoch"].unique())
                values = []
                for ep in epochs:
                    ep_data = cross_df[cross_df["epoch"] == ep]
                    corr_col = "abs_correlation" if "abs_correlation" in ep_data.columns else "correlation"
                    if corr_col in ep_data.columns:
                        max_corr = ep_data[corr_col].abs().max()
                        values.append(max_corr)
                return np.array(values) if values else np.array([])

    # AU fraction for residual
    if metric_name in ("au_frac_residual", "residual_au_frac"):
        partition_df = data.get("partition_stats")
        if partition_df is not None and not partition_df.empty:
            residual = partition_df[partition_df["partition"] == "z_residual"]
            if not residual.empty and "au_frac" in residual.columns:
                return residual["au_frac"].dropna().values

    # ODE readiness expanded - compute from components
    if metric_name == "ode_readiness_expanded":
        vol = _get_metric_history(data, "vol_r2")
        loc = _get_metric_history(data, "loc_r2")
        shape = _get_metric_history(data, "shape_r2")
        indep = 1.0 - _get_metric_history(data, "max_cross_corr")
        resid = _get_metric_history(data, "au_frac_residual")

        # Align lengths
        min_len = min(len(vol), len(loc), len(shape), len(indep)) if all(
            len(x) > 0 for x in [vol, loc, shape, indep]
        ) else 0

        if min_len > 0:
            vol = vol[:min_len]
            loc = loc[:min_len]
            shape = shape[:min_len]
            indep = indep[:min_len]
            resid_health = np.minimum(resid[:min_len] / 0.10, 1.0) if len(resid) >= min_len else np.zeros(min_len)

            expanded = (
                0.40 * vol +
                0.20 * loc +
                0.15 * np.maximum(0.0, shape) +
                0.15 * np.maximum(0.0, indep) +
                0.10 * resid_health
            )
            return expanded

    # Fallback: try metrics.csv columns
    metrics_df = data.get("metrics")
    if metrics_df is not None and not metrics_df.empty:
        col_attempts = [
            f"val_epoch/{metric_name}",
            f"train_epoch/{metric_name}",
            metric_name,
        ]
        for col in col_attempts:
            if col in metrics_df.columns:
                return metrics_df[col].dropna().values

    return np.array([])


def _get_metric_history_with_epochs(
    data: Dict[str, Any],
    metric_name: str,
) -> List[tuple]:
    """Extract metric history with epoch numbers.

    Returns list of (epoch, value) tuples.
    """
    semantic_df = data.get("semantic_quality")
    if semantic_df is not None and not semantic_df.empty:
        if metric_name in ("vol_r2", "loc_r2", "shape_r2"):
            if "partition" in semantic_df.columns:
                partition_map = {"vol_r2": "z_vol", "loc_r2": "z_loc", "shape_r2": "z_shape"}
                partition = partition_map.get(metric_name)
                if partition:
                    part_data = semantic_df[semantic_df["partition"] == partition].copy()
                    if not part_data.empty and "r2" in part_data.columns and "epoch" in part_data.columns:
                        valid = part_data.dropna(subset=["r2", "epoch"])
                        return list(zip(valid["epoch"].values, valid["r2"].values))

    if metric_name == "max_cross_corr":
        cross_df = data.get("cross_correlation")
        if cross_df is not None and not cross_df.empty and "partition_i" in cross_df.columns:
            epochs = sorted(cross_df["epoch"].unique())
            result = []
            for ep in epochs:
                ep_data = cross_df[cross_df["epoch"] == ep]
                corr_col = "abs_correlation" if "abs_correlation" in ep_data.columns else "correlation"
                if corr_col in ep_data.columns:
                    max_corr = ep_data[corr_col].abs().max()
                    result.append((ep, max_corr))
            return result

    if metric_name in ("au_frac_residual", "residual_au_frac"):
        partition_df = data.get("partition_stats")
        if partition_df is not None and not partition_df.empty:
            residual = partition_df[partition_df["partition"] == "z_residual"]
            if not residual.empty and "au_frac" in residual.columns and "epoch" in residual.columns:
                valid = residual.dropna(subset=["au_frac", "epoch"])
                return list(zip(valid["epoch"].values, valid["au_frac"].values))

    return []


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
    name_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Create a DataFrame comparing key metrics across runs.

    Args:
        summaries: Dictionary of run summaries
        name_map: Optional mapping from run_id to display name

    Returns:
        DataFrame with runs as rows and metrics as columns
    """
    if name_map is None:
        name_map = {k: k for k in summaries.keys()}

    records = []

    for run_id, summary in summaries.items():
        display_name = name_map.get(run_id, run_id)
        record = {
            "run_name": display_name,
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
            "ode_readiness_expanded": summary.ode_utility.ode_readiness_expanded,
            "residual_health": summary.ode_utility.residual_health,
            "grade": summary.overall_grade.value,
            "loss_final": summary.trends.loss_final,
            "loss_converged": summary.trends.loss_converged,
        }
        records.append(record)

    return pd.DataFrame(records)
