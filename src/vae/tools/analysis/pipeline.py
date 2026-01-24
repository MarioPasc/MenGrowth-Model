"""Pipeline orchestration for analysis stages.

Coordinates Stage 1 (statistics) and Stage 2 (visualization) execution.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .loaders import load_experiment_data, validate_experiment_directory
from .schemas import AnalysisSummary, ComparisonSummary
from .writers import (
    setup_output_directory,
    write_summary_json,
    write_metrics_csv,
    write_comparison_json,
    write_manifest,
)

logger = logging.getLogger(__name__)


def run_stage1(
    run_dir: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Stage 1: Statistical analysis.

    Loads experiment data, computes all metrics, and writes outputs.

    Args:
        run_dir: Path to experiment run directory
        output_dir: Output directory (default: {run_dir}/analysis)

    Returns:
        Dictionary with:
            - summary: AnalysisSummary
            - output_dir: Path to output directory
            - files: List of generated files
    """
    from .statistics.summary import generate_summary
    from .statistics.performance import compute_performance_history
    from .statistics.collapse import compute_collapse_history
    from .statistics.ode_utility import compute_ode_utility_history
    from .statistics.trends import compute_loss_history, compute_schedule_history

    logger.info(f"Stage 1: Loading experiment data from {run_dir}")

    # Validate and load
    is_valid, found_files, missing = validate_experiment_directory(run_dir)
    if not is_valid:
        raise ValueError(f"Invalid experiment directory. Missing: {missing}")

    data = load_experiment_data(run_dir)

    # Setup output
    if output_dir is None:
        output_path = setup_output_directory(run_dir)
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    # Generate summary
    logger.info("Computing analysis summary...")
    summary = generate_summary(data)
    filepath = write_summary_json(summary, output_path)
    generated_files.append(filepath)

    # Compute and write metric histories
    logger.info("Computing performance history...")
    perf_df = compute_performance_history(data)
    if not perf_df.empty:
        filepath = write_metrics_csv(perf_df, output_path, "performance_metrics.csv")
        generated_files.append(filepath)

    logger.info("Computing collapse history...")
    collapse_df = compute_collapse_history(data)
    if not collapse_df.empty:
        filepath = write_metrics_csv(collapse_df, output_path, "collapse_metrics.csv")
        generated_files.append(filepath)

    logger.info("Computing ODE utility history...")
    ode_df = compute_ode_utility_history(data)
    if not ode_df.empty:
        filepath = write_metrics_csv(ode_df, output_path, "ode_utility_metrics.csv")
        generated_files.append(filepath)

    logger.info("Computing training dynamics...")
    loss_df = compute_loss_history(data)
    if not loss_df.empty:
        filepath = write_metrics_csv(loss_df, output_path, "training_dynamics.csv")
        generated_files.append(filepath)

    schedule_df = compute_schedule_history(data)
    if not schedule_df.empty:
        filepath = write_metrics_csv(schedule_df, output_path, "schedule_history.csv")
        generated_files.append(filepath)

    logger.info(f"Stage 1 complete. Generated {len(generated_files)} files.")

    return {
        "summary": summary,
        "data": data,
        "output_dir": str(output_path),
        "files": generated_files,
    }


def run_stage2(
    run_dir: str,
    output_dir: Optional[str] = None,
    summary: Optional[AnalysisSummary] = None,
    data: Optional[Dict[str, Any]] = None,
    dpi: int = 150,
    format: str = "png",
) -> Dict[str, Any]:
    """Run Stage 2: Visualization.

    Generates plots from experiment data and/or Stage 1 outputs.

    Args:
        run_dir: Path to experiment run directory
        output_dir: Output directory (default: {run_dir}/analysis)
        summary: Pre-computed AnalysisSummary (optional)
        data: Pre-loaded experiment data (optional)
        dpi: Plot resolution
        format: Image format ("png" or "pdf")

    Returns:
        Dictionary with:
            - output_dir: Path to output directory
            - plots: Dictionary mapping plot names to file paths
    """
    from .visualization.performance_plots import plot_performance_metrics
    from .visualization.collapse_plots import plot_collapse_metrics
    from .visualization.ode_plots import plot_ode_utility
    from .visualization.training_plots import plot_training_dynamics
    from .visualization.dashboard import create_dashboard
    from .statistics.summary import generate_summary

    logger.info(f"Stage 2: Generating visualizations")

    # Load data if not provided
    if data is None:
        data = load_experiment_data(run_dir)

    # Generate summary if not provided
    if summary is None:
        summary = generate_summary(data)

    # Setup output
    if output_dir is None:
        output_path = setup_output_directory(run_dir)
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    all_plots = {}

    # Generate all plot types
    logger.info("Generating performance plots...")
    plots = plot_performance_metrics(data, str(plots_dir), dpi=dpi, format=format)
    all_plots.update(plots)

    logger.info("Generating collapse diagnostic plots...")
    plots = plot_collapse_metrics(data, str(plots_dir), dpi=dpi, format=format)
    all_plots.update(plots)

    logger.info("Generating ODE utility plots...")
    plots = plot_ode_utility(data, str(plots_dir), dpi=dpi, format=format)
    all_plots.update(plots)

    logger.info("Generating training dynamics plots...")
    plots = plot_training_dynamics(data, str(plots_dir), dpi=dpi, format=format)
    all_plots.update(plots)

    logger.info("Creating dashboard...")
    dashboard_path = create_dashboard(data, summary, str(plots_dir), dpi=dpi, format=format)
    if dashboard_path:
        all_plots["dashboard"] = dashboard_path

    logger.info(f"Stage 2 complete. Generated {len(all_plots)} plots.")

    return {
        "output_dir": str(output_path),
        "plots": all_plots,
    }


def run_full_pipeline(
    run_dir: str,
    output_dir: Optional[str] = None,
    dpi: int = 150,
    format: str = "png",
) -> Dict[str, Any]:
    """Run complete analysis pipeline (Stage 1 + Stage 2).

    Args:
        run_dir: Path to experiment run directory
        output_dir: Output directory (default: {run_dir}/analysis)
        dpi: Plot resolution
        format: Image format

    Returns:
        Dictionary with combined results from both stages
    """
    logger.info(f"Running full analysis pipeline on {run_dir}")

    # Stage 1
    stage1_result = run_stage1(run_dir, output_dir)

    # Stage 2 (reuse data and summary from stage 1)
    stage2_result = run_stage2(
        run_dir,
        output_dir=stage1_result["output_dir"],
        summary=stage1_result["summary"],
        data=stage1_result["data"],
        dpi=dpi,
        format=format,
    )

    # Write manifest
    manifest_path = write_manifest(stage1_result["output_dir"])

    result = {
        "run_dir": run_dir,
        "output_dir": stage1_result["output_dir"],
        "summary": stage1_result["summary"],
        "files": stage1_result["files"],
        "plots": stage2_result["plots"],
        "manifest": manifest_path,
    }

    logger.info(f"Full pipeline complete. Output: {stage1_result['output_dir']}")
    return result


def _resolve_run_names(
    run_ids: List[str],
    run_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Create mapping from folder-based run_id to display name.

    Args:
        run_ids: List of folder-based run identifiers
        run_names: Optional user-provided display names

    Returns:
        Dictionary mapping run_id -> display_name
    """
    name_map = {}
    for i, run_id in enumerate(run_ids):
        if run_names and i < len(run_names):
            name_map[run_id] = run_names[i]
        else:
            name_map[run_id] = run_id
    return name_map


def run_comparison(
    run_dirs: List[str],
    output_dir: str,
    run_names: Optional[List[str]] = None,
    include_statistical_tests: bool = True,
    dpi: int = 150,
    format: str = "png",
) -> Dict[str, Any]:
    """Run comparison analysis across multiple runs.

    Args:
        run_dirs: List of paths to experiment run directories
        output_dir: Output directory for comparison results
        run_names: Optional human-readable names for runs
        include_statistical_tests: Whether to run statistical tests
        dpi: Plot resolution
        format: Image format

    Returns:
        Dictionary with comparison results
    """
    from .statistics.comparison import (
        compare_runs,
        create_comparison_dataframe,
        compare_convergence_rates,
        compute_config_diff,
    )
    from .visualization.comparison_plots import plot_comparison, create_comparison_table

    logger.info(f"Running comparison analysis on {len(run_dirs)} runs")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all run data
    run_data = {}
    run_id_order = []
    for run_dir in run_dirs:
        try:
            is_valid, _, _ = validate_experiment_directory(run_dir)
            if is_valid:
                data = load_experiment_data(run_dir)
                run_id = Path(run_dir).name
                run_data[run_id] = data
                run_id_order.append(run_id)
                logger.info(f"Loaded: {run_id}")
            else:
                logger.warning(f"Skipping invalid directory: {run_dir}")
        except Exception as e:
            logger.error(f"Failed to load {run_dir}: {e}")

    if len(run_data) < 2:
        raise ValueError("Need at least 2 valid runs for comparison")

    # Resolve display names
    name_map = _resolve_run_names(run_id_order, run_names)
    logger.info(f"Run names: {name_map}")

    # Run comparison
    comparison = compare_runs(
        run_data,
        include_statistical_tests=include_statistical_tests,
        name_map=name_map,
    )

    # Write outputs
    files = []

    # Comparison summary JSON
    filepath = write_comparison_json(comparison, output_path)
    files.append(filepath)

    # Comparison table CSV
    comparison_df = create_comparison_dataframe(comparison.summaries, name_map=name_map)
    if not comparison_df.empty:
        filepath = write_metrics_csv(comparison_df, output_path, "comparison_metrics.csv")
        files.append(filepath)

    # Config diff
    config_diff = compute_config_diff(run_data, name_map=name_map)
    if config_diff:
        import pandas as pd
        diff_records = []
        for param, values in config_diff.items():
            record = {"parameter": param}
            record.update(values)
            diff_records.append(record)
        if diff_records:
            diff_df = pd.DataFrame(diff_records)
            filepath = write_metrics_csv(diff_df, output_path, "config_diff.csv")
            files.append(filepath)

    # Convergence timeline
    convergence = compare_convergence_rates(run_data, name_map=name_map)
    if convergence:
        import pandas as pd
        conv_records = []
        thresholds_info = {
            "vol_r2": (">=", 0.85),
            "loc_r2": (">=", 0.90),
            "shape_r2": (">=", 0.35),
            "max_cross_corr": ("<", 0.30),
            "residual_au_frac": (">=", 0.10),
        }
        for threshold_name, (direction, target) in thresholds_info.items():
            record = {"threshold": threshold_name, "target": target, "direction": direction}
            epochs = []
            for run_id in run_id_order:
                display_name = name_map.get(run_id, run_id)
                epoch = convergence.get(run_id, {}).get(threshold_name, -1)
                record[display_name] = epoch if epoch >= 0 else "never"
                if epoch >= 0:
                    epochs.append((display_name, epoch))
            if epochs:
                record["fastest_run"] = min(epochs, key=lambda x: x[1])[0]
            else:
                record["fastest_run"] = "none"
            conv_records.append(record)
        if conv_records:
            conv_df = pd.DataFrame(conv_records)
            filepath = write_metrics_csv(conv_df, output_path, "convergence_timeline.csv")
            files.append(filepath)

    # Statistical tests CSV
    if comparison.test_results:
        import pandas as pd
        test_records = []
        for t in comparison.test_results:
            test_records.append({
                "test_name": t.test_name,
                "statistic": t.statistic,
                "p_value": t.p_value,
                "p_adjusted": getattr(t, "p_adjusted", t.p_value),
                "significant": t.significant,
                "effect_size": t.effect_size,
                "effect_interpretation": getattr(t, "effect_interpretation", ""),
                "notes": t.notes,
            })
        if test_records:
            tests_df = pd.DataFrame(test_records)
            filepath = write_metrics_csv(tests_df, output_path, "statistical_tests.csv")
            files.append(filepath)

    # Stability metrics CSV
    if hasattr(comparison, "stability_metrics") and comparison.stability_metrics:
        import pandas as pd
        stab_records = []
        for run_id, metrics in comparison.stability_metrics.items():
            display_name = name_map.get(run_id, run_id)
            record = {"run": display_name}
            record.update(metrics)
            stab_records.append(record)
        if stab_records:
            stab_df = pd.DataFrame(stab_records)
            filepath = write_metrics_csv(stab_df, output_path, "stability_metrics.csv")
            files.append(filepath)

    # Generate comparison plots
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    plots = plot_comparison(
        run_data, str(plots_dir),
        name_map=name_map,
        dpi=dpi,
        format=format,
    )

    # Table as CSV
    table_path = create_comparison_table(run_data, str(output_path), name_map=name_map)
    if table_path:
        files.append(table_path)

    # Write manifest
    manifest_path = write_manifest(str(output_path))
    files.append(manifest_path)

    logger.info(f"Comparison complete. Output: {output_path}")

    return {
        "output_dir": str(output_path),
        "comparison": comparison,
        "name_map": name_map,
        "files": files,
        "plots": plots,
        "best_run": comparison.best_run_overall,
    }
