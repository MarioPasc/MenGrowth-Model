"""SemiVAE Experiment Analysis Module.

This module provides a two-stage pipeline for comprehensive analysis
of SemiVAE training experiments:

Stage 1 - Statistics:
    Pure computation of metrics from experiment artifacts (CSVs, configs).
    Outputs machine-readable JSON/CSV files.

Stage 2 - Visualization:
    Generate plots and dashboards from Stage 1 outputs.
    Can run independently if Stage 1 outputs exist.

Usage:
    # Full pipeline
    python -m vae.tools.analysis analyze /path/to/run_dir

    # Stage 1 only
    python -m vae.tools.analysis stats /path/to/run_dir

    # Stage 2 only (requires Stage 1 outputs)
    python -m vae.tools.analysis visualize /path/to/analysis_dir

    # Multi-run comparison
    python -m vae.tools.analysis compare run1_dir run2_dir --output comparison/
"""

from .loaders import (
    load_experiment_data,
    load_metrics_csv,
    load_config,
    validate_experiment_directory,
)
from .schemas import (
    ExperimentMetadata,
    PerformanceMetrics,
    CollapseMetrics,
    ODEUtilityMetrics,
    AnalysisSummary,
)
from .pipeline import (
    run_stage1,
    run_stage2,
    run_full_pipeline,
    run_comparison,
)

__all__ = [
    # Loaders
    "load_experiment_data",
    "load_metrics_csv",
    "load_config",
    "validate_experiment_directory",
    # Schemas
    "ExperimentMetadata",
    "PerformanceMetrics",
    "CollapseMetrics",
    "ODEUtilityMetrics",
    "AnalysisSummary",
    # Pipeline
    "run_stage1",
    "run_stage2",
    "run_full_pipeline",
    "run_comparison",
]
