"""Stage 1: Statistical Analysis Modules.

This subpackage contains pure functions for computing analysis metrics
from experiment artifacts. No visualization or side effects.

Modules:
    performance: Reconstruction quality, semantic R² metrics
    collapse: Active units, variance analysis, decoder bypass detection
    ode_utility: Factor independence, ODE readiness scores
    trends: Convergence analysis, stability metrics
    summary: Aggregate metrics into summary.json
    statistical_tests: Mann-Whitney U, bootstrap CI, Spearman correlation
    comparison: Multi-run comparison logic
"""

from .performance import compute_performance_metrics
from .collapse import (
    compute_collapse_metrics,
    compute_collapse_metrics_with_trajectory,
    analyze_variance_trajectory,
)
from .ode_utility import compute_ode_utility_metrics
from .trends import compute_trend_metrics
from .summary import generate_summary
from .statistical_tests import (
    mann_whitney_u_test,
    bootstrap_confidence_interval,
    spearman_correlation,
    levene_variance_test,
)
from .comparison import compare_runs

__all__ = [
    "compute_performance_metrics",
    "compute_collapse_metrics",
    "compute_collapse_metrics_with_trajectory",
    "analyze_variance_trajectory",
    "compute_ode_utility_metrics",
    "compute_trend_metrics",
    "generate_summary",
    "mann_whitney_u_test",
    "bootstrap_confidence_interval",
    "spearman_correlation",
    "levene_variance_test",
    "compare_runs",
]
