"""Stage 2: Visualization Modules.

This subpackage generates plots and dashboards from Stage 1 outputs.
Can run independently if Stage 1 CSVs/JSONs exist.

Modules:
    performance_plots: Reconstruction quality, semantic R² curves
    collapse_plots: AU progression, partition activity heatmaps
    ode_plots: Cross-partition correlations, factor independence
    training_plots: Loss breakdown, schedule progression
    dashboard: Multi-panel summary dashboard
    comparison_plots: Multi-run overlay and comparison plots
"""

from .performance_plots import plot_performance_metrics
from .collapse_plots import plot_collapse_metrics
from .ode_plots import plot_ode_utility
from .training_plots import plot_training_dynamics
from .dashboard import create_dashboard
from .comparison_plots import plot_comparison

__all__ = [
    "plot_performance_metrics",
    "plot_collapse_metrics",
    "plot_ode_utility",
    "plot_training_dynamics",
    "create_dashboard",
    "plot_comparison",
]
