"""Registry of all figure modules for the ensemble plotting suite."""

from __future__ import annotations

from experiments.uncertainty_segmentation.plotting.figures import (
    fig_best_worst,
    fig_boundary_disagreement,
    fig_calibration,
    fig_convergence,
    fig_dice_compartments,
    fig_epistemic_diagnosis,
    fig_forest_plot,
    fig_inter_member_agreement,
    fig_paired_comparison,
    fig_performance_comparison,
    fig_threshold_sensitivity,
    fig_training_curves,
    fig_uncertainty_overlay,
    fig_volume_bland_altman,
    fig_volume_trajectories,
    fig_volume_uncertainty,
)

FIGURE_REGISTRY: dict[str, object] = {
    "fig_training_curves": fig_training_curves,
    "fig_performance_comparison": fig_performance_comparison,
    "fig_paired_comparison": fig_paired_comparison,
    "fig_forest_plot": fig_forest_plot,
    "fig_convergence": fig_convergence,
    "fig_threshold_sensitivity": fig_threshold_sensitivity,
    "fig_calibration": fig_calibration,
    "fig_best_worst": fig_best_worst,
    "fig_dice_compartments": fig_dice_compartments,
    "fig_inter_member_agreement": fig_inter_member_agreement,
    "fig_volume_bland_altman": fig_volume_bland_altman,
    "fig_volume_trajectories": fig_volume_trajectories,
    "fig_volume_uncertainty": fig_volume_uncertainty,
    "fig_boundary_disagreement": fig_boundary_disagreement,
    "fig_uncertainty_overlay": fig_uncertainty_overlay,
    "fig_epistemic_diagnosis": fig_epistemic_diagnosis,
}

__all__ = ["FIGURE_REGISTRY"]
