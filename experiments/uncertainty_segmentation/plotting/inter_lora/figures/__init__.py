"""Registry of inter-LoRA figure and table modules."""

from __future__ import annotations

from experiments.uncertainty_segmentation.plotting.inter_lora.figures import (
    qual1_slice_grid,
    qual2_clustered_heatmap,
    quant1_dice_vs_rank,
    quant2_calib_epistemic,
    quant3_convergence_vs_rank,
)
from experiments.uncertainty_segmentation.plotting.inter_lora.tables import (
    tab1_summary,
    tab2_paired,
    tab3_model_complexity,
)

INTER_LORA_FIGURE_REGISTRY: dict[str, object] = {
    "quant1_dice_vs_rank": quant1_dice_vs_rank,
    "quant2_calib_epistemic": quant2_calib_epistemic,
    "quant3_convergence_vs_rank": quant3_convergence_vs_rank,
    "qual1_slice_grid": qual1_slice_grid,
    "qual2_clustered_heatmap": qual2_clustered_heatmap,
}

INTER_LORA_TABLE_REGISTRY: dict[str, object] = {
    "tab1_summary": tab1_summary,
    "tab2_paired": tab2_paired,
    "tab3_model_complexity": tab3_model_complexity,
}

FIGURE_OUTPUT_NAMES: dict[str, str] = {
    "quant1_dice_vs_rank": "quant1_dice_vs_rank",
    "quant2_calib_epistemic": "quant2_calibration_epistemic_vs_rank",
    "quant3_convergence_vs_rank": "quant3_convergence_vs_rank",
    "qual1_slice_grid": "qual1_slice_grid",
    "qual2_clustered_heatmap": "qual2_clustered_heatmap",
}

__all__ = [
    "INTER_LORA_FIGURE_REGISTRY",
    "INTER_LORA_TABLE_REGISTRY",
    "FIGURE_OUTPUT_NAMES",
]
