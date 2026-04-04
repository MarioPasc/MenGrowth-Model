"""Plotting suite for LoRA-Ensemble uncertainty segmentation results.

Generates publication-quality figures from evaluation CSVs, JSONs,
volume data, and NIfTI predictions produced by the uncertainty_segmentation
module.

Usage:
    python -m experiments.uncertainty_segmentation.plotting.orchestrator \\
        /path/to/r8_M10_s42/ --output ./figures/ --format pdf
"""

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
    load_results,
)
from experiments.uncertainty_segmentation.plotting.style import setup_style

__all__ = ["EnsembleResultsData", "load_results", "setup_style"]
