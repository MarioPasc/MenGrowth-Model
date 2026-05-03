# experiments/stage1_volumetric/analysis/plots.py
"""Figure generation for Stage 1 UQ growth prediction results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from growth.shared.lopo import LOPOResults

logger = logging.getLogger(__name__)


def generate_pit_histograms(
    calib_metrics: dict[str, dict],
    output_dir: Path,
    n_bins: int = 10,
) -> None:
    """Generate PIT histogram panel and save as PDF.

    Args:
        calib_metrics: Dict mapping model_name -> calibration metrics dict
            (must contain 'pit_values' key).
        output_dir: Root output directory. Figures saved to output_dir/figures/.
        n_bins: Number of histogram bins.
    """
    from growth.shared.calibration_plots import plot_pit_histogram_panel

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pit_dict = {name: cm["pit_values"] for name, cm in calib_metrics.items() if "pit_values" in cm}
    if not pit_dict:
        logger.warning("No PIT values available for histogram generation")
        return

    fig = plot_pit_histogram_panel(pit_dict, n_bins=n_bins)
    path = figures_dir / "pit_histogram_panel.pdf"
    fig.savefig(path, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info(f"Saved PIT panel to {path}")


def generate_sharpness_scatter(
    lopo_results: dict[str, LOPOResults],
    output_dir: Path,
    nominal: float = 0.95,
) -> None:
    """Generate sharpness-calibration scatter plot and save as PDF.

    Args:
        lopo_results: Dict mapping model_name -> LOPOResults.
        output_dir: Root output directory. Figures saved to output_dir/figures/.
        nominal: Target coverage level for reference line.
    """
    from growth.shared.calibration_plots import plot_sharpness_calibration_scatter

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    scatter_metrics = {}
    for model_name, results in lopo_results.items():
        m = results.aggregate_metrics
        cov_95 = m.get("last_from_rest/coverage_95", float("nan"))
        ci_w = m.get("last_from_rest/mean_ci_width_log", float("nan"))
        if np.isfinite(cov_95) and np.isfinite(ci_w):
            scatter_metrics[model_name] = {
                "coverage_95": cov_95,
                "mean_ci_width": ci_w,
            }

    if not scatter_metrics:
        logger.warning("No coverage/CI-width data available for scatter plot")
        return

    fig = plot_sharpness_calibration_scatter(scatter_metrics, nominal=nominal)
    path = figures_dir / "sharpness_calibration_scatter.pdf"
    fig.savefig(path, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info(f"Saved scatter to {path}")
