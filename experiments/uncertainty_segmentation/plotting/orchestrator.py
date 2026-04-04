"""Orchestrator for the LoRA-Ensemble plotting suite.

Loads config + data, calls each figure module, saves outputs.

Usage:
    python -m experiments.uncertainty_segmentation.plotting.orchestrator \\
        /path/to/r8_M10_s42/ \\
        --config experiments/uncertainty_segmentation/plotting/config.yaml \\
        --output ./figures/ \\
        --format pdf \\
        --only fig_training_curves fig_paired_comparison
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from experiments.uncertainty_segmentation.plotting.data_loader import load_results
from experiments.uncertainty_segmentation.plotting.figures import FIGURE_REGISTRY
from experiments.uncertainty_segmentation.plotting.style import setup_style

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Map from registry name → output filename prefix
FIGURE_NAMES = {
    "fig_training_curves": "fig01_training_curves",
    "fig_performance_comparison": "fig02_performance_comparison",
    "fig_paired_comparison": "fig03_paired_comparison",
    "fig_forest_plot": "fig04_forest_plot",
    "fig_convergence": "fig05_convergence",
    "fig_calibration": "fig06_calibration",
    "fig_best_worst": "fig07_best_worst",
    "fig_dice_compartments": "fig08_dice_compartments",
    "fig_inter_member_agreement": "fig09_inter_member_agreement",
    "fig_volume_bland_altman": "fig10_volume_bland_altman",
    "fig_volume_trajectories": "fig11_volume_trajectories",
    "fig_volume_uncertainty": "fig12_volume_uncertainty",
    "fig_boundary_disagreement": "fig13_boundary_disagreement",
    "fig_uncertainty_overlay": "fig14_uncertainty_overlay",
}


def _load_config(config_path: Path | None) -> dict:
    """Load plot configuration YAML.

    Args:
        config_path: Path to config YAML.  If None, uses the default
            config.yaml bundled with the plotting package.

    Returns:
        Parsed configuration dict.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        logger.warning("Config file not found: %s — using defaults", config_path)
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def generate_figures(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    fmt: str | None = None,
    dpi: int | None = None,
    only: list[str] | None = None,
) -> None:
    """Generate all (or a subset of) figures.

    Args:
        run_dir: Path to the experiment run directory.
        output_dir: Where to save figures.  Defaults to ``{run_dir}/figures/``.
        config_path: Path to config YAML.
        fmt: Output format override (pdf, png, svg).
        dpi: DPI override.
        only: If given, generate only these figure names.
    """
    run_dir = Path(run_dir)
    config = _load_config(Path(config_path) if config_path else None)

    style_config = config.get("style", {})
    figures_config = config.get("figures", {})

    save_fmt = fmt or style_config.get("save_format", "pdf")
    save_dpi = dpi or style_config.get("save_dpi", 300)
    transparent = style_config.get("transparent", False)

    out = Path(output_dir) if output_dir else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_results(run_dir)
    logger.info("Data loaded: %d test scans, %d members",
                data.n_test_scans, data.n_members)

    # Apply style
    setup_style(style_config)

    # Select which figures to generate
    if only:
        selected = {k: v for k, v in FIGURE_REGISTRY.items() if k in only}
        if not selected:
            logger.error("No matching figures for --only %s", only)
            return
    else:
        selected = FIGURE_REGISTRY

    t_total = time.time()
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for name, module in selected.items():
        fig_config = figures_config.get(
            name.replace("fig_", ""), {},
        )

        # Check enabled flag
        if not fig_config.get("enabled", True):
            logger.info("[SKIP] %s (disabled in config)", name)
            n_skip += 1
            continue

        t0 = time.time()
        try:
            fig = module.plot(data, fig_config)
        except Exception:
            logger.exception("[FAIL] %s", name)
            n_fail += 1
            continue

        if fig is None:
            logger.info("[SKIP] %s (data not available)", name)
            n_skip += 1
            continue

        # Save
        out_name = FIGURE_NAMES.get(name, name)
        out_path = out / f"{out_name}.{save_fmt}"
        fig.savefig(out_path, dpi=save_dpi, transparent=transparent)
        plt.close(fig)

        elapsed = time.time() - t0
        logger.info("[OK]   %s  (%.1fs) -> %s", name, elapsed, out_path.name)
        n_ok += 1

    total_time = time.time() - t_total
    logger.info(
        "Done: %d generated, %d skipped, %d failed (%.1fs total)",
        n_ok, n_skip, n_fail, total_time,
    )
    logger.info("Figures saved to %s/", out)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate LoRA-Ensemble publication figures.",
    )
    parser.add_argument(
        "run_dir", type=str,
        help="Path to run directory (e.g., r8_M10_s42/)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to plot config YAML",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format", type=str, default=None,
        choices=["pdf", "png", "svg"],
        help="Output format",
    )
    parser.add_argument(
        "--dpi", type=int, default=None,
        help="Output DPI",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Generate only these figures (by registry name)",
    )

    args = parser.parse_args()

    generate_figures(
        run_dir=args.run_dir,
        output_dir=args.output,
        config_path=args.config,
        fmt=args.format,
        dpi=args.dpi,
        only=args.only,
    )


if __name__ == "__main__":
    main()
