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
import time
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from experiments.uncertainty_segmentation.plotting import epistemic_metrics
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
    "fig_threshold_sensitivity": "fig05b_threshold_sensitivity",
    "fig_calibration": "fig06_calibration",
    "fig_best_worst": "fig07_best_worst",
    "fig_dice_compartments": "fig08_dice_compartments",
    "fig_violin_regions": "fig08b_violin_regions",
    "fig_inter_member_agreement": "fig09_inter_member_agreement",
    "fig_volume_bland_altman": "fig10_volume_bland_altman",
    "fig_volume_trajectories": "fig11_volume_trajectories",
    "fig_volume_uncertainty": "fig12_volume_uncertainty",
    "fig_boundary_disagreement": "fig13_boundary_disagreement",
    "fig_uncertainty_overlay": "fig14_uncertainty_overlay",
    "fig_epistemic_diagnosis": "fig15_bias_variance_landscape",
}

# Figures that should be saved to {run_dir.parent}/epistemic_summary/ in
# addition to the per-run figures/ directory, so the cross-rank artifact
# has a single canonical home.
CROSS_RANK_FIGURES: set[str] = {"fig_epistemic_diagnosis"}

SUPPLEMENTARY_ELIGIBLE: set[str] = {
    "fig_performance_comparison",
    "fig_paired_comparison",
    "fig_forest_plot",
    "fig_best_worst",
    "fig_inter_member_agreement",
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
    force_recompute: bool = False,
) -> None:
    """Generate all (or a subset of) figures.

    Args:
        run_dir: Path to the experiment run directory.
        output_dir: Where to save figures.  Defaults to ``{run_dir}/figures/``.
        config_path: Path to config YAML.
        fmt: Output format override (pdf, png, svg).
        dpi: DPI override.
        only: If given, generate only these figure names.
        force_recompute: If True, bypass cached epistemic-diagnostics CSVs
            and recompute them from raw evaluation CSVs.
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

    # Epistemic-uncertainty compute step: populates / refreshes cached
    # CSVs and JSON under {run_dir}/evaluation/ and
    # {run_dir.parent}/epistemic_summary/. Non-fatal if raw inputs are
    # missing — downstream figures will simply skip.
    try:
        epistemic_metrics.run_for_rank(run_dir, force=force_recompute)
    except FileNotFoundError as exc:
        logger.warning(
            "Epistemic compute skipped for %s: %s",
            run_dir.name,
            exc,
        )
    try:
        epistemic_metrics.run_cross_rank(run_dir, force=force_recompute)
    except Exception:
        logger.exception("Cross-rank epistemic aggregation failed")

    # Load data
    data = load_results(run_dir)
    logger.info("Data loaded: %d test scans, %d members", data.n_test_scans, data.n_members)
    logger.info("Label system: 3-channel BraTS-hierarchical (TC, WT, ET)")
    logger.info("  TC (ch0): labels 1|3 = meningioma mass (tumor core)")
    logger.info("  WT (ch1): seg > 0   = whole tumor incl. edema")
    logger.info("  ET (ch2): label 3   = enhancing tumor")

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
            name.replace("fig_", ""),
            {},
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

        # Also save cross-rank figure to the central epistemic_summary/
        # folder, so one canonical copy lives next to the summary CSVs.
        extra_path: Path | None = None
        if name in CROSS_RANK_FIGURES:
            extra_dir = run_dir.parent / "epistemic_summary"
            extra_dir.mkdir(parents=True, exist_ok=True)
            extra_path = extra_dir / f"{out_name}.{save_fmt}"
            fig.savefig(extra_path, dpi=save_dpi, transparent=transparent)

        plt.close(fig)

        elapsed = time.time() - t0
        if extra_path is not None:
            logger.info(
                "[OK]   %s  (%.1fs) -> %s (+ %s)",
                name,
                elapsed,
                out_path.name,
                extra_path,
            )
        else:
            logger.info("[OK]   %s  (%.1fs) -> %s", name, elapsed, out_path.name)
        n_ok += 1

        # Supplementary TC / ET variants for single-region figures
        if name in SUPPLEMENTARY_ELIGIBLE:
            for supp_region in ("tc", "et"):
                supp_config = {**fig_config, "region": supp_region}
                try:
                    fig_supp = module.plot(data, supp_config)
                except Exception:
                    logger.exception("[FAIL] %s_%s (supplementary)", name, supp_region)
                    n_fail += 1
                    continue
                if fig_supp is None:
                    continue
                supp_path = out / f"{out_name}_{supp_region}.{save_fmt}"
                fig_supp.savefig(supp_path, dpi=save_dpi, transparent=transparent)
                plt.close(fig_supp)
                logger.info("[OK]   %s_%s (supplementary) -> %s", name, supp_region, supp_path.name)
                n_ok += 1

    total_time = time.time() - t_total
    logger.info(
        "Done: %d generated, %d skipped, %d failed (%.1fs total)",
        n_ok,
        n_skip,
        n_fail,
        total_time,
    )
    logger.info("Figures saved to %s/", out)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate LoRA-Ensemble publication figures.",
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to run directory (e.g., r8_M10_s42/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to plot config YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["pdf", "png", "svg"],
        help="Output format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Output DPI",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Generate only these figures (by registry name)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Bypass cached epistemic-diagnostics CSVs and recompute.",
    )

    # Inter-LoRA comparison mode
    inter_group = parser.add_argument_group(
        "inter-lora",
        "Inter-rank LoRA comparison report",
    )
    inter_group.add_argument(
        "--inter-lora",
        action="store_true",
        help="Generate inter-rank comparison report. "
        "run_dir must be the ROOT directory containing r*_M*_s* subdirs.",
    )
    inter_group.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=None,
        help="Filter to specific ranks (e.g., --ranks 4 8 16 32)",
    )
    inter_group.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Explicit baseline directory override",
    )
    inter_group.add_argument(
        "--select-subjects",
        nargs="*",
        default=["auto"],
        help="Scan IDs for Qual1 (default: auto-select)",
    )
    inter_group.add_argument(
        "--bootstrap-n",
        type=int,
        default=1000,
        help="Number of bootstrap resamples",
    )
    inter_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    inter_group.add_argument(
        "--skip",
        nargs="+",
        default=None,
        choices=["quant1", "quant2", "qual1", "qual2", "tables"],
        help="Skip specific outputs",
    )
    inter_group.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any missing artefact",
    )
    inter_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and validate only, no rendering",
    )

    args = parser.parse_args()

    if args.inter_lora:
        from experiments.uncertainty_segmentation.plotting.inter_lora.orchestrate import (
            generate_inter_lora_report,
        )

        subjects = args.select_subjects
        if len(subjects) == 1 and subjects[0] == "auto":
            subjects = "auto"

        generate_inter_lora_report(
            root_dir=args.run_dir,
            out_root=args.output,
            config_path=args.config,
            ranks=args.ranks,
            baseline_dir=args.baseline_dir,
            select_subjects=subjects,
            bootstrap_n=args.bootstrap_n,
            seed=args.seed,
            dpi=args.dpi or 600,
            fmt=args.format or "pdf",
            skip=set(args.skip) if args.skip else None,
            strict=args.strict,
            dry_run=args.dry_run,
            force_recompute=args.force_recompute,
        )
        return

    generate_figures(
        run_dir=args.run_dir,
        output_dir=args.output,
        config_path=args.config,
        fmt=args.format,
        dpi=args.dpi,
        only=args.only,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
