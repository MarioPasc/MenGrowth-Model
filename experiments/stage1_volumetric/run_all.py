# experiments/stage1_volumetric/run_all.py
"""Stage 1: Uncertainty-Propagated Volume Prediction — Full Pipeline.

Runs all configured models sequentially (with resume support), then
computes calibration metrics, paired comparisons, and generates figures.

Replaces the monolithic run_stage1_uq.py. For model-level parallelism
on Picasso, use run_single_model.py via SLURM instead.

Usage::

    cd /home/mpascual/research/code/MenGrowth-Model
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_all \\
        --config experiments/stage1_volumetric/configs/config_uq.yaml

    # Resume after interruption (cached models are skipped):
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_all \\
        --config experiments/stage1_volumetric/configs/config_uq.yaml

    # Force re-run everything:
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_all \\
        --config experiments/stage1_volumetric/configs/config_uq.yaml --force
"""

import argparse
import logging
import time
from pathlib import Path

from experiments.stage1_volumetric.analysis.plots import (
    generate_pit_histograms,
    generate_sharpness_scatter,
)
from experiments.stage1_volumetric.analysis.summary import (
    print_comparison_tables,
    print_summary_table,
)
from experiments.stage1_volumetric.engine.data import load_config, load_trajectories
from experiments.stage1_volumetric.engine.model_registry import build_model_configs
from experiments.stage1_volumetric.engine.runner import run_single_model
from experiments.stage1_volumetric.stats.calibration import compute_calibration_metrics
from experiments.stage1_volumetric.stats.comparisons import (
    run_paired_comparisons,
    write_comparison_table,
)
from experiments.utils.experiment_output import save_experiment_metadata
from growth.shared.bootstrap import BootstrapResult
from growth.shared.lopo import LOPOResults

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full Stage 1 UQ pipeline."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Uncertainty-Propagated Volume Prediction — Full Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/stage1_volumetric/configs/config_uq.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Use days-from-baseline instead of ordinal time",
    )
    parser.add_argument(
        "--estimator",
        choices=["mean_std", "median_mad", "mask_mean"],
        default=None,
        help="Override uncertainty estimator",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all models even if cached results exist",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])

    if args.real_time:
        cfg["time"]["variable"] = "days_from_baseline"
        logger.info("CLI override: time_variable=days_from_baseline")
    if args.estimator:
        cfg["uncertainty"]["estimator"] = args.estimator
        logger.info(f"CLI override: estimator={args.estimator}")

    # =========================================================================
    # Step 1: Load trajectories
    # =========================================================================
    logger.info("=== Step 1: Load Trajectories ===")
    trajectories = load_trajectories(cfg)

    # =========================================================================
    # Step 2: LOPO-CV for all models (sequential, with resume)
    # =========================================================================
    logger.info("=== Step 2: LOPO-CV ===")
    model_configs = build_model_configs(cfg)
    n_models = len(model_configs)

    lopo_results: dict[str, LOPOResults] = {}
    all_bootstrap_cis: dict[str, dict[str, BootstrapResult]] = {}

    bootstrap_cfg = cfg.get("bootstrap", {})
    pipeline_start = time.monotonic()
    run_times: list[float] = []

    for model_idx, (model_name, (model_cls, kwargs)) in enumerate(model_configs.items(), 1):
        model_start = time.monotonic()
        logger.info(f"--- [{model_idx}/{n_models}] {model_name} ---")

        results, ci_results, was_cached = run_single_model(
            model_name=model_name,
            model_cls=model_cls,
            model_kwargs=kwargs,
            trajectories=trajectories,
            output_dir=output_dir,
            bootstrap_n=bootstrap_cfg.get("n_samples", 2000),
            bootstrap_seed=bootstrap_cfg.get("seed", 42),
            force=args.force,
            cfg=cfg,
        )

        if results is not None:
            lopo_results[model_name] = results
            if ci_results:
                all_bootstrap_cis[model_name] = ci_results

        elapsed = time.monotonic() - model_start
        if not was_cached:
            run_times.append(elapsed)

        remaining = n_models - model_idx
        if run_times and remaining > 0:
            avg = sum(run_times) / len(run_times)
            eta_s = avg * remaining
            eta_str = f"{eta_s / 60:.1f}min" if eta_s > 60 else f"{eta_s:.0f}s"
            logger.info(f"  ETA for {remaining} remaining: {eta_str}")

    total_elapsed = time.monotonic() - pipeline_start
    n_cached = len(model_configs) - len(run_times)
    logger.info(
        f"=== LOPO-CV complete: {len(run_times)} ran, {n_cached} cached, "
        f"total {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min) ==="
    )

    # =========================================================================
    # Step 3: Post-hoc analysis (calibration, comparisons, plots)
    # =========================================================================
    _run_posthoc_analysis(cfg, output_dir, model_configs, lopo_results)

    # =========================================================================
    # Step 4: Save experiment metadata
    # =========================================================================
    logger.info("=== Step 4: Save Metadata ===")
    save_experiment_metadata(
        output_dir=output_dir,
        trajectories=trajectories,
        config=cfg,
        stage_name="stage1_volumetric_uq",
        all_model_results=lopo_results,
        all_bootstrap_cis=all_bootstrap_cis,
    )

    logger.info(f"Results saved to: {output_dir}")


def _run_posthoc_analysis(
    cfg: dict,
    output_dir: Path,
    model_configs: dict[str, tuple[type, dict]],
    lopo_results: dict[str, LOPOResults],
) -> None:
    """Run calibration metrics, paired comparisons, plots, and summary.

    Shared between run_all.py and run_analysis.py.
    """
    # --- Calibration metrics ---
    logger.info("=== Step 3a: Calibration Metrics ===")
    calib_metrics: dict[str, dict] = {}
    for model_name, results in lopo_results.items():
        calib = compute_calibration_metrics(results)
        if calib:
            calib_metrics[model_name] = calib
            logger.info(
                f"  {model_name}: DSS={calib['dss']:.4f}, "
                f"NLPD={calib['nlpd']:.4f}, PIT KS p={calib['pit_ks_p']:.4f}"
            )

    # --- Homo vs hetero comparisons ---
    logger.info("=== Step 3b: Paired Comparisons ===")
    vd_cfg = cfg.get("variance_decomposition", {})
    homo_hetero: list[dict] = []
    if vd_cfg.get("enabled", True):
        homo_hetero = run_paired_comparisons(
            lopo_results,
            vd_cfg.get("pairs", []),
            n_permutations=vd_cfg.get("n_permutations", 10000),
            seed=cfg["experiment"]["seed"],
        )
        write_comparison_table(homo_hetero, output_dir)

    # --- Classical vs propagated ---
    reporting_cfg = cfg.get("reporting", {})
    cvp_cfg = reporting_cfg.get("classical_vs_propagated", {})
    classical_propagated: list[dict] = []
    if cvp_cfg.get("pairs"):
        classical_propagated = run_paired_comparisons(
            lopo_results,
            cvp_cfg["pairs"],
            n_permutations=cvp_cfg.get("n_permutations", 10000),
            seed=cfg["experiment"]["seed"],
        )
        write_comparison_table(
            classical_propagated,
            output_dir,
            filename_prefix="comparison_classical_vs_propagated",
            title="Classical (NLME) vs Propagated (Heteroscedastic) Comparison",
        )

    # --- Analytical vs homo ---
    avh_cfg = reporting_cfg.get("analytical_vs_homo", {})
    analytical_homo: list[dict] = []
    if avh_cfg.get("pairs"):
        analytical_homo = run_paired_comparisons(
            lopo_results,
            avh_cfg["pairs"],
            n_permutations=avh_cfg.get("n_permutations", 10000),
            seed=cfg["experiment"]["seed"],
        )
        write_comparison_table(
            analytical_homo,
            output_dir,
            filename_prefix="comparison_analytical_vs_homo",
            title="Analytical (NLME) vs Homoscedastic Comparison",
        )

    # --- Figures ---
    if reporting_cfg.get("pit_histogram", False) and calib_metrics:
        generate_pit_histograms(calib_metrics, output_dir)

    if reporting_cfg.get("sharpness_scatter", False) and lopo_results:
        generate_sharpness_scatter(lopo_results, output_dir)

    # --- Summary ---
    model_names = list(model_configs.keys()) if model_configs else list(lopo_results.keys())
    print_summary_table(model_names, lopo_results, calib_metrics)
    print_comparison_tables(homo_hetero, classical_propagated, analytical_homo)
    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
