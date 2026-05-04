# experiments/stage1_volumetric/run_single_model.py
"""Run LOPO-CV for a single growth model.

Designed for model-level parallelism: each SLURM worker calls this
script with a different --model argument. Results are saved to
output_dir/model_name/ and can be aggregated post-hoc by run_analysis.py.

Usage::

    cd /home/mpascual/research/code/MenGrowth-Model

    # Single model (local):
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_single_model \\
        --model LME --config experiments/stage1_volumetric/configs/config_uq.yaml

    # Force re-run (ignore cache):
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_single_model \\
        --model NLME_Exponential --config experiments/stage1_volumetric/configs/config_uq.yaml --force

    # List available models:
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_single_model \\
        --list-models --config experiments/stage1_volumetric/configs/config_uq.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from experiments.stage1_volumetric.engine.data import load_config, load_trajectories
from experiments.stage1_volumetric.engine.model_registry import build_model_configs
from experiments.stage1_volumetric.engine.runner import run_single_model


def main() -> None:
    """Run LOPO-CV for one model."""
    parser = argparse.ArgumentParser(description="Run LOPO-CV for a single growth model")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g. LME, NLME_Exponential, HGPHetero)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if cached results exist",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names and exit",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)

    if args.real_time:
        cfg["time"]["variable"] = "days_from_baseline"
    if args.estimator:
        cfg["uncertainty"]["estimator"] = args.estimator

    all_models = build_model_configs(cfg)

    if args.list_models:
        print("Available models:")
        for name in all_models:
            print(f"  {name}")
        sys.exit(0)

    if args.model is None:
        parser.error("--model is required (use --list-models to see options)")

    if args.model not in all_models:
        logger.error(f"Unknown model: {args.model}. Available: {list(all_models.keys())}")
        sys.exit(1)

    output_dir = Path(cfg["paths"]["output_dir"])
    bootstrap_cfg = cfg.get("bootstrap", {})

    logger.info(f"Loading trajectories for model {args.model}")
    trajectories = load_trajectories(cfg)

    model_cls, model_kwargs = all_models[args.model]
    logger.info(f"Running LOPO-CV for {args.model}")

    results, ci_results, was_cached = run_single_model(
        model_name=args.model,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        trajectories=trajectories,
        output_dir=output_dir,
        bootstrap_n=bootstrap_cfg.get("n_samples", 2000),
        bootstrap_seed=bootstrap_cfg.get("seed", 42),
        force=args.force,
        cfg=cfg,
    )

    if results is None:
        logger.error(f"Model {args.model} failed")
        sys.exit(1)

    status = "cached" if was_cached else "completed"
    r2 = results.aggregate_metrics.get("last_from_rest/r2_log", float("nan"))
    logger.info(f"Model {args.model} {status}: R2={r2:.4f}")


if __name__ == "__main__":
    main()
