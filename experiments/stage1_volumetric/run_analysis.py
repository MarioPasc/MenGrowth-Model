# experiments/stage1_volumetric/run_analysis.py
"""Post-hoc analysis on existing LOPO-CV results.

Loads cached results from disk (no model fitting), then computes
calibration metrics, paired comparisons, generates figures, and prints
summary tables. Designed for use after SLURM parallel model runs.

Usage::

    cd /home/mpascual/research/code/MenGrowth-Model
    ~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_analysis \\
        --config experiments/stage1_volumetric/configs/config_uq.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from experiments.stage1_volumetric.engine.data import load_config
from experiments.stage1_volumetric.engine.model_registry import build_model_configs
from experiments.stage1_volumetric.run_all import _run_posthoc_analysis
from growth.shared.lopo import LOPOResults

logger = logging.getLogger(__name__)


def main() -> None:
    """Run post-hoc analysis on existing results."""
    parser = argparse.ArgumentParser(description="Post-hoc analysis on existing LOPO-CV results")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])
    model_configs = build_model_configs(cfg)

    # Scan output_dir for model subdirectories with lopo_results.json
    logger.info(f"Scanning {output_dir} for cached results...")
    lopo_results: dict[str, LOPOResults] = {}

    for model_name in model_configs:
        cached_path = output_dir / model_name / "lopo_results.json"
        if cached_path.exists():
            try:
                with open(cached_path) as f:
                    data = json.load(f)
                lopo_results[model_name] = LOPOResults.from_dict(data)
                logger.info(f"  Loaded {model_name}")
            except Exception as e:
                logger.warning(f"  Failed to load {model_name}: {e}")
        else:
            logger.warning(f"  Missing: {model_name}")

    if not lopo_results:
        logger.error("No cached results found. Run models first.")
        sys.exit(1)

    n_expected = len(model_configs)
    n_found = len(lopo_results)
    logger.info(f"Loaded {n_found}/{n_expected} models")

    if n_found < n_expected:
        missing = set(model_configs.keys()) - set(lopo_results.keys())
        logger.warning(f"Missing models: {missing}")

    _run_posthoc_analysis(cfg, output_dir, model_configs, lopo_results)
    logger.info("Post-hoc analysis complete")


if __name__ == "__main__":
    main()
