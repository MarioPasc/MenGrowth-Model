# experiments/stage2_severity/run_severity.py
"""Stage 2: Latent Severity Model experiment orchestrator.

Runs the severity model under LOPO-CV, compares against Stage 1 baseline,
and produces diagnostic outputs.

Usage::

    ~/.conda/envs/growth/bin/python -m experiments.stage2_severity.run_severity \\
        --config experiments/stage2_severity/config.yaml \\
        --trajectories /path/to/trajectories.json \\
        --output-dir /path/to/results
"""

import argparse
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from growth.shared import LOPOEvaluator, load_trajectories
from growth.stages.stage2_severity import SeverityModel

logger = logging.getLogger(__name__)


def main() -> None:
    """Run Stage 2 severity model experiment."""
    parser = argparse.ArgumentParser(description="Stage 2: Latent Severity Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--trajectories", type=str, required=True, help="Path to trajectories JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    cfg = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectories
    patients = load_trajectories(args.trajectories)
    logger.info(f"Loaded {len(patients)} patients")

    # Configure model
    severity_cfg = cfg.severity
    model_kwargs = {
        "growth_function": severity_cfg.growth_function,
        "lambda_reg": severity_cfg.optimization.lambda_reg,
        "n_restarts": severity_cfg.optimization.n_restarts,
        "max_iter": severity_cfg.optimization.max_iter,
        "seed": cfg.seed,
    }

    # Run LOPO-CV
    evaluator = LOPOEvaluator(
        prediction_protocols=list(cfg.evaluation.prediction_protocols),
    )
    results = evaluator.evaluate(SeverityModel, patients, **model_kwargs)

    # Save results
    results_path = output_dir / "severity_lopo_results.json"
    with open(results_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Aggregate metrics: {results.aggregate_metrics}")


if __name__ == "__main__":
    main()
