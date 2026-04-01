#!/usr/bin/env python
# experiments/uncertainty_segmentation/run_inference.py
"""CLI entry point: ensemble inference on MenGrowth longitudinal cohort.

Runs M LoRA adapters on each scan, computes volume ± uncertainty, and saves
a CSV for downstream GP/LME growth models.

Usage:
    python -m experiments.uncertainty_segmentation.run_inference \
        --config experiments/uncertainty_segmentation/config.yaml \
        --device cuda:0
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .engine.ensemble_inference import EnsemblePredictor
from .engine.paths import get_run_dir
from .engine.volume_extraction import extract_ensemble_volumes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Run ensemble inference on MenGrowth cohort and save volume CSV."""
    parser = argparse.ArgumentParser(
        description="Ensemble inference on MenGrowth longitudinal cohort."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config-override", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mengrowth",
        choices=["mengrowth", "men"],
        help="Which dataset to run inference on (default: mengrowth).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Restrict to a specific split (e.g., 'test'). Default: all scans.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override run directory (from SLURM). If not set, derived from config.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.config_override:
        config = OmegaConf.merge(config, OmegaConf.load(args.config_override))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Select H5 file
    if args.dataset == "mengrowth":
        h5_path = Path(config.paths.mengrowth_h5_file)
        output_name = "mengrowth_ensemble_volumes.csv"
    else:
        h5_path = Path(config.paths.men_h5_file)
        output_name = "men_ensemble_volumes.csv"

    if not h5_path.exists():
        logger.error(f"H5 file not found: {h5_path}")
        return 1

    # Resolve run directory (must be before any path that depends on it)
    resolved_run_dir = get_run_dir(config, override=args.run_dir)

    # Create predictor
    predictor = EnsemblePredictor(config, device=device, run_dir=args.run_dir)

    if len(predictor.available_members) == 0:
        logger.error("No trained ensemble members found. Train first.")
        return 1

    # Predictions directory (for NIfTI masks and sample predictions)
    predictions_dir = resolved_run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Extract volumes
    df = extract_ensemble_volumes(
        predictor=predictor,
        h5_path=h5_path,
        config=config,
        split=args.split,
        predictions_dir=predictions_dir,
    )

    # Save CSV
    output_dir = resolved_run_dir / "volumes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    df.to_csv(output_path, index=False)
    logger.info(f"Volume CSV saved to {output_path} ({len(df)} rows)")

    # Summary statistics
    if len(df) > 0:
        logger.info(
            f"Volume summary:\n"
            f"  Mean volume: {df['vol_mean'].mean():.0f} ± {df['vol_std'].mean():.0f} mm³\n"
            f"  Log-volume: {df['logvol_mean'].mean():.3f} ± {df['logvol_std'].mean():.4f}\n"
            f"  N patients: {df['patient_id'].nunique()}\n"
            f"  N scans: {len(df)}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
