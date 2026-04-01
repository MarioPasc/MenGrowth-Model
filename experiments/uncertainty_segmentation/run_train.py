#!/usr/bin/env python
# experiments/uncertainty_segmentation/run_train.py
"""CLI entry point: train a single LoRA ensemble member.

Usage:
    python -m experiments.uncertainty_segmentation.run_train \
        --config experiments/uncertainty_segmentation/config.yaml \
        --member-id 0 \
        --device cuda:0

Each member is trained independently with seed = base_seed + member_id.
For SLURM array jobs, member_id maps to SLURM_ARRAY_TASK_ID.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .engine.train_member import train_single_member

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Parse arguments and train one ensemble member."""
    parser = argparse.ArgumentParser(
        description="Train a single LoRA ensemble member for uncertainty segmentation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML (or base + picasso override).",
    )
    parser.add_argument(
        "--config-override",
        type=str,
        default=None,
        help="Optional override YAML (e.g., picasso config). Merged on top of --config.",
    )
    parser.add_argument(
        "--member-id",
        type=int,
        required=True,
        help="Ensemble member index (0-based). Maps to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (default: cuda).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override run directory (from SLURM). If not set, derived from config.",
    )
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    if args.config_override is not None:
        override = OmegaConf.load(args.config_override)
        config = OmegaConf.merge(config, override)

    # Validate member_id
    n_members = config.ensemble.n_members
    if not 0 <= args.member_id < n_members:
        logger.error(
            f"member-id {args.member_id} out of range [0, {n_members})"
        )
        return 1

    # Validate paths
    checkpoint_path = Path(config.paths.checkpoint_dir) / config.paths.checkpoint_filename
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    h5_path = Path(config.paths.men_h5_file)
    if not h5_path.exists():
        logger.error(f"H5 file not found: {h5_path}")
        return 1

    # Device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Training member {args.member_id}/{n_members} on {device}")

    # Train
    metrics = train_single_member(config, args.member_id, device, run_dir=args.run_dir)

    logger.info(f"Training complete. Best metrics: {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
