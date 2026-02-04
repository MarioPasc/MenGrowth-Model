#!/usr/bin/env python
# experiments/lora_ablation/data_splits.py
"""Generate and save fixed train/val/test splits for LoRA ablation.

Creates four non-overlapping splits:
- lora_train: For training LoRA/baseline segmentation
- lora_val: For early stopping during LoRA training
- probe_train: For training linear probes (separate from LoRA training)
- test: For final evaluation (never touched during training)

Usage:
    python -m experiments.lora_ablation.data_splits --config experiments/lora_ablation/config/ablation.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from growth.data.bratsmendata import BraTSMENDataset, save_splits, split_subjects_multi
from growth.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_subjects_four_way(
    subject_ids: List[str],
    lora_train_size: int,
    lora_val_size: int,
    probe_train_size: int,
    test_size: int,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split subjects into four non-overlapping sets.

    This is a convenience wrapper around split_subjects_multi() for the
    four-way split pattern used in LoRA ablation experiments.

    Args:
        subject_ids: List of all subject IDs.
        lora_train_size: Number of subjects for LoRA training.
        lora_val_size: Number of subjects for LoRA validation.
        probe_train_size: Number of subjects for probe training.
        test_size: Number of subjects for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'lora_train', 'lora_val', 'probe_train', 'test'.

    Raises:
        ValueError: If total size exceeds available subjects.
    """
    return split_subjects_multi(
        subject_ids=subject_ids,
        split_sizes={
            "lora_train": lora_train_size,
            "lora_val": lora_val_size,
            "probe_train": probe_train_size,
            "test": test_size,
        },
        seed=seed,
    )


def main(config_path: str, force: bool = False) -> Dict[str, List[str]]:
    """Generate and save data splits.

    Args:
        config_path: Path to ablation.yaml configuration.
        force: If True, overwrite existing splits file.

    Returns:
        The generated splits dictionary.
    """
    # Load configuration
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    seed = config["experiment"]["seed"]
    set_seed(seed)

    # Set up output directory
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_path = output_dir / "data_splits.json"

    # Check if splits already exist
    if splits_path.exists() and not force:
        logger.info(f"Splits already exist at {splits_path}. Use --force to regenerate.")
        with open(splits_path) as f:
            splits = json.load(f)
        _print_split_statistics(splits)
        return splits

    # Discover all subject IDs
    data_root = Path(config["paths"]["data_root"])
    logger.info(f"Discovering subjects in {data_root}")

    all_subjects = BraTSMENDataset.get_all_subject_ids(data_root)
    logger.info(f"Found {len(all_subjects)} subjects")

    # Generate splits
    split_config = config["data_splits"]
    splits = split_subjects_four_way(
        subject_ids=all_subjects,
        lora_train_size=split_config["lora_train"],
        lora_val_size=split_config["lora_val"],
        probe_train_size=split_config["probe_train"],
        test_size=split_config["test"],
        seed=seed,
    )

    # Save splits
    save_splits(splits, splits_path)
    logger.info(f"Saved splits to {splits_path}")

    # Also save a copy of the config for reference
    config_copy_path = output_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config copy to {config_copy_path}")

    # Print statistics
    _print_split_statistics(splits)

    return splits


def _print_split_statistics(splits: Dict[str, List[str]]) -> None:
    """Print split statistics."""
    print("\n" + "=" * 50)
    print("Data Split Statistics")
    print("=" * 50)

    total = sum(len(v) for v in splits.values())
    print(f"\nTotal subjects: {total}")
    print("-" * 30)

    for split_name, subjects in splits.items():
        pct = 100 * len(subjects) / total
        print(f"  {split_name:15s}: {len(subjects):4d} ({pct:5.1f}%)")

    print("-" * 30)

    # Show first few subjects from each split for verification
    print("\nFirst 3 subjects per split:")
    for split_name, subjects in splits.items():
        print(f"  {split_name}: {subjects[:3]}")

    print("=" * 50 + "\n")


def load_splits(config_path: str) -> Dict[str, List[str]]:
    """Load existing splits from config's output directory.

    Args:
        config_path: Path to ablation.yaml configuration.

    Returns:
        Loaded splits dictionary.

    Raises:
        FileNotFoundError: If splits file doesn't exist.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    splits_path = output_dir / "data_splits.json"

    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits not found at {splits_path}. "
            "Run data_splits.py first to generate them."
        )

    with open(splits_path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data splits for LoRA ablation experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of splits even if they exist",
    )

    args = parser.parse_args()
    main(args.config, args.force)
