#!/usr/bin/env python
# experiments/sdp/extract_all_features.py
"""Extract encoder features + semantic targets for all subjects.

Saves per-split HDF5 files: {features_dir}/{split_name}.h5
Each file contains:
    features/encoder10  [N, 768]
    targets/volume      [N, 4]
    targets/location    [N, 3]
    targets/shape       [N, 3]
    subject_ids         [N] (string dataset)

Usage:
    python -m experiments.sdp.extract_all_features \
        --config experiments/sdp/config/sdp_default.yaml
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from experiments.lora_ablation.data_splits import load_splits
from experiments.lora_ablation.extract_features import (
    extract_features_for_split,
    load_encoder_for_condition,
)
from src.growth.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_features_h5(
    features_dict: dict[str, np.ndarray],
    targets_dict: dict[str, np.ndarray],
    subject_ids: list[str],
    save_path: Path,
) -> None:
    """Save features and targets to a single HDF5 file.

    Args:
        features_dict: Dict of feature arrays {level: [N, D]}.
        targets_dict: Dict of target arrays {name: [N, K]}.
        subject_ids: List of subject ID strings.
        save_path: Output .h5 file path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, "w") as f:
        # Features group
        feat_grp = f.create_group("features")
        for key, arr in features_dict.items():
            feat_grp.create_dataset(key, data=arr, compression="gzip")

        # Targets group
        tgt_grp = f.create_group("targets")
        for key, arr in targets_dict.items():
            tgt_grp.create_dataset(key, data=arr, compression="gzip")

        # Subject IDs (variable-length strings)
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("subject_ids", data=subject_ids, dtype=dt)

    logger.info(f"Saved {len(subject_ids)} subjects to {save_path}")


def main(config_path: str, device: str = "cuda") -> None:
    """Extract features for all splits and save as HDF5.

    Args:
        config_path: Path to SDP experiment config.
        device: Device for inference.
    """
    cfg = OmegaConf.load(config_path)

    set_seed(cfg.training.seed)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Load data splits from the LoRA ablation config
    splits = load_splits(cfg.data.splits_config)

    # Load the LoRA-adapted encoder config for loading the model
    with open(cfg.data.splits_config) as f:
        lora_config = yaml.safe_load(f)

    # Load merged LoRA encoder
    encoder = load_encoder_for_condition("lora_r8", lora_config, device)

    # Feature extraction config
    fe_cfg = cfg.get("feature_extraction", {})
    batch_size = fe_cfg.get("batch_size", 2)
    num_workers = fe_cfg.get("num_workers", 4)

    features_dir = Path(cfg.paths.features_dir)

    # Extract for each split
    all_splits = list(splits.keys())
    logger.info(f"Extracting features for splits: {all_splits}")

    for split_name in all_splits:
        subject_ids = splits[split_name]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Split: {split_name} ({len(subject_ids)} subjects)")
        logger.info(f"{'=' * 60}")

        features_dict, targets_dict, extracted_ids = extract_features_for_split(
            encoder=encoder,
            subject_ids=subject_ids,
            data_root=cfg.paths.data_root,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            feature_level="encoder10",
        )

        # Save as HDF5
        h5_path = features_dir / f"{split_name}.h5"
        save_features_h5(features_dict, targets_dict, extracted_ids, h5_path)

    logger.info(f"\nAll features saved to {features_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for SDP training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.device)
