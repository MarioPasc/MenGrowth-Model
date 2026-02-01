#!/usr/bin/env python
# experiments/lora_ablation/extract_features.py
"""Extract 768-dim encoder features for all subjects in probe_train + test splits.

For each trained condition, this script:
1. Loads the encoder (with merged LoRA weights if applicable)
2. Extracts features from probe_train and test subjects
3. Also extracts semantic targets from segmentation masks
4. Saves features and targets as .pt files

Usage:
    python -m experiments.lora_ablation.extract_features \
        --config experiments/lora_ablation/config/ablation.yaml \
        --condition lora_r8
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from growth.data.bratsmendata import BraTSMENDataset
from growth.data.transforms import get_val_transforms
from growth.models.encoder.feature_extractor import FeatureExtractor
from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_swin_encoder
from growth.utils.seed import set_seed

from .data_splits import load_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_encoder_for_condition(
    condition_name: str,
    config: dict,
    device: str,
) -> torch.nn.Module:
    """Load encoder for a trained condition.

    For baseline: loads the checkpoint directly
    For LoRA: loads base encoder, loads LoRA adapter, and merges weights

    Returns:
        Encoder model with merged weights (if LoRA) in eval mode.
    """
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    # Check condition type
    condition_config = None
    for cond in config["conditions"]:
        if cond["name"] == condition_name:
            condition_config = cond
            break

    if condition_config is None:
        raise ValueError(f"Unknown condition: {condition_name}")

    is_baseline = condition_config.get("lora_rank") is None

    # Load base encoder
    base_encoder = load_swin_encoder(
        config["paths"]["checkpoint"],
        freeze=True,
        device=device,
    )

    if is_baseline:
        logger.info(f"Loaded baseline encoder (no LoRA)")
        encoder = base_encoder
    else:
        # Load LoRA adapter and merge
        adapter_path = condition_dir / "adapter"
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_path}. "
                "Train the condition first."
            )

        logger.info(f"Loading LoRA adapter from {adapter_path}")

        # Create LoRA model and load adapter
        # We need to recreate the LoRA wrapper with the right rank
        lora_encoder = LoRASwinViT.load_lora(
            base_encoder,
            adapter_path,
            device=device,
            trainable=False,
        )

        # Merge LoRA weights into base model for efficient inference
        encoder = lora_encoder.merge_lora()
        logger.info("Merged LoRA weights into base encoder")

    encoder.eval()
    return encoder


@torch.no_grad()
def extract_features_for_split(
    encoder: torch.nn.Module,
    subject_ids: List[str],
    data_root: str,
    device: str,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Extract features and semantic targets for a split.

    Args:
        encoder: SwinUNETR encoder.
        subject_ids: List of subject IDs to process.
        data_root: Path to BraTS-MEN data.
        device: Device to use.
        batch_size: Batch size for inference.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of:
        - features: [N, 768] numpy array
        - targets: dict with 'volume', 'location', 'shape', 'all' arrays
        - ordered_ids: List of subject IDs in order of features
    """
    # Create dataset with semantic features
    dataset = BraTSMENDataset(
        data_root=data_root,
        subject_ids=subject_ids,
        transform=get_val_transforms(),
        compute_semantic=True,
        cache_semantic=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create feature extractor
    feature_extractor = FeatureExtractor(encoder, level="encoder10")

    # Extract features
    all_features = []
    all_volumes = []
    all_locations = []
    all_shapes = []
    all_ids = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        images = batch["image"].to(device)

        # Extract features using global average pooling
        features = feature_extractor(images)
        all_features.append(features.cpu().numpy())

        # Collect semantic targets
        all_volumes.append(batch["semantic_features"]["volume"].numpy())
        all_locations.append(batch["semantic_features"]["location"].numpy())
        all_shapes.append(batch["semantic_features"]["shape"].numpy())

        # Track subject IDs (handle both list and tuple batch results)
        ids = batch["subject_id"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        all_ids.extend(ids)

    # Concatenate all batches
    features = np.concatenate(all_features, axis=0)
    targets = {
        "volume": np.concatenate(all_volumes, axis=0),
        "location": np.concatenate(all_locations, axis=0),
        "shape": np.concatenate(all_shapes, axis=0),
    }
    # Concatenate all targets for convenience
    targets["all"] = np.concatenate([
        targets["volume"],
        targets["location"],
        targets["shape"],
    ], axis=1)

    logger.info(f"Extracted features shape: {features.shape}")
    logger.info(f"Targets shapes: volume={targets['volume'].shape}, "
                f"location={targets['location'].shape}, shape={targets['shape'].shape}")

    return features, targets, all_ids


def save_features(
    features: np.ndarray,
    targets: Dict[str, np.ndarray],
    subject_ids: List[str],
    save_dir: Path,
    split_name: str,
) -> None:
    """Save features and targets to files.

    Saves:
    - features_{split_name}.pt
    - targets_{split_name}.pt
    - subject_ids_{split_name}.txt
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    features_path = save_dir / f"features_{split_name}.pt"
    torch.save(torch.from_numpy(features), features_path)
    logger.info(f"Saved features to {features_path}")

    # Save targets
    targets_path = save_dir / f"targets_{split_name}.pt"
    torch.save({k: torch.from_numpy(v) for k, v in targets.items()}, targets_path)
    logger.info(f"Saved targets to {targets_path}")

    # Save subject IDs for reference
    ids_path = save_dir / f"subject_ids_{split_name}.txt"
    with open(ids_path, "w") as f:
        f.write("\n".join(subject_ids))
    logger.info(f"Saved subject IDs to {ids_path}")


def extract_features(
    condition_name: str,
    config: dict,
    splits: dict,
    device: str = "cuda",
) -> Dict[str, Path]:
    """Extract features for probe_train and test splits.

    Args:
        condition_name: Name of the trained condition.
        config: Full experiment configuration.
        splits: Data splits dictionary.
        device: Device to use.

    Returns:
        Dict mapping split names to feature file paths.
    """
    logger.info(f"Extracting features for condition: {condition_name}")

    # Set up output directory
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    # Load encoder
    encoder = load_encoder_for_condition(condition_name, config, device)

    # Feature extraction config
    fe_config = config.get("feature_extraction", {})
    batch_size = fe_config.get("batch_size", 8)
    num_workers = config["training"].get("num_workers", 4)

    # Extract features for probe_train split
    logger.info(f"\nExtracting probe_train features ({len(splits['probe_train'])} subjects)")
    probe_features, probe_targets, probe_ids = extract_features_for_split(
        encoder=encoder,
        subject_ids=splits["probe_train"],
        data_root=config["paths"]["data_root"],
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    save_features(probe_features, probe_targets, probe_ids, condition_dir, "probe")

    # Extract features for test split
    logger.info(f"\nExtracting test features ({len(splits['test'])} subjects)")
    test_features, test_targets, test_ids = extract_features_for_split(
        encoder=encoder,
        subject_ids=splits["test"],
        data_root=config["paths"]["data_root"],
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    save_features(test_features, test_targets, test_ids, condition_dir, "test")

    # Return paths
    return {
        "probe_features": condition_dir / "features_probe.pt",
        "probe_targets": condition_dir / "targets_probe.pt",
        "test_features": condition_dir / "features_test.pt",
        "test_targets": condition_dir / "targets_test.pt",
    }


def main(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Main entry point for feature extraction.

    Args:
        config_path: Path to ablation.yaml.
        condition: Condition name.
        device: Device to use.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Load splits
    splits = load_splits(config_path)

    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Extract features
    extract_features(
        condition_name=condition,
        config=config,
        splits=splits,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features for a trained condition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["baseline", "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32"],
        help="Condition to extract features for",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
