#!/usr/bin/env python
# experiments/lora_ablation/extract_features.py
"""Feature extraction with multi-scale support.

Supports both feature levels via config:
- "encoder10": Single-scale 768-dim features (v1 behavior)
- "multi_scale": Layers 2+3+4 concatenated (192+384+768=1344-dim, recommended)

Usage:
    python -m experiments.lora_ablation.extract_features \
        --config experiments/lora_ablation/config/ablation.yaml \
        --condition lora_r8
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from growth.data.bratsmendata import BraTSMENDataset
from growth.data.transforms import get_val_transforms
from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_swin_encoder
from growth.utils.seed import set_seed

from .data_splits import load_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiScaleFeatureExtractor:
    """Extract features at multiple scales from SwinUNETR.

    Extracts GAP features from:
    - layers2: 192-dim
    - layers3: 384-dim
    - layers4: 768-dim
    - encoder10: 768-dim (bottleneck)

    The multi-scale representation (1344-dim) captures both
    local and global information.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        device: str = "cuda",
    ):
        self.encoder = encoder
        self.device = device
        self.encoder.eval()

    @torch.no_grad()
    def extract(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor [B, 4, 96, 96, 96].

        Returns:
            Dict with 'layers2', 'layers3', 'layers4', 'encoder10', 'multi_scale'.
        """
        # Get hidden states from swinViT
        hidden_states = self.encoder.swinViT(x, self.encoder.normalize)

        features = {}

        # Stage outputs with GAP
        # hidden_states[2]: layers2 output [B, 192, 12, 12, 12]
        features['layers2'] = F.adaptive_avg_pool3d(
            hidden_states[2], 1
        ).flatten(1)  # [B, 192]

        # hidden_states[3]: layers3 output [B, 384, 6, 6, 6]
        features['layers3'] = F.adaptive_avg_pool3d(
            hidden_states[3], 1
        ).flatten(1)  # [B, 384]

        # hidden_states[4]: layers4 output [B, 768, 3, 3, 3]
        features['layers4'] = F.adaptive_avg_pool3d(
            hidden_states[4], 1
        ).flatten(1)  # [B, 768]

        # encoder10: bottleneck [B, 768, 3, 3, 3]
        enc10 = self.encoder.encoder10(hidden_states[4])
        features['encoder10'] = F.adaptive_avg_pool3d(
            enc10, 1
        ).flatten(1)  # [B, 768]

        # Multi-scale concatenation
        features['multi_scale'] = torch.cat([
            features['layers2'],
            features['layers3'],
            features['layers4'],
        ], dim=1)  # [B, 1344]

        return features


def load_encoder_for_condition(
    condition_name: str,
    config: dict,
    device: str,
) -> torch.nn.Module:
    """Load encoder for a condition with proper weight merging.

    Returns:
        SwinUNETR encoder in eval mode.
    """
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    # Get condition config
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
        logger.info("Loaded baseline encoder (no LoRA)")
        encoder = base_encoder
    else:
        # Load LoRA adapter and merge
        adapter_path = condition_dir / "adapter"
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        logger.info(f"Loading LoRA adapter from {adapter_path}")

        # Load LoRA model
        lora_encoder = LoRASwinViT.load_lora(
            base_encoder,
            adapter_path,
            device=device,
            trainable=False,
        )

        # Merge LoRA weights for efficient inference
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
    feature_level: str = "multi_scale",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Extract features and semantic targets for a split.

    Args:
        encoder: SwinUNETR encoder.
        subject_ids: Subject IDs to process.
        data_root: Path to data.
        device: Device to use.
        batch_size: Batch size.
        num_workers: Data loading workers.
        feature_level: 'encoder10', 'multi_scale', or 'all'.

    Returns:
        Tuple of (features_dict, targets_dict, subject_ids).
    """
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

    extractor = MultiScaleFeatureExtractor(encoder, device)

    # Storage
    all_features = {
        'encoder10': [],
        'multi_scale': [],
        'layers2': [],
        'layers3': [],
        'layers4': [],
    }
    all_volumes = []
    all_locations = []
    all_shapes = []
    all_ids = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        images = batch["image"].to(device)

        # Extract multi-scale features
        features = extractor.extract(images)

        for key in all_features:
            all_features[key].append(features[key].cpu().numpy())

        # Semantic targets
        all_volumes.append(batch["semantic_features"]["volume"].numpy())
        all_locations.append(batch["semantic_features"]["location"].numpy())
        all_shapes.append(batch["semantic_features"]["shape"].numpy())

        # Subject IDs
        ids = batch["subject_id"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        all_ids.extend(ids)

    # Concatenate
    features_dict = {
        key: np.concatenate(vals, axis=0)
        for key, vals in all_features.items()
    }

    targets_dict = {
        "volume": np.concatenate(all_volumes, axis=0),
        "location": np.concatenate(all_locations, axis=0),
        "shape": np.concatenate(all_shapes, axis=0),
    }

    # Log shapes
    logger.info("Feature shapes:")
    for key, arr in features_dict.items():
        logger.info(f"  {key}: {arr.shape}")
    logger.info("Target shapes:")
    for key, arr in targets_dict.items():
        logger.info(f"  {key}: {arr.shape}")

    return features_dict, targets_dict, all_ids


def save_features(
    features_dict: Dict[str, np.ndarray],
    targets_dict: Dict[str, np.ndarray],
    subject_ids: List[str],
    save_dir: Path,
    split_name: str,
) -> None:
    """Save features and targets."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save each feature type
    for key, arr in features_dict.items():
        path = save_dir / f"features_{split_name}_{key}.pt"
        torch.save(torch.from_numpy(arr), path)
        logger.info(f"Saved {key} features to {path}")

    # Save primary features (for backwards compatibility)
    torch.save(
        torch.from_numpy(features_dict['encoder10']),
        save_dir / f"features_{split_name}.pt"
    )

    # Save targets
    targets_path = save_dir / f"targets_{split_name}.pt"
    torch.save({k: torch.from_numpy(v) for k, v in targets_dict.items()}, targets_path)
    logger.info(f"Saved targets to {targets_path}")

    # Save subject IDs
    ids_path = save_dir / f"subject_ids_{split_name}.txt"
    with open(ids_path, "w") as f:
        f.write("\n".join(subject_ids))


def extract_features(
    condition_name: str,
    config: dict,
    splits: dict,
    device: str = "cuda",
) -> Dict[str, Path]:
    """Extract features for a condition."""
    logger.info(f"Extracting features for condition: {condition_name}")

    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    # Load encoder
    encoder = load_encoder_for_condition(condition_name, config, device)

    # Config
    fe_config = config.get("feature_extraction", {})
    batch_size = fe_config.get("batch_size", 8)
    num_workers = config["training"].get("num_workers", 4)
    feature_level = fe_config.get("level", "multi_scale")

    # Extract for sdp_train split (used for SDP projection and probe training)
    sdp_split_key = "sdp_train" if "sdp_train" in splits else "probe_train"
    logger.info(f"\nExtracting sdp_train features ({len(splits[sdp_split_key])} subjects)")
    probe_features, probe_targets, probe_ids = extract_features_for_split(
        encoder=encoder,
        subject_ids=splits[sdp_split_key],
        data_root=config["paths"]["data_root"],
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_level=feature_level,
    )
    save_features(probe_features, probe_targets, probe_ids, condition_dir, "probe")

    # Extract for test split
    logger.info(f"\nExtracting test features ({len(splits['test'])} subjects)")
    test_features, test_targets, test_ids = extract_features_for_split(
        encoder=encoder,
        subject_ids=splits["test"],
        data_root=config["paths"]["data_root"],
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_level=feature_level,
    )
    save_features(test_features, test_targets, test_ids, condition_dir, "test")

    return {
        "probe_features": condition_dir / "features_probe.pt",
        "test_features": condition_dir / "features_test.pt",
    }


def main(config_path: str, condition: str, device: str = "cuda") -> None:
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])
    splits = load_splits(config_path)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    extract_features(
        condition_name=condition,
        config=config,
        splits=splits,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract multi-scale features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
