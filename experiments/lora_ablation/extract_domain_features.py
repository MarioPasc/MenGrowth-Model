#!/usr/bin/env python
# experiments/lora_ablation/extract_domain_features.py
"""Extract features for domain shift visualization (Glioma vs Meningioma).

This script extracts a subset of features from:
1. BraTS-GLI (Glioma) dataset - source domain
2. BraTS-MEN (Meningioma) dataset - target domain

The features are used for UMAP visualization to show how LoRA adaptation
affects the feature space relative to the source domain.

Usage:
    python -m experiments.lora_ablation.extract_domain_features \
        --config experiments/lora_ablation/config/ablation.yaml \
        --condition baseline \
        --n-samples 200
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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


# BraTS-GLI uses similar structure but different prefix
GLIOMA_MODALITY_SUFFIXES = {
    "t1c": "-t1c.nii.gz",
    "t1n": "-t1n.nii.gz",
    "t2f": "-t2f.nii.gz",
    "t2w": "-t2w.nii.gz",
}
GLIOMA_SEG_SUFFIX = "-seg.nii.gz"


class BraTSGLIDataset(torch.utils.data.Dataset):
    """BraTS-GLI (Glioma) dataset for feature extraction.

    Similar to BraTSMENDataset but for glioma data.
    Uses the same transform pipeline.
    """

    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[str]] = None,
        transform=None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform or get_val_transforms()

        # Discover subjects if not provided
        if subject_ids is None:
            self.subject_ids = self._discover_subjects()
        else:
            self.subject_ids = subject_ids

        logger.info(f"BraTSGLIDataset initialized with {len(self.subject_ids)} subjects")

    def _discover_subjects(self) -> List[str]:
        """Discover all subject IDs in the glioma dataset."""
        subject_ids = []
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name.startswith("BraTS-GLI-"):
                subject_ids.append(item.name)
        return sorted(subject_ids)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict:
        subject_id = self.subject_ids[idx]
        subject_dir = self.data_root / subject_id

        # Build data dict for transforms
        data = {}
        for modality, suffix in GLIOMA_MODALITY_SUFFIXES.items():
            path = subject_dir / f"{subject_id}{suffix}"
            if path.exists():
                data[modality] = str(path)
            else:
                raise FileNotFoundError(f"Missing {modality}: {path}")

        # Segmentation (optional for feature extraction)
        seg_path = subject_dir / f"{subject_id}{GLIOMA_SEG_SUFFIX}"
        if seg_path.exists():
            data["seg"] = str(seg_path)

        # Apply transforms
        transformed = self.transform(data)

        return {
            "image": transformed["image"],
            "subject_id": subject_id,
            "domain": "glioma",
        }


def load_encoder_for_condition(
    condition_name: str,
    config: dict,
    device: str,
) -> torch.nn.Module:
    """Load encoder for a trained condition (same as extract_features.py)."""
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
        logger.info("Loaded baseline encoder (no LoRA)")
        encoder = base_encoder
    else:
        adapter_path = condition_dir / "adapter"
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_path}. Train the condition first."
            )

        logger.info(f"Loading LoRA adapter from {adapter_path}")
        lora_encoder = LoRASwinViT.load_lora(
            base_encoder,
            adapter_path,
            device=device,
            trainable=False,
        )
        encoder = lora_encoder.merge_lora()
        logger.info("Merged LoRA weights into base encoder")

    encoder.eval()
    return encoder


@torch.no_grad()
def extract_features_subset(
    encoder: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    n_samples: int,
    device: str,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Extract features from a random subset of the dataset.

    Args:
        encoder: SwinUNETR encoder.
        dataset: Dataset to extract from.
        n_samples: Number of samples to extract.
        device: Device to use.
        batch_size: Batch size.
        num_workers: Data loading workers.
        seed: Random seed for subset selection.

    Returns:
        Tuple of (features [N, 768], subject_ids [N]).
    """
    # Select random subset
    rng = np.random.RandomState(seed)
    n_total = len(dataset)
    n_samples = min(n_samples, n_total)

    indices = rng.choice(n_total, n_samples, replace=False)
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create feature extractor
    feature_extractor = FeatureExtractor(encoder, level="encoder10")

    all_features = []
    all_ids = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        images = batch["image"].to(device)
        features = feature_extractor(images)
        all_features.append(features.cpu().numpy())

        ids = batch["subject_id"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        all_ids.extend(ids)

    features = np.concatenate(all_features, axis=0)
    logger.info(f"Extracted features shape: {features.shape}")

    return features, all_ids


def extract_glioma_features(
    config: dict,
    condition_name: str,
    n_samples: int = 200,
    device: str = "cuda",
) -> Path:
    """Extract features from BraTS-GLI (glioma) dataset.

    Args:
        config: Experiment configuration.
        condition_name: Which condition's encoder to use.
        n_samples: Number of samples to extract.
        device: Device to use.

    Returns:
        Path to saved features file.
    """
    # Get glioma data path from foundation config or ablation config
    glioma_root = config.get("paths", {}).get("glioma_root")

    if glioma_root is None:
        # Try loading from foundation.yaml
        foundation_config_path = Path("src/growth/config/foundation.yaml")
        if foundation_config_path.exists():
            with open(foundation_config_path) as f:
                foundation_config = yaml.safe_load(f)
            glioma_root = foundation_config.get("paths", {}).get("brats_gli_root")

    if glioma_root is None:
        raise ValueError(
            "Glioma dataset path not found. Set paths.glioma_root in config "
            "or paths.brats_gli_root in foundation.yaml"
        )

    glioma_root = Path(glioma_root)
    if not glioma_root.exists():
        raise FileNotFoundError(f"Glioma dataset not found at {glioma_root}")

    logger.info(f"Loading glioma dataset from {glioma_root}")

    # Create dataset
    dataset = BraTSGLIDataset(
        data_root=glioma_root,
        transform=get_val_transforms(),
    )

    # Load encoder
    encoder = load_encoder_for_condition(condition_name, config, device)

    # Extract features
    features, subject_ids = extract_features_subset(
        encoder=encoder,
        dataset=dataset,
        n_samples=n_samples,
        device=device,
        seed=config["experiment"]["seed"],
    )

    # Save features
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    output_path = condition_dir / "features_glioma.pt"
    torch.save({
        "features": torch.from_numpy(features),
        "subject_ids": subject_ids,
        "domain": "glioma",
        "n_samples": len(subject_ids),
    }, output_path)

    logger.info(f"Saved glioma features to {output_path}")
    return output_path


def extract_meningioma_subset_features(
    config: dict,
    condition_name: str,
    n_samples: int = 200,
    device: str = "cuda",
    config_path: str = "experiments/lora_ablation/config/ablation.yaml",
) -> Path:
    """Extract features from a subset of BraTS-MEN test set.

    Args:
        config: Experiment configuration.
        condition_name: Which condition's encoder to use.
        n_samples: Number of samples to extract.
        device: Device to use.
        config_path: Path to config file (for loading splits).

    Returns:
        Path to saved features file.
    """
    # Load splits to get test set subjects
    splits = load_splits(config_path)

    test_subjects = splits["test"]
    logger.info(f"Test set has {len(test_subjects)} subjects")

    # Create dataset with test subjects only
    dataset = BraTSMENDataset(
        data_root=config["paths"]["data_root"],
        subject_ids=test_subjects,
        transform=get_val_transforms(),
        compute_semantic=False,
    )

    # Load encoder
    encoder = load_encoder_for_condition(condition_name, config, device)

    # Extract features
    features, subject_ids = extract_features_subset(
        encoder=encoder,
        dataset=dataset,
        n_samples=n_samples,
        device=device,
        seed=config["experiment"]["seed"],
    )

    # Save features
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    output_path = condition_dir / "features_meningioma_subset.pt"
    torch.save({
        "features": torch.from_numpy(features),
        "subject_ids": subject_ids,
        "domain": "meningioma",
        "n_samples": len(subject_ids),
    }, output_path)

    logger.info(f"Saved meningioma subset features to {output_path}")
    return output_path


def extract_domain_features(
    config_path: str,
    condition_name: str,
    n_glioma: int = 200,
    n_meningioma: int = 200,
    device: str = "cuda",
) -> Dict[str, Path]:
    """Extract features for domain comparison UMAP.

    Args:
        config_path: Path to ablation config.
        condition_name: Condition to use for extraction.
        n_glioma: Number of glioma samples.
        n_meningioma: Number of meningioma samples.
        device: Device to use.

    Returns:
        Dict with paths to saved feature files.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])

    paths = {}

    # Extract glioma features
    logger.info(f"\n{'='*50}")
    logger.info(f"Extracting {n_glioma} glioma features")
    logger.info(f"{'='*50}")
    try:
        paths["glioma"] = extract_glioma_features(
            config, condition_name, n_glioma, device
        )
    except Exception as e:
        logger.error(f"Failed to extract glioma features: {e}")
        paths["glioma"] = None

    # Extract meningioma subset features
    logger.info(f"\n{'='*50}")
    logger.info(f"Extracting {n_meningioma} meningioma features")
    logger.info(f"{'='*50}")
    paths["meningioma"] = extract_meningioma_subset_features(
        config, condition_name, n_meningioma, device, config_path
    )

    return paths


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract features for domain shift visualization"
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
        default="baseline",
        help="Condition to use for feature extraction",
    )
    parser.add_argument(
        "--n-glioma",
        type=int,
        default=200,
        help="Number of glioma samples to extract",
    )
    parser.add_argument(
        "--n-meningioma",
        type=int,
        default=200,
        help="Number of meningioma samples to extract",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    extract_domain_features(
        config_path=args.config,
        condition_name=args.condition,
        n_glioma=args.n_glioma,
        n_meningioma=args.n_meningioma,
        device=args.device,
    )


if __name__ == "__main__":
    main()
