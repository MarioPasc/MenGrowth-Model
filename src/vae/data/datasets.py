"""Dataset utilities for BraTS MRI data.

This module provides functions for:
- Building subject index from directory structure
- Creating deterministic train/val splits
- Building MONAI PersistentDataset and PyTorch DataLoaders
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import PersistentDataset
from omegaconf import DictConfig

from .transforms import get_train_transforms, get_val_transforms


logger = logging.getLogger(__name__)


def safe_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that handles both numpy arrays and tensors.

    This collate function ensures that data is always converted to tensors,
    even if PersistentDataset returns cached numpy arrays. It also filters
    out MONAI metadata keys that can cause issues with PyTorch Lightning's
    batch size extraction.

    Only processes keys needed for training: "image", "seg", and "id".
    This prevents issues with MONAI MetaTensor metadata (scalars, 0-d arrays).

    Args:
        batch: List of data dictionaries from the dataset.

    Returns:
        Collated batch dictionary with tensor values for image/seg keys and explicit batch_size.
    """
    if not batch:
        return {}

    # Only collate the keys we need for training
    # Filter out MONAI metadata keys (like "image_meta_dict", "affine", etc.)
    # that can contain scalar values causing PyTorch Lightning batch size errors
    keys_to_collate = ["image", "seg", "id"]

    collated = {}

    for key in keys_to_collate:
        # Skip if key doesn't exist in batch
        if key not in batch[0]:
            continue

        values = [item[key] for item in batch]

        # Check if first value is a numpy array or tensor
        first_val = values[0]

        if isinstance(first_val, np.ndarray):
            # Convert numpy arrays to tensors
            tensors = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in values]
            collated[key] = torch.stack(tensors, dim=0)
        elif isinstance(first_val, torch.Tensor):
            # Stack tensors normally
            collated[key] = torch.stack(values, dim=0)
        else:
            # For other types (strings, etc.), keep as list
            collated[key] = values

    # Add explicit batch_size to help PyTorch Lightning's batch size inference
    # This prevents warnings about ambiguous batch size detection
    collated["batch_size"] = len(batch)

    return collated


def build_subject_index(root_dir: str, modalities: List[str]) -> List[Dict[str, str]]:
    """Build a list of subject data dictionaries by scanning the root directory.

    Each subject directory should contain NIfTI files matching the pattern:
        *-{modality}.nii.gz (e.g., BraTS-MEN-00001-000-t1c.nii.gz)

    Args:
        root_dir: Path to the root data directory containing subject folders.
        modalities: List of modality names (e.g., ["t1c", "t1n", "t2f", "t2w"]).

    Returns:
        List of dicts, each with keys: modality names, "seg", and "id".
        Example: {"t1c": "/path/to/t1c.nii.gz", ..., "seg": "/path/to/seg.nii.gz", "id": "BraTS-MEN-00001"}

    Raises:
        FileNotFoundError: If root_dir does not exist.
        ValueError: If no valid subjects are found.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Data root directory not found: {root_dir}")

    subjects = []
    subject_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        subject_data = {"id": subject_id}
        valid = True

        # Find modality files
        for mod in modalities:
            mod_files = list(subject_dir.glob(f"*-{mod}.nii.gz"))
            if len(mod_files) != 1:
                logger.warning(f"Subject {subject_id}: expected 1 {mod} file, found {len(mod_files)}")
                valid = False
                break
            subject_data[mod] = str(mod_files[0])

        # Find segmentation file
        if valid:
            seg_files = list(subject_dir.glob("*-seg.nii.gz"))
            if len(seg_files) != 1:
                logger.warning(f"Subject {subject_id}: expected 1 seg file, found {len(seg_files)}")
                valid = False
            else:
                subject_data["seg"] = str(seg_files[0])

        if valid:
            subjects.append(subject_data)

    if not subjects:
        raise ValueError(f"No valid subjects found in {root_dir}")

    logger.info(f"Found {len(subjects)} valid subjects in {root_dir}")
    return subjects


def create_train_val_split(
    subjects: List[Dict[str, str]],
    val_split: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Create deterministic train/val split from subject list.

    Args:
        subjects: List of subject data dictionaries.
        val_split: Fraction of subjects to use for validation (0.0 to 1.0).
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple of (train_subjects, val_subjects).
    """
    n_subjects = len(subjects)
    n_val = max(1, int(n_subjects * val_split))
    n_train = n_subjects - n_val

    # Deterministic shuffle using seeded generator
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_subjects, generator=generator).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subjects = [subjects[i] for i in sorted(train_indices)]
    val_subjects = [subjects[i] for i in sorted(val_indices)]

    logger.info(f"Split: {len(train_subjects)} train, {len(val_subjects)} val subjects")

    return train_subjects, val_subjects


def get_dataloaders(
    cfg: DictConfig,
    run_dir: str,
    train_subjects: List[Dict[str, str]],
    val_subjects: List[Dict[str, str]],
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders with PersistentDataset caching.

    Args:
        cfg: Configuration object with data parameters.
        run_dir: Path to run directory for cache storage.
        train_subjects: List of training subject data dicts.
        val_subjects: List of validation subject data dicts.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Extract config values
    spacing = tuple(cfg.data.spacing)
    orientation = cfg.data.orientation
    roi_size = tuple(cfg.data.roi_size)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers
    cache_subdir = cfg.data.get("persistent_cache_subdir", "cache")

    # Setup cache directory
    cache_dir = Path(run_dir) / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "val"
    train_cache_dir.mkdir(exist_ok=True)
    val_cache_dir.mkdir(exist_ok=True)

    # Get transforms
    train_transforms = get_train_transforms(spacing, orientation, roi_size)
    val_transforms = get_val_transforms(spacing, orientation, roi_size)

    # Create PersistentDatasets
    train_dataset = PersistentDataset(
        data=train_subjects,
        transform=train_transforms,
        cache_dir=str(train_cache_dir),
    )

    val_dataset = PersistentDataset(
        data=val_subjects,
        transform=val_transforms,
        cache_dir=str(val_cache_dir),
    )

    logger.info(f"Created PersistentDataset with cache at {cache_dir}")

    # Determine DataLoader settings
    use_cuda = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=persistent_workers,
        drop_last=True,
        collate_fn=safe_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=persistent_workers,
        drop_last=False,
        collate_fn=safe_collate,
    )

    logger.info(
        f"DataLoaders created: train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, batch_size={batch_size}"
    )

    return train_loader, val_loader
