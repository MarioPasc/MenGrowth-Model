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
from .semantic_features import SemanticFeatureNormalizer


logger = logging.getLogger(__name__)


def safe_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that handles both numpy arrays and tensors.

    This collate function ensures that data is always converted to tensors,
    even if PersistentDataset returns cached numpy arrays. It also filters
    out MONAI metadata keys that can cause issues with PyTorch Lightning's
    batch size extraction.

    Processes keys needed for training: "image", "seg", "id", and "semantic_features".
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

    # Handle semantic_features (nested dict of tensors)
    if "semantic_features" in batch[0]:
        semantic_features = {}
        # Get all partition keys from first sample
        partition_keys = batch[0]["semantic_features"].keys()

        for pkey in partition_keys:
            # Stack tensors for each partition
            values = [item["semantic_features"][pkey] for item in batch]
            if isinstance(values[0], torch.Tensor):
                semantic_features[pkey] = torch.stack(values, dim=0)
            elif isinstance(values[0], np.ndarray):
                tensors = [torch.from_numpy(v) for v in values]
                semantic_features[pkey] = torch.stack(tensors, dim=0)

        collated["semantic_features"] = semantic_features

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


def create_train_val_test_split(
    subjects: List[Dict[str, str]],
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Create deterministic train/val/test split from subject list.

    The test set is held out for final evaluation after training completes.
    Only train and val sets are used during the training loop.

    Args:
        subjects: List of subject data dictionaries.
        val_split: Fraction of subjects for validation (0.0 to 1.0).
        test_split: Fraction of subjects for test (0.0 to 1.0).
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects).

    Raises:
        ValueError: If val_split + test_split >= 1.0
    """
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) must be < 1.0"
        )

    n_subjects = len(subjects)
    n_test = max(1, int(n_subjects * test_split)) if test_split > 0 else 0
    n_val = max(1, int(n_subjects * val_split))
    n_train = n_subjects - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Not enough subjects for training. "
            f"Total: {n_subjects}, val: {n_val}, test: {n_test}, train: {n_train}"
        )

    # Deterministic shuffle using seeded generator
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_subjects, generator=generator).tolist()

    # Split indices: train | val | test
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Sort indices for reproducibility (same subjects always in same split)
    train_subjects = [subjects[i] for i in sorted(train_indices)]
    val_subjects = [subjects[i] for i in sorted(val_indices)]
    test_subjects = [subjects[i] for i in sorted(test_indices)]

    logger.info(
        f"Split: {len(train_subjects)} train, {len(val_subjects)} val, "
        f"{len(test_subjects)} test subjects"
    )

    return train_subjects, val_subjects, test_subjects


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
    modalities = list(cfg.data.modalities)
    spacing = tuple(cfg.data.spacing)
    orientation = cfg.data.orientation
    roi_size = tuple(cfg.data.roi_size)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers
    cache_subdir = cfg.data.get("persistent_cache_subdir", "cache")

    # Check if semantic feature extraction is enabled (for semi-supervised VAE)
    extract_semantic = cfg.data.get("extract_semantic_features", False)
    seg_labels = None
    semantic_normalizer = None
    require_normalizer = cfg.data.get("require_normalizer", False)

    if extract_semantic:
        # Get segmentation labels from logging config (shared with callbacks)
        seg_labels = cfg.logging.get("seg_labels", None)
        if seg_labels is not None:
            seg_labels = dict(seg_labels)  # Convert DictConfig to dict
        logger.info(f"Semantic feature extraction enabled with labels: {seg_labels}")

    # Setup cache directory
    cache_dir = Path(run_dir) / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "val"
    train_cache_dir.mkdir(exist_ok=True)
    val_cache_dir.mkdir(exist_ok=True)

    # Load semantic normalizer if semantic extraction is enabled
    if extract_semantic:
        normalizer_path = cache_dir / "semantic_normalizer.json"
        if normalizer_path.exists():
            semantic_normalizer = SemanticFeatureNormalizer.load(str(normalizer_path))
            logger.info(f"Loaded semantic normalizer from {normalizer_path}")
            logger.info(f"  Features: {len(semantic_normalizer.feature_names)} dimensions")
        else:
            msg = (
                f"Semantic normalizer not found at {normalizer_path}. "
                "Run scripts/precompute_semantic_normalizer.py first to compute "
                "training set statistics for z-score normalization."
            )
            if require_normalizer:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg)
                logger.warning("Features will NOT be normalized - gradient imbalance may occur!")

    # Get transforms (with optional semantic extraction and normalizer)
    residualize_shape = cfg.data.get("residualize_shape", False)
    residual_params_path = cfg.data.get("residual_params_path", None)

    train_transforms = get_train_transforms(
        modalities, spacing, orientation, roi_size,
        extract_semantic=extract_semantic,
        seg_labels=seg_labels,
        semantic_normalizer=semantic_normalizer,
        residualize_shape=residualize_shape,
        residual_params_path=residual_params_path,
    )
    val_transforms = get_val_transforms(
        modalities, spacing, orientation, roi_size,
        extract_semantic=extract_semantic,
        seg_labels=seg_labels,
        semantic_normalizer=semantic_normalizer,
        residualize_shape=residualize_shape,
        residual_params_path=residual_params_path,
    )

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

    # DDP gather safety: drop_last must be True for training when use_ddp_gather is enabled
    # This is CRITICAL because DIP-VAE covariance all_gather happens in TRAINING step (loss computation)
    num_devices = cfg.train.get("devices", 1)
    # Safe access: cfg.loss may not exist in baseline VAE config (vae.yaml)
    loss_cfg = cfg.get("loss", {})
    use_ddp_gather = loss_cfg.get("use_ddp_gather", False) if loss_cfg else False
    train_drop_last = True  # Always True for stable training

    # Log DDP-specific configuration if applicable
    if use_ddp_gather and num_devices > 1:
        logger.info(
            f"DDP mode with use_ddp_gather=True (devices={num_devices}): "
            f"enforcing drop_last=True on TRAINING loader to ensure equal batch sizes "
            f"for all_gather in DIP-VAE covariance computation."
        )

    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=persistent_workers,
        drop_last=train_drop_last,  # CRITICAL: Must be True for DDP gather safety
        collate_fn=safe_collate,
    )

    # Validation loader: also use drop_last=True for callback safety (latent diagnostics)
    # Secondary concern compared to training, but helps avoid uneven batches in callbacks
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=persistent_workers,
        drop_last=True,  # For callback safety in multi-GPU scenarios
        collate_fn=safe_collate,
    )

    # Validate DDP configuration
    if use_ddp_gather and num_devices > 1 and not train_drop_last:
        raise RuntimeError(
            "use_ddp_gather=True requires drop_last=True on training loader to prevent "
            "all_gather shape mismatch during DIP-VAE covariance computation. "
            f"Current: drop_last={train_drop_last}"
        )

    logger.info(
        f"DataLoaders created: train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, batch_size={batch_size}"
    )

    return train_loader, val_loader


def get_dataloaders_with_test(
    cfg: DictConfig,
    run_dir: str,
    train_subjects: List[Dict[str, str]],
    val_subjects: List[Dict[str, str]],
    test_subjects: Optional[List[Dict[str, str]]] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train, validation, and test DataLoaders with PersistentDataset caching.

    The test loader is created for final evaluation after training completes.
    Test data uses validation transforms (no augmentation).

    Args:
        cfg: Configuration object with data parameters.
        run_dir: Path to run directory for cache storage.
        train_subjects: List of training subject data dicts.
        val_subjects: List of validation subject data dicts.
        test_subjects: Optional list of test subject data dicts.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        test_loader is None if test_subjects is None or empty.
    """
    # Extract config values
    modalities = list(cfg.data.modalities)
    spacing = tuple(cfg.data.spacing)
    orientation = cfg.data.orientation
    roi_size = tuple(cfg.data.roi_size)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers
    cache_subdir = cfg.data.get("persistent_cache_subdir", "cache")

    # Check if semantic feature extraction is enabled
    extract_semantic = cfg.data.get("extract_semantic_features", False)
    seg_labels = None
    semantic_normalizer = None
    require_normalizer = cfg.data.get("require_normalizer", False)

    if extract_semantic:
        seg_labels = cfg.logging.get("seg_labels", None)
        if seg_labels is not None:
            seg_labels = dict(seg_labels)
        logger.info(f"Semantic feature extraction enabled with labels: {seg_labels}")

    # Setup cache directories
    cache_dir = Path(run_dir) / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "val"
    test_cache_dir = cache_dir / "test"
    train_cache_dir.mkdir(exist_ok=True)
    val_cache_dir.mkdir(exist_ok=True)
    test_cache_dir.mkdir(exist_ok=True)

    # Load semantic normalizer if enabled
    if extract_semantic:
        normalizer_path = cache_dir / "semantic_normalizer.json"
        if normalizer_path.exists():
            semantic_normalizer = SemanticFeatureNormalizer.load(str(normalizer_path))
            logger.info(f"Loaded semantic normalizer from {normalizer_path}")
            logger.info(f"  Features: {len(semantic_normalizer.feature_names)} dimensions")
        else:
            msg = (
                f"Semantic normalizer not found at {normalizer_path}. "
                "Run vae.utils.precompute_semantic_normalizer first."
            )
            if require_normalizer:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg)
                logger.warning("Features will NOT be normalized!")

    # Get transforms
    residualize_shape = cfg.data.get("residualize_shape", False)
    residual_params_path = cfg.data.get("residual_params_path", None)

    train_transforms = get_train_transforms(
        modalities, spacing, orientation, roi_size,
        extract_semantic=extract_semantic,
        seg_labels=seg_labels,
        semantic_normalizer=semantic_normalizer,
        residualize_shape=residualize_shape,
        residual_params_path=residual_params_path,
    )
    val_transforms = get_val_transforms(
        modalities, spacing, orientation, roi_size,
        extract_semantic=extract_semantic,
        seg_labels=seg_labels,
        semantic_normalizer=semantic_normalizer,
        residualize_shape=residualize_shape,
        residual_params_path=residual_params_path,
    )

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

    # Test dataset uses val_transforms (no augmentation)
    test_dataset = None
    if test_subjects:
        test_dataset = PersistentDataset(
            data=test_subjects,
            transform=val_transforms,  # Same as val - no augmentation
            cache_dir=str(test_cache_dir),
        )

    logger.info(f"Created PersistentDataset with cache at {cache_dir}")

    # DataLoader settings
    use_cuda = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    num_devices = cfg.train.get("devices", 1)
    loss_cfg = cfg.get("loss", {})
    use_ddp_gather = loss_cfg.get("use_ddp_gather", False) if loss_cfg else False

    if use_ddp_gather and num_devices > 1:
        logger.info(
            f"DDP mode with use_ddp_gather=True (devices={num_devices}): "
            f"enforcing drop_last=True for DDP gather safety."
        )

    # Create DataLoaders
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
        drop_last=True,
        collate_fn=safe_collate,
    )

    # Test loader: drop_last=False to evaluate ALL test samples
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=persistent_workers,
            drop_last=False,  # Evaluate all test samples
            collate_fn=safe_collate,
        )

    n_test = len(test_loader) if test_loader else 0
    logger.info(
        f"DataLoaders created: train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, test={n_test} batches, "
        f"batch_size={batch_size}"
    )

    return train_loader, val_loader, test_loader
