# src/growth/data/bratsmendata.py
"""BraTS-MEN dataset for Phases 1 and 2.

Loads 4-channel MRI volumes and segmentation masks from BraTS Meningioma dataset.
Provides semantic features (volume, location, shape) extracted from segmentations.

Directory structure expected:
    BraTS_Men_Train/
    ├── BraTS-MEN-00004-000/
    │   ├── BraTS-MEN-00004-000-t1c.nii.gz
    │   ├── BraTS-MEN-00004-000-t1n.nii.gz
    │   ├── BraTS-MEN-00004-000-t2f.nii.gz
    │   ├── BraTS-MEN-00004-000-t2w.nii.gz
    │   └── BraTS-MEN-00004-000-seg.nii.gz
    ├── BraTS-MEN-00005-000/
    │   └── ...
    └── ...
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .semantic_features import extract_semantic_features
from .transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

# Modality file suffixes
MODALITY_SUFFIXES = {
    "t1c": "-t1c.nii.gz",
    "t1n": "-t1n.nii.gz",
    "t2f": "-t2f.nii.gz",
    "t2w": "-t2w.nii.gz",
}
SEG_SUFFIX = "-seg.nii.gz"


class BraTSMENDataset(Dataset):
    """BraTS-MEN dataset for encoder training and evaluation.

    Loads 4-channel MRI volumes (T1c, T1n, T2-FLAIR, T2w) and segmentation masks
    from the BraTS Meningioma dataset. Optionally computes semantic features
    (volume, location, shape) from segmentation masks.

    Args:
        data_root: Path to BraTS-MEN directory.
        subject_ids: List of subject IDs to include. If None, discover all.
        transform: MONAI transform pipeline. If None, uses default val transforms.
        compute_semantic: If True, compute semantic features from segmentation.
        cache_semantic: If True, cache semantic features to disk for faster loading.
        cache_dir: Directory for caching. Defaults to data_root/.semantic_cache.

    Returns per sample:
        dict with:
        - 'image': [4, D, H, W] MRI tensor
        - 'seg': [1, D, H, W] segmentation mask
        - 'subject_id': str
        - 'semantic_features': dict (if compute_semantic=True)
            - 'volume': [4] tensor
            - 'location': [3] tensor
            - 'shape': [3] tensor (sphericity, surface_area_log, solidity)
            - 'all': [10] tensor

    Example:
        >>> dataset = BraTSMENDataset(
        ...     data_root="/path/to/BraTS_Men_Train",
        ...     compute_semantic=True,
        ... )
        >>> sample = dataset[0]
        >>> sample['image'].shape
        torch.Size([4, 96, 96, 96])
        >>> sample['semantic_features']['volume'].shape
        torch.Size([4])
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        subject_ids: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        compute_semantic: bool = True,
        cache_semantic: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.data_root = Path(data_root)
        self.compute_semantic = compute_semantic
        self.cache_semantic = cache_semantic

        # Set up semantic cache directory
        if cache_dir is None:
            self.cache_dir = self.data_root / ".semantic_cache"
        else:
            self.cache_dir = Path(cache_dir)

        if self.cache_semantic:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Discover or validate subject IDs
        if subject_ids is None:
            self.subject_ids = self.get_all_subject_ids(self.data_root)
        else:
            self.subject_ids = subject_ids

        # Validate subjects exist
        self._validate_subjects()

        # Set up transforms
        if transform is None:
            self.transform = get_val_transforms()
        else:
            self.transform = transform

        logger.info(
            f"BraTSMENDataset initialized with {len(self.subject_ids)} subjects"
        )

    def _validate_subjects(self) -> None:
        """Validate that all subject directories exist."""
        missing = []
        for subject_id in self.subject_ids:
            subject_dir = self.data_root / subject_id
            if not subject_dir.is_dir():
                missing.append(subject_id)

        if missing:
            raise ValueError(
                f"Missing {len(missing)} subject directories. "
                f"First 5: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        subject_id = self.subject_ids[idx]

        # Get file paths
        paths = self.load_subject_paths(self.data_root, subject_id)

        # Build data dict for transforms
        data = {
            "t1c": str(paths["t1c"]),
            "t1n": str(paths["t1n"]),
            "t2f": str(paths["t2f"]),
            "t2w": str(paths["t2w"]),
            "seg": str(paths["seg"]),
        }

        # Apply transforms
        transformed = self.transform(data)

        # Build output dict
        output = {
            "image": transformed["image"],
            "seg": transformed["seg"],
            "subject_id": subject_id,
        }

        # Compute semantic features if requested
        if self.compute_semantic:
            semantic = self._get_semantic_features(subject_id, paths["seg"])
            output["semantic_features"] = {
                "volume": torch.from_numpy(semantic["volume"]),
                "location": torch.from_numpy(semantic["location"]),
                "shape": torch.from_numpy(semantic["shape"]),
                "all": torch.from_numpy(semantic["all"]),
            }

        return output

    def _get_semantic_features(
        self, subject_id: str, seg_path: Path
    ) -> Dict[str, np.ndarray]:
        """Get semantic features, using cache if available."""
        if self.cache_semantic:
            cache_path = self._get_cache_path(subject_id)

            if cache_path.exists():
                # Load from cache
                return self._load_cached_features(cache_path)

        # Compute features from original segmentation (not transformed)
        # This ensures features are computed on native resolution
        import nibabel as nib

        seg_img = nib.load(seg_path)
        seg_data = np.asarray(seg_img.dataobj).astype(np.int32)
        spacing = tuple(seg_img.header.get_zooms()[:3])

        features = extract_semantic_features(seg_data, spacing)

        # Cache if enabled
        if self.cache_semantic:
            self._save_cached_features(cache_path, features)

        return features

    def _get_cache_path(self, subject_id: str) -> Path:
        """Get cache file path for a subject."""
        return self.cache_dir / f"{subject_id}_semantic.npz"

    def _load_cached_features(self, cache_path: Path) -> Dict[str, np.ndarray]:
        """Load features from cache file."""
        data = np.load(cache_path)
        return {
            "volume": data["volume"],
            "location": data["location"],
            "shape": data["shape"],
            "all": data["all"],
        }

    def _save_cached_features(
        self, cache_path: Path, features: Dict[str, np.ndarray]
    ) -> None:
        """Save features to cache file."""
        np.savez(
            cache_path,
            volume=features["volume"],
            location=features["location"],
            shape=features["shape"],
            all=features["all"],
        )

    @staticmethod
    def get_all_subject_ids(data_root: Union[str, Path]) -> List[str]:
        """Discover all subject IDs in directory.

        Args:
            data_root: Path to BraTS-MEN directory.

        Returns:
            Sorted list of subject IDs (directory names starting with 'BraTS-MEN-').
        """
        data_root = Path(data_root)
        subject_ids = []

        for item in data_root.iterdir():
            if item.is_dir() and item.name.startswith("BraTS-MEN-"):
                subject_ids.append(item.name)

        return sorted(subject_ids)

    @staticmethod
    def load_subject_paths(
        data_root: Union[str, Path], subject_id: str
    ) -> Dict[str, Path]:
        """Get paths to all modalities and segmentation for a subject.

        Args:
            data_root: Path to BraTS-MEN directory.
            subject_id: Subject directory name.

        Returns:
            Dict with keys: 't1c', 't1n', 't2f', 't2w', 'seg'

        Raises:
            FileNotFoundError: If any required file is missing.
        """
        data_root = Path(data_root)
        subject_dir = data_root / subject_id

        paths = {}

        # Load modalities
        for modality, suffix in MODALITY_SUFFIXES.items():
            path = subject_dir / f"{subject_id}{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"Missing {modality} file: {path}")
            paths[modality] = path

        # Load segmentation
        seg_path = subject_dir / f"{subject_id}{SEG_SUFFIX}"
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
        paths["seg"] = seg_path

        return paths


def create_dataloaders(
    data_root: Union[str, Path],
    train_ids: List[str],
    val_ids: List[str],
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    compute_semantic: bool = True,
    cache_semantic: bool = True,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        data_root: Path to BraTS-MEN directory.
        train_ids: List of subject IDs for training.
        val_ids: List of subject IDs for validation.
        batch_size: Batch size for both loaders.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        compute_semantic: Whether to compute semantic features.
        cache_semantic: Whether to cache semantic features.
        augment_train: Whether to apply augmentation to training data.

    Returns:
        Tuple of (train_loader, val_loader).

    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     data_root="/path/to/BraTS_Men_Train",
        ...     train_ids=["BraTS-MEN-00004-000", "BraTS-MEN-00005-000"],
        ...     val_ids=["BraTS-MEN-00006-000"],
        ... )
    """
    # Create datasets
    train_dataset = BraTSMENDataset(
        data_root=data_root,
        subject_ids=train_ids,
        transform=get_train_transforms() if augment_train else get_val_transforms(),
        compute_semantic=compute_semantic,
        cache_semantic=cache_semantic,
    )

    val_dataset = BraTSMENDataset(
        data_root=data_root,
        subject_ids=val_ids,
        transform=get_val_transforms(),
        compute_semantic=compute_semantic,
        cache_semantic=cache_semantic,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def split_subjects(
    subject_ids: List[str],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split subjects into train/val/test sets.

    Uses deterministic random shuffling based on seed.

    Args:
        subject_ids: List of all subject IDs.
        train_size: Number of subjects for training.
        val_size: Number of subjects for validation.
        test_size: Number of subjects for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to subject ID lists.

    Raises:
        ValueError: If total size exceeds available subjects.

    Example:
        >>> all_ids = BraTSMENDataset.get_all_subject_ids(data_root)
        >>> splits = split_subjects(all_ids, train=200, val=100, test=500)
        >>> len(splits['train'])
        200
    """
    total_needed = train_size + val_size + test_size
    if total_needed > len(subject_ids):
        raise ValueError(
            f"Requested {total_needed} subjects but only {len(subject_ids)} available"
        )

    # Deterministic shuffle
    rng = np.random.RandomState(seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)

    # Split
    train_end = train_size
    val_end = train_end + val_size
    test_end = val_end + test_size

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:test_end],
    }


def save_splits(splits: Dict[str, List[str]], path: Union[str, Path]) -> None:
    """Save data splits to JSON file.

    Args:
        splits: Dict mapping split names to subject ID lists.
        path: Output JSON file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Saved splits to {path}")


def load_splits(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Load data splits from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Dict mapping split names to subject ID lists.
    """
    with open(path) as f:
        return json.load(f)
