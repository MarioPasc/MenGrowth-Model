# src/growth/data/bratsmendata.py
"""BraTS-MEN dataset for Phases 1 and 2.

Supports two backends:
  - **NIfTI** (original): loads from per-subject directories with 5 NIfTI files each.
  - **HDF5** (fast): loads from a single pre-preprocessed H5 file (192³ volumes).

The H5 backend eliminates NIfTI decompression, resampling, and multi-file seek
overhead, reducing I/O from ~5000 file opens per epoch (1000 subjects) to a
single memory-mapped file with chunked reads.

NIfTI directory structure expected:
    BraTS_Men_Train/
    ├── BraTS-MEN-00004-000/
    │   ├── BraTS-MEN-00004-000-t1c.nii.gz
    │   ├── BraTS-MEN-00004-000-t1n.nii.gz
    │   ├── BraTS-MEN-00004-000-t2f.nii.gz
    │   ├── BraTS-MEN-00004-000-t2w.nii.gz
    │   └── BraTS-MEN-00004-000-seg.nii.gz
    └── ...

H5 file schema (see scripts/convert_nifti_to_h5.py):
    images       [N, 4, 192, 192, 192] float32
    segs         [N, 1, 192, 192, 192] int8
    subject_ids  [N] string
    semantic/    {volume, location, shape} arrays
    splits/      {lora_train, lora_val, sdp_train, test} index arrays
"""

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .semantic_features import extract_semantic_features
from .transforms import (
    get_h5_train_transforms,
    get_h5_val_transforms,
    get_train_transforms,
    get_val_transforms,
)

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
        torch.Size([4, 128, 128, 128])
        >>> sample['semantic_features']['volume'].shape
        torch.Size([4])
    """

    def __init__(
        self,
        data_root: str | Path,
        subject_ids: list[str] | None = None,
        transform: Callable | None = None,
        compute_semantic: bool = True,
        cache_semantic: bool = True,
        cache_dir: str | Path | None = None,
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

        logger.info(f"BraTSMENDataset initialized with {len(self.subject_ids)} subjects")

    def _validate_subjects(self) -> None:
        """Validate that all subject directories exist."""
        missing = []
        for subject_id in self.subject_ids:
            subject_dir = self.data_root / subject_id
            if not subject_dir.is_dir():
                missing.append(subject_id)

        if missing:
            raise ValueError(f"Missing {len(missing)} subject directories. First 5: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
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

    def _get_semantic_features(self, subject_id: str, seg_path: Path) -> dict[str, np.ndarray]:
        """Get semantic features, using cache if available.

        Validates cached features match expected dimensions. If cache is stale
        (e.g., from older code version with different feature count), it will
        be recomputed and updated.
        """
        if self.cache_semantic:
            cache_path = self._get_cache_path(subject_id)

            if cache_path.exists():
                # Load from cache (returns None if format is stale)
                cached = self._load_cached_features(cache_path)
                if cached is not None:
                    return cached
                # Stale cache - fall through to recompute

        # Compute features from original segmentation (not transformed)
        # This ensures features are computed on native resolution
        import nibabel as nib

        seg_img = nib.load(seg_path)
        seg_data = np.asarray(seg_img.dataobj).astype(np.int32)
        spacing = tuple(seg_img.header.get_zooms()[:3])

        features = extract_semantic_features(seg_data, spacing)

        # Cache if enabled (overwrites stale cache)
        if self.cache_semantic:
            self._save_cached_features(cache_path, features)

        return features

    def _get_cache_path(self, subject_id: str) -> Path:
        """Get cache file path for a subject."""
        return self.cache_dir / f"{subject_id}_semantic.npz"

    def _load_cached_features(self, cache_path: Path) -> dict[str, np.ndarray] | None:
        """Load features from cache file.

        Returns:
            Feature dict if cache is valid, None if cache format is stale.
        """
        data = np.load(cache_path)

        # Validate expected dimensions (catches stale cache from older code versions)
        # Expected: volume=[4], location=[3], shape=[3], all=[10]
        volume = data["volume"]
        location = data["location"]
        shape = data["shape"]
        all_feats = data["all"]

        if (
            volume.shape != (4,)
            or location.shape != (3,)
            or shape.shape != (3,)
            or all_feats.shape != (10,)
        ):
            logger.warning(
                f"Stale cache detected at {cache_path}: "
                f"shape={shape.shape} (expected (3,)), all={all_feats.shape} (expected (10,)). "
                f"Recomputing features."
            )
            return None

        return {
            "volume": volume,
            "location": location,
            "shape": shape,
            "all": all_feats,
        }

    def _save_cached_features(self, cache_path: Path, features: dict[str, np.ndarray]) -> None:
        """Save features to cache file."""
        np.savez(
            cache_path,
            volume=features["volume"],
            location=features["location"],
            shape=features["shape"],
            all=features["all"],
        )

    @staticmethod
    def get_all_subject_ids(data_root: str | Path) -> list[str]:
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
    def load_subject_paths(data_root: str | Path, subject_id: str) -> dict[str, Path]:
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


class BraTSMENDatasetH5(Dataset):
    """BraTS-MEN dataset backed by a single HDF5 file.

    Reads pre-preprocessed 192³ volumes from an H5 file created by
    ``scripts/convert_nifti_to_h5.py``. Volumes are already Orient→Resample→
    CropForeground→SpatialPad→CenterCrop but NOT z-score normalized (applied
    at runtime by H5 transforms).

    Uses lazy per-worker file handles for multi-worker DataLoader safety:
    each worker opens its own HDF5 file handle on first access.

    Args:
        h5_path: Path to the HDF5 file.
        split: Split name to load indices from (e.g., ``"lora_train"``).
            If None, uses all subjects.
        indices: Explicit index array (overrides ``split``).
        transform: MONAI transform pipeline. If None, uses
            :func:`get_h5_val_transforms`.
        compute_semantic: If True, load semantic features from H5.

    Returns per sample:
        dict with:
        - ``'image'``: ``[4, D, H, W]`` MRI tensor
        - ``'seg'``: ``[1, D, H, W]`` segmentation mask
        - ``'subject_id'``: str
        - ``'semantic_features'``: dict (if ``compute_semantic=True``)
    """

    def __init__(
        self,
        h5_path: str | Path,
        split: str | None = None,
        indices: np.ndarray | None = None,
        transform: Callable | None = None,
        compute_semantic: bool = True,
    ):
        import h5py

        self.h5_path = str(h5_path)
        self.compute_semantic = compute_semantic

        # Read metadata from file (open once, then close)
        with h5py.File(self.h5_path, "r") as f:
            n_total = f.attrs["n_subjects"]
            self._subject_ids = [
                s.decode() if isinstance(s, bytes) else s for s in f["subject_ids"][:]
            ]

            # Determine indices
            if indices is not None:
                self._indices = np.asarray(indices, dtype=np.int64)
            elif split is not None:
                if f"splits/{split}" not in f:
                    available = list(f["splits"].keys()) if "splits" in f else []
                    raise KeyError(f"Split '{split}' not found in H5 file. Available: {available}")
                self._indices = f[f"splits/{split}"][:].astype(np.int64)
            else:
                self._indices = np.arange(n_total, dtype=np.int64)

        # Per-worker lazy file handle (thread-local for multi-worker safety)
        self._local = threading.local()

        # Transform pipeline
        if transform is None:
            self.transform = get_h5_val_transforms()
        else:
            self.transform = transform

        logger.info(
            f"BraTSMENDatasetH5 initialized: {len(self._indices)} subjects "
            f"from {Path(self.h5_path).name}" + (f" (split={split})" if split else "")
        )

    @property
    def subject_ids(self) -> list[str]:
        """Subject IDs for the active split/indices."""
        return [self._subject_ids[int(i)] for i in self._indices]

    def _get_h5(self):
        """Get or create a per-worker HDF5 file handle."""
        import h5py

        if not hasattr(self._local, "h5_file") or self._local.h5_file is None:
            self._local.h5_file = h5py.File(self.h5_path, "r")
        return self._local.h5_file

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        h5_idx = int(self._indices[idx])
        f = self._get_h5()

        # Read image and seg from H5 (already preprocessed to 192³)
        image = f["images"][h5_idx]  # [4, 192, 192, 192] float32
        seg = f["segs"][h5_idx]  # [1, 192, 192, 192] int8

        subject_id = self._subject_ids[h5_idx]

        # Build dict for MONAI transforms
        data = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "seg": torch.from_numpy(seg.astype(np.float32)),
        }

        # Apply transforms (normalization, crop, augmentation)
        if self.transform is not None:
            data = self.transform(data)

        output: dict[str, Any] = {
            "image": data["image"],
            "seg": data["seg"],
            "subject_id": subject_id,
        }

        # Load semantic features from H5
        if self.compute_semantic:
            volume = f["semantic/volume"][h5_idx]  # [4]
            location = f["semantic/location"][h5_idx]  # [3]
            shape = f["semantic/shape"][h5_idx]  # [3]
            all_feats = np.concatenate([volume, location, shape])

            output["semantic_features"] = {
                "volume": torch.from_numpy(volume.astype(np.float32)),
                "location": torch.from_numpy(location.astype(np.float32)),
                "shape": torch.from_numpy(shape.astype(np.float32)),
                "all": torch.from_numpy(all_feats.astype(np.float32)),
            }

        return output

    def __del__(self) -> None:
        """Close H5 file handle if open."""
        if hasattr(self, "_local") and hasattr(self._local, "h5_file"):
            if self._local.h5_file is not None:
                try:
                    self._local.h5_file.close()
                except Exception:
                    pass

    @staticmethod
    def load_splits_from_h5(h5_path: str | Path) -> dict[str, np.ndarray]:
        """Load all split index arrays from an H5 file.

        Args:
            h5_path: Path to the HDF5 file.

        Returns:
            Dict mapping split names to index arrays.
        """
        import h5py

        splits: dict[str, np.ndarray] = {}
        with h5py.File(str(h5_path), "r") as f:
            if "splits" in f:
                for name in f["splits"]:
                    splits[name] = f[f"splits/{name}"][:].astype(np.int64)
        return splits

    @staticmethod
    def load_subject_ids_from_h5(h5_path: str | Path) -> list[str]:
        """Load subject ID list from an H5 file.

        Args:
            h5_path: Path to the HDF5 file.

        Returns:
            List of subject ID strings.
        """
        import h5py

        with h5py.File(str(h5_path), "r") as f:
            return [s.decode() if isinstance(s, bytes) else s for s in f["subject_ids"][:]]


def create_dataloaders(
    data_root: str | Path | None = None,
    train_ids: list[str] | None = None,
    val_ids: list[str] | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    compute_semantic: bool = True,
    cache_semantic: bool = True,
    augment_train: bool = True,
    h5_path: str | Path | None = None,
    train_split: str = "lora_train",
    val_split: str = "lora_val",
    roi_size: tuple[int, int, int] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Supports two backends:
    - **NIfTI** (default): when ``data_root`` is set with ``train_ids``/``val_ids``.
    - **HDF5** (fast): when ``h5_path`` is set, uses pre-preprocessed volumes.

    Args:
        data_root: Path to BraTS-MEN directory (NIfTI backend).
        train_ids: List of subject IDs for training (NIfTI backend).
        val_ids: List of subject IDs for validation (NIfTI backend).
        batch_size: Batch size for both loaders.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        compute_semantic: Whether to compute/load semantic features.
        cache_semantic: Whether to cache semantic features (NIfTI only).
        augment_train: Whether to apply augmentation to training data.
        h5_path: Path to HDF5 file (H5 backend). Takes precedence over
            ``data_root`` when both are set.
        train_split: Split name for training in H5 file.
        val_split: Split name for validation in H5 file.
        roi_size: Override ROI size for training transforms.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if h5_path is not None:
        # ---- H5 backend ----
        from .transforms import DEFAULT_ROI_SIZE, FEATURE_ROI_SIZE

        train_roi = roi_size or DEFAULT_ROI_SIZE

        train_dataset = BraTSMENDatasetH5(
            h5_path=h5_path,
            split=train_split,
            transform=get_h5_train_transforms(roi_size=train_roi, augment=augment_train),
            compute_semantic=compute_semantic,
        )

        val_dataset = BraTSMENDatasetH5(
            h5_path=h5_path,
            split=val_split,
            transform=get_h5_val_transforms(roi_size=FEATURE_ROI_SIZE),
            compute_semantic=compute_semantic,
        )
    else:
        # ---- NIfTI backend (original) ----
        if data_root is None:
            raise ValueError("Either h5_path or data_root must be provided")

        train_dataset = BraTSMENDataset(
            data_root=data_root,
            subject_ids=train_ids,
            transform=(get_train_transforms() if augment_train else get_val_transforms()),
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
    subject_ids: list[str],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> dict[str, list[str]]:
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
        raise ValueError(f"Requested {total_needed} subjects but only {len(subject_ids)} available")

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


def save_splits(splits: dict[str, list[str]], path: str | Path) -> None:
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


def load_splits(path: str | Path) -> dict[str, list[str]]:
    """Load data splits from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Dict mapping split names to subject ID lists.
    """
    with open(path) as f:
        return json.load(f)


def split_subjects_multi(
    subject_ids: list[str],
    split_sizes: dict[str, int],
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split subjects into multiple non-overlapping sets.

    Generic n-way splitting function that supports any number of splits
    with configurable sizes.

    Args:
        subject_ids: List of all subject IDs.
        split_sizes: Dict mapping split names to sizes.
            Example: {"train": 200, "val": 100, "test": 500}
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping split names to subject ID lists.

    Raises:
        ValueError: If total requested size exceeds available subjects.

    Example:
        >>> all_ids = BraTSMENDataset.get_all_subject_ids(data_root)
        >>> splits = split_subjects_multi(
        ...     all_ids,
        ...     {"lora_train": 400, "lora_val": 100, "probe_train": 200, "test": 300},
        ...     seed=42,
        ... )
        >>> len(splits["lora_train"])
        400
    """
    total_needed = sum(split_sizes.values())
    if total_needed > len(subject_ids):
        raise ValueError(f"Requested {total_needed} subjects but only {len(subject_ids)} available")

    # Deterministic shuffle
    rng = np.random.RandomState(seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)

    # Split sequentially
    idx = 0
    splits = {}

    for split_name, size in split_sizes.items():
        splits[split_name] = shuffled[idx : idx + size]
        idx += size

    return splits
