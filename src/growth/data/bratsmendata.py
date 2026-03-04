# src/growth/data/bratsmendata.py
"""BraTS dataset backed by HDF5 files.

All three datasets (BraTS-MEN, BraTS-GLI, MenGrowth) use the **unified v2.0
H5 schema**.  For cross-sectional data (MEN), the longitudinal structure is
trivial: each patient has exactly 1 scan at timepoint 0, so
``patient_offsets = [0, 1, ..., N]``.

Unified H5 v2.0 schema (all datasets):
    attrs:           {n_scans, n_patients, roi_size, spacing, channel_order,
                      version, dataset_type, domain}
    images           [N_scans, 4, 192, 192, 192] float32
    segs             [N_scans, 1, 192, 192, 192] int8
    scan_ids         [N_scans] string
    patient_ids      [N_scans] string
    timepoint_idx    [N_scans] int32
    semantic/        {volume [N,4], location [N,3], shape [N,3]}
    longitudinal/    {patient_offsets [N_patients+1], patient_list [N_patients]}
    splits/          {lora_train, lora_val, test}  (patient-level indices)
    metadata/        {grade, age, sex}

Legacy note: older MEN H5 files (v1.0) used ``subject_ids`` instead of
``scan_ids``.  The reader still supports this via auto-detection.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .transforms import (
    get_h5_train_transforms,
    get_h5_val_transforms,
)

logger = logging.getLogger(__name__)


class BraTSDatasetH5(Dataset):
    """BraTS dataset backed by a single HDF5 file.

    Reads pre-preprocessed 192^3 volumes from an H5 file. Volumes are already
    Orient -> Resample -> CropForeground -> SpatialPad -> CenterCrop but NOT
    z-score normalized (applied at runtime by H5 transforms).

    All datasets (MEN, GLI, MenGrowth) use the unified v2.0 schema with
    ``scan_ids``, ``patient_ids``, ``timepoint_idx``, and a ``longitudinal/``
    group.  For cross-sectional MEN data, the longitudinal CSR is trivial
    (1 scan per patient).

    Legacy v1.0 MEN files (``subject_ids`` only, no ``scan_ids``) are still
    supported via auto-detection.

    Args:
        h5_path: Path to the HDF5 file.
        split: Split name to load indices from (e.g., ``"lora_train"``).
            If None, uses all scans.
        indices: Explicit scan-level index array (overrides ``split``).
        transform: MONAI transform pipeline. If None, uses
            :func:`get_h5_val_transforms`.
        compute_semantic: If True, load semantic features from H5.

    Returns per sample:
        dict with:
        - ``'image'``: ``[4, D, H, W]`` MRI tensor
        - ``'seg'``: ``[1, D, H, W]`` segmentation mask
        - ``'subject_id'``: str (scan_id for longitudinal, subject_id for cross-sectional)
        - ``'domain'``: str (H5 attr ``domain``, default ``"MEN"``)
        - ``'patient_id'``: str (longitudinal only)
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
            # Detect schema: longitudinal vs cross-sectional
            self._is_longitudinal = "scan_ids" in f

            if self._is_longitudinal:
                n_total = f.attrs["n_scans"]
                self._scan_ids = [
                    s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]
                ]
                self._patient_ids = [
                    s.decode() if isinstance(s, bytes) else s for s in f["patient_ids"][:]
                ]
                self._patient_offsets = f["longitudinal/patient_offsets"][:].astype(np.int64)
                self._patient_list = [
                    s.decode() if isinstance(s, bytes) else s
                    for s in f["longitudinal/patient_list"][:]
                ]
                # subject_ids alias for compatibility
                self._subject_ids = self._scan_ids
            else:
                n_total = f.attrs["n_subjects"]
                self._subject_ids = [
                    s.decode() if isinstance(s, bytes) else s for s in f["subject_ids"][:]
                ]
                self._scan_ids = None
                self._patient_ids = None
                self._patient_offsets = None
                self._patient_list = None

            self._domain = f.attrs.get("domain", "MEN")
            if isinstance(self._domain, bytes):
                self._domain = self._domain.decode()

            # Determine indices
            if indices is not None:
                self._indices = np.asarray(indices, dtype=np.int64)
            elif split is not None:
                if f"splits/{split}" not in f:
                    available = list(f["splits"].keys()) if "splits" in f else []
                    raise KeyError(f"Split '{split}' not found in H5 file. Available: {available}")
                split_indices = f[f"splits/{split}"][:].astype(np.int64)

                if self._is_longitudinal:
                    # Split indices are patient-level; expand to scan indices via CSR
                    self._indices = self._expand_patient_indices(split_indices)
                else:
                    self._indices = split_indices
            else:
                self._indices = np.arange(n_total, dtype=np.int64)

        # Per-worker lazy file handle (thread-local for multi-worker safety)
        self._local = threading.local()

        # Transform pipeline
        if transform is None:
            self.transform = get_h5_val_transforms()
        else:
            self.transform = transform

        schema_type = "longitudinal" if self._is_longitudinal else "cross-sectional"
        logger.info(
            f"BraTSDatasetH5 initialized: {len(self._indices)} scans ({schema_type}, "
            f"domain={self._domain}) from {Path(self.h5_path).name}"
            + (f" (split={split})" if split else "")
        )

    def _expand_patient_indices(self, patient_indices: np.ndarray) -> np.ndarray:
        """Expand patient-level split indices to scan-level indices via CSR offsets.

        Args:
            patient_indices: Indices into ``patient_list``.

        Returns:
            Array of scan-level indices.
        """
        scan_indices = []
        for pi in patient_indices:
            start = self._patient_offsets[pi]
            end = self._patient_offsets[pi + 1]
            scan_indices.extend(range(start, end))
        return np.array(scan_indices, dtype=np.int64)

    @property
    def subject_ids(self) -> list[str]:
        """Subject/scan IDs for the active split/indices."""
        return [self._subject_ids[int(i)] for i in self._indices]

    @property
    def domain(self) -> str:
        """Domain identifier (e.g. 'MEN', 'GLI')."""
        return self._domain

    @property
    def is_longitudinal(self) -> bool:
        """Whether this H5 uses the longitudinal schema."""
        return self._is_longitudinal

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

        # Read image and seg from H5 (already preprocessed to 192^3)
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
            "domain": self._domain,
        }

        # Add longitudinal fields
        if self._is_longitudinal and self._patient_ids is not None:
            output["patient_id"] = self._patient_ids[h5_idx]

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
        """Load subject/scan ID list from an H5 file.

        For cross-sectional H5: returns ``subject_ids``.
        For longitudinal H5: returns ``scan_ids``.

        Args:
            h5_path: Path to the HDF5 file.

        Returns:
            List of ID strings.
        """
        import h5py

        with h5py.File(str(h5_path), "r") as f:
            key = "scan_ids" if "scan_ids" in f else "subject_ids"
            return [s.decode() if isinstance(s, bytes) else s for s in f[key][:]]


# Backward-compatibility alias
BraTSMENDatasetH5 = BraTSDatasetH5


def create_dataloaders(
    h5_path: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    compute_semantic: bool = True,
    augment_train: bool = True,
    train_split: str = "lora_train",
    val_split: str = "lora_val",
    roi_size: tuple[int, int, int] | None = None,
    val_roi_size: tuple[int, int, int] | None = None,
    val_batch_size: int | None = None,
    ddp_rank: int | None = None,
    ddp_world_size: int | None = None,
    persistent_workers: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from an H5 file.

    Args:
        h5_path: Path to HDF5 file (required).
        batch_size: Batch size for both loaders.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        compute_semantic: Whether to load semantic features.
        augment_train: Whether to apply augmentation to training data.
        train_split: Split name for training in H5 file.
        val_split: Split name for validation in H5 file.
        roi_size: Override ROI size for training transforms.
        val_roi_size: Override ROI size for validation transforms. If None,
            defaults to FEATURE_ROI_SIZE (192^3) for full tumor coverage.
        val_batch_size: Override batch size for validation loader. If None,
            defaults to ``batch_size``.
        ddp_rank: DDP rank (None for single-GPU).
        ddp_world_size: DDP world size (None for single-GPU).
        persistent_workers: Keep workers alive across epochs.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from .transforms import DEFAULT_ROI_SIZE, FEATURE_ROI_SIZE

    train_roi = roi_size or DEFAULT_ROI_SIZE
    val_roi = val_roi_size or FEATURE_ROI_SIZE

    train_dataset = BraTSDatasetH5(
        h5_path=h5_path,
        split=train_split,
        transform=get_h5_train_transforms(roi_size=train_roi, augment=augment_train),
        compute_semantic=compute_semantic,
    )

    val_dataset = BraTSDatasetH5(
        h5_path=h5_path,
        split=val_split,
        transform=get_h5_val_transforms(roi_size=val_roi),
        compute_semantic=compute_semantic,
    )

    # DDP samplers
    train_sampler = None
    val_sampler = None
    if ddp_rank is not None and ddp_world_size is not None:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
        )

    pw = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=pw,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size or batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=pw,
    )

    return train_loader, val_loader


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
        >>> splits = split_subjects_multi(
        ...     all_ids,
        ...     {"lora_train": 400, "lora_val": 100, "test": 300},
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


def save_splits(splits: dict[str, list[str]], path: str | Path) -> None:
    """Save data splits to JSON file.

    Args:
        splits: Dict mapping split names to subject ID lists.
        path: Output JSON file path.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Saved splits to {path}")
