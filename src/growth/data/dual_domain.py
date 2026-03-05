# src/growth/data/dual_domain.py
"""Dual-domain DataLoader factory for mixed-batch MEN + GLI training.

Creates a ConcatDataset + WeightedRandomSampler that yields approximately
balanced MEN/GLI batches from two HDF5 files.
"""

import logging

from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    WeightedRandomSampler,
)

from .bratsmendata import BraTSDatasetH5
from .transforms import (
    DEFAULT_ROI_SIZE,
    get_h5_train_transforms,
    get_h5_val_transforms,
)

logger = logging.getLogger(__name__)


def create_dual_domain_train_loader(
    men_h5_path: str,
    gli_h5_path: str,
    domain_ratio: float = 0.5,
    batch_size: int = 2,
    num_workers: int = 4,
    roi_size: tuple[int, int, int] = DEFAULT_ROI_SIZE,
    compute_semantic: bool = True,
    augment: bool = True,
    train_split: str = "lora_train",
    pin_memory: bool = True,
    seed: int = 0,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a mixed MEN + GLI training DataLoader.

    Uses ConcatDataset + WeightedRandomSampler to balance domain representation.

    Args:
        men_h5_path: Path to BraTS-MEN H5 file.
        gli_h5_path: Path to BraTS-GLI H5 file.
        domain_ratio: Target fraction of MEN samples per batch (0.5 = balanced).
        batch_size: Physical batch size.
        num_workers: Data loading workers.
        roi_size: Spatial crop size for training.
        compute_semantic: Whether to load semantic features.
        augment: Whether to apply data augmentation.
        train_split: Split name in H5 files.
        pin_memory: Pin memory for GPU transfer.
        seed: Random seed for sampler determinism.
        persistent_workers: Keep workers alive across epochs.

    Returns:
        DataLoader yielding mixed-domain batches.
    """
    transform = get_h5_train_transforms(roi_size=roi_size, augment=augment)

    men_dataset = BraTSDatasetH5(
        h5_path=men_h5_path,
        split=train_split,
        transform=transform,
        compute_semantic=compute_semantic,
    )
    gli_dataset = BraTSDatasetH5(
        h5_path=gli_h5_path,
        split=train_split,
        transform=transform,
        compute_semantic=compute_semantic,
    )

    n_men = len(men_dataset)
    n_gli = len(gli_dataset)
    combined = ConcatDataset([men_dataset, gli_dataset])

    # WeightedRandomSampler: inverse-domain-size weights for balance
    men_weight = 1.0 / n_men if n_men > 0 else 0.0
    gli_weight = 1.0 / n_gli if n_gli > 0 else 0.0

    # Scale by target ratio
    men_weight *= domain_ratio
    gli_weight *= 1.0 - domain_ratio

    weights = [men_weight] * n_men + [gli_weight] * n_gli
    num_samples = 2 * max(n_men, n_gli)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )
    logger.info(
        f"Dual-domain train loader: {n_men} MEN + {n_gli} GLI = "
        f"{len(combined)} total, {num_samples} samples/epoch, "
        f"ratio={domain_ratio:.2f}"
    )

    return DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def create_per_domain_val_loaders(
    men_h5_path: str,
    gli_h5_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    roi_size: tuple[int, int, int] = DEFAULT_ROI_SIZE,
    compute_semantic: bool = True,
    val_split: str = "lora_val",
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> dict[str, DataLoader]:
    """Create separate MEN and GLI validation DataLoaders.

    Args:
        men_h5_path: Path to BraTS-MEN H5 file.
        gli_h5_path: Path to BraTS-GLI H5 file.
        batch_size: Validation batch size.
        num_workers: Data loading workers.
        roi_size: Spatial size for validation.
        compute_semantic: Whether to load semantic features.
        val_split: Split name in H5 files.
        pin_memory: Pin memory for GPU transfer.
        persistent_workers: Keep workers alive across epochs.

    Returns:
        Dict with 'men' and 'gli' DataLoaders.
    """
    transform = get_h5_val_transforms(roi_size=roi_size)

    loaders = {}
    for name, h5_path in [("men", men_h5_path), ("gli", gli_h5_path)]:
        dataset = BraTSDatasetH5(
            h5_path=h5_path,
            split=val_split,
            transform=transform,
            compute_semantic=compute_semantic,
        )

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers and num_workers > 0,
        )
        logger.info(f"Validation loader [{name.upper()}]: {len(dataset)} samples")

    return loaders
