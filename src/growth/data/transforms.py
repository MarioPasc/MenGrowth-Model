# src/growth/data/transforms.py
"""MONAI transforms for MRI preprocessing.

Provides H5-based transforms for pre-preprocessed 192^3 volumes:
  - Training: z-score normalize -> RandSpatialCrop(128^3) -> augmentation
  - Validation: z-score normalize -> (optional center crop)

Also exports shared constants and reusable sub-pipelines (intensity,
augmentation, finalize) used by both the H5 dataset and standalone
conversion scripts.
"""

import logging

from monai.transforms import (
    Compose,
    EnsureTyped,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
)

logger = logging.getLogger(__name__)

# Default keys for BraTS data
# Channel order matches BrainSegFounder training: [FLAIR, T1ce, T1, T2]
# This order is critical - wrong order causes near-zero Dice scores.
MODALITY_KEYS: list[str] = ["t2f", "t1c", "t1n", "t2w"]
SEG_KEY: str = "seg"
IMAGE_KEY: str = "image"

# Default spatial settings matching BrainSegFounder fine-tuning
# 128^3 for LoRA training (matches BrainSegFounder fine-tuning)
DEFAULT_ROI_SIZE: tuple[int, int, int] = (128, 128, 128)
# 192^3 for feature extraction - guarantees 100% tumor containment
# (128^3 center crop only captures 38.8% MEN / 30.0% GLI tumors fully)
FEATURE_ROI_SIZE: tuple[int, int, int] = (192, 192, 192)
DEFAULT_SPACING: tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_ORIENTATION: str = "RAS"


def get_intensity_transforms(
    modality_keys: list[str] | None = None,
) -> list:
    """Get intensity normalization transforms.

    Applies z-score normalization per modality, using only
    nonzero voxels (standard for brain MRI with skull-stripped data).

    Args:
        modality_keys: List of modality keys.

    Returns:
        List of MONAI intensity transforms.
    """
    if modality_keys is None:
        modality_keys = MODALITY_KEYS.copy()

    return [
        NormalizeIntensityd(
            keys=modality_keys,
            nonzero=True,
            channel_wise=True,
        ),
    ]


def get_augmentation_transforms(
    image_key: str = IMAGE_KEY,
    seg_key: str = SEG_KEY,
    include_seg: bool = True,
    include_flip: bool = True,
    include_rotate: bool = True,
    include_intensity: bool = True,
    flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    intensity_scale: float = 0.1,
    intensity_shift: float = 0.1,
) -> list:
    """Get data augmentation transforms.

    Args:
        image_key: Key for image tensor.
        seg_key: Key for segmentation tensor.
        include_seg: Whether segmentation is present.
        include_flip: Include random flipping along each axis.
        include_rotate: Include random 90-degree rotations.
        include_intensity: Include intensity augmentation (image only).
        flip_prob: Probability for each flip.
        rotate_prob: Probability for rotation.
        intensity_scale: Scale factor range for intensity scaling.
        intensity_shift: Offset range for intensity shifting.

    Returns:
        List of MONAI augmentation transforms.
    """
    transforms = []

    # Keys to apply spatial augmentation to
    spatial_keys = [image_key]
    if include_seg:
        spatial_keys.append(seg_key)

    # Random flipping along each spatial axis
    if include_flip:
        for axis in [0, 1, 2]:  # D, H, W
            transforms.append(RandFlipd(keys=spatial_keys, prob=flip_prob, spatial_axis=axis))

    # Random 90-degree rotations (not in BrainSegFounder's original pipeline,
    # which only uses flips; kept as additional augmentation for domain adaptation)
    if include_rotate:
        transforms.append(RandRotate90d(keys=spatial_keys, prob=rotate_prob, max_k=3))

    # Intensity augmentation (only on image, not segmentation)
    # prob=1.0 matches BrainSegFounder's data_utils.py (lines 132-133)
    if include_intensity:
        transforms.append(
            RandScaleIntensityd(
                keys=[image_key],
                factors=intensity_scale,
                prob=1.0,
            )
        )
        transforms.append(
            RandShiftIntensityd(
                keys=[image_key],
                offsets=intensity_shift,
                prob=1.0,
            )
        )

    return transforms


def get_finalize_transforms(
    image_key: str = IMAGE_KEY,
    include_seg: bool = True,
    seg_key: str = SEG_KEY,
) -> list:
    """Get final transforms for tensor conversion.

    Args:
        image_key: Key for image tensor.
        include_seg: Whether segmentation is present.
        seg_key: Key for segmentation tensor.

    Returns:
        List with EnsureTyped transforms.
    """
    keys = [image_key]
    if include_seg:
        keys.append(seg_key)

    return [
        EnsureTyped(keys=keys, dtype="float32", track_meta=False),
    ]


def get_h5_train_transforms(
    roi_size: tuple[int, int, int] = DEFAULT_ROI_SIZE,
    augment: bool = True,
) -> Compose:
    """Get training transforms for pre-preprocessed H5 data.

    H5 volumes are already Orient -> Resample -> CropForeground -> SpatialPad -> CenterCrop
    at 192^3 but NOT z-score normalized. This pipeline applies:
      1. Z-score normalization (nonzero, channel-wise)
      2. RandSpatialCrop to roi_size (e.g., 128^3 for LoRA training)
      3. Augmentation (optional)
      4. EnsureType

    Input dict: {"image": [4, 192, 192, 192], "seg": [1, 192, 192, 192]}.

    Args:
        roi_size: Target spatial size for random crop.
        augment: Whether to apply data augmentation.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = []

    # Z-score normalize (nonzero voxels, per-channel)
    transforms.append(
        NormalizeIntensityd(
            keys=[IMAGE_KEY],
            nonzero=True,
            channel_wise=True,
        )
    )

    # Random crop to target size (e.g., 192->128 for LoRA training)
    spatial_keys = [IMAGE_KEY, SEG_KEY]
    if tuple(roi_size) != FEATURE_ROI_SIZE:
        transforms.append(
            RandSpatialCropd(
                keys=spatial_keys,
                roi_size=roi_size,
                random_size=False,
            )
        )

    # Augmentation
    if augment:
        transforms.extend(
            get_augmentation_transforms(
                image_key=IMAGE_KEY,
                seg_key=SEG_KEY,
                include_seg=True,
            )
        )

    # Final tensor conversion
    transforms.extend(get_finalize_transforms(IMAGE_KEY, include_seg=True))

    pipeline = Compose(transforms)

    logger.info(
        f"Created H5 training transform pipeline: "
        f"roi_size={roi_size}, augment={augment}, {len(transforms)} transforms"
    )

    return pipeline


def get_h5_val_transforms(
    roi_size: tuple[int, int, int] = FEATURE_ROI_SIZE,
) -> Compose:
    """Get validation transforms for pre-preprocessed H5 data.

    H5 volumes are already at 192^3. This pipeline applies:
      1. Z-score normalization (nonzero, channel-wise)
      2. ResizeWithPadOrCrop to roi_size (no-op when roi_size=192^3)
      3. EnsureType

    Input dict: {"image": [4, 192, 192, 192], "seg": [1, 192, 192, 192]}.

    Args:
        roi_size: Target spatial size (192^3 = no-op for feature extraction).

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = []

    # Z-score normalize (nonzero voxels, per-channel)
    transforms.append(
        NormalizeIntensityd(
            keys=[IMAGE_KEY],
            nonzero=True,
            channel_wise=True,
        )
    )

    # Center crop/pad to target size (no-op when roi_size matches H5 volume)
    spatial_keys = [IMAGE_KEY, SEG_KEY]
    if tuple(roi_size) != FEATURE_ROI_SIZE:
        transforms.append(ResizeWithPadOrCropd(keys=spatial_keys, spatial_size=roi_size))

    # Final tensor conversion
    transforms.extend(get_finalize_transforms(IMAGE_KEY, include_seg=True))

    pipeline = Compose(transforms)

    logger.info(
        f"Created H5 validation transform pipeline: "
        f"roi_size={roi_size}, {len(transforms)} transforms"
    )

    return pipeline
