# src/growth/data/transforms.py
"""MONAI transforms for MRI preprocessing.

Shared transforms for loading, resampling, normalization, and augmentation.
Matches BrainSegFounder's preprocessing pipeline:
  - Training: CropForeground + RandSpatialCrop(128^3) + augmentation
  - Validation: CropForeground + center crop to 128^3
  - Sliding window: no spatial crop (full volume for sliding_window_inference)
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ConcatItemsd,
    CropForegroundd,
    RandSpatialCropd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
)

logger = logging.getLogger(__name__)

# Default keys for BraTS data
MODALITY_KEYS: List[str] = ["t1c", "t1n", "t2f", "t2w"]
SEG_KEY: str = "seg"
IMAGE_KEY: str = "image"

# Default spatial settings matching BrainSegFounder fine-tuning
DEFAULT_ROI_SIZE: Tuple[int, int, int] = (128, 128, 128)
DEFAULT_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_ORIENTATION: str = "RAS"


def get_load_transforms(
    modality_keys: List[str] = None,
    include_seg: bool = True,
) -> List:
    """Get transforms for loading NIfTI files.

    Args:
        modality_keys: List of modality keys (default: t1c, t1n, t2f, t2w).
        include_seg: Whether to include segmentation mask.

    Returns:
        List of MONAI transforms for loading.
    """
    if modality_keys is None:
        modality_keys = MODALITY_KEYS.copy()

    keys = list(modality_keys)
    if include_seg:
        keys.append(SEG_KEY)

    return [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
    ]


def get_spatial_transforms(
    modality_keys: List[str] = None,
    include_seg: bool = True,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
    orientation: str = DEFAULT_ORIENTATION,
    roi_size: Tuple[int, int, int] = DEFAULT_ROI_SIZE,
    random_crop: bool = False,
) -> List:
    """Get spatial preprocessing transforms.

    Matches BrainSegFounder pipeline:
      1. Reorient to RAS
      2. Resample to 1mm isotropic
      3. CropForeground to brain bounding box (removes background/air)
      4. SpatialPad to ensure volume >= roi_size
      5. RandSpatialCrop (training) or center crop (validation) to roi_size

    Args:
        modality_keys: List of modality keys.
        include_seg: Whether to include segmentation.
        spacing: Target voxel spacing in mm (isotropic recommended).
        orientation: Target orientation (e.g., "RAS", "LPS").
        roi_size: Target spatial size (default: 128^3 matching BrainSegFounder).
        random_crop: If True, use RandSpatialCropd (training).
            If False, use ResizeWithPadOrCropd for deterministic center crop.

    Returns:
        List of MONAI spatial transforms.
    """
    if modality_keys is None:
        modality_keys = MODALITY_KEYS.copy()

    keys = list(modality_keys)
    seg_keys = [SEG_KEY] if include_seg else []
    all_keys = keys + seg_keys

    transforms = []

    # Reorient to standard orientation
    transforms.append(
        Orientationd(keys=all_keys, axcodes=orientation)
    )

    # Resample images to target spacing (bilinear interpolation)
    transforms.append(
        Spacingd(
            keys=modality_keys,
            pixdim=spacing,
            mode=["bilinear"] * len(modality_keys),
        )
    )

    # Resample segmentation with nearest neighbor
    if include_seg:
        transforms.append(
            Spacingd(
                keys=[SEG_KEY],
                pixdim=spacing,
                mode=("nearest",),
            )
        )

    # Crop to brain foreground (removes background/air around skull-stripped brain)
    transforms.append(
        CropForegroundd(keys=all_keys, source_key=modality_keys[0])
    )

    # Ensure volume is at least roi_size (pad with zeros if brain is smaller)
    transforms.append(
        SpatialPadd(keys=all_keys, spatial_size=roi_size)
    )

    if random_crop:
        # Training: random crop within brain volume
        transforms.append(
            RandSpatialCropd(
                keys=all_keys, roi_size=roi_size, random_size=False,
            )
        )
    else:
        # Validation: deterministic center crop within brain
        transforms.append(
            ResizeWithPadOrCropd(keys=all_keys, spatial_size=roi_size)
        )

    return transforms


def get_reorient_and_spacing_transforms(
    modality_keys: List[str] = None,
    include_seg: bool = True,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
    orientation: str = DEFAULT_ORIENTATION,
) -> List:
    """Get orientation + spacing transforms only (no crop).

    Used by sliding window transforms where no spatial cropping is applied.

    Args:
        modality_keys: List of modality keys.
        include_seg: Whether to include segmentation.
        spacing: Target voxel spacing in mm.
        orientation: Target orientation.

    Returns:
        List of MONAI transforms for reorientation and resampling.
    """
    if modality_keys is None:
        modality_keys = MODALITY_KEYS.copy()

    keys = list(modality_keys)
    seg_keys = [SEG_KEY] if include_seg else []
    all_keys = keys + seg_keys

    transforms = []

    transforms.append(
        Orientationd(keys=all_keys, axcodes=orientation)
    )

    transforms.append(
        Spacingd(
            keys=modality_keys,
            pixdim=spacing,
            mode=["bilinear"] * len(modality_keys),
        )
    )

    if include_seg:
        transforms.append(
            Spacingd(
                keys=[SEG_KEY],
                pixdim=spacing,
                mode=("nearest",),
            )
        )

    return transforms


def get_intensity_transforms(
    modality_keys: List[str] = None,
) -> List:
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


def get_concat_transforms(
    modality_keys: List[str] = None,
    output_key: str = IMAGE_KEY,
) -> List:
    """Get transforms to concatenate modalities into single tensor.

    Args:
        modality_keys: List of modality keys to concatenate.
        output_key: Key for concatenated output (default: "image").

    Returns:
        List with ConcatItemsd transform.
    """
    if modality_keys is None:
        modality_keys = MODALITY_KEYS.copy()

    return [
        ConcatItemsd(keys=modality_keys, name=output_key, dim=0),
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
) -> List:
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
            transforms.append(
                RandFlipd(keys=spatial_keys, prob=flip_prob, spatial_axis=axis)
            )

    # Random 90-degree rotations
    if include_rotate:
        transforms.append(
            RandRotate90d(keys=spatial_keys, prob=rotate_prob, max_k=3)
        )

    # Intensity augmentation (only on image, not segmentation)
    if include_intensity:
        transforms.append(
            RandScaleIntensityd(
                keys=[image_key],
                factors=intensity_scale,
                prob=0.5,
            )
        )
        transforms.append(
            RandShiftIntensityd(
                keys=[image_key],
                offsets=intensity_shift,
                prob=0.5,
            )
        )

    return transforms


def get_finalize_transforms(
    image_key: str = IMAGE_KEY,
    include_seg: bool = True,
    seg_key: str = SEG_KEY,
) -> List:
    """Get final transforms for tensor conversion.

    Args:
        image_key: Key for image tensor.
        include_seg: Whether segmentation is present.
        seg_key: Key for segmentation tensor.

    Returns:
        List with EnsureTyped/ToTensord transforms.
    """
    keys = [image_key]
    if include_seg:
        keys.append(seg_key)

    return [
        EnsureTyped(keys=keys, dtype="float32", track_meta=False),
    ]


def get_train_transforms(
    modality_keys: List[str] = None,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
    orientation: str = DEFAULT_ORIENTATION,
    roi_size: Tuple[int, int, int] = DEFAULT_ROI_SIZE,
    include_seg: bool = True,
    augment: bool = True,
) -> Compose:
    """Get complete training transform pipeline.

    Matches BrainSegFounder preprocessing:
      1. Load NIfTI files
      2. Ensure channel-first format
      3. Reorient to RAS
      4. Resample to 1mm isotropic
      5. CropForeground to brain bounding box
      6. SpatialPad to ensure volume >= roi_size
      7. RandSpatialCrop to roi_size (random position within brain)
      8. Z-score normalize intensities (nonzero voxels)
      9. Concatenate modalities into single tensor
      10. Apply augmentation (optional)
      11. Convert to PyTorch tensors

    Args:
        modality_keys: List of modality keys (default: t1c, t1n, t2f, t2w).
        spacing: Target voxel spacing (default: 1mm isotropic).
        orientation: Target orientation (default: RAS).
        roi_size: Target spatial size (default: 128^3).
        include_seg: Include segmentation mask in pipeline.
        augment: Apply data augmentation (for training).

    Returns:
        MONAI Compose transform pipeline.

    Example:
        >>> transforms = get_train_transforms(augment=True)
        >>> data = {"t1c": "path/to/t1c.nii.gz", ...}
        >>> result = transforms(data)
        >>> result["image"].shape  # [4, 128, 128, 128]
    """
    if modality_keys is None:
        modality_keys = MODALITY_KEYS.copy()

    transforms = []

    # Load and basic preprocessing
    transforms.extend(get_load_transforms(modality_keys, include_seg))

    # Spatial preprocessing (CropForeground + random crop)
    transforms.extend(get_spatial_transforms(
        modality_keys, include_seg, spacing, orientation, roi_size,
        random_crop=True,
    ))

    # Intensity normalization
    transforms.extend(get_intensity_transforms(modality_keys))

    # Concatenate modalities
    transforms.extend(get_concat_transforms(modality_keys))

    # Augmentation (training only)
    if augment:
        transforms.extend(get_augmentation_transforms(
            image_key=IMAGE_KEY,
            seg_key=SEG_KEY,
            include_seg=include_seg,
        ))

    # Final tensor conversion
    transforms.extend(get_finalize_transforms(IMAGE_KEY, include_seg))

    pipeline = Compose(transforms)

    logger.info(
        f"Created training transform pipeline: "
        f"roi_size={roi_size}, spacing={spacing}, orientation={orientation}, "
        f"include_seg={include_seg}, augment={augment}, random_crop=True, "
        f"{len(transforms)} transforms"
    )

    return pipeline


def get_val_transforms(
    modality_keys: List[str] = None,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
    orientation: str = DEFAULT_ORIENTATION,
    roi_size: Tuple[int, int, int] = DEFAULT_ROI_SIZE,
    include_seg: bool = True,
) -> Compose:
    """Get validation transform pipeline.

    Same spatial preprocessing as training (CropForeground + crop to roi_size)
    but uses deterministic center crop instead of random crop, and no
    augmentation.

    Args:
        modality_keys: List of modality keys.
        spacing: Target voxel spacing.
        orientation: Target orientation.
        roi_size: Target spatial size.
        include_seg: Include segmentation mask.

    Returns:
        MONAI Compose transform pipeline.
    """
    if modality_keys is None:
        modality_keys = MODALITY_KEYS.copy()

    transforms = []
    transforms.extend(get_load_transforms(modality_keys, include_seg))

    # Spatial preprocessing (CropForeground + deterministic center crop)
    transforms.extend(get_spatial_transforms(
        modality_keys, include_seg, spacing, orientation, roi_size,
        random_crop=False,
    ))

    transforms.extend(get_intensity_transforms(modality_keys))
    transforms.extend(get_concat_transforms(modality_keys))
    transforms.extend(get_finalize_transforms(IMAGE_KEY, include_seg))

    pipeline = Compose(transforms)

    logger.info(
        f"Created validation transform pipeline: "
        f"roi_size={roi_size}, spacing={spacing}, orientation={orientation}, "
        f"include_seg={include_seg}, augment=False, random_crop=False, "
        f"{len(transforms)} transforms"
    )

    return pipeline


def get_sliding_window_transforms(
    modality_keys: List[str] = None,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
    orientation: str = DEFAULT_ORIENTATION,
    include_seg: bool = True,
) -> Compose:
    """Get transforms for sliding window inference (no spatial crop).

    Applies only orientation, spacing, and normalization. The full-resolution
    volume is passed to monai.inferers.sliding_window_inference which handles
    patching internally. Use with batch_size=1 since volumes have variable
    spatial dimensions.

    This matches BrainSegFounder's inference-time preprocessing where the
    full brain volume is processed via sliding window with overlapping patches.

    Args:
        modality_keys: List of modality keys.
        spacing: Target voxel spacing.
        orientation: Target orientation.
        include_seg: Include segmentation mask.

    Returns:
        MONAI Compose transform pipeline.
    """
    if modality_keys is None:
        modality_keys = MODALITY_KEYS.copy()

    transforms = []
    transforms.extend(get_load_transforms(modality_keys, include_seg))

    # Orientation + spacing only (no crop)
    transforms.extend(get_reorient_and_spacing_transforms(
        modality_keys, include_seg, spacing, orientation,
    ))

    transforms.extend(get_intensity_transforms(modality_keys))
    transforms.extend(get_concat_transforms(modality_keys))
    transforms.extend(get_finalize_transforms(IMAGE_KEY, include_seg))

    pipeline = Compose(transforms)

    logger.info(
        f"Created sliding window transform pipeline: "
        f"spacing={spacing}, orientation={orientation}, "
        f"include_seg={include_seg}, no_crop=True, "
        f"{len(transforms)} transforms"
    )

    return pipeline


def get_inference_transforms(
    modality_keys: List[str] = None,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
    orientation: str = DEFAULT_ORIENTATION,
    roi_size: Tuple[int, int, int] = DEFAULT_ROI_SIZE,
) -> Compose:
    """Get inference-only transform pipeline (no segmentation).

    Uses CropForeground + center crop for fixed-size batched inference.
    For sliding window inference on full volumes, use
    get_sliding_window_transforms() instead.

    Args:
        modality_keys: List of modality keys.
        spacing: Target voxel spacing.
        orientation: Target orientation.
        roi_size: Target spatial size.

    Returns:
        MONAI Compose transform pipeline.
    """
    if modality_keys is None:
        modality_keys = MODALITY_KEYS.copy()

    transforms = []
    transforms.extend(get_load_transforms(modality_keys, include_seg=False))

    transforms.extend(get_spatial_transforms(
        modality_keys, include_seg=False, spacing=spacing,
        orientation=orientation, roi_size=roi_size,
        random_crop=False,
    ))

    transforms.extend(get_intensity_transforms(modality_keys))
    transforms.extend(get_concat_transforms(modality_keys))
    transforms.extend(get_finalize_transforms(IMAGE_KEY, include_seg=False))

    pipeline = Compose(transforms)

    logger.info(
        f"Created inference transform pipeline: "
        f"roi_size={roi_size}, spacing={spacing}, orientation={orientation}, "
        f"include_seg=False, {len(transforms)} transforms"
    )

    return pipeline
