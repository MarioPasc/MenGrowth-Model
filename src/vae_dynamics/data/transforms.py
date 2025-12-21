"""MONAI transforms for 3D MRI preprocessing.

This module defines transform pipelines for train and validation data.
All transforms are dict-based and operate on keys: t1c, t1n, t2f, t2w, seg.
After transforms, the batch contains:
    - batch["image"]: shape [C=4, D=128, H=128, W=128]
    - batch["seg"]: shape [1, D=128, H=128, W=128]
"""

from typing import List, Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ConcatItemsd,
    ResizeWithPadOrCropd,
    RandSpatialCropd,
    ToTensord,
)


MODALITY_KEYS = ["t1c", "t1n", "t2f", "t2w"]
SEG_KEY = "seg"
ALL_KEYS = MODALITY_KEYS + [SEG_KEY]


def get_common_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "RAS",
    roi_size: Tuple[int, int, int] = (128, 128, 128),
) -> List:
    """Get common transforms applied to both train and val data.

    These transforms ensure deterministic geometry and intensity standardization.

    Args:
        spacing: Target voxel spacing in mm.
        orientation: Target orientation code (e.g., "RAS").
        roi_size: Target spatial size after resize/crop.

    Returns:
        List of MONAI transforms.
    """
    return [
        # 1. Load NIfTI files
        LoadImaged(keys=ALL_KEYS, image_only=False),
        # 2. Ensure channel dimension is first
        EnsureChannelFirstd(keys=ALL_KEYS),
        # 3. Reorient to standard orientation
        Orientationd(keys=ALL_KEYS, axcodes=orientation),
        # 4a. Resample modalities with bilinear interpolation
        Spacingd(
            keys=MODALITY_KEYS,
            pixdim=spacing,
            mode=("bilinear", "bilinear", "bilinear", "bilinear"),
        ),
        # 4b. Resample segmentation with nearest-neighbor interpolation
        Spacingd(
            keys=[SEG_KEY],
            pixdim=spacing,
            mode=("nearest",),
        ),
        # 5. Z-score normalize each modality channel-wise per subject
        NormalizeIntensityd(keys=MODALITY_KEYS, nonzero=False, channel_wise=True),
        # 6. Concatenate modalities into single "image" tensor with C=4
        ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
        # 7. Resize/pad/crop to exact target size (128x128x128)
        ResizeWithPadOrCropd(keys=["image", SEG_KEY], spatial_size=roi_size),
    ]


def get_train_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "RAS",
    roi_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    """Get training transforms including stochastic augmentation.

    The train pipeline includes random spatial cropping after size guarantee.
    Even though crop size equals target size, it randomizes location if volumes
    are larger before the resize step.

    Args:
        spacing: Target voxel spacing in mm.
        orientation: Target orientation code.
        roi_size: Target spatial size.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = get_common_transforms(spacing, orientation, roi_size)

    # Add train-only stochastic transforms
    transforms.extend([
        # 8. Random spatial crop (same size, randomizes location)
        RandSpatialCropd(keys=["image", SEG_KEY], roi_size=roi_size, random_size=False),
        # 9. Convert to PyTorch tensors
        ToTensord(keys=["image", SEG_KEY]),
    ])

    return Compose(transforms)


def get_val_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "RAS",
    roi_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    """Get validation transforms (deterministic only).

    The validation pipeline contains no random transforms to ensure
    reproducible evaluation.

    Args:
        spacing: Target voxel spacing in mm.
        orientation: Target orientation code.
        roi_size: Target spatial size.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = get_common_transforms(spacing, orientation, roi_size)

    # Val: only deterministic transforms, no random crop
    transforms.append(
        ToTensord(keys=["image", SEG_KEY]),
    )

    return Compose(transforms)
