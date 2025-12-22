"""MONAI transforms for 3D MRI preprocessing.

This module defines transform pipelines for train and validation data.
All transforms are dict-based and operate on keys: t1c, t1n, t2f, t2w, seg.
After transforms, the batch contains:
    - batch["image"]: shape [C=4, D=128, H=128, W=128]
    - batch["seg"]: shape [1, D=128, H=128, W=128]
"""

from typing import List, Tuple

from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd, 
    ConcatItemsd, 
    ToTensord
)
from monai.transforms.spatial.dictionary import (
    Orientationd,
    Spacingd,
)
from monai.transforms.croppad.dictionary import (
    ResizeWithPadOrCropd,
    RandSpatialCropd
)
from monai.transforms.intensity.dictionary import NormalizeIntensityd


MODALITY_KEYS = ["t1c", "t1n", "t2f", "t2w"]
SEG_KEY = "seg"
ALL_KEYS = MODALITY_KEYS + [SEG_KEY]



def get_common_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "RAS",
    roi_size: Tuple[int, int, int] = (128, 128, 128),
) -> List:
    """Get common transforms applied to both train and val data.

    These transforms enforce:
      (i) a shared atlas orientation,
      (ii) isotropic voxel spacing (mm/voxel),
      (iii) per-subject intensity normalization, and
      (iv) a deterministic 128³ spatial tensor (via pad/crop).

    Args:
        spacing: Target voxel spacing in mm (recommended for SRI24 whole-brain: (1.875, 1.875, 1.875)).
        orientation: Target orientation code (e.g., "RAS").
        roi_size: Target spatial size after pad/crop (default: 128³).

    Returns:
        List of MONAI transforms.
    """
    return [
        # 1. Load NIfTI files (keeps affine/metadata for spacing/orientation ops)
        LoadImaged(keys=ALL_KEYS, image_only=False),
        # 2. Ensure channel dimension is first for all keys
        EnsureChannelFirstd(keys=ALL_KEYS),
        # 3. Reorient to a standard axis code
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
        # 5. Z-score normalize each modality per subject over nonzero voxels
        NormalizeIntensityd(keys=MODALITY_KEYS, nonzero=True, channel_wise=True),
        # 6. Concatenate modalities into a single "image" tensor with C=4
        ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
        # 7. Deterministic pad/crop to exact target size (128×128×128)
        ResizeWithPadOrCropd(keys=["image", SEG_KEY], spatial_size=roi_size),
    ]


def get_train_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "RAS",
    roi_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    """Get training transforms.

    For Neural ODE compatibility, the training pipeline is geometrically deterministic:
    it produces a single, atlas-anchored whole-brain tensor (128³) per scan.

    If we later decide we want patch-based augmentation, reintroduce random crops
    *before* the final `ResizeWithPadOrCropd`, and be explicit that the ODE state then
    becomes crop-dependent.

    Args:
        spacing: Target voxel spacing in mm.
        orientation: Target orientation code.
        roi_size: Target spatial size.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = get_common_transforms(spacing, orientation, roi_size)

    transforms.append(ToTensord(keys=["image", SEG_KEY]))
    return Compose(transforms)


def get_val_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "RAS",
    roi_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    """Get validation transforms (deterministic only).

    Args:
        spacing: Target voxel spacing in mm.
        orientation: Target orientation code.
        roi_size: Target spatial size.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = get_common_transforms(spacing, orientation, roi_size)
    transforms.append(ToTensord(keys=["image", SEG_KEY]))
    return Compose(transforms)