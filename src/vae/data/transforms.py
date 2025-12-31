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
    ToTensord,
)
from monai.transforms.spatial.dictionary import (
    Orientationd,
    Spacingd,
)
from monai.transforms.croppad.dictionary import (
    ResizeWithPadOrCropd,
    RandSpatialCropd,
)
from monai.transforms.intensity.dictionary import NormalizeIntensityd


SEG_KEY = "seg"


def get_common_transforms(
    modality_keys: List[str],
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
        modality_keys: List of modality key strings (e.g., ["t1c", "t1n"]).
        spacing: Target voxel spacing in mm (recommended for SRI24 whole-brain: (1.875, 1.875, 1.875)).
        orientation: Target orientation code (e.g., "RAS").
        roi_size: Target spatial size after pad/crop (default: 128³).

    Returns:
        List of MONAI transforms.
    """
    all_keys = modality_keys + [SEG_KEY]

    return [
        # 1. Load NIfTI files (keeps affine/metadata for spacing/orientation ops)
        LoadImaged(keys=all_keys, image_only=False),
        # 2. Ensure channel dimension is first for all keys
        EnsureChannelFirstd(keys=all_keys),
        # 3. Reorient to a standard axis code
        Orientationd(keys=all_keys, axcodes=orientation),
        # 4a. Resample modalities with bilinear interpolation
        Spacingd(
            keys=modality_keys,
            pixdim=spacing,
            mode=["bilinear"] * len(modality_keys),
        ),
        # 4b. Resample segmentation with nearest-neighbor interpolation
        Spacingd(
            keys=[SEG_KEY],
            pixdim=spacing,
            mode=("nearest",),
        ),
        # 5. Z-score normalize each modality per subject over nonzero voxels
        NormalizeIntensityd(keys=modality_keys, nonzero=True, channel_wise=True),
        # 6. Concatenate modalities into a single "image" tensor with C=len(modality_keys)
        ConcatItemsd(keys=modality_keys, name="image", dim=0),
        # 7. Deterministic pad/crop to exact target size (128×128×128)
        ResizeWithPadOrCropd(keys=["image", SEG_KEY], spatial_size=roi_size),
    ]


def get_train_transforms(
    modality_keys: List[str],
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
        modality_keys: List of modality key strings.
        spacing: Target voxel spacing in mm.
        orientation: Target orientation code.
        roi_size: Target spatial size.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = get_common_transforms(modality_keys, spacing, orientation, roi_size)

    transforms.append(ToTensord(keys=["image", SEG_KEY]))
    return Compose(transforms)


def get_val_transforms(
    modality_keys: List[str],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "RAS",
    roi_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    """Get validation transforms (deterministic only).

    Args:
        modality_keys: List of modality key strings.
        spacing: Target voxel spacing in mm.
        orientation: Target orientation code.
        roi_size: Target spatial size.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = get_common_transforms(modality_keys, spacing, orientation, roi_size)
    transforms.append(ToTensord(keys=["image", SEG_KEY]))
    return Compose(transforms)