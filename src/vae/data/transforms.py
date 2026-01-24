"""MONAI transforms for 3D MRI preprocessing.

This module defines transform pipelines for train and validation data.
All transforms are dict-based and operate on keys: t1c, t1n, t2f, t2w, seg.
After transforms, the batch contains:
    - batch["image"]: shape [C=4, D=128, H=128, W=128]
    - batch["seg"]: shape [1, D=128, H=128, W=128]
    - batch["semantic_features"]: dict of feature tensors (if extract_semantic=True)
"""

from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np
import torch
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
from monai.transforms import MapTransform

from .semantic_features import (
    extract_semantic_features,
    get_feature_groups,
    features_to_tensor,
    residualize_shape_features,
    SemanticFeatureNormalizer,
)


SEG_KEY = "seg"


class ExtractSemanticFeaturesd(MapTransform):
    """Extract semantic features from segmentation mask.

    This transform extracts volume, location, and shape features from
    the segmentation mask for semi-supervised VAE training.

    The features are stored in a nested dictionary under "semantic_features"
    with keys matching the latent partition names (z_vol, z_loc, z_shape).
    """

    def __init__(
        self,
        seg_key: str = "seg",
        spacing: Tuple[float, float, float] = (1.875, 1.875, 1.875),
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        seg_labels: Optional[Dict[str, int]] = None,
        compute_shape: bool = True,
        normalizer: Optional[SemanticFeatureNormalizer] = None,
        residualize_shape: bool = False,
        residual_params_path: Optional[str] = None,
    ):
        """Initialize the transform.

        Args:
            seg_key: Key for segmentation in data dict
            spacing: Voxel spacing in mm
            roi_size: ROI dimensions for coordinate normalization
            seg_labels: Segmentation label mapping
            compute_shape: Whether to compute shape descriptors
            normalizer: Optional normalizer for z-score standardization
            residualize_shape: Whether to remove volume-predictable component
                from shape features before normalization
            residual_params_path: Path to JSON with OLS coefficients for
                shape residualization (required if residualize_shape=True)
        """
        super().__init__(keys=[seg_key])
        self.seg_key = seg_key
        self.spacing = spacing
        self.roi_size = roi_size
        self.seg_labels = seg_labels
        self.compute_shape = compute_shape
        self.normalizer = normalizer
        self.residualize_shape = residualize_shape

        # Load residualization parameters if configured
        self.residual_params: Optional[Dict[str, Dict[str, float]]] = None
        if residualize_shape and residual_params_path:
            with open(residual_params_path) as f:
                self.residual_params = json.load(f)

        # Get feature groups for partitioning
        self.feature_groups = get_feature_groups()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic features from segmentation.

        Args:
            data: Data dictionary with segmentation

        Returns:
            Data dictionary with added "semantic_features" key
        """
        d = dict(data)

        seg = d[self.seg_key]

        # Handle tensor or numpy array
        if isinstance(seg, torch.Tensor):
            seg_np = seg.cpu().numpy()
        else:
            seg_np = np.asarray(seg)

        # Extract features
        features = extract_semantic_features(
            seg=seg_np,
            spacing=self.spacing,
            roi_size=self.roi_size,
            seg_labels=self.seg_labels,
            compute_shape=self.compute_shape,
        )

        # Residualize shape features (remove volume-predictable component)
        if self.residualize_shape and self.residual_params is not None:
            features = residualize_shape_features(
                features, self.residual_params, volume_feature="vol_total"
            )

        # Apply normalization if available
        if self.normalizer is not None:
            features = self.normalizer.transform(features)

        # Organize features by partition
        semantic_features = {}

        # Volume features -> z_vol
        vol_features = [features.get(k, 0.0) for k in self.feature_groups["volume"]]
        semantic_features["z_vol"] = torch.tensor(vol_features, dtype=torch.float32)

        # Location features -> z_loc
        loc_features = [features.get(k, 0.0) for k in self.feature_groups["location"]]
        semantic_features["z_loc"] = torch.tensor(loc_features, dtype=torch.float32)

        # Shape features -> z_shape
        shape_features = [features.get(k, 0.0) for k in self.feature_groups["shape"]]
        semantic_features["z_shape"] = torch.tensor(shape_features, dtype=torch.float32)

        d["semantic_features"] = semantic_features

        return d


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
    extract_semantic: bool = False,
    seg_labels: Optional[Dict[str, int]] = None,
    semantic_normalizer: Optional[SemanticFeatureNormalizer] = None,
    residualize_shape: bool = False,
    residual_params_path: Optional[str] = None,
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
        extract_semantic: Whether to extract semantic features for semi-supervised VAE.
        seg_labels: Segmentation label mapping for semantic extraction.
        semantic_normalizer: Optional normalizer for semantic features.
        residualize_shape: Whether to residualize shape features against volume.
        residual_params_path: Path to JSON with OLS coefficients.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = get_common_transforms(modality_keys, spacing, orientation, roi_size)

    # Add semantic feature extraction before tensor conversion (if enabled)
    if extract_semantic:
        transforms.append(
            ExtractSemanticFeaturesd(
                seg_key=SEG_KEY,
                spacing=spacing,
                roi_size=roi_size,
                seg_labels=seg_labels,
                compute_shape=True,
                normalizer=semantic_normalizer,
                residualize_shape=residualize_shape,
                residual_params_path=residual_params_path,
            )
        )

    transforms.append(ToTensord(keys=["image", SEG_KEY]))
    return Compose(transforms)


def get_val_transforms(
    modality_keys: List[str],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "RAS",
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    extract_semantic: bool = False,
    seg_labels: Optional[Dict[str, int]] = None,
    semantic_normalizer: Optional[SemanticFeatureNormalizer] = None,
    residualize_shape: bool = False,
    residual_params_path: Optional[str] = None,
) -> Compose:
    """Get validation transforms (deterministic only).

    Args:
        modality_keys: List of modality key strings.
        spacing: Target voxel spacing in mm.
        orientation: Target orientation code.
        roi_size: Target spatial size.
        extract_semantic: Whether to extract semantic features for semi-supervised VAE.
        seg_labels: Segmentation label mapping for semantic extraction.
        semantic_normalizer: Optional normalizer for semantic features.
        residualize_shape: Whether to residualize shape features against volume.
        residual_params_path: Path to JSON with OLS coefficients.

    Returns:
        MONAI Compose transform pipeline.
    """
    transforms = get_common_transforms(modality_keys, spacing, orientation, roi_size)

    # Add semantic feature extraction before tensor conversion (if enabled)
    if extract_semantic:
        transforms.append(
            ExtractSemanticFeaturesd(
                seg_key=SEG_KEY,
                spacing=spacing,
                roi_size=roi_size,
                seg_labels=seg_labels,
                compute_shape=True,
                normalizer=semantic_normalizer,
                residualize_shape=residualize_shape,
                residual_params_path=residual_params_path,
            )
        )

    transforms.append(ToTensord(keys=["image", SEG_KEY]))
    return Compose(transforms)