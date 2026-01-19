"""Semantic feature extraction from segmentation masks.

This module extracts interpretable features from 3D segmentation masks for
semi-supervised VAE training. Features are designed to be meaningful for
downstream Neural ODE tumor growth prediction.

Features extracted:
- Volume: Total and per-label tumor volumes (log-scaled)
- Location: Tumor centroid in normalized coordinates
- Shape: Morphological descriptors (sphericity, surface area, aspect ratios)

References:
- BraTS segmentation labels: NCR=1, ED=2, ET=3
- Gompertz model: dV/dt = αV ln(K/V) requires volume as primary feature
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import torch
from scipy import ndimage
from skimage import measure

logger = logging.getLogger(__name__)

# Default BraTS segmentation labels
DEFAULT_SEG_LABELS = {
    "ncr": 1,  # Necrotic core
    "ed": 2,   # Peritumoral edema
    "et": 3,   # Enhancing tumor
}


def validate_semantic_features(
    features: Dict[str, float],
    fix_invalid: bool = True,
    log_warnings: bool = True,
) -> Tuple[Dict[str, float], bool]:
    """Validate semantic features for NaN/Inf values.

    Args:
        features: Dictionary of semantic features
        fix_invalid: If True, replace NaN/Inf with safe defaults
        log_warnings: If True, log warnings for invalid values

    Returns:
        Tuple of (validated_features, had_invalid) where had_invalid is True
        if any NaN/Inf values were found
    """
    had_invalid = False
    validated = {}

    for name, value in features.items():
        if not np.isfinite(value):
            had_invalid = True
            if log_warnings:
                logger.warning(
                    f"Invalid value in semantic feature '{name}': {value}"
                )

            if fix_invalid:
                # Use safe defaults based on feature type
                if name.startswith("vol_"):
                    # Log volume: log(1) = 0 for empty/invalid
                    validated[name] = 0.0
                elif name.startswith("loc_"):
                    # Location: center of ROI
                    validated[name] = 0.5
                elif name.startswith("sphericity_") or name.startswith("solidity_"):
                    # Sphericity/solidity: 0 for invalid
                    validated[name] = 0.0
                elif name.startswith("surface_area_"):
                    # Log surface area: log(1) = 0
                    validated[name] = 0.0
                elif name.startswith("aspect_"):
                    # Aspect ratio: 1.0 (no distortion)
                    validated[name] = 1.0
                else:
                    # Unknown: default to 0
                    validated[name] = 0.0
            else:
                validated[name] = value
        else:
            validated[name] = value

    return validated, had_invalid


def extract_semantic_features(
    seg: Union[np.ndarray, torch.Tensor],
    spacing: Tuple[float, float, float] = (1.875, 1.875, 1.875),
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    seg_labels: Optional[Dict[str, int]] = None,
    compute_shape: bool = True,
) -> Dict[str, float]:
    """Extract semantic features from a segmentation mask.

    Args:
        seg: Segmentation mask of shape [D, H, W] or [1, D, H, W].
             Integer labels: 0=background, 1=NCR, 2=ED, 3=ET
        spacing: Voxel spacing in mm (D, H, W)
        roi_size: ROI dimensions for coordinate normalization
        seg_labels: Label mapping, defaults to BraTS convention
        compute_shape: Whether to compute shape descriptors (slower)

    Returns:
        Dictionary of semantic features with keys:
        - vol_total, vol_ncr, vol_ed, vol_et: Log-scaled volumes
        - loc_x, loc_y, loc_z: Normalized centroid coordinates [0, 1]
        - sphericity_*, surface_area_*, aspect_*: Shape descriptors
    """
    if seg_labels is None:
        seg_labels = DEFAULT_SEG_LABELS

    # Convert to numpy if tensor
    if isinstance(seg, torch.Tensor):
        seg = seg.cpu().numpy()

    # Remove channel dimension if present
    if seg.ndim == 4:
        seg = seg[0]

    # Ensure integer type for label comparison
    seg = seg.astype(np.int32)

    features = {}

    # Voxel volume in mm³
    voxel_vol = spacing[0] * spacing[1] * spacing[2]

    # ==========================================================================
    # VOLUME FEATURES (log-scaled for numerical stability)
    # ==========================================================================

    # Total tumor (union of all labels > 0)
    tumor_mask = seg > 0
    vol_total = np.sum(tumor_mask) * voxel_vol
    features["vol_total"] = np.log(vol_total + 1.0)  # +1 to handle empty masks

    # Per-label volumes
    for label_name, label_val in seg_labels.items():
        label_mask = seg == label_val
        vol_label = np.sum(label_mask) * voxel_vol
        features[f"vol_{label_name}"] = np.log(vol_label + 1.0)

    # ==========================================================================
    # LOCATION FEATURES (normalized centroid)
    # ==========================================================================

    if np.any(tumor_mask):
        # Compute centroid using center of mass
        centroid = ndimage.center_of_mass(tumor_mask)

        # Normalize to [0, 1] based on ROI size
        features["loc_x"] = centroid[2] / roi_size[2]  # Width (last dim)
        features["loc_y"] = centroid[1] / roi_size[1]  # Height
        features["loc_z"] = centroid[0] / roi_size[0]  # Depth (first dim)
    else:
        # No tumor: place at center
        features["loc_x"] = 0.5
        features["loc_y"] = 0.5
        features["loc_z"] = 0.5

    # ==========================================================================
    # SHAPE FEATURES (optional, computationally expensive)
    # ==========================================================================

    if compute_shape:
        # Total tumor shape
        shape_total = _compute_shape_features(tumor_mask, spacing, "total")
        features.update(shape_total)

        # Per-label shape (only for NCR and ET, ED is often too diffuse)
        for label_name in ["ncr", "et"]:
            label_val = seg_labels.get(label_name)
            if label_val is not None:
                label_mask = seg == label_val
                if np.sum(label_mask) > 10:  # Minimum voxels for shape
                    shape_label = _compute_shape_features(
                        label_mask, spacing, label_name
                    )
                    features.update(shape_label)
                else:
                    # Set default values for missing shapes
                    features[f"sphericity_{label_name}"] = 0.0
                    features[f"surface_area_{label_name}"] = 0.0
                    features[f"aspect_xy_{label_name}"] = 1.0
                    features[f"aspect_xz_{label_name}"] = 1.0

    # Validate all features for NaN/Inf and fix if necessary
    features, had_invalid = validate_semantic_features(features, fix_invalid=True)
    if had_invalid:
        logger.warning("Some semantic features had invalid values and were fixed")

    return features


def _compute_shape_features(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    suffix: str,
) -> Dict[str, float]:
    """Compute shape descriptors for a binary mask.

    Args:
        mask: Binary 3D mask
        spacing: Voxel spacing in mm
        suffix: Suffix for feature names (e.g., "total", "ncr")

    Returns:
        Dictionary with shape features
    """
    features = {}

    if np.sum(mask) == 0:
        # Empty mask: return defaults
        features[f"sphericity_{suffix}"] = 0.0
        features[f"surface_area_{suffix}"] = 0.0
        features[f"aspect_xy_{suffix}"] = 1.0
        features[f"aspect_xz_{suffix}"] = 1.0
        features[f"solidity_{suffix}"] = 0.0
        return features

    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    volume = np.sum(mask) * voxel_vol

    # Surface area via marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask.astype(np.float32),
            level=0.5,
            spacing=spacing,
        )
        surface_area = measure.mesh_surface_area(verts, faces)
    except (ValueError, RuntimeError):
        # Marching cubes can fail on very small or irregular masks
        surface_area = 0.0

    features[f"surface_area_{suffix}"] = np.log(surface_area + 1.0)

    # Sphericity: ratio of surface area to that of equal-volume sphere
    # sphericity = (36π V²)^(1/3) / A, where A is surface area
    if surface_area > 0:
        sphere_factor = (36 * np.pi * volume**2) ** (1/3)
        sphericity = sphere_factor / surface_area
        # Clamp to [0, 1] (can exceed 1 due to discretization)
        sphericity = min(max(sphericity, 0.0), 1.0)
    else:
        sphericity = 0.0

    features[f"sphericity_{suffix}"] = sphericity

    # Bounding box aspect ratios
    coords = np.argwhere(mask)
    if len(coords) > 0:
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        bbox_size = (max_coords - min_coords + 1).astype(float)

        # Aspect ratios (avoid division by zero)
        bbox_size = np.maximum(bbox_size, 1.0)
        features[f"aspect_xy_{suffix}"] = bbox_size[2] / bbox_size[1]  # W/H
        features[f"aspect_xz_{suffix}"] = bbox_size[2] / bbox_size[0]  # W/D
    else:
        features[f"aspect_xy_{suffix}"] = 1.0
        features[f"aspect_xz_{suffix}"] = 1.0

    # Solidity: volume / convex hull volume
    try:
        from scipy.spatial import ConvexHull
        if len(coords) >= 4:  # Minimum for 3D hull
            hull = ConvexHull(coords * np.array(spacing))
            hull_volume = hull.volume
            solidity = volume / hull_volume if hull_volume > 0 else 0.0
            solidity = min(max(solidity, 0.0), 1.0)
        else:
            solidity = 1.0  # Single point is "solid"
    except Exception:
        solidity = 0.0

    features[f"solidity_{suffix}"] = solidity

    return features


def get_feature_names(
    include_shape: bool = True,
    seg_labels: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Get ordered list of all semantic feature names.

    Args:
        include_shape: Whether to include shape descriptors
        seg_labels: Label mapping for per-label features

    Returns:
        Ordered list of feature names
    """
    if seg_labels is None:
        seg_labels = DEFAULT_SEG_LABELS

    names = []

    # Volume features
    names.append("vol_total")
    for label_name in seg_labels.keys():
        names.append(f"vol_{label_name}")

    # Location features
    names.extend(["loc_x", "loc_y", "loc_z"])

    # Shape features
    if include_shape:
        for suffix in ["total", "ncr", "et"]:
            names.extend([
                f"sphericity_{suffix}",
                f"surface_area_{suffix}",
                f"aspect_xy_{suffix}",
                f"aspect_xz_{suffix}",
            ])
        # Solidity only for total and et
        names.append("solidity_total")
        names.append("solidity_et")

    return names


def get_feature_groups() -> Dict[str, List[str]]:
    """Get feature names grouped by semantic category.

    Returns:
        Dictionary mapping group name to list of feature names
    """
    return {
        "volume": ["vol_total", "vol_ncr", "vol_ed", "vol_et"],
        "location": ["loc_x", "loc_y", "loc_z"],
        "shape": [
            "sphericity_total", "sphericity_ncr", "sphericity_et",
            "surface_area_total", "surface_area_ncr", "surface_area_et",
            "aspect_xy_total", "aspect_xz_total",
            "aspect_xy_ncr", "aspect_xz_ncr",
            "solidity_total", "solidity_et",
        ],
    }


class SemanticFeatureNormalizer:
    """Normalizer for semantic features using z-score standardization.

    Computes mean and std from a dataset and applies normalization.
    Should be fitted on training data and applied to train/val/test.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """Initialize normalizer.

        Args:
            feature_names: List of feature names to normalize.
                           If None, uses all features from get_feature_names().
        """
        self.feature_names = feature_names or get_feature_names()
        self.mean: Optional[Dict[str, float]] = None
        self.std: Optional[Dict[str, float]] = None
        self._fitted = False

    def fit(self, feature_dicts: List[Dict[str, float]]) -> "SemanticFeatureNormalizer":
        """Fit normalizer on a list of feature dictionaries.

        Args:
            feature_dicts: List of feature dictionaries from extract_semantic_features()

        Returns:
            Self for chaining
        """
        if len(feature_dicts) == 0:
            raise ValueError("Cannot fit on empty list")

        # Collect values per feature
        values = {name: [] for name in self.feature_names}
        for fd in feature_dicts:
            for name in self.feature_names:
                if name in fd:
                    values[name].append(fd[name])

        # Compute statistics
        self.mean = {}
        self.std = {}
        for name in self.feature_names:
            if len(values[name]) > 0:
                arr = np.array(values[name])
                self.mean[name] = float(np.mean(arr))
                self.std[name] = float(np.std(arr))
                # Prevent division by zero
                if self.std[name] < 1e-6:
                    self.std[name] = 1.0
            else:
                self.mean[name] = 0.0
                self.std[name] = 1.0

        self._fitted = True
        return self

    def transform(
        self, features: Dict[str, float]
    ) -> Dict[str, float]:
        """Normalize a feature dictionary.

        Args:
            features: Feature dictionary from extract_semantic_features()

        Returns:
            Normalized feature dictionary
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        normalized = {}
        for name in self.feature_names:
            if name in features:
                normalized[name] = (features[name] - self.mean[name]) / self.std[name]
            else:
                normalized[name] = 0.0  # Missing features default to mean

        return normalized

    def fit_transform(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """Fit and transform in one call.

        Args:
            feature_dicts: List of feature dictionaries

        Returns:
            List of normalized feature dictionaries
        """
        self.fit(feature_dicts)
        return [self.transform(fd) for fd in feature_dicts]

    def to_tensor(
        self, features: Dict[str, float]
    ) -> torch.Tensor:
        """Convert normalized features to tensor.

        Args:
            features: Normalized feature dictionary

        Returns:
            Tensor of shape [num_features]
        """
        values = [features.get(name, 0.0) for name in self.feature_names]
        return torch.tensor(values, dtype=torch.float32)

    def save(self, path: str) -> None:
        """Save normalizer statistics to file."""
        import json
        data = {
            "feature_names": self.feature_names,
            "mean": self.mean,
            "std": self.std,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SemanticFeatureNormalizer":
        """Load normalizer from file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)

        normalizer = cls(feature_names=data["feature_names"])
        normalizer.mean = data["mean"]
        normalizer.std = data["std"]
        normalizer._fitted = True
        return normalizer


def features_to_tensor(
    features: Dict[str, float],
    feature_names: Optional[List[str]] = None,
) -> torch.Tensor:
    """Convert feature dictionary to tensor.

    Args:
        features: Feature dictionary
        feature_names: Ordered list of features to include.
                       If None, uses get_feature_names().

    Returns:
        Tensor of shape [num_features]
    """
    if feature_names is None:
        feature_names = get_feature_names()

    values = [features.get(name, 0.0) for name in feature_names]
    return torch.tensor(values, dtype=torch.float32)
