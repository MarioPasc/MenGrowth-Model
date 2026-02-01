# src/growth/data/semantic_features.py
"""Semantic feature extraction from segmentation masks.

Computes volume (total, NCR, ED, ET), location (centroid), and shape
features (sphericity, surface area, solidity, aspect ratios) from masks.

These features serve as regression targets for linear probing experiments
and for the SDP (Supervised Disentangled Projection) training.

Label convention (BraTS):
    0: Background
    1: NCR (Necrotic Core)
    2: ED (Peritumoral Edema)
    3: ET (Enhancing Tumor)
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

# Default spacing in mm (isotropic after preprocessing)
DEFAULT_SPACING = (1.0, 1.0, 1.0)

# BraTS label values
LABEL_BACKGROUND = 0
LABEL_NCR = 1
LABEL_ED = 2
LABEL_ET = 3


def compute_volumes(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> Dict[str, float]:
    """Compute tumor component volumes in mm^3.

    Args:
        mask: Segmentation mask [D, H, W] with labels {0: background, 1: NCR, 2: ED, 3: ET}
        spacing: Voxel spacing in mm (z, y, x) or (d, h, w)

    Returns:
        Dictionary with keys:
        - 'total': Total tumor volume (NCR + ED + ET)
        - 'ncr': Necrotic core volume
        - 'ed': Peritumoral edema volume
        - 'et': Enhancing tumor volume

    Example:
        >>> mask = np.zeros((96, 96, 96), dtype=np.int32)
        >>> mask[40:60, 40:60, 40:60] = 1  # 20x20x20 = 8000 voxels
        >>> volumes = compute_volumes(mask, spacing=(1.0, 1.0, 1.0))
        >>> volumes['ncr']
        8000.0
    """
    voxel_volume = float(np.prod(spacing))

    # Count voxels for each label
    ncr_count = np.sum(mask == LABEL_NCR)
    ed_count = np.sum(mask == LABEL_ED)
    et_count = np.sum(mask == LABEL_ET)
    total_count = ncr_count + ed_count + et_count

    return {
        "total": float(total_count * voxel_volume),
        "ncr": float(ncr_count * voxel_volume),
        "ed": float(ed_count * voxel_volume),
        "et": float(et_count * voxel_volume),
    }


def compute_log_volumes(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> np.ndarray:
    """Compute log-transformed volumes as feature array.

    Log transformation provides better numerical properties for regression
    and handles the large dynamic range of tumor volumes.

    Args:
        mask: Segmentation mask [D, H, W]
        spacing: Voxel spacing in mm

    Returns:
        Array of shape [4]: [log(V_total+1), log(V_NCR+1), log(V_ED+1), log(V_ET+1)]
    """
    volumes = compute_volumes(mask, spacing)

    return np.array(
        [
            np.log1p(volumes["total"]),
            np.log1p(volumes["ncr"]),
            np.log1p(volumes["ed"]),
            np.log1p(volumes["et"]),
        ],
        dtype=np.float32,
    )


def compute_centroid(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
    normalize: bool = True,
) -> np.ndarray:
    """Compute tumor centroid (center of mass).

    Uses the whole tumor mask (all non-background labels) to compute
    the centroid, weighted by voxel presence.

    Args:
        mask: Segmentation mask [D, H, W]
        spacing: Voxel spacing in mm (used for physical coordinates if not normalizing)
        normalize: If True, normalize to [0, 1] based on image dimensions.
                  If False, return physical coordinates in mm.

    Returns:
        Array of shape [3]: [cz, cy, cx] or [cd, ch, cw]
        Returns [0.5, 0.5, 0.5] (center) if mask is empty.

    Example:
        >>> mask = np.zeros((96, 96, 96), dtype=np.int32)
        >>> mask[20:40, 30:50, 40:60] = 1  # Tumor in upper-left-back region
        >>> centroid = compute_centroid(mask, normalize=True)
        >>> centroid  # Should be around [0.31, 0.42, 0.52]
    """
    # Create binary tumor mask (all tumor labels)
    tumor_mask = mask > LABEL_BACKGROUND

    if not np.any(tumor_mask):
        # Empty mask: return center
        if normalize:
            return np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            shape = np.array(mask.shape, dtype=np.float32)
            spacing_arr = np.array(spacing, dtype=np.float32)
            return (shape * spacing_arr) / 2.0

    # Compute center of mass
    centroid = ndimage.center_of_mass(tumor_mask)
    centroid = np.array(centroid, dtype=np.float32)

    if normalize:
        # Normalize to [0, 1] based on image dimensions
        shape = np.array(mask.shape, dtype=np.float32)
        centroid = centroid / shape
    else:
        # Convert to physical coordinates (mm)
        spacing_arr = np.array(spacing, dtype=np.float32)
        centroid = centroid * spacing_arr

    return centroid


def compute_bounding_box(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute axis-aligned bounding box of tumor.

    Args:
        mask: Segmentation mask [D, H, W]

    Returns:
        Tuple of (min_coords, max_coords), each of shape [3].
        Returns ((0,0,0), (1,1,1)) if mask is empty.
    """
    tumor_mask = mask > LABEL_BACKGROUND

    if not np.any(tumor_mask):
        return np.zeros(3, dtype=np.int32), np.ones(3, dtype=np.int32)

    # Find bounding box
    coords = np.where(tumor_mask)
    min_coords = np.array([c.min() for c in coords], dtype=np.int32)
    max_coords = np.array([c.max() for c in coords], dtype=np.int32)

    return min_coords, max_coords


def compute_surface_area(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> float:
    """Compute approximate surface area of tumor.

    Uses the marching cubes approximation via counting surface voxels.

    Args:
        mask: Segmentation mask [D, H, W]
        spacing: Voxel spacing in mm

    Returns:
        Surface area in mm^2
    """
    tumor_mask = (mask > LABEL_BACKGROUND).astype(np.uint8)

    if not np.any(tumor_mask):
        return 0.0

    # Erode to find interior, surface = original - eroded
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(tumor_mask, structure=structure)
    surface_voxels = np.sum(tumor_mask) - np.sum(eroded)

    # Approximate surface area (each surface voxel contributes ~1 face)
    # Average face area for non-isotropic voxels
    dz, dy, dx = spacing
    avg_face_area = (dz * dy + dy * dx + dz * dx) / 3.0

    return float(surface_voxels * avg_face_area)


def compute_sphericity(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> float:
    """Compute sphericity of tumor.

    Sphericity is defined as:
        psi = (pi^(1/3) * (6V)^(2/3)) / A

    Where V is volume and A is surface area. A perfect sphere has psi = 1.

    Args:
        mask: Segmentation mask [D, H, W]
        spacing: Voxel spacing in mm

    Returns:
        Sphericity value in [0, 1]. Returns 0 if tumor is empty.
    """
    volumes = compute_volumes(mask, spacing)
    volume = volumes["total"]

    if volume == 0:
        return 0.0

    surface_area = compute_surface_area(mask, spacing)

    if surface_area == 0:
        return 0.0

    # Sphericity formula
    sphericity = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surface_area

    # Clamp to [0, 1] due to discretization artifacts
    return float(np.clip(sphericity, 0.0, 1.0))


def compute_solidity(mask: np.ndarray) -> float:
    """Compute solidity (ratio of volume to convex hull volume).

    Solidity measures how "solid" or "compact" the tumor is.
    A convex shape has solidity = 1.

    Args:
        mask: Segmentation mask [D, H, W]

    Returns:
        Solidity value in [0, 1]. Returns 0 if tumor is empty or too small.
    """
    tumor_mask = mask > LABEL_BACKGROUND
    tumor_voxels = np.sum(tumor_mask)

    if tumor_voxels < 4:  # Need at least 4 points for 3D convex hull
        return 0.0

    # Get coordinates of tumor voxels
    coords = np.array(np.where(tumor_mask)).T  # [N, 3]

    try:
        hull = ConvexHull(coords)
        hull_volume = hull.volume
        solidity = float(tumor_voxels / hull_volume) if hull_volume > 0 else 0.0
        return np.clip(solidity, 0.0, 1.0)
    except Exception:
        # ConvexHull can fail for degenerate cases
        return 0.0


def compute_aspect_ratios(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> Dict[str, float]:
    """Compute aspect ratios from bounding box.

    Args:
        mask: Segmentation mask [D, H, W]
        spacing: Voxel spacing in mm

    Returns:
        Dictionary with aspect ratios:
        - 'aspect_dh': depth / height
        - 'aspect_dw': depth / width
        - 'aspect_hw': height / width
    """
    min_coords, max_coords = compute_bounding_box(mask)
    extents = (max_coords - min_coords + 1).astype(np.float32)

    # Apply spacing to get physical extents
    spacing_arr = np.array(spacing, dtype=np.float32)
    physical_extents = extents * spacing_arr

    d, h, w = physical_extents

    # Avoid division by zero
    eps = 1e-6

    return {
        "aspect_dh": float(d / (h + eps)),
        "aspect_dw": float(d / (w + eps)),
        "aspect_hw": float(h / (w + eps)),
    }


def compute_shape_features(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> Dict[str, float]:
    """Compute all shape descriptors.

    Args:
        mask: Segmentation mask [D, H, W]
        spacing: Voxel spacing in mm

    Returns:
        Dictionary with shape features:
        - 'sphericity': Shape compactness measure
        - 'surface_area_log': log(surface_area + 1)
        - 'solidity': Ratio to convex hull
        - 'aspect_dh': Depth/height ratio
        - 'aspect_dw': Depth/width ratio
        - 'aspect_hw': Height/width ratio
    """
    surface_area = compute_surface_area(mask, spacing)
    aspect_ratios = compute_aspect_ratios(mask, spacing)

    return {
        "sphericity": compute_sphericity(mask, spacing),
        "surface_area_log": float(np.log1p(surface_area)),
        "solidity": compute_solidity(mask),
        "aspect_dh": aspect_ratios["aspect_dh"],
        "aspect_dw": aspect_ratios["aspect_dw"],
        "aspect_hw": aspect_ratios["aspect_hw"],
    }


def compute_shape_array(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> np.ndarray:
    """Compute shape features as array.

    Args:
        mask: Segmentation mask [D, H, W]
        spacing: Voxel spacing in mm

    Returns:
        Array of shape [6]: [sphericity, surface_area_log, solidity,
                            aspect_dh, aspect_dw, aspect_hw]
    """
    features = compute_shape_features(mask, spacing)

    return np.array(
        [
            features["sphericity"],
            features["surface_area_log"],
            features["solidity"],
            features["aspect_dh"],
            features["aspect_dw"],
            features["aspect_hw"],
        ],
        dtype=np.float32,
    )


def extract_semantic_features(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> Dict[str, np.ndarray]:
    """Extract all semantic features from segmentation mask.

    This is the main entry point for feature extraction, returning
    all features needed for linear probing and SDP training.

    Args:
        mask: Segmentation mask [D, H, W] with BraTS labels
        spacing: Voxel spacing in mm

    Returns:
        Dictionary with:
        - 'volume': [4] log-transformed volumes
        - 'location': [3] normalized centroid
        - 'shape': [6] shape descriptors
        - 'all': [13] concatenation of all features

    Example:
        >>> mask = load_segmentation("patient_001.nii.gz")
        >>> features = extract_semantic_features(mask)
        >>> features['volume'].shape
        (4,)
        >>> features['all'].shape
        (13,)
    """
    volume = compute_log_volumes(mask, spacing)
    location = compute_centroid(mask, spacing, normalize=True)
    shape = compute_shape_array(mask, spacing)

    return {
        "volume": volume,
        "location": location,
        "shape": shape,
        "all": np.concatenate([volume, location, shape]),
    }


def extract_semantic_features_from_file(
    mask_path: str,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Extract semantic features from a NIfTI segmentation file.

    Args:
        mask_path: Path to segmentation NIfTI file
        spacing: Voxel spacing in mm. If None, read from file header.

    Returns:
        Dictionary with volume, location, shape, and all features.
    """
    import nibabel as nib

    img = nib.load(mask_path)
    mask = np.asarray(img.dataobj).astype(np.int32)

    if spacing is None:
        # Get spacing from NIfTI header (first 3 elements of pixdim)
        spacing = tuple(img.header.get_zooms()[:3])

    return extract_semantic_features(mask, spacing)
