"""Pre-compute linear regression coefficients for shape ~ volume residualization.

Run once before training to compute: for each shape feature,
    residual = shape_value - (slope * vol_total + intercept)

This removes the natural correlation between shape and volume in tumors
(e.g., surface_area ~ V^(2/3)), enabling the latent partitions z_vol and
z_shape to encode genuinely independent factors.

Usage:
    python -m vae.data.compute_shape_residuals \
        --data_root /path/to/brats_men \
        --output src/vae/config/shape_residual_params.json \
        --shape_features sphericity_total surface_area_total solidity_total \
                         aspect_xy_total sphericity_ncr surface_area_ncr \
        --volume_feature vol_total \
        --spacing 1.25 1.25 1.25 \
        --roi_size 160 160 160

Output JSON format:
    {
        "feature_name": {
            "slope": float,
            "intercept": float,
            "r2": float
        },
        ...
    }
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .semantic_features import extract_semantic_features

logger = logging.getLogger(__name__)


def compute_residualization_params(
    data_root: str,
    shape_features: List[str],
    volume_feature: str = "vol_total",
    spacing: Tuple[float, float, float] = (1.25, 1.25, 1.25),
    roi_size: Tuple[int, int, int] = (160, 160, 160),
) -> Dict[str, Dict[str, float]]:
    """Compute OLS regression coefficients for shape ~ volume.

    For each shape feature, fits:
        shape_feature_i = slope_i * vol_total + intercept_i

    and stores the coefficients for residualization during training.

    Args:
        data_root: Path to the BraTS dataset root
        shape_features: List of shape feature names to residualize
        volume_feature: Name of the volume feature (predictor)
        spacing: Voxel spacing in mm for feature extraction
        roi_size: ROI dimensions for coordinate normalization

    Returns:
        Dictionary mapping feature names to regression parameters:
            {feature_name: {"slope": float, "intercept": float, "r2": float}}
    """
    from monai.transforms import LoadImage
    import nibabel as nib

    data_path = Path(data_root)

    # Find all segmentation files
    seg_files = sorted(data_path.rglob("*seg*.nii.gz"))
    if not seg_files:
        # Try alternative pattern
        seg_files = sorted(data_path.rglob("*_seg.nii*"))
    if not seg_files:
        raise FileNotFoundError(f"No segmentation files found in {data_root}")

    logger.info(f"Found {len(seg_files)} segmentation files")

    # Extract features from all subjects
    all_features: List[Dict[str, float]] = []
    for seg_path in seg_files:
        try:
            seg_img = nib.load(str(seg_path))
            seg_data = seg_img.get_fdata().astype(np.int32)

            features = extract_semantic_features(
                seg=seg_data,
                spacing=spacing,
                roi_size=roi_size,
                compute_shape=True,
            )
            all_features.append(features)
        except Exception as e:
            logger.warning(f"Failed to process {seg_path}: {e}")
            continue

    if len(all_features) < 10:
        raise ValueError(
            f"Only {len(all_features)} valid samples found. "
            "Need at least 10 for reliable regression."
        )

    logger.info(f"Successfully extracted features from {len(all_features)} subjects")

    # Extract volume and shape values
    vol_values = np.array([f.get(volume_feature, 0.0) for f in all_features])

    # Fit OLS for each shape feature
    results = {}
    for feat_name in shape_features:
        feat_values = np.array([f.get(feat_name, 0.0) for f in all_features])

        # Skip features with zero variance
        if np.std(feat_values) < 1e-8:
            logger.warning(f"Feature '{feat_name}' has zero variance, skipping")
            results[feat_name] = {"slope": 0.0, "intercept": 0.0, "r2": 0.0}
            continue

        # OLS: y = slope * x + intercept
        # Using numpy polyfit (degree 1)
        slope, intercept = np.polyfit(vol_values, feat_values, deg=1)

        # Compute R² to understand explained variance
        predicted = slope * vol_values + intercept
        ss_res = np.sum((feat_values - predicted) ** 2)
        ss_tot = np.sum((feat_values - np.mean(feat_values)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        results[feat_name] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
        }

        logger.info(
            f"  {feat_name}: slope={slope:.6f}, intercept={intercept:.6f}, R²={r2:.4f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute shape-volume regression coefficients for residualization"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to BraTS dataset root directory"
    )
    parser.add_argument(
        "--output", type=str, default="src/vae/config/shape_residual_params.json",
        help="Output JSON path for regression coefficients"
    )
    parser.add_argument(
        "--shape_features", nargs="+",
        default=[
            "sphericity_total", "surface_area_total", "solidity_total",
            "aspect_xy_total", "sphericity_ncr", "surface_area_ncr",
        ],
        help="Shape features to residualize"
    )
    parser.add_argument(
        "--volume_feature", type=str, default="vol_total",
        help="Volume feature used as predictor"
    )
    parser.add_argument(
        "--spacing", nargs=3, type=float, default=[1.25, 1.25, 1.25],
        help="Voxel spacing in mm (D H W)"
    )
    parser.add_argument(
        "--roi_size", nargs=3, type=int, default=[160, 160, 160],
        help="ROI dimensions for coordinate normalization"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(f"Computing residualization params from: {args.data_root}")
    logger.info(f"Shape features: {args.shape_features}")
    logger.info(f"Volume feature: {args.volume_feature}")

    results = compute_residualization_params(
        data_root=args.data_root,
        shape_features=args.shape_features,
        volume_feature=args.volume_feature,
        spacing=tuple(args.spacing),
        roi_size=tuple(args.roi_size),
    )

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved residualization params to: {output_path}")

    # Print summary
    logger.info("\nSummary of volume-shape correlations:")
    for feat, params in results.items():
        logger.info(f"  {feat}: R²={params['r2']:.4f} (slope={params['slope']:.6f})")


if __name__ == "__main__":
    main()
