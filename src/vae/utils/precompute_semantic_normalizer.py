#!/usr/bin/env python
"""Pre-compute semantic features and fit normalizer on training set only.

This script:
1. Loads all training subjects (NOT val/test - prevents data leakage)
2. Extracts semantic features from segmentation masks
3. Fits SemanticFeatureNormalizer on training statistics
4. Saves normalizer JSON to cache folder

Usage:
    python -m vae.utils.precompute_semantic_normalizer \
        --data-root /media/mpascual/PortableSSD/Meningiomas/BraTS/BraTS_Men_Train \
        --output-dir experiments/runs/semivae/cache_semivae \
        --val-split 0.1 \
        --seed 42

The normalizer JSON contains mean/std for all 19 features:
- Volume (4): vol_total, vol_ncr, vol_ed, vol_et
- Location (3): loc_x, loc_y, loc_z
- Shape (12): sphericity_*, surface_area_*, aspect_*, solidity_*
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from vae.data.datasets import build_subject_index, create_train_val_split
from vae.data.semantic_features import (
    extract_semantic_features,
    SemanticFeatureNormalizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_resample_segmentation(
    seg_path: str,
    target_spacing: Tuple[float, float, float] = (1.875, 1.875, 1.875),
) -> np.ndarray:
    """Load segmentation NIfTI and resample to target spacing.

    Args:
        seg_path: Path to segmentation NIfTI file
        target_spacing: Target voxel spacing in mm (D, H, W)

    Returns:
        Resampled segmentation as numpy array
    """
    nii = nib.load(seg_path)
    seg = nii.get_fdata().astype(np.int32)

    # Get original spacing from affine
    original_spacing = np.abs(np.diag(nii.affine)[:3])

    # Compute zoom factors
    zoom_factors = original_spacing / np.array(target_spacing)

    # Resample with nearest neighbor (for labels)
    if not np.allclose(zoom_factors, 1.0, atol=0.01):
        seg = ndimage.zoom(seg, zoom_factors, order=0, mode='nearest')

    return seg


def extract_features_from_subjects(
    subjects: List[Dict[str, str]],
    spacing: Tuple[float, float, float],
    roi_size: Tuple[int, int, int],
    seg_labels: Dict[str, int],
    compute_shape: bool = True,
) -> List[Dict[str, float]]:
    """Extract semantic features from all subjects.

    Args:
        subjects: List of subject dictionaries with 'seg' key
        spacing: Voxel spacing in mm
        roi_size: ROI dimensions for coordinate normalization
        seg_labels: Segmentation label mapping
        compute_shape: Whether to compute shape descriptors

    Returns:
        List of feature dictionaries, one per subject
    """
    all_features = []

    for subject in tqdm(subjects, desc="Extracting features"):
        seg_path = subject["seg"]
        subject_id = subject["id"]

        try:
            seg = load_and_resample_segmentation(seg_path, spacing)
            features = extract_semantic_features(
                seg=seg,
                spacing=spacing,
                roi_size=roi_size,
                seg_labels=seg_labels,
                compute_shape=compute_shape,
            )
            features["_subject_id"] = subject_id
            all_features.append(features)

        except Exception as e:
            logger.warning(f"Failed to extract features for {subject_id}: {e}")
            continue

    return all_features


def validate_features(features_list: List[Dict[str, float]]) -> None:
    """Validate extracted features and log statistics.

    Args:
        features_list: List of feature dictionaries
    """
    if not features_list:
        raise ValueError("No features extracted!")

    feature_names = [k for k in features_list[0].keys() if not k.startswith("_")]
    logger.info(f"Extracted {len(features_list)} samples with {len(feature_names)} features")

    for name in feature_names:
        values = [f[name] for f in features_list if name in f]
        if values:
            arr = np.array(values)
            logger.info(
                f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                f"min={arr.min():.4f}, max={arr.max():.4f}"
            )

    # Check for potential issues
    vol_total = [f["vol_total"] for f in features_list]
    if any(v < 1.0 for v in vol_total):
        n_small = sum(1 for v in vol_total if v < 1.0)
        logger.warning(f"{n_small} samples have very small tumor volumes!")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute semantic features normalizer for SemiVAE"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/media/mpascual/PortableSSD/Meningiomas/BraTS/BraTS_Men_Train",
        help="Root directory containing BraTS subject folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for normalizer JSON (e.g., cache_semivae/)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.875, 1.875, 1.875],
        help="Target voxel spacing in mm (default: 1.875 1.875 1.875)",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="ROI size for coordinate normalization (default: 128 128 128)",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["t1c", "t1n", "t2f", "t2w"],
        help="Modality names for subject index building",
    )
    parser.add_argument(
        "--no-shape",
        action="store_true",
        help="Skip shape feature computation (faster)",
    )

    args = parser.parse_args()

    # Segmentation labels (BraTS convention)
    seg_labels = {
        "ncr": 1,
        "ed": 2,
        "et": 3,
    }

    # Build subject index
    logger.info(f"Building subject index from {args.data_root}")
    subjects = build_subject_index(args.data_root, args.modalities)
    logger.info(f"Found {len(subjects)} total subjects")

    # Create train/val split
    logger.info(f"Creating train/val split with val_split={args.val_split}, seed={args.seed}")
    train_subjects, val_subjects = create_train_val_split(
        subjects, args.val_split, args.seed
    )
    logger.info(f"Training set: {len(train_subjects)} subjects")
    logger.info(f"Validation set: {len(val_subjects)} subjects (NOT used for normalizer)")

    # Extract features from TRAINING SET ONLY
    logger.info("Extracting semantic features from training set...")
    spacing = tuple(args.spacing)
    roi_size = tuple(args.roi_size)

    train_features = extract_features_from_subjects(
        train_subjects,
        spacing=spacing,
        roi_size=roi_size,
        seg_labels=seg_labels,
        compute_shape=not args.no_shape,
    )

    validate_features(train_features)

    # Fit normalizer on training features
    logger.info("Fitting SemanticFeatureNormalizer on training statistics...")
    clean_features = [
        {k: v for k, v in f.items() if not k.startswith("_")}
        for f in train_features
    ]

    normalizer = SemanticFeatureNormalizer()
    normalizer.fit(clean_features)

    # Create output directory and save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer_path = output_dir / "semantic_normalizer.json"
    normalizer.save(str(normalizer_path))
    logger.info(f"Saved normalizer to {normalizer_path}")

    # Save metadata
    metadata = {
        "data_root": args.data_root,
        "n_train_subjects": len(train_subjects),
        "n_val_subjects": len(val_subjects),
        "val_split": args.val_split,
        "seed": args.seed,
        "spacing": list(spacing),
        "roi_size": list(roi_size),
        "seg_labels": seg_labels,
        "compute_shape": not args.no_shape,
        "feature_names": normalizer.feature_names,
    }

    metadata_path = output_dir / "normalizer_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    # Log final statistics
    logger.info("\n" + "=" * 60)
    logger.info("NORMALIZER STATISTICS (mean / std)")
    logger.info("=" * 60)
    for name in normalizer.feature_names:
        mean = normalizer.mean[name]
        std = normalizer.std[name]
        logger.info(f"  {name:25s}: {mean:8.4f} / {std:8.4f}")

    logger.info("\nDone! Normalizer ready for SemiVAE training.")


if __name__ == "__main__":
    main()
