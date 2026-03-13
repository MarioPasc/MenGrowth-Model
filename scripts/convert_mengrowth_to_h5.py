#!/usr/bin/env python
# scripts/convert_mengrowth_to_h5.py
"""Convert MenGrowth NIfTI files to a single HDF5 file with longitudinal hierarchy.

Pre-applies spatial preprocessing (Orient -> Resample -> CropForeground -> SpatialPad ->
CenterCrop at 192^3) but does NOT apply z-score normalization.

MenGrowth has 33 patients, 100 scans (2-6 timepoints each), with labels {0,1,2,3}.
10 scans have empty segmentation (no tumor).

Directory layout::

    data_root/
        MenGrowth-XXXX/
            MenGrowth-XXXX-YYY/
                {t2f,t1c,t1n,t2w,seg}.nii.gz   (bare names)
                OR
                MenGrowth-XXXX-YYY-{t2f,...}.nii.gz  (prefixed names)

H5 Schema:
    mengrowth.h5
    |-- attrs: {n_scans, n_patients, roi_size, spacing, channel_order, version,
    |           dataset_type="longitudinal", domain="MenGrowth"}
    |-- images           [N_scans, 4, 192, 192, 192] float32
    |-- segs             [N_scans, 1, 192, 192, 192] int8
    |-- scan_ids         [N_scans] str  ("MenGrowth-0001-000")
    |-- patient_ids      [N_scans] str  ("MenGrowth-0001")
    |-- timepoint_idx    [N_scans] int32
    |-- semantic/{volume, location, shape}
    |-- longitudinal/{patient_offsets [N_patients+1], patient_list [N_patients]}
    |-- splits/{lora_train, lora_val, test}  (patient-level)
    +-- metadata/{grade, age, sex}  (from --metadata-csv if provided; else placeholders)

Usage:
    # Dry-run with 3 patients
    python scripts/convert_mengrowth_to_h5.py \
        --data-root /media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/MenGrowth-2025 \
        --output /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/MenGrowth.h5 \
        --max-patients 3

    # Full conversion with metadata
    python scripts/convert_mengrowth_to_h5.py \
        --data-root /media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/MenGrowth-2025 \
        --output /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/source/MenGrowth.h5 \
        --metadata-csv /media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/dataset/metadata_enriched.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
    SpatialPadd,
)
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from growth.data.bratsmendata import split_subjects_multi
from growth.data.semantic_features import extract_semantic_features
from growth.data.transforms import (
    DEFAULT_SPACING,
    FEATURE_ROI_SIZE,
    MODALITY_KEYS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# H5 file version
H5_VERSION = "2.0"

# Default split sizes (patient-level, ~70/10/20 split for 31 patients)
DEFAULT_SPLIT_SIZES = {
    "lora_train": 21,
    "lora_val": 3,
    "test": 7,
}

# Patterns for parsing directory names
_PATIENT_PATTERN = re.compile(r"^MenGrowth-\d+$")
_SCAN_PATTERN = re.compile(r"^(MenGrowth-\d+)-(\d+)$")


# =========================================================================
# Metadata Loading
# =========================================================================


def load_metadata(
    metadata_csv: Path,
) -> dict[str, dict[str, float | str]]:
    """Load clinical metadata from the enriched CSV.

    The CSV has columns: patient_id, age, sex, ..., MenGrowth_ID.
    Only rows with a non-empty MenGrowth_ID are included patients.

    Sex encoding: 0 → "F", 1 → "M".

    Args:
        metadata_csv: Path to metadata_enriched.csv.

    Returns:
        Dict mapping MenGrowth patient ID → {"age": float|NaN, "sex": str}.
    """
    metadata: dict[str, dict[str, float | str]] = {}

    with open(metadata_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mg_id = row.get("MenGrowth_ID", "").strip()
            if not mg_id:
                continue

            # Parse age
            age_str = row.get("age", "").strip()
            age = float(age_str) if age_str else float("nan")

            # Parse sex: 0 → F, 1 → M
            sex_str = row.get("sex", "").strip()
            if sex_str == "0.0" or sex_str == "0":
                sex = "F"
            elif sex_str == "1.0" or sex_str == "1":
                sex = "M"
            else:
                sex = "unknown"

            metadata[mg_id] = {"age": age, "sex": sex}

    logger.info(
        f"Loaded metadata for {len(metadata)} patients from {metadata_csv}"
    )
    n_with_age = sum(1 for m in metadata.values() if not np.isnan(m["age"]))
    n_with_sex = sum(1 for m in metadata.values() if m["sex"] != "unknown")
    logger.info(f"  Age available: {n_with_age}, Sex available: {n_with_sex}")

    return metadata


# =========================================================================
# NIfTI Discovery
# =========================================================================


def _find_nifti(scan_dir: Path, name: str) -> Path:
    """Find a NIfTI file in *scan_dir* by modality/seg name.

    Supports two naming conventions:
      1. Bare names:     ``{name}.nii.gz``  (e.g. ``t2f.nii.gz``)
      2. Prefixed names: ``{scan_id}-{name}.nii.gz``
         (e.g. ``MenGrowth-0001-000-t2f.nii.gz``)

    Args:
        scan_dir: Directory containing NIfTI files for one scan.
        name: Modality key (``t2f``, ``t1c``, ...) or ``seg``.

    Returns:
        Path to the matching NIfTI file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    # Try bare name first (current convention)
    bare = scan_dir / f"{name}.nii.gz"
    if bare.exists():
        return bare

    # Try prefixed name: {scan_dir.name}-{name}.nii.gz
    prefixed = scan_dir / f"{scan_dir.name}-{name}.nii.gz"
    if prefixed.exists():
        return prefixed

    raise FileNotFoundError(
        f"Missing {name} in {scan_dir}: tried {bare.name} and {prefixed.name}"
    )


# =========================================================================
# Discovery
# =========================================================================


def discover_mengrowth_scans(
    data_root: Path,
) -> list[dict[str, str | int | Path]]:
    """Discover all MenGrowth scans via two-level directory walk.

    MenGrowth uses a patient/scan nesting:
        MenGrowth-XXXX/MenGrowth-XXXX-YYY/

    Args:
        data_root: Path containing MenGrowth-* patient directories.

    Returns:
        List of dicts with keys: 'scan_id', 'patient_id', 'timepoint', 'scan_dir'.
        Sorted by (patient_id, timepoint).
    """
    scans: list[dict[str, str | int | Path]] = []

    for patient_dir in sorted(data_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        if not _PATIENT_PATTERN.match(patient_dir.name):
            continue

        patient_id = patient_dir.name

        for scan_dir in sorted(patient_dir.iterdir()):
            if not scan_dir.is_dir():
                continue

            match = _SCAN_PATTERN.match(scan_dir.name)
            if match is None:
                continue

            # Verify the patient portion matches the parent directory
            if match.group(1) != patient_id:
                continue

            timepoint = int(match.group(2))
            scans.append({
                "scan_id": scan_dir.name,
                "patient_id": patient_id,
                "timepoint": timepoint,
                "scan_dir": scan_dir,
            })

    # Sort by patient then timepoint
    scans.sort(key=lambda s: (s["patient_id"], s["timepoint"]))
    return scans


def build_longitudinal_structure(
    scans: list[dict],
) -> tuple[list[str], np.ndarray, list[int]]:
    """Build CSR-style longitudinal indexing from scan list.

    Args:
        scans: Sorted list of scan dicts (from discover_mengrowth_scans).

    Returns:
        Tuple of (patient_list, patient_offsets, timepoint_indices).
        - patient_list: Unique patient IDs in order.
        - patient_offsets: CSR array [N_patients+1] where patient i's scans
          are at indices [offsets[i]:offsets[i+1]].
        - timepoint_indices: Sequential 0-based timepoint index per scan.
    """
    patient_list: list[str] = []
    patient_offsets = [0]
    timepoint_indices: list[int] = []

    current_patient = None
    tp_counter = 0

    for scan in scans:
        pid = scan["patient_id"]
        if pid != current_patient:
            if current_patient is not None:
                patient_offsets.append(len(timepoint_indices))
            patient_list.append(pid)
            current_patient = pid
            tp_counter = 0

        timepoint_indices.append(tp_counter)
        tp_counter += 1

    # Final offset
    patient_offsets.append(len(timepoint_indices))

    return patient_list, np.array(patient_offsets, dtype=np.int32), timepoint_indices


def load_scan_paths(scan_dir: Path) -> dict[str, Path]:
    """Get paths to all modalities and segmentation for a scan.

    Uses bare-name convention with prefixed-name fallback.

    Args:
        scan_dir: Path to scan directory.

    Returns:
        Dict with keys: 't1c', 't1n', 't2f', 't2w', 'seg'.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    paths: dict[str, Path] = {}
    for modality in MODALITY_KEYS:
        paths[modality] = _find_nifti(scan_dir, modality)

    paths["seg"] = _find_nifti(scan_dir, "seg")
    return paths


# =========================================================================
# Preprocessing
# =========================================================================


def build_preprocessing_transforms() -> Compose:
    """Build MONAI transforms for spatial preprocessing without normalization.

    Pipeline: Load -> EnsureChannelFirst -> Orient(RAS) -> Spacing(1mm) ->
              CropForeground -> SpatialPad(192^3) -> CenterCrop(192^3) ->
              Concat -> EnsureType

    Returns:
        MONAI Compose pipeline.
    """
    modality_keys = list(MODALITY_KEYS)
    seg_key = "seg"
    all_keys = modality_keys + [seg_key]
    roi_size = list(FEATURE_ROI_SIZE)

    transforms = [
        LoadImaged(keys=all_keys, image_only=False),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=modality_keys,
            pixdim=DEFAULT_SPACING,
            mode=["bilinear"] * len(modality_keys),
        ),
        Spacingd(keys=[seg_key], pixdim=DEFAULT_SPACING, mode=("nearest",)),
        CropForegroundd(
            keys=all_keys,
            source_key=modality_keys[0],
            k_divisible=roi_size,
        ),
        SpatialPadd(keys=all_keys, spatial_size=roi_size),
        ResizeWithPadOrCropd(keys=all_keys, spatial_size=roi_size),
        ConcatItemsd(keys=modality_keys, name="image", dim=0),
        EnsureTyped(keys=["image", seg_key], dtype="float32", track_meta=False),
    ]

    return Compose(transforms)


# =========================================================================
# Conversion
# =========================================================================


def convert(
    data_root: str,
    output_path: str,
    max_patients: int | None = None,
    seed: int = 42,
    compression: str = "gzip",
    compression_level: int = 4,
    metadata_csv: str | None = None,
) -> None:
    """Convert MenGrowth NIfTI files to HDF5 with longitudinal hierarchy.

    Args:
        data_root: Path to MenGrowth data root directory.
        output_path: Output H5 file path.
        max_patients: Max patients for dry-run (None = all).
        seed: Random seed for split generation.
        compression: HDF5 compression algorithm.
        compression_level: Compression level (1-9).
        metadata_csv: Optional path to metadata_enriched.csv with age/sex.
    """
    data_root_path = Path(data_root)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Discover scans
    all_scans = discover_mengrowth_scans(data_root_path)
    logger.info(f"Found {len(all_scans)} scans in {data_root_path}")

    # Build longitudinal structure
    patient_list, patient_offsets, timepoint_indices = build_longitudinal_structure(
        all_scans
    )
    n_patients = len(patient_list)
    logger.info(f"Found {n_patients} patients")

    # Optionally limit patients
    if max_patients is not None and max_patients < n_patients:
        last_scan_idx = int(patient_offsets[max_patients])
        all_scans = all_scans[:last_scan_idx]
        patient_list = patient_list[:max_patients]
        patient_offsets = patient_offsets[: max_patients + 1]
        timepoint_indices = timepoint_indices[:last_scan_idx]
        n_patients = max_patients
        logger.info(f"Limiting to {max_patients} patients ({len(all_scans)} scans)")

    n_scans = len(all_scans)
    roi = list(FEATURE_ROI_SIZE)

    # Build preprocessing pipeline
    transforms = build_preprocessing_transforms()

    # Generate patient-level splits
    if max_patients is None or n_patients >= sum(DEFAULT_SPLIT_SIZES.values()):
        split_sizes = DEFAULT_SPLIT_SIZES
    else:
        total_default = sum(DEFAULT_SPLIT_SIZES.values())
        split_sizes = {}
        allocated = 0
        for name, size in DEFAULT_SPLIT_SIZES.items():
            if name == list(DEFAULT_SPLIT_SIZES.keys())[-1]:
                split_sizes[name] = n_patients - allocated
            else:
                split_sizes[name] = max(1, round(size * n_patients / total_default))
                allocated += split_sizes[name]

    patient_splits = split_subjects_multi(patient_list, split_sizes, seed=seed)

    # Build patient -> index mapping
    patient_to_idx = {pid: i for i, pid in enumerate(patient_list)}

    # Log timepoint distribution
    tp_counts = np.diff(patient_offsets)
    logger.info("Timepoint distribution:")
    for n_tp in sorted(set(tp_counts)):
        count = (tp_counts == n_tp).sum()
        logger.info(f"  {n_tp} tp: {count} patients")

    # Create H5 file
    logger.info(f"Creating H5 file: {output_path_obj}")
    logger.info(f"  Scans: {n_scans}, Patients: {n_patients}")
    logger.info(f"  ROI size: {roi}")
    logger.info(f"  Compression: {compression} level={compression_level}")

    with h5py.File(output_path_obj, "w") as f:
        # Global attributes
        f.attrs["n_scans"] = n_scans
        f.attrs["n_patients"] = n_patients
        f.attrs["roi_size"] = roi
        f.attrs["spacing"] = list(DEFAULT_SPACING)
        f.attrs["channel_order"] = list(MODALITY_KEYS)
        f.attrs["version"] = H5_VERSION
        f.attrs["dataset_type"] = "longitudinal"
        f.attrs["domain"] = "MenGrowth"

        # Pre-allocate datasets
        images_ds = f.create_dataset(
            "images",
            shape=(n_scans, 4, *roi),
            dtype="float32",
            chunks=(1, 4, *roi),
            compression=compression,
            compression_opts=compression_level,
        )

        segs_ds = f.create_dataset(
            "segs",
            shape=(n_scans, 1, *roi),
            dtype="int8",
            chunks=(1, 1, *roi),
            compression=compression,
            compression_opts=compression_level,
        )

        # IDs
        dt = h5py.special_dtype(vlen=str)
        scan_id_list = [s["scan_id"] for s in all_scans]
        patient_id_list = [s["patient_id"] for s in all_scans]

        f.create_dataset("scan_ids", data=scan_id_list, dtype=dt)
        f.create_dataset("patient_ids", data=patient_id_list, dtype=dt)
        f.create_dataset(
            "timepoint_idx", data=np.array(timepoint_indices, dtype=np.int32)
        )

        # Longitudinal group
        long_grp = f.create_group("longitudinal")
        long_grp.create_dataset("patient_offsets", data=patient_offsets)
        long_grp.create_dataset("patient_list", data=patient_list, dtype=dt)

        # Metadata group — load from CSV if provided, else placeholders
        patient_metadata: dict[str, dict[str, float | str]] = {}
        if metadata_csv is not None:
            patient_metadata = load_metadata(Path(metadata_csv))

        age_arr = np.full(n_scans, np.nan, dtype=np.float32)
        sex_arr = ["unknown"] * n_scans
        for i, scan_info in enumerate(all_scans):
            pid = scan_info["patient_id"]
            if pid in patient_metadata:
                age_arr[i] = patient_metadata[pid]["age"]
                sex_arr[i] = patient_metadata[pid]["sex"]

        n_age_filled = np.sum(np.isfinite(age_arr))
        n_sex_filled = sum(1 for s in sex_arr if s != "unknown")
        logger.info(
            f"Metadata coverage: age={n_age_filled}/{n_scans} scans, "
            f"sex={n_sex_filled}/{n_scans} scans"
        )

        meta_grp = f.create_group("metadata")
        meta_grp.create_dataset("grade", data=np.full(n_scans, -1, dtype=np.int8))
        meta_grp.create_dataset("age", data=age_arr)
        meta_grp.create_dataset("sex", data=sex_arr, dtype=dt)

        # Semantic features group
        sem_grp = f.create_group("semantic")
        vol_ds = sem_grp.create_dataset("volume", shape=(n_scans, 4), dtype="float32")
        loc_ds = sem_grp.create_dataset(
            "location", shape=(n_scans, 3), dtype="float32"
        )
        shape_ds = sem_grp.create_dataset("shape", shape=(n_scans, 3), dtype="float32")

        # Process each scan
        n_errors = 0
        n_empty_segs = 0
        for i, scan_info in enumerate(tqdm(all_scans, desc="Converting")):
            try:
                scan_dir = scan_info["scan_dir"]
                paths = load_scan_paths(scan_dir)

                data = {
                    "t2f": str(paths["t2f"]),
                    "t1c": str(paths["t1c"]),
                    "t1n": str(paths["t1n"]),
                    "t2w": str(paths["t2w"]),
                    "seg": str(paths["seg"]),
                }
                result = transforms(data)

                image_np = result["image"].numpy()
                seg_np = result["seg"].numpy()

                assert image_np.shape == (4, *roi), (
                    f"Image shape mismatch: {image_np.shape}"
                )
                assert seg_np.shape == (1, *roi), (
                    f"Seg shape mismatch: {seg_np.shape}"
                )

                images_ds[i] = image_np
                segs_ds[i] = seg_np.astype(np.int8)

                # Semantic features (no label 4 in MenGrowth, no merge needed)
                seg_for_semantic = seg_np[0].astype(np.int32)
                if not np.any(seg_for_semantic > 0):
                    n_empty_segs += 1

                sem = extract_semantic_features(
                    seg_for_semantic,
                    spacing=(1.0, 1.0, 1.0),
                    merge_rc_into_ncr=False,
                )
                vol_ds[i] = sem["volume"]
                loc_ds[i] = sem["location"]
                shape_ds[i] = sem["shape"]

            except Exception as e:
                logger.error(f"Failed to process {scan_info['scan_id']}: {e}")
                n_errors += 1
                continue

        # Splits group (patient-level indices into patient_list)
        splits_grp = f.create_group("splits")
        for split_name, pid_list in patient_splits.items():
            indices = np.array(
                [patient_to_idx[pid] for pid in pid_list], dtype=np.int32
            )
            splits_grp.create_dataset(split_name, data=indices)
            # Count scans in this split
            n_split_scans = sum(
                int(patient_offsets[idx + 1] - patient_offsets[idx]) for idx in indices
            )
            logger.info(
                f"  Split '{split_name}': {len(indices)} patients, "
                f"{n_split_scans} scans"
            )

    # Summary
    file_size_gb = output_path_obj.stat().st_size / (1024**3)
    logger.info("\nConversion complete!")
    logger.info(f"  Output: {output_path_obj}")
    logger.info(f"  File size: {file_size_gb:.2f} GB")
    logger.info(f"  Scans: {n_scans} ({n_errors} errors)")
    logger.info(f"  Patients: {n_patients}")
    logger.info(f"  Empty segmentations: {n_empty_segs}")

    # Verification
    _verify_h5(output_path_obj, n_scans, n_patients, roi)


def _verify_h5(
    h5_path: Path,
    n_scans: int,
    n_patients: int,
    roi: list[int],
) -> None:
    """Spot-check the H5 file for correctness."""
    logger.info("\nVerification:")

    with h5py.File(h5_path, "r") as f:
        # Check attributes
        assert f.attrs["n_scans"] == n_scans
        assert f.attrs["n_patients"] == n_patients
        assert list(f.attrs["roi_size"]) == roi
        assert list(f.attrs["channel_order"]) == list(MODALITY_KEYS)
        assert f.attrs["domain"] == "MenGrowth"
        assert f.attrs["dataset_type"] == "longitudinal"

        # Check shapes
        assert f["images"].shape == (n_scans, 4, *roi)
        assert f["segs"].shape == (n_scans, 1, *roi)
        assert len(f["scan_ids"]) == n_scans
        assert len(f["patient_ids"]) == n_scans

        # Check longitudinal structure
        offsets = f["longitudinal/patient_offsets"][:]
        assert len(offsets) == n_patients + 1
        assert offsets[0] == 0
        assert offsets[-1] == n_scans

        # Check semantic shapes
        assert f["semantic/volume"].shape == (n_scans, 4)
        assert f["semantic/location"].shape == (n_scans, 3)
        assert f["semantic/shape"].shape == (n_scans, 3)

        # Spot-check 3 random scans
        rng = np.random.RandomState(42)
        check_indices = rng.choice(n_scans, min(3, n_scans), replace=False)

        for idx in check_indices:
            sid = f["scan_ids"][idx]
            if isinstance(sid, bytes):
                sid = sid.decode()

            img = f["images"][idx]
            seg = f["segs"][idx]

            assert not np.isnan(img).any(), f"NaN in image for {sid}"
            assert not np.isinf(img).any(), f"Inf in image for {sid}"
            assert img.max() > 0, f"All-zero image for {sid}"

            unique_labels = np.unique(seg)
            valid_labels = {0, 1, 2, 3}
            assert set(unique_labels).issubset(valid_labels), (
                f"Invalid labels {unique_labels} for {sid}"
            )

            logger.info(
                f"  [{idx}] {sid}: "
                f"img range=[{img.min():.1f}, {img.max():.1f}], "
                f"seg labels={sorted(unique_labels)}"
            )

        # Check splits (patient-level)
        if "splits" in f:
            total_patients_in_splits = sum(
                len(f[f"splits/{s}"]) for s in f["splits"]
            )
            logger.info(f"  Total patients in splits: {total_patients_in_splits}")

    logger.info("  Verification passed!")


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert MenGrowth NIfTI files to HDF5 with longitudinal hierarchy"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to MenGrowth data root (containing MenGrowth-XXXX dirs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mengrowth.h5",
        help="Output H5 file path",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Max patients for dry-run testing (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split generation",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level 1-9 (default: 4)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Path to metadata_enriched.csv with age/sex columns and MenGrowth_ID",
    )

    args = parser.parse_args()

    convert(
        data_root=args.data_root,
        output_path=args.output,
        max_patients=args.max_patients,
        seed=args.seed,
        compression_level=args.compression_level,
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()
