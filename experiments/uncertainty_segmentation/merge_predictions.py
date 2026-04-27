#!/usr/bin/env python
# experiments/uncertainty_segmentation/merge_predictions.py
"""Merge LoRA-ensemble segmentation predictions into MenGrowth dataset and H5.

Pipeline:
    1. Invert ensemble segmentations from preprocessed space (192³) to original
       image space using MONAI inverse transforms, save as seg.nii.gz
    2. Re-run scripts/convert_mengrowth_to_h5.py for full H5 conversion
    3. Append uncertainty/ group to H5 from ensemble volume CSV

The spatial inversion is necessary because ensemble predictions are produced in
the preprocessed coordinate frame (192³, 1mm isotropic, identity affine) while
the MenGrowth dataset stores NIfTI files in original acquisition space (e.g.,
240×240×155). MONAI's ``Invertd`` reverses CropForeground → SpatialPad →
ResizeWithPadOrCrop using transform metadata recorded from the forward pass
on the original t2f image.

Usage::

    python experiments/uncertainty_segmentation/merge_predictions.py \\
        --rank 8 \\
        --results-base /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/uncertainty_segmentation/frozen_decoder/kqv_proj_fc1_fc2/stages_1234 \\
        --data-root /media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/MenGrowth-2025 \\
        --h5-path /media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/h5_format/MenGrowth.h5 \\
        --metadata-csv /media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/metadata_enriched.csv \\
        --n-members 20 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
    SpatialPadd,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from growth.data.transforms import DEFAULT_SPACING, FEATURE_ROI_SIZE

logger = logging.getLogger(__name__)

_PATIENT_PATTERN = re.compile(r"^MenGrowth-\d+$")
_SCAN_PATTERN = re.compile(r"^(MenGrowth-\d+)-(\d+)$")


# =========================================================================
# Discovery
# =========================================================================


def discover_prediction_scans(predictions_dir: Path) -> dict[str, Path]:
    """Discover MenGrowth scan directories under a predictions/ folder.

    Skips non-MenGrowth directories (e.g. ``brats_men_test/``).

    Args:
        predictions_dir: Path to ``r{rank}_M{n}_s{seed}/predictions/``.

    Returns:
        Mapping of scan_id → prediction directory path, sorted by scan_id.
    """
    scans: dict[str, Path] = {}
    for d in sorted(predictions_dir.iterdir()):
        if d.is_dir() and _SCAN_PATTERN.match(d.name):
            scans[d.name] = d
    return scans


def discover_dataset_scans(data_root: Path) -> dict[str, Path]:
    """Discover MenGrowth study directories via two-level directory walk.

    Args:
        data_root: Path containing ``MenGrowth-XXXX/`` patient directories.

    Returns:
        Mapping of scan_id → study directory path, sorted by scan_id.
    """
    scans: dict[str, Path] = {}
    for patient_dir in sorted(data_root.iterdir()):
        if not patient_dir.is_dir() or not _PATIENT_PATTERN.match(patient_dir.name):
            continue
        for scan_dir in sorted(patient_dir.iterdir()):
            if not scan_dir.is_dir():
                continue
            match = _SCAN_PATTERN.match(scan_dir.name)
            if match and match.group(1) == patient_dir.name:
                scans[scan_dir.name] = scan_dir
    return scans


# =========================================================================
# Step 1: Invert + Copy Segmentations
# =========================================================================


def build_inversion_transforms(
    roi_size: tuple[int, ...] | None = None,
) -> Compose:
    """Build MONAI forward transforms for recording spatial operations.

    The pipeline mirrors ``scripts/convert_mengrowth_to_h5.py`` exactly so
    that inversion produces segmentations in the original image coordinate
    frame.

    Args:
        roi_size: Spatial size for crop/pad/resize. Defaults to
            ``FEATURE_ROI_SIZE`` (192³).

    Returns:
        MONAI Compose pipeline (invertible, metadata-tracking).
    """
    roi = list(roi_size or FEATURE_ROI_SIZE)
    keys = ["t2f", "seg"]
    return Compose(
        [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=["t2f"], pixdim=DEFAULT_SPACING, mode="bilinear"),
            Spacingd(keys=["seg"], pixdim=DEFAULT_SPACING, mode="nearest"),
            CropForegroundd(keys=keys, source_key="t2f", k_divisible=roi),
            SpatialPadd(keys=keys, spatial_size=roi),
            ResizeWithPadOrCropd(keys=keys, spatial_size=roi),
        ]
    )


def invert_single_seg(
    ensemble_seg_path: Path,
    t2f_path: Path,
    forward_transforms: Compose,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert one ensemble segmentation from preprocessed to original space.

    Uses ``t2f`` as both image and seg proxy for recording the forward
    transform chain. The actual seg data is swapped in after the forward
    pass, then the inverse is applied.

    Args:
        ensemble_seg_path: Path to 192³ ``segmentation.nii.gz``.
        t2f_path: Path to original ``t2f.nii.gz`` (for spatial metadata).
        forward_transforms: Invertible MONAI Compose pipeline.

    Returns:
        Tuple of (inverted_seg [D,H,W] int8, original_affine [4,4] float64).
    """
    # Forward pass: t2f loaded as both "t2f" and "seg" (same spatial metadata)
    data = {"t2f": str(t2f_path), "seg": str(t2f_path)}
    preprocessed = forward_transforms(data)

    # Load ensemble seg and replace preprocessed seg data in-place
    ensemble_nii = nib.load(str(ensemble_seg_path))
    ensemble_data = torch.from_numpy(ensemble_nii.get_fdata().astype(np.float32))
    if ensemble_data.ndim == 3:
        ensemble_data = ensemble_data.unsqueeze(0)

    assert preprocessed["seg"].shape == ensemble_data.shape, (
        f"Shape mismatch: preprocessed seg {preprocessed['seg'].shape} "
        f"vs ensemble seg {ensemble_data.shape}"
    )
    preprocessed["seg"][:] = ensemble_data

    # Invert spatial transforms on seg
    inverter = Invertd(
        keys=["seg"],
        transform=forward_transforms,
        orig_keys=["seg"],
        nearest_interp=True,
        to_tensor=True,
    )
    inverted = inverter(preprocessed)

    seg_out = inverted["seg"]
    if hasattr(seg_out, "numpy"):
        seg_np = seg_out.numpy()
    else:
        seg_np = np.asarray(seg_out)

    if seg_np.ndim == 4:
        seg_np = seg_np[0]

    seg_np = np.round(seg_np).astype(np.int8)
    original_affine = nib.load(str(t2f_path)).affine

    return seg_np, original_affine


def invert_and_copy_segmentations(
    predictions_dir: Path,
    data_root: Path,
    roi_size: tuple[int, ...] | None = None,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Invert ensemble segs from preprocessed to original space, save as seg.nii.gz.

    For each scan, records the forward MONAI transform chain using the
    original ``t2f.nii.gz``, swaps in the ensemble segmentation data, then
    inverts to recover the original image geometry.

    Args:
        predictions_dir: Path to ``r{rank}_M{n}_s{seed}/predictions/``.
        data_root: Path to ``MenGrowth-2025/`` dataset root.
        roi_size: Override ROI size (for testing). Defaults to production 192³.
        dry_run: If True, log actions but do not write files.

    Returns:
        Mapping of scan_id �� destination seg.nii.gz path.

    Raises:
        ValueError: If prediction and dataset scan IDs don't match 1:1.
        FileNotFoundError: If required NIfTI files are missing.
    """
    pred_scans = discover_prediction_scans(predictions_dir)
    dataset_scans = discover_dataset_scans(data_root)

    pred_ids = set(pred_scans.keys())
    data_ids = set(dataset_scans.keys())
    if pred_ids != data_ids:
        only_pred = sorted(pred_ids - data_ids)
        only_data = sorted(data_ids - pred_ids)
        raise ValueError(
            f"Scan ID mismatch: {len(only_pred)} only in predictions, "
            f"{len(only_data)} only in dataset.\n"
            f"  Predictions-only (first 5): {only_pred[:5]}\n"
            f"  Dataset-only (first 5): {only_data[:5]}"
        )

    forward_transforms = build_inversion_transforms(roi_size)
    results: dict[str, Path] = {}

    for scan_id in tqdm(sorted(pred_scans.keys()), desc="Inverting segmentations"):
        pred_dir = pred_scans[scan_id]
        dataset_dir = dataset_scans[scan_id]

        ensemble_seg_path = pred_dir / "segmentation.nii.gz"
        if not ensemble_seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation.nii.gz in {pred_dir}")

        t2f_path = dataset_dir / "t2f.nii.gz"
        if not t2f_path.exists():
            raise FileNotFoundError(f"Missing t2f.nii.gz in {dataset_dir}")

        dest_path = dataset_dir / "seg.nii.gz"

        if dry_run:
            logger.info(f"[DRY RUN] Would invert+copy {scan_id}")
            results[scan_id] = dest_path
            continue

        seg_np, affine = invert_single_seg(ensemble_seg_path, t2f_path, forward_transforms)
        nib.save(nib.Nifti1Image(seg_np, affine), str(dest_path))
        logger.debug(f"Saved {scan_id}: shape={seg_np.shape}, labels={np.unique(seg_np).tolist()}")

        results[scan_id] = dest_path

    logger.info(f"Inverted and copied {len(results)} segmentations")
    return results


# =========================================================================
# Step 2: H5 Conversion
# =========================================================================


def _import_convert_function():
    """Import ``convert()`` from ``scripts/convert_mengrowth_to_h5.py``."""
    script_path = PROJECT_ROOT / "scripts" / "convert_mengrowth_to_h5.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Convert script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("convert_mengrowth_to_h5", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


def run_h5_conversion(
    data_root: Path,
    h5_output: Path,
    metadata_csv: Path | None = None,
    seed: int = 42,
) -> None:
    """Run ``scripts/convert_mengrowth_to_h5.py`` programmatically.

    Args:
        data_root: MenGrowth-2025/ directory.
        h5_output: Destination H5 file path.
        metadata_csv: Optional path to metadata_enriched.csv.
        seed: Random seed for split generation.

    Raises:
        RuntimeError: If conversion fails or H5 is not created.
    """
    convert = _import_convert_function()

    logger.info(f"Running H5 conversion: {data_root} → {h5_output}")
    try:
        convert(
            data_root=str(data_root),
            output_path=str(h5_output),
            metadata_csv=str(metadata_csv) if metadata_csv else None,
            seed=seed,
        )
    except Exception as e:
        raise RuntimeError(f"H5 conversion failed: {e}") from e

    if not h5_output.exists():
        raise RuntimeError(f"H5 not created after conversion: {h5_output}")

    file_size_gb = h5_output.stat().st_size / (1024**3)
    logger.info(f"H5 conversion complete: {h5_output} ({file_size_gb:.2f} GB)")


# =========================================================================
# Step 3: Append Uncertainty Group
# =========================================================================

_SCALAR_COLUMNS: dict[str, str] = {
    "vol_mean": "vol_mean",
    "vol_std": "vol_std",
    "logvol_mean": "logvol_mean",
    "logvol_std": "logvol_std",
    "vol_median": "vol_median",
    "vol_mad": "vol_mad",
    "logvol_median": "logvol_median",
    "logvol_mad": "logvol_mad",
    "logvol_mad_scaled": "logvol_mad_scaled",
    "vol_ensemble": "vol_ensemble_mask",
    "logvol_ensemble": "logvol_ensemble_mask",
    "mean_entropy": "mean_entropy",
    "mean_mi": "mean_mi",
    "mean_var": "mean_var",
    "men_mean_entropy": "men_mean_entropy",
    "men_mean_mi": "men_mean_mi",
    "men_boundary_entropy": "men_boundary_entropy",
    "men_boundary_mi": "men_boundary_mi",
}


def append_uncertainty_group(
    h5_path: Path,
    volumes_csv: Path,
    rank: int,
    n_members: int,
    seed: int,
) -> None:
    """Append ``uncertainty/`` group to H5 from ensemble volume CSV.

    Reads ``scan_ids`` from the H5 to establish authoritative ordering, then
    reindexes CSV rows to match. Idempotent: deletes existing uncertainty/
    group before writing.

    Args:
        h5_path: Path to the H5 file (must already exist).
        volumes_csv: Path to ``mengrowth_ensemble_volumes.csv``.
        rank: LoRA rank (stored as attribute).
        n_members: Number of ensemble members M.
        seed: Base seed (stored as attribute).

    Raises:
        KeyError: If H5 scan_ids are missing from the CSV.
        ValueError: If required CSV columns are absent.
    """
    df = pd.read_csv(volumes_csv)
    df = df.set_index("scan_id")

    with h5py.File(h5_path, "a") as f:
        scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
        n_scans = len(scan_ids)

        missing = [sid for sid in scan_ids if sid not in df.index]
        if missing:
            raise KeyError(f"{len(missing)} H5 scan_ids not in CSV: {missing[:5]}...")

        df_ordered = df.loc[scan_ids]

        if "uncertainty" in f:
            del f["uncertainty"]
            logger.info("Deleted existing uncertainty/ group (idempotent)")

        unc = f.create_group("uncertainty")
        unc.attrs["n_members"] = n_members
        unc.attrs["rank"] = rank
        unc.attrs["seed"] = seed
        unc.attrs["source_csv"] = str(volumes_csv)

        for ds_name, csv_col in _SCALAR_COLUMNS.items():
            if csv_col not in df_ordered.columns:
                raise ValueError(f"Missing column '{csv_col}' in CSV")
            unc.create_dataset(
                ds_name,
                data=df_ordered[csv_col].values.astype(np.float32),
            )

        member_cols = [f"vol_m{i}" for i in range(n_members)]
        missing_cols = [c for c in member_cols if c not in df_ordered.columns]
        if missing_cols:
            raise ValueError(f"Missing per-member columns: {missing_cols[:5]}...")
        unc.create_dataset(
            "per_member_volumes",
            data=df_ordered[member_cols].values.astype(np.float32),
        )

        f.attrs["uncertainty_rank"] = rank
        f.attrs["ensemble_source"] = f"r{rank}_M{n_members}_s{seed}"

    logger.info(f"Appended uncertainty group: {n_scans} scans, {n_members} members, rank={rank}")


# =========================================================================
# Orchestrator
# =========================================================================


def merge_predictions(
    rank: int,
    results_base: Path,
    data_root: Path,
    h5_path: Path,
    metadata_csv: Path | None = None,
    n_members: int = 20,
    seed: int = 42,
    dry_run: bool = False,
    skip_copy: bool = False,
    skip_convert: bool = False,
    skip_uncertainty: bool = False,
) -> None:
    """Full merge pipeline: invert+copy → H5 convert → append uncertainty.

    Args:
        rank: LoRA rank to use.
        results_base: Path containing ``r{rank}_M{n}_s{seed}/`` directories.
        data_root: MenGrowth-2025/ dataset root.
        h5_path: Output H5 file path.
        metadata_csv: Optional metadata_enriched.csv path.
        n_members: Number of ensemble members.
        seed: Base seed used for the ensemble.
        dry_run: If True, log actions but do not write.
        skip_copy: Skip Step 1 (invert + copy segmentations).
        skip_convert: Skip Step 2 (H5 conversion).
        skip_uncertainty: Skip Step 3 (append uncertainty group).
    """
    run_dir = results_base / f"r{rank}_M{n_members}_s{seed}"
    predictions_dir = run_dir / "predictions"
    volumes_csv = run_dir / "volumes" / "mengrowth_ensemble_volumes.csv"

    for name, path in [
        ("run directory", run_dir),
        ("predictions", predictions_dir),
        ("volumes CSV", volumes_csv),
        ("dataset root", data_root),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    logger.info(f"=== Merge Predictions Pipeline (rank={rank}) ===")
    logger.info(f"  Results: {run_dir}")
    logger.info(f"  Dataset: {data_root}")
    logger.info(f"  H5 output: {h5_path}")

    # Step 1: Invert + Copy
    if not skip_copy:
        logger.info("--- Step 1: Invert and copy segmentations ---")
        invert_and_copy_segmentations(predictions_dir, data_root, dry_run=dry_run)
    else:
        logger.info("--- Step 1: SKIPPED (--skip-copy) ---")

    # Step 2: H5 Conversion
    if not skip_convert and not dry_run:
        logger.info("--- Step 2: H5 conversion ---")
        run_h5_conversion(data_root, h5_path, metadata_csv, seed)
    else:
        reason = "dry-run" if dry_run else "--skip-convert"
        logger.info(f"--- Step 2: SKIPPED ({reason}) ---")

    # Step 3: Append uncertainty
    if not skip_uncertainty and not dry_run:
        logger.info("--- Step 3: Append uncertainty group ---")
        append_uncertainty_group(h5_path, volumes_csv, rank, n_members, seed)
    else:
        reason = "dry-run" if dry_run else "--skip-uncertainty"
        logger.info(f"--- Step 3: SKIPPED ({reason}) ---")

    logger.info("=== Pipeline complete ===")


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Merge ensemble predictions into MenGrowth dataset and H5"
    )
    parser.add_argument("--rank", type=int, required=True, help="LoRA rank")
    parser.add_argument(
        "--results-base",
        type=str,
        required=True,
        help="Base dir containing r{rank}_M{n}_s{seed}/ directories",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to MenGrowth-2025/ dataset root",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        required=True,
        help="Output H5 file path",
    )
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--n-members", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-uncertainty", action="store_true")

    args = parser.parse_args()

    merge_predictions(
        rank=args.rank,
        results_base=Path(args.results_base),
        data_root=Path(args.data_root),
        h5_path=Path(args.h5_path),
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
        n_members=args.n_members,
        seed=args.seed,
        dry_run=args.dry_run,
        skip_copy=args.skip_copy,
        skip_convert=args.skip_convert,
        skip_uncertainty=args.skip_uncertainty,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
