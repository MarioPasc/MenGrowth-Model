#!/usr/bin/env python3
"""Extract BraTS-MEN test split for benchmark inference.

Two modes:

1. ``--raw-dir`` (recommended): Symlink canonical (240, 240, 155) NIfTIs from
   the raw BraTS-MEN-Train release into ``output/nifti/``. The H5 is consulted
   only to select the 150 test scan IDs and to dump ground-truth segmentations
   in the same canonical canvas (also taken from raw). External BraTS25/23
   containers (qing, mmdp, blackbean, cnmc_pmi2023, nvauto) require this
   shape — the H5's 192³ ROI crops are rejected by the dataloaders.

2. ``--h5-only`` (legacy / fallback): Write the 192³ ROI crops as before. Kept
   only for offline sanity checks; produces inputs that **all** external
   containers will reject.

Usage::

    python extract_h5_to_nifti.py --h5 BraTS_MEN.h5 --raw-dir /path/to/BraTS_Men_Train --output out/
    python extract_h5_to_nifti.py --h5 BraTS_MEN.h5 --output out/ --h5-only
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

H5_CHANNEL_ORDER = ["t2f", "t1c", "t1n", "t2w"]
BRATS_CANONICAL_SHAPE = (240, 240, 155)
AFFINE_FALLBACK = np.eye(4, dtype=np.float64)


def _load_test_scan_ids(h5_path: Path, limit: int | None) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        test_indices = f["splits/test"][:]
        id_key = "scan_ids" if "scan_ids" in f else "subject_ids"
        all_ids = f[id_key][:]
        all_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in all_ids]
        ids = [all_ids[int(i)] for i in test_indices]
    if limit is not None:
        ids = ids[:limit]
    return ids


def extract_from_raw(
    h5_path: Path,
    raw_dir: Path,
    output_dir: Path,
    limit: int | None,
    use_symlinks: bool,
) -> dict:
    """Symlink (or copy) canonical BraTS-MEN NIfTIs for the test split."""
    nifti_dir = output_dir / "nifti"
    gt_dir = output_dir / "ground_truth"
    nifti_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    scan_ids = _load_test_scan_ids(h5_path, limit)
    logger.info("Test split: %d scan IDs (limit=%s)", len(scan_ids), limit)

    manifest_entries: dict[str, dict] = {}
    missing: list[str] = []
    bad_shape: list[str] = []
    t0 = time.time()

    for i, scan_id in enumerate(scan_ids):
        src = raw_dir / scan_id
        if not src.is_dir():
            missing.append(scan_id)
            continue

        scan_nifti_dir = nifti_dir / scan_id
        scan_gt_dir = gt_dir / scan_id
        scan_nifti_dir.mkdir(parents=True, exist_ok=True)
        scan_gt_dir.mkdir(parents=True, exist_ok=True)

        # Modalities: BraTS canonical ordering t1c/t1n/t2f/t2w (filename suffix)
        for modality in ("t1c", "t1n", "t2f", "t2w"):
            src_file = src / f"{scan_id}-{modality}.nii.gz"
            if not src_file.is_file():
                missing.append(f"{scan_id}/{modality}")
                continue
            dst_file = scan_nifti_dir / src_file.name
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            if use_symlinks:
                dst_file.symlink_to(src_file.resolve())
            else:
                shutil.copy2(src_file, dst_file)

        seg_src = src / f"{scan_id}-seg.nii.gz"
        if seg_src.is_file():
            seg_dst = scan_gt_dir / "seg.nii.gz"
            if seg_dst.exists() or seg_dst.is_symlink():
                seg_dst.unlink()
            if use_symlinks:
                seg_dst.symlink_to(seg_src.resolve())
            else:
                shutil.copy2(seg_src, seg_dst)

        # Validate shape on the first modality
        try:
            ref = nib.load(str(scan_nifti_dir / f"{scan_id}-t1c.nii.gz"))
            shp = tuple(ref.shape)
            if shp != BRATS_CANONICAL_SHAPE:
                bad_shape.append(f"{scan_id}={shp}")
            manifest_entries[scan_id] = {"spatial_shape": list(shp)}
        except Exception as e:
            logger.warning("Could not validate shape for %s: %s", scan_id, e)
            manifest_entries[scan_id] = {"spatial_shape": list(BRATS_CANONICAL_SHAPE)}

        if (i + 1) % 25 == 0 or (i + 1) == len(scan_ids):
            logger.info("  Linked %d/%d (%.1fs)", i + 1, len(scan_ids), time.time() - t0)

    if missing:
        logger.error("Missing %d files in raw-dir; first 5: %s",
                     len(missing), missing[:5])
    if bad_shape:
        logger.error("Non-canonical shapes (%d); first 5: %s",
                     len(bad_shape), bad_shape[:5])

    manifest = {
        "source": "raw",
        "h5_file": str(h5_path),
        "raw_dir": str(raw_dir),
        "n_patients": len(manifest_entries),
        "n_missing": len(missing),
        "n_bad_shape": len(bad_shape),
        "expected_shape": list(BRATS_CANONICAL_SHAPE),
        "channel_order_filenames": ["t1c", "t1n", "t2f", "t2w"],
        "scans": manifest_entries,
    }
    with open(output_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    logger.info("Manifest written: %s", output_dir / "manifest.json")
    return manifest


def extract_from_h5(
    h5_path: Path,
    output_dir: Path,
    limit: int | None,
) -> dict:
    """Legacy path: write 192³ ROI crops. External containers will reject these."""
    logger.error(
        "H5-only extraction is the legacy path: external BraTS25/23 containers "
        "expect canonical (240,240,155). They WILL reject 192³ inputs."
    )
    nifti_dir = output_dir / "nifti"
    gt_dir = output_dir / "ground_truth"

    with h5py.File(h5_path, "r") as f:
        test_indices = f["splits/test"][:]
        id_key = "scan_ids" if "scan_ids" in f else "subject_ids"
        all_ids = f[id_key][:]
        all_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in all_ids]
        images_ds = f["images"]
        segs_ds = f["segs"]
        img_shape = images_ds.shape[1:]
        if limit is not None:
            test_indices = test_indices[:limit]

        manifest_entries: dict[str, dict] = {}
        t0 = time.time()
        for i, idx in enumerate(test_indices):
            idx = int(idx)
            scan_id = all_ids[idx]
            scan_nifti_dir = nifti_dir / scan_id
            scan_gt_dir = gt_dir / scan_id
            scan_nifti_dir.mkdir(parents=True, exist_ok=True)
            scan_gt_dir.mkdir(parents=True, exist_ok=True)

            image = images_ds[idx]
            seg = segs_ds[idx]
            for ch_idx, modality in enumerate(H5_CHANNEL_ORDER):
                vol = image[ch_idx]
                fname = f"{scan_id}-{modality}.nii.gz"
                nii = nib.Nifti1Image(vol.astype(np.float32), AFFINE_FALLBACK)
                nib.save(nii, scan_nifti_dir / fname)

            seg_vol = seg[0]
            seg_nii = nib.Nifti1Image(seg_vol.astype(np.int8), AFFINE_FALLBACK)
            nib.save(seg_nii, scan_gt_dir / "seg.nii.gz")
            manifest_entries[scan_id] = {
                "h5_index": idx,
                "spatial_shape": list(img_shape[1:]),
            }
            if (i + 1) % 25 == 0 or (i + 1) == len(test_indices):
                logger.info("  Extracted %d/%d (%.1fs)",
                            i + 1, len(test_indices), time.time() - t0)

    manifest = {
        "source": "h5",
        "h5_file": str(h5_path),
        "n_patients": len(manifest_entries),
        "spatial_shape": list(img_shape[1:]),
        "channel_order": H5_CHANNEL_ORDER,
        "affine": "identity_1mm_isotropic",
        "scans": manifest_entries,
    }
    with open(output_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BraTS-MEN test split for benchmark inference")
    parser.add_argument("--h5", type=str, required=True, help="Path to BraTS_MEN.h5")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to raw BraTS_Men_Train directory (recommended). "
             "If set, NIfTIs are symlinked from raw at canonical (240,240,155).",
    )
    parser.add_argument(
        "--h5-only",
        action="store_true",
        help="Force H5 path (legacy 192³ crops). External containers will reject these.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy raw files instead of symlinking. Defaults to symlinks.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N patients")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    h5_path = Path(args.h5)
    output_dir = Path(args.output)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    if args.h5_only:
        extract_from_h5(h5_path, output_dir, args.limit)
    else:
        if args.raw_dir is None:
            raise SystemExit(
                "ERROR: --raw-dir is required (canonical 240×240×155 NIfTIs).\n"
                "       Pass --h5-only to force the legacy 192³ extraction "
                "(external containers will reject those inputs)."
            )
        raw_dir = Path(args.raw_dir)
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Raw dataset dir not found: {raw_dir}")
        extract_from_raw(
            h5_path=h5_path,
            raw_dir=raw_dir,
            output_dir=output_dir,
            limit=args.limit,
            use_symlinks=not args.copy,
        )


if __name__ == "__main__":
    main()
