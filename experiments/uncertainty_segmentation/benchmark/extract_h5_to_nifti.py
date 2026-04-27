#!/usr/bin/env python3
"""Extract BraTS-MEN test split from H5 to NIfTI for benchmark inference.

Reads the 150-patient test split from the BraTS_MEN.h5 archive and saves
each scan as individual modality NIfTI files in BraTS naming convention,
plus ground-truth segmentations for downstream Dice evaluation.

Usage:
    python extract_h5_to_nifti.py --h5 /path/to/BraTS_MEN.h5 --output /path/to/output
    python extract_h5_to_nifti.py --h5 /path/to/BraTS_MEN.h5 --output /path/to/output --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

H5_CHANNEL_ORDER = ["t2f", "t1c", "t1n", "t2w"]
AFFINE = np.eye(4, dtype=np.float64)


def extract_test_split(
    h5_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict:
    """Extract test-split scans from H5 to NIfTI files.

    Parameters
    ----------
    h5_path : str | Path
        Path to BraTS_MEN.h5 archive.
    output_dir : str | Path
        Root output directory. Creates ``nifti/`` and ``ground_truth/`` subdirs.
    limit : int | None
        If set, extract only the first ``limit`` patients (for debugging).

    Returns
    -------
    dict
        Manifest with scan IDs, indices, and extraction metadata.
    """
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    nifti_dir = output_dir / "nifti"
    gt_dir = output_dir / "ground_truth"

    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        test_indices = f["splits/test"][:]
        logger.info("Test split: %d patients", len(test_indices))

        id_key = "scan_ids" if "scan_ids" in f else "subject_ids"
        all_ids = f[id_key][:]
        all_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in all_ids]

        images_ds = f["images"]
        segs_ds = f["segs"]
        img_shape = images_ds.shape[1:]
        logger.info("H5 shape: images=%s, segs=%s", images_ds.shape, segs_ds.shape)

        if limit is not None:
            test_indices = test_indices[:limit]
            logger.info("Limited to %d patients (--limit)", limit)

        manifest_entries = {}
        t0 = time.time()

        for i, idx in enumerate(test_indices):
            idx = int(idx)
            scan_id = all_ids[idx]

            scan_nifti_dir = nifti_dir / scan_id
            scan_nifti_dir.mkdir(parents=True, exist_ok=True)

            scan_gt_dir = gt_dir / scan_id
            scan_gt_dir.mkdir(parents=True, exist_ok=True)

            image = images_ds[idx]
            seg = segs_ds[idx]

            for ch_idx, modality in enumerate(H5_CHANNEL_ORDER):
                vol = image[ch_idx]
                fname = f"{scan_id}-{modality}.nii.gz"
                nii = nib.Nifti1Image(vol.astype(np.float32), AFFINE)
                nib.save(nii, scan_nifti_dir / fname)

            seg_vol = seg[0]
            seg_nii = nib.Nifti1Image(seg_vol.astype(np.int8), AFFINE)
            nib.save(seg_nii, scan_gt_dir / "seg.nii.gz")

            manifest_entries[scan_id] = {
                "h5_index": idx,
                "spatial_shape": list(img_shape[1:]),
            }

            if (i + 1) % 25 == 0 or (i + 1) == len(test_indices):
                elapsed = time.time() - t0
                logger.info(
                    "  Extracted %d/%d (%.1fs elapsed)",
                    i + 1,
                    len(test_indices),
                    elapsed,
                )

    manifest = {
        "h5_file": str(h5_path),
        "n_patients": len(manifest_entries),
        "spatial_shape": list(img_shape[1:]),
        "channel_order": H5_CHANNEL_ORDER,
        "affine": "identity_1mm_isotropic",
        "scans": manifest_entries,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    elapsed_total = time.time() - t0
    logger.info(
        "Extraction complete: %d patients in %.1fs → %s",
        len(manifest_entries),
        elapsed_total,
        output_dir,
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BraTS-MEN test split from H5 to NIfTI")
    parser.add_argument(
        "--h5",
        type=str,
        required=True,
        help="Path to BraTS_MEN.h5",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for extracted NIfTIs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit extraction to first N patients (for debugging)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    extract_test_split(
        h5_path=args.h5,
        output_dir=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
