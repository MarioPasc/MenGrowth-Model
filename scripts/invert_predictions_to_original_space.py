#!/usr/bin/env python
"""Invert 192^3 ensemble predictions back onto the original MenGrowth NIfTI grid.

The H5 build pipeline (``scripts/convert_mengrowth_to_h5.py``) applies, in order,

    LoadImaged -> EnsureChannelFirstd -> Orientationd(RAS)
    -> Spacingd(pixdim=1mm) -> CropForegroundd(source_key=t2f, k_divisible=192)
    -> SpatialPadd(192^3) -> ResizeWithPadOrCropd(192^3)

and then discards all spatial metadata (``track_meta=False``). The inference
engine subsequently saves predictions with an identity affine, so the masks
cannot be overlaid on the raw BraTS-layout originals (240x240x155 with a
non-trivial affine).

This script replays the forward pipeline on each scan's ``t2f.nii.gz`` with
``track_meta=True`` to recover the chain of ``applied_operations``, wraps the
saved prediction as a :class:`monai.data.MetaTensor` carrying that same chain,
then uses :class:`monai.transforms.Invertd` (with nearest-neighbor resampling
for label preservation) to map the prediction back onto the source grid. The
resulting NIfTI is written next to the original modalities with the native
affine and qform/sform codes.

Usage
-----
    python scripts/invert_predictions_to_original_space.py \
        --predictions-dir /media/.../predictions \
        --dataset-dir     /media/.../MenGrowth-2025 \
        [--source-filename segmentation.nii.gz] \
        [--output-filename seg.nii.gz] \
        [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
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

# Project constants — kept in sync with ``src/growth/data/transforms.py`` and
# ``scripts/convert_mengrowth_to_h5.py``. Imported defensively to avoid pulling
# the whole ``growth`` package on systems where it is not installed.
MODALITY_KEYS: list[str] = ["t2f", "t1c", "t1n", "t2w"]
FEATURE_ROI_SIZE: tuple[int, int, int] = (192, 192, 192)
DEFAULT_SPACING: tuple[float, float, float] = (1.0, 1.0, 1.0)

logger = logging.getLogger("invert_predictions")


# =============================================================================
# Forward pipeline — mirrors the H5 build transforms
# =============================================================================


def build_forward_pipeline() -> Compose:
    """Return the exact forward pipeline used to build the H5 volumes.

    We declare the pipeline over all four MRI modalities (even though only
    ``t2f`` is strictly required to recover the CropForeground bbox) because
    :class:`Invertd` inspects the full set of applied operations and must see
    a consistent metadata chain on the reference key.

    Returns
    -------
    Compose
        Dict-based MONAI pipeline yielding ``[1, 192, 192, 192]`` MetaTensors.
    """
    keys = list(MODALITY_KEYS)
    roi_size = list(FEATURE_ROI_SIZE)
    return Compose(
        [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=DEFAULT_SPACING,
                mode=["bilinear"] * len(keys),
            ),
            CropForegroundd(
                keys=keys,
                source_key=keys[0],  # "t2f"
                k_divisible=roi_size,
            ),
            SpatialPadd(keys=keys, spatial_size=roi_size),
            ResizeWithPadOrCropd(keys=keys, spatial_size=roi_size),
        ]
    )


# =============================================================================
# Scan discovery
# =============================================================================


@dataclass(frozen=True)
class ScanJob:
    """Paths for one scan to invert.

    Attributes
    ----------
    scan_id
        Scan identifier, e.g. ``MenGrowth-0001-000``.
    patient_id
        Patient identifier, e.g. ``MenGrowth-0001``.
    prediction_path
        Saved 192^3 NIfTI prediction (ensemble mask or multi-label seg).
    modality_paths
        Mapping ``{modality_key: path_to_original_nifti}`` for all four
        modalities. All must exist; ``t2f`` is required to recover the
        foreground bbox.
    output_path
        Destination NIfTI in the source dataset folder.
    """

    scan_id: str
    patient_id: str
    prediction_path: Path
    modality_paths: dict[str, Path]
    output_path: Path


def parse_scan_id(scan_id: str) -> str:
    """Derive the patient identifier from a scan identifier.

    Scan ids follow the pattern ``MenGrowth-XXXX-YYY`` where the trailing
    three-digit block is a per-patient timepoint index. The patient id is
    the scan id with that trailing block stripped.

    Parameters
    ----------
    scan_id
        Full scan identifier.

    Returns
    -------
    str
        Patient identifier.

    Raises
    ------
    ValueError
        If the scan id does not match the expected pattern.
    """
    parts = scan_id.rsplit("-", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(
            f"Scan id {scan_id!r} does not match pattern MenGrowth-XXXX-YYY"
        )
    return parts[0]


def discover_jobs(
    predictions_dir: Path,
    dataset_dir: Path,
    source_filename: str,
    output_filename: str,
    overwrite: bool,
) -> list[ScanJob]:
    """Enumerate scans with both a prediction and the four source modalities.

    Parameters
    ----------
    predictions_dir
        Directory containing ``{scan_id}/`` subfolders produced by the
        uncertainty-segmentation pipeline.
    dataset_dir
        Directory containing ``{patient_id}/{scan_id}/{modality}.nii.gz``.
    source_filename
        File within each prediction folder to invert
        (e.g. ``segmentation.nii.gz`` or ``ensemble_mask.nii.gz``).
    output_filename
        Destination filename inside each scan's dataset folder.
    overwrite
        If ``False``, scans whose output already exists are skipped.

    Returns
    -------
    list[ScanJob]
        Sorted list of jobs ready to invert.
    """
    jobs: list[ScanJob] = []
    scan_dirs = sorted(p for p in predictions_dir.iterdir() if p.is_dir())
    for scan_dir in scan_dirs:
        scan_id = scan_dir.name
        pred_path = scan_dir / source_filename
        if not pred_path.exists():
            logger.debug("skip %s: missing prediction %s", scan_id, source_filename)
            continue

        try:
            patient_id = parse_scan_id(scan_id)
        except ValueError as exc:
            logger.warning("skip %s: %s", scan_id, exc)
            continue

        scan_src_dir = dataset_dir / patient_id / scan_id
        modality_paths = {k: scan_src_dir / f"{k}.nii.gz" for k in MODALITY_KEYS}
        missing = [k for k, p in modality_paths.items() if not p.exists()]
        if missing:
            logger.warning("skip %s: missing source modalities %s", scan_id, missing)
            continue

        output_path = scan_src_dir / output_filename
        if output_path.exists() and not overwrite:
            logger.info("skip %s: %s already exists (use --overwrite)", scan_id, output_filename)
            continue

        jobs.append(
            ScanJob(
                scan_id=scan_id,
                patient_id=patient_id,
                prediction_path=pred_path,
                modality_paths=modality_paths,
                output_path=output_path,
            )
        )
    return jobs


# =============================================================================
# Inversion
# =============================================================================


def _load_prediction_array(path: Path) -> np.ndarray:
    """Read a 192^3 (optionally channel-leading) prediction NIfTI as int array.

    The prediction may be a 3D hard mask (``[192, 192, 192]``) or a 4D
    multi-label file with singleton channel axis; both collapse to a 3D
    integer volume here.
    """
    arr = np.asarray(nib.load(str(path)).dataobj)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 3D prediction volume at {path}, got shape {arr.shape}"
        )
    if arr.shape != FEATURE_ROI_SIZE:
        raise ValueError(
            f"Prediction at {path} has shape {arr.shape}, "
            f"expected {FEATURE_ROI_SIZE} (H5 feature ROI)"
        )
    return arr.astype(np.int16, copy=False)


def invert_single(
    job: ScanJob,
    forward: Compose,
) -> None:
    """Invert one scan's prediction and write the result to the source folder.

    Parameters
    ----------
    job
        Scan paths.
    forward
        Prebuilt forward pipeline.

    Notes
    -----
    Implementation details:

    * We run the forward pipeline on the original ``t2f/t1c/t1n/t2w`` NIfTIs
      so that each yielded ``MetaTensor`` carries the full
      ``applied_operations`` chain (Orient, Spacing, CropForeground, Pad,
      ResizeWithPadOrCrop). Only the ``t2f`` MetaTensor is needed afterwards.
    * The saved prediction is loaded as a numpy array and wrapped in a new
      :class:`MetaTensor` whose ``meta`` and ``applied_operations`` are
      cloned from the ``t2f`` tensor. This is the standard MONAI idiom for
      post-hoc inversion of predictions that were produced outside the
      transform pipeline (see MONAI tutorials, ``sliding_window_inference``
      examples).
    * :class:`Invertd` with ``nearest_interp=True`` inverts every
      resampling step using nearest-neighbor interpolation, preserving
      integer BraTS labels. CropForeground and Pad are slice/pad
      operations and invert exactly.
    * The inverted tensor is written with the affine that MONAI restored
      from the ``applied_operations`` chain, i.e. the affine of the original
      NIfTI modulo the RAS-axis reordering that ``Orientationd`` imposed.
      For this dataset all originals are already RAS + 1 mm iso, so the
      recovered affine equals the original t2f affine.
    """
    data_in = {k: str(p) for k, p in job.modality_paths.items()}
    processed = forward(data_in)

    ref: MetaTensor = processed[MODALITY_KEYS[0]]  # t2f, [1, 192, 192, 192]
    if tuple(ref.shape[-3:]) != FEATURE_ROI_SIZE:
        raise RuntimeError(
            f"Forward pipeline produced {tuple(ref.shape)} for {job.scan_id}, "
            f"expected a trailing {FEATURE_ROI_SIZE} ROI"
        )

    pred_arr = _load_prediction_array(job.prediction_path)
    pred_tensor = torch.from_numpy(pred_arr).unsqueeze(0).to(ref.dtype)  # [1,D,H,W]

    # Attach the reference metadata and applied-operations chain to the
    # prediction so Invertd can reverse it.
    processed["pred"] = MetaTensor(
        pred_tensor,
        meta=dict(ref.meta),
        applied_operations=list(ref.applied_operations),
    )

    invert = Invertd(
        keys="pred",
        transform=forward,
        orig_keys=MODALITY_KEYS[0],
        meta_keys="pred_meta_dict",
        orig_meta_keys=f"{MODALITY_KEYS[0]}_meta_dict",
        nearest_interp=True,
        to_tensor=True,
    )
    inverted = invert(processed)["pred"]

    # Strip the leading channel axis and cast to int8 (BraTS label range is
    # 0..3, well within int8). Take the affine from the MetaTensor — this is
    # the native affine restored by Invertd.
    data_out = inverted[0].detach().cpu().numpy().astype(np.int8)
    if isinstance(inverted, MetaTensor):
        affine = np.asarray(inverted.affine.detach().cpu().numpy(), dtype=np.float64)
    else:
        affine = np.eye(4, dtype=np.float64)

    # Cross-check against the source affine. If they disagree, prefer the
    # original NIfTI's affine: MONAI's Invertd is correct in theory, but this
    # guard ensures a mis-trimmed header from an unusual source scan can't
    # silently shift the overlay.
    src_img = nib.load(str(job.modality_paths[MODALITY_KEYS[0]]))
    src_shape = src_img.shape
    src_affine = src_img.affine
    if data_out.shape != src_shape:
        raise RuntimeError(
            f"Inverted shape {data_out.shape} != source {src_shape} for {job.scan_id}"
        )
    if not np.allclose(affine, src_affine, atol=1e-5):
        logger.warning(
            "affine mismatch for %s (max |Δ| = %.3g); using source affine",
            job.scan_id,
            float(np.max(np.abs(affine - src_affine))),
        )
        affine = src_affine

    out_img = nib.Nifti1Image(data_out, affine)
    # Copy qform/sform codes from the source so Slicer chooses the same
    # transform Slicer uses for the original modalities.
    out_img.header.set_qform(affine, code=int(src_img.header["qform_code"]))
    out_img.header.set_sform(affine, code=int(src_img.header["sform_code"]))
    out_img.header.set_zooms(src_img.header.get_zooms()[:3])

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(job.output_path))
    logger.info(
        "wrote %s  shape=%s  labels=%s",
        job.output_path,
        data_out.shape,
        sorted(int(v) for v in np.unique(data_out)),
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map 192^3 uncertainty-segmentation predictions back onto the "
            "original MenGrowth NIfTI grid."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing {scan_id}/ subfolders with saved predictions.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=(
            "Source dataset root containing {patient_id}/{scan_id}/*.nii.gz "
            "(e.g. .../v5_final/MenGrowth-2025)."
        ),
    )
    parser.add_argument(
        "--source-filename",
        default="segmentation.nii.gz",
        help=(
            "Prediction file to invert (default: segmentation.nii.gz — "
            "multi-label BraTS ensemble). Use ensemble_mask.nii.gz for WT binary."
        ),
    )
    parser.add_argument(
        "--output-filename",
        default="seg.nii.gz",
        help="Destination filename inside each source scan folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List jobs without writing anything.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N scans (useful for smoke tests).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if not args.predictions_dir.is_dir():
        logger.error("predictions-dir does not exist: %s", args.predictions_dir)
        return 2
    if not args.dataset_dir.is_dir():
        logger.error("dataset-dir does not exist: %s", args.dataset_dir)
        return 2

    jobs = discover_jobs(
        args.predictions_dir,
        args.dataset_dir,
        source_filename=args.source_filename,
        output_filename=args.output_filename,
        overwrite=args.overwrite,
    )
    logger.info("discovered %d scans to invert", len(jobs))
    if args.limit is not None:
        jobs = jobs[: args.limit]
        logger.info("limiting to first %d", len(jobs))

    if args.dry_run:
        for j in jobs:
            logger.info(
                "dry-run: %s -> %s", j.prediction_path, j.output_path
            )
        return 0

    forward = build_forward_pipeline()
    n_ok = 0
    n_err = 0
    for j in jobs:
        try:
            invert_single(j, forward)
            n_ok += 1
        except Exception:
            n_err += 1
            logger.error("failed on %s:\n%s", j.scan_id, traceback.format_exc())

    logger.info("done: %d ok, %d failed", n_ok, n_err)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
