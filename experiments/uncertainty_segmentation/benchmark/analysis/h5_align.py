"""Push a subject-space (240×240×155, LPS) NIfTI into the H5 192³ frame.

Replicates the canonical pipeline used to build the project's BraTS-MEN H5
(documented in ``src/growth/data/transforms.py:207``):

    Orient(RAS) → CropForeground(source=t1n) → SpatialPad(192³) → CenterCrop(192³)

The cropping bbox is taken from the T1n channel (non-zero foreground), so the
same bbox can be applied to (image, segmentation, prediction) triplets and they
will align voxel-for-voxel with each other in the resulting 192³ canvas.

This is the only piece of the analysis pipeline that needs the canonical
bbox; the per-case metric computation in ``compute.py`` evaluates each
prediction in its native frame instead.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    SpatialPadd,
)

ROI_192: tuple[int, int, int] = (192, 192, 192)


def _build_pipeline(keys: list[str]) -> Compose:
    """Construct the canonical H5-build transform on the requested dict keys.

    The first key is treated as the foreground reference (image-like).
    """
    if not keys:
        raise ValueError("at least one key required")
    return Compose(
        [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            CropForegroundd(keys=keys, source_key=keys[0], allow_smaller=False),
            SpatialPadd(keys=keys, spatial_size=ROI_192),
            ResizeWithPadOrCropd(keys=keys, spatial_size=ROI_192),
        ]
    )


def align_triplet(
    t1n_path: Path,
    seg_path: Path | None,
    pred_path: Path | None,
) -> dict[str, np.ndarray]:
    """Run the H5-build pipeline on a (T1n, GT, prediction) triplet.

    Args:
        t1n_path: subject-space T1n NIfTI used to derive the foreground bbox.
        seg_path: optional subject-space GT NIfTI (integer labels).
        pred_path: optional subject-space prediction NIfTI (integer labels).

    Returns:
        Dictionary with up to three numpy arrays of shape ``(192, 192, 192)``:
        ``t1n`` (float32), ``seg`` (int8), ``pred`` (int8). Keys whose source
        path was ``None`` are omitted from the returned dictionary.
    """
    keys = ["t1n"]
    payload: dict[str, str] = {"t1n": str(t1n_path)}
    if seg_path is not None:
        keys.append("seg")
        payload["seg"] = str(seg_path)
    if pred_path is not None:
        keys.append("pred")
        payload["pred"] = str(pred_path)

    pipeline = _build_pipeline(keys)
    out = pipeline(payload)

    def _strip(arr) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]
        return arr

    result: dict[str, np.ndarray] = {"t1n": _strip(out["t1n"]).astype(np.float32)}
    if "seg" in out:
        result["seg"] = _strip(out["seg"]).astype(np.int8)
    if "pred" in out:
        result["pred"] = _strip(out["pred"]).astype(np.int8)
    return result


def align_array_to_h5(
    arr: np.ndarray,
    affine: np.ndarray,
    t1n_path: Path,
    is_label: bool,
) -> np.ndarray:
    """Push an in-memory array (sharing T1n's subject-space affine) into 192³.

    Useful when the prediction NIfTI does not carry a meaningful affine but
    its voxel grid matches the subject T1n. Wraps the array as a transient
    NIfTI and reuses :func:`align_triplet`.
    """
    img = nib.Nifti1Image(arr.astype(np.int16 if is_label else np.float32), affine)
    tmp = Path("/tmp") / f"_align_{abs(hash((arr.tobytes()[:32], is_label)))}.nii.gz"
    nib.save(img, tmp)
    try:
        triplet = align_triplet(
            t1n_path=t1n_path,
            seg_path=tmp if is_label else None,
            pred_path=None if is_label else tmp,
        )
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return triplet["seg"] if is_label else triplet["pred"]
