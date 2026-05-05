"""Per-case, per-label segmentation metrics: Dice, HD95, lesion recall.

Labels (BraTS-MEN integer convention):
    NETC = 1
    SNFH = 2
    ET   = 3
    TC   = NETC | ET (= label in {1, 3})
    WT   = (label > 0)

All metrics return ``NaN`` when the comparison is ill-defined (e.g. empty GT
for lesion recall). The empty/empty Dice case returns NaN by convention so
that easy negatives do not artificially inflate model means.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.ndimage import label as cc_label

logger = logging.getLogger(__name__)

LABELS: tuple[str, ...] = ("NETC", "SNFH", "ET", "TC", "WT")
METRICS: tuple[str, ...] = ("dice", "hd95", "lesion_recall")
HIGHER_IS_BETTER: dict[str, bool] = {"dice": True, "hd95": False, "lesion_recall": True}


def label_mask(volume: np.ndarray, label: str) -> np.ndarray:
    """Return a boolean mask for one of the five named labels."""
    if label == "NETC":
        return volume == 1
    if label == "SNFH":
        return volume == 2
    if label == "ET":
        return volume == 3
    if label == "TC":
        return (volume == 1) | (volume == 3)
    if label == "WT":
        return volume > 0
    raise ValueError(f"Unknown label: {label}")


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Standard Dice on boolean masks. NaN when both masks are empty."""
    g = gt.sum()
    p = pred.sum()
    if g == 0 and p == 0:
        return float("nan")
    inter = float(np.logical_and(gt, pred).sum())
    return 2.0 * inter / float(g + p)


def hd95(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Symmetric 95th-percentile Hausdorff distance in mm.

    NaN when either mask is empty (HD is undefined). Uses a chamfer distance
    transform via ``scipy.ndimage.distance_transform_edt`` for an exact result
    on isotropic-sampled grids.
    """
    if not gt.any() or not pred.any():
        return float("nan")
    from scipy.ndimage import distance_transform_edt

    # Surface voxels = mask voxels with at least one non-mask 6-neighbour.
    def _surface(mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(mask, dtype=bool)
        out[1:, :, :] |= mask[1:, :, :] & ~mask[:-1, :, :]
        out[:-1, :, :] |= mask[:-1, :, :] & ~mask[1:, :, :]
        out[:, 1:, :] |= mask[:, 1:, :] & ~mask[:, :-1, :]
        out[:, :-1, :] |= mask[:, :-1, :] & ~mask[:, 1:, :]
        out[:, :, 1:] |= mask[:, :, 1:] & ~mask[:, :, :-1]
        out[:, :, :-1] |= mask[:, :, :-1] & ~mask[:, :, 1:]
        # voxels on the volume border that belong to the mask are also surface
        for axis in range(3):
            slc_lo = [slice(None)] * 3
            slc_hi = [slice(None)] * 3
            slc_lo[axis] = 0
            slc_hi[axis] = mask.shape[axis] - 1
            out[tuple(slc_lo)] |= mask[tuple(slc_lo)]
            out[tuple(slc_hi)] |= mask[tuple(slc_hi)]
        return out

    s_gt = _surface(gt)
    s_pred = _surface(pred)
    if not s_gt.any() or not s_pred.any():
        return float("nan")

    # Distance from each voxel to the nearest surface point of the OTHER mask,
    # in physical units.
    dt_gt = distance_transform_edt(~s_gt, sampling=spacing)
    dt_pred = distance_transform_edt(~s_pred, sampling=spacing)

    d_pred_to_gt = dt_gt[s_pred]
    d_gt_to_pred = dt_pred[s_gt]

    p95_a = float(np.percentile(d_pred_to_gt, 95))
    p95_b = float(np.percentile(d_gt_to_pred, 95))
    return max(p95_a, p95_b)


def lesion_recall(gt: np.ndarray, pred: np.ndarray, mode: Literal["any_overlap"] = "any_overlap") -> float:
    """Instance-level recall: fraction of GT connected components hit by ``pred``.

    A 26-connected component in ``gt`` counts as recovered if it shares at
    least one voxel with ``pred``. Returns NaN when the GT is empty (no
    lesions to recall).
    """
    if not gt.any():
        return float("nan")
    structure = np.ones((3, 3, 3), dtype=bool)
    labelled, n_components = cc_label(gt, structure=structure)
    if n_components == 0:
        return float("nan")
    hit = 0
    for k in range(1, n_components + 1):
        component = labelled == k
        if np.logical_and(component, pred).any():
            hit += 1
    return hit / n_components


def compute_case_metrics(
    gt_volume: np.ndarray,
    pred_volume: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[dict[str, float | str]]:
    """Compute (dice, hd95, lesion_recall) for the five labels.

    Returns:
        A list of rows ``{label, dice, hd95, lesion_recall}`` ready for
        long-format Parquet storage.
    """
    rows: list[dict[str, float | str]] = []
    for lbl in LABELS:
        g = label_mask(gt_volume, lbl)
        p = label_mask(pred_volume, lbl)
        rows.append(
            {
                "label": lbl,
                "dice": dice(g, p),
                "hd95": hd95(g, p, spacing=spacing),
                "lesion_recall": lesion_recall(g, p),
            }
        )
    return rows
