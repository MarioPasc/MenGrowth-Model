"""Fig 13: Boundary Disagreement (NIfTI-based).

Three-panel qualitative illustration of where ensemble members disagree:
  a) Ensemble contour on MRI slice
  b) Per-member contours overlaid
  c) Voxel-wise agreement map (0..M)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import C_ENSEMBLE

logger = logging.getLogger(__name__)


def _load_nifti_data(path: Path) -> np.ndarray | None:
    """Load a NIfTI file, returning the numpy array or None."""
    try:
        import nibabel as nib
    except ImportError:
        logger.warning("nibabel not installed — cannot load NIfTI files")
        return None

    if not path.exists():
        logger.warning("NIfTI file not found: %s", path)
        return None

    img = nib.load(str(path))
    return np.asarray(img.dataobj)


def _find_best_slice(mask_3d: np.ndarray, axis: str = "axial") -> int:
    """Find the slice with the largest tumour cross-section.

    Args:
        mask_3d: 3D binary mask array.
        axis: Slice orientation ("axial", "coronal", "sagittal").

    Returns:
        Slice index with maximum foreground area.
    """
    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    ax = axis_map.get(axis, 2)
    sums = mask_3d.sum(axis=tuple(i for i in range(3) if i != ax))
    return int(np.argmax(sums))


def _get_slice(vol: np.ndarray, idx: int, axis: str = "axial") -> np.ndarray:
    """Extract a 2D slice from a 3D volume."""
    if axis == "axial":
        return vol[:, :, idx]
    elif axis == "coronal":
        return vol[:, idx, :]
    else:  # sagittal
        return vol[idx, :, :]


def _load_mri_background(
    data: EnsembleResultsData,
    scan_id: str,
) -> np.ndarray | None:
    """Load T1ce MRI background for a scan.

    Tries local MenGrowth H5 first, falls back to ensemble_probs.nii.gz.
    """
    pred_dir = data.predictions_dir / scan_id

    # Try ensemble_probs as fallback (channel 1 = WT probability)
    probs_path = pred_dir / "ensemble_probs.nii.gz"
    probs = _load_nifti_data(probs_path)
    if probs is not None:
        # ensemble_probs is [D,H,W,C] or [D,H,W] — use as-is for background
        if probs.ndim == 4:
            return probs[..., 1]  # WT channel
        return probs

    return None


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | None:
    """Generate the boundary disagreement figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Ignored (three-panel figure creates its own axes).

    Returns:
        The Figure object, or None if prediction data is unavailable.
    """
    if not data.has_predictions or not data.sample_scans:
        logger.warning("No per-member predictions available — skipping Fig 13")
        return None

    try:
        import nibabel as nib  # noqa: F401
    except ImportError:
        logger.warning("nibabel not installed — skipping Fig 13")
        return None

    figsize = config.get("figsize", [8, 4])
    scan_id = config.get("scan_id") or data.sample_scans[0]
    slice_axis = config.get("slice_axis", "axial")

    pred_dir = data.predictions_dir / scan_id
    if not pred_dir.exists():
        logger.warning("Prediction dir not found for %s — skipping Fig 13",
                       scan_id)
        return None

    # Load ensemble mask
    ens_mask = _load_nifti_data(pred_dir / "ensemble_mask.nii.gz")
    if ens_mask is None:
        return None

    # Determine number of members
    member_masks: list[np.ndarray] = []
    m = 0
    while True:
        mask = _load_nifti_data(pred_dir / f"member_{m}_mask.nii.gz")
        if mask is None:
            break
        member_masks.append(mask)
        m += 1

    if len(member_masks) < 2:
        logger.warning("Fewer than 2 member masks found — skipping Fig 13")
        return None

    M = len(member_masks)

    # WT mask: channel 1 in the 4D mask, or the full mask if 3D
    def _get_wt(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 4:
            return (mask[..., 1] > 0.5).astype(np.uint8)
        return (mask > 0).astype(np.uint8)

    ens_wt = _get_wt(ens_mask)
    member_wts = [_get_wt(m) for m in member_masks]

    # Find best slice
    slice_idx = _find_best_slice(ens_wt, slice_axis)

    # Load MRI background
    bg = _load_mri_background(data, scan_id)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=figsize)

    # Get 2D slices
    ens_slice = _get_slice(ens_wt, slice_idx, slice_axis)
    member_slices = [_get_slice(mw, slice_idx, slice_axis) for mw in member_wts]

    # Background image
    if bg is not None:
        bg_slice = _get_slice(bg, slice_idx, slice_axis)
    else:
        bg_slice = np.zeros_like(ens_slice, dtype=float)

    # --- Panel A: MRI with ensemble contour ---
    ax_a.imshow(bg_slice.T, cmap="gray", origin="lower", aspect="equal")
    ax_a.contour(ens_slice.T, levels=[0.5], colors=[C_ENSEMBLE],
                 linewidths=1.5)
    ax_a.set_title("a) Ensemble prediction", fontweight="bold", fontsize=9)
    ax_a.axis("off")

    # --- Panel B: MRI with all member contours ---
    ax_b.imshow(bg_slice.T, cmap="gray", origin="lower", aspect="equal")
    cmap_members = plt.cm.tab10
    for i, ms in enumerate(member_slices):
        color = cmap_members(i / max(M - 1, 1))
        ax_b.contour(ms.T, levels=[0.5], colors=[color], linewidths=0.7,
                     alpha=0.8)
    ax_b.set_title("b) Per-member contours", fontweight="bold", fontsize=9)
    ax_b.axis("off")

    # --- Panel C: Agreement map ---
    agreement = np.sum(member_slices, axis=0)
    im = ax_c.imshow(agreement.T, cmap="YlOrRd", origin="lower",
                     aspect="equal", vmin=0, vmax=M)
    ax_c.contour(ens_slice.T, levels=[0.5], colors=["white"],
                 linewidths=0.8, linestyles="--")
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.8, pad=0.02)
    cbar.set_label(f"Agreement (0\u2013{M})", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax_c.set_title("c) Agreement map", fontweight="bold", fontsize=9)
    ax_c.axis("off")

    fig.suptitle(f"Boundary disagreement \u2014 {scan_id} "
                 f"({slice_axis} slice {slice_idx})",
                 fontweight="bold", fontsize=10, y=1.02)
    fig.tight_layout()
    return fig
