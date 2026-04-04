"""Fig 14: Uncertainty Heatmap Overlay.

Spatial distribution of epistemic uncertainty (mutual information and
predictive entropy) overlaid on the MRI.  High MI at the tumour boundary
demonstrates that the ensemble "knows what it doesn't know."
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

logger = logging.getLogger(__name__)


def _load_nifti_data(path: Path) -> np.ndarray | None:
    """Load a NIfTI file, returning the numpy array or None."""
    try:
        import nibabel as nib
    except ImportError:
        return None
    if not path.exists():
        return None
    return np.asarray(nib.load(str(path)).dataobj)


def _find_best_slice(vol_3d: np.ndarray, axis: str = "axial") -> int:
    """Find the slice with the maximum signal."""
    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    ax = axis_map.get(axis, 2)
    sums = vol_3d.sum(axis=tuple(i for i in range(3) if i != ax))
    return int(np.argmax(sums))


def _get_slice(vol: np.ndarray, idx: int, axis: str = "axial") -> np.ndarray:
    """Extract a 2D slice from a 3D volume."""
    if axis == "axial":
        return vol[:, :, idx]
    elif axis == "coronal":
        return vol[:, idx, :]
    else:
        return vol[idx, :, :]


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure | None:
    """Generate the uncertainty overlay figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Ignored (two-panel figure creates its own axes).

    Returns:
        The Figure object, or None if prediction data is unavailable.
    """
    if not data.has_predictions or not data.sample_scans:
        logger.warning("No per-member predictions available — skipping Fig 14")
        return None

    try:
        import nibabel as nib  # noqa: F401
    except ImportError:
        logger.warning("nibabel not installed — skipping Fig 14")
        return None

    figsize = config.get("figsize", [7, 3])
    scan_id = config.get("scan_id") or data.sample_scans[0]

    pred_dir = data.predictions_dir / scan_id
    if not pred_dir.exists():
        logger.warning("Prediction dir not found for %s — skipping Fig 14",
                       scan_id)
        return None

    # Load entropy and MI maps
    entropy_vol = _load_nifti_data(pred_dir / "entropy.nii.gz")
    mi_vol = _load_nifti_data(pred_dir / "mutual_information.nii.gz")

    if entropy_vol is None and mi_vol is None:
        logger.warning("No entropy/MI NIfTI found for %s — skipping Fig 14",
                       scan_id)
        return None

    # Extract WT channel if 4D (channel 1)
    def _extract_wt(vol: np.ndarray | None) -> np.ndarray | None:
        if vol is None:
            return None
        if vol.ndim == 4:
            return vol[..., 1]
        return vol

    entropy_3d = _extract_wt(entropy_vol)
    mi_3d = _extract_wt(mi_vol)

    # Load ensemble mask for contour reference
    ens_mask = _load_nifti_data(pred_dir / "ensemble_mask.nii.gz")
    ens_wt = None
    if ens_mask is not None:
        ens_wt = (ens_mask[..., 1] > 0.5).astype(np.uint8) if ens_mask.ndim == 4 else (ens_mask > 0).astype(np.uint8)

    # Load background
    bg_vol = _load_nifti_data(pred_dir / "ensemble_probs.nii.gz")
    if bg_vol is not None and bg_vol.ndim == 4:
        bg_3d = bg_vol[..., 1]
    elif bg_vol is not None:
        bg_3d = bg_vol
    else:
        # Use MI/entropy map extent for zero background
        ref = entropy_3d if entropy_3d is not None else mi_3d
        bg_3d = np.zeros(ref.shape, dtype=float)

    # Find best slice (use MI if available, else entropy)
    ref_map = mi_3d if mi_3d is not None else entropy_3d
    slice_idx = _find_best_slice(ref_map, "axial")

    # Determine which panels to show
    panels = []
    if entropy_3d is not None:
        panels.append(("Predictive entropy (nats)", entropy_3d, "hot"))
    if mi_3d is not None:
        panels.append(("Mutual information (nats)", mi_3d, "inferno"))

    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]

    for ax_i, (title, vol_3d, cmap) in zip(axes, panels):
        bg_slice = _get_slice(bg_3d, slice_idx, "axial")
        unc_slice = _get_slice(vol_3d, slice_idx, "axial")

        ax_i.imshow(bg_slice.T, cmap="gray", origin="lower", aspect="equal")

        # Mask out near-zero uncertainty for cleaner overlay
        unc_masked = np.ma.masked_where(unc_slice < 1e-6, unc_slice)
        im = ax_i.imshow(unc_masked.T, cmap=cmap, origin="lower",
                         aspect="equal", alpha=0.6,
                         vmin=0, vmax=float(np.nanpercentile(unc_slice, 99)))

        # Ensemble contour reference
        if ens_wt is not None:
            ens_slice = _get_slice(ens_wt, slice_idx, "axial")
            ax_i.contour(ens_slice.T, levels=[0.5], colors=["white"],
                         linewidths=0.8, linestyles="--")

        cbar = fig.colorbar(im, ax=ax_i, shrink=0.8, pad=0.02)
        cbar.set_label(title.split("(")[1].rstrip(")"), fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        ax_i.set_title(title, fontweight="bold", fontsize=9)
        ax_i.axis("off")

    fig.suptitle(f"Uncertainty overlay \u2014 {scan_id} (axial slice {slice_idx})",
                 fontweight="bold", fontsize=10, y=1.02)
    fig.tight_layout()
    return fig
