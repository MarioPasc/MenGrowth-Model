"""Qual1 — Multi-rank slice grid with MRI overlay.

Generates 8 figure variants (vertical + horizontal for each):
  (i)   best_brats / best_brats_horizontal
  (ii)  worst_brats / worst_brats_horizontal
  (iii) mengrowth_low_uncertainty / mengrowth_low_uncertainty_horizontal
  (iv)  mengrowth_high_uncertainty / mengrowth_high_uncertainty_horizontal

BraTS variants merge GT+Ensemble into a single overlay column:
  - Ground truth only → orange
  - Ensemble only → blue
  - Overlap → green (solid)

The uncertainty column overlays a voxelwise uncertainty map on the T1n MRI
background. The map is selected via ``map_type`` in the figure config:
  - ``entropy`` (default) → predictive binary entropy of the ensemble-mean
    probability on the meningioma channel, ``H[mean_m p_m]``; magma colormap.
  - ``variance`` → inter-member predictive standard deviation summed over
    channels, ``sum_c sqrt(Var_m[p_m,c])``; plasma colormap.
In both cases the overlay alpha is proportional to the normalised value, so
low-uncertainty regions stay transparent (raw MRI anatomy visible).
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..io_layer import (
    InterLoraData,
    RankRun,
    compute_scan_mean_entropy,
    compute_voxelwise_entropy_slice,
    compute_voxelwise_variance_slice,
    find_largest_tumor_slice,
    get_nifti_affine,
    load_nifti_slice,
)
from ..style import (
    DOUBLE_COL_MM,
    MM_TO_INCH,
)

logger = logging.getLogger(__name__)

_OVERLAY_ALPHA: float = 0.7
_VARIANCE_CMAP: str = "plasma_r"
_ENTROPY_CMAP: str = "magma"
_REF_CMAP: str = "gray"
_PAD_VOXELS: int = 15
_T1N_CHANNEL: int = 2

# Default uncertainty-map settings (overridable via figure config).
_DEFAULT_MAP_TYPE: str = "entropy"
_DEFAULT_ENTROPY_CHANNEL: int = 0  # BSF ch0 = meningioma mass (TC)
_DEFAULT_N_MEMBERS: int = 20

# Merged overlay colours
_GT_ONLY_RGBA: tuple[float, ...] = (0.9, 0.5, 0.0, _OVERLAY_ALPHA)  # orange
_ENS_ONLY_RGBA: tuple[float, ...] = (0.0, 0.45, 0.7, _OVERLAY_ALPHA)  # blue
_OVERLAP_RGBA: tuple[float, ...] = (0.0, 0.7, 0.3, _OVERLAY_ALPHA)  # green

# Fallback for binary masks (MenGrowth ensemble)
_ENSEMBLE_RGBA: tuple[float, ...] = (0.0, 0.45, 0.7, _OVERLAY_ALPHA)

# Variant definitions
VARIANT_BEST_BRATS = "best_brats"
VARIANT_WORST_BRATS = "worst_brats"
VARIANT_MENGROWTH_LOW_UNC = "mengrowth_low_uncertainty"
VARIANT_MENGROWTH_HIGH_UNC = "mengrowth_high_uncertainty"

_VERTICAL_VARIANTS: list[str] = [
    VARIANT_BEST_BRATS,
    VARIANT_WORST_BRATS,
    VARIANT_MENGROWTH_LOW_UNC,
    VARIANT_MENGROWTH_HIGH_UNC,
]

_HORIZONTAL_VARIANTS: list[str] = [f"{v}_horizontal" for v in _VERTICAL_VARIANTS]

ALL_VARIANTS: list[str] = _VERTICAL_VARIANTS + _HORIZONTAL_VARIANTS


def _uncertainty_label(map_type: str) -> str:
    """Human-readable column/row label for the uncertainty map."""
    return "Entropy" if map_type == "entropy" else "Variance"


def _uncertainty_cmap(map_type: str) -> str:
    """Matplotlib colormap name for the uncertainty map."""
    return _ENTROPY_CMAP if map_type == "entropy" else _VARIANCE_CMAP


def _uncertainty_cbar_label(map_type: str) -> str:
    """Colorbar label (LaTeX mathtext) for the uncertainty map."""
    if map_type == "entropy":
        return r"$\mathcal{H}\left[\bar{p}_{\mathrm{men}}(\mathbf{x})\right]$ (nats)"
    return r"$\sum_c \sqrt{\mathrm{Var}_m\left[p_{m,c}(\mathbf{x})\right]}$"


def _compute_uncertainty_slice(
    scan_dir: Path,
    slice_idx: int,
    map_type: str,
    entropy_channel: int,
    n_members: int,
) -> np.ndarray:
    """Dispatch to the entropy or variance voxelwise slice estimator."""
    if map_type == "entropy":
        return compute_voxelwise_entropy_slice(
            scan_dir,
            slice_idx,
            channel=entropy_channel,
            n_members=n_members,
        )
    return compute_voxelwise_variance_slice(scan_dir, slice_idx, n_members=n_members)


def _is_horizontal(variant: str) -> bool:
    return variant.endswith("_horizontal")


def _base_variant(variant: str) -> str:
    return variant.removesuffix("_horizontal")


def _is_brats(variant: str) -> bool:
    base = _base_variant(variant)
    return base in (VARIANT_BEST_BRATS, VARIANT_WORST_BRATS)


def _find_scan_dir(run: RankRun, scan_id: str) -> Path | None:
    """Locate scan directory under predictions/."""
    if run.predictions_dir is None:
        return None
    for subdir in [
        run.predictions_dir / scan_id,
        run.predictions_dir / "brats_men_test" / scan_id,
    ]:
        if subdir.is_dir():
            return subdir
    for d in run.predictions_dir.iterdir():
        if d.is_dir() and d.name.startswith(scan_id[:15]):
            return d
    return None


def _resolve_h5_path(
    scan_id: str,
    brats_h5: Path | None,
    mengrowth_h5: Path | None,
) -> Path | None:
    if scan_id.startswith("BraTS-MEN") and brats_h5 is not None and brats_h5.exists():
        return brats_h5
    if scan_id.startswith("MenGrowth") and mengrowth_h5 is not None and mengrowth_h5.exists():
        return mengrowth_h5
    return None


def _load_t1n_slice(
    scan_id: str,
    slice_idx: int,
    brats_h5: Path | None = None,
    mengrowth_h5: Path | None = None,
) -> np.ndarray | None:
    h5_path = _resolve_h5_path(scan_id, brats_h5, mengrowth_h5)
    if h5_path is None:
        return None
    try:
        import h5py

        with h5py.File(h5_path, "r") as f:
            if "scan_ids" in f:
                scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
            elif "subject_ids" in f:
                scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["subject_ids"][:]]
            else:
                return None

            if scan_id not in scan_ids:
                return None
            idx = scan_ids.index(scan_id)
            img = f["images"][idx, _T1N_CHANNEL, :, :, slice_idx]
            return np.array(img, dtype=np.float32)
    except Exception:
        logger.debug("Failed to load T1n for %s from H5", scan_id, exc_info=True)
        return None


def _make_merged_overlay(
    gt_slice: np.ndarray,
    ens_slice: np.ndarray,
) -> np.ndarray:
    """Create merged GT+Ensemble overlay: GT-only=orange, Ens-only=blue, overlap=green."""
    h, w = gt_slice.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    gt_labels = np.round(gt_slice).astype(int)
    if gt_labels.ndim == 3:
        gt_labels = gt_labels[..., 0]
    gt_mask = gt_labels > 0

    ens_mask = ens_slice > 0.5
    if ens_mask.ndim == 3:
        ens_mask = ens_mask[..., 0]

    gt_only = gt_mask & ~ens_mask
    ens_only = ens_mask & ~gt_mask
    overlap = gt_mask & ens_mask

    overlay[gt_only] = _GT_ONLY_RGBA
    overlay[ens_only] = _ENS_ONLY_RGBA
    overlay[overlap] = _OVERLAP_RGBA

    return overlay


def _make_overlay_binary(mask_slice: np.ndarray) -> np.ndarray:
    if mask_slice.ndim == 3:
        mask_slice = mask_slice[..., 0]
    h, w = mask_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    binary = mask_slice > 0.5
    overlay[binary] = _ENSEMBLE_RGBA
    return overlay


def _compute_crop_bbox(
    masks: list[np.ndarray],
    pad: int = _PAD_VOXELS,
) -> tuple[slice, slice]:
    union = np.zeros_like(
        masks[0][..., 0] if masks[0].ndim == 3 else masks[0],
        dtype=bool,
    )
    for m in masks:
        if m.ndim == 3:
            union |= m.sum(axis=-1) > 0.5
        else:
            union |= m > 0.5

    rows = np.any(union, axis=1)
    cols = np.any(union, axis=0)
    if not rows.any():
        return slice(None), slice(None)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = union.shape
    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)
    return slice(rmin, rmax + 1), slice(cmin, cmax + 1)


def _add_scale_bar(ax: plt.Axes, affine: np.ndarray, length_mm: float = 10.0) -> None:
    """Add a white scale bar sitting 1 pixel below the image, text underneath."""
    voxel_size = abs(affine[0, 0])
    if voxel_size < 1e-6:
        return
    bar_voxels = length_mm / voxel_size
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    # Place bar 1 pixel below the bottom edge of the image (imshow y-axis inverted)
    y_pos = ylim[0] + 1
    x_start = xlim[0] + (xlim[1] - xlim[0]) * 0.03
    ax.plot(
        [x_start, x_start + bar_voxels],
        [y_pos, y_pos],
        color="white",
        linewidth=2,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        x_start + bar_voxels / 2,
        y_pos + 4,
        f"{length_mm:.0f} mm",
        color="white",
        fontsize=6,
        ha="center",
        va="top",
        clip_on=False,
    )


def _find_rank_with_predictions(data: InterLoraData, prefix: str = "") -> RankRun | None:
    for run in sorted(data.ranks, key=lambda r: r.rank, reverse=True):
        if run.rank == 0 or run.predictions_dir is None:
            continue
        for subdir in [run.predictions_dir, run.predictions_dir / "brats_men_test"]:
            if not subdir.is_dir():
                continue
            for d in subdir.iterdir():
                if not d.is_dir() or (prefix and not d.name.startswith(prefix)):
                    continue
                if (d / "ensemble_mask.nii.gz").exists():
                    return run
    return None


def _scans_with_predictions(run: RankRun, prefix: str) -> set[str]:
    result: set[str] = set()
    if run.predictions_dir is None:
        return result
    for subdir in [run.predictions_dir, run.predictions_dir / "brats_men_test"]:
        if not subdir.is_dir():
            continue
        for d in subdir.iterdir():
            if d.is_dir() and d.name.startswith(prefix) and (d / "ensemble_mask.nii.gz").exists():
                result.add(d.name)
    return result


def _cache_path(out_root: Path, variant: str, key: str) -> Path:
    """Return path for a cached intermediate CSV/JSON."""
    cache_dir = out_root / "data" / "qual1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{variant}_{key}.json"


def _save_cache(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _select_scan(
    data: InterLoraData,
    variant: str,
    *,
    map_type: str = _DEFAULT_MAP_TYPE,
    entropy_channel: int = _DEFAULT_ENTROPY_CHANNEL,
    n_members: int = _DEFAULT_N_MEMBERS,
    force_recompute: bool = False,
) -> tuple[str | None, bool]:
    """Select scan_id for a variant. Returns (scan_id, has_gt).

    The MenGrowth ``low/high_uncertainty`` variants pick the scans with the
    minimum / maximum value of the *displayed* uncertainty metric: mean
    predictive entropy when ``map_type == "entropy"``, or inter-member tumour
    volume standard deviation when ``map_type == "variance"``.
    """
    base = _base_variant(variant)

    cache_p = _cache_path(data.out_root, base, f"selected_scan_{map_type}")
    if not force_recompute:
        cached = _load_cache(cache_p)
        if cached is not None:
            return cached.get("scan_id"), cached.get("has_gt", False)

    scan_id: str | None = None
    has_gt = False

    if base in (VARIANT_BEST_BRATS, VARIANT_WORST_BRATS):
        ref_run = _find_rank_with_predictions(data, "BraTS-MEN")
        if ref_run is None:
            return None, False
        valid_scans = _scans_with_predictions(ref_run, "BraTS-MEN")
        if not valid_scans:
            return None, False

        ens = ref_run.ensemble_dice
        brats = ens[ens["scan_id"].str.startswith("BraTS-MEN") & ens["scan_id"].isin(valid_scans)]
        if brats.empty:
            return None, False

        if base == VARIANT_BEST_BRATS:
            idx = brats["dice_mean"].idxmax()
        else:
            idx = brats["dice_mean"].idxmin()
        scan_id = str(brats.loc[idx, "scan_id"])
        has_gt = True

    elif base in (VARIANT_MENGROWTH_LOW_UNC, VARIANT_MENGROWTH_HIGH_UNC):
        ref_run = _find_rank_with_predictions(data, "MenGrowth")
        if ref_run is None:
            return None, False
        valid_scans = _scans_with_predictions(ref_run, "MenGrowth")
        if not valid_scans:
            return None, False

        stats_cache = _cache_path(data.out_root, "mengrowth", f"{map_type}_select_stats")
        stats_cached = None if force_recompute else _load_cache(stats_cache)

        if stats_cached is not None:
            scan_stats = stats_cached
        else:
            scan_stats = _mengrowth_selection_stats(
                ref_run,
                sorted(valid_scans),
                map_type=map_type,
                entropy_channel=entropy_channel,
                n_members=n_members,
            )
            _save_cache(stats_cache, scan_stats)

        if not scan_stats:
            scan_id = sorted(valid_scans)[0]
        elif map_type == "entropy":
            # Pick by the displayed metric directly (mean predictive entropy).
            if base == VARIANT_MENGROWTH_LOW_UNC:
                scan_id = min(scan_stats, key=lambda k: scan_stats[k]["score"])
            else:
                scan_id = max(scan_stats, key=lambda k: scan_stats[k]["score"])
        else:
            # Variance mode: prefer larger tumours, then pick by volume std.
            volumes = [s.get("mean_vol", 0.0) for s in scan_stats.values()]
            vol_median = float(np.median(volumes)) if volumes else 0.0
            large = {k: v for k, v in scan_stats.items() if v.get("mean_vol", 0.0) >= vol_median}
            pool = large if large else scan_stats
            if base == VARIANT_MENGROWTH_LOW_UNC:
                scan_id = min(pool, key=lambda k: pool[k]["score"])
            else:
                scan_id = max(pool, key=lambda k: pool[k]["score"])
        has_gt = False

    if scan_id is not None:
        _save_cache(cache_p, {"scan_id": scan_id, "has_gt": has_gt})

    return scan_id, has_gt


def _mengrowth_selection_stats(
    ref_run: RankRun,
    scan_ids: list[str],
    *,
    map_type: str,
    entropy_channel: int,
    n_members: int,
) -> dict[str, dict[str, float]]:
    """Per-scan selection statistic for the MenGrowth uncertainty variants.

    For ``entropy`` mode ``score`` is the mean predictive entropy inside the
    predicted meningioma mask. For ``variance`` mode ``score`` is the
    inter-member tumour volume standard deviation, with ``mean_vol`` kept so
    larger tumours can be preferred.
    """
    scan_stats: dict[str, dict[str, float]] = {}

    if map_type == "entropy":
        for sid in scan_ids:
            scan_dir = _find_scan_dir(ref_run, sid)
            if scan_dir is None:
                continue
            score = compute_scan_mean_entropy(scan_dir, entropy_channel, n_members)
            if np.isfinite(score):
                scan_stats[sid] = {"score": float(score)}
            gc.collect()
        return scan_stats

    import nibabel as nib

    sample_members = [0, 4, 9, 14, 19]
    for sid in scan_ids:
        scan_dir = _find_scan_dir(ref_run, sid)
        if scan_dir is None:
            continue
        member_vols: list[float] = []
        for m in sample_members:
            mask_path = scan_dir / f"member_{m}_mask.nii.gz"
            if not mask_path.exists():
                continue
            img = nib.load(str(mask_path))
            member_vols.append(float((img.get_fdata() > 0.5).sum()))
            del img
        if len(member_vols) >= 2:
            scan_stats[sid] = {
                "score": float(np.std(member_vols)),
                "mean_vol": float(np.mean(member_vols)),
            }
        gc.collect()
    return scan_stats


def _render_variant(
    data: InterLoraData,
    scan_id: str,
    has_gt: bool,
    variant: str,
    brats_h5: Path | None = None,
    mengrowth_h5: Path | None = None,
    slice_idx: int | None = None,
    *,
    map_type: str = _DEFAULT_MAP_TYPE,
    entropy_channel: int = _DEFAULT_ENTROPY_CHANNEL,
    n_members: int = _DEFAULT_N_MEMBERS,
) -> Figure | None:
    """Render one variant's grid figure.

    BraTS: columns = Overlay (merged GT+Ens) | Uncertainty-on-MRI
    MenGrowth: columns = Ensemble | Uncertainty-on-MRI
    The uncertainty column shows predictive entropy or inter-member variance
    depending on ``map_type``. Horizontal layout transposes rows↔columns.
    """
    horizontal = _is_horizontal(variant)
    is_brats = _is_brats(variant)
    unc_label = _uncertainty_label(map_type)
    unc_cmap_name = _uncertainty_cmap(map_type)

    runs_with_preds: list[RankRun] = []
    for run in data.ranks:
        if run.rank == 0:
            continue
        sd = _find_scan_dir(run, scan_id)
        if sd is not None and (sd / "ensemble_mask.nii.gz").exists():
            runs_with_preds.append(run)

    if not runs_with_preds:
        logger.warning("No ranks have predictions for %s — skipping", scan_id)
        return None

    n_ranks = len(runs_with_preds)
    n_content_cols = 2  # overlay/ensemble + uncertainty

    if slice_idx is None:
        ref_run = runs_with_preds[-1]
        scan_dir = _find_scan_dir(ref_run, scan_id)
        mask_path = scan_dir / "ensemble_mask.nii.gz"
        slice_idx = find_largest_tumor_slice(mask_path)

    # Collect masks for bounding box
    all_masks: list[np.ndarray] = []
    for run in runs_with_preds:
        scan_dir = _find_scan_dir(run, scan_id)
        mask_path = scan_dir / "ensemble_mask.nii.gz"
        if mask_path.exists():
            m = load_nifti_slice(mask_path, slice_idx)
            all_masks.append(m)

    if not all_masks:
        logger.warning("No masks for %s — skipping", scan_id)
        return None

    crop_r, crop_c = _compute_crop_bbox(all_masks)
    del all_masks
    gc.collect()

    # Load T1n background
    t1n_slice = _load_t1n_slice(scan_id, slice_idx, brats_h5, mengrowth_h5)
    if t1n_slice is not None:
        t1n_cropped = t1n_slice[crop_r, crop_c]
        nz = t1n_cropped[t1n_cropped > 0]
        p1 = float(np.percentile(nz, 1)) if len(nz) > 0 else 0
        p99 = float(np.percentile(nz, 99)) if len(nz) > 0 else 1
    else:
        t1n_cropped = None
        p1, p99 = 0, 1

    is_mengrowth = not is_brats
    if horizontal:
        base_fontsize = 9 if is_mengrowth else 8
    else:
        base_fontsize = 17 if is_mengrowth else 16

    if horizontal:
        col_labels = [f"$r={r.rank}$" for r in runs_with_preds]
        row_labels = ["Overlay", unc_label] if is_brats else ["Ensemble", unc_label]
        n_grid_rows = n_content_cols
        n_grid_cols = n_ranks

        fig_w = DOUBLE_COL_MM * MM_TO_INCH
        col_w = fig_w / (n_grid_cols + 1.5)
        row_h = col_w * 0.9
        fig_h = row_h * n_grid_rows + 0.6

        fig, axes = plt.subplots(
            n_grid_rows,
            n_grid_cols,
            figsize=(fig_w, fig_h),
            gridspec_kw={"wspace": 0.02, "hspace": 0.15},
        )
        fig.subplots_adjust(bottom=0.12)
        if axes.ndim == 1:
            axes = axes[np.newaxis, :]
    else:
        row_labels_list = [f"$r={r.rank}$" for r in runs_with_preds]
        col_labels = ["Overlay", unc_label] if is_brats else ["Ensemble", unc_label]
        n_grid_rows = n_ranks
        n_grid_cols = n_content_cols

        fig_w = DOUBLE_COL_MM * MM_TO_INCH
        row_h = fig_w / n_grid_cols * 0.7
        fig_h = row_h * n_grid_rows

        fig, axes = plt.subplots(
            n_grid_rows,
            n_grid_cols,
            figsize=(fig_w, fig_h),
            gridspec_kw={"wspace": 0.02, "hspace": 0.12},
        )
        fig.subplots_adjust(bottom=0.06)
        if axes.ndim == 1:
            axes = axes[np.newaxis, :]

    fig.patch.set_facecolor("black")

    for ax in axes.ravel():
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Column/row headers
    if horizontal:
        for j, label in enumerate(col_labels):
            axes[0, j].set_title(label, color="white", fontsize=base_fontsize, pad=4)
        for i, label in enumerate(row_labels):
            axes[i, 0].set_ylabel(
                label,
                color="white",
                fontsize=base_fontsize - 1,
                rotation=90,
                labelpad=10,
                va="center",
            )
    else:
        for j, label in enumerate(col_labels):
            axes[0, j].set_title(label, color="white", fontsize=base_fontsize, pad=4)
        for i, label in enumerate(row_labels_list):
            axes[i, 0].set_ylabel(
                label,
                color="white",
                fontsize=base_fontsize - 1,
                rotation=0,
                labelpad=25,
                va="center",
            )

    affine_found = None
    # Two-pass uncertainty: collect data first, render after computing global vmax
    unc_deferred: list[tuple[plt.Axes, np.ndarray]] = []

    for rank_i, run in enumerate(runs_with_preds):
        scan_dir = _find_scan_dir(run, scan_id)

        if horizontal:
            overlay_ax = axes[0, rank_i]
            unc_ax = axes[1, rank_i]
        else:
            overlay_ax = axes[rank_i, 0]
            unc_ax = axes[rank_i, 1]

        # Overlay / Ensemble column (T1n background + coloured mask)
        if t1n_cropped is not None:
            overlay_ax.imshow(t1n_cropped, cmap=_REF_CMAP, vmin=p1, vmax=p99)

        if is_brats and scan_dir is not None:
            gt_path = scan_dir / "segmentation.nii.gz"
            mask_path = scan_dir / "ensemble_mask.nii.gz"
            if gt_path.exists() and mask_path.exists():
                gt_slc = load_nifti_slice(gt_path, slice_idx)
                gt_cropped = gt_slc[crop_r, crop_c]
                mask_slc = load_nifti_slice(mask_path, slice_idx)
                mask_cropped = mask_slc[crop_r, crop_c]
                merged = _make_merged_overlay(gt_cropped, mask_cropped)
                overlay_ax.imshow(merged)
                if affine_found is None:
                    affine_found = get_nifti_affine(gt_path)
        elif scan_dir is not None:
            mask_path = scan_dir / "ensemble_mask.nii.gz"
            if mask_path.exists():
                mask_slc = load_nifti_slice(mask_path, slice_idx)
                mask_cropped = mask_slc[crop_r, crop_c]
                overlay = _make_overlay_binary(mask_cropped)
                overlay_ax.imshow(overlay)
                if affine_found is None:
                    affine_found = get_nifti_affine(mask_path)

        # Uncertainty column — show T1n background now, overlay the map later
        if t1n_cropped is not None:
            unc_ax.imshow(t1n_cropped, cmap=_REF_CMAP, vmin=p1, vmax=p99)

        if scan_dir is not None:
            try:
                unc_map = _compute_uncertainty_slice(
                    scan_dir,
                    slice_idx,
                    map_type,
                    entropy_channel,
                    n_members,
                )
                unc_cropped = unc_map[crop_r, crop_c]
                unc_deferred.append((unc_ax, unc_cropped))
            except (MemoryError, Exception):
                logger.warning("%s map failed for %s r=%d", unc_label, scan_id, run.rank)
                gc.collect()

        gc.collect()

    # Deferred uncertainty rendering: proportional alpha so low-uncertainty
    # regions stay transparent (MRI visible) rather than darkened.
    unc_vmax = 1.0
    if unc_deferred:
        all_unc = np.concatenate([v.ravel() for _, v in unc_deferred])
        unc_vmax = float(np.percentile(all_unc, 99))
        if unc_vmax < 1e-12:
            unc_vmax = 1.0
        unc_cmap = plt.cm.get_cmap(unc_cmap_name)
        for unc_ax, unc_cropped in unc_deferred:
            unc_norm = np.clip(unc_cropped / unc_vmax, 0.0, 1.0)
            unc_rgba = unc_cmap(unc_norm)
            # Alpha scales with normalized uncertainty: 0 → transparent, 1 → full overlay
            unc_rgba[..., 3] = unc_norm * _OVERLAY_ALPHA
            unc_ax.imshow(unc_rgba)

    # Scale bar on first cell
    if affine_found is not None:
        _add_scale_bar(axes[0, 0], affine_found)

    # Segmentation legend at the TOP of the figure
    if is_brats:
        legend_patches = [
            mpatches.Patch(facecolor=_GT_ONLY_RGBA[:3], alpha=_OVERLAY_ALPHA, label="Ground Truth"),
            mpatches.Patch(facecolor=_ENS_ONLY_RGBA[:3], alpha=_OVERLAY_ALPHA, label="Ensemble"),
            mpatches.Patch(facecolor=_OVERLAP_RGBA[:3], alpha=_OVERLAY_ALPHA, label="Overlap"),
        ]
        fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=3,
            frameon=False,
            fontsize=base_fontsize - 1,
            labelcolor="white",
            bbox_to_anchor=(0.5, 1.02),
        )

    # Uncertainty colorbar at the BOTTOM, spanning full figure width
    if unc_deferred:
        from matplotlib.colors import Normalize as MplNormalize

        fig.canvas.draw()
        sm = plt.cm.ScalarMappable(
            cmap=unc_cmap_name,
            norm=MplNormalize(vmin=0, vmax=unc_vmax),
        )
        sm.set_array([])

        if horizontal:
            left_ax = axes[-1, 0].get_position()
            right_ax = axes[-1, -1].get_position()
            cbar_x = left_ax.x0
            cbar_w = right_ax.x1 - left_ax.x0
            cbar_y = left_ax.y0 - 0.07
            cbar_h = 0.025
        else:
            # Span BOTH columns (Overlay + Uncertainty)
            left_pos = axes[-1, 0].get_position()
            right_pos = axes[-1, -1].get_position()
            cbar_x = left_pos.x0
            cbar_w = right_pos.x1 - left_pos.x0
            cbar_y = left_pos.y0 - 0.04
            cbar_h = 0.012

        cbar_fontsize = min(base_fontsize - 1, 10)
        ax_cbar = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
        cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
        cbar.set_label(
            _uncertainty_cbar_label(map_type),
            fontsize=cbar_fontsize,
            color="white",
        )
        cbar.ax.tick_params(labelsize=cbar_fontsize - 1, colors="white")
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(0.5)

    return fig


def plot(data: InterLoraData, config: dict[str, Any]) -> Figure | None:
    """Generate qual1 slice grid for a specific variant.

    Config keys:
        variant: One of ALL_VARIANTS (default: best_brats).
        brats_h5: Path to BraTS-MEN H5 for T1n background.
        mengrowth_h5: Path to MenGrowth H5 for T1n background.
        map_type: ``"entropy"`` (default) or ``"variance"`` — selects the
            voxelwise uncertainty map shown in the right-hand column.
        entropy_channel: Probability channel for entropy mode (default 0 =
            BSF ch0 = meningioma mass).
        n_members: Ensemble size used when averaging per-member probabilities.
        force_recompute: If True, ignore cached scan selections.

    Returns:
        Figure or None if scan not found.
    """
    variant = config.get("variant", VARIANT_BEST_BRATS)
    brats_h5_str = config.get("brats_h5")
    mengrowth_h5_str = config.get("mengrowth_h5")
    brats_h5 = Path(brats_h5_str) if brats_h5_str else None
    mengrowth_h5 = Path(mengrowth_h5_str) if mengrowth_h5_str else None
    force_recompute = config.get("force_recompute", False)

    map_type = str(config.get("map_type", _DEFAULT_MAP_TYPE)).lower()
    if map_type not in ("entropy", "variance"):
        logger.warning("qual1: unknown map_type=%r — falling back to entropy", map_type)
        map_type = "entropy"
    entropy_channel = int(config.get("entropy_channel", _DEFAULT_ENTROPY_CHANNEL))
    n_members = int(config.get("n_members", _DEFAULT_N_MEMBERS))

    scan_id, has_gt = _select_scan(
        data,
        variant,
        map_type=map_type,
        entropy_channel=entropy_channel,
        n_members=n_members,
        force_recompute=force_recompute,
    )
    if scan_id is None:
        logger.warning("qual1 %s: no suitable scan found", variant)
        return None

    logger.info(
        "qual1 %s: scan_id=%s, has_gt=%s, map_type=%s",
        variant,
        scan_id,
        has_gt,
        map_type,
    )
    return _render_variant(
        data,
        scan_id,
        has_gt,
        variant,
        brats_h5=brats_h5,
        mengrowth_h5=mengrowth_h5,
        map_type=map_type,
        entropy_channel=entropy_channel,
        n_members=n_members,
    )
