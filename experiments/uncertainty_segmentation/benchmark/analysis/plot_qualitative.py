"""Qualitative figure: T1n + GT TC + per-model predicted TC, all in the H5 192³ frame.

For each row's selected case (best/median/worst) we replicate the H5-build
pipeline (Orient RAS → CropForeground → Pad → CenterCrop 192³) on the
subject-space NIfTIs of competitor models so that *every* column shares one
canvas. Our LoRA ensemble predictions are already in that frame and are
loaded as-is.

Colour convention (per user spec):
    * GT TC: red filled (α=0.5) with a red contour (α=1.0).
    * Prediction TC (any model, including Ours): green filled (α=0.5) drawn
      on top of the GT, with a thin green contour.
    * Visually:
        - True positives (GT ∩ Pred): green on red → muted green/olive.
        - False negatives (GT ∖ Pred): red exposed → highlights missed tumour.
        - False positives (Pred ∖ GT): green on T1n → highlights over-predictions.
    * The metric value annotated above each cell is the full 3D score
      (Dice/HD95/lesion-recall). The displayed slice is the H5-frame axial
      slice with the largest GT TC area, so spurious distant components
      (e.g. the 304-voxel false-positive that drives HD95 to 133 mm on
      ``BraTS-MEN-00717-008``) may not appear in this slice — the
      annotation tells the truth, the slice gives intuition.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.ndimage import label as cc_label

from experiments.uncertainty_segmentation.plotting.inter_lora.style import save_figure
from experiments.uncertainty_segmentation.plotting.style import setup_style

from .h5_align import align_triplet
from .io import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_GT_ROOT,
    OURS_MODEL_ID,
    ModelEntry,
    load_ground_truth,
    load_prediction,
    load_t1n,
)
from .metrics import METRICS, label_mask

logger = logging.getLogger(__name__)

ROW_KEYS: tuple[str, str, str] = ("best", "median", "worst")
ROW_TITLES: dict[str, str] = {"best": "Best", "median": "Median", "worst": "Worst"}

# GT in red, predictions in green. Greens chosen for high contrast against
# the red GT and the gray T1n underlay. Both alphas at 0.5 so over-painting
# greens on reds yields a clearly mixed (olive-ish) hue at true positives,
# while pure red marks false negatives and pure green marks false positives.
GT_COLOR = "#E63946"
PRED_COLOR = "#2A9D8F"


def _normalize_t1n_slice(slc: np.ndarray) -> np.ndarray:
    """Min-max stretch a 2D T1n slice to [0, 1] using brain-region percentiles."""
    finite = slc[np.isfinite(slc)]
    if finite.size == 0:
        return np.zeros_like(slc)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        return np.zeros_like(slc)
    out = (slc - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


CONTOUR_LW: float = 0.8


def _paint_overlays(
    ax: plt.Axes,
    t1n_slice: np.ndarray,
    gt_tc_slice: np.ndarray,
    pred_tc_slice: np.ndarray | None,
) -> None:
    """Render the T1n + GT (red) + pred (green) stack on ``ax``."""
    from matplotlib.colors import to_rgb

    ax.set_facecolor("black")
    ax.imshow(
        _normalize_t1n_slice(t1n_slice).T,
        cmap="gray",
        origin="lower",
        interpolation="nearest",
    )
    gr, gg, gb = to_rgb(GT_COLOR)
    pr, pg, pb = to_rgb(PRED_COLOR)
    if gt_tc_slice.any():
        gt_rgba = np.zeros((*gt_tc_slice.T.shape, 4), dtype=float)
        gt_rgba[gt_tc_slice.T.astype(bool)] = [gr, gg, gb, 0.5]
        ax.imshow(gt_rgba, origin="lower", interpolation="nearest")
        ax.contour(
            gt_tc_slice.T.astype(np.uint8),
            levels=[0.5],
            colors=[(gr, gg, gb, 1.0)],
            linewidths=CONTOUR_LW,
            origin="lower",
        )
    if pred_tc_slice is not None and pred_tc_slice.any():
        pred_rgba = np.zeros((*pred_tc_slice.T.shape, 4), dtype=float)
        pred_rgba[pred_tc_slice.T.astype(bool)] = [pr, pg, pb, 0.5]
        ax.imshow(pred_rgba, origin="lower", interpolation="nearest")
        ax.contour(
            pred_tc_slice.T.astype(np.uint8),
            levels=[0.5],
            colors=[(pr, pg, pb, 1.0)],
            linewidths=CONTOUR_LW,
            origin="lower",
        )


def _strip_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _square_zoom_window(
    gt_tc_slice: np.ndarray,
    pred_tc_slice: np.ndarray | None,
    pad: int = 6,
) -> tuple[int, int, int, int] | None:
    """Find the bounding square around the largest GT/Pred disagreement.

    Returns ``(x_lo, x_hi, y_lo, y_hi)`` in display coordinates
    (i.e. after the ``.T`` transpose used by :func:`_paint_overlays`),
    or ``None`` if there is no error to highlight.
    """
    if pred_tc_slice is None:
        if not gt_tc_slice.any():
            return None
        error = gt_tc_slice
    else:
        error = gt_tc_slice ^ pred_tc_slice
    if not error.any():
        return None
    error_T = error.T
    cc, n = cc_label(error_T, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return None
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    k = int(sizes.argmax())
    rows, cols = np.where(cc == k)  # rows = display y, cols = display x
    if rows.size == 0:
        return None
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    side = max(rmax - rmin, cmax - cmin) + 2 * pad
    side = max(side, 16)
    cy = 0.5 * (rmin + rmax)
    cx = 0.5 * (cmin + cmax)
    H, W = error_T.shape
    half = side / 2
    y_lo = max(0, int(round(cy - half)))
    y_hi = min(H, int(round(cy + half)))
    x_lo = max(0, int(round(cx - half)))
    x_hi = min(W, int(round(cx + half)))
    if (y_hi - y_lo) < 6 or (x_hi - x_lo) < 6:
        return None
    return x_lo, x_hi, y_lo, y_hi


def _pick_inset_corner(
    gt_tc_slice: np.ndarray,
    pred_tc_slice: np.ndarray | None,
    image_shape_xy: tuple[int, int],
    inset_frac: float = 0.42,
) -> tuple[float, float, float, float]:
    """Choose an axes-fraction bbox for the inset that avoids the masks.

    The returned tuple is ``(x, y, w, h)`` suitable for
    :meth:`matplotlib.axes.Axes.inset_axes`. The corner farthest from the
    centroid of (GT ∪ Pred) is selected.
    """
    occupied = gt_tc_slice.copy()
    if pred_tc_slice is not None:
        occupied = occupied | pred_tc_slice
    H, W = image_shape_xy  # display y, display x
    if occupied.any():
        ys, xs = np.where(occupied.T)
        cy = float(ys.mean()) / H
        cx = float(xs.mean()) / W
    else:
        cx = cy = 0.5
    candidates = {
        "TL": (0.02, 1 - inset_frac - 0.02),
        "TR": (1 - inset_frac - 0.02, 1 - inset_frac - 0.02),
        "BL": (0.02, 0.02),
        "BR": (1 - inset_frac - 0.02, 0.02),
    }
    centers = {
        "TL": (candidates["TL"][0] + inset_frac / 2, candidates["TL"][1] + inset_frac / 2),
        "TR": (candidates["TR"][0] + inset_frac / 2, candidates["TR"][1] + inset_frac / 2),
        "BL": (candidates["BL"][0] + inset_frac / 2, candidates["BL"][1] + inset_frac / 2),
        "BR": (candidates["BR"][0] + inset_frac / 2, candidates["BR"][1] + inset_frac / 2),
    }
    best = max(candidates, key=lambda k: (centers[k][0] - cx) ** 2 + (centers[k][1] - cy) ** 2)
    x0, y0 = candidates[best]
    return x0, y0, inset_frac, inset_frac


def _draw_cell(
    ax: plt.Axes,
    t1n_slice: np.ndarray,
    gt_tc_slice: np.ndarray,
    pred_tc_slice: np.ndarray | None,
    title: str,
    zoom: bool = False,
) -> None:
    """Render one panel cell, optionally with a zoom inset on the largest error."""
    _paint_overlays(ax, t1n_slice, gt_tc_slice, pred_tc_slice)
    _strip_axis(ax)
    ax.set_title(title, color="white", fontsize=7, pad=2)
    if not zoom:
        return
    window = _square_zoom_window(gt_tc_slice, pred_tc_slice)
    if window is None:
        return
    x_lo, x_hi, y_lo, y_hi = window
    H, W = gt_tc_slice.T.shape
    bounds = _pick_inset_corner(gt_tc_slice, pred_tc_slice, (H, W))
    axins = ax.inset_axes(bounds)
    _paint_overlays(axins, t1n_slice, gt_tc_slice, pred_tc_slice)
    axins.set_xlim(x_lo, x_hi)
    axins.set_ylim(y_lo, y_hi)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("white")
        spine.set_linewidth(0.6)
    ax.indicate_inset_zoom(axins, edgecolor="white", linewidth=0.5, alpha=0.9)


def _aligned_external_pred(case_id: str, pred_path: Path, gt_root: Path) -> np.ndarray:
    """Push a subject-space external prediction into the 192³ H5 frame."""
    t1n_path = gt_root / case_id / f"{case_id}-t1n.nii.gz"
    seg_path = gt_root / case_id / f"{case_id}-seg.nii.gz"
    triplet = align_triplet(t1n_path=t1n_path, seg_path=seg_path, pred_path=pred_path)
    return triplet["pred"]


def make_qualitative_figure(
    metric: str,
    cases: dict[str, str],
    entries: list[ModelEntry],
    df_metric: pd.DataFrame,
    gt_root: Path = DEFAULT_GT_ROOT,
    zoom_inset: bool = False,
) -> Figure:
    """Build the qualitative panel for one metric, all columns in 192³.

    Args:
        metric: which metric this figure is keyed on (used only for the score
            annotation in cell titles).
        cases: ``{"best": case_id, "median": case_id, "worst": case_id}``.
        entries: ordered ModelEntry list (external models first, ``Ours`` last).
        df_metric: long-format DataFrame restricted to one metric column.
        gt_root: subject-space NIfTI root.
    """
    setup_style()
    n_cols = len(entries)
    n_rows = len(ROW_KEYS)
    fig_w = max(n_cols * 1.4, 6.0)
    fig_h = n_rows * 1.5 + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("black")
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        left=0.08,    # space for vertical row labels
        right=0.99,
        top=0.93,
        bottom=0.12,  # space for the bottom legend
        hspace=0.22,
        wspace=0.04,
    )

    cached_h5_t1n: dict[str, np.ndarray] = {}
    cached_h5_gt: dict[str, np.ndarray] = {}

    def _get_h5_pair(case_id: str) -> tuple[np.ndarray, np.ndarray]:
        if case_id not in cached_h5_t1n:
            cached_h5_t1n[case_id] = load_t1n(case_id, frame="h5_192", gt_root=gt_root)
            cached_h5_gt[case_id] = load_ground_truth(case_id, frame="h5_192", gt_root=gt_root)[0]
        return cached_h5_t1n[case_id], cached_h5_gt[case_id]

    for row_idx, row_key in enumerate(ROW_KEYS):
        case_id = cases.get(row_key)
        if case_id is None:
            continue
        try:
            t1n_192, gt_192 = _get_h5_pair(case_id)
        except FileNotFoundError as exc:
            logger.warning("qual: skipping case %s: %s", case_id, exc)
            continue

        gt_tc_full = label_mask(gt_192, "TC")
        if gt_tc_full.any():
            z = int(np.argmax(gt_tc_full.sum(axis=(0, 1))))
        else:
            z = t1n_192.shape[2] // 2
        t1n_slice = t1n_192[:, :, z]
        gt_tc_slice = gt_tc_full[:, :, z]

        row_label = f"{ROW_TITLES[row_key]} ({case_id.replace('BraTS-MEN-', '')})"

        for col_idx, entry in enumerate(entries):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            try:
                if entry.model_id == OURS_MODEL_ID:
                    pred_arr, _ = load_prediction(entry, case_id)  # already 192³
                else:
                    pred_arr = _aligned_external_pred(
                        case_id, entry.prediction_path(case_id), gt_root
                    )
                pred_tc_slice = label_mask(pred_arr, "TC")[:, :, z]
            except (FileNotFoundError, ValueError) as exc:
                logger.warning(
                    "qual: skipping %s on %s: %s", entry.model_id, case_id, exc
                )
                pred_tc_slice = None

            score_str = ""
            if pred_tc_slice is not None:
                row_df = df_metric[
                    (df_metric["model"] == entry.model_id)
                    & (df_metric["case_id"] == case_id)
                    & (df_metric["label"] == "TC")
                ]
                if len(row_df):
                    val = float(row_df[metric].iloc[0])
                    if np.isfinite(val):
                        if metric == "hd95":
                            score_str = f"\nHD95={val:.1f} mm"
                        elif metric == "lesion_recall":
                            score_str = f"\nrec={val:.2f}"
                        else:
                            score_str = f"\nDice={val:.2f}"

            cell_title = f"{entry.model_id}{score_str}"
            _draw_cell(
                ax, t1n_slice, gt_tc_slice, pred_tc_slice, cell_title, zoom=zoom_inset
            )
            if col_idx == 0:
                ax.set_ylabel(
                    row_label,
                    color="white",
                    fontsize=9,
                    rotation=90,
                    labelpad=8,
                )

    legend_handles = [
        Patch(facecolor=GT_COLOR, alpha=0.5, edgecolor=GT_COLOR, label="Ground Truth TC"),
        Patch(facecolor=PRED_COLOR, alpha=0.5, edgecolor=PRED_COLOR, label="Predicted TC"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor="white",
        handlelength=2.0,
        columnspacing=2.5,
    )
    return fig


def write_qualitative(
    entries: list[ModelEntry],
    df: pd.DataFrame,
    best_median_worst: dict[str, dict[str, str]],
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    gt_root: Path = DEFAULT_GT_ROOT,
    zoom_dice: bool = False,
) -> list[Path]:
    """Write one qualitative figure per metric (best/median/worst rows).

    When ``zoom_dice`` is true, the dice qualitative figure also shows an
    inset that zooms into the largest error region of each cell.
    """
    out_dir = analysis_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for metric in METRICS:
        cases = best_median_worst.get(metric, {})
        if not cases:
            logger.warning("qual: no best/median/worst for metric=%s; skipping", metric)
            continue
        sub_cases = {k: cases[k] for k in ROW_KEYS if k in cases}
        df_metric = df[["model", "case_id", "label", metric]]
        fig = make_qualitative_figure(
            metric=metric,
            cases=sub_cases,
            entries=entries,
            df_metric=df_metric,
            gt_root=gt_root,
            zoom_inset=(zoom_dice and metric == "dice"),
        )
        for ext in (".pdf", ".png"):
            target = out_dir / f"qual_{metric}{ext}"
            save_figure(
                fig,
                target,
                title=f"BraTS-MEN qualitative comparison: {metric}",
                description=(
                    f"Best/median/worst on TC by mean across models; metric={metric}. "
                    "All columns aligned to the H5 192³ canvas."
                ),
                transparent=False,
            )
            paths.append(target)
        plt.close(fig)
    logger.info("plot: wrote %d qualitative figures to %s", len(paths), out_dir)
    return paths


# Kept for backwards compatibility with any direct importers; the cached
# slices file is no longer consulted (we recompute the slice from H5 GT TC
# inside make_qualitative_figure to guarantee 192³-frame alignment).
def _legacy_slices_loader(analysis_root: Path) -> dict[str, dict[str, int]]:
    p = analysis_root / "cache" / "case_t1n_slices.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())
