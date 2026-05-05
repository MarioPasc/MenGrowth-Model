"""Tumor-size sensitivity figure: TC Dice vs effective tumour diameter.

For each segmentation model we fit a linear mixed-effects regression

    Dice = β₀ + β₁ · d_eff + u_patient + ε,    u_patient ~ N(0, σ²_p)

where ``d_eff`` is the effective spherical diameter in mm derived from the
GT TC volume (``d_eff = (6 V / π)^{1/3}`` with ``V`` in mm³). The grouping
factor is the patient id parsed from the case id, which keeps the very small
number of multi-timepoint patients (2 out of 147 in the BraTS-MEN test
split) properly clustered. With single-observation groups the random
intercept collapses to zero and the slope/intercept estimates coincide with
OLS, so the design is robust to the typical case where each patient
contributes a single scan.

Outputs:
    cache/tumor_sizes.json
    cache/size_sensitivity_summary.json
    figures/size_sensitivity_dice.{pdf,png}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from experiments.uncertainty_segmentation.plotting.inter_lora.style import (
    DOUBLE_COL_MM,
    MM_TO_INCH,
    save_figure,
)
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
from .metrics import label_mask
from .plot_qualitative import (
    GT_COLOR,
    _normalize_t1n_slice,
    _pick_inset_corner,
    _square_zoom_window,
)

logger = logging.getLogger(__name__)

CASE_RE = re.compile(r"BraTS-MEN-(\d+)-\d+")
TUMOR_SIZES_JSON = "tumor_sizes.json"
SUMMARY_JSON = "size_sensitivity_summary.json"

# Up to ~6 distinct, perceptually separable model colours.
_PALETTE: tuple[str, ...] = (
    "#E76F51",  # warm red-orange
    "#F4A261",  # amber
    "#264653",  # dark teal
    "#9D4EDD",  # purple
    "#E9C46A",  # mustard
    "#1D3557",  # navy
)
_OURS_COLOR = "#2A9D8F"  # the production teal we already use elsewhere


def model_colors(model_order: list[str]) -> dict[str, str]:
    """Assign one palette colour per model, reserving teal for ``Ours``."""
    out: dict[str, str] = {}
    palette_idx = 0
    for model in model_order:
        if model == OURS_MODEL_ID:
            out[model] = _OURS_COLOR
        else:
            out[model] = _PALETTE[palette_idx % len(_PALETTE)]
            palette_idx += 1
    return out


def patient_of(case_id: str) -> str:
    m = CASE_RE.match(case_id)
    return m.group(1) if m is not None else case_id


def _effective_diameter_mm(tc_voxels: int, spacing_mm: float = 1.0) -> float:
    """Spherical-equivalent diameter in mm for a TC mass of given volume."""
    if tc_voxels <= 0:
        return 0.0
    volume_mm3 = float(tc_voxels) * (spacing_mm**3)
    return float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0))


def compute_tumor_sizes(
    case_ids: list[str],
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    force: bool = False,
) -> dict[str, dict[str, float]]:
    """Cache (and return) ``{case_id: {tc_voxels, d_eff_mm}}`` from the H5 GT."""
    cache = analysis_root / "cache" / TUMOR_SIZES_JSON
    cache.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, float]] = {}
    if cache.exists() and not force:
        existing = json.loads(cache.read_text())
    todo = [c for c in case_ids if c not in existing]
    if todo:
        logger.info("size: computing TC volume for %d new cases", len(todo))
        for case_id in todo:
            try:
                gt, spacing = load_ground_truth(case_id, frame="h5_192")
            except FileNotFoundError as exc:
                logger.warning("size: skipping %s: %s", case_id, exc)
                continue
            tc_voxels = int(label_mask(gt, "TC").sum())
            sx = float(spacing[0]) if spacing else 1.0
            existing[case_id] = {
                "tc_voxels": tc_voxels,
                "d_eff_mm": _effective_diameter_mm(tc_voxels, sx),
            }
        cache.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return existing


def _fit_one(df_one: pd.DataFrame) -> dict:
    """Fit ``Dice ~ d_eff`` MixedLM with random intercept per patient."""
    df_one = df_one.dropna(subset=["dice", "d_eff_mm"]).copy()
    df_one = df_one[df_one["d_eff_mm"] > 0]
    md = smf.mixedlm("dice ~ d_eff_mm", df_one, groups=df_one["patient_id"])
    res = md.fit(method="lbfgs", reml=True)
    fe = res.fe_params
    cov = res.cov_params().loc[fe.index, fe.index]
    return {
        "result": res,
        "beta0": float(fe["Intercept"]),
        "beta1": float(fe["d_eff_mm"]),
        "p_intercept": float(res.pvalues["Intercept"]),
        "p_slope": float(res.pvalues["d_eff_mm"]),
        "cov": cov.values.copy(),
        "n": int(len(df_one)),
        "x_min": float(df_one["d_eff_mm"].min()),
        "x_max": float(df_one["d_eff_mm"].max()),
    }


def _predict_with_ci(
    fit: dict,
    x_grid: np.ndarray,
    ci_z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = np.array([fit["beta0"], fit["beta1"]])
    cov = fit["cov"]
    X = np.column_stack([np.ones_like(x_grid), x_grid])
    mean = X @ beta
    var = np.einsum("ij,jk,ik->i", X, cov, X)
    se = np.sqrt(np.clip(var, 0.0, None))
    return mean, mean - ci_z * se, mean + ci_z * se


def _format_formula(fit: dict) -> str:
    sign = "-" if fit["beta1"] < 0 else "+"
    return (
        rf"$\widehat{{D}} = {fit['beta0']:.3f} {sign} "
        rf"{abs(fit['beta1']):.4f}\,d_{{\mathrm{{eff}}}}$, "
        rf"$p = {fit['p_slope']:.2g}$"
    )


def _draw_size_patch(
    ax: plt.Axes,
    case_id: str,
    entries: list[ModelEntry],
    colors: dict[str, str],
    gt_root: Path,
    d_eff_mm: float,
    add_zoom: bool = True,
) -> None:
    """Render the T1n slice + GT contour + each model's TC contour for a case.

    Optionally adds a zoom inset focusing on the largest GT/Ours disagreement.
    """
    t1n = load_t1n(case_id, frame="h5_192", gt_root=gt_root)
    gt, _ = load_ground_truth(case_id, frame="h5_192", gt_root=gt_root)
    gt_tc = label_mask(gt, "TC")
    z = int(np.argmax(gt_tc.sum(axis=(0, 1)))) if gt_tc.any() else t1n.shape[2] // 2

    ax.set_facecolor("black")
    ax.imshow(
        _normalize_t1n_slice(t1n[:, :, z]).T,
        cmap="gray",
        origin="lower",
        interpolation="nearest",
    )
    if gt_tc[:, :, z].any():
        ax.contour(
            gt_tc[:, :, z].T.astype(np.uint8),
            levels=[0.5],
            colors=[GT_COLOR],
            linewidths=0.6,
            origin="lower",
        )

    pred_slices: dict[str, np.ndarray] = {}
    for entry in entries:
        try:
            if entry.model_id == OURS_MODEL_ID:
                arr, _ = load_prediction(entry, case_id)
            else:
                triplet = align_triplet(
                    t1n_path=gt_root / case_id / f"{case_id}-t1n.nii.gz",
                    seg_path=gt_root / case_id / f"{case_id}-seg.nii.gz",
                    pred_path=entry.prediction_path(case_id),
                )
                arr = triplet["pred"]
            pred_slices[entry.model_id] = label_mask(arr, "TC")[:, :, z]
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("size patch: skipping %s on %s: %s", entry.model_id, case_id, exc)
    for model_id, mask in pred_slices.items():
        if not mask.any():
            continue
        ax.contour(
            mask.T.astype(np.uint8),
            levels=[0.5],
            colors=[colors[model_id]],
            linewidths=0.7,
            origin="lower",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("white")
        spine.set_linewidth(0.5)
    ax.set_title(
        f"{case_id.replace('BraTS-MEN-', '')} — {d_eff_mm:.0f} mm",
        color="white",
        fontsize=6,
        pad=1.5,
    )

    if not add_zoom:
        return
    ours_mask = pred_slices.get(OURS_MODEL_ID)
    if ours_mask is None:
        return
    window = _square_zoom_window(gt_tc[:, :, z], ours_mask)
    if window is None:
        return
    x_lo, x_hi, y_lo, y_hi = window
    H, W = t1n[:, :, z].T.shape
    bounds = _pick_inset_corner(gt_tc[:, :, z], ours_mask, (H, W), inset_frac=0.40)
    axins = ax.inset_axes(bounds)
    axins.imshow(
        _normalize_t1n_slice(t1n[:, :, z]).T,
        cmap="gray",
        origin="lower",
        interpolation="nearest",
    )
    if gt_tc[:, :, z].any():
        axins.contour(
            gt_tc[:, :, z].T.astype(np.uint8),
            levels=[0.5],
            colors=[GT_COLOR],
            linewidths=0.5,
            origin="lower",
        )
    for model_id, mask in pred_slices.items():
        if not mask.any():
            continue
        axins.contour(
            mask.T.astype(np.uint8),
            levels=[0.5],
            colors=[colors[model_id]],
            linewidths=0.55,
            origin="lower",
        )
    axins.set_xlim(x_lo, x_hi)
    axins.set_ylim(y_lo, y_hi)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.4)
    ax.indicate_inset_zoom(axins, edgecolor="white", linewidth=0.35, alpha=0.85)


def _add_size_patches(
    fig: Figure,
    ax: plt.Axes,
    sizes: dict[str, dict[str, float]],
    entries: list[ModelEntry],
    colors: dict[str, str],
    gt_root: Path,
    bin_width_mm: float = 10.0,
    bottom_y_frac: float = 0.04,
    max_height_frac: float = 0.30,
) -> None:
    """Embed one MRI patch per 10 mm-wide x-bin, anchored along the bottom."""
    fig_w, fig_h = fig.get_size_inches()
    bbox = ax.get_position()
    ax_w_in = bbox.width * fig_w
    ax_h_in = bbox.height * fig_h
    xlim = ax.get_xlim()
    x_left_data, x_right_data = xlim  # (large, small) thanks to invert
    x_range = abs(x_left_data - x_right_data)
    if x_range <= 0:
        return

    bin_w_frac = bin_width_mm / x_range
    patch_h_frac = min((bin_w_frac * ax_w_in) / ax_h_in, max_height_frac)

    # Walk integer-aligned bin edges from the smaller d_eff up to the larger.
    x_lo, x_hi = sorted([x_left_data, x_right_data])
    edge = bin_width_mm * np.ceil(x_lo / bin_width_mm)
    bins: list[tuple[float, float]] = []
    while edge + bin_width_mm <= x_hi + 1e-6:
        bins.append((float(edge), float(edge + bin_width_mm)))
        edge += bin_width_mm

    for lo, hi in bins:
        center = (lo + hi) / 2
        cases = [c for c, s in sizes.items() if lo <= s.get("d_eff_mm", -1) < hi]
        if not cases:
            continue
        case_id = min(cases, key=lambda c: abs(sizes[c]["d_eff_mm"] - center))
        # x_left_data corresponds to axes-fraction 0; x_right_data to 1.
        x_frac_left = (x_left_data - hi) / (x_left_data - x_right_data)
        bounds = (x_frac_left, bottom_y_frac, bin_w_frac, patch_h_frac)
        axins = ax.inset_axes(bounds)
        _draw_size_patch(axins, case_id, entries, colors, gt_root, sizes[case_id]["d_eff_mm"])


def make_size_sensitivity_figure(
    df: pd.DataFrame,
    sizes: dict[str, dict[str, float]],
    model_order: list[str],
    entries: list[ModelEntry] | None = None,
    gt_root: Path = DEFAULT_GT_ROOT,
    label: str = "TC",
    metric: str = "dice",
    add_patches: bool = True,
) -> tuple[Figure, dict[str, dict]]:
    """Scatter + LME regression of TC Dice against effective tumour diameter."""
    setup_style()
    sub = df[df["label"] == label][["model", "case_id", metric]].copy()
    sub = sub.rename(columns={metric: "dice"})
    sub["d_eff_mm"] = sub["case_id"].map(lambda c: sizes.get(c, {}).get("d_eff_mm", np.nan))
    sub["patient_id"] = sub["case_id"].map(patient_of)
    sub = sub.dropna(subset=["dice", "d_eff_mm"]).reset_index(drop=True)

    colors = model_colors(model_order)
    fig_w = DOUBLE_COL_MM * MM_TO_INCH
    fig_h = 110.0 * MM_TO_INCH  # extra height to host the bottom legend
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Reserve bottom whitespace for the outside (model→colour) legend.
    ax = fig.add_axes([0.10, 0.22, 0.85, 0.72])

    fits: dict[str, dict] = {}
    inside_handles: list = []
    inside_labels: list[str] = []

    x_lo_global = float(sub["d_eff_mm"].min())
    x_hi_global = float(sub["d_eff_mm"].max())
    x_grid_global = np.linspace(x_lo_global, x_hi_global, 200)

    for model in model_order:
        df_m = sub[sub["model"] == model]
        if df_m.empty:
            continue
        color = colors[model]
        ax.scatter(
            df_m["d_eff_mm"].values,
            df_m["dice"].values,
            c=color,
            s=14,
            alpha=0.5,
            linewidths=0,
            zorder=2,
        )
        try:
            fit = _fit_one(df_m)
        except Exception as exc:  # pragma: no cover (defensive)
            logger.warning("LME failed for %s: %s", model, exc)
            continue
        fits[model] = {k: v for k, v in fit.items() if k not in ("result", "cov")}
        x_grid = np.linspace(fit["x_min"], fit["x_max"], 200)
        mean, lo, hi = _predict_with_ci(fit, x_grid)
        ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=3)
        line, = ax.plot(x_grid, mean, color=color, linewidth=1.6, zorder=4)
        inside_handles.append(line)
        inside_labels.append(_format_formula(fit))

    ax.set_xlabel(r"GT TC effective diameter $d_{\mathrm{eff}}$ (mm)", fontsize=8)
    ax.set_ylabel("TC Dice coefficient", fontsize=8)
    # Invert the x-axis so larger tumours appear first (left → right is large → small).
    x_lo = x_lo_global * 0.95 if x_lo_global > 0 else 0
    x_hi = x_hi_global * 1.02
    ax.set_xlim(x_hi, x_lo)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=7)

    # Inside legend: anchored at data coordinates (d_eff = 70 mm, Dice = 0.8)
    # via the upper-left corner of the legend box.
    inside_legend = ax.legend(
        inside_handles,
        inside_labels,
        loc="upper left",
        bbox_to_anchor=(70.0, 0.8),
        bbox_transform=ax.transData,
        fontsize=7,
        frameon=True,
        framealpha=0.92,
        edgecolor="#999999",
        title="LME fits",
        title_fontsize=7,
    )
    ax.add_artist(inside_legend)

    if add_patches and entries is not None:
        _add_size_patches(
            fig=fig,
            ax=ax,
            sizes=sizes,
            entries=entries,
            colors=colors,
            gt_root=gt_root,
        )

    # Outside legend: colour → model, anchored below the axes.
    outside_handles = [
        Patch(facecolor=colors[m], edgecolor=colors[m], label=m) for m in model_order if m in fits
    ]
    fig.legend(
        outside_handles,
        [m for m in model_order if m in fits],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(outside_handles), 6),
        fontsize=8,
        frameon=False,
        title="Model",
        title_fontsize=8,
        columnspacing=2.0,
        handlelength=2.0,
    )
    return fig, fits


def write_size_sensitivity(
    df: pd.DataFrame,
    model_order: list[str],
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    entries: list[ModelEntry] | None = None,
    gt_root: Path = DEFAULT_GT_ROOT,
    force_sizes: bool = False,
) -> list[Path]:
    """Compute tumour sizes (cached) and write the size-sensitivity figure.

    When ``entries`` is provided, the figure is annotated with one MRI patch
    per 10-mm-wide x-bin (each patch overlays GT and per-model TC contours and
    a zoom inset on Ours's largest disagreement).
    """
    out_dir = analysis_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_ids = sorted(df["case_id"].unique())
    sizes = compute_tumor_sizes(case_ids, analysis_root=analysis_root, force=force_sizes)
    fig, fits = make_size_sensitivity_figure(
        df, sizes, model_order, entries=entries, gt_root=gt_root
    )
    paths: list[Path] = []
    for ext in (".pdf", ".png"):
        target = out_dir / f"size_sensitivity_dice{ext}"
        save_figure(
            fig,
            target,
            title="BraTS-MEN size sensitivity (TC Dice vs effective diameter)",
            description="LME fits per segmentation model; random intercept per patient.",
        )
        paths.append(target)
    plt.close(fig)
    summary_path = analysis_root / "cache" / SUMMARY_JSON
    summary_path.write_text(json.dumps(fits, indent=2, sort_keys=True, default=float))
    logger.info(
        "plot: wrote size-sensitivity figure (LME for %d models) to %s",
        len(fits),
        out_dir,
    )
    return paths
