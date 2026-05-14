"""Real-patient figure: a BMA win, with the segmentation-variance overlay.

Patient: MenGrowth-0048 -- the cleanest case where Ensemble BMA beats the
homoscedastic LME baseline. The held-out scan's log-volume drops sharply
(log(V_MEN+1): 8.29 -> 8.86 -> 5.80); the homoscedastic LME *misses* the
held-out point (lower bound 5.94 > y = 5.80, IS@95 = 11.36) while BMA
*covers* it (lower bound 5.72, IS@95 = 5.92). The win is driven by the
LoRA-ensemble disagreement: at the held-out scan the 20 ensemble members
span tumour volumes from 748 to 91219 mm^3 (sigma^2_v = 9.39, the cohort
maximum), so BMA's between-member variance widens its interval just enough
to catch the drop the baseline cannot.

Layout
------
    [ row 0:  predictive-interval panel spanning the figure width        ]
    [ row 1:  T1ce slice @ t0 | @ t1 | @ t2, ensemble mask + variance     ]
    [ horizontal colorbar for the segmentation-variance overlay          ]

Row 0 is the per-patient predictive-interval visualisation of
``real_patient_homo_vs_bma.py`` (LME homoscedastic vs Ensemble BMA, 95%
parametric interval, ``last_from_rest`` protocol), with a miss arrow drawn
when a model's interval excludes the held-out observation.

Row 1 shows the T1ce slice (channel 1) at the ensemble-MEN-centroid axial
plane for each scan. Overlaid:

* the **voxel-wise segmentation variance** ``p(1 - p)`` of the per-voxel MEN
  indicator across the M=20 LoRA ensemble members (``p`` = fraction of
  members labelling the voxel as MEN; max 0.25 at maximal disagreement) --
  a perceptual heatmap of *where* the ensemble disagrees;
* the **ensemble consensus** MEN contour (majority-vote ``ensemble_mask``).

The MEN mask and per-member masks are taken from the LoRA ensemble
directory (``per_member_segmentations_deep_ensemble/per_scan``), the same
source as the ``logvol_mean`` trajectory used by the conformal experiment --
*not* the H5 ``segs`` dataset, which for this patient is a different,
inconsistent segmentation.

References
----------
Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and
Estimation*, JASA 102:359, 2007 -- the Interval Score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from scipy.stats import norm

from growth.stages.stage1_volumetric.trajectory_loader import (
    load_ensemble_trajectories_from_h5,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "figure.dpi": 120,
    }
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RESULTS_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction/conformal_calibration"
)
H5_PATH = Path("/media/mpascual/MeningD2/MENINGIOMAS/MENGROWTH/050526/h5_format/MenGrowth.h5")
PER_MEMBER_DIR = Path(
    "/media/mpascual/MeningD2/MENINGIOMAS/MENGROWTH/050526/"
    "per_member_segmentations_deep_ensemble/per_scan"
)
THESIS_FIG_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/"
    "bachelor_thesis/68596a200c0e0e3876880afa/figures/results"
)

PATIENT_ID = "MenGrowth-0048"
LAYER = "parametric"
SEED = 0  # parametric layer is seed-deterministic at tau=0.
ALPHA = 0.05
N_MEMBERS = 20

# (base_model, display label, colour, time-offset at t*)
MODELS: tuple[tuple[str, str, str, float], ...] = (
    ("lme_homo", "LME homoscedastic", "#4477AA", -0.11),
    ("ensemble_bma", "Ensemble BMA", "#CC6677", +0.11),
)

# Slice display
CROP_HALF = 72  # 144x144 window; the ensemble disagreement is diffuse for this patient
T1C_CHAN = 1  # channel index of "t1c" in MODALITY_KEYS = [t2f, t1c, t1n, t2w]
VAR_CMAP = "magma"
VAR_VMAX = 0.25  # theoretical max of the Bernoulli variance p(1-p)
VAR_FLOOR = 0.04  # below this the overlay is transparent (drops single-member stragglers)
ENSEMBLE_CONTOUR = "#39d0d8"

STAR_COLOR = "#b30000"
PAST_COLOR = "0.25"

COHORT_KWARGS = {
    "time_variable": "ordinal",
    "variance_key": "logvol_var",
    "mean_key": "logvol_mean",
    "scaling": "raw",
    "floor_variance": 1e-6,
    "exclude": ["MenGrowth-0028"],
    "min_timepoints": 2,
    "skip_all_zero_volume": True,
    "max_logvol_std": None,
}


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def _load_predictions() -> dict[str, pd.Series]:
    """Load the parametric-layer predictions for the patient, per model.

    Returns:
        Mapping ``base_model -> Series`` with the predictive mean, variance,
        interval bounds, IS@95 and held-out actual for ``PATIENT_ID``.
    """
    pp = pd.read_parquet(RESULTS_ROOT / "aggregated" / "per_patient_table.parquet")
    out: dict[str, pd.Series] = {}
    for base_model, *_ in MODELS:
        rows = pp[
            (pp["base_model"] == base_model)
            & (pp["seed"] == SEED)
            & (pp["layer"] == LAYER)
            & (pp["patient_id"] == PATIENT_ID)
        ]
        if rows.empty:
            raise RuntimeError(f"No prediction for {base_model}/{PATIENT_ID}.")
        out[base_model] = rows.iloc[0]
    return out


def _find_patient_scans(f: h5py.File, pid: str) -> tuple[list[int], np.ndarray, list[str]]:
    """Return time-ordered H5 row indices, timepoint indices and scan ids.

    Args:
        f: Open HDF5 handle.
        pid: Patient identifier.

    Returns:
        Tuple ``(row_indices, timepoint_idx, scan_ids)`` sorted by timepoint.
    """
    plist = f["longitudinal"]["patient_list"][:].astype(str)
    offs = f["longitudinal"]["patient_offsets"][:]
    pi = int(np.where(plist == pid)[0][0])
    i0, i1 = int(offs[pi]), int(offs[pi + 1])
    tpi = f["timepoint_idx"][i0:i1]
    sids = f["scan_ids"][i0:i1].astype(str)
    order = np.argsort(tpi)
    return (
        [int(i0 + k) for k in order],
        tpi[order].astype(float),
        [str(sids[k]) for k in order],
    )


def _crop_xy(arr2d: np.ndarray, cy: int, cx: int, half: int) -> np.ndarray:
    """Crop a 2-D array to a ``2*half`` window centred at ``(cy, cx)``, zero-padded.

    Args:
        arr2d: Source 2-D array.
        cy: Row centre.
        cx: Column centre.
        half: Half window size.

    Returns:
        ``(2*half, 2*half)`` crop, zero-padded where the window hits a border.
    """
    h, w = arr2d.shape
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    crop = arr2d[y0:y1, x0:x1]
    out = np.zeros((2 * half, 2 * half), dtype=crop.dtype)
    out[: crop.shape[0], : crop.shape[1]] = crop
    return out


def _ensemble_mask(scan_id: str) -> np.ndarray:
    """Load the majority-vote MEN ensemble mask for a scan.

    Args:
        scan_id: Scan identifier (e.g. ``MenGrowth-0048-002``).

    Returns:
        Boolean ``(192, 192, 192)`` MEN mask.
    """
    path = PER_MEMBER_DIR / scan_id / "ensemble_mask.nii.gz"
    return np.asarray(nib.load(str(path)).dataobj) > 0


def _variance_map(scan_id: str) -> np.ndarray:
    """Voxel-wise Bernoulli variance ``p(1-p)`` of the MEN indicator across members.

    Args:
        scan_id: Scan identifier.

    Returns:
        Float32 ``(192, 192, 192)`` variance map; ``p`` is the fraction of the
        M ensemble members labelling each voxel as MEN.
    """
    members = []
    for k in range(N_MEMBERS):
        path = PER_MEMBER_DIR / scan_id / f"member_{k}_mask.nii.gz"
        members.append(np.asarray(nib.load(str(path)).dataobj) > 0)
    p = np.mean(np.stack(members).astype(np.float32), axis=0)
    return (p * (1.0 - p)).astype(np.float32)


def _load_slices() -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Load T1ce slices, ensemble MEN masks and variance maps for every scan.

    Returns:
        Tuple ``(t1c_slices, men_masks, var_maps, timepoint_idx)``; all arrays
        are cropped to a common window around the cross-scan mean ensemble
        centroid (clamped so the window stays in-volume), at the per-scan z
        that maximises the in-slice ensemble MEN area.
    """
    with h5py.File(H5_PATH, "r") as f:
        idxs, ts, scan_ids = _find_patient_scans(f, PATIENT_ID)
        images = [f["images"][gi, T1C_CHAN] for gi in idxs]

    masks = [_ensemble_mask(sid) for sid in scan_ids]
    var_vols = [_variance_map(sid) for sid in scan_ids]
    if all(m.sum() == 0 for m in masks):
        raise RuntimeError("All ensemble MEN masks are empty.")

    # Common crop centre = mean of the per-scan ensemble centroids, clamped so
    # the CROP_HALF window stays inside the volume (the disagreement is diffuse,
    # so a fixed generous window beats per-scan re-centring).
    cents = np.stack([np.argwhere(m).mean(axis=0) for m in masks if m.sum() > 0])
    cen = cents.mean(axis=0)
    dim = masks[0].shape[0]
    cy0 = int(np.clip(round(cen[0]), CROP_HALF, dim - CROP_HALF))
    cx0 = int(np.clip(round(cen[1]), CROP_HALF, dim - CROP_HALF))

    t1c_slices: list[np.ndarray] = []
    men_masks: list[np.ndarray] = []
    var_maps: list[np.ndarray] = []
    for img, men, var in zip(images, masks, var_vols):
        # Per-scan z that maximises in-slice ensemble MEN area keeps the
        # tumour visible as it grows; the xy crop stays common across scans.
        cz = int(np.argmax(men.sum(axis=(0, 1)))) if men.sum() > 0 else int(round(cen[2]))
        t1c_slices.append(_crop_xy(img[:, :, cz], cy0, cx0, CROP_HALF))
        men_masks.append(_crop_xy(men[:, :, cz].astype(np.uint8), cy0, cx0, CROP_HALF))
        var_maps.append(_crop_xy(var[:, :, cz], cy0, cx0, CROP_HALF))
    return t1c_slices, men_masks, var_maps, ts


# --------------------------------------------------------------------------- #
# Prediction panel
# --------------------------------------------------------------------------- #
def _draw_prediction_panel(
    ax: plt.Axes,
    past_t: np.ndarray,
    past_y: np.ndarray,
    t_star: float,
    y_true: float,
    preds: dict[str, pd.Series],
) -> None:
    """Draw the row-0 predictive-interval panel with both models overlaid.

    Args:
        ax: Target axes.
        past_t: Conditioning time-points.
        past_y: Conditioning mean log-volumes.
        t_star: Held-out follow-up index.
        y_true: Held-out observed log-volume.
        preds: Mapping ``base_model -> prediction Series``.
    """
    half = 0.34  # sideways-density half-width in time units

    ax.plot(
        past_t,
        past_y,
        "o-",
        color=PAST_COLOR,
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=0.4,
        linewidth=1.2,
        zorder=3,
    )
    ax.axvline(t_star, color="0.6", linestyle=":", linewidth=0.9, zorder=0)

    is_lines: list[str] = []
    for base_model, label, color, dx in MODELS:
        row = preds[base_model]
        mu = float(row["pred_mean"])
        sigma = float(np.sqrt(max(row["pred_var"], 1e-12)))
        lo, hi = float(row["lower"]), float(row["upper"])
        is_val = float(row["interval_score"])
        x0 = t_star + dx

        ax.plot(
            [past_t[-1], x0],
            [past_y[-1], mu],
            "--",
            color=color,
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )

        # Sideways Gaussian predictive density.
        y_grid = np.linspace(min(lo, y_true) - 0.7, max(hi, y_true) + 0.7, 400)
        pdf = norm.pdf(y_grid, loc=mu, scale=max(sigma, 1e-6))
        pdf_w = pdf / pdf.max() * half
        in_ci = (y_grid >= lo) & (y_grid <= hi)
        ax.fill_betweenx(y_grid, x0 - pdf_w, x0 + pdf_w, color=color, alpha=0.10, zorder=1)
        ax.fill_betweenx(
            y_grid[in_ci],
            (x0 - pdf_w)[in_ci],
            (x0 + pdf_w)[in_ci],
            color=color,
            alpha=0.32,
            zorder=2,
        )
        ax.plot(x0 + pdf_w, y_grid, color=color, linewidth=1.1, alpha=0.9, zorder=3)
        ax.plot(x0 - pdf_w, y_grid, color=color, linewidth=1.1, alpha=0.9, zorder=3)

        # Interval bounds as a capped vertical bar at the offset position.
        ax.plot([x0, x0], [lo, hi], color=color, linewidth=1.3, alpha=0.9, zorder=4)
        for bound in (lo, hi):
            ax.plot([x0 - 0.05, x0 + 0.05], [bound, bound], color=color, linewidth=1.3, zorder=4)
        ax.plot(
            x0,
            mu,
            "o",
            color=color,
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.4,
            zorder=5,
        )

        # Miss arrow when the held-out point falls outside the interval.
        covered = bool(row["covered"])
        if not covered:
            y_a, y_b = (y_true, lo) if y_true < lo else (hi, y_true)
            ax.annotate(
                "",
                xy=(x0, y_a),
                xytext=(x0, y_b),
                arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.5},
                zorder=6,
            )
        is_lines.append(rf"{label}: IS@95 $= {is_val:.2f}$ ({'covered' if covered else 'miss'})")

    ax.plot(
        t_star,
        y_true,
        marker="*",
        color=STAR_COLOR,
        markersize=16,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=7,
    )

    ax.text(
        0.015,
        0.97,
        "\n".join(is_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    all_t = np.concatenate([past_t, [t_star]])
    ax.set_xlim(all_t.min() - 0.45, t_star + 0.85)
    ax.set_xticks(np.unique(all_t))
    ax.set_xlabel(r"follow-up index $t$")
    ax.set_ylabel(r"$\log(V_{\mathrm{MEN}}+1)$")
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_title(
        rf"{PATIENT_ID}: predict $t^\ast={int(t_star)}$ from $t=0,\dots,{int(t_star) - 1}$ "
        r"(LME homoscedastic vs. Ensemble BMA, 95% parametric interval)",
        fontsize=10,
    )


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def main(out_pdf: Path, out_png: Path) -> None:
    """Build and write the MenGrowth-0048 BMA-win figure with variance overlay.

    Args:
        out_pdf: Destination PDF path.
        out_png: Destination PNG path.
    """
    preds = _load_predictions()
    trajs = {
        t.patient_id: t
        for t in load_ensemble_trajectories_from_h5(h5_path=str(H5_PATH), **COHORT_KWARGS)
    }
    traj = trajs[PATIENT_ID]
    past_t = traj.times[:-1].astype(float)
    past_y = np.asarray(traj.observations[:-1]).reshape(-1).astype(float)
    t_star = float(traj.times[-1])
    y_true = float(np.asarray(traj.observations[-1]).reshape(-1)[0])
    full_y = np.asarray(traj.observations).reshape(-1).astype(float)
    sigma_v_sq = np.asarray(traj.observation_variance).reshape(-1).astype(float)

    t1c_slices, men_masks, var_maps, ts = _load_slices()
    n = len(t1c_slices)
    logger.info(
        "Patient %s: %d scans, log-volumes %s, sigma^2_v %s",
        PATIENT_ID,
        n,
        np.round(full_y, 3).tolist(),
        np.round(sigma_v_sq, 3).tolist(),
    )

    fig = plt.figure(figsize=(11.0, 7.0))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=n,
        height_ratios=[1.45, 1.0],
        hspace=0.34,
        wspace=0.10,
        left=0.075,
        right=0.90,
        top=0.93,
        bottom=0.12,
        figure=fig,
    )

    # Row 0: prediction panel.
    ax_pred = fig.add_subplot(gs[0, :])
    _draw_prediction_panel(ax_pred, past_t, past_y, t_star, y_true, preds)

    # Row 1: slice panels with the variance overlay, shared display ranges.
    all_intens = np.concatenate([s.ravel() for s in t1c_slices])
    vmin, vmax = np.percentile(all_intens[all_intens > 0], (1.0, 99.0))
    men_cmap = ListedColormap([ENSEMBLE_CONTOUR])
    var_im = None

    for k in range(n):
        ax = fig.add_subplot(gs[1, k])
        img = np.rot90(t1c_slices[k], k=1)
        mask = np.rot90(men_masks[k], k=1)
        var = np.rot90(var_maps[k], k=1)

        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
        var_disp = np.ma.masked_where(var < VAR_FLOOR, var)
        var_im = ax.imshow(
            var_disp, cmap=VAR_CMAP, vmin=0.0, vmax=VAR_VMAX, alpha=0.70, interpolation="bilinear"
        )
        ax.contour(mask, levels=[0.5], colors=ENSEMBLE_CONTOUR, linewidths=1.3, alpha=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        marker = r"  $\leftarrow t^\ast$" if int(ts[k]) == int(t_star) else ""
        ax.set_title(
            rf"$t = {int(ts[k])}$,  $\log(V+1) = {full_y[k]:.2f}$,  "
            rf"$\sigma^2_v = {sigma_v_sq[k]:.2f}${marker}",
            fontsize=8.5,
        )

    # Vertical colorbar for the variance overlay, to the right of the slice row.
    row1_box = gs[1, n - 1].get_position(fig)
    cax = fig.add_axes((0.915, row1_box.y0, 0.015, row1_box.height))
    cb = fig.colorbar(var_im, cax=cax, orientation="vertical")
    cb.set_label(
        r"segmentation variance $p(1-p)$" + "\n" + r"across $M=20$ ensemble members",
        fontsize=8.5,
    )
    cb.ax.tick_params(labelsize=8)

    # Shared legend.
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="-",
            marker="o",
            markersize=6,
            color=PAST_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="conditioning observations",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            markersize=16,
            color=STAR_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=r"held-out observation $y$ at $t^\ast$",
        ),
        plt.Line2D([], [], linestyle="-", linewidth=1.3, color=MODELS[0][2], label=MODELS[0][1]),
        plt.Line2D([], [], linestyle="-", linewidth=1.3, color=MODELS[1][2], label=MODELS[1][1]),
        plt.Line2D(
            [],
            [],
            linestyle="-",
            linewidth=1.4,
            color=ENSEMBLE_CONTOUR,
            label="ensemble consensus MEN contour",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", out_pdf)
    logger.info("Wrote %s", out_png)
    plt.close(fig)


if __name__ == "__main__":
    main(
        out_pdf=THESIS_FIG_DIR / "real_patient_homo_vs_bma_0048_uncertainty.pdf",
        out_png=RESULTS_ROOT / "figures" / "real_patient_homo_vs_bma_0048_uncertainty.png",
    )
