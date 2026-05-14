"""Real-patient predictive-interval figure: LME Homo vs Ensemble BMA.

Patient: MenGrowth-0055 -- a clean monotonic grower (4 timepoints,
log(V_MEN+1) from 7.38 to 9.43) with a confident LoRA-ensemble
segmentation. The same patient is used by
``main_experiment/analyses/plot_real_patient_log_linearisation.py``.

Layout
------
    [ row 0:  predictive-interval panel spanning the figure width        ]
    [ row 1:  T1ce slice @ t0 | @ t1 | @ t2 | @ t3, MEN segmentation     ]

Row 0 reproduces the per-patient predictive-interval visualisation of
``per_patient_predictions.py`` under the ``last_from_rest`` protocol, but
overlays *both* primary-showcase models at the held-out time-point t*: the
homoscedastic LME baseline and the ensemble Bayesian model averaging (BMA)
model. Each is drawn as a sideways Gaussian predictive density with its 95%
parametric interval; the two are offset slightly in time so the overlap
stays readable. The per-model IS@95 (Gneiting & Raftery, 2007) is annotated.

Row 1 shows the T1ce slice (channel 1) at the MEN-centroid axial plane for
each scan, cropped to a fixed window around the first-scan tumour centroid,
with the MEN mask (labels 1|3) overlaid -- the segmentation evidence behind
the growth trajectory plotted above.

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
THESIS_FIG_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/"
    "bachelor_thesis/68596a200c0e0e3876880afa/figures/results"
)

PATIENT_ID = "MenGrowth-0055"
LAYER = "parametric"
SEED = 0  # parametric layer is seed-deterministic at tau=0.
ALPHA = 0.05

# (base_model, display label, colour, time-offset at t*)
MODELS: tuple[tuple[str, str, str, float], ...] = (
    ("lme_homo", "LME homoscedastic", "#4477AA", -0.11),
    ("ensemble_bma", "Ensemble BMA", "#CC6677", +0.11),
)

# Slice display
CROP_HALF = 48  # 96x96 window around the first-scan tumour centroid
T1C_CHAN = 1  # channel index of "t1c" in MODALITY_KEYS = [t2f, t1c, t1n, t2w]
MEN_LABELS = (1, 3)  # MEN meningioma = BraTS-TC (labels 1|3)
MEN_COLOR = "#1f77ff"

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


def _find_patient_indices(f: h5py.File, pid: str) -> tuple[list[int], np.ndarray]:
    """Return the time-ordered H5 row indices and timepoint indices for a patient.

    Args:
        f: Open HDF5 handle.
        pid: Patient identifier.

    Returns:
        Tuple ``(row_indices, timepoint_idx)`` sorted by timepoint.
    """
    plist = f["longitudinal"]["patient_list"][:].astype(str)
    offs = f["longitudinal"]["patient_offsets"][:]
    pi = int(np.where(plist == pid)[0][0])
    i0, i1 = int(offs[pi]), int(offs[pi + 1])
    tpi = f["timepoint_idx"][i0:i1]
    order = np.argsort(tpi)
    return [int(i0 + k) for k in order], tpi[order].astype(float)


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


def _load_slices() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Load T1ce slices and MEN masks for every scan of the patient.

    Returns:
        Tuple ``(t1c_slices, men_masks, timepoint_idx)``; slices and masks are
        cropped to a fixed window around the first-scan tumour centroid.
    """
    with h5py.File(H5_PATH, "r") as f:
        idxs, ts = _find_patient_indices(f, PATIENT_ID)

        seg0 = f["segs"][idxs[0], 0]
        men0 = np.isin(seg0, MEN_LABELS)
        if men0.sum() == 0:
            raise RuntimeError("Empty MEN mask at the first scan.")
        cent0 = np.argwhere(men0).mean(axis=0)
        cy0, cx0 = int(round(cent0[0])), int(round(cent0[1]))

        t1c_slices: list[np.ndarray] = []
        men_masks: list[np.ndarray] = []
        for gi in idxs:
            seg = f["segs"][gi, 0]
            men = np.isin(seg, MEN_LABELS)
            # Per-scan z that maximises in-slice MEN area keeps the tumour
            # visible as it grows; the xy crop stays at the scan-0 centroid.
            cz = int(np.argmax(men.sum(axis=(0, 1)))) if men.sum() > 0 else int(round(cent0[2]))
            t1c_slices.append(_crop_xy(f["images"][gi, T1C_CHAN][:, :, cz], cy0, cx0, CROP_HALF))
            men_masks.append(_crop_xy(men[:, :, cz].astype(np.uint8), cy0, cx0, CROP_HALF))
    return t1c_slices, men_masks, ts


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

    # Conditioning trajectory.
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

        # Dashed link from the last conditioning point to the predictive mean.
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
        # Predictive mean.
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

        covered = bool(row["covered"])
        is_lines.append(rf"{label}: IS@95 $= {is_val:.2f}$ ({'covered' if covered else 'miss'})")

    # Held-out observation.
    ax.plot(
        t_star,
        y_true,
        marker="*",
        color=STAR_COLOR,
        markersize=16,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=6,
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
    """Build and write the MenGrowth-0055 Homo-vs-BMA figure.

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

    t1c_slices, men_masks, ts = _load_slices()
    n = len(t1c_slices)
    logger.info("Patient %s: %d scans, log-volumes %s", PATIENT_ID, n, np.round(full_y, 3).tolist())

    fig = plt.figure(figsize=(11.0, 6.4))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=n,
        height_ratios=[1.45, 1.0],
        hspace=0.34,
        wspace=0.10,
        left=0.075,
        right=0.975,
        top=0.92,
        bottom=0.07,
        figure=fig,
    )

    # Row 0: prediction panel.
    ax_pred = fig.add_subplot(gs[0, :])
    _draw_prediction_panel(ax_pred, past_t, past_y, t_star, y_true, preds)

    # Row 1: slice panels with a shared display intensity range.
    all_intens = np.concatenate([s.ravel() for s in t1c_slices])
    vmin, vmax = np.percentile(all_intens[all_intens > 0], (1.0, 99.0))
    men_cmap = ListedColormap([MEN_COLOR])

    for k in range(n):
        ax = fig.add_subplot(gs[1, k])
        img = np.rot90(t1c_slices[k], k=1)
        mask = np.rot90(men_masks[k], k=1)
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=men_cmap, alpha=0.30, vmin=0, vmax=1)
        ax.contour(mask, levels=[0.5], colors=MEN_COLOR, linewidths=1.2, alpha=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        marker = r"  $\leftarrow t^\ast$" if int(ts[k]) == int(t_star) else ""
        ax.set_title(rf"$t = {int(ts[k])}$,  $\log(V+1) = {full_y[k]:.2f}${marker}", fontsize=9)

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
            color=MEN_COLOR,
            label=r"MEN segmentation (labels 1$|$3)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
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
        out_pdf=THESIS_FIG_DIR / "real_patient_homo_vs_bma_0055.pdf",
        out_png=RESULTS_ROOT / "figures" / "real_patient_homo_vs_bma_0055.png",
    )
