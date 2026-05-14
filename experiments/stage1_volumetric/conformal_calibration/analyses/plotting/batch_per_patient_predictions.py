"""Batch per-patient predictive-interval figures for the BSc thesis.

Produces one figure per MenGrowth patient in the conformal_calibration cohort:

    [ row 0:  predictive-interval panel — LME homoscedastic vs Ensemble BMA  ]
    [ row 1:  whole-slice T1ce @ each timepoint, MEN segmentation overlaid   ]

Same style as ``real_patient_homo_vs_bma.py`` but (a) batched over every
patient and (b) the slices are shown **whole** (full 192³ axial plane, no
crop) at the per-scan z that maximises the MEN-mask area.

Segmentation-variance overlay (codepath, off by default)
--------------------------------------------------------
The slice row can additionally show the voxel-wise segmentation variance
``p(1-p)`` of the M=20 LoRA ensemble. That requires per-member voxel masks
*consistent with the conformal experiment's trajectory* — i.e. the World-A
re-inference produced by
``experiments/stage1_volumetric/test_candidate_uncertainty_signals/`` (see
``REINFER_WORLDA.md``). The masks on disk before that job runs
(``per_member_segmentations_deep_ensemble/per_scan/``) are the broken World-B
inference and must NOT be used — see
``memory/project_h5_uncertainty_two_inferences.md``.

To enable the overlay once the World-A masks exist, set ``PER_MEMBER_DIR`` to
the re-inference ``per_scan/`` directory. While it is ``None`` the figures
show the consensus MEN contour only; everything downstream is unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
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
OUT_DIR = RESULTS_ROOT / "figures" / "per_patient_batch"

LAYER = "parametric"
SEED = 0  # parametric layer is seed-deterministic at tau=0.
ALPHA = 0.05

# (base_model, display label, colour, time-offset at t*)
MODELS: tuple[tuple[str, str, str, float], ...] = (
    ("lme_homo", "LME homoscedastic", "#4477AA", -0.11),
    ("ensemble_bma", "Ensemble BMA", "#CC6677", +0.11),
)

# --- Segmentation-variance overlay codepath -------------------------------- #
# Set to the World-A re-inference ``per_scan/`` directory once it exists
# (see test_candidate_uncertainty_signals/REINFER_WORLDA.md). While None, the
# slices show only the consensus MEN contour.
PER_MEMBER_DIR: Path | None = None
N_MEMBERS = 20
VAR_CMAP = "magma"
VAR_VMAX = 0.25  # theoretical max of the Bernoulli variance p(1-p)
VAR_FLOOR = 0.04  # below this the overlay is transparent

T1C_CHAN = 1  # channel index of "t1c" in MODALITY_KEYS = [t2f, t1c, t1n, t2w]
MEN_LABELS = (1, 3)  # MEN meningioma = BraTS-TC (labels 1|3)
MEN_COLOR = "#39d0d8"
STAR_COLOR = "#b30000"
PAST_COLOR = "0.25"

# QC filter matching the executed conformal variance-ablation run.
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
def _load_predictions() -> pd.DataFrame:
    """Load per-patient parametric-layer predictions, indexed by (model, patient).

    Returns:
        Frame indexed by ``(base_model, patient_id)`` for ``SEED`` / ``LAYER``.
    """
    pp = pd.read_parquet(RESULTS_ROOT / "aggregated" / "per_patient_table.parquet")
    keep = pp[(pp["seed"] == SEED) & (pp["layer"] == LAYER)].copy()
    if keep.empty:
        raise RuntimeError(f"No rows for seed={SEED}, layer={LAYER} in per_patient_table.")
    return keep.set_index(["base_model", "patient_id"]).sort_index()


def _patient_scan_rows(f: h5py.File) -> dict[str, list[tuple[int, int]]]:
    """Map each patient to its time-ordered ``(h5_row, timepoint_idx)`` list.

    Args:
        f: Open HDF5 handle.

    Returns:
        Mapping ``patient_id -> [(h5_row, timepoint_idx), ...]`` sorted by time.
    """
    plist = f["longitudinal"]["patient_list"][:].astype(str)
    offs = f["longitudinal"]["patient_offsets"][:]
    tpi = f["timepoint_idx"][:]
    out: dict[str, list[tuple[int, int]]] = {}
    for pi, pid in enumerate(plist):
        i0, i1 = int(offs[pi]), int(offs[pi + 1])
        rows = sorted(((int(j), int(tpi[j])) for j in range(i0, i1)), key=lambda t: t[1])
        out[str(pid)] = rows
    return out


def _variance_map(scan_id: str) -> np.ndarray | None:
    """Voxel-wise Bernoulli variance ``p(1-p)`` of the MEN indicator over members.

    Args:
        scan_id: Scan identifier (e.g. ``MenGrowth-0048-002``).

    Returns:
        Float32 ``[D, H, W]`` variance map, or ``None`` if ``PER_MEMBER_DIR`` is
        unset or the per-member masks for this scan are missing.
    """
    if PER_MEMBER_DIR is None:
        return None
    import nibabel as nib  # local import: only needed when the overlay is on

    scan_dir = PER_MEMBER_DIR / scan_id
    members = []
    for k in range(N_MEMBERS):
        path = scan_dir / f"member_{k}_mask.nii.gz"
        if not path.exists():
            return None
        members.append(np.asarray(nib.load(str(path)).dataobj) > 0)
    p = np.mean(np.stack(members).astype(np.float32), axis=0)
    return (p * (1.0 - p)).astype(np.float32)


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
    patient_id: str,
) -> None:
    """Draw the row-0 predictive-interval panel with both models overlaid.

    Args:
        ax: Target axes.
        past_t: Conditioning time-points.
        past_y: Conditioning mean log-volumes.
        t_star: Held-out follow-up index.
        y_true: Held-out observed log-volume.
        preds: Mapping ``base_model -> prediction Series``.
        patient_id: Patient identifier (for the title).
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
        rf"{patient_id}: predict $t^\ast={int(t_star)}$ from $t=0,\dots,{int(t_star) - 1}$ "
        r"(LME homoscedastic vs. Ensemble BMA, 95% parametric interval)",
        fontsize=10,
    )


# --------------------------------------------------------------------------- #
# Per-patient figure
# --------------------------------------------------------------------------- #
def _make_patient_figure(
    patient_id: str,
    traj,
    preds: pd.DataFrame,
    scan_rows: list[tuple[int, int]],
    out_path: Path,
) -> bool:
    """Build and write one patient's figure.

    Args:
        patient_id: Patient identifier.
        traj: ``PatientTrajectory`` for this patient (QC-filtered).
        preds: Full per-patient prediction frame indexed by (model, patient).
        scan_rows: ``[(h5_row, timepoint_idx), ...]`` for this patient.
        out_path: Destination PNG path.

    Returns:
        ``True`` if the figure was written, ``False`` if the patient was
        skipped (missing predictions for one of the two models).
    """
    try:
        pred_by_model = {bm: preds.loc[(bm, patient_id)] for bm, *_ in MODELS}
    except KeyError:
        logger.warning("  %s: missing predictions for one model — skipped", patient_id)
        return False

    past_t = traj.times[:-1].astype(float)
    past_y = np.asarray(traj.observations[:-1]).reshape(-1).astype(float)
    t_star = float(traj.times[-1])
    y_true = float(np.asarray(traj.observations[-1]).reshape(-1)[0])
    full_y = np.asarray(traj.observations).reshape(-1).astype(float)

    # Slices: only the timepoints that survived QC (present in traj.times).
    traj_tps = {int(t) for t in traj.times}
    shown = [(row, tp) for row, tp in scan_rows if tp in traj_tps]
    n = len(shown)
    if n == 0:
        logger.warning("  %s: no scans match the trajectory timepoints — skipped", patient_id)
        return False

    with h5py.File(H5_PATH, "r") as f:
        scan_ids = f["scan_ids"][:].astype(str)
        t1c_slices: list[np.ndarray] = []
        men_masks: list[np.ndarray] = []
        var_maps: list[np.ndarray | None] = []
        for row, _tp in shown:
            seg = np.isin(f["segs"][row, 0], MEN_LABELS)
            cz = int(np.argmax(seg.sum(axis=(0, 1)))) if seg.sum() > 0 else seg.shape[2] // 2
            t1c_slices.append(np.asarray(f["images"][row, T1C_CHAN, :, :, cz], dtype=np.float32))
            men_masks.append(seg[:, :, cz].astype(np.uint8))
            var_vol = _variance_map(str(scan_ids[row]))
            var_maps.append(var_vol[:, :, cz] if var_vol is not None else None)

    fig = plt.figure(figsize=(max(9.0, 2.7 * n), 6.6))
    has_var = any(v is not None for v in var_maps)
    right = 0.90 if has_var else 0.975
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=n,
        height_ratios=[1.45, 1.0],
        hspace=0.34,
        wspace=0.08,
        left=0.075,
        right=right,
        top=0.92,
        bottom=0.11,
        figure=fig,
    )

    ax_pred = fig.add_subplot(gs[0, :])
    _draw_prediction_panel(ax_pred, past_t, past_y, t_star, y_true, pred_by_model, patient_id)

    all_intens = np.concatenate([s.ravel() for s in t1c_slices])
    pos = all_intens[all_intens > 0]
    vmin, vmax = np.percentile(pos, (1.0, 99.0)) if pos.size else (0.0, 1.0)
    var_im = None

    for k, (_row, tp) in enumerate(shown):
        ax = fig.add_subplot(gs[1, k])
        img = np.rot90(t1c_slices[k], k=1)
        mask = np.rot90(men_masks[k], k=1)
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
        if var_maps[k] is not None:
            var = np.rot90(var_maps[k], k=1)
            var_im = ax.imshow(
                np.ma.masked_where(var < VAR_FLOOR, var),
                cmap=VAR_CMAP,
                vmin=0.0,
                vmax=VAR_VMAX,
                alpha=0.70,
                interpolation="bilinear",
            )
        if mask.sum() > 0:
            ax.contour(mask, levels=[0.5], colors=MEN_COLOR, linewidths=1.2, alpha=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        marker = r"  $\leftarrow t^\ast$" if int(tp) == int(t_star) else ""
        # full_y is indexed by trajectory order, which matches `shown` order.
        ax.set_title(rf"$t = {int(tp)}$,  $\log(V+1) = {full_y[k]:.2f}${marker}", fontsize=8.5)

    if has_var and var_im is not None:
        row1_box = gs[1, n - 1].get_position(fig)
        cax = fig.add_axes((0.915, row1_box.y0, 0.015, row1_box.height))
        cb = fig.colorbar(var_im, cax=cax, orientation="vertical")
        cb.set_label(
            r"segmentation variance $p(1-p)$" + "\n" + r"across $M=20$ ensemble members",
            fontsize=8.5,
        )
        cb.ax.tick_params(labelsize=8)

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
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(out_dir: Path) -> None:
    """Generate one prediction figure per patient in the conformal cohort.

    Args:
        out_dir: Directory for the per-patient PNGs.
    """
    preds = _load_predictions()
    trajs = {
        t.patient_id: t
        for t in load_ensemble_trajectories_from_h5(h5_path=str(H5_PATH), **COHORT_KWARGS)
    }
    with h5py.File(H5_PATH, "r") as f:
        scan_rows = _patient_scan_rows(f)

    overlay = "ON" if PER_MEMBER_DIR is not None else "OFF (consensus contour only)"
    logger.info("Variance-map overlay: %s", overlay)
    logger.info("Rendering %d patients -> %s", len(trajs), out_dir)

    n_done = 0
    for patient_id in sorted(trajs):
        traj = trajs[patient_id]
        rows = scan_rows.get(patient_id, [])
        out_path = out_dir / f"{patient_id}.png"
        if _make_patient_figure(patient_id, traj, preds, rows, out_path):
            n_done += 1
            logger.info("  [%d] wrote %s", n_done, out_path.name)

    logger.info("Done: %d/%d patient figures in %s", n_done, len(trajs), out_dir)


if __name__ == "__main__":
    main(out_dir=OUT_DIR)
