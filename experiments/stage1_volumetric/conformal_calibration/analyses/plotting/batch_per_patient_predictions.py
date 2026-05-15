"""Batch per-patient predictive-interval figures for the BSc thesis.

Produces one figure per MenGrowth patient in the conformal_calibration cohort:

    [ row 0:  predictive-interval panel — LME homoscedastic vs Ensemble BMA  ]
    [ row 1:  whole-slice T1ce @ each timepoint, MEN segmentation overlaid   ]

Same style as ``real_patient_homo_vs_bma.py`` but (a) batched over every
patient and (b) the slices are shown **whole** (full 192³ axial plane, no
crop) at a *single* axis-2 index anchored to the most clearly-segmented
timepoint — so every panel shows the same anatomical plane. A timepoint
whose MEN mask falls outside that plane (a separate lesion, or an unreliable
segmentation) is annotated rather than shown contour-free.

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
# Both predictions are rendered at the held-out x = t^ast (no offset) so the
# point estimates land exactly on the GT vertical; the two violin densities
# overlap with their own alphas, which is acceptable since the table reports
# the numeric separation.
MODELS: tuple[tuple[str, str, str, float], ...] = (
    ("lme_homo", "LME homoscedastic", "#4477AA", 0.0),
    ("ensemble_bma", "Ensemble BMA", "#CC6677", 0.0),
)

# --- Segmentation-variance overlay codepath -------------------------------- #
# World-A re-inference ``per_scan/`` directory (per-member masks consistent
# with the logvol_mean trajectory used by the conformal experiment). See
# test_candidate_uncertainty_signals/REINFER_WORLDA.md for context.
PER_MEMBER_DIR: Path | None = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction/per_member_segmentations_r32_worldA/per_scan"
)
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


# --------------------------------------------------------------------------- #
# Physical-units helpers
# --------------------------------------------------------------------------- #
# H5 spacing is 1 mm isotropic (see ``attrs['spacing']``), so 1 voxel = 1 mm^3
# and the H5 ``logvol_mean`` field is :math:`\\log(V_{\\mathrm{mm}^3} + 1)`.
VOXEL_SPACING_MM = 1.0
SCALE_BAR_LENGTH_MM = 30.0  # ruler length drawn on each slice


def _log_to_diam_mm(y: np.ndarray) -> np.ndarray:
    """Convert ``log(V+1)`` (V in mm^3) to equivalent spherical diameter (mm).

    Args:
        y: Log-volume values.

    Returns:
        Equivalent spherical diameter, :math:`d = 2 (3V / 4\\pi)^{1/3}`, mm.
    """
    v = np.maximum(np.exp(y) - 1.0, 0.0)
    return 2.0 * (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0)


def _diam_mm_to_log(d: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_log_to_diam_mm`: equivalent diameter (mm) → log(V+1).

    Args:
        d: Equivalent spherical diameter in mm.

    Returns:
        :math:`\\log(\\pi d^3 / 6 + 1)`.
    """
    d = np.maximum(d, 0.0)
    return np.log1p((np.pi / 6.0) * d ** 3)


def _add_scale_bar(
    ax: plt.Axes,
    length_mm: float = SCALE_BAR_LENGTH_MM,
    voxel_size_mm: float = VOXEL_SPACING_MM,
) -> None:
    """Draw a horizontal ruler on the slice (white, with endpoint tick marks).

    Args:
        ax: Image axes (imshow already drawn; y-axis inverted).
        length_mm: Ruler length in millimetres.
        voxel_size_mm: Image voxel pitch in millimetres (in-plane).
    """
    if voxel_size_mm < 1e-6:
        return
    bar_voxels = length_mm / voxel_size_mm
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_start = xlim[0] + (xlim[1] - xlim[0]) * 0.04
    x_end = x_start + bar_voxels
    y_pos = ylim[0] - 0.025 * abs(ylim[1] - ylim[0])  # just above bottom edge of image
    tick_h = 0.018 * abs(ylim[1] - ylim[0])
    ax.plot(
        [x_start, x_end],
        [y_pos, y_pos],
        color="white",
        linewidth=1.6,
        solid_capstyle="butt",
        clip_on=False,
        zorder=11,
    )
    for x_t in (x_start, x_end):
        ax.plot(
            [x_t, x_t],
            [y_pos - tick_h, y_pos + tick_h],
            color="white",
            linewidth=1.6,
            solid_capstyle="butt",
            clip_on=False,
            zorder=11,
        )
    # Label sits clearly above the bar (image y-axis is inverted, so subtracting
    # moves visually upward). va="bottom" anchors the text's lower edge to y,
    # leaving a clean gap between the label and the ruler ticks.
    ax.text(
        (x_start + x_end) / 2,
        y_pos - 1.6 * tick_h,
        f"{length_mm:.0f} mm",
        color="white",
        fontsize=7,
        ha="center",
        va="bottom",
        zorder=11,
    )


def _worldA_dir(patient_id: str, traj_index: int) -> Path | None:
    """Resolve the World-A per-scan directory for a given trajectory position.

    The World-A re-inference enumerates scans **per patient, 0-based,
    contiguous, in trajectory (timepoint) order** — independently of the
    H5 ``scan_ids`` strings, which retain the original clinical numbering
    and can skip values (e.g. ``MenGrowth-0025`` has scan ids ``-000,
    -002, -003, -004, -005, -006`` for trajectory positions 0..5; disk has
    ``-000..-005``). Indexing by scan_id therefore mis-reads or misses
    variance maps; the correct key is ``f"{patient_id}-{traj_index:03d}"``.

    Args:
        patient_id: Patient identifier (e.g. ``MenGrowth-0048``).
        traj_index: 0-based position within the patient's trajectory.

    Returns:
        Path to the per-scan directory, or ``None`` if ``PER_MEMBER_DIR``
        is unset or the directory is missing on disk.
    """
    if PER_MEMBER_DIR is None:
        return None
    d = PER_MEMBER_DIR / f"{patient_id}-{traj_index:03d}"
    return d if d.is_dir() else None


def _variance_map(patient_id: str, traj_index: int) -> np.ndarray | None:
    """Voxel-wise Bernoulli variance ``p(1-p)`` of the MEN indicator over members.

    Args:
        patient_id: Patient identifier.
        traj_index: 0-based trajectory position (see ``_worldA_dir``).

    Returns:
        Float32 ``[D, H, W]`` variance map, or ``None`` if ``PER_MEMBER_DIR`` is
        unset or the per-member masks for this scan are missing.
    """
    scan_dir = _worldA_dir(patient_id, traj_index)
    if scan_dir is None:
        return None
    import nibabel as nib  # local import: only needed when the overlay is on

    members = []
    for k in range(N_MEMBERS):
        path = scan_dir / f"member_{k}_mask.nii.gz"
        if not path.exists():
            return None
        members.append(np.asarray(nib.load(str(path)).dataobj) > 0)
    p = np.mean(np.stack(members).astype(np.float32), axis=0)
    return (p * (1.0 - p)).astype(np.float32)


def _ensemble_mask(patient_id: str, traj_index: int) -> np.ndarray | None:
    """Load the World-A LoRA-ensemble consensus MEN mask.

    Args:
        patient_id: Patient identifier.
        traj_index: 0-based trajectory position (see ``_worldA_dir``).

    Returns:
        Boolean ``[D, H, W]`` consensus mask, or ``None`` if missing on disk.
    """
    scan_dir = _worldA_dir(patient_id, traj_index)
    if scan_dir is None:
        return None
    import nibabel as nib  # local import: only needed when the overlay is on

    path = scan_dir / "ensemble_mask.nii.gz"
    if not path.exists():
        return None
    return np.asarray(nib.load(str(path)).dataobj) > 0


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
        patient_id: Patient identifier (unused — title removed).
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

    metrics_rows: list[tuple[str, str, str, str, str]] = []
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
        abs_log_err = abs(mu - y_true)
        abs_diam_err_mm = abs(
            float(_log_to_diam_mm(np.array(mu))) - float(_log_to_diam_mm(np.array(y_true)))
        )
        metrics_rows.append(
            (
                label,
                color,
                f"{is_val:.2f}",
                f"{abs_log_err:.2f}",
                f"{abs_diam_err_mm:.1f}",
            )
        )

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

    all_t = np.concatenate([past_t, [t_star]])
    ax.set_xlim(all_t.min() - 0.45, t_star + 0.85)
    ax.set_xticks(np.unique(all_t))
    ax.set_xlabel(r"Follow-up Index $t$")
    ax.set_ylabel(r"$\log(V_{\mathrm{MEN}}+1)$")
    ax.grid(alpha=0.25, linestyle=":")

    # Secondary y-axis: equivalent spherical diameter (mm), d = 2(3V/4π)^{1/3}.
    # Gives a physical / clinical sense alongside the log-volume axis.
    sec = ax.secondary_yaxis("right", functions=(_log_to_diam_mm, _diam_mm_to_log))
    sec.set_ylabel(r"Equivalent Spherical Diameter $d$ (mm)")

    # --- Top-left metrics table (booktabs-style; LaTeX paper convention).
    # Columns: model | IS@95 (this patient's Winkler interval score) |
    # |Δlog(V+1)| (absolute log-volume residual, |μ̂ - y|, pure point-prediction
    # error) | |Δd| (mm) (the same residual mapped to equivalent spherical
    # diameter, a clinical length scale).
    col_labels = (
        r"Model",
        r"IS@95",
        r"$|\Delta \log(V{+}1)|$",
        r"$|\Delta d|$ (mm)",
    )
    cell_text = [[row[0], row[2], row[3], row[4]] for row in metrics_rows]
    tbl_x0, tbl_y0, tbl_w, tbl_h = 0.015, 0.66, 0.47, 0.30  # axes-fraction
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        bbox=(tbl_x0, tbl_y0, tbl_w, tbl_h),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    col_widths = (0.40, 0.16, 0.24, 0.20)
    n_rows = len(metrics_rows) + 1  # +1 for the header
    # Strip all cell edges; the booktabs rules below are drawn separately so
    # the top/bottom rules can be thicker than the header mid-rule.
    for (r_idx, c_idx), cell in tbl.get_celld().items():
        cell.set_width(col_widths[c_idx])
        cell.set_facecolor("white")
        cell.visible_edges = ""
        cell.set_linewidth(0.0)
        if r_idx == 0:
            pass  # header row: plain weight per request
        elif c_idx == 0:
            model_color = metrics_rows[r_idx - 1][1]
            cell.set_text_props(color=model_color, weight="bold")

    # Booktabs rules in axes coords: thick top-rule, thin mid-rule under
    # the header, thick bottom-rule under the last data row.
    row_h = tbl_h / n_rows
    y_top = tbl_y0 + tbl_h
    y_mid = y_top - row_h  # under the header
    y_bot = tbl_y0
    rule_x0, rule_x1 = tbl_x0, tbl_x0 + tbl_w
    for y_val, lw in ((y_top, 1.2), (y_mid, 0.6), (y_bot, 1.2)):
        ax.plot(
            [rule_x0, rule_x1],
            [y_val, y_val],
            transform=ax.transAxes,
            color="black",
            linewidth=lw,
            solid_capstyle="butt",
            clip_on=False,
            zorder=10,
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
        # Load every timepoint's full 3-D MEN mask first, so the display slice
        # can be anchored consistently across the longitudinal series. The
        # scans are co-registered (brain centroids agree across timepoints),
        # so a single axis-2 index shows the same anatomical plane everywhere.
        # Prefer the World-A LoRA-ensemble consensus mask — it is the segmentation
        # the conformal trajectory was built from. Index by within-patient
        # trajectory position (0-based), NOT by H5 scan_id: the latter retains
        # clinical numbering with gaps that does not align with the World-A
        # per_scan directory layout. Fall back to H5 ``segs`` only when the
        # per-member directory is unavailable for a scan.
        men_volumes: list[np.ndarray] = []
        var_volumes: list[np.ndarray | None] = []
        for k, (row, _tp) in enumerate(shown):
            ens = _ensemble_mask(patient_id, k)
            if ens is None:
                ens = np.isin(f["segs"][row, 0], MEN_LABELS)
            men_volumes.append(ens)
            var_volumes.append(_variance_map(patient_id, k))
        vox_counts = [int(m.sum()) for m in men_volumes]

        # Anchor the display plane to where the overlay carries lesion-localised
        # content. LoRA members can disagree on non-tumour voxels (skull base,
        # cerebellum edge), so a global "max variance" anchor drifts off-lesion.
        # Constrain to slices inside the lesion: among the z's covered by the
        # most timepoints' MEN masks (so the contour is informative), pick the
        # one whose per-member disagreement, *restricted to the union of masks
        # across timepoints*, is largest. Tie-break to the slice nearest the
        # timepoints' median peak-area z. Fallback: contour-only anchor.
        D2 = men_volumes[0].shape[2]
        z_present = np.array(
            [m.any(axis=(0, 1)) for m in men_volumes], dtype=int
        )  # [n_tp, D2]
        coverage = z_present.sum(axis=0)  # [D2]
        peak_z_list = [
            int(np.argmax(m.sum(axis=(0, 1)))) for m, n in zip(men_volumes, vox_counts) if n > 0
        ]
        med = float(np.median(peak_z_list)) if peak_z_list else D2 / 2.0

        if int(coverage.max()) > 0:
            band = np.flatnonzero(coverage == coverage.max())
            # Per-z lesion-restricted variance summed across timepoints. The
            # mask used for restriction is the OR of MEN masks across the
            # timepoints whose mask actually reaches that z, so we never include
            # variance from voxels outside any lesion at that plane.
            in_mask_var = np.zeros(D2, dtype=np.float64)
            for v, m in zip(var_volumes, men_volumes):
                if v is None:
                    continue
                in_mask_var += (v * m.astype(np.float32)).sum(axis=(0, 1))
            if in_mask_var[band].max() > 0:
                top = in_mask_var[band].max()
                cand = band[in_mask_var[band] >= 0.95 * top]
            else:
                cand = band
            ref_cz = int(min(cand, key=lambda z: abs(z - med)))
        else:
            ref_cz = D2 // 2

        t1c_slices: list[np.ndarray] = []
        men_masks: list[np.ndarray] = []
        var_maps: list[np.ndarray | None] = []
        mask_off_plane: list[bool] = []  # rendered at its own z, not ref_cz
        slice_z: list[int] = []  # actual axial index used for each panel
        for k, ((row, _tp), men_vol, var_vol, n_vox) in enumerate(
            zip(shown, men_volumes, var_volumes, vox_counts)
        ):
            # If the mass exists in 3D but does not reach ``ref_cz``, fall back
            # to this timepoint's own peak-area slice (per-scan z). The panel
            # is then labelled with its own z so the reader is not misled into
            # thinking the lesion has disappeared. The variance overlay follows
            # the same z so it remains lesion-localised on that timepoint.
            men_at_ref = men_vol[:, :, ref_cz].astype(np.uint8)
            if n_vox > 0 and int(men_at_ref.sum()) == 0:
                z_k = int(np.argmax(men_vol.sum(axis=(0, 1))))
                off = True
            else:
                z_k = ref_cz
                off = False
            t1c_slices.append(
                np.asarray(f["images"][row, T1C_CHAN, :, :, z_k], dtype=np.float32)
            )
            men_masks.append(men_vol[:, :, z_k].astype(np.uint8))
            mask_off_plane.append(off)
            slice_z.append(z_k)
            var_maps.append(var_vol[:, :, z_k] if var_vol is not None else None)

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
        top=0.965,
        bottom=0.11,
        figure=fig,
    )

    ax_pred = fig.add_subplot(gs[0, :])
    _draw_prediction_panel(ax_pred, past_t, past_y, t_star, y_true, pred_by_model, patient_id)

    # Per-slice grayscale normalisation: each timepoint's slice is stretched to
    # its own 1–99 percentile of positive pixels, so every panel fills the same
    # display range despite genuine inter-study intensity differences (scanner /
    # session / bias field). The MR intensities themselves are not modified —
    # only the imshow vmin/vmax for visual comparability across timepoints.
    var_im = None

    for k, (_row, tp) in enumerate(shown):
        ax = fig.add_subplot(gs[1, k])
        img = np.rot90(t1c_slices[k], k=1)
        mask = np.rot90(men_masks[k], k=1)
        pos = img[img > 0]
        if pos.size:
            vmin_k, vmax_k = np.percentile(pos, (1.0, 99.0))
            if vmax_k <= vmin_k:
                vmax_k = vmin_k + 1.0
        else:
            vmin_k, vmax_k = 0.0, 1.0
        ax.imshow(img, cmap="gray", vmin=vmin_k, vmax=vmax_k, interpolation="bilinear")
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
        if mask_off_plane[k]:
            # This timepoint's lesion did not reach the cohort-anchored plane,
            # so the panel falls back to the scan's own peak-area slice — flag
            # this so the reader is not misled by the anatomical shift.
            ax.text(
                0.5,
                0.035,
                f"Different Axial Plane (z={slice_z[k]})",
                transform=ax.transAxes,
                fontsize=7.5,
                color=MEN_COLOR,
                ha="center",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "black",
                    "alpha": 0.55,
                    "edgecolor": "none",
                },
            )
        ax.set_xticks([])
        ax.set_yticks([])
        marker = r"  $\leftarrow t^\ast$" if int(tp) == int(t_star) else ""
        # full_y is indexed by trajectory order, which matches `shown` order.
        ax.set_title(rf"$t = {int(tp)}$,  $\log(V+1) = {full_y[k]:.2f}${marker}", fontsize=8.5)
        _add_scale_bar(ax)

    if has_var and var_im is not None:
        row1_box = gs[1, n - 1].get_position(fig)
        cax = fig.add_axes((0.915, row1_box.y0, 0.015, row1_box.height))
        cb = fig.colorbar(var_im, cax=cax, orientation="vertical")
        cb.set_label(
            r"Segmentation Variance $p(1-p)$" + "\n" + r"across $M=20$ Ensemble Members",
            fontsize=8.5,
        )
        cb.ax.tick_params(labelsize=8)

    # Model identity is conveyed by the table's colour-coded model rows and the
    # plot ribbons, so the legend lists only the entries that do not appear in
    # the table: conditioning observations, held-out observation, segmentation.
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
            label="Conditioning Observations",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            markersize=14,
            color=STAR_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=r"Held-out Observation $y$ at $t^\ast$",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="-",
            linewidth=1.5,
            color=MEN_COLOR,
            label="Tumor Core Segmentation",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=170, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
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
