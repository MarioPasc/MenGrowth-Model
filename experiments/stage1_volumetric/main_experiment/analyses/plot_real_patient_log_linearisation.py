"""Real-patient log-linearisation figure for the BSc thesis.

Patient: MenGrowth-0055 (n=4 timepoints, monotonic ~8x growth on
log(V_MEN+1), the steepest 4-scan trajectory in the cohort after
excluding patients 0004/0008/0022 with zero-volume scans).

Layout:
    [ thin colorbar across the figure top                                   ]
    [ row 0:  mm^3 space (Gompertz family + real)  |  log(V+1) space        ]
    [ row 1:  slice@scan0  |  slice@scan1  |  slice@scan2  |  slice@scan3   ]

Top-row curves are Gompertz V(t) = V_0 exp((a/b)(1 - e^{-bt})) with
V_0 fixed at the patient's first scan, 1/b = 15 yr (Engelhardt 2023
WHO-I deceleration timescale), and a swept across a viridis colormap.
The real trajectory is overlaid in dark red.

Bottom-row slices are T1ce (channel 1 = "t1c") at the MEN-centroid
axial slice, cropped to a fixed window around the tumour centroid
of scan 0 (so the brain context stays consistent across timepoints).
The MEN mask (labels 1|3) is overlaid in blue (fill alpha=0.3,
contour alpha=1.0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

H5_PATH = Path("/media/mpascual/MeningD2/MENINGIOMAS/MENGROWTH/050526/h5_format/MenGrowth.h5")
PATIENT_ID = "MenGrowth-0055"

# Display settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "figure.dpi": 120,
    }
)

# Gompertz parameters
INV_B_YR = 15.0  # deceleration timescale (Engelhardt 2023, WHO-I)
B = 1.0 / INV_B_YR
N_A = 11
A_MIN, A_MAX = 0.10, 1.00

# Slice display
CROP_HALF = 48  # 96x96 window around tumour centroid
T1C_CHAN = 1  # channel index of "t1c"


def gompertz(t: np.ndarray, V0: float, a: float, b: float = B) -> np.ndarray:
    return V0 * np.exp((a / b) * (1.0 - np.exp(-b * t)))


def find_patient_indices(f: h5py.File, pid: str) -> tuple[list[int], np.ndarray]:
    plist = f["longitudinal"]["patient_list"][:].astype(str)
    offs = f["longitudinal"]["patient_offsets"][:]
    pi = int(np.where(plist == pid)[0][0])
    i0, i1 = int(offs[pi]), int(offs[pi + 1])
    tpi = f["timepoint_idx"][i0:i1]
    order = np.argsort(tpi)
    return [int(i0 + k) for k in order], tpi[order].astype(float)


def crop_xy(arr2d: np.ndarray, cy: int, cx: int, half: int) -> np.ndarray:
    H, W = arr2d.shape
    y0, y1 = max(cy - half, 0), min(cy + half, H)
    x0, x1 = max(cx - half, 0), min(cx + half, W)
    crop = arr2d[y0:y1, x0:x1]
    # pad to (2*half, 2*half) if hit a border
    out = np.zeros((2 * half, 2 * half), dtype=crop.dtype)
    out[: crop.shape[0], : crop.shape[1]] = crop
    return out


def main(out_pdf: Path, out_png: Path) -> None:
    with h5py.File(H5_PATH, "r") as f:
        idxs, ts = find_patient_indices(f, PATIENT_ID)
        n = len(idxs)
        logger.info("Patient %s indices=%s timepoints=%s", PATIENT_ID, idxs, ts.tolist())

        # log-volume MEN (col 3 = labels 1|3)
        ys = f["semantic"]["volume"][:, 3][idxs].astype(float)
        Vs = np.exp(ys) - 1.0
        logger.info("V_MEN per scan (mm^3): %s", [f"{v:.0f}" for v in Vs])

        # Slices and masks
        slices_t1c: list[np.ndarray] = []
        masks_men: list[np.ndarray] = []
        # Use first-scan tumour centroid as the common crop center.
        seg0 = f["segs"][idxs[0], 0]
        men0 = np.isin(seg0, [1, 3])
        if men0.sum() == 0:
            raise RuntimeError("Empty MEN mask at first scan")
        cent0 = np.argwhere(men0).mean(axis=0)
        cy0 = int(round(cent0[0]))  # volume axis 0 -> slice row
        cx0 = int(round(cent0[1]))  # volume axis 1 -> slice col
        cz0 = int(round(cent0[2]))  # volume axis 2 -> slice index

        for k, gi in enumerate(idxs):
            t1c_vol = f["images"][gi, T1C_CHAN]
            seg = f["segs"][gi, 0]
            men = np.isin(seg, [1, 3])
            # Use the per-scan z that maximises the in-slice MEN area to
            # keep the tumour visible even as it grows; xy crop stays at
            # scan-0 centroid for visual stability.
            if men.sum() > 0:
                areas = men.sum(axis=(0, 1))
                cz = int(np.argmax(areas))
            else:
                cz = cz0
            t1c_slice = t1c_vol[:, :, cz]
            men_slice = men[:, :, cz].astype(np.uint8)
            slices_t1c.append(crop_xy(t1c_slice, cy0, cx0, CROP_HALF))
            masks_men.append(crop_xy(men_slice, cy0, cx0, CROP_HALF))

    # Real-trajectory slope on log scale (estimate of patient's a*).
    a_real = float(np.polyfit(ts, ys, 1)[0])
    logger.info("Real log-vol slope a* = %.3f /timepoint (1yr/tp assumed)", a_real)

    # Time grid and Gompertz family
    t_dense = np.linspace(0.0, ts[-1] + 0.5, 300)
    a_grid = np.linspace(A_MIN, A_MAX, N_A)
    V0 = float(Vs[0])
    norm = Normalize(vmin=A_MIN, vmax=A_MAX)
    cmap = plt.get_cmap("viridis")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(10.5, 7.2))
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=n,
        height_ratios=[0.05, 1.0, 1.0],
        hspace=0.55,
        wspace=0.22,
        figure=fig,
        top=0.94,
        bottom=0.06,
        left=0.07,
        right=0.97,
    )

    # Colorbar (top)
    cax = fig.add_subplot(gs[0, :])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"Gompertz initial growth rate $a$ (yr$^{-1}$)", fontsize=10)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")

    # Row 0: mm^3 (col 0 spans n//2 cols, log span the rest)
    half = n // 2
    ax_lin = fig.add_subplot(gs[1, :half])
    ax_log = fig.add_subplot(gs[1, half:])

    for a in a_grid:
        V = gompertz(t_dense, V0, a)
        ax_lin.plot(t_dense, V, color=cmap(norm(a)), alpha=0.55, linewidth=1.2)
        ax_log.plot(t_dense, np.log(V + 1.0), color=cmap(norm(a)), alpha=0.55, linewidth=1.2)

    ax_lin.plot(
        ts,
        Vs,
        color="darkred",
        marker="o",
        markersize=6,
        linewidth=1.8,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=5,
        label=f"{PATIENT_ID}",
    )
    ax_log.plot(
        ts,
        ys,
        color="darkred",
        marker="o",
        markersize=6,
        linewidth=1.8,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=5,
        label=f"{PATIENT_ID}",
    )

    ax_lin.set_xlabel("Follow-up (timepoints, $\\approx$ years)")
    ax_lin.set_ylabel(r"Tumour volume $V$ (mm$^3$)")
    ax_lin.set_ylim(0.0, 1.5 * float(Vs.max()))
    ax_lin.grid(alpha=0.25, linestyle=":")
    ax_lin.legend(loc="upper left", frameon=False)

    ax_log.set_xlabel("Follow-up (timepoints, $\\approx$ years)")
    ax_log.set_ylabel(r"$\log(V+1)$")
    ax_log.set_ylim(float(ys.min()) - 0.3, float(ys.max()) + 1.2)
    ax_log.grid(alpha=0.25, linestyle=":")
    ax_log.legend(loc="upper left", frameon=False)

    # Row 1: slice panels
    # Use shared display intensity range across scans for fair comparison.
    all_intens = np.concatenate([s.ravel() for s in slices_t1c])
    vmin, vmax = np.percentile(all_intens[all_intens > 0], (1.0, 99.0))

    for k in range(n):
        ax = fig.add_subplot(gs[2, k])
        # Rotate 90 deg CCW for a more conventional axial orientation.
        img = np.rot90(slices_t1c[k], k=1)
        m = np.rot90(masks_men[k], k=1)
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
        # Filled mask (alpha=0.3) using a single-color cmap.
        mask_disp = np.ma.masked_where(m == 0, m)
        ax.imshow(
            mask_disp,
            cmap=plt.matplotlib.colors.ListedColormap(["#1f77ff"]),
            alpha=0.30,
            vmin=0,
            vmax=1,
        )
        # Contour (alpha=1.0)
        ax.contour(m, levels=[0.5], colors="#1f77ff", linewidths=1.2, alpha=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {int(ts[k])}    $V$ = {Vs[k]:,.0f} mm$^3$", fontsize=9)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", out_pdf)
    logger.info("Wrote %s", out_png)
    plt.close(fig)


if __name__ == "__main__":
    THESIS_FIG_DIR = Path(
        "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/"
        "bachelor_thesis/68596a200c0e0e3876880afa/figures/methodology"
    )
    main(
        out_pdf=THESIS_FIG_DIR / "log_linearisation_real_patient.pdf",
        out_png=THESIS_FIG_DIR / "log_linearisation_real_patient.png",
    )
