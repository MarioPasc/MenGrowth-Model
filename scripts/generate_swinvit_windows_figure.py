"""Generate Figure: Input representation with Swin Transformer windows and segmentation overlays.

Panel (a): 2×2 grid of the four MRI modalities (FLAIR, T1ce, T1n, T2w) with:
  - Segmentation label overlays (TC, WT, ET) with alpha=0.2 fill, alpha=1.0 contour
  - Swin Transformer local attention window grid overlaid on top
  - All-black background, IEEE publication style from project style system

Output: PDF figure for the thesis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from scipy import ndimage

from experiments.uncertainty_segmentation.plotting.style import (
    C_ET,
    C_TC,
    C_WT,
    REGION_DISPLAY_SHORT,
    setup_style,
)
from experiments.utils.settings import (
    IEEE_TEXT_WIDTH_INCHES,
    PLOT_SETTINGS,
)

# ── Configuration ────────────────────────────────────────────────────────────

H5_PATH = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/source/BraTS_MEN.h5"
)
OUTPUT_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/bachelor_thesis/"
    "68596a200c0e0e3876880afa/figures/dataset/methodology"
)
OUTPUT_NAME = "swinvit_windows.pdf"

# Swin Transformer architecture constants (BrainSegFounder)
PATCH_SIZE = 2
WINDOW_SIZE = 7
WINDOW_VOXELS = PATCH_SIZE * WINDOW_SIZE  # 14 voxels per window in image space

# Channel order: [FLAIR, T1ce, T1n, T2w]
MODALITY_NAMES = ["T2-FLAIR", "T1ce", "T1n", "T2w"]

# Region colors from project style (colorblind-safe)
REGION_COLORS = {"WT": C_WT, "TC": C_TC, "ET": C_ET}

FILL_ALPHA = 0.2
GRID_COLOR = "#FFFFFF"
GRID_ALPHA = 0.30
GRID_LINEWIDTH = 0.4


# ── Helper functions ─────────────────────────────────────────────────────────


def find_best_sample_and_slice(f: h5py.File, n_candidates: int = 50) -> tuple[int, int]:
    """Find sample and axial slice with the most diverse label visibility.

    Searches the first n_candidates samples for the axial slice that maximises
    the number of distinct non-background labels visible, then breaks ties by
    total tumour cross-sectional area.
    """
    best_idx, best_slice, best_score = 0, 96, 0

    for idx in range(min(n_candidates, f["segs"].shape[0])):
        seg = f["segs"][idx, 0]  # [192, 192, 192]
        for z in range(40, 160):
            axial = seg[:, :, z]
            labels_present = set(np.unique(axial)) - {0}
            n_labels = len(labels_present)
            area = int((axial > 0).sum())
            score = n_labels * 100_000 + area
            if score > best_score:
                best_score = score
                best_idx = idx
                best_slice = z

    return best_idx, best_slice


def derive_hierarchical_masks(seg_slice: np.ndarray) -> dict[str, np.ndarray]:
    """Derive TC, WT, ET binary masks from raw integer labels (2D axial slice)."""
    return {
        "WT": (seg_slice > 0).astype(np.uint8),
        "TC": ((seg_slice == 1) | (seg_slice == 3)).astype(np.uint8),
        "ET": (seg_slice == 3).astype(np.uint8),
    }


def get_contour(mask: np.ndarray) -> np.ndarray:
    """Extract 1-pixel-wide contour from a binary mask."""
    dilated = ndimage.binary_dilation(mask, iterations=1)
    eroded = ndimage.binary_erosion(mask, iterations=1)
    return (dilated.astype(np.uint8) - eroded.astype(np.uint8)).clip(0, 1)


def normalize_slice(img_slice: np.ndarray) -> np.ndarray:
    """Z-score normalise a 2D slice over nonzero voxels, then clip to [0, 1]."""
    nonzero = img_slice[img_slice != 0]
    if len(nonzero) == 0:
        return np.zeros_like(img_slice, dtype=np.float32)
    mu, sigma = nonzero.mean(), nonzero.std()
    if sigma < 1e-8:
        return np.zeros_like(img_slice, dtype=np.float32)
    normed = (img_slice - mu) / sigma
    normed = np.clip(normed, -3, 3)
    return ((normed + 3) / 6).astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────


def render_panel(
    ax: plt.Axes,
    img: np.ndarray,
    masks: dict[str, np.ndarray],
    name: str,
) -> None:
    """Render a single modality panel with segmentation overlays and window grid."""
    draw_order = ["WT", "TC", "ET"]
    ax.set_facecolor("black")
    ax.imshow(img.T, cmap="gray", origin="lower", vmin=0, vmax=1)

    for label_name in draw_order:
        mask = masks[label_name]
        if mask.sum() == 0:
            continue
        color_rgb = to_rgba(REGION_COLORS[label_name])[:3]

        fill_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        fill_rgba[mask == 1, :3] = color_rgb
        fill_rgba[mask == 1, 3] = FILL_ALPHA
        ax.imshow(fill_rgba.transpose(1, 0, 2), origin="lower")

        contour = get_contour(mask)
        contour_rgba = np.zeros((*contour.shape, 4), dtype=np.float32)
        contour_rgba[contour == 1, :3] = color_rgb
        contour_rgba[contour == 1, 3] = 1.0
        ax.imshow(contour_rgba.transpose(1, 0, 2), origin="lower")

    h, w = img.shape
    for x in range(0, w + 1, WINDOW_VOXELS):
        ax.axvline(x, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    for y in range(0, h + 1, WINDOW_VOXELS):
        ax.axhline(y, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

    ax.set_title(
        name,
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        fontweight="bold",
        color="white",
        pad=4,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_legend_handles() -> list:
    """Build shared legend handles for both layouts."""
    draw_order = ["WT", "TC", "ET"]
    handles = [
        mpatches.Patch(
            facecolor=to_rgba(REGION_COLORS[k], FILL_ALPHA),
            edgecolor=REGION_COLORS[k],
            linewidth=1.5,
            label=REGION_DISPLAY_SHORT[k.lower()],
        )
        for k in draw_order
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color=GRID_COLOR,
            linewidth=1.0,
            alpha=0.6,
            label=f"Swin window ({WINDOW_VOXELS}" + r"$\times$" + f"{WINDOW_VOXELS} vox)",
        )
    )
    return handles


def save_figure(
    fig: plt.Figure,
    img_normed: list[np.ndarray],
    masks: dict[str, np.ndarray],
    axes: np.ndarray,
    out_path: Path,
    *,
    wspace: float,
    hspace: float,
    legend_y: float,
    rect: list[float],
) -> None:
    """Populate axes, add legend, and save."""
    for ax, img, name in zip(axes, img_normed, MODALITY_NAMES):
        render_panel(ax, img, masks, name)

    leg = fig.legend(
        handles=make_legend_handles(),
        loc="lower center",
        ncol=4,
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=True,
        fancybox=False,
        edgecolor="0.4",
        facecolor="black",
        labelcolor="white",
        bbox_to_anchor=(0.5, legend_y),
    )
    leg.get_frame().set_linewidth(0.6)

    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    fig.savefig(
        str(out_path),
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=PLOT_SETTINGS["dpi_print"],
        facecolor="black",
    )
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    if not H5_PATH.exists():
        print(f"Error: H5 file not found at {H5_PATH}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    with h5py.File(str(H5_PATH), "r") as f:
        print("Searching for best sample and slice...")
        sample_idx, slice_idx = find_best_sample_and_slice(f)
        print(f"  Selected sample {sample_idx}, axial slice {slice_idx}")
        images = f["images"][sample_idx]
        seg = f["segs"][sample_idx, 0]

    img_slices = [images[ch, :, :, slice_idx] for ch in range(4)]
    seg_slice = seg[:, :, slice_idx]
    img_normed = [normalize_slice(s) for s in img_slices]
    masks = derive_hierarchical_masks(seg_slice)

    fig_w = IEEE_TEXT_WIDTH_INCHES

    # ── 2×2 layout ───────────────────────────────────────────────────────
    fig_2x2, axes_2x2 = plt.subplots(
        2,
        2,
        figsize=(fig_w, fig_w),
        dpi=PLOT_SETTINGS["dpi_screen"],
        facecolor="black",
    )
    fig_2x2.patch.set_facecolor("black")
    save_figure(
        fig_2x2,
        img_normed,
        masks,
        axes_2x2.ravel(),
        OUTPUT_DIR / "swinvit_windows.pdf",
        wspace=0.04,
        hspace=0.10,
        legend_y=-0.01,
        rect=[0, 0.04, 1, 1],
    )

    # ── 1×4 layout ───────────────────────────────────────────────────────
    fig_h = fig_w * 0.28
    fig_1x4, axes_1x4 = plt.subplots(
        1,
        4,
        figsize=(fig_w, fig_h),
        dpi=PLOT_SETTINGS["dpi_screen"],
        facecolor="black",
    )
    fig_1x4.patch.set_facecolor("black")
    save_figure(
        fig_1x4,
        img_normed,
        masks,
        axes_1x4.ravel(),
        OUTPUT_DIR / "swinvit_windows_1x4.pdf",
        wspace=0.04,
        hspace=0.0,
        legend_y=-0.08,
        rect=[0, 0.10, 1, 1],
    )


if __name__ == "__main__":
    main()
