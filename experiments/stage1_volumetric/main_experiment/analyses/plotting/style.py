"""Centralised style for the main-experiment figures.

All colours, font sizes, line widths, and matplotlib rcParams used by the
plotting scripts in this package live here so the visual identity of the
experiment stays consistent across panels and figures.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------

#: Sequential, perceptually uniform colormap for the (π, log σ²_v, density)
#: surface. A monotone map keeps the eye on the slice highlights.
SURFACE_CMAP = cm.viridis

#: Slight transparency lets the highlighted slice curves stand out.
SURFACE_ALPHA: float = 0.85
SURFACE_RSTRIDE: int = 1
SURFACE_CSTRIDE: int = 1
SURFACE_LINEWIDTH: float = 0.0  # no facet edges
SURFACE_ANTIALIASED: bool = True

# ---------------------------------------------------------------------------
# Slice colours and labels
# ---------------------------------------------------------------------------

SLICE_COLORS: dict[str, str] = {
    "pi_min": "#d62728",  # red — π=0 (all-noisy)
    "pi_empirical": "#2ca02c",  # green — π=π̂ (empirical match)
    "pi_balanced": "#ff7f0e",  # orange — π=0.5 (balanced bimodal)
    "pi_max": "#1f77b4",  # blue — π=1 (all-clean)
}

SLICE_LABELS: dict[str, str] = {
    "pi_min": r"$\pi=0$ (all-noisy)",
    "pi_empirical": r"$\pi=\hat{\pi}$ (empirical match)",
    "pi_balanced": r"$\pi=0.5$ (balanced bimodal)",
    "pi_max": r"$\pi=1$ (all-clean)",
}

#: Order in which slices are stacked in legends and grids.
SLICE_ORDER: tuple[str, ...] = ("pi_max", "pi_empirical", "pi_balanced", "pi_min")

#: Width of the highlighted slice curves both on the surface and in the
#: 2-D cross-section panel.
SLICE_LINE_WIDTH: float = 2.5
SLICE_LINE_ALPHA: float = 1.0

# Marker for "this is exactly the empirical-fit slice"
EMPIRICAL_MARKER_COLOR: str = "#2ca02c"
EMPIRICAL_MARKER_SIZE: float = 60.0
EMPIRICAL_MARKER_LINEWIDTH: float = 1.4

# ---------------------------------------------------------------------------
# Empirical histogram overlay
# ---------------------------------------------------------------------------

HIST_COLOR: str = "#555555"
HIST_ALPHA: float = 0.35
HIST_EDGE_COLOR: str = "white"
HIST_BINS: int = 24

# ---------------------------------------------------------------------------
# Figure / axes
# ---------------------------------------------------------------------------

FIG_DPI: int = 300
FIG_SIZE_INCHES: tuple[float, float] = (15.5, 6.4)
TITLE_SIZE: float = 13.0
SECTION_TITLE_SIZE: float = 11.5
LABEL_SIZE: float = 10.5
TICK_SIZE: float = 9.0
LEGEND_SIZE: float = 9.0
GRID_ALPHA: float = 0.35

# 3-D axis pane backgrounds (matplotlib defaults are too dark on prints)
PANE_FACE_COLOR: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0)
PANE_EDGE_COLOR: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)


def apply_global_style() -> None:
    """Mutate matplotlib rcParams to match the experiment style."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": SECTION_TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "savefig.bbox": "tight",
            "axes.grid": False,
        }
    )


def style_3d_axes(ax) -> None:
    """Apply consistent 3-D axes cosmetics."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE_FACE_COLOR)
        axis.pane.set_edgecolor(PANE_EDGE_COLOR)
    ax.grid(True, alpha=GRID_ALPHA)


def style_2d_axes(ax) -> None:
    """Apply consistent 2-D axes cosmetics."""
    ax.grid(True, linestyle=":", alpha=GRID_ALPHA)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)


__all__ = [
    "SURFACE_CMAP",
    "SURFACE_ALPHA",
    "SURFACE_RSTRIDE",
    "SURFACE_CSTRIDE",
    "SURFACE_LINEWIDTH",
    "SURFACE_ANTIALIASED",
    "SLICE_COLORS",
    "SLICE_LABELS",
    "SLICE_ORDER",
    "SLICE_LINE_WIDTH",
    "SLICE_LINE_ALPHA",
    "EMPIRICAL_MARKER_COLOR",
    "EMPIRICAL_MARKER_SIZE",
    "EMPIRICAL_MARKER_LINEWIDTH",
    "HIST_COLOR",
    "HIST_ALPHA",
    "HIST_EDGE_COLOR",
    "HIST_BINS",
    "FIG_DPI",
    "FIG_SIZE_INCHES",
    "TITLE_SIZE",
    "SECTION_TITLE_SIZE",
    "LABEL_SIZE",
    "TICK_SIZE",
    "LEGEND_SIZE",
    "GRID_ALPHA",
    "apply_global_style",
    "style_3d_axes",
    "style_2d_axes",
    "plt",
]
