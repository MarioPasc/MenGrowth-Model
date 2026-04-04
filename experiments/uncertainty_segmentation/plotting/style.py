"""Shared style constants and annotation helpers for ensemble figures.

All figure modules import from this module. No figure module defines its
own colors or annotation helpers.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt

from experiments.utils.settings import (
    ENSEMBLE_COLORS,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_significance_stars,
)

# ---------------------------------------------------------------------------
# Re-exported color constants (short aliases used by every figure module)
# ---------------------------------------------------------------------------

C_BASELINE: str = ENSEMBLE_COLORS["baseline"]
C_ENSEMBLE: str = ENSEMBLE_COLORS["ensemble"]
C_MEMBERS: str = ENSEMBLE_COLORS["members"]
C_BEST: str = ENSEMBLE_COLORS["best_member"]
C_DELTA_POS: str = ENSEMBLE_COLORS["delta_positive"]
C_DELTA_NEG: str = ENSEMBLE_COLORS["delta_negative"]
C_FILL: str = ENSEMBLE_COLORS["fill"]
C_MEDIAN: str = ENSEMBLE_COLORS["median"]
MEMBER_CMAP = plt.cm.Set3


def setup_style(style_config: dict | None = None) -> None:
    """Configure matplotlib for publication-quality ensemble figures.

    Args:
        style_config: Optional dict from config.yaml ``style`` section.
            If None, uses sensible defaults.
    """
    # Start from the IEEE base defined in settings.py
    apply_ieee_style()

    cfg = style_config or {}

    # Ensemble-specific overrides
    mpl.rcParams.update({
        "font.family": cfg.get("font_family", "serif"),
        "font.serif": cfg.get("font_serif",
                               ["CMU Serif", "DejaVu Serif", "Times New Roman"]),
        "font.size": cfg.get("font_size", 9),
        "axes.titlesize": cfg.get("axes_title_size", 10),
        "axes.labelsize": cfg.get("font_size", 9),
        "xtick.labelsize": cfg.get("tick_size", 8),
        "ytick.labelsize": cfg.get("tick_size", 8),
        "legend.fontsize": cfg.get("legend_size", 8),
        "figure.dpi": cfg.get("figure_dpi", 150),
        "savefig.dpi": cfg.get("save_dpi", 300),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": cfg.get("pdf_fonttype", 42),
        "ps.fonttype": 42,
    })


def significance_label(p: float) -> str:
    """Convert p-value to significance stars.

    Args:
        p: P-value from statistical test.

    Returns:
        String with stars or "n.s.".
    """
    return get_significance_stars(p)


def add_stat_bracket(
    ax: matplotlib.axes.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    fontsize: int = 7,
) -> None:
    """Draw a significance bracket between two x positions.

    Args:
        ax: Target axes.
        x1: Left position.
        x2: Right position.
        y: Baseline y-coordinate of the bracket.
        h: Height of the bracket arms.
        text: Text to display above the bracket.
        fontsize: Font size for the annotation text.
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c="k")
    ax.text(
        (x1 + x2) / 2, y + h, text,
        ha="center", va="bottom", fontsize=fontsize,
    )
