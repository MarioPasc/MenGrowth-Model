"""Inter-LoRA-specific style constants, palettes, and figure-saving helpers."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from experiments.utils.settings import apply_ieee_style, get_significance_stars

__all__ = [
    "setup_inter_lora_style",
    "save_figure",
    "get_significance_stars",
    "LABEL_COLORS",
    "BASELINE_COLOR",
    "REGION_KEYS",
    "REGION_DISPLAY",
    "rank_color",
    "MM_TO_INCH",
    "SINGLE_COL_MM",
    "DOUBLE_COL_MM",
    "FULL_PAGE_MM",
]

# ---------- Label colours (spec-mandated, differ from intra-rank) ----------
LABEL_COLORS: dict[str, str] = {
    "tc": "#009E73",
    "wt": "#0072B2",
    "et": "#D55E00",
}

BASELINE_COLOR: str = "#3a3a3a"

REGION_KEYS: tuple[str, ...] = ("tc", "wt", "et")
REGION_DISPLAY: dict[str, str] = {
    "tc": "TC",
    "wt": "WT",
    "et": "ET",
}

# ---------- Figure sizes ----------
MM_TO_INCH: float = 1.0 / 25.4
SINGLE_COL_MM: float = 88.0
DOUBLE_COL_MM: float = 180.0
FULL_PAGE_MM: tuple[float, float] = (180.0, 220.0)

DEFAULT_DPI: int = 600


def rank_colormap(
    ranks: list[int],
) -> tuple[Normalize, mpl.colors.Colormap]:
    """Return a viridis normalizer on log2(rank) scale."""
    log2_ranks = [np.log2(r) for r in ranks if r > 0]
    norm = Normalize(vmin=min(log2_ranks), vmax=max(log2_ranks))
    return norm, plt.cm.viridis


def rank_color(rank: int, ranks: list[int]) -> Any:
    """Return the viridis colour for a single rank."""
    norm, cmap = rank_colormap(ranks)
    return cmap(norm(np.log2(rank)))


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def setup_inter_lora_style(config: dict | None = None) -> None:
    """Apply IEEE base + inter-lora overrides."""
    apply_ieee_style()
    cfg = config or {}
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": cfg.get("font_size", 9),
            "axes.titlesize": cfg.get("axes_title_size", 10),
            "xtick.labelsize": cfg.get("tick_size", 8),
            "ytick.labelsize": cfg.get("tick_size", 8),
            "legend.fontsize": cfg.get("legend_size", 8),
            "figure.dpi": 150,
            "savefig.dpi": cfg.get("save_dpi", DEFAULT_DPI),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
        },
    )


def save_figure(
    fig: Figure,
    path: Path,
    *,
    title: str = "",
    description: str = "",
    dpi: int | None = None,
    transparent: bool = False,
) -> None:
    """Save figure with git SHA + ISO timestamp in metadata."""
    sha = _git_sha()
    ts = datetime.now(tz=UTC).isoformat(timespec="seconds")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        metadata = {
            "Title": title,
            "Subject": description,
            "Creator": f"inter_lora_report (git:{sha})",
            "CreationDate": ts,
        }
    else:
        metadata = {
            "Title": title,
            "Description": description,
            "Software": f"inter_lora_report (git:{sha})",
            "Creation Time": ts,
        }
    fig.savefig(
        path,
        dpi=dpi or DEFAULT_DPI,
        transparent=transparent,
        metadata=metadata,
    )
