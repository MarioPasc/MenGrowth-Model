#!/usr/bin/env python
# experiments/lora_ablation/v3_style.py
"""Style constants for LoRA v3 figures and tables.

Single source of truth for condition names, colors (colorblind-safe, Wong 2011),
display labels, and figure settings used across v3_cache, v3_figures, and
regenerate_analysis.

Usage:
    from experiments.lora_ablation.v3_style import (
        V3_CONDITIONS, V3_COLORS, V3_LABELS, V3_SHAPE_LABELS,
        apply_v3_style,
    )
"""

from typing import Dict, List

# ============================================================================
# Conditions (ordered for display)
# ============================================================================

V3_CONDITIONS: List[str] = [
    "baseline_frozen",
    "baseline",
    "lora_r4_full",
    "lora_r8_full",
    "lora_r16_full",
    "lora_r32_full",
    "lora_r64_full",
]

# ============================================================================
# Colorblind-safe palette (Wong 2011, Nature Methods)
# ============================================================================

V3_COLORS: Dict[str, str] = {
    "baseline_frozen": "#999999",   # Gray
    "baseline":        "#E69F00",   # Orange
    "lora_r4_full":    "#56B4E9",   # Sky blue
    "lora_r8_full":    "#009E73",   # Bluish green
    "lora_r16_full":   "#F0E442",   # Yellow
    "lora_r32_full":   "#0072B2",   # Blue
    "lora_r64_full":   "#D55E00",   # Vermillion
}

# ============================================================================
# Display labels (short, for figures)
# ============================================================================

V3_LABELS: Dict[str, str] = {
    "baseline_frozen": "Frozen",
    "baseline":        "Baseline",
    "lora_r4_full":    "LoRA r=4",
    "lora_r8_full":    "LoRA r=8",
    "lora_r16_full":   "LoRA r=16",
    "lora_r32_full":   "LoRA r=32",
    "lora_r64_full":   "LoRA r=64",
}

# ============================================================================
# Semantic target labels (v3: revised shape targets)
# ============================================================================

V3_SHAPE_LABELS: List[str] = [
    "Sphericity",
    "Enhancement Ratio",
    "Infiltration Index",
]

V3_FEATURE_TYPES: List[str] = ["volume", "location", "shape"]

V3_FEATURE_LABELS: Dict[str, str] = {
    "volume": "Volume",
    "location": "Location",
    "shape": "Shape",
}

# ============================================================================
# Figure settings
# ============================================================================

FIGURE_DPI: int = 300
FIGURE_FONT_SIZE: int = 10
FIGURE_AXES_LABEL: int = 11
FIGURE_AXES_TITLE: int = 12
FIGURE_TICK_LABEL: int = 9
FIGURE_LEGEND_FONT: int = 9

# Probe type colors
PROBE_COLORS: Dict[str, str] = {
    "linear": "#0072B2",   # Blue
    "mlp":    "#D55E00",   # Vermillion
}

# DCI bar colors
DCI_COLORS: Dict[str, str] = {
    "D": "#0072B2",   # Blue
    "C": "#009E73",   # Green
    "I": "#D55E00",   # Vermillion
}


def apply_v3_style() -> None:
    """Apply publication style settings for v3 figures.

    Configures matplotlib rcParams for thesis-quality output.
    Must be called before any figure creation.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.rcParams.update({
        "font.size": FIGURE_FONT_SIZE,
        "axes.labelsize": FIGURE_AXES_LABEL,
        "axes.titlesize": FIGURE_AXES_TITLE,
        "xtick.labelsize": FIGURE_TICK_LABEL,
        "ytick.labelsize": FIGURE_TICK_LABEL,
        "legend.fontsize": FIGURE_LEGEND_FONT,
        "figure.dpi": 150,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })


def get_color(condition: str) -> str:
    """Get color for a condition, with fallback for unknown conditions.

    Args:
        condition: Condition name.

    Returns:
        Hex color string.
    """
    return V3_COLORS.get(condition, "#333333")


def get_label(condition: str) -> str:
    """Get display label for a condition, with fallback.

    Args:
        condition: Condition name.

    Returns:
        Human-readable label.
    """
    return V3_LABELS.get(condition, condition)
