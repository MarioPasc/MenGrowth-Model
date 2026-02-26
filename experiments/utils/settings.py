"""Publication-ready plot settings for LoRA Ablation thesis figures.

IEEE-compliant settings with Paul Tol colorblind-friendly palettes
adapted for encoder adaptation ablation study.

References:
    - Paul Tol's color schemes: https://personal.sron.nl/~pault/
    - IEEE publication guidelines
    - scienceplots: https://github.com/garrettj403/SciencePlots
"""

from __future__ import annotations

# =============================================================================
# Paul Tol Color Palettes (SRON - colorblind safe)
# =============================================================================

PAUL_TOL_BRIGHT = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

PAUL_TOL_HIGH_CONTRAST = {
    "blue": "#004488",
    "yellow": "#DDAA33",
    "red": "#BB5566",
}

PAUL_TOL_MUTED = [
    "#CC6677",  # rose
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # green
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
]

# =============================================================================
# LoRA Ablation Condition Visual Encoding
# =============================================================================

CONDITION_COLORS = {
    # v2 conditions (Paul Tol)
    "baseline": "#BBBBBB",  # Grey - frozen encoder (reference)
    "lora_r2": "#88CCEE",  # Light cyan - minimal adaptation (lower bound)
    "lora_r4": "#66CCEE",  # Cyan - light adaptation
    "lora_r8": "#4477AA",  # Blue - moderate adaptation
    "lora_r16": "#228833",  # Green - strong adaptation
    "lora_r32": "#117733",  # Dark green - maximum adaptation (saturation test)
    # v3 conditions (Wong 2011)
    "baseline_frozen": "#999999",  # Gray
    "lora_r4_full": "#56B4E9",  # Sky blue
    "lora_r8_full": "#009E73",  # Bluish green
    "lora_r16_full": "#F0E442",  # Yellow
    "lora_r32_full": "#0072B2",  # Blue
    "lora_r64_full": "#D55E00",  # Vermillion
    # DoRA conditions
    "dora_r2": "#b2df8a",
    "dora_r4": "#6a3d9a",
    "dora_r8": "#cab2d6",
    "dora_r16": "#fb9a99",
    "dora_r32": "#fdbf6f",
}

CONDITION_MARKERS = {
    "baseline": "o",  # Circle
    "baseline_frozen": "X",  # X marker
    "lora_r2": "v",  # Triangle down
    "lora_r4": "s",  # Square
    "lora_r8": "D",  # Diamond
    "lora_r16": "^",  # Triangle up
    "lora_r32": "p",  # Pentagon
    "lora_r4_full": "s",
    "lora_r8_full": "D",
    "lora_r16_full": "^",
    "lora_r32_full": "p",
    "lora_r64_full": "h",  # Hexagon
    "dora_r2": "v",
    "dora_r4": "s",
    "dora_r8": "D",
    "dora_r16": "^",
    "dora_r32": "p",
}

CONDITION_LINESTYLES = {
    "baseline": "--",  # Dashed (reference)
    "baseline_frozen": ":",  # Dotted (frozen reference)
    "lora_r2": "-",  # Solid
    "lora_r4": "-",  # Solid
    "lora_r8": "-",  # Solid
    "lora_r16": "-",  # Solid
    "lora_r32": "-",  # Solid
    "lora_r4_full": "-",
    "lora_r8_full": "-",
    "lora_r16_full": "-",
    "lora_r32_full": "-",
    "lora_r64_full": "-",
    "dora_r2": "-.",  # Dash-dot (DoRA)
    "dora_r4": "-.",
    "dora_r8": "-.",
    "dora_r16": "-.",
    "dora_r32": "-.",
}

CONDITION_HATCHES = {
    "baseline": "//",  # Diagonal stripes
    "baseline_frozen": "xx",  # Cross-hatch (frozen)
    "lora_r2": None,  # Solid
    "lora_r4": None,  # Solid
    "lora_r8": None,  # Solid
    "lora_r16": None,  # Solid
    "lora_r32": "\\\\",  # Back diagonal (saturation marker)
    "lora_r4_full": None,
    "lora_r8_full": None,
    "lora_r16_full": None,
    "lora_r32_full": None,
    "lora_r64_full": "\\\\",
    "dora_r2": None,
    "dora_r4": None,
    "dora_r8": None,
    "dora_r16": None,
    "dora_r32": None,
}

# Display labels for conditions (human-readable)
CONDITION_LABELS = {
    "baseline": "Baseline",
    "baseline_frozen": "Frozen",
    "lora_r2": "LoRA r=2",
    "lora_r4": "LoRA r=4",
    "lora_r8": "LoRA r=8",
    "lora_r16": "LoRA r=16",
    "lora_r32": "LoRA r=32",
    "lora_r4_full": "LoRA r=4",
    "lora_r8_full": "LoRA r=8",
    "lora_r16_full": "LoRA r=16",
    "lora_r32_full": "LoRA r=32",
    "lora_r64_full": "LoRA r=64",
    "dora_r2": "DoRA r=2",
    "dora_r4": "DoRA r=4",
    "dora_r8": "DoRA r=8",
    "dora_r16": "DoRA r=16",
    "dora_r32": "DoRA r=32",
}

# Short labels for tight spaces (axis ticks)
CONDITION_LABELS_SHORT = {
    "baseline": "Base",
    "baseline_frozen": "Frozen",
    "lora_r2": "r=2",
    "lora_r4": "r=4",
    "lora_r8": "r=8",
    "lora_r16": "r=16",
    "lora_r32": "r=32",
    "lora_r4_full": "r=4",
    "lora_r8_full": "r=8",
    "lora_r16_full": "r=16",
    "lora_r32_full": "r=32",
    "lora_r64_full": "r=64",
    "dora_r2": "dr=2",
    "dora_r4": "dr=4",
    "dora_r8": "dr=8",
    "dora_r16": "dr=16",
    "dora_r32": "dr=32",
}

# =============================================================================
# Semantic Feature Visual Encoding
# =============================================================================

SEMANTIC_COLORS = {
    "volume": "#EE6677",  # Red/Rose
    "location": "#4477AA",  # Blue
    "shape": "#228833",  # Green
}

SEMANTIC_LABELS = {
    "volume": r"Volume ($R^2_\mathrm{vol}$)",
    "location": r"Location ($R^2_\mathrm{loc}$)",
    "shape": r"Shape ($R^2_\mathrm{shape}$)",
}

SEMANTIC_LABELS_SHORT = {
    "volume": r"$R^2_\mathrm{vol}$",
    "location": r"$R^2_\mathrm{loc}$",
    "shape": r"$R^2_\mathrm{shape}$",
}

# =============================================================================
# Dice Score Visual Encoding (BraTS classes)
# =============================================================================

DICE_COLORS = {
    "NCR": "#CC6677",  # Rose - Necrotic Core
    "ED": "#DDCC77",  # Sand - Edema
    "ET": "#882255",  # Wine - Enhancing Tumor
    "mean": "#332288",  # Indigo - Mean Dice
}

DICE_LABELS = {
    "NCR": "NCR (Necrotic Core)",
    "ED": "ED (Edema)",
    "ET": "ET (Enhancing Tumor)",
    "mean": "Mean Dice",
}

# =============================================================================
# Domain/Dataset Visual Encoding (for latent space viz)
# =============================================================================

DOMAIN_COLORS = {
    "glioma": "#EE6677",  # Red/Rose - BraTS Glioma (source)
    "meningioma": "#4477AA",  # Blue - BraTS-MEN (target)
}

DOMAIN_MARKERS = {
    "glioma": "o",  # Circle
    "meningioma": "^",  # Triangle
}

DOMAIN_LABELS = {
    "glioma": "BraTS Glioma (Source)",
    "meningioma": "BraTS-MEN (Target)",
}

# =============================================================================
# IEEE Column Width Specifications
# =============================================================================

IEEE_COLUMN_WIDTH_INCHES = 3.39  # Single column (86 mm)
IEEE_COLUMN_GAP_INCHES = 0.24  # Gap between columns (6 mm)
IEEE_TEXT_WIDTH_INCHES = 7.0  # Full print area width (178 mm)
IEEE_TEXT_HEIGHT_INCHES = 9.0  # Full print area height (229 mm)

# =============================================================================
# Main Plot Settings Dictionary
# =============================================================================

PLOT_SETTINGS = {
    # Figure dimensions (IEEE compliant)
    "figure_width_single": IEEE_COLUMN_WIDTH_INCHES,  # 3.39 inches
    "figure_width_double": IEEE_TEXT_WIDTH_INCHES,  # 7.0 inches
    "figure_height_max": IEEE_TEXT_HEIGHT_INCHES,  # 9.0 inches (max)
    "figure_height_ratio": 0.75,  # Height = width * ratio (for plots)
    # Fonts (IEEE requires Times or similar serif)
    "font_family": "serif",
    "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext_fontset": "stix",  # STIX for math (matches Times)
    "text_usetex": False,  # Set True if LaTeX is installed
    # Font sizes (IEEE guidelines)
    "font_size": 10,
    "axes_labelsize": 11,
    "axes_titlesize": 12,
    "tick_labelsize": 9,
    "legend_fontsize": 9,
    "annotation_fontsize": 8,
    "panel_label_fontsize": 11,
    # Line properties
    "line_width": 1.2,
    "line_width_thick": 1.8,
    "marker_size": 5,
    "marker_edge_width": 0.5,
    # Error bars
    "errorbar_capsize": 2,
    "errorbar_capthick": 0.8,
    "errorbar_linewidth": 0.8,
    # Error bands (for confidence intervals)
    "error_band_alpha": 0.2,
    # Boxplot properties
    "boxplot_linewidth": 0.8,
    "boxplot_flier_size": 3,
    "boxplot_width": 0.6,
    # Bar plot properties
    "bar_width": 0.18,
    "bar_alpha": 0.85,
    # Grid
    "grid_alpha": 0.4,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    # Spines
    "spine_linewidth": 0.8,
    "spine_color": "0.2",
    # Ticks
    "tick_direction": "in",
    "tick_major_width": 0.8,
    "tick_minor_width": 0.5,
    "tick_major_length": 3.5,
    "tick_minor_length": 2.0,
    # Legend
    "legend_frameon": False,
    "legend_framealpha": 0.9,
    "legend_edgecolor": "0.8",
    "legend_borderpad": 0.4,
    "legend_columnspacing": 1.0,
    "legend_handletextpad": 0.5,
    # UMAP/t-SNE scatter
    "scatter_alpha": 0.6,
    "scatter_size": 15,
    "scatter_edgewidth": 0.3,
    # DPI for output
    "dpi_print": 300,
    "dpi_screen": 150,
    # Significance annotations
    "significance_bracket_linewidth": 0.8,
    "significance_text_fontsize": 9,
    "effect_size_fontsize": 8,
}


def apply_ieee_style() -> None:
    """Apply IEEE publication style using scienceplots if available.

    Falls back to manual style settings if scienceplots is not installed.
    Overrides default color cycle with Paul Tol colorblind-safe palette.
    """
    import matplotlib.pyplot as plt

    # Try to use scienceplots if available
    try:
        plt.style.use(["science", "ieee"])
        _scienceplots_available = True
    except OSError:
        _scienceplots_available = False
        _apply_fallback_ieee_style()

    # Override with condition colors and custom settings
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=list(CONDITION_COLORS.values())),
            # Ensure math rendering
            "mathtext.fontset": PLOT_SETTINGS["mathtext_fontset"],
            "font.family": PLOT_SETTINGS["font_family"],
            # Grid settings
            "axes.grid": True,
            "grid.alpha": PLOT_SETTINGS["grid_alpha"],
            "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
            "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
            # Tick settings
            "xtick.direction": PLOT_SETTINGS["tick_direction"],
            "ytick.direction": PLOT_SETTINGS["tick_direction"],
            "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
            "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
        }
    )


def _apply_fallback_ieee_style() -> None:
    """Apply IEEE-like style without scienceplots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # Fonts
            "font.family": PLOT_SETTINGS["font_family"],
            "font.serif": PLOT_SETTINGS["font_serif"],
            "font.size": PLOT_SETTINGS["font_size"],
            "mathtext.fontset": PLOT_SETTINGS["mathtext_fontset"],
            # Axes
            "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
            "axes.titlesize": PLOT_SETTINGS["axes_titlesize"],
            "axes.linewidth": PLOT_SETTINGS["spine_linewidth"],
            "axes.grid": True,
            # Ticks
            "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
            "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
            "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
            "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
            "xtick.minor.width": PLOT_SETTINGS["tick_minor_width"],
            "ytick.minor.width": PLOT_SETTINGS["tick_minor_width"],
            "xtick.direction": PLOT_SETTINGS["tick_direction"],
            "ytick.direction": PLOT_SETTINGS["tick_direction"],
            "xtick.major.size": PLOT_SETTINGS["tick_major_length"],
            "ytick.major.size": PLOT_SETTINGS["tick_major_length"],
            "xtick.minor.size": PLOT_SETTINGS["tick_minor_length"],
            "ytick.minor.size": PLOT_SETTINGS["tick_minor_length"],
            # Legend
            "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
            "legend.frameon": PLOT_SETTINGS["legend_frameon"],
            "legend.framealpha": PLOT_SETTINGS["legend_framealpha"],
            "legend.edgecolor": PLOT_SETTINGS["legend_edgecolor"],
            # Grid
            "grid.alpha": PLOT_SETTINGS["grid_alpha"],
            "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
            "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
            # Figure
            "figure.figsize": (
                PLOT_SETTINGS["figure_width_double"],
                PLOT_SETTINGS["figure_width_double"] * PLOT_SETTINGS["figure_height_ratio"],
            ),
            "figure.dpi": PLOT_SETTINGS["dpi_screen"],
            "savefig.dpi": PLOT_SETTINGS["dpi_print"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


# =============================================================================
# SDP Partition Visual Encoding
# =============================================================================

PARTITION_COLORS = {
    "vol": "#EE6677",  # Red/Rose (matches SEMANTIC_COLORS["volume"])
    "loc": "#4477AA",  # Blue (matches SEMANTIC_COLORS["location"])
    "shape": "#228833",  # Green (matches SEMANTIC_COLORS["shape"])
    "residual": "#BBBBBB",  # Grey
}

PARTITION_LABELS = {
    "vol": r"Volume ($z_\mathrm{vol}$)",
    "loc": r"Location ($z_\mathrm{loc}$)",
    "shape": r"Shape ($z_\mathrm{shape}$)",
    "residual": r"Residual ($z_\mathrm{res}$)",
}

# SDP Loss Term Colors (for training curves)
LOSS_TERM_COLORS = {
    "loss_total": "#332288",  # Indigo
    "mse_vol": "#EE6677",  # Red (matches volume)
    "mse_loc": "#4477AA",  # Blue (matches location)
    "mse_shape": "#228833",  # Green (matches shape)
    "loss_var": "#DDCC77",  # Sand
    "loss_cov": "#88CCEE",  # Cyan
    "loss_dcor": "#AA4499",  # Purple
}

# Curriculum Phase Markers
CURRICULUM_PHASES = {
    "warmup": {"start": 0, "end": 10, "color": "#BBBBBB", "label": "Warm-up"},
    "semantic": {"start": 10, "end": 40, "color": "#88CCEE", "label": "Semantic"},
    "independence": {"start": 40, "end": 60, "color": "#44AA99", "label": "Independence"},
    "full": {"start": 60, "end": None, "color": "#332288", "label": "Full"},
}


def get_significance_stars(p_val: float) -> str:
    """Convert p-value to significance stars.

    Args:
        p_val: P-value from statistical test.

    Returns:
        String with stars: "***" (p<0.001), "**" (p<0.01),
        "*" (p<0.05), or "n.s." (not significant).
    """
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "n.s."


def get_effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def get_figure_size(
    width: str = "single",
    height_ratio: float = None,
) -> tuple[float, float]:
    """Get figure size tuple for IEEE format.

    Args:
        width: "single" for column width, "double" for full width.
        height_ratio: Custom height/width ratio. If None, uses default.

    Returns:
        Tuple of (width, height) in inches.
    """
    if width == "single":
        w = PLOT_SETTINGS["figure_width_single"]
    elif width == "double":
        w = PLOT_SETTINGS["figure_width_double"]
    else:
        raise ValueError(f"Unknown width: {width}")

    ratio = height_ratio or PLOT_SETTINGS["figure_height_ratio"]
    return (w, w * ratio)


# =============================================================================
# V3 Condition Registries
# =============================================================================

V3_CONDITIONS: list[str] = [
    "baseline_frozen",
    "baseline",
    "lora_r4_full",
    "lora_r8_full",
    "lora_r16_full",
    "lora_r32_full",
    "lora_r64_full",
]

V3_SHAPE_LABELS: list[str] = [
    "Sphericity",
    "Enhancement Ratio",
    "Infiltration Index",
]

V3_FEATURE_TYPES: list[str] = ["volume", "location", "shape"]

V3_FEATURE_LABELS: dict[str, str] = {
    "volume": "Volume",
    "location": "Location",
    "shape": "Shape",
}

# =============================================================================
# Condition Ordering Lists
# =============================================================================

CONDITION_ORDER_V2: list[str] = [
    "baseline", "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32",
]

CONDITION_ORDER_V3: list[str] = list(V3_CONDITIONS)

CONDITION_ORDER_LORA: list[str] = [
    "baseline_frozen", "baseline",
    "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32",
]

CONDITION_ORDER_DORA: list[str] = [
    "baseline_frozen", "baseline",
    "dora_r2", "dora_r4", "dora_r8", "dora_r16", "dora_r32",
]

CONDITION_ORDER_ALL: list[str] = [
    "baseline_frozen", "baseline",
    "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32",
    "dora_r2", "dora_r4", "dora_r8", "dora_r16", "dora_r32",
]

RANKS: list[int] = [2, 4, 8, 16, 32]

# =============================================================================
# Probe / DCI / Adapter Colors
# =============================================================================

PROBE_COLORS: dict[str, str] = {
    "linear": "#0072B2",  # Blue
    "mlp": "#D55E00",  # Vermillion
}

DCI_COLORS: dict[str, str] = {
    "D": "#0072B2",  # Blue
    "C": "#009E73",  # Green
    "I": "#D55E00",  # Vermillion
}

ADAPTER_COLORS: dict[str, str] = {
    "lora": "#1f78b4",
    "dora": "#6a3d9a",
}

# =============================================================================
# Experiment Labels (for report directory naming)
# =============================================================================

EXPERIMENT_LABELS: dict[str, str] = {
    "lora_ablation_semantic_heads": "LoRA + Semantic",
    "lora_ablation_no_semantic_heads": "LoRA",
    "dora_ablation_semantic_heads": "DoRA + Semantic",
    "dora_ablation_no_semantic_heads": "DoRA",
}

# =============================================================================
# Utility Functions
# =============================================================================


def get_color(condition: str) -> str:
    """Get color for a condition, with fallback for unknown conditions.

    Args:
        condition: Condition name.

    Returns:
        Hex color string.
    """
    return CONDITION_COLORS.get(condition, "#808080")


def get_label(condition: str) -> str:
    """Get display label for a condition, with fallback.

    Args:
        condition: Condition name.

    Returns:
        Human-readable label.
    """
    return CONDITION_LABELS.get(condition, condition)


def short_label(condition: str) -> str:
    """Convert condition name to short plot label.

    Args:
        condition: Full condition name (e.g. 'lora_r8').

    Returns:
        Short label for axis ticks (e.g. 'r8').
    """
    return CONDITION_LABELS_SHORT.get(
        condition,
        condition.replace("baseline_frozen", "frozen")
        .replace("baseline", "base")
        .replace("lora_", "r")
        .replace("dora_", "dr"),
    )
