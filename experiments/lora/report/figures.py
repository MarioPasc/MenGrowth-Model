"""Figure generation for the LoRA ablation report.

Contains 8 figure-generation functions that produce publication-quality
PNG + PDF figures plus base64-encoded PNGs for HTML embedding.
"""

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.lora.report.data_loader import (
    ConditionData,
    ExperimentData,
)
from experiments.lora.report.style import (
    ADAPTER_COLORS,
    CONDITION_LABELS,
    PROBE_COLORS,
    RANKS,
    SEMANTIC_COLORS,
    apply_style,
    get_color,
    short_label,
)

logger = logging.getLogger(__name__)

# Apply publication style on import
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None

try:
    import seaborn as sns

    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    sns = None


# ─────────────────────────────────────────────────────────────────────
# Figure result container
# ─────────────────────────────────────────────────────────────────────


@dataclass
class FigureResult:
    """Result of a single figure generation."""

    name: str
    png_path: Path | None = None
    pdf_path: Path | None = None
    png_base64: str = ""
    caption: str = ""


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _save_and_encode(
    fig: "matplotlib.figure.Figure",
    name: str,
    output_dir: Path,
    caption: str = "",
) -> FigureResult:
    """Save figure as PNG + PDF and encode PNG to base64.

    Args:
        fig: Matplotlib figure.
        name: Figure filename stem.
        output_dir: Directory to save into.
        caption: Figure caption text.

    Returns:
        FigureResult with paths and base64 encoding.
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    png_path = figures_dir / f"{name}.png"
    pdf_path = figures_dir / f"{name}.pdf"

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    # Encode PNG to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    buf.close()

    plt.close(fig)

    return FigureResult(
        name=name,
        png_path=png_path,
        pdf_path=pdf_path,
        png_base64=b64,
        caption=caption,
    )


def _get_condition_list(exp: ExperimentData) -> list[str]:
    """Get ordered condition names from experiment."""
    return list(exp.conditions.keys())


def _get_rank_conditions(exp: ExperimentData) -> list[str]:
    """Get only the rank-based conditions (exclude baselines)."""
    prefix = "dora_r" if exp.adapter_type == "dora" else "lora_r"
    return [c for c in exp.conditions if c.startswith(prefix)]


def _rank_from_condition(condition: str) -> int:
    """Extract rank integer from condition name like 'lora_r8'."""
    parts = condition.split("_r")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return 0


# ─────────────────────────────────────────────────────────────────────
# Figure 1: MEN Dice by rank
# ─────────────────────────────────────────────────────────────────────


def fig_dice_men_by_rank(
    exp: ExperimentData,
    output_dir: Path,
) -> FigureResult | None:
    """Grouped bar chart of MEN Dice (mean/TC/WT/ET) across conditions.

    Args:
        exp: Loaded experiment data.
        output_dir: Output directory.

    Returns:
        FigureResult or None if no data.
    """
    if not HAS_MPL:
        return None

    conditions = _get_condition_list(exp)
    classes = [("Mean", "dice_mean"), ("TC", "dice_TC"), ("WT", "dice_WT"), ("ET", "dice_ET")]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))

    for ax, (cls_name, key) in zip(axes, classes):
        values = [exp.conditions[c].dice_men.get(key, 0) for c in conditions]
        x = np.arange(len(conditions))
        colors = [get_color(c) for c in conditions]

        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(
                    f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

        ax.set_ylabel("Dice Score")
        ax.set_title(cls_name, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(c) for c in conditions], rotation=45, ha="right")
        ax.set_ylim(0, 1)

    fig.suptitle("BraTS-MEN Segmentation Performance", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_and_encode(
        fig,
        "dice_men_by_rank",
        output_dir,
        "BraTS-MEN Dice scores (Mean, TC, WT, ET) across adaptation conditions.",
    )


# ─────────────────────────────────────────────────────────────────────
# Figure 2: Probe R²
# ─────────────────────────────────────────────────────────────────────


def fig_probe_r2(
    exp: ExperimentData,
    output_dir: Path,
) -> FigureResult | None:
    """Linear vs GP-RBF R² grouped bars for volume/location/shape.

    Args:
        exp: Loaded experiment data.
        output_dir: Output directory.

    Returns:
        FigureResult or None.
    """
    if not HAS_MPL:
        return None

    conditions = _get_condition_list(exp)
    conditions = [c for c in conditions if exp.conditions[c].metrics_enhanced]
    if not conditions:
        return None

    feature_types = ["volume", "location", "shape"]
    titles = ["Volume R²", "Location R²", "Shape R²"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, feat, title in zip(axes, feature_types, titles):
        linear_r2 = []
        rbf_r2 = []
        for c in conditions:
            m = exp.conditions[c].metrics_enhanced
            linear_r2.append(m.get(f"r2_{feat}_linear", m.get(f"r2_{feat}", 0)))
            rbf_r2.append(m.get(f"r2_{feat}_rbf", 0))

        x = np.arange(len(conditions))
        width = 0.35

        ax.bar(
            x - width / 2,
            linear_r2,
            width,
            label="Linear",
            color=PROBE_COLORS["linear"],
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            rbf_r2,
            width,
            label="GP-RBF",
            color=PROBE_COLORS["rbf"],
            alpha=0.8,
        )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("R²")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(c) for c in conditions], rotation=45, ha="right")
        ax.legend(fontsize=8)

    fig.suptitle("Semantic Feature Decodability", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_and_encode(
        fig,
        "probe_r2",
        output_dir,
        "Linear vs GP-RBF probe R² for volume, location, and shape predictions.",
    )


# ─────────────────────────────────────────────────────────────────────
# Figure 7: Nonlinearity Evidence
# ─────────────────────────────────────────────────────────────────────


def fig_nonlinearity_evidence(
    exp: ExperimentData,
    output_dir: Path,
) -> FigureResult | None:
    """GP-RBF − Linear R² per feature type.

    Args:
        exp: Loaded experiment data.
        output_dir: Output directory.

    Returns:
        FigureResult or None.
    """
    if not HAS_MPL:
        return None

    conditions = _get_condition_list(exp)
    conditions = [c for c in conditions if exp.conditions[c].metrics_enhanced]
    if not conditions:
        return None

    feature_types = ["volume", "location", "shape"]
    gaps: dict[str, list[float]] = {feat: [] for feat in feature_types}

    for c in conditions:
        m = exp.conditions[c].metrics_enhanced
        for feat in feature_types:
            linear = m.get(f"r2_{feat}_linear", m.get(f"r2_{feat}", 0))
            rbf = m.get(f"r2_{feat}_rbf", 0)
            gaps[feat].append(rbf - linear)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(conditions))
    width = 0.25
    feat_colors = ["steelblue", "seagreen", "coral"]

    for i, (feat, color) in enumerate(zip(feature_types, feat_colors)):
        ax.bar(x + i * width, gaps[feat], width, label=feat.capitalize(), color=color, alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Nonlinearity Evidence (GP-RBF R² − Linear R²)")
    ax.set_xlabel("Condition")
    ax.set_title("Nonlinearly Encoded Information", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([short_label(c) for c in conditions], rotation=45, ha="right")
    ax.legend(fontsize=9)

    fig.tight_layout()

    return _save_and_encode(
        fig,
        "nonlinearity_evidence",
        output_dir,
        "Nonlinearity Evidence: GP-RBF R² minus Linear R² per semantic feature type.",
    )


# ─────────────────────────────────────────────────────────────────────
# Figure 8: Training curves
# ─────────────────────────────────────────────────────────────────────


def fig_training_curves(
    exp: ExperimentData,
    output_dir: Path,
) -> FigureResult | None:
    """Validation Dice over epochs, overlaid per condition.

    Args:
        exp: Loaded experiment data.
        output_dir: Output directory.

    Returns:
        FigureResult or None.
    """
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    any_plotted = False
    for cond_name, cond in exp.conditions.items():
        if cond.training_log is None or cond_name == "baseline_frozen":
            continue
        log = cond.training_log
        if "val_dice_mean" not in log.columns:
            continue

        epochs = log["epoch"].values
        dice = log["val_dice_mean"].values
        color = get_color(cond_name)
        label = CONDITION_LABELS.get(cond_name, cond_name)
        ax.plot(epochs, dice, color=color, label=label, alpha=0.8, linewidth=1.5)

        # Mark best epoch
        best_epoch = cond.training_summary.get("best_epoch")
        if best_epoch is not None and best_epoch <= len(epochs):
            best_idx = best_epoch - 1
            if 0 <= best_idx < len(dice):
                ax.scatter([epochs[best_idx]], [dice[best_idx]], color=color, s=40, zorder=5)

        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice (Mean)")
    ax.set_title("Training Convergence", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return _save_and_encode(
        fig,
        "training_curves",
        output_dir,
        "Validation Dice over epochs. Dots mark best epoch per condition.",
    )


# ─────────────────────────────────────────────────────────────────────
# Figure 9: Statistical heatmap
# ─────────────────────────────────────────────────────────────────────


def fig_statistical_heatmap(
    exp: ExperimentData,
    output_dir: Path,
) -> FigureResult | None:
    """Corrected p-values heatmap from statistical comparisons.

    Args:
        exp: Loaded experiment data.
        output_dir: Output directory.

    Returns:
        FigureResult or None.
    """
    if not HAS_MPL or not HAS_SNS:
        return None

    stats = exp.statistical_comparisons
    if stats is None:
        logger.warning("No statistical comparisons for %s", exp.name)
        return None

    # Collect conditions (rows) and metrics (columns)
    cond_names = [c for c in stats if c != "baseline_frozen"]
    if not cond_names:
        return None

    # Gather all metric keys from first condition
    first_cond = cond_names[0]
    metric_keys = list(stats[first_cond].keys())

    # Filter to key metrics for readability
    priority_metrics = [
        "volume_neg_mse",
        "location_neg_mse",
        "shape_neg_mse",
        "dice_men_mean",
    ]
    metric_keys = [m for m in priority_metrics if m in metric_keys]
    if not metric_keys:
        metric_keys = list(stats[first_cond].keys())[:8]

    # Build matrix of corrected p-values
    data = np.full((len(cond_names), len(metric_keys)), np.nan)
    for i, cond in enumerate(cond_names):
        for j, metric in enumerate(metric_keys):
            entry = stats.get(cond, {}).get(metric, {})
            p = entry.get("p_corrected", entry.get("p_value", np.nan))
            if p is not None and not (isinstance(p, float) and np.isnan(p)):
                data[i, j] = p

    # Clean labels
    metric_labels = [m.replace("_neg_mse", " R²").replace("dice_men_", "MEN ") for m in metric_keys]
    cond_labels = [CONDITION_LABELS.get(c, c) for c in cond_names]

    fig, ax = plt.subplots(figsize=(max(8, len(metric_keys) * 1.5), max(4, len(cond_names) * 0.6)))

    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".3f",
        xticklabels=metric_labels,
        yticklabels=cond_labels,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Corrected p-value"},
    )

    ax.set_title("Statistical Significance (vs Frozen Baseline)", fontweight="bold")
    fig.tight_layout()

    return _save_and_encode(
        fig,
        "statistical_heatmap",
        output_dir,
        "Corrected p-values for each condition vs frozen baseline. "
        "Green = significant improvement (p < 0.05).",
    )


# ─────────────────────────────────────────────────────────────────────
# Figure 5: Rank summary (2-panel)
# ─────────────────────────────────────────────────────────────────────


def fig_rank_summary(
    exp: ExperimentData,
    output_dir: Path,
) -> FigureResult | None:
    """2-panel line plot: Dice/R² vs rank.

    Args:
        exp: Loaded experiment data.
        output_dir: Output directory.

    Returns:
        FigureResult or None.
    """
    if not HAS_MPL:
        return None

    rank_conds = _get_rank_conditions(exp)
    if not rank_conds:
        return None

    ranks = [_rank_from_condition(c) for c in rank_conds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: Dice (MEN only)
    men_dice = [exp.conditions[c].dice_men.get("dice_mean", 0) for c in rank_conds]
    ax1.plot(ranks, men_dice, "o-", color="#1f77b4", label="BraTS-MEN", linewidth=2)

    # Add frozen baseline reference
    if "baseline_frozen" in exp.conditions:
        frozen_men = exp.conditions["baseline_frozen"].dice_men.get("dice_mean", 0)
        ax1.axhline(y=frozen_men, color="gray", linestyle=":", alpha=0.4, label="Frozen")
        ax1.legend(fontsize=8)

    ax1.set_xlabel("LoRA Rank")
    ax1.set_ylabel("Dice Score")
    ax1.set_title("Segmentation vs Rank", fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ranks)
    ax1.set_xticklabels(ranks)

    # Panel 2: R² mean
    r2_linear = [exp.conditions[c].metrics_enhanced.get("r2_mean_linear", 0) for c in rank_conds]
    r2_rbf = [exp.conditions[c].metrics_enhanced.get("r2_mean_rbf", 0) for c in rank_conds]
    ax2.plot(ranks, r2_linear, "o-", color=PROBE_COLORS["linear"], label="Linear", linewidth=2)
    ax2.plot(ranks, r2_rbf, "s--", color=PROBE_COLORS["rbf"], label="GP-RBF", linewidth=2)

    if "baseline_frozen" in exp.conditions:
        frozen_r2 = exp.conditions["baseline_frozen"].metrics_enhanced.get("r2_mean_linear", 0)
        ax2.axhline(y=frozen_r2, color="gray", linestyle=":", alpha=0.4)

    ax2.set_xlabel("LoRA Rank")
    ax2.set_ylabel("R² (mean)")
    ax2.set_title("Feature Quality vs Rank", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ranks)
    ax2.set_xticklabels(ranks)

    fig.suptitle("Rank Selection Summary", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_and_encode(
        fig,
        "rank_summary",
        output_dir,
        "Summary of key metrics vs LoRA rank. Dashed lines show frozen baseline reference.",
    )


# ─────────────────────────────────────────────────────────────────────
# Figure 6: LoRA vs DoRA comparison
# ─────────────────────────────────────────────────────────────────────


def fig_lora_vs_dora(
    experiments: list[ExperimentData],
    output_dir: Path,
) -> FigureResult | None:
    """2-panel comparison: paired bars LoRA vs DoRA per rank.

    Args:
        experiments: List of experiments (should contain both LoRA and DoRA).
        output_dir: Output directory.

    Returns:
        FigureResult or None.
    """
    if not HAS_MPL:
        return None

    # Find LoRA and DoRA experiments (prefer semantic_heads)
    lora_exp = None
    dora_exp = None
    for exp in experiments:
        if exp.adapter_type == "lora" and lora_exp is None:
            lora_exp = exp
        elif exp.adapter_type == "dora" and dora_exp is None:
            dora_exp = exp

    if lora_exp is None or dora_exp is None:
        logger.info("Need both LoRA and DoRA experiments for comparison figure")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    width = 0.35

    for rank in RANKS:
        lora_cond = f"lora_r{rank}"
        dora_cond = f"dora_r{rank}"
        if lora_cond not in lora_exp.conditions or dora_cond not in dora_exp.conditions:
            continue

    x = np.arange(len(RANKS))
    rank_labels = [str(r) for r in RANKS]

    # Panel 1: MEN Dice
    lora_dice = [
        lora_exp.conditions.get(f"lora_r{r}", ConditionData(name="")).dice_men.get("dice_mean", 0)
        for r in RANKS
    ]
    dora_dice = [
        dora_exp.conditions.get(f"dora_r{r}", ConditionData(name="")).dice_men.get("dice_mean", 0)
        for r in RANKS
    ]

    ax1.bar(x - width / 2, lora_dice, width, label="LoRA", color=ADAPTER_COLORS["lora"], alpha=0.8)
    ax1.bar(x + width / 2, dora_dice, width, label="DoRA", color=ADAPTER_COLORS["dora"], alpha=0.8)
    ax1.set_ylabel("Dice (MEN)")
    ax1.set_title("Segmentation", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rank_labels)
    ax1.set_xlabel("Rank")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1)

    # Panel 2: R² mean
    lora_r2 = [
        lora_exp.conditions.get(f"lora_r{r}", ConditionData(name="")).metrics_enhanced.get(
            "r2_mean_linear", 0
        )
        for r in RANKS
    ]
    dora_r2 = [
        dora_exp.conditions.get(f"dora_r{r}", ConditionData(name="")).metrics_enhanced.get(
            "r2_mean_linear", 0
        )
        for r in RANKS
    ]

    ax2.bar(x - width / 2, lora_r2, width, label="LoRA", color=ADAPTER_COLORS["lora"], alpha=0.8)
    ax2.bar(x + width / 2, dora_r2, width, label="DoRA", color=ADAPTER_COLORS["dora"], alpha=0.8)
    ax2.set_ylabel("R² (mean linear)")
    ax2.set_title("Feature Quality", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(rank_labels)
    ax2.set_xlabel("Rank")
    ax2.legend(fontsize=8)

    fig.suptitle("LoRA vs DoRA Across Ranks", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_and_encode(
        fig,
        "lora_vs_dora",
        output_dir,
        "Direct comparison of LoRA vs DoRA at each rank for Dice and R².",
    )


# ─────────────────────────────────────────────────────────────────────
# Figure 12: Semantic head comparison
# ─────────────────────────────────────────────────────────────────────


def fig_semantic_comparison(
    experiments: list[ExperimentData],
    output_dir: Path,
) -> FigureResult | None:
    """2-panel Dice + R² comparison: semantic vs no-semantic.

    Args:
        experiments: List of experiments (should contain both variants).
        output_dir: Output directory.

    Returns:
        FigureResult or None.
    """
    if not HAS_MPL:
        return None

    # Find semantic and no-semantic experiments (same adapter type)
    sem_exp = None
    no_sem_exp = None
    for exp in experiments:
        if exp.semantic_heads and sem_exp is None:
            sem_exp = exp
        elif not exp.semantic_heads and no_sem_exp is None:
            no_sem_exp = exp

    if sem_exp is None or no_sem_exp is None:
        logger.info("Need both semantic and no-semantic experiments for comparison")
        return None

    prefix = "dora_r" if sem_exp.adapter_type == "dora" else "lora_r"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(RANKS))
    width = 0.35

    # Panel 1: MEN Dice
    sem_dice = [
        sem_exp.conditions.get(f"{prefix}{r}", ConditionData(name="")).dice_men.get("dice_mean", 0)
        for r in RANKS
    ]
    nosem_dice = [
        no_sem_exp.conditions.get(f"{prefix}{r}", ConditionData(name="")).dice_men.get(
            "dice_mean", 0
        )
        for r in RANKS
    ]

    ax1.bar(
        x - width / 2,
        sem_dice,
        width,
        label="+ Semantic",
        color=SEMANTIC_COLORS["semantic"],
        alpha=0.8,
    )
    ax1.bar(
        x + width / 2,
        nosem_dice,
        width,
        label="No Semantic",
        color=SEMANTIC_COLORS["no_semantic"],
        alpha=0.8,
    )
    ax1.set_ylabel("Dice (MEN)")
    ax1.set_title("Segmentation Performance", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(r) for r in RANKS])
    ax1.set_xlabel("Rank")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1)

    # Panel 2: R² mean
    sem_r2 = [
        sem_exp.conditions.get(f"{prefix}{r}", ConditionData(name="")).metrics_enhanced.get(
            "r2_mean_linear", 0
        )
        for r in RANKS
    ]
    nosem_r2 = [
        no_sem_exp.conditions.get(f"{prefix}{r}", ConditionData(name="")).metrics_enhanced.get(
            "r2_mean_linear", 0
        )
        for r in RANKS
    ]

    ax2.bar(
        x - width / 2,
        sem_r2,
        width,
        label="+ Semantic",
        color=SEMANTIC_COLORS["semantic"],
        alpha=0.8,
    )
    ax2.bar(
        x + width / 2,
        nosem_r2,
        width,
        label="No Semantic",
        color=SEMANTIC_COLORS["no_semantic"],
        alpha=0.8,
    )
    ax2.set_ylabel("R² (mean linear)")
    ax2.set_title("Feature Quality", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(r) for r in RANKS])
    ax2.set_xlabel("Rank")
    ax2.legend(fontsize=8)

    adapter_label = sem_exp.adapter_type.upper()
    fig.suptitle(
        f"Effect of Semantic Heads ({adapter_label})",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    return _save_and_encode(
        fig,
        "semantic_comparison",
        output_dir,
        f"Effect of semantic auxiliary heads on {adapter_label} adaptation: "
        "Dice and R² at each rank.",
    )


# ─────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────


def generate_all_figures(
    experiments: list[ExperimentData],
    results_dir: Path,
    output_dir: Path,
    compare_semantic: bool = False,
) -> list[FigureResult]:
    """Generate all report figures.

    Args:
        experiments: Loaded experiment data list.
        results_dir: Root results directory (for locating feature files).
        output_dir: Report output directory.
        compare_semantic: Generate semantic comparison figures.

    Returns:
        List of generated FigureResult objects.
    """
    if not HAS_MPL:
        logger.error("Matplotlib not available, cannot generate figures")
        return []

    figures: list[FigureResult] = []

    # Use first experiment as primary
    primary = experiments[0]
    logger.info("Generating figures for primary experiment: %s", primary.name)

    # Single-experiment figures
    generators = [
        ("MEN Dice by rank", lambda: fig_dice_men_by_rank(primary, output_dir)),
        ("Probe R²", lambda: fig_probe_r2(primary, output_dir)),
        ("Nonlinearity Evidence", lambda: fig_nonlinearity_evidence(primary, output_dir)),
        ("Training curves", lambda: fig_training_curves(primary, output_dir)),
        ("Statistical heatmap", lambda: fig_statistical_heatmap(primary, output_dir)),
        ("Rank summary", lambda: fig_rank_summary(primary, output_dir)),
    ]

    for desc, gen_fn in generators:
        logger.info("Generating: %s", desc)
        try:
            result = gen_fn()
            if result is not None:
                figures.append(result)
                logger.info("  -> %s", result.png_path)
            else:
                logger.warning("  -> skipped (no data)")
        except Exception:
            logger.exception("  -> FAILED: %s", desc)

    # Multi-experiment figures
    if len(experiments) >= 2:
        logger.info("Generating multi-experiment figures...")
        try:
            result = fig_lora_vs_dora(experiments, output_dir)
            if result is not None:
                figures.append(result)
        except Exception:
            logger.exception("Failed: LoRA vs DoRA")

    if compare_semantic and len(experiments) >= 2:
        logger.info("Generating semantic comparison figure...")
        try:
            result = fig_semantic_comparison(experiments, output_dir)
            if result is not None:
                figures.append(result)
        except Exception:
            logger.exception("Failed: Semantic comparison")

    logger.info("Generated %d figures total", len(figures))
    return figures
