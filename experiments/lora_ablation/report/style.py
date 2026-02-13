"""Shared style constants for the LoRA ablation report.

Consolidates color palettes, condition ordering, labels, and
matplotlib configuration used across all figures.
"""

from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────
# Condition ordering and labels
# ─────────────────────────────────────────────────────────────────────

CONDITION_ORDER_LORA: List[str] = [
    "baseline_frozen", "baseline",
    "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32",
]

CONDITION_ORDER_DORA: List[str] = [
    "baseline_frozen", "baseline",
    "dora_r2", "dora_r4", "dora_r8", "dora_r16", "dora_r32",
]

CONDITION_ORDER_ALL: List[str] = [
    "baseline_frozen", "baseline",
    "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32",
    "dora_r2", "dora_r4", "dora_r8", "dora_r16", "dora_r32",
]

CONDITION_LABELS: Dict[str, str] = {
    "baseline_frozen": "Frozen",
    "baseline": "Baseline",
    "lora_r2": "LoRA r=2",
    "lora_r4": "LoRA r=4",
    "lora_r8": "LoRA r=8",
    "lora_r16": "LoRA r=16",
    "lora_r32": "LoRA r=32",
    "dora_r2": "DoRA r=2",
    "dora_r4": "DoRA r=4",
    "dora_r8": "DoRA r=8",
    "dora_r16": "DoRA r=16",
    "dora_r32": "DoRA r=32",
}

# ─────────────────────────────────────────────────────────────────────
# Color palettes
# ─────────────────────────────────────────────────────────────────────

CONDITION_COLORS: Dict[str, str] = {
    "baseline_frozen": "#a0a0a0",
    "baseline": "#808080",
    "lora_r2": "#a6cee3",
    "lora_r4": "#1f78b4",
    "lora_r8": "#33a02c",
    "lora_r16": "#ff7f00",
    "lora_r32": "#e31a1c",
    "dora_r2": "#b2df8a",
    "dora_r4": "#6a3d9a",
    "dora_r8": "#cab2d6",
    "dora_r16": "#fb9a99",
    "dora_r32": "#fdbf6f",
}

DOMAIN_COLORS: Dict[str, str] = {
    "meningioma": "#2166ac",
    "glioma": "#b2182b",
}

PROBE_COLORS: Dict[str, str] = {
    "linear": "steelblue",
    "mlp": "darkorange",
}

ADAPTER_COLORS: Dict[str, str] = {
    "lora": "#1f78b4",
    "dora": "#6a3d9a",
}

SEMANTIC_COLORS: Dict[str, str] = {
    "semantic": "#33a02c",
    "no_semantic": "#e31a1c",
}

# ─────────────────────────────────────────────────────────────────────
# Experiment directory naming
# ─────────────────────────────────────────────────────────────────────

EXPERIMENT_LABELS: Dict[str, str] = {
    "lora_ablation_semantic_heads": "LoRA + Semantic",
    "lora_ablation_no_semantic_heads": "LoRA",
    "dora_ablation_semantic_heads": "DoRA + Semantic",
    "dora_ablation_no_semantic_heads": "DoRA",
}

# Ranks used in the ablation (excluding baselines)
RANKS: List[int] = [2, 4, 8, 16, 32]


def short_label(condition: str) -> str:
    """Convert condition name to short plot label.

    Args:
        condition: Full condition name (e.g. 'lora_r8').

    Returns:
        Short label for axis ticks (e.g. 'r8').
    """
    return (
        condition.replace("baseline_frozen", "frozen")
        .replace("baseline", "base")
        .replace("lora_", "r")
        .replace("dora_", "dr")
    )


def get_color(condition: str) -> str:
    """Get color for a condition, with fallback.

    Args:
        condition: Condition name.

    Returns:
        Hex color string.
    """
    return CONDITION_COLORS.get(condition, "#808080")


def apply_style() -> None:
    """Apply publication-quality matplotlib style for report figures."""
    from growth.evaluation.visualization import set_publication_style

    set_publication_style(figure_dpi=150, save_dpi=200)
