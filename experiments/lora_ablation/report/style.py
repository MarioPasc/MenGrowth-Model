"""Shared style constants for the LoRA ablation report.

Thin re-export layer — all constants live in ``experiments.utils.settings``.
Only ``SEMANTIC_COLORS`` is defined locally (semantic-heads vs no-semantic-heads,
distinct from settings.py's volume/location/shape ``SEMANTIC_COLORS``).
"""

# Re-export everything that report/ consumers need
from experiments.utils.settings import (  # noqa: F401
    ADAPTER_COLORS,
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER_ALL,
    CONDITION_ORDER_DORA,
    CONDITION_ORDER_LORA,
    DOMAIN_COLORS,
    EXPERIMENT_LABELS,
    PROBE_COLORS,
    RANKS,
    get_color,
    get_label,
    short_label,
)

# Local: semantic-heads experiment variant colors (different meaning from
# settings.py's SEMANTIC_COLORS which maps volume/location/shape).
SEMANTIC_COLORS: dict[str, str] = {
    "semantic": "#33a02c",
    "no_semantic": "#e31a1c",
}


def apply_style() -> None:
    """Apply publication-quality matplotlib style for report figures."""
    from growth.evaluation.visualization import set_publication_style

    set_publication_style(figure_dpi=150, save_dpi=200)
