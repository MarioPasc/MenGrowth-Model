"""CKA adaptation drift grouped bar chart.

One group per stage (x-axis), one bar per adapted config (colour-coded).
y-axis = CKA (1.0 = no drift, lower = more drift). Non-adapted stages show
CKA ~ 1.0.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def render_cka_drift(
    drift_csv: Path,
    stages: tuple[int, ...],
    out_path: Path,
    figsize: tuple[float, float] = (10, 5),
    dpi: int = 300,
) -> None:
    """Render grouped bar chart of CKA adaptation drift.

    Args:
        drift_csv: Path to ``cka_drift.csv``.
        stages: Stage indices to display.
        out_path: Output PDF/PNG path.
        figsize: Figure size in inches.
        dpi: Resolution.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(drift_csv)
    config_names = list(df["config_name"].unique())
    n_configs = len(config_names)
    stage_list = sorted(stages)
    n_stages = len(stage_list)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_stages)
    total_width = 0.7
    bar_width = total_width / max(n_configs, 1)

    cmap = plt.cm.Set2
    colors = [cmap(i / max(n_configs - 1, 1)) for i in range(n_configs)]

    for i, name in enumerate(config_names):
        row = df[df["config_name"] == name].iloc[0]
        cka_values = [float(row.get(f"cka_stage_{s}", float("nan"))) for s in stage_list]
        offset = (i - n_configs / 2 + 0.5) * bar_width
        ax.bar(x + offset, cka_values, bar_width, label=name, color=colors[i], alpha=0.85)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="No drift")
    ax.set_xlabel("Encoder stage")
    ax.set_ylabel("CKA (frozen vs adapted)")
    ax.set_title("CKA Adaptation Drift per Stage")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in stage_list])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("CKA drift figure saved to %s", out_path)
