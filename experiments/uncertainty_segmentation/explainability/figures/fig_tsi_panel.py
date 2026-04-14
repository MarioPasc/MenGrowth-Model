"""Brain-masked TSI summary figure (per spec §7.1).

The previous five-row panel from ``figure_tsi.py`` required full hidden
states + reference scans, which couples figure generation to the GPU
pipeline.  For the refactored module we ship a much simpler bar-chart
view that only needs the per-scan CSV — making
``run_figures_only.py`` truly GPU-free.

The detailed activation/heatmap panels remain available in the legacy
``figure_tsi.py`` (kept until that file is removed in §Final Cleanup).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def render_tsi_panel(
    per_scan_df: pd.DataFrame,
    out_path: Path,
    title: str,
    figsize: tuple[float, float] = (8.0, 4.5),
    dpi: int = 300,
    lora_stages: set[int] | None = None,
) -> None:
    """Bar chart of brain-masked mean TSI per stage with 95% CI."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stages = sorted(per_scan_df["stage"].unique())
    means = []
    err = []
    for s in stages:
        vals = per_scan_df.loc[per_scan_df["stage"] == s, "mean_tsi"].dropna().to_numpy()
        m = float(np.nanmean(vals)) if len(vals) else float("nan")
        sd = float(np.nanstd(vals)) if len(vals) else 0.0
        n = max(1, len(vals))
        means.append(m)
        err.append(1.96 * sd / np.sqrt(n))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(stages, means, yerr=err, capsize=4, color="#3b6ea5")
    if lora_stages:
        for s, bar in zip(stages, bars):
            if s in lora_stages:
                bar.set_color("#d65a31")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Encoder stage")
    ax.set_ylabel("Brain-masked TSI (mean ± 95% CI)")
    ax.set_title(title)
    ax.set_xticks(stages)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
