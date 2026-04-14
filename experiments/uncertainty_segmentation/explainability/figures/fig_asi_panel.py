"""ASI per-stage figure (spec §7.2).

Box plot of per-window ASI distributions, one box per head per stage.
Reads the per-scan CSV produced by ``run_analysis.py`` whose schema is::

    scan_id, condition, stage, block, head, asi_value
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def render_asi_panel(
    per_scan_df: pd.DataFrame,
    out_path: Path,
    title: str,
    figsize: tuple[float, float] = (10.0, 5.0),
    dpi: int = 300,
) -> None:
    """1×4 panel of per-head ASI box plots, one panel per stage."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stages = sorted(per_scan_df["stage"].unique())
    fig, axes = plt.subplots(1, max(1, len(stages)), figsize=figsize, sharey=True)
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        sub = per_scan_df[per_scan_df["stage"] == stage]
        heads = sorted(sub["head"].unique())
        data = [
            sub.loc[sub["head"] == h, "asi_value"].dropna().to_numpy() for h in heads
        ]
        bp = ax.boxplot(
            data, positions=heads, widths=0.6, patch_artist=True,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#3b6ea5")
            patch.set_alpha(0.6)
        # Highlight heads whose median ASI > 1.5.
        medians = [float(np.nanmedian(d)) if len(d) else float("nan") for d in data]
        for h, m, patch in zip(heads, medians, bp["boxes"]):
            if np.isfinite(m) and m > 1.5:
                patch.set_facecolor("#d65a31")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Stage {stage}")
        ax.set_xlabel("Head")
        ax.set_xticks(heads)
    axes[0].set_ylabel("ASI (per boundary window)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
