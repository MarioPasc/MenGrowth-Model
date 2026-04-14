"""DAD per-stage bar chart with permutation significance (spec §7.3)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _significance_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def render_dad_bar(
    per_head_df: pd.DataFrame,
    out_path: Path,
    title: str,
    figsize: tuple[float, float] = (6.0, 4.0),
    dpi: int = 300,
    condition: str | None = None,
) -> None:
    """Bar chart of mean DAD per stage; error bars = SD across heads/blocks.

    Stars above each bar indicate the minimum permutation-test p-value
    among the heads of that stage.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = per_head_df.copy()
    if condition is not None and "condition" in df.columns:
        df = df[df["condition"] == condition]
    if df.empty:
        logger.warning("Empty DAD dataframe for condition=%s — skipping plot", condition)
        return

    stages = sorted(df["stage"].unique())
    means = []
    err = []
    sig_markers = []
    for s in stages:
        sub = df[df["stage"] == s]
        vals = sub["dad"].to_numpy()
        means.append(float(np.mean(vals)))
        err.append(float(np.std(vals)))
        sig_markers.append(_significance_marker(float(sub["p_value"].min())))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(stages, means, yerr=err, capsize=4, color="#4a7c59")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    for s, bar, marker in zip(stages, bars, sig_markers):
        if marker:
            ax.text(
                s, bar.get_height() + bar.get_height() * 0.05, marker,
                ha="center", va="bottom", fontsize=11,
            )
    ax.set_xlabel("Encoder stage")
    ax.set_ylabel("DAD (symmetric KL, mean ± SD across heads)")
    ax.set_title(title)
    ax.set_xticks(stages)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
