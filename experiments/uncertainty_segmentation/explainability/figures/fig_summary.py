"""Combined thesis summary figure (spec §7.4).

A single 1×3 panel: TSI, ASI, DAD per stage, with the LoRA target
stages highlighted as a coloured background band on each panel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _agg_per_stage(
    df: pd.DataFrame,
    value_col: str,
) -> tuple[list[int], list[float], list[float]]:
    stages = sorted(df["stage"].unique())
    means = []
    err = []
    for s in stages:
        vals = df.loc[df["stage"] == s, value_col].dropna().to_numpy()
        means.append(float(np.nanmean(vals)) if len(vals) else float("nan"))
        n = max(1, len(vals))
        err.append(1.96 * float(np.nanstd(vals)) / np.sqrt(n) if len(vals) else 0.0)
    return stages, means, err


def render_summary(
    tsi_df: pd.DataFrame,
    asi_df: pd.DataFrame | None,
    dad_df: pd.DataFrame | None,
    lora_stages: set[int],
    out_path: Path,
    figsize: tuple[float, float] = (14.0, 5.0),
    dpi: int = 300,
    condition: str = "frozen",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A — TSI
    sub = tsi_df[tsi_df["condition"] == condition]
    s_t, m_t, e_t = _agg_per_stage(sub, "mean_tsi")
    axes[0].bar(s_t, m_t, yerr=e_t, capsize=4, color="#3b6ea5")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("(a) Brain-masked TSI")
    axes[0].set_xlabel("Stage")
    axes[0].set_ylabel("TSI (mean ± 95% CI)")

    # Panel B — ASI
    if asi_df is not None and not asi_df.empty:
        sub = asi_df[asi_df["condition"] == condition]
        s_a, m_a, e_a = _agg_per_stage(sub, "asi_value")
        axes[1].bar(s_a, m_a, yerr=e_a, capsize=4, color="#3b6ea5")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("(b) Attention Selectivity Index")
    axes[1].set_xlabel("Stage")
    axes[1].set_ylabel("ASI (mean ± 95% CI)")

    # Panel C — DAD
    if dad_df is not None and not dad_df.empty:
        sub = dad_df[dad_df.get("condition", condition) == condition] \
            if "condition" in dad_df.columns else dad_df
        stages = sorted(sub["stage"].unique())
        means = [float(sub.loc[sub["stage"] == s, "dad"].mean()) for s in stages]
        std = [float(sub.loc[sub["stage"] == s, "dad"].std()) for s in stages]
        axes[2].bar(stages, means, yerr=std, capsize=4, color="#4a7c59")
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("(c) Domain Attention Divergence")
    axes[2].set_xlabel("Stage")
    axes[2].set_ylabel("DAD (mean ± SD)")

    # Highlight LoRA stages on each panel.
    for ax in axes:
        for s in lora_stages:
            ax.axvspan(s - 0.4, s + 0.4, color="#d65a31", alpha=0.15)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Encoder explainability — {condition}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
