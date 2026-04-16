"""Four-panel domain divergence figure.

Panels:
    (a) Domain classifier accuracy (linear + MLP) per stage, bootstrap CIs.
    (b) FSD per stage.
    (c) DAD per stage (mean over blocks), if available.
    (d) Cross-stage CKA heatmap (5x5).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colours consistent with TSI panel: blue = default, orange = LoRA stage.
_BLUE = "#3b6ea5"
_ORANGE = "#d65a31"
_GREEN = "#2ca02c"


def render_domain_divergence_panel(
    metrics_csv: Path,
    cka_matrix_path: Path,
    dad_csv_path: Path | str | None,
    stages: tuple[int, ...],
    out_path: Path,
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 300,
    lora_stages: set[int] | None = None,
) -> None:
    """Render the 2x2 domain divergence panel figure.

    Args:
        metrics_csv: Path to ``domain_metrics.csv``.
        cka_matrix_path: Path to ``cka_cross_stage_men.npy``.
        dad_csv_path: Optional path to ``dad_per_head.csv`` for panel (c).
        stages: Stage indices to display.
        out_path: Output PDF/PNG path.
        figsize: Figure size in inches.
        dpi: Resolution.
        lora_stages: Stages with LoRA adaptation (highlighted in orange).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    cka_matrix = np.load(cka_matrix_path)

    if lora_stages is None:
        lora_stages = set()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # --- Panel (a): Domain classifier accuracy ---
    ax = axes[0, 0]
    stage_list = sorted(df["stage"].unique())
    x = np.arange(len(stage_list))
    width = 0.35

    linear_acc = [float(df.loc[df["stage"] == s, "domain_acc_linear"].values[0]) for s in stage_list]
    mlp_acc = [float(df.loc[df["stage"] == s, "domain_acc_mlp"].values[0]) for s in stage_list]
    ci_lo = [float(df.loc[df["stage"] == s, "domain_acc_ci_lower"].values[0]) for s in stage_list]
    ci_hi = [float(df.loc[df["stage"] == s, "domain_acc_ci_upper"].values[0]) for s in stage_list]

    linear_err_lo = [a - lo for a, lo in zip(linear_acc, ci_lo)]
    linear_err_hi = [hi - a for a, hi in zip(linear_acc, ci_hi)]

    colors_linear = [_ORANGE if s in lora_stages else _BLUE for s in stage_list]
    colors_mlp = [_ORANGE if s in lora_stages else _GREEN for s in stage_list]

    bars1 = ax.bar(x - width / 2, linear_acc, width, yerr=[linear_err_lo, linear_err_hi],
                   capsize=3, label="Linear", color=colors_linear, alpha=0.8)
    bars2 = ax.bar(x + width / 2, mlp_acc, width, label="MLP",
                   color=colors_mlp, alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("Encoder stage")
    ax.set_ylabel("Domain classifier accuracy")
    ax.set_title("(a) Domain Classifier Accuracy per Stage")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in stage_list])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel (b): FSD ---
    ax = axes[0, 1]
    fsd_vals = [float(df.loc[df["stage"] == s, "fsd"].values[0]) for s in stage_list]
    colors_fsd = [_ORANGE if s in lora_stages else _BLUE for s in stage_list]
    ax.bar(stage_list, fsd_vals, color=colors_fsd)
    ax.set_xlabel("Encoder stage")
    ax.set_ylabel("FSD (Cohen's d² avg)")
    ax.set_title("(b) Feature Statistics Divergence")
    ax.set_xticks(stage_list)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel (c): DAD or placeholder ---
    ax = axes[1, 0]
    if dad_csv_path is not None and Path(dad_csv_path).exists():
        dad_df = pd.read_csv(dad_csv_path)
        # Aggregate: mean DAD per stage (across blocks and heads)
        frozen_dad = dad_df[dad_df["condition"] == "frozen"]
        if not frozen_dad.empty:
            dad_per_stage = frozen_dad.groupby("stage")["dad"].mean()
            dad_stages = sorted(dad_per_stage.index)
            dad_vals = [float(dad_per_stage[s]) for s in dad_stages]
            colors_dad = [_ORANGE if s in lora_stages else _BLUE for s in dad_stages]
            ax.bar(dad_stages, dad_vals, color=colors_dad)
            ax.set_xlabel("Encoder stage")
            ax.set_ylabel("Mean DAD")
            ax.set_title("(c) Domain Attention Divergence (frozen)")
            ax.set_xticks(dad_stages)
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No frozen DAD data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title("(c) DAD — not available")
    else:
        ax.text(0.5, 0.5, "DAD CSV not provided", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("(c) DAD — not available")

    # --- Panel (d): CKA heatmap ---
    ax = axes[1, 1]
    n_stages = cka_matrix.shape[0]
    im = ax.imshow(cka_matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    for i in range(n_stages):
        for j in range(n_stages):
            val = cka_matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xticks(range(n_stages))
    ax.set_yticks(range(n_stages))
    ax.set_xticklabels([str(s) for s in sorted(stages)[:n_stages]])
    ax.set_yticklabels([str(s) for s in sorted(stages)[:n_stages]])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Stage")
    ax.set_title("(d) Cross-Stage CKA (MEN domain)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Domain divergence panel saved to %s", out_path)
