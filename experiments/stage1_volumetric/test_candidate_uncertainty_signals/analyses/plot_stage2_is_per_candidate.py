"""Stage 2 figure — ΔIS@95 vs LME-homo per candidate, with bootstrap CIs.

Two-panel layout:
  (A) Forest plot of ΔIS (cell - homo) ± 95 % CI, ranked.
  (B) Stacked bars of mean width + mean miss per candidate, in cell order
      from panel A, exposing whether a candidate trades sharpness for
      coverage (the "same IS, different routes" trade-off).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from experiments.stage1_volumetric.engine.data import load_config

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 120,
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    out_root = Path(cfg["paths"]["output_dir"])
    rank_csv = out_root / "aggregated" / "candidate_ranking.csv"
    if not rank_csv.exists():
        logger.error("ranking CSV missing: %s — run aggregate_stage2 first", rank_csv)
        return 2
    df = pd.read_csv(rank_csv).copy()
    df = df.sort_values("delta_is", ascending=True).reset_index(drop=True)

    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.0, max(4.0, 0.32 * len(df) + 2.5)))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[1.10, 1.0],
        wspace=0.30,
        left=0.18,
        right=0.97,
        top=0.92,
        bottom=0.18,
        figure=fig,
    )

    # Panel A: ΔIS forest plot
    ax_a = fig.add_subplot(gs[0, 0])
    ypos = np.arange(len(df))
    color = ["#2a8c4a" if d < 0 else "#a93226" for d in df["delta_is"]]
    err_lo = df["delta_is"] - df["delta_is_lo"]
    err_hi = df["delta_is_hi"] - df["delta_is"]
    ax_a.errorbar(
        df["delta_is"],
        ypos,
        xerr=[err_lo, err_hi],
        fmt="none",
        ecolor="0.4",
        capsize=3,
        linewidth=1.0,
    )
    ax_a.scatter(df["delta_is"], ypos, c=color, s=42, zorder=5, edgecolors="black", linewidth=0.4)
    ax_a.axvline(0.0, color="0.6", linestyle="--", linewidth=0.8, zorder=2)
    if "p_delta_is_bh" in df.columns:
        for i, p in enumerate(df["p_delta_is_bh"].fillna(1.0)):
            if p < 0.05:
                ax_a.text(
                    df["delta_is_hi"].iloc[i] + 0.05,
                    i,
                    "*",
                    va="center",
                    ha="left",
                    fontsize=11,
                    color="black",
                )
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels(df["cell"], fontsize=8)
    ax_a.set_xlabel(r"$\Delta\,$IS@95 (cell $-$ LME-homo)")
    ax_a.set_title("(A) IS gain vs homo (BH-FDR * = significant)", fontsize=10)
    ax_a.grid(axis="x", alpha=0.25, linestyle=":")

    # Panel B: stacked sharpness + miss
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.barh(
        ypos,
        df["width_mean"],
        color="#1f6fb4",
        alpha=0.55,
        edgecolor="#1f6fb4",
        linewidth=0.6,
        label="width $U-L$",
    )
    ax_b.barh(
        ypos,
        df["miss_mean"],
        left=df["width_mean"],
        color="#1f6fb4",
        alpha=1.0,
        edgecolor="black",
        linewidth=0.4,
        hatch="///",
        label=r"miss $(2/\alpha)\,[(L-y)_+ + (y-U)_+]$",
    )
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([])
    ax_b.set_xlabel("score (lower is better)")
    ax_b.set_title("(B) IS decomposition (mean across patients)", fontsize=10)
    ax_b.grid(axis="x", alpha=0.25, linestyle=":")
    ax_b.legend(loc="lower right", frameon=False, fontsize=8)

    out_pdf = fig_dir / "stage2_is_per_candidate.pdf"
    out_png = fig_dir / "stage2_is_per_candidate.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s, %s", out_pdf, out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
