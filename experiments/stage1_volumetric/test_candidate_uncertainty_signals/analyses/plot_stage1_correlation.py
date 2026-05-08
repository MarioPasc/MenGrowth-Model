"""Stage 1 figure — Spearman ρ + 95 % BCa CI per candidate, ranked.

Single panel: forest plot of Spearman ρ between each candidate signal
and the homoscedastic LOPO residual, with 95 % bootstrap CI bars and the
ρ = 0 reference line. Linear-fit R² annotated next to each point.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    csv = out_root / "stage1_diagnostic" / "correlations.csv"
    if not csv.exists():
        logger.error("Stage 1 correlations CSV missing: %s", csv)
        return 2

    df = pd.read_csv(csv).copy()
    df = df.sort_values("spearman_rho").reset_index(drop=True)

    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.32 * len(df) + 1.5)))
    ypos = np.arange(len(df))
    color = ["#2a8c4a" if r > 0 else "#a93226" for r in df["spearman_rho"]]

    err_lo = df["spearman_rho"] - df["spearman_lo"]
    err_hi = df["spearman_hi"] - df["spearman_rho"]
    ax.errorbar(
        df["spearman_rho"],
        ypos,
        xerr=[err_lo, err_hi],
        fmt="none",
        ecolor="0.4",
        capsize=3,
        linewidth=1.0,
    )
    ax.scatter(df["spearman_rho"], ypos, c=color, s=42, zorder=5, edgecolors="black", linewidth=0.4)
    ax.axvline(0.0, color="0.6", linestyle="--", linewidth=0.8, zorder=2)

    for i, (_, row) in enumerate(df.iterrows()):
        x_label = max(row["spearman_hi"], row["spearman_rho"]) + 0.02
        ax.text(
            x_label,
            i,
            rf"$R^2_{{\mathrm{{lin}}}}={row['r2_linear']:.3f}$",
            va="center",
            ha="left",
            fontsize=8,
            color="0.3",
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(df["candidate"])
    ax.set_xlabel(
        r"Spearman $\rho$ between candidate $\sigma^2_v$ and $|y - \hat\mu^{\mathrm{homo}}|$"
    )
    n = int(df["n"].max() if not df["n"].isna().all() else 0)
    ax.set_title(
        f"Stage 1 — information content of segmentation-uncertainty signals (n={n})", fontsize=10
    )
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    xrng = max(0.6, float(df["spearman_hi"].abs().max() * 1.15) if not df.empty else 0.6)
    ax.set_xlim(-xrng, xrng + 0.2)
    fig.tight_layout()

    out_pdf = fig_dir / "stage1_correlation_panel.pdf"
    out_png = fig_dir / "stage1_correlation_panel.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s, %s", out_pdf, out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
