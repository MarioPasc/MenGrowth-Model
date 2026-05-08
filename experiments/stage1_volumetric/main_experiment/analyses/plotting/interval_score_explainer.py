"""Interval Score explainer figure for the BSc thesis methodology section.

The figure illustrates the Interval Score (Gneiting & Raftery, 2007;
*Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 102:359),
the headline calibration metric used to compare LME homoscedastic and
LME heteroscedastic predictive intervals on the meningioma volume
trajectories.

For nominal level :math:`(1-\\alpha)`, prediction interval :math:`[L, U]`
and observation :math:`y`,

.. math::
   \\mathrm{IS}_\\alpha(L, U;\\, y) = (U - L)
       + \\frac{2}{\\alpha} (L - y)\\,\\mathbf{1}\\{y < L\\}
       + \\frac{2}{\\alpha} (y - U)\\,\\mathbf{1}\\{y > U\\}.

Lower IS is better. The score simultaneously rewards sharpness
(small width :math:`U-L`) and coverage (no miss penalty), which makes it
a strictly proper scoring rule for central prediction intervals.

Layout
------
Top row, three panels — synthetic patient with three past
:math:`\\log(V+1)` observations and a predictive interval at the next
follow-up :math:`t^\\ast`. Cases:

1. sharp & calibrated   — narrow interval, observation inside (low IS).
2. wide  & calibrated   — interval covers but is uninformative
   (medium IS, dominated by width).
3. sharp & over-confident — narrow interval, observation outside
   (high IS, dominated by miss penalty).

Bottom row — horizontal stacked-bar decomposition of IS into width
and miss components for the three cases, with the IS value annotated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Display
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "figure.dpi": 120,
    }
)

ALPHA = 0.05
Z = float(norm.ppf(1.0 - ALPHA / 2.0))  # 1.95996...

# Synthetic patient (illustrative, not real data) ---------------------------
T_PAST = np.array([0.0, 1.0, 2.0])
Y_PAST = np.array([6.40, 6.78, 7.15])
T_FUT = 3.0
Y_TRUE = 7.50  # held-out observation at t*

# Three forecasts at t*: (label, mu_hat, sigma, color) ----------------------
CASES = [
    (r"sharp \& calibrated", 7.45, 0.18, "#2a8c4a"),
    (r"wide \& calibrated", 7.45, 0.65, "#d68a1f"),
    (r"sharp \& over-confident", 8.30, 0.22, "#c0392b"),
]


def interval_score(mu: float, sigma: float, y: float, alpha: float = ALPHA) -> dict:
    """Compute IS_alpha for a Gaussian predictive distribution N(mu, sigma^2)."""
    L = mu - Z * sigma
    U = mu + Z * sigma
    width = U - L
    miss_low = (2.0 / alpha) * max(0.0, L - y)
    miss_high = (2.0 / alpha) * max(0.0, y - U)
    return {
        "L": L,
        "U": U,
        "width": width,
        "miss_low": miss_low,
        "miss_high": miss_high,
        "miss": miss_low + miss_high,
        "IS": width + miss_low + miss_high,
    }


def _draw_panel(
    ax: plt.Axes,
    mu: float,
    sigma: float,
    color: str,
    title: str,
    res: dict,
    y_lo: float,
    y_hi: float,
) -> None:
    """One predictive-interval panel."""
    # Past trajectory
    ax.plot(
        T_PAST,
        Y_PAST,
        "o-",
        color="0.25",
        markersize=5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        linewidth=1.0,
        zorder=3,
    )
    # Dashed link from last observation to predictive mean
    ax.plot(
        [T_PAST[-1], T_FUT],
        [Y_PAST[-1], mu],
        "--",
        color=color,
        linewidth=1.0,
        alpha=0.85,
        zorder=2,
    )

    # Predictive density (sideways) — visualises sharp vs wide
    y_grid = np.linspace(y_lo, y_hi, 400)
    pdf = norm.pdf(y_grid, loc=mu, scale=sigma)
    pdf_norm = pdf / pdf.max() * 0.42  # half-width in time units
    x_left = T_FUT - pdf_norm
    x_right = T_FUT + pdf_norm
    in_ci = (y_grid >= res["L"]) & (y_grid <= res["U"])

    # Light fill outside [L,U], stronger inside
    ax.fill_betweenx(y_grid, x_left, x_right, color=color, alpha=0.08, zorder=1)
    ax.fill_betweenx(
        y_grid[in_ci], x_left[in_ci], x_right[in_ci], color=color, alpha=0.30, zorder=2
    )
    ax.plot(x_right, y_grid, color=color, linewidth=1.0, alpha=0.85, zorder=3)
    ax.plot(x_left, y_grid, color=color, linewidth=1.0, alpha=0.85, zorder=3)

    # Interval bounds (horizontal ticks at L and U)
    ax.hlines(
        [res["L"], res["U"]],
        xmin=T_FUT - 0.50,
        xmax=T_FUT + 0.50,
        colors=color,
        linestyles=":",
        linewidth=0.9,
        zorder=4,
    )
    ax.text(
        T_FUT + 0.55,
        res["U"],
        r"$U$",
        color=color,
        fontsize=9,
        va="center",
        ha="left",
    )
    ax.text(
        T_FUT + 0.55,
        res["L"],
        r"$L$",
        color=color,
        fontsize=9,
        va="center",
        ha="left",
    )

    # Predictive mean
    ax.plot(
        T_FUT,
        mu,
        "o",
        color=color,
        markersize=5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        zorder=5,
    )

    # True held-out observation
    ax.plot(
        T_FUT,
        Y_TRUE,
        marker="*",
        color="#b30000",
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=6,
    )

    # Miss arrow (only when y* outside [L,U])
    if Y_TRUE < res["L"] or Y_TRUE > res["U"]:
        y_a, y_b = (Y_TRUE, res["L"]) if Y_TRUE < res["L"] else (res["U"], Y_TRUE)
        x_arrow = T_FUT + 0.85
        ax.annotate(
            "",
            xy=(x_arrow, y_a),
            xytext=(x_arrow, y_b),
            arrowprops={"arrowstyle": "<->", "color": "#b30000", "lw": 1.2},
            zorder=6,
        )
        ax.text(
            x_arrow + 0.08,
            0.5 * (y_a + y_b),
            r"$|y - L|$" if Y_TRUE < res["L"] else r"$|y - U|$",
            color="#b30000",
            fontsize=9,
            va="center",
            ha="left",
        )

    ax.set_xlim(-0.4, T_FUT + 1.35)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_xlabel("follow-up $t$ (yr)")
    ax.set_title(f"{title}\nIS = {res['IS']:.2f}", fontsize=10)


def _draw_decomposition(
    ax: plt.Axes, results: list[dict], colors: list[str], labels: list[str]
) -> None:
    """Stacked horizontal bar chart: IS = width + miss."""
    widths = np.array([r["width"] for r in results])
    misses = np.array([r["miss"] for r in results])
    iss = widths + misses
    y_pos = np.arange(len(results))[::-1]
    bw = 0.55

    ax.barh(
        y_pos,
        widths,
        height=bw,
        color=colors,
        alpha=0.45,
        edgecolor=colors,
        linewidth=0.8,
        label=r"width $\;U-L$",
    )
    ax.barh(
        y_pos,
        misses,
        height=bw,
        left=widths,
        color=colors,
        alpha=1.0,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        label=r"miss penalty $\;\frac{2}{\alpha}(L-y)_+ + \frac{2}{\alpha}(y-U)_+$",
    )

    x_pad = max(iss) * 0.02
    for yi, total in zip(y_pos, iss):
        ax.text(total + x_pad, yi, f"IS = {total:.2f}", va="center", ha="left", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("score (lower is better)")
    ax.set_xlim(0.0, max(iss) * 1.20)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", frameon=False, ncol=1, fontsize=9)


def main(out_pdf: Path, out_png: Path) -> None:
    # Compute IS decomposition for each case
    results = [interval_score(mu, sigma, Y_TRUE) for _, mu, sigma, _ in CASES]
    for (label, mu, sigma, _), res in zip(CASES, results):
        logger.info(
            "%-26s  mu=%.2f  sigma=%.2f  [L,U]=[%.2f, %.2f]  width=%.2f  miss=%.2f  IS=%.2f",
            label,
            mu,
            sigma,
            res["L"],
            res["U"],
            res["width"],
            res["miss"],
            res["IS"],
        )

    # Shared y-axis range covers all 95% intervals + true y
    y_lo = float(min(min(Y_PAST), min(r["L"] for r in results), Y_TRUE) - 0.4)
    y_hi = float(max(max(Y_PAST), max(r["U"] for r in results), Y_TRUE) + 0.4)

    fig = plt.figure(figsize=(11.0, 4.2))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=3,
        wspace=0.22,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.20,
        figure=fig,
    )

    # 3 trajectory panels
    axes_top = []
    for i, ((label, mu, sigma, color), res) in enumerate(zip(CASES, results)):
        ax = fig.add_subplot(gs[0, i])
        axes_top.append(ax)
        _draw_panel(ax, mu, sigma, color, label, res, y_lo, y_hi)
        if i == 0:
            ax.set_ylabel(r"$\log(V_{\mathrm{MEN}}+1)$")

    # Bottom legend explaining the red star (held-out observation)
    star_handle = plt.Line2D(
        [],
        [],
        linestyle="none",
        marker="*",
        markersize=14,
        color="#b30000",
        markeredgecolor="black",
        markeredgewidth=0.5,
        label=r"held-out observation $y$ at $t^\ast$",
    )
    past_handle = plt.Line2D(
        [],
        [],
        linestyle="-",
        marker="o",
        markersize=5,
        color="0.25",
        markeredgecolor="black",
        markeredgewidth=0.4,
        label="past observations",
    )
    fig.legend(
        handles=[past_handle, star_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", out_pdf)
    logger.info("Wrote %s", out_png)
    plt.close(fig)


if __name__ == "__main__":
    THESIS_FIG_DIR = Path(
        "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/"
        "bachelor_thesis/68596a200c0e0e3876880afa/figures/methodology"
    )
    main(
        out_pdf=THESIS_FIG_DIR / "interval_score_explainer.pdf",
        out_png=THESIS_FIG_DIR / "interval_score_explainer.png",
    )
