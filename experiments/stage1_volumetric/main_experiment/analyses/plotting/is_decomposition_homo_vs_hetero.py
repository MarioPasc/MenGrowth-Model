"""Per-fold IS decomposition: LME homoscedastic vs heteroscedastic.

For each held-out LOPO fold and each model, the Interval Score at
nominal level :math:`(1-\\alpha) = 0.95` decomposes additively as

.. math::
   \\mathrm{IS}_\\alpha = \\underbrace{(U - L)}_{\\text{width}}
       \\;+\\; \\underbrace{\\frac{2}{\\alpha}\\bigl[(L-y)_+ + (y-U)_+\\bigr]}_{\\text{miss penalty}}.

The figure shows the "same IS, different routes" trade-off behind the
marginally-null ΔIS@95 between LME-homo and LME-hetero on the τ=0
(empirical σ²_v) main-experiment runs:

* Panel A — per-patient scatter in (width, miss) space, with iso-IS
  contours of slope -1. Each patient appears once per model
  (○ homo, ▲ hetero), coloured by the tertile of the held-out scan's
  σ²_v. A grey arrow connects the homo→hetero pair, exposing how
  hetero re-allocates mass between width and miss along iso-IS lines.

* Panel B — tertile-stratified mean (width, miss) as stacked bars,
  side-by-side per model. The total height is the mean IS for that
  (tertile, model) cell. The textbook signature of the trade-off is
  hetero having larger width and smaller miss in the high-σ²_v tertile,
  homo the opposite, with marginal averages cancelling.

Inputs
------
* LME homo:  ``LME_baseline/lopo_results.json``
* LME hetero (τ=0): mean over 20 seeds of
  ``runs/empirical_shift_tau_+0.000/seed_*/lopo_results.json``

Outputs
-------
``main_experiment/figures/is_decomposition_homo_vs_hetero.{pdf,png}``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths ---------------------------------------------------------------------
RES_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction/main_experiment"
)
HOMO_PATH = RES_ROOT / "LME_baseline" / "lopo_results.json"
HETERO_RUNS_DIR = RES_ROOT / "runs" / "empirical_shift_tau_+0.000"
OUT_DIR = RES_ROOT / "figures"

ALPHA = 0.05
SCALE_MISS = 2.0 / ALPHA  # 40

# Style ---------------------------------------------------------------------
COLOR_HOMO = "#1f6fb4"
COLOR_HETERO = "#c0392b"
TERTILE_COLORS = ["#2a8c4a", "#7f7f7f", "#a93226"]  # low / mid / high σ²_v
TERTILE_LABELS = [r"low $\sigma^2_v$", r"mid $\sigma^2_v$", r"high $\sigma^2_v$"]

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
        "lines.linewidth": 1.2,
        "figure.dpi": 120,
    }
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _flatten_lopo(json_path: Path) -> dict[str, dict]:
    """Return {patient_id: prediction_dict} using the last_from_rest split.

    Each patient has one held-out LOPO observation in the τ=0 main-experiment
    run; if multiple are present we keep the last one.
    """
    with open(json_path) as f:
        data = json.load(f)
    out: dict[str, dict] = {}
    for fold in data["fold_results"]:
        pid = fold["patient_id"]
        preds = fold["predictions"].get("last_from_rest", [])
        if not preds:
            continue
        # Use last prediction (= furthest held-out scan)
        pred = preds[-1]
        out[pid] = {
            "mu": float(pred["pred_mean"]),
            "y": float(pred["actual"]),
            "L": float(pred["lower_95"]),
            "U": float(pred["upper_95"]),
            "sigma_v_sq": float(pred.get("sigma_v_sq_target", 0.0)),
            "time": float(pred["time"]),
        }
    return out


def load_homo() -> dict[str, dict]:
    return _flatten_lopo(HOMO_PATH)


def load_hetero_avg_over_seeds() -> tuple[dict[str, dict], int]:
    """Average L, U, mu, σ²_v across all τ=0 seeds, per patient."""
    seed_files = sorted(HETERO_RUNS_DIR.glob("seed_*/lopo_results.json"))
    if not seed_files:
        raise FileNotFoundError(f"No seed dirs under {HETERO_RUNS_DIR}")
    accum: dict[str, dict[str, list[float]]] = {}
    for sf in seed_files:
        for pid, rec in _flatten_lopo(sf).items():
            d = accum.setdefault(pid, {k: [] for k in ("mu", "y", "L", "U", "sigma_v_sq", "time")})
            for k, v in rec.items():
                d[k].append(v)
    out: dict[str, dict] = {}
    for pid, d in accum.items():
        out[pid] = {k: float(np.mean(v)) for k, v in d.items()}
    return out, len(seed_files)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def is_components(rec: dict) -> tuple[float, float, float]:
    """Return (width, miss, IS) under α=0.05."""
    width = rec["U"] - rec["L"]
    miss_lo = max(0.0, rec["L"] - rec["y"])
    miss_hi = max(0.0, rec["y"] - rec["U"])
    miss = SCALE_MISS * (miss_lo + miss_hi)
    return width, miss, width + miss


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_iso_is_contours(ax: plt.Axes, levels: list[float], xmax: float, ymax: float) -> None:
    for IS_lev in levels:
        xs = np.linspace(0.0, IS_lev, 200)
        ys = IS_lev - xs
        ax.plot(xs, ys, ":", color="0.65", linewidth=0.7, alpha=0.8, zorder=1)
        # Label near upper edge
        if IS_lev <= xmax:
            ax.text(
                IS_lev * 0.96,
                0.04 * ymax,
                f"IS={IS_lev:g}",
                color="0.45",
                fontsize=7.5,
                ha="right",
                va="bottom",
                rotation=0,
                zorder=2,
            )
        else:
            x_at_top = max(0.0, IS_lev - ymax * 0.95)
            y_at_top = IS_lev - x_at_top
            ax.text(
                x_at_top + xmax * 0.005,
                y_at_top,
                f"IS={IS_lev:g}",
                color="0.45",
                fontsize=7.5,
                ha="left",
                va="bottom",
                rotation=-45,
                zorder=2,
            )


def plot_panel_A(
    ax: plt.Axes,
    homo_W: np.ndarray,
    homo_M: np.ndarray,
    het_W: np.ndarray,
    het_M: np.ndarray,
    tertile: np.ndarray,
) -> None:
    xmax = max(homo_W.max(), het_W.max()) * 1.15
    ymax = max(homo_M.max(), het_M.max()) * 1.10
    if ymax <= 0.0:
        ymax = 1.0

    # Iso-IS contours
    is_levels = [1.0, 2.5, 5.0, 10.0, 25.0, 50.0]
    is_levels = [lv for lv in is_levels if lv <= xmax + ymax]
    _draw_iso_is_contours(ax, is_levels, xmax, ymax)

    # Per-patient connectors
    for i in range(len(homo_W)):
        ax.plot(
            [homo_W[i], het_W[i]],
            [homo_M[i], het_M[i]],
            "-",
            color="0.55",
            linewidth=0.4,
            alpha=0.40,
            zorder=2,
        )

    # Markers, coloured by tertile
    for t in range(3):
        sel = tertile == t
        ax.scatter(
            homo_W[sel],
            homo_M[sel],
            marker="o",
            facecolors=TERTILE_COLORS[t],
            edgecolors="black",
            linewidths=0.4,
            s=32,
            alpha=0.85,
            zorder=4,
        )
        ax.scatter(
            het_W[sel],
            het_M[sel],
            marker="^",
            facecolors=TERTILE_COLORS[t],
            edgecolors="black",
            linewidths=0.4,
            s=34,
            alpha=0.85,
            zorder=4,
        )

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"sharpness  $U - L$  [log-volume units]")
    ax.set_ylabel(r"miss penalty  $(2/\alpha)\,[(L-y)_+ + (y-U)_+]$")
    ax.set_title("(A) per-patient IS decomposition", fontsize=10)
    ax.grid(alpha=0.20, linestyle=":")


def plot_panel_B(
    ax: plt.Axes,
    homo_W: np.ndarray,
    homo_M: np.ndarray,
    het_W: np.ndarray,
    het_M: np.ndarray,
    tertile: np.ndarray,
) -> None:
    n_t = 3
    # Tertile means
    hW = np.array([homo_W[tertile == t].mean() for t in range(n_t)])
    hM = np.array([homo_M[tertile == t].mean() for t in range(n_t)])
    eW = np.array([het_W[tertile == t].mean() for t in range(n_t)])
    eM = np.array([het_M[tertile == t].mean() for t in range(n_t)])

    bw = 0.36
    x = np.arange(n_t)
    x_homo = x - bw / 2 - 0.01
    x_het = x + bw / 2 + 0.01

    # Stacked bars: width (light fill) + miss (hatched)
    ax.bar(
        x_homo,
        hW,
        bw,
        color=COLOR_HOMO,
        alpha=0.45,
        edgecolor=COLOR_HOMO,
        linewidth=0.6,
        label="width",
    )
    ax.bar(
        x_homo,
        hM,
        bw,
        bottom=hW,
        color=COLOR_HOMO,
        alpha=1.0,
        edgecolor="black",
        linewidth=0.4,
        hatch="///",
        label="miss penalty",
    )
    ax.bar(x_het, eW, bw, color=COLOR_HETERO, alpha=0.45, edgecolor=COLOR_HETERO, linewidth=0.6)
    ax.bar(
        x_het,
        eM,
        bw,
        bottom=eW,
        color=COLOR_HETERO,
        alpha=1.0,
        edgecolor="black",
        linewidth=0.4,
        hatch="///",
    )

    # IS labels above bars
    ymax_bar = float(max(hW + hM).max() if (hW + hM).size else 1.0)
    ymax_bar = max(ymax_bar, float((eW + eM).max()))
    pad = 0.04 * ymax_bar
    for i in range(n_t):
        ax.text(
            x_homo[i],
            hW[i] + hM[i] + pad,
            f"{hW[i] + hM[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLOR_HOMO,
        )
        ax.text(
            x_het[i],
            eW[i] + eM[i] + pad,
            f"{eW[i] + eM[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLOR_HETERO,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TERTILE_LABELS)
    ax.set_ylabel("score (lower is better)")
    ax.set_title("(B) tertile-conditional mean IS = width + miss", fontsize=10)
    ax.set_ylim(0, ymax_bar * 1.18)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    # Model legend
    homo_patch = plt.Rectangle((0, 0), 1, 1, color=COLOR_HOMO, alpha=0.55, label="LME homo")
    het_patch = plt.Rectangle((0, 0), 1, 1, color=COLOR_HETERO, alpha=0.55, label="LME hetero")
    width_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=0.4, label="width"
    )
    miss_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=0.4, hatch="///", label="miss"
    )
    ax.legend(
        handles=[homo_patch, het_patch, width_patch, miss_patch],
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncol=2,
    )


def make_figure(
    homo_recs: dict[str, dict],
    het_recs: dict[str, dict],
    n_seeds: int,
) -> plt.Figure:
    common = sorted(set(homo_recs) & set(het_recs))
    logger.info(
        "Common patients: %d (homo=%d, hetero=%d, n_seeds=%d)",
        len(common),
        len(homo_recs),
        len(het_recs),
        n_seeds,
    )

    # Components
    homo_W = np.array([is_components(homo_recs[p])[0] for p in common])
    homo_M = np.array([is_components(homo_recs[p])[1] for p in common])
    het_W = np.array([is_components(het_recs[p])[0] for p in common])
    het_M = np.array([is_components(het_recs[p])[1] for p in common])

    # Tertile by held-out σ²_v from hetero (homo placeholder ≈ 0)
    sv = np.array([het_recs[p]["sigma_v_sq"] for p in common])
    q1, q2 = np.quantile(sv, [1 / 3, 2 / 3])
    tertile = np.full(len(common), 1, dtype=int)
    tertile[sv <= q1] = 0
    tertile[sv > q2] = 2

    # Marginal IS for log
    homo_IS = homo_W + homo_M
    het_IS = het_W + het_M
    logger.info(
        "Marginal mean IS — homo=%.3f  hetero=%.3f  Δ=%.3f",
        homo_IS.mean(),
        het_IS.mean(),
        het_IS.mean() - homo_IS.mean(),
    )
    for t, name in enumerate(["low", "mid", "high"]):
        sel = tertile == t
        if sel.sum() == 0:
            continue
        logger.info(
            "%-4s tertile (n=%d, mean σ²_v=%.4f):  homo IS=%.2f  (W=%.2f, M=%.2f)  "
            "hetero IS=%.2f  (W=%.2f, M=%.2f)",
            name,
            sel.sum(),
            sv[sel].mean(),
            homo_IS[sel].mean(),
            homo_W[sel].mean(),
            homo_M[sel].mean(),
            het_IS[sel].mean(),
            het_W[sel].mean(),
            het_M[sel].mean(),
        )

    # Layout
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[1.25, 1.0],
        wspace=0.25,
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.20,
        figure=fig,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    plot_panel_A(ax_A, homo_W, homo_M, het_W, het_M, tertile)
    plot_panel_B(ax_B, homo_W, homo_M, het_W, het_M, tertile)

    # Bottom legend (model marker + tertile colour)
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="LME homo",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="^",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="LME hetero",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=8,
            markerfacecolor=TERTILE_COLORS[0],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=TERTILE_LABELS[0],
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=8,
            markerfacecolor=TERTILE_COLORS[1],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=TERTILE_LABELS[1],
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=8,
            markerfacecolor=TERTILE_COLORS[2],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=TERTILE_LABELS[2],
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    homo = load_homo()
    hetero, n_seeds = load_hetero_avg_over_seeds()
    fig = make_figure(homo, hetero, n_seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "is_decomposition_homo_vs_hetero.pdf"
    out_png = OUT_DIR / "is_decomposition_homo_vs_hetero.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", out_pdf)
    logger.info("Wrote %s", out_png)
    plt.close(fig)


if __name__ == "__main__":
    main()
