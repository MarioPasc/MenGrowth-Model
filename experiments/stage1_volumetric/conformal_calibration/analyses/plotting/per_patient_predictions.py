"""Per-patient predictive-interval figure for the BSc thesis results section.

Primary results showcase: the homoscedastic LME baseline (``lme_homo``) versus
the ensemble Bayesian model averaging model (``ensemble_bma``), both with the
native parametric (Gaussian) 95% predictive interval. The figure adapts the
methodology explainer ``main_experiment/.../interval_score_explainer.py`` to
real ``conformal_calibration`` LOPO-CV output.

Layout
------
A 2 (model) x 3 (case) grid. Columns are three patients selected by the
*ensemble BMA* per-patient Interval Score IS@95 (Gneiting & Raftery, 2007):
the best (lowest IS), the median, and the worst (highest IS). The same three
patients are shown for both models so the rows are directly comparable; the
y-axis is shared within each column (per patient).

Each panel shows, under the ``last_from_rest`` protocol:

* the conditioning observations (mean log-volume, the shared regression
  target) as a grey line;
* the held-out last observation ``y`` as a red star;
* the predictive mean and its 95% parametric interval ``[L, U]``, with the
  Gaussian predictive density drawn sideways at ``t*``;
* a miss arrow when ``y`` falls outside ``[L, U]``;
* the per-patient IS@95 in the panel title.

On the BMA row only, the M=20 LoRA-ensemble per-member log-volumes are
overlaid as faint dots at each conditioning time-point: these are the inputs
whose between-member disagreement BMA propagates into the (wider, better
calibrated) predictive variance via the law of total variance.

References
----------
Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and
Estimation*, JASA 102:359, 2007 -- the Interval Score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import norm

from growth.stages.stage1_volumetric.trajectory_loader import (
    load_ensemble_trajectories_from_h5,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RESULTS_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction/conformal_calibration"
)
H5_PATH = Path("/media/mpascual/MeningD2/MENINGIOMAS/MENGROWTH/050526/h5_format/MenGrowth.h5")
THESIS_FIG_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/"
    "bachelor_thesis/68596a200c0e0e3876880afa/figures/results"
)

ALPHA = 0.05
Z = float(norm.ppf(1.0 - ALPHA / 2.0))
LAYER = "parametric"
SEED = 0  # parametric layer is seed-deterministic at tau=0; any seed is identical.

# Patient-selection model and the two models plotted (top -> bottom rows).
SELECTION_MODEL = "ensemble_bma"
MODEL_ROWS: tuple[tuple[str, str, str], ...] = (
    ("lme_homo", r"LME homoscedastic", "#4477AA"),
    ("ensemble_bma", r"Ensemble BMA", "#CC6677"),
)
CASE_ORDER: tuple[str, ...] = ("best", "median", "worst")

# QC filter used by the executed run (cohort_meta.json: 56 patients, 173 scans).
COHORT_KWARGS = {
    "time_variable": "ordinal",
    "variance_key": "logvol_var",
    "mean_key": "logvol_mean",
    "scaling": "raw",
    "floor_variance": 1e-6,
    "exclude": ["MenGrowth-0028"],
    "min_timepoints": 2,
    "skip_all_zero_volume": True,
    "max_logvol_std": None,
}

STAR_COLOR = "#b30000"
PAST_COLOR = "0.25"


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def _load_predictions() -> pd.DataFrame:
    """Load the per-patient parametric-layer predictions for both models.

    Returns:
        Frame indexed by ``(base_model, patient_id)`` with the predictive mean,
        variance, interval bounds and IS@95 for ``SEED`` and ``LAYER``.
    """
    pp = pd.read_parquet(RESULTS_ROOT / "aggregated" / "per_patient_table.parquet")
    keep = pp[(pp["seed"] == SEED) & (pp["layer"] == LAYER)].copy()
    if keep.empty:
        raise RuntimeError(f"No rows for seed={SEED}, layer={LAYER} in per_patient_table.")
    return keep.set_index(["base_model", "patient_id"]).sort_index()


def _select_cases(preds: pd.DataFrame) -> dict[str, str]:
    """Pick best / median / worst patients by the selection model's IS@95.

    Args:
        preds: Per-patient prediction frame from :func:`_load_predictions`.

    Returns:
        Mapping ``{"best": pid, "median": pid, "worst": pid}``.
    """
    s = preds.loc[SELECTION_MODEL].sort_values("interval_score")
    return {
        "best": s.index[0],
        "median": s.index[len(s) // 2],
        "worst": s.index[-1],
    }


# --------------------------------------------------------------------------- #
# Panel drawing
# --------------------------------------------------------------------------- #
def _draw_panel(
    ax: plt.Axes,
    past_t: np.ndarray,
    past_y: np.ndarray,
    t_star: float,
    y_true: float,
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
    is_val: float,
    color: str,
    member_t: np.ndarray | None = None,
    member_y: np.ndarray | None = None,
) -> None:
    """Draw one predictive-interval panel for a single (model, patient) cell.

    Args:
        ax: Target axes.
        past_t: Conditioning time-points (ordinal follow-up index).
        past_y: Conditioning mean log-volumes ``log(V_MEN + 1)``.
        t_star: Held-out follow-up index.
        y_true: Held-out observed log-volume.
        mu: Parametric predictive mean at ``t_star``.
        sigma: Parametric predictive standard deviation at ``t_star``.
        lo: Lower 95% interval bound.
        hi: Upper 95% interval bound.
        is_val: Per-patient IS@95 for this cell.
        color: Model colour.
        member_t: Optional time-points for the ensemble-member overlay.
        member_y: Optional ``[n_cond, M]`` per-member log-volumes (BMA row only).
    """
    span = max(t_star - past_t.min(), 1.0)
    half = 0.22 * span  # sideways-density half-width in time units

    # Conditioning ensemble-member cloud (BMA row only).
    if member_t is not None and member_y is not None:
        for k, tk in enumerate(member_t):
            ax.scatter(
                np.full(member_y.shape[1], tk),
                member_y[k],
                s=6,
                color=color,
                alpha=0.18,
                edgecolors="none",
                zorder=1,
            )

    # Conditioning trajectory (mean log-volume).
    ax.plot(
        past_t,
        past_y,
        "o-",
        color=PAST_COLOR,
        markersize=5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        linewidth=1.0,
        zorder=3,
    )
    # Dashed link from the last conditioning point to the predictive mean.
    ax.plot(
        [past_t[-1], t_star],
        [past_y[-1], mu],
        "--",
        color=color,
        linewidth=1.0,
        alpha=0.85,
        zorder=2,
    )

    # Sideways Gaussian predictive density at t*.
    y_grid = np.linspace(min(lo, y_true) - 0.6, max(hi, y_true) + 0.6, 400)
    pdf = norm.pdf(y_grid, loc=mu, scale=max(sigma, 1e-6))
    pdf_w = pdf / pdf.max() * half
    in_ci = (y_grid >= lo) & (y_grid <= hi)
    ax.fill_betweenx(y_grid, t_star - pdf_w, t_star + pdf_w, color=color, alpha=0.08, zorder=1)
    ax.fill_betweenx(
        y_grid[in_ci],
        (t_star - pdf_w)[in_ci],
        (t_star + pdf_w)[in_ci],
        color=color,
        alpha=0.30,
        zorder=2,
    )
    ax.plot(t_star + pdf_w, y_grid, color=color, linewidth=1.0, alpha=0.85, zorder=3)
    ax.plot(t_star - pdf_w, y_grid, color=color, linewidth=1.0, alpha=0.85, zorder=3)

    # Interval bounds.
    ax.hlines(
        [lo, hi],
        xmin=t_star - 0.42 * span,
        xmax=t_star + 0.42 * span,
        colors=color,
        linestyles=":",
        linewidth=0.9,
        zorder=4,
    )
    for bound, label in ((hi, r"$U$"), (lo, r"$L$")):
        ax.text(
            t_star + 0.46 * span,
            bound,
            label,
            color=color,
            fontsize=9,
            va="center",
            ha="left",
        )

    # Predictive mean.
    ax.plot(
        t_star,
        mu,
        "o",
        color=color,
        markersize=5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        zorder=5,
    )
    # Held-out observation.
    ax.plot(
        t_star,
        y_true,
        marker="*",
        color=STAR_COLOR,
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=6,
    )

    # Miss arrow when the held-out point falls outside [L, U]. The arrow sits
    # clear of the U/L bound labels; it is explained once in the figure legend.
    if y_true < lo or y_true > hi:
        y_a, y_b = (y_true, lo) if y_true < lo else (hi, y_true)
        x_arrow = t_star + 0.72 * span
        ax.annotate(
            "",
            xy=(x_arrow, y_a),
            xytext=(x_arrow, y_b),
            arrowprops={"arrowstyle": "<->", "color": STAR_COLOR, "lw": 1.2},
            zorder=6,
        )

    ax.set_xlim(past_t.min() - 0.35 * span, t_star + 0.92 * span)
    ax.set_xticks(np.unique(np.concatenate([past_t, [t_star]])))
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_title(rf"IS@95 $= {is_val:.2f}$", fontsize=9)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def main(out_pdf: Path, out_png: Path) -> None:
    """Build and write the per-patient predictive-interval figure.

    Args:
        out_pdf: Destination PDF path.
        out_png: Destination PNG path.
    """
    preds = _load_predictions()
    cases = _select_cases(preds)
    logger.info("Cases selected by %s IS@95: %s", SELECTION_MODEL, cases)

    trajs = {
        t.patient_id: t
        for t in load_ensemble_trajectories_from_h5(h5_path=str(H5_PATH), **COHORT_KWARGS)
    }

    n_rows, n_cols = len(MODEL_ROWS), len(CASE_ORDER)
    fig = plt.figure(figsize=(11.0, 6.4))
    gs = gridspec.GridSpec(
        nrows=n_rows,
        ncols=n_cols,
        wspace=0.20,
        hspace=0.34,
        left=0.085,
        right=0.965,
        top=0.90,
        bottom=0.13,
        figure=fig,
    )

    for j, case in enumerate(CASE_ORDER):
        pid = cases[case]
        traj = trajs[pid]
        past_t = traj.times[:-1].astype(float)
        past_y = np.asarray(traj.observations[:-1]).reshape(-1).astype(float)
        t_star = float(traj.times[-1])

        # Shared y-range for this patient (column), covering both models.
        col_lo, col_hi = [], []
        for base_model, _, _ in MODEL_ROWS:
            row = preds.loc[(base_model, pid)]
            col_lo += [row["lower"], row["actual"]]
            col_hi += [row["upper"], row["actual"]]
        y_lo = float(min(past_y.min(), min(col_lo)) - 0.5)
        y_hi = float(max(past_y.max(), max(col_hi)) + 0.5)

        for i, (base_model, model_label, color) in enumerate(MODEL_ROWS):
            ax = fig.add_subplot(gs[i, j])
            row = preds.loc[(base_model, pid)]
            mu = float(row["pred_mean"])
            sigma = float(np.sqrt(max(row["pred_var"], 1e-12)))

            member_t = member_y = None
            if base_model == "ensemble_bma" and traj.observation_ensemble is not None:
                member_t = past_t
                member_y = np.asarray(traj.observation_ensemble[:-1], dtype=float)

            _draw_panel(
                ax,
                past_t=past_t,
                past_y=past_y,
                t_star=t_star,
                y_true=float(row["actual"]),
                mu=mu,
                sigma=sigma,
                lo=float(row["lower"]),
                hi=float(row["upper"]),
                is_val=float(row["interval_score"]),
                color=color,
                member_t=member_t,
                member_y=member_y,
            )
            ax.set_ylim(y_lo, y_hi)

            if i == 0:
                ax.annotate(
                    rf"\textbf{{{case}}} -- {pid.replace('MenGrowth-', 'Patient ')}",
                    xy=(0.5, 1.16),
                    xycoords="axes fraction",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            if i == n_rows - 1:
                ax.set_xlabel(r"follow-up index $t$")
            if j == 0:
                ax.set_ylabel(rf"{model_label}" + "\n" + r"$\log(V_{\mathrm{MEN}}+1)$")

    # Shared legend.
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="-",
            marker="o",
            markersize=5,
            color=PAST_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="conditioning observations",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            markersize=14,
            color=STAR_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=r"held-out observation $y$ at $t^\ast$",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5,
            color="0.45",
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=r"predictive mean $\widehat{\mu}$ with 95\% interval $[L,U]$",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5,
            color=MODEL_ROWS[1][2],
            alpha=0.35,
            markeredgecolor="none",
            label=r"LoRA ensemble members ($M=20$, BMA row)",
        ),
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=r"$\updownarrow$",
            markersize=11,
            color=STAR_COLOR,
            label=r"miss penalty: $y$ outside $[L,U]$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", out_pdf)
    logger.info("Wrote %s", out_png)
    plt.close(fig)


if __name__ == "__main__":
    local_dir = RESULTS_ROOT / "figures"
    main(
        out_pdf=THESIS_FIG_DIR / "per_patient_predictions_homo_bma.pdf",
        out_png=local_dir / "per_patient_predictions_homo_bma.png",
    )
