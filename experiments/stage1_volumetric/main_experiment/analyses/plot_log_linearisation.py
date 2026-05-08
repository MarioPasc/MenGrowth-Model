"""Simulated Gompertz meningioma growth: V(t) vs log(V+1)(t).

Justifies a homoscedastic LME with random intercept + slope on
log-volume as the working scale for short watch-and-wait windows.

Mathematics (cf. Eq. log-linear-approx in
sections/methods/uncertainty_propagation.tex):

    V(t)        = V_0 * exp( (a/b) * (1 - exp(-b t)) )           (Gompertz)
    log V(t)    = log V_0 + (a/b) * (1 - exp(-b t))
                approx.
                = log V_0 + a * t            when  b T << 1.

Engelhardt et al. 2023 (eBioMedicine) report a deceleration
timescale 1/b on the order of 10-20 years for WHO Grade I
meningioma; with watch-and-wait follow-up T = 3-5 years,
b T in [0.2, 0.33] and the second-order error of the linear
approximation is (b T)^2 / 2 in [0.02, 0.055].

The linear-in-log structure is exactly the mean function of the
random-intercept-and-slope LME of Eq. lme-homo:

    log V_i(t) = (beta_0 + u_{0i}) + (beta_1 + u_{1i}) t + eps,

so the LME is not a generic linear baseline but the first-order
truncation of the canonical clinical growth model on the working
scale where the per-scan LoRA-ensemble uncertainty is computed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "figure.dpi": 120,
    }
)


def gompertz_volume(t: np.ndarray, V0: float, a: float, b: float) -> np.ndarray:
    """Closed-form Gompertz volume V(t).

    Parameters
    ----------
    t : np.ndarray
        Time grid in years.
    V0 : float
        Volume at t = 0 (mm^3).
    a : float
        Initial specific growth rate (1/year).
    b : float
        Deceleration rate (1/year). 1/b is the deceleration timescale.

    Returns
    -------
    V : np.ndarray
        Volume at each t.
    """
    return V0 * np.exp((a / b) * (1.0 - np.exp(-b * t)))


def main(out_pdf: Path, out_png: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)

    # Continuous time grid covering a 5-year watch-and-wait window.
    T_max = 5.0
    t_dense = np.linspace(0.0, T_max, 400)

    # Simulate 6 synthetic patients with parameters bracketed by
    # Engelhardt et al. 2023 (Grade I meningioma):
    #   * 1/b in [10, 20] years  -> b in [0.05, 0.10] /year
    #   * a chosen so that V doubles in ~3-7 years on average
    n_patients = 6
    V0_samples = rng.uniform(low=300.0, high=8_000.0, size=n_patients)
    inv_b_samples = rng.uniform(low=10.0, high=20.0, size=n_patients)  # years
    b_samples = 1.0 / inv_b_samples
    a_samples = rng.uniform(low=0.10, high=0.30, size=n_patients)  # /year

    # Discrete observation timepoints (clinical follow-up cadence,
    # roughly 6-15 months between scans).
    obs_times_per_patient = []
    for _ in range(n_patients):
        n_obs = rng.integers(low=3, high=6)
        t_obs = np.sort(rng.uniform(low=0.2, high=T_max, size=n_obs))
        t_obs = np.concatenate(([0.0], t_obs))  # baseline scan at t=0
        obs_times_per_patient.append(t_obs)

    sigma_log = 0.08  # additive log-volume measurement noise (LoRA-ensemble scale)

    # ------------------------------------------------------------------
    # Single panel with twin y-axes: solid = V (mm^3), dashed = log(V+1).
    # ------------------------------------------------------------------
    fig, ax_lin = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax_log = ax_lin.twinx()

    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.10, 0.85, n_patients)]

    for i in range(n_patients):
        V0_i = V0_samples[i]
        a_i = a_samples[i]
        b_i = b_samples[i]

        V_dense = gompertz_volume(t_dense, V0_i, a_i, b_i)
        logV_dense = np.log(V_dense + 1.0)

        # Discrete observations with measurement noise on log-scale.
        t_obs = obs_times_per_patient[i]
        V_obs_clean = gompertz_volume(t_obs, V0_i, a_i, b_i)
        log_noise = rng.normal(loc=0.0, scale=sigma_log, size=t_obs.shape)
        logV_obs = np.log(V_obs_clean + 1.0) + log_noise
        V_obs = np.exp(logV_obs) - 1.0

        ax_lin.plot(t_dense, V_dense, color=colors[i], alpha=0.9, linestyle="-")
        ax_lin.scatter(
            t_obs,
            V_obs,
            color=colors[i],
            s=20,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )

        ax_log.plot(t_dense, logV_dense, color=colors[i], alpha=0.9, linestyle="--")
        ax_log.scatter(
            t_obs,
            logV_obs,
            color=colors[i],
            s=20,
            marker="x",
            linewidth=1.0,
            zorder=3,
        )

    ax_lin.set_xlabel("Follow-up time $t$ (years)")
    ax_lin.set_ylabel(r"Tumour volume $V$ (mm$^3$)")
    ax_lin.set_xlim(0.0, T_max)
    ax_lin.grid(alpha=0.25, linestyle=":")

    ax_log.set_ylabel(r"$\log(V+1)$")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="0.2", linestyle="-", linewidth=1.4, label=r"$V(t)$ (mm$^3$)"),
        Line2D([0], [0], color="0.2", linestyle="--", linewidth=1.4, label=r"$\log(V+1)$"),
    ]
    ax_lin.legend(handles=legend_handles, loc="upper left", frameon=False)

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
    out_pdf = THESIS_FIG_DIR / "log_linearisation_gompertz.pdf"
    out_png = THESIS_FIG_DIR / "log_linearisation_gompertz.png"
    main(out_pdf=out_pdf, out_png=out_png, seed=7)
