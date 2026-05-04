"""Growth ODE profiles figure for the thesis related-work section.

Compares analytical solutions of exponential, logistic, and Gompertz
tumor growth models for varying initial specific growth rate *a*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.uncertainty_segmentation.plotting.style import setup_style

OUTPUT_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/"
    "bachelor_thesis/68596a200c0e0e3876880afa/figures/related_work"
)


def _exponential(t: np.ndarray, a: float, V0: float = 1.0) -> np.ndarray:
    return V0 * np.exp(a * t)


def _logistic(t: np.ndarray, a: float, V0: float = 1.0, K: float = 10.0) -> np.ndarray:
    return K / (1.0 + (K / V0 - 1.0) * np.exp(-a * t))


def _gompertz(t: np.ndarray, a: float, V0: float = 1.0, K: float = 10.0) -> np.ndarray:
    b = a / np.log(K / V0)
    return V0 * np.exp((a / b) * (1.0 - np.exp(-b * t)))


def main() -> None:
    setup_style()

    V0, K = 1.0, 10.0
    t = np.linspace(0, 15, 500)
    a_values = np.linspace(0.1, 0.8, 8)
    ylim_max = 12.0

    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=a_values.min(), vmax=a_values.max())

    models: list[tuple] = [
        (
            _exponential,
            r"$\frac{\mathrm{d}V}{\mathrm{d}t} = aV$",
            "(a)",
            False,
        ),
        (
            _logistic,
            r"$\frac{\mathrm{d}V}{\mathrm{d}t} = aV"
            r"\!\left(1 - \frac{V}{K}\right)$",
            "(b)",
            True,
        ),
        (
            _gompertz,
            r"$\frac{\mathrm{d}V}{\mathrm{d}t} = aV"
            r"\ln\!\left(\frac{K}{V}\right)$",
            "(c)",
            True,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.08, bottom=0.22)

    for ax, (model_fn, eq_text, panel_label, show_K) in zip(axes, models):
        for a_val in a_values:
            V = model_fn(t, a_val)
            V_clipped = np.clip(V, 0, ylim_max * 5)
            ax.plot(
                t,
                V_clipped,
                color=cmap(norm(a_val)),
                lw=1.0,
                clip_on=True,
                solid_capstyle="round",
            )

        if show_K:
            ax.axhline(K, color="0.6", ls="--", lw=0.6, zorder=0)
            ax.text(
                t.max() * 0.97,
                K + 0.25,
                r"$K$",
                ha="right",
                va="bottom",
                fontsize=7,
                color="0.5",
            )

        ax.set_ylim(0, ylim_max)
        ax.set_xlim(0, t.max())
        ax.set_xlabel(r"$t$")
        ax.set_xticks([0, 5, 10, 15])
        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.7, linewidth=0.4, color="0.7")
        ax.set_axisbelow(True)

        ax.text(
            0.04,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            0.96,
            0.96,
            eq_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
        )

    axes[0].set_ylabel(r"$V(t)$")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$a$ (initial specific growth rate)", fontsize=8)
    cbar.set_ticks(a_values)
    cbar.ax.tick_params(labelsize=7)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "growth_ode_profiles.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
