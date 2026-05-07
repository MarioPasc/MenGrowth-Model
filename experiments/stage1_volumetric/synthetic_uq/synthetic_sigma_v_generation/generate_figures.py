"""Generate diagnostic figures for the smooth σ²_v shape-shift family.

Produces two figures:

1. ``sigma_v_shape_heatmap_2d.{pdf,png}`` — 2D heatmap with α on the
   x-axis, σ²_v bin on the y-axis, cell colour = sample count. Shows
   how the σ²_v distribution morphs from a peak at 0 (α=-1) through
   uniform (α=0) to a peak at σ²_max (α=+1).
2. ``sigma_v_shape_surface_3d.{pdf,png}`` — 3D surface version of the
   same data, height = sample count.

Run via::

    cd /home/mpascual/research/code/MenGrowth-Model
    ~/.conda/envs/growth/bin/python -m \\
        experiments.stage1_volumetric.synthetic_uq.synthetic_sigma_v_generation.generate_figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .beta_sampler import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_SIGMA_V_SQ_MAX,
    DEFAULT_STEEPNESS,
    sample_sigma_v_grid,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = (
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction"
)


def build_count_grid(
    samples: np.ndarray,
    sigma_v_sq_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin the per-α samples on a shared σ²_v grid and return counts.

    Args:
        samples: 2-D array, shape ``(n_alphas, n_per_alpha)``.
        sigma_v_sq_max: Right edge of the σ²_v axis.
        n_bins: Number of σ²_v bins.

    Returns:
        ``(counts, alpha_edges, sigma_edges)`` where ``counts`` has
        shape ``(n_bins, n_alphas)`` (rows = σ²_v bins, columns = α).
    """
    n_alphas, _ = samples.shape
    sigma_edges = np.linspace(0.0, sigma_v_sq_max, n_bins + 1)
    counts = np.zeros((n_bins, n_alphas), dtype=np.int64)
    for k in range(n_alphas):
        c, _ = np.histogram(samples[k], bins=sigma_edges)
        counts[:, k] = c
    alpha_edges = np.arange(n_alphas + 1, dtype=np.float64)
    return counts, alpha_edges, sigma_edges


def plot_heatmap_2d(
    counts: np.ndarray,
    alphas: np.ndarray,
    sigma_centres: np.ndarray,
    output_path: Path,
    *,
    cmap: str = "viridis",
) -> None:
    """Render the 2D σ²_v × α heatmap, coloured by sample count.

    Args:
        counts: ``(n_sigma_bins, n_alphas)`` integer counts.
        alphas: Length-n_alphas vector of α values.
        sigma_centres: Length-n_sigma_bins vector of σ²_v bin centres.
        output_path: Destination *base* path (extensions appended).
        cmap: matplotlib colormap name.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # pcolormesh edges: extend by one half-step on each side.
    da = float(alphas[1] - alphas[0]) if len(alphas) > 1 else 1.0
    alpha_edges = np.concatenate([[alphas[0] - da / 2], alphas + da / 2])
    ds = float(sigma_centres[1] - sigma_centres[0]) if len(sigma_centres) > 1 else 1.0
    sigma_edges = np.concatenate([[sigma_centres[0] - ds / 2], sigma_centres + ds / 2])

    norm = LogNorm(vmin=max(counts[counts > 0].min(), 1), vmax=counts.max())
    pcm = ax.pcolormesh(
        alpha_edges,
        sigma_edges,
        counts,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Sample count (log scale)", fontsize=11)

    ax.set_xlabel(r"Beta-family shape knob $\alpha$", fontsize=12)
    ax.set_ylabel(r"$\sigma^2_v$ bin centre", fontsize=12)
    ax.set_title(
        r"Smooth $\sigma^2_v$ shape sweep — counts per ($\alpha$, $\sigma^2_v$) cell",
        fontsize=12,
    )

    # Annotate the three reference α values.
    for x, label in [(-1.0, "peak @ 0"), (0.0, "uniform"), (1.0, r"peak @ $\sigma^2_\max$")]:
        if x in set(alphas):
            ax.axvline(x, color="white", lw=0.5, ls="--", alpha=0.6)
            ax.text(
                x,
                sigma_edges[-1] * 0.97,
                label,
                rotation=90,
                ha="right",
                va="top",
                fontsize=8,
                color="white",
            )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_surface_3d(
    counts: np.ndarray,
    alphas: np.ndarray,
    sigma_centres: np.ndarray,
    output_path: Path,
    *,
    cmap: str = "viridis",
) -> None:
    """Render a 3D surface where height = sample count.

    Args:
        counts: ``(n_sigma_bins, n_alphas)`` integer counts.
        alphas: Length-n_alphas vector of α values.
        sigma_centres: Length-n_sigma_bins vector of σ²_v bin centres.
        output_path: Destination base path (extensions appended).
        cmap: matplotlib colormap name.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    A, S = np.meshgrid(alphas, sigma_centres)
    Z = counts.astype(np.float64)
    surf = ax.plot_surface(
        A,
        S,
        Z,
        cmap=cmap,
        edgecolor="none",
        antialiased=True,
        alpha=0.95,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.10)
    cbar.set_label("Sample count", fontsize=10)

    ax.set_xlabel(r"$\alpha$", fontsize=11, labelpad=8)
    ax.set_ylabel(r"$\sigma^2_v$", fontsize=11, labelpad=8)
    ax.set_zlabel("count", fontsize=11, labelpad=4)
    ax.set_title(
        r"Smooth $\sigma^2_v$ shape sweep — 3D count surface",
        fontsize=11,
    )

    ax.view_init(elev=22, azim=-55)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the figures (a 'sigma_v_shape_family' subfolder is created).",
    )
    parser.add_argument(
        "--n-per-alpha",
        type=int,
        default=20000,
        help="Samples drawn at each α (default 20 000 → smooth heatmap).",
    )
    parser.add_argument(
        "--n-sigma-bins",
        type=int,
        default=80,
        help="Number of σ²_v bins on the y-axis (default 80).",
    )
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=None,
        help="Override α grid (default: 9 levels in [-1, 1] step 0.25).",
    )
    parser.add_argument(
        "--sigma-v-sq-max",
        type=float,
        default=DEFAULT_SIGMA_V_SQ_MAX,
        help=f"σ²_v upper bound (default {DEFAULT_SIGMA_V_SQ_MAX}).",
    )
    parser.add_argument(
        "--steepness",
        type=float,
        default=DEFAULT_STEEPNESS,
        help=f"Beta peak sharpness at |α|=1 (default {DEFAULT_STEEPNESS}).",
    )
    parser.add_argument(
        "--fixed-mean",
        type=float,
        default=None,
        help="Optional mean pinning across α (free-mean if omitted).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-finer-alphas",
        type=int,
        default=33,
        help="Density of α used for the count heatmap (default 33 → step 0.0625).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    out_root = Path(args.output_dir) / "sigma_v_shape_family"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.alphas is not None:
        alpha_coarse = tuple(args.alphas)
    else:
        alpha_coarse = DEFAULT_ALPHA_GRID

    # Use a finer α grid for the count heatmap so the shape transition is
    # visually smooth, and the coarse grid for the metadata + reference lines.
    alpha_fine = tuple(np.linspace(-1.0, 1.0, args.n_finer_alphas).round(4).tolist())

    logger.info(
        "Sampling Beta family: %d α levels (fine), %d samples/α, σ²_max=%.3f, s=%.1f, fixed_mean=%s",
        len(alpha_fine),
        args.n_per_alpha,
        args.sigma_v_sq_max,
        args.steepness,
        args.fixed_mean,
    )
    samples = sample_sigma_v_grid(
        alphas=alpha_fine,
        n=args.n_per_alpha,
        sigma_v_sq_max=args.sigma_v_sq_max,
        steepness=args.steepness,
        fixed_mean=args.fixed_mean,
        seed=args.seed,
    )

    # Diagnostics — verify the family limits behave correctly.
    means = samples.mean(axis=1)
    medians = np.median(samples, axis=1)
    logger.info(
        "α=%.2f mean=%.4f median=%.4f | α=%.2f mean=%.4f median=%.4f | α=%.2f mean=%.4f median=%.4f",
        alpha_fine[0],
        means[0],
        medians[0],
        alpha_fine[len(alpha_fine) // 2],
        means[len(alpha_fine) // 2],
        medians[len(alpha_fine) // 2],
        alpha_fine[-1],
        means[-1],
        medians[-1],
    )

    counts, _, sigma_edges = build_count_grid(samples, args.sigma_v_sq_max, args.n_sigma_bins)
    sigma_centres = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])

    # ------- Save raw arrays + metadata for reproducibility -------
    np.savez(
        out_root / "sigma_v_shape_counts.npz",
        counts=counts,
        alphas=np.asarray(alpha_fine),
        sigma_v_sq_centres=sigma_centres,
        sigma_v_sq_edges=sigma_edges,
        sample_means=means,
        sample_medians=medians,
    )
    metadata = {
        "n_per_alpha": int(args.n_per_alpha),
        "n_sigma_bins": int(args.n_sigma_bins),
        "alphas_fine": list(alpha_fine),
        "alphas_coarse": list(alpha_coarse),
        "sigma_v_sq_max": float(args.sigma_v_sq_max),
        "steepness": float(args.steepness),
        "fixed_mean": args.fixed_mean,
        "seed": int(args.seed),
        "alpha_to_mean": dict(zip([float(a) for a in alpha_fine], [float(m) for m in means], strict=True)),
        "alpha_to_median": dict(zip([float(a) for a in alpha_fine], [float(m) for m in medians], strict=True)),
    }
    with open(out_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ------- 2D heatmap -------
    plot_heatmap_2d(
        counts,
        np.asarray(alpha_fine),
        sigma_centres,
        out_root / "sigma_v_shape_heatmap_2d",
    )
    logger.info("Wrote sigma_v_shape_heatmap_2d.{pdf,png} → %s", out_root)

    # ------- 3D surface -------
    plot_surface_3d(
        counts,
        np.asarray(alpha_fine),
        sigma_centres,
        out_root / "sigma_v_shape_surface_3d",
    )
    logger.info("Wrote sigma_v_shape_surface_3d.{pdf,png} → %s", out_root)

    logger.info("Done. Artifacts in %s", out_root)


if __name__ == "__main__":
    main()
