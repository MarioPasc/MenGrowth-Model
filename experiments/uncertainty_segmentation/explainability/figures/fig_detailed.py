"""Detailed per-metric panel figures (spec §7, richer than bar charts).

Each renderer produces a multi-row figure combining a **visual example** on
row 1 (requires a one-shot GPU re-run by the caller) with **aggregate
statistics** on rows 2-3 (read from the cached raw CSVs/NPZs written by
``run_analysis.py``).  The layout echoes the legacy ``figure_tsi.py`` 3×5
panel but now covers all three explainability metrics.

Outputs (placed by the CLI in ``{output_dir}/figures/``):

- ``tsi_detailed_{condition}.{fmt}``  — 3 × 5 (stages 0–4)
- ``asi_detailed_{condition}.{fmt}``  — 3 × 4 (stages 1–4)
- ``dad_detailed_{condition}.{fmt}``  — 3 × 4 (stages 1–4)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..engine.tsi import ScanTSIResult

logger = logging.getLogger(__name__)

# Paul-Tol vibrant palette.
COL_FROZEN = "#0077BB"
COL_ADAPTED = "#EE7733"
COL_MEN = "#CC3311"
COL_GLI = "#009988"
COL_NULL = "#BBBBBB"


# =============================================================================
# TSI — 3 × 5 (stages 0-4)
# =============================================================================


def _upsample_to_input(act_map: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    t = torch.from_numpy(act_map).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=target_size, mode="trilinear", align_corners=False)
    return up.squeeze().numpy()


def render_tsi_detailed(
    scan_tsi: ScanTSIResult,
    mri_volume: np.ndarray,
    wt_mask: np.ndarray,
    slice_idx: int,
    channels_npz_path: Path,
    condition: str,
    out_path: Path,
    lora_stages: set[int] | None = None,
    dpi: int = 300,
) -> None:
    """3×5 TSI panel: mean activation + top-3 channels + aggregate histogram.

    Parameters
    ----------
    scan_tsi : ScanTSIResult
        Per-stage TSI for one representative scan (must be computed with
        ``return_maps=True`` so the activation overlays are populated).
    mri_volume : np.ndarray
        The scan's MRI volume ``[D, H, W]`` (typically T1ce channel).
    wt_mask : np.ndarray
        Binary whole-tumor mask ``[D, H, W]``.
    slice_idx : int
        Axial slice index for the visualisation row.
    channels_npz_path : pathlib.Path
        Path to ``tsi_{condition}_channels.npz`` — pooled per-channel TSI
        across every processed scan.  Used for the row-3 histograms.
    condition : str
        ``"frozen"`` or ``"adapted"`` (drives the NPZ filename and title).
    out_path : pathlib.Path
        Output figure path.
    lora_stages : set[int] | None
        Stages with LoRA adapters (highlighted in the column header).
    dpi : int
        Output DPI.
    """
    lora = lora_stages or set()
    target_size = mri_volume.shape
    mri_slice = mri_volume[slice_idx]
    mask_slice = wt_mask[slice_idx]

    fig, axes = plt.subplots(3, 5, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(
        f"Brain-masked TSI — {condition} (scan = {scan_tsi.scan_id})",
        fontsize=13, fontweight="bold",
    )

    # Pool per-channel TSI across all scans for the aggregate histogram.
    all_tsi_by_stage: dict[int, list[float]] = {s: [] for s in range(5)}
    if channels_npz_path.exists():
        npz = np.load(channels_npz_path, allow_pickle=True)
        for key in npz.files:
            if "_stage" not in key:
                continue
            stage = int(key.rsplit("_stage", 1)[1])
            all_tsi_by_stage[stage].extend(npz[key].tolist())
    else:
        logger.warning("channels_npz not found: %s", channels_npz_path)

    for s in range(5):
        tsi_res = scan_tsi.stages[s]
        lora_tag = "  [LoRA]" if s in lora else ""
        axes[0, s].set_title(
            f"Stage {s}{lora_tag}\n"
            f"$C_s$={tsi_res.n_channels}, {tuple(tsi_res.resolution)}",
            fontsize=9,
        )

        # ---- Row 1: mean |h| on MRI ----
        ax = axes[0, s]
        ax.imshow(mri_slice.T, cmap="gray", origin="lower", aspect="equal")
        if tsi_res.mean_activation_map is not None:
            up = _upsample_to_input(tsi_res.mean_activation_map, target_size)[slice_idx]
            ax.imshow(up.T, cmap="inferno", alpha=0.5, origin="lower", aspect="equal")
        ax.contour(mask_slice.T, levels=[0.5], colors="white",
                   linestyles="dashed", linewidths=0.8)
        ax.axis("off")

        # ---- Row 2: top-3 channels on MRI ----
        ax = axes[1, s]
        ax.imshow(mri_slice.T, cmap="gray", origin="lower", aspect="equal")
        if tsi_res.top_channels_map is not None:
            up = _upsample_to_input(tsi_res.top_channels_map, target_size)[slice_idx]
            ax.imshow(up.T, cmap="plasma", alpha=0.5, origin="lower", aspect="equal")
            pos = up[up > 0]
            if pos.size:
                p75 = float(np.percentile(pos, 75))
                ax.contour(up.T, levels=[p75],
                           colors="cyan", linewidths=0.8)
        ax.contour(mask_slice.T, levels=[0.5], colors="white",
                   linestyles="dashed", linewidths=0.8)
        ax.axis("off")

        # ---- Row 3: aggregate TSI histogram across all scans ----
        ax = axes[2, s]
        pooled = np.asarray(all_tsi_by_stage[s], dtype=np.float32)
        pooled = pooled[~np.isnan(pooled)]
        if pooled.size:
            clipped = np.clip(pooled, 0, 4)
            ax.hist(clipped, bins=30, range=(0, 4),
                    color=COL_NULL, edgecolor="black", linewidth=0.4)
            selective = pooled[pooled > 1.5]
            if selective.size:
                ax.hist(np.clip(selective, 0, 4), bins=30, range=(0, 4),
                        color=COL_FROZEN, alpha=0.75,
                        edgecolor="black", linewidth=0.4)
            frac_15 = float(np.mean(pooled > 1.5))
            ax.text(
                0.97, 0.95, f">1.5: {frac_15:.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
            )
        ax.axvline(1.0, color="black", linestyle="-", linewidth=0.8)
        ax.axvline(1.5, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 4)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("TSI" if s == 2 else "", fontsize=9)

    axes[0, 0].set_ylabel("Mean |h|", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Top-3 channels", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("TSI dist. (all scans)", fontsize=10, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# =============================================================================
# ASI — 3 × 4 (stages 1-4)
# =============================================================================


def render_asi_detailed(
    asi_visual: dict[int, dict],
    asi_per_scan_df: pd.DataFrame,
    scan_id: str,
    condition: str,
    out_path: Path,
    stages: tuple[int, ...] = (1, 2, 3, 4),
    lora_stages: set[int] | None = None,
    dpi: int = 300,
) -> None:
    """3×4 ASI panel: attention heatmap + per-stage ASI dist + per-head bar.

    Parameters
    ----------
    asi_visual : dict[int, dict]
        For each stage, a dict with keys:
            ``attn_matrix`` : np.ndarray ``[H, N, N]`` — post-softmax for one
                boundary window (tokens reordered tumor-first).
            ``n_tumor`` : int — number of tumor tokens (position of the
                boundary line on the heatmap).
            ``block`` : int — which block the window came from (0 or 1).
    asi_per_scan_df : pd.DataFrame
        Long-format ASI table (``stage, block, head, asi_value`` columns).
        Used for rows 2-3.
    scan_id, condition : str
        For the figure title.
    out_path : pathlib.Path
    stages : tuple[int, ...]
        Stages shown in the columns (default ``(1, 2, 3, 4)``).
    lora_stages : set[int] | None
    dpi : int
    """
    lora = lora_stages or set()
    n = len(stages)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 9), constrained_layout=True)
    fig.suptitle(
        f"Attention Selectivity Index (ASI) — {condition} "
        f"(visual scan = {scan_id})",
        fontsize=13, fontweight="bold",
    )

    for col, stage in enumerate(stages):
        lora_tag = "  [LoRA]" if stage in lora else ""
        vis = asi_visual.get(stage, {})
        axes[0, col].set_title(f"Stage {stage}{lora_tag}", fontsize=10)

        # ---- Row 1: attention heatmap (one boundary window, tumor-first) ----
        ax = axes[0, col]
        attn = vis.get("attn_matrix")
        n_tumor = vis.get("n_tumor", 0)
        if attn is not None and attn.size > 0:
            # Mean over heads for interpretability; individual heads available in raw.
            attn_mean = attn.mean(axis=0)
            im = ax.imshow(attn_mean, cmap="viridis", aspect="equal", origin="upper")
            ax.axhline(n_tumor - 0.5, color="white", linewidth=0.6, linestyle="--")
            ax.axvline(n_tumor - 0.5, color="white", linewidth=0.6, linestyle="--")
            N = attn_mean.shape[0]
            ax.text(0.02, 0.98, f"N={N}, H={attn.shape[0]}\n"
                                f"block={vis.get('block', '?')}, n_T={n_tumor}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="black", alpha=0.55))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.text(0.5, 0.5, "no boundary window", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])

        # ---- Row 2: aggregate ASI histogram (all scans, windows, heads) ----
        ax = axes[1, col]
        vals = asi_per_scan_df.loc[
            asi_per_scan_df["stage"] == stage, "asi_value"
        ].to_numpy(dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            clipped = np.clip(vals, 0, 4)
            ax.hist(clipped, bins=40, range=(0, 4),
                    color=COL_NULL, edgecolor="black", linewidth=0.4)
            above_1 = vals[vals > 1.0]
            if above_1.size:
                ax.hist(np.clip(above_1, 0, 4), bins=40, range=(0, 4),
                        color=COL_FROZEN, alpha=0.7,
                        edgecolor="black", linewidth=0.4)
            frac_1 = float(np.mean(vals > 1.0))
            ax.text(0.97, 0.95, f">1: {frac_1:.0%}\nN={vals.size:,}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", alpha=0.85))
        ax.axvline(1.0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xlim(0, 4)
        ax.set_xlabel("ASI" if col == n // 2 else "", fontsize=9)
        ax.tick_params(labelsize=7)

        # ---- Row 3: per-head mean ASI bar (across all scans × windows) ----
        ax = axes[2, col]
        subset = asi_per_scan_df[asi_per_scan_df["stage"] == stage]
        if not subset.empty:
            agg = subset.groupby("head")["asi_value"].agg(["mean", "std", "count"])
            heads = agg.index.to_numpy()
            means = agg["mean"].to_numpy()
            err = 1.96 * agg["std"].to_numpy() / np.sqrt(np.clip(agg["count"].to_numpy(), 1, None))
            ax.bar(heads, means, yerr=err, capsize=2, color=COL_FROZEN,
                   edgecolor="black", linewidth=0.3)
            ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7)
            ax.set_xticks(heads)
        ax.set_xlabel("head" if col == n // 2 else "", fontsize=9)
        ax.tick_params(labelsize=7)

    axes[0, 0].set_ylabel("Attn (head-mean)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("ASI dist. (all scans)", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Per-head mean ASI", fontsize=10, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# =============================================================================
# DAD — 3 × 4 (stages 1-4)
# =============================================================================


def render_dad_detailed(
    dad_visual: dict[int, dict],
    dad_df: pd.DataFrame,
    null_npz_path: Path,
    condition: str,
    out_path: Path,
    stages: tuple[int, ...] = (1, 2, 3, 4),
    lora_stages: set[int] | None = None,
    dpi: int = 300,
) -> None:
    """3×4 DAD panel: MEN vs GLI distribution + DAD bar + significance heatmap.

    Parameters
    ----------
    dad_visual : dict[int, dict]
        For each stage, a dict with keys:
            ``men`` : np.ndarray ``[H, N]`` — row-averaged attention for
                one representative MEN scan.
            ``gli`` : np.ndarray ``[H, N]`` — same for one GLI scan.
            ``head`` : int — which head is shown on row 1.
            ``block`` : int — which block the distribution came from.
    dad_df : pd.DataFrame
        ``dad_per_head.csv`` contents.  Rows 2-3 use the ``condition`` filter.
    null_npz_path : pathlib.Path
        Path to ``dad_permutation_null.npz`` (per-head null mean/std).
    condition : str
        Which condition to plot ``(frozen | adapted_r8 | …)``.
    out_path : pathlib.Path
    stages : tuple[int, ...]
    lora_stages : set[int] | None
    dpi : int
    """
    lora = lora_stages or set()
    n = len(stages)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 9), constrained_layout=True)
    fig.suptitle(
        f"Domain Attention Divergence (DAD) — {condition}",
        fontsize=13, fontweight="bold",
    )

    null_npz = dict(np.load(null_npz_path, allow_pickle=True)) if null_npz_path.exists() else {}

    cond_df = dad_df[dad_df["condition"] == condition]
    if cond_df.empty:
        logger.warning("No DAD rows for condition=%s; using first available", condition)
        cond_df = dad_df
        condition = str(dad_df["condition"].iloc[0])

    for col, stage in enumerate(stages):
        lora_tag = "  [LoRA]" if stage in lora else ""
        vis = dad_visual.get(stage, {})
        axes[0, col].set_title(f"Stage {stage}{lora_tag}", fontsize=10)

        # ---- Row 1: sorted row-avg attention MEN vs GLI (one head) ----
        ax = axes[0, col]
        men = vis.get("men")
        gli = vis.get("gli")
        head = int(vis.get("head", 0))
        block = int(vis.get("block", 0))
        if men is not None and gli is not None and men.size and gli.size:
            p_men = np.sort(men[head])[::-1]
            p_gli = np.sort(gli[head])[::-1]
            x = np.arange(p_men.size)
            ax.plot(x, p_men, color=COL_MEN, linewidth=1.2, label="MEN")
            ax.plot(x, p_gli, color=COL_GLI, linewidth=1.2, label="GLI")
            ax.fill_between(x, p_men, p_gli,
                            color="#999999", alpha=0.25, linewidth=0)
            ax.set_yscale("log")
            ax.set_xlabel("sorted token rank" if col == n // 2 else "", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(labelsize=7)
            ax.text(0.02, 0.02, f"head={head}, block={block}",
                    transform=ax.transAxes, va="bottom", ha="left", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", alpha=0.85))
        else:
            ax.text(0.5, 0.5, "no row-avg captured", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.axis("off")

        # ---- Row 2: per-head DAD bar vs null mean/std ----
        ax = axes[1, col]
        sub = cond_df[(cond_df["stage"] == stage) & (cond_df["block"] == 0)].sort_values("head")
        if not sub.empty:
            heads = sub["head"].to_numpy()
            dad_vals = sub["dad"].to_numpy()
            null_mu = sub["null_mean"].to_numpy()
            null_sd = sub["null_std"].to_numpy()
            p_vals = sub["p_value"].to_numpy()
            ax.bar(heads, dad_vals, color=COL_FROZEN,
                   edgecolor="black", linewidth=0.3, label="DAD")
            ax.fill_between(heads, null_mu - 2 * null_sd, null_mu + 2 * null_sd,
                            step="mid", color=COL_NULL, alpha=0.55,
                            label="null ±2σ")
            ax.plot(heads, null_mu, color="black", linewidth=0.8, marker="_")
            for h, v, p in zip(heads, dad_vals, p_vals):
                if p < 0.05:
                    ax.text(h, v, "*", ha="center", va="bottom",
                            fontsize=11, fontweight="bold", color="red")
            ax.set_xticks(heads)
            if col == n - 1:
                ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("head (block=0)" if col == n // 2 else "", fontsize=8)
        ax.tick_params(labelsize=7)

        # ---- Row 3: -log10(p) heatmap across both blocks × heads ----
        ax = axes[2, col]
        pivot = (
            cond_df[cond_df["stage"] == stage]
            .pivot_table(index="block", columns="head", values="p_value",
                         aggfunc="first")
        )
        if not pivot.empty:
            vals = -np.log10(pivot.to_numpy(dtype=float).clip(min=1e-4))
            im = ax.imshow(vals, cmap="magma", aspect="auto",
                           vmin=0, vmax=3.5)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels([f"blk {b}" for b in pivot.index], fontsize=7)
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels([str(h) for h in pivot.columns], fontsize=7)
            # Annotate with * where p<0.05.
            for (i, j), p in np.ndenumerate(pivot.to_numpy(dtype=float)):
                if p < 0.05:
                    ax.text(j, i, "*", ha="center", va="center",
                            color="white", fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                         label="−log₁₀ p" if col == n - 1 else "")
        ax.set_xlabel("head" if col == n // 2 else "", fontsize=8)

    axes[0, 0].set_ylabel("Sorted attention (log)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("DAD (block 0)", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Signif. (−log₁₀ p)", fontsize=10, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
