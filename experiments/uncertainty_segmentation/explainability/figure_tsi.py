"""Figure generation for TSI analysis.

Produces three figure types:
1. 5×3 panel figure (per condition): mean activation heatmap, top-K channels, TSI histogram
2. 1×5 delta summary (adapted − frozen mean TSI per stage)
3. Cross-rank comparison (delta TSI across r=4, 8, 16)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from omegaconf import DictConfig

from growth.data.bratsmendata import BraTSDatasetH5

from .tsi_analysis import STAGE_META, ScanTSIResult


def _resolve_stage_meta(stage_meta: dict | None) -> dict:
    """Fall back to the legacy [3,4] STAGE_META when callers pass None.

    Why: thread-through is opt-in; old call sites keep working but figures will
    mislabel stages as in the [3,4] default. New callers should always pass
    stage_meta built from config.lora.target_stages.
    """
    return stage_meta if stage_meta is not None else STAGE_META

logger = logging.getLogger(__name__)

# Paul Tol vibrant palette for ranks
RANK_COLORS = {4: "#0077BB", 8: "#33BBEE", 16: "#EE7733"}
FROZEN_COLOR = "#BBBBBB"
ADAPTED_COLOR = "#0077BB"


def save_viz_data(
    scan_result: ScanTSIResult,
    mri_volume: np.ndarray,
    wt_mask: np.ndarray,
    slice_idx: int,
    output_path: Path,
) -> None:
    """Save all data needed for panel figure regeneration.

    Saves the 2D slices (compact) and per-channel TSI arrays.
    Total size: ~5 MB for 192³, much smaller than full 3D volumes.

    Args:
        scan_result: TSI results with activation maps for one scan.
        mri_volume: T1ce volume [D, H, W].
        wt_mask: Binary WT mask [D, H, W].
        slice_idx: Axial slice index used for visualization.
        output_path: Path to .npz file.
    """
    target_size = mri_volume.shape
    data: dict[str, np.ndarray] = {
        "mri_slice": mri_volume[slice_idx],
        "mask_slice": wt_mask[slice_idx],
        "slice_idx": np.array(slice_idx),
        "scan_id": np.array(scan_result.scan_id),
        "condition": np.array(scan_result.condition),
    }

    for s, sr in enumerate(scan_result.stages):
        # Per-channel TSI array (for histogram regeneration)
        data[f"stage{s}_tsi"] = sr.tsi_per_channel

        # Upsampled 2D activation slices (for heatmap overlays)
        if sr.mean_activation_map is not None:
            act_up = _upsample_to_input(sr.mean_activation_map, target_size)
            data[f"stage{s}_mean_act_slice"] = act_up[slice_idx]
        if sr.top_channels_map is not None:
            top_up = _upsample_to_input(sr.top_channels_map, target_size)
            data[f"stage{s}_top_act_slice"] = top_up[slice_idx]

    np.savez_compressed(output_path, **data)
    logger.info(f"Saved visualization data: {output_path}")


def load_viz_data(path: Path) -> dict[str, np.ndarray]:
    """Load saved visualization data for figure regeneration.

    Args:
        path: Path to viz_data.npz file.

    Returns:
        Dict of numpy arrays keyed by field name.
    """
    data = dict(np.load(path, allow_pickle=True))
    logger.info(f"Loaded visualization data: {path}")
    return data


def generate_panel_figure_from_saved(
    viz_data: dict[str, np.ndarray],
    config: DictConfig,
    title_suffix: str = "",
    stage_meta: dict | None = None,
) -> Figure:
    """Generate 5×3 panel figure from saved visualization data.

    Same layout as generate_panel_figure() but reads from saved NPZ
    instead of requiring the model or dataset.

    Args:
        viz_data: Dict from load_viz_data().
        config: TSI config with figure settings.
        title_suffix: Optional suffix for figure title.

    Returns:
        Matplotlib Figure.
    """
    fig_cfg = config.figure
    cmap_act = fig_cfg.colormap_activation
    cmap_sel = fig_cfg.colormap_selective
    stage_meta = _resolve_stage_meta(stage_meta)

    mri_slice = viz_data["mri_slice"]
    mask_slice = viz_data["mask_slice"]
    condition = str(viz_data["condition"])

    fig, axes = plt.subplots(
        3, 5,
        figsize=tuple(fig_cfg.figsize_main),
        constrained_layout=True,
    )

    condition_label = condition.replace("_", " ").title()
    fig.suptitle(
        f"Tumor Selectivity Index — {condition_label}{title_suffix}",
        fontsize=12,
        fontweight="bold",
    )

    for s in range(5):
        meta = stage_meta[s]
        tsi_vals = viz_data.get(f"stage{s}_tsi", np.array([]))

        # Column header
        lora_tag = " [LoRA]" if meta["has_lora"] else ""
        n_ch = meta["channels"]
        axes[0, s].set_title(
            f"Stage {s}{lora_tag}\n$C_s$={n_ch}",
            fontsize=8,
        )

        # --- Row A: Mean activation heatmap ---
        ax_a = axes[0, s]
        ax_a.imshow(mri_slice.T, cmap="gray", origin="lower", aspect="equal")
        mean_key = f"stage{s}_mean_act_slice"
        if mean_key in viz_data:
            ax_a.imshow(
                viz_data[mean_key].T,
                cmap=cmap_act, alpha=0.5, origin="lower", aspect="equal",
            )
        ax_a.contour(mask_slice.T, levels=[0.5], colors="white", linestyles="dashed", linewidths=0.8)
        ax_a.axis("off")

        # --- Row B: Top-K channels ---
        ax_b = axes[1, s]
        ax_b.imshow(mri_slice.T, cmap="gray", origin="lower", aspect="equal")
        top_key = f"stage{s}_top_act_slice"
        if top_key in viz_data:
            top_slice = viz_data[top_key]
            ax_b.imshow(top_slice.T, cmap=cmap_sel, alpha=0.5, origin="lower", aspect="equal")
            p75 = np.percentile(top_slice[top_slice > 0], 75) if (top_slice > 0).any() else 0
            if p75 > 0:
                ax_b.contour(top_slice.T, levels=[p75], colors="cyan", linewidths=0.8)
        ax_b.contour(mask_slice.T, levels=[0.5], colors="white", linestyles="dashed", linewidths=0.8)
        ax_b.axis("off")

        # --- Row C: TSI histogram ---
        ax_c = axes[2, s]
        valid = tsi_vals[~np.isnan(tsi_vals)] if len(tsi_vals) > 0 else np.array([])
        if len(valid) > 0:
            clipped = np.clip(valid, 0, 4)
            ax_c.hist(clipped, bins=20, range=(0, 4), color="#BBBBBB", edgecolor="black", linewidth=0.5)
            above_15 = valid[valid > 1.5]
            if len(above_15) > 0:
                ax_c.hist(np.clip(above_15, 0, 4), bins=20, range=(0, 4),
                          color=ADAPTED_COLOR, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax_c.axvline(1.0, color="grey", linestyle="-", linewidth=1.0)
        ax_c.axvline(1.5, color="red", linestyle="--", linewidth=1.0)
        frac_15 = np.mean(valid > 1.5) if len(valid) > 0 else 0
        ax_c.text(0.95, 0.95, f">{1.5}: {frac_15:.0%}",
                  transform=ax_c.transAxes, ha="right", va="top", fontsize=7,
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax_c.set_xlim(0, 4)
        ax_c.set_xlabel("TSI" if s == 2 else "", fontsize=8)
        ax_c.set_ylabel("Channels" if s == 0 else "", fontsize=8)
        ax_c.tick_params(labelsize=7)

    for row_idx, label in enumerate(["Mean activation", "Top-3 channels", "TSI distribution"]):
        axes[row_idx, 0].set_ylabel(label, fontsize=9, fontweight="bold")

    return fig


def _get_mri_and_mask(
    dataset: BraTSDatasetH5,
    scan_idx: int,
    mri_channel: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MRI volume and GT mask for visualization.

    Args:
        dataset: Test dataset.
        scan_idx: Dataset index.
        mri_channel: MRI channel for background (1 = T1ce).

    Returns:
        Tuple of (mri_volume [D,H,W], wt_mask [D,H,W]) as numpy arrays.
    """
    sample = dataset[scan_idx]
    mri = sample["image"][mri_channel].numpy()  # [D, H, W]
    wt_mask = (sample["seg"].squeeze(0) > 0).float().numpy()
    return mri, wt_mask


def _find_max_tumor_slice(wt_mask: np.ndarray) -> int:
    """Find the axial slice with maximum tumor cross-section.

    Args:
        wt_mask: Binary WT mask [D, H, W].

    Returns:
        Axial slice index (along dim 0).
    """
    return int(np.argmax(wt_mask.sum(axis=(1, 2))))


def _upsample_to_input(
    activation_map: np.ndarray,
    target_size: tuple[int, int, int],
) -> np.ndarray:
    """Trilinearly upsample an activation map to input resolution.

    Args:
        activation_map: [D_s, H_s, W_s] activation map.
        target_size: Target spatial dims (e.g., (192, 192, 192)).

    Returns:
        Upsampled map [D, H, W] as numpy array.
    """
    t = torch.from_numpy(activation_map).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=target_size, mode="trilinear", align_corners=False)
    return up.squeeze().numpy()


def generate_panel_figure(
    scan_result: ScanTSIResult,
    mri_volume: np.ndarray,
    wt_mask: np.ndarray,
    slice_idx: int,
    config: DictConfig,
    title_suffix: str = "",
    stage_meta: dict | None = None,
) -> Figure:
    """Generate a 5×3 panel figure for one condition.

    Row A: Mean activation heatmap overlaid on MRI.
    Row B: Top-K tumor-selective channels overlaid on MRI.
    Row C: TSI histogram per stage.

    Args:
        scan_result: TSI results for one scan under one condition.
        mri_volume: T1ce volume [D, H, W].
        wt_mask: Binary WT mask [D, H, W].
        slice_idx: Axial slice index for visualization.
        config: TSI config with figure settings.
        title_suffix: Optional suffix for figure title.

    Returns:
        Matplotlib Figure.
    """
    fig_cfg = config.figure
    cmap_act = fig_cfg.colormap_activation
    cmap_sel = fig_cfg.colormap_selective
    stage_meta = _resolve_stage_meta(stage_meta)

    target_size = mri_volume.shape
    mri_slice = mri_volume[slice_idx]
    mask_slice = wt_mask[slice_idx]

    fig, axes = plt.subplots(
        3, 5,
        figsize=tuple(fig_cfg.figsize_main),
        constrained_layout=True,
    )

    condition_label = scan_result.condition.replace("_", " ").title()
    fig.suptitle(
        f"Tumor Selectivity Index — {condition_label}{title_suffix}",
        fontsize=12,
        fontweight="bold",
    )

    for s in range(5):
        sr = scan_result.stages[s]
        meta = stage_meta[s]

        # Column header
        lora_tag = " [LoRA]" if meta["has_lora"] else ""
        axes[0, s].set_title(
            f"Stage {s}{lora_tag}\n$C_s$={meta['channels']}, {sr.resolution[0]}³",
            fontsize=8,
        )

        # --- Row A: Mean activation heatmap ---
        ax_a = axes[0, s]
        ax_a.imshow(mri_slice.T, cmap="gray", origin="lower", aspect="equal")

        if sr.mean_activation_map is not None:
            act_up = _upsample_to_input(sr.mean_activation_map, target_size)
            act_slice = act_up[slice_idx]
            ax_a.imshow(
                act_slice.T,
                cmap=cmap_act,
                alpha=0.5,
                origin="lower",
                aspect="equal",
            )

        # GT contour
        ax_a.contour(
            mask_slice.T,
            levels=[0.5],
            colors="white",
            linestyles="dashed",
            linewidths=0.8,
        )
        ax_a.axis("off")

        # --- Row B: Top-K channels ---
        ax_b = axes[1, s]
        ax_b.imshow(mri_slice.T, cmap="gray", origin="lower", aspect="equal")

        if sr.top_channels_map is not None:
            top_up = _upsample_to_input(sr.top_channels_map, target_size)
            top_slice = top_up[slice_idx]
            ax_b.imshow(
                top_slice.T,
                cmap=cmap_sel,
                alpha=0.5,
                origin="lower",
                aspect="equal",
            )

            # Contour at 75th percentile of top-channel activation
            p75 = np.percentile(top_slice[top_slice > 0], 75) if (top_slice > 0).any() else 0
            if p75 > 0:
                ax_b.contour(
                    top_slice.T,
                    levels=[p75],
                    colors="cyan",
                    linewidths=0.8,
                )

        ax_b.contour(
            mask_slice.T,
            levels=[0.5],
            colors="white",
            linestyles="dashed",
            linewidths=0.8,
        )
        ax_b.axis("off")

        # --- Row C: TSI histogram ---
        ax_c = axes[2, s]
        tsi_vals = sr.tsi_per_channel
        valid = tsi_vals[~np.isnan(tsi_vals)]

        # Clip for display range [0, 4] with overflow
        clipped = np.clip(valid, 0, 4)
        ax_c.hist(
            clipped,
            bins=20,
            range=(0, 4),
            color="#BBBBBB",
            edgecolor="black",
            linewidth=0.5,
        )

        # Fill TSI > 1.5 region
        above_15 = valid[valid > 1.5]
        if len(above_15) > 0:
            ax_c.hist(
                np.clip(above_15, 0, 4),
                bins=20,
                range=(0, 4),
                color=ADAPTED_COLOR,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

        # Reference lines
        ax_c.axvline(1.0, color="grey", linestyle="-", linewidth=1.0, label="TSI=1")
        ax_c.axvline(1.5, color="red", linestyle="--", linewidth=1.0, label="τ=1.5")

        # Annotation
        frac_15 = sr.frac_above.get(1.5, 0)
        ax_c.text(
            0.95, 0.95,
            f">{1.5}: {frac_15:.0%}",
            transform=ax_c.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        ax_c.set_xlim(0, 4)
        ax_c.set_xlabel("TSI" if s == 2 else "", fontsize=8)
        ax_c.set_ylabel("Channels" if s == 0 else "", fontsize=8)
        ax_c.tick_params(labelsize=7)

    # Row labels
    for row_idx, label in enumerate(["Mean activation", "Top-3 channels", "TSI distribution"]):
        axes[row_idx, 0].set_ylabel(label, fontsize=9, fontweight="bold")

    return fig


def generate_delta_figure(
    frozen_df: pd.DataFrame,
    adapted_df: pd.DataFrame,
    config: DictConfig,
    rank: int,
    stage_meta: dict | None = None,
) -> Figure:
    """Generate 1×5 delta summary figure (adapted − frozen mean TSI per stage).

    Args:
        frozen_df: Per-scan frozen results.
        adapted_df: Per-scan adapted results.
        config: TSI config.
        rank: LoRA rank for labeling.

    Returns:
        Matplotlib Figure.
    """
    stage_meta = _resolve_stage_meta(stage_meta)
    fig, axes = plt.subplots(
        1, 5,
        figsize=tuple(config.figure.figsize_delta),
        constrained_layout=True,
        sharey=True,
    )
    fig.suptitle(f"Δ Mean TSI (Adapted r={rank} − Frozen)", fontsize=11, fontweight="bold")

    for s in range(5):
        ax = axes[s]
        f_stage = frozen_df[frozen_df["stage"] == s].set_index("scan_id")["mean_tsi"]
        a_stage = adapted_df[adapted_df["stage"] == s].set_index("scan_id")["mean_tsi"]
        common = f_stage.index.intersection(a_stage.index)
        fv = f_stage.loc[common].values
        av = a_stage.loc[common].values
        valid = ~(np.isnan(fv) | np.isnan(av))
        delta = av[valid] - fv[valid]
        n = len(delta)

        mean_delta = np.mean(delta) if n > 0 else 0.0

        # 95% CI
        if n >= 2:
            se = np.std(delta, ddof=1) / np.sqrt(n)
            ci = scipy.stats.t.interval(0.95, df=n - 1, loc=mean_delta, scale=se)
            err = [[mean_delta - ci[0]], [ci[1] - mean_delta]]
        else:
            err = [[0], [0]]

        color = ADAPTED_COLOR if stage_meta[s]["has_lora"] else FROZEN_COLOR
        ax.bar(
            0, mean_delta,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            width=0.6,
        )
        ax.errorbar(
            0, mean_delta,
            yerr=err,
            fmt="none",
            color="black",
            capsize=4,
            linewidth=1.5,
        )
        ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)

        lora_tag = " *" if stage_meta[s]["has_lora"] else ""
        ax.set_title(f"Stage {s}{lora_tag}", fontsize=9)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)

        # Annotate delta value
        ax.text(
            0, mean_delta + (err[1][0] if mean_delta >= 0 else -err[0][0]),
            f"{mean_delta:+.3f}",
            ha="center",
            va="bottom" if mean_delta >= 0 else "top",
            fontsize=7,
            fontweight="bold",
        )

    axes[0].set_ylabel("Δ Mean TSI", fontsize=9)

    return fig


def generate_cross_rank_figure(
    frozen_df: pd.DataFrame,
    all_adapted_dfs: dict[int, pd.DataFrame],
    config: DictConfig,
    output_dir: Path,
    stage_meta: dict | None = None,
) -> None:
    """Generate cross-rank comparison figure.

    Shows Δ(mean TSI) at each stage for each rank, with error bars.
    Stages 0-2 should show Δ ≈ 0; stages 3-4 should vary by rank.

    Args:
        frozen_df: Per-scan frozen results.
        all_adapted_dfs: Dict of rank -> adapted per-scan DataFrame.
        config: TSI config.
        output_dir: Directory to save figure.
    """
    stage_meta = _resolve_stage_meta(stage_meta)
    ranks = sorted(all_adapted_dfs.keys())
    n_ranks = len(ranks)

    fig, ax = plt.subplots(
        figsize=tuple(config.figure.figsize_cross_rank),
        constrained_layout=True,
    )
    fig.suptitle("Δ Mean TSI Across LoRA Ranks", fontsize=11, fontweight="bold")

    bar_width = 0.25
    x = np.arange(5)

    for i, rank in enumerate(ranks):
        adapted_df = all_adapted_dfs[rank]
        deltas = []
        errs_lo = []
        errs_hi = []

        for s in range(5):
            f_stage = frozen_df[frozen_df["stage"] == s].set_index("scan_id")["mean_tsi"]
            a_stage = adapted_df[adapted_df["stage"] == s].set_index("scan_id")["mean_tsi"]
            common = f_stage.index.intersection(a_stage.index)
            fv = f_stage.loc[common].values
            av = a_stage.loc[common].values
            valid = ~(np.isnan(fv) | np.isnan(av))
            delta = av[valid] - fv[valid]
            n = len(delta)

            mean_d = np.mean(delta) if n > 0 else 0.0
            deltas.append(mean_d)

            if n >= 2:
                se = np.std(delta, ddof=1) / np.sqrt(n)
                ci = scipy.stats.t.interval(0.95, df=n - 1, loc=mean_d, scale=se)
                errs_lo.append(mean_d - ci[0])
                errs_hi.append(ci[1] - mean_d)
            else:
                errs_lo.append(0)
                errs_hi.append(0)

        offset = (i - (n_ranks - 1) / 2) * bar_width
        color = RANK_COLORS.get(rank, "#999999")
        ax.bar(
            x + offset,
            deltas,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=f"r={rank}",
        )
        ax.errorbar(
            x + offset,
            deltas,
            yerr=[errs_lo, errs_hi],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1.0,
        )

    ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"Stage {s}" + (" *" if stage_meta[s]["has_lora"] else "")
        for s in range(5)
    ], fontsize=9)
    ax.set_ylabel("Δ Mean TSI (Adapted − Frozen)", fontsize=9)
    ax.legend(fontsize=8, frameon=True)
    ax.tick_params(labelsize=8)

    # Save
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_path = figures_dir / "tsi_cross_rank.pdf"
    fig.savefig(save_path, dpi=config.figure.save_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cross-rank figure: {save_path}")


def generate_all_figures(
    frozen_results: list[ScanTSIResult] | None,
    adapted_results: list[ScanTSIResult],
    frozen_df: pd.DataFrame,
    adapted_df: pd.DataFrame,
    dataset: BraTSDatasetH5,
    scan_indices: list[int],
    all_scan_ids: list[str],
    test_indices: np.ndarray,
    rank: int,
    config: DictConfig,
    output_dir: Path,
    stage_meta: dict | None = None,
) -> None:
    """Generate all figures for one rank.

    Args:
        frozen_results: Frozen ScanTSIResults (None if cached; panel figure skipped).
        adapted_results: Adapted ScanTSIResults.
        frozen_df: Per-scan frozen DataFrame.
        adapted_df: Per-scan adapted DataFrame.
        dataset: Test dataset for MRI slices.
        scan_indices: Which dataset indices were used.
        all_scan_ids: All scan IDs from H5.
        test_indices: Test split indices.
        rank: LoRA rank.
        config: TSI config.
        output_dir: Rank output directory.
    """
    stage_meta = _resolve_stage_meta(stage_meta)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    mri_channel = config.figure.mri_channel
    save_fmt = config.figure.save_format
    save_dpi = config.figure.save_dpi

    # Load MRI + mask for the first scan (the one with activation maps)
    first_scan_idx = scan_indices[0]
    mri_vol, wt_mask = _get_mri_and_mask(dataset, first_scan_idx, mri_channel)

    # Determine axial slice
    slice_sel = config.figure.slice_selection
    if slice_sel == "max_tumor":
        slice_idx = _find_max_tumor_slice(wt_mask)
    elif slice_sel == "center":
        slice_idx = mri_vol.shape[0] // 2
    else:
        slice_idx = int(slice_sel)

    logger.info(f"Figure slice index: {slice_idx} (selection: {slice_sel})")

    # Panel figure: frozen condition + save viz data
    if frozen_results is not None and len(frozen_results) > 0:
        frozen_scan = frozen_results[0]
        fig_frozen = generate_panel_figure(
            frozen_scan, mri_vol, wt_mask, slice_idx, config,
            stage_meta=stage_meta,
        )
        path = figures_dir / f"tsi_frozen.{save_fmt}"
        fig_frozen.savefig(path, dpi=save_dpi, bbox_inches="tight")
        plt.close(fig_frozen)
        logger.info(f"Saved frozen panel figure: {path}")

        # Save viz data for offline figure regeneration
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        save_viz_data(frozen_scan, mri_vol, wt_mask, slice_idx,
                      data_dir / "viz_frozen.npz")

    # Panel figure: adapted condition + save viz data
    if len(adapted_results) > 0:
        adapted_scan = adapted_results[0]
        fig_adapted = generate_panel_figure(
            adapted_scan, mri_vol, wt_mask, slice_idx, config,
            title_suffix=f" (r={rank})",
            stage_meta=stage_meta,
        )
        path = figures_dir / f"tsi_adapted_r{rank}.{save_fmt}"
        fig_adapted.savefig(path, dpi=save_dpi, bbox_inches="tight")
        plt.close(fig_adapted)
        logger.info(f"Saved adapted panel figure: {path}")

        # Save viz data for offline figure regeneration
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        save_viz_data(adapted_scan, mri_vol, wt_mask, slice_idx,
                      data_dir / f"viz_adapted_r{rank}.npz")

    # Delta figure
    fig_delta = generate_delta_figure(frozen_df, adapted_df, config, rank, stage_meta=stage_meta)
    path = figures_dir / f"tsi_delta_r{rank}.{save_fmt}"
    fig_delta.savefig(path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig_delta)
    logger.info(f"Saved delta figure: {path}")
