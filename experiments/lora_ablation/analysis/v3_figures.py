#!/usr/bin/env python
# experiments/lora_ablation/v3_figures.py
"""Eleven thesis-quality figures for LoRA v3 ablation.

All figures read from the precomputed ``figure_cache/`` directory (no GPU
needed) and use the colorblind-safe Wong 2011 palette from v3_style.

Usage:
    from experiments.lora_ablation.v3_figures import generate_all_v3_figures
    generate_all_v3_figures(cache_dir, figures_dir, config)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.utils.settings import (
    V3_CONDITIONS,
    V3_SHAPE_LABELS,
    PROBE_COLORS,
    DCI_COLORS,
    DOMAIN_COLORS,
    DOMAIN_MARKERS,
    SEMANTIC_COLORS,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_color,
    get_label,
)

FIGURE_DPI = PLOT_SETTINGS["dpi_print"]
apply_v3_style = apply_ieee_style

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# ============================================================================
# Helpers
# ============================================================================

def _load_json(path: Path) -> Optional[Dict]:
    """Load JSON, returning None on failure."""
    if not path.exists():
        logger.warning(f"Cache file not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _save_fig(fig: Any, figures_dir: Path, name: str) -> None:
    """Save figure as PDF + PNG."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_dir = figures_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(figures_dir / f"{name}.pdf", dpi=FIGURE_DPI)
    fig.savefig(png_dir / f"{name}.png", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"Saved {name}.pdf + png")


def _ordered_conditions(data: Dict) -> List[str]:
    """Return conditions from data in canonical V3 order."""
    return [c for c in V3_CONDITIONS if c in data]


# ============================================================================
# Fig 1: Training Dynamics
# ============================================================================

def fig1_training_dynamics(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """Training Dynamics (3-row, 10x8).

    (a) Training loss vs epoch
    (b) Validation Dice vs epoch (vertical dashed at best_epoch)
    (c) VICReg components (var + cov) vs epoch, LoRA conditions only
    """
    logs = _load_json(cache_dir / "training_logs.json")
    if not logs:
        logger.warning("No training logs; skipping fig1.")
        return

    conditions = _ordered_conditions(logs)
    if not conditions:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # (a) Training loss
    ax = axes[0]
    for cond in conditions:
        epochs = [r.get("epoch", i) for i, r in enumerate(logs[cond])]
        losses = [r.get("train_loss", r.get("loss", None)) for r in logs[cond]]
        if any(v is not None for v in losses):
            ax.plot(epochs, losses, label=get_label(cond), color=get_color(cond), alpha=0.8)
    ax.set_ylabel("Training Loss")
    ax.set_title("(a) Training Loss")
    ax.legend(fontsize=7, ncol=2)

    # (b) Validation Dice
    ax = axes[1]
    for cond in conditions:
        epochs = [r.get("epoch", i) for i, r in enumerate(logs[cond])]
        dice = [r.get("val_dice", r.get("val_dice_mean", None)) for r in logs[cond]]
        if any(v is not None for v in dice):
            ax.plot(epochs, dice, label=get_label(cond), color=get_color(cond), alpha=0.8)
            # Best epoch marker
            valid_dice = [(e, d) for e, d in zip(epochs, dice) if d is not None]
            if valid_dice:
                best_epoch, best_dice = max(valid_dice, key=lambda x: x[1])
                ax.axvline(x=best_epoch, color=get_color(cond), linestyle="--",
                           alpha=0.3, linewidth=0.8)
    ax.set_ylabel("Validation Dice")
    ax.set_title("(b) Validation Dice")
    ax.legend(fontsize=7, ncol=2)

    # (c) VICReg components (LoRA conditions only)
    ax = axes[2]
    lora_conds = [c for c in conditions if "lora" in c]
    for cond in lora_conds:
        epochs = [r.get("epoch", i) for i, r in enumerate(logs[cond])]
        var_loss = [r.get("vicreg_var", r.get("var_loss", None)) for r in logs[cond]]
        cov_loss = [r.get("vicreg_cov", r.get("cov_loss", None)) for r in logs[cond]]
        if any(v is not None for v in var_loss):
            ax.plot(epochs, var_loss, label=f"{get_label(cond)} var",
                    color=get_color(cond), alpha=0.8, linestyle="-")
        if any(v is not None for v in cov_loss):
            ax.plot(epochs, cov_loss, label=f"{get_label(cond)} cov",
                    color=get_color(cond), alpha=0.6, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VICReg Loss")
    ax.set_title("(c) VICReg Components (LoRA only)")
    if lora_conds:
        ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig1_training_dynamics")


# ============================================================================
# Fig 2: Segmentation Quality
# ============================================================================

def _get_men_dice(dice_data: Dict, condition: str) -> Dict:
    """Extract MEN dice dict from either new or legacy format.

    New format: ``{cond: {"men": {...}, "gli": {...}}}``
    Legacy format: ``{cond: {"dice_mean": ..., ...}}``
    """
    entry = dice_data.get(condition, {})
    if "men" in entry:
        return entry["men"]
    # Legacy flat format
    return entry


def _get_gli_dice(dice_data: Dict, condition: str) -> Optional[Dict]:
    """Extract GLI dice dict (new format only)."""
    entry = dice_data.get(condition, {})
    if "gli" in entry:
        return entry["gli"]
    return None


def fig2_dice_comparison(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """Segmentation Quality (10x5) - grouped bar: TC, WT, ET Dice per condition."""
    dice_data = _load_json(cache_dir / "dice_data.json")
    if not dice_data:
        logger.warning("No dice data; skipping fig2.")
        return

    conditions = _ordered_conditions(dice_data)
    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(conditions))
    width = 0.25
    channels = ["dice_TC", "dice_WT", "dice_ET"]
    channel_labels = ["TC", "WT", "ET"]
    channel_colors = ["#0072B2", "#009E73", "#D55E00"]

    for i, (ch, ch_label, ch_color) in enumerate(zip(channels, channel_labels, channel_colors)):
        values = [_get_men_dice(dice_data, c).get(ch, 0) for c in conditions]
        ax.bar(x + i * width, values, width, label=ch_label, color=ch_color, alpha=0.85)

    ax.axhline(y=0.88, color="red", linestyle="--", alpha=0.5, label="Threshold (0.88)")
    ax.set_ylabel("Dice Score")
    ax.set_title("Test Dice by Channel (BraTS-MEN)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([get_label(c) for c in conditions], rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig2_dice_comparison")


# ============================================================================
# Fig 3: Feature Quality Dashboard
# ============================================================================

def fig3_feature_quality(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """Feature Quality Dashboard (2x2, 10x8).

    (a) Effective rank, threshold at 20
    (b) Mean inter-dim |r|, threshold at 0.30
    (c) PCA explained variance at 10/50/100 dims
    (d) Collapsed dimensions count
    """
    fq_data = _load_json(cache_dir / "feature_quality_data.json")
    if not fq_data:
        logger.warning("No feature quality data; skipping fig3.")
        return

    conditions = _ordered_conditions(fq_data)
    if not conditions:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    x = np.arange(len(conditions))
    labels = [get_label(c) for c in conditions]
    colors = [get_color(c) for c in conditions]

    # (a) Effective rank
    ax = axes[0, 0]
    ranks = [fq_data[c].get("effective_rank", 0) for c in conditions]
    ax.bar(x, ranks, color=colors, alpha=0.85)
    ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="Threshold (20)")
    ax.set_ylabel("Effective Rank")
    ax.set_title("(a) Effective Rank")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=7)

    # (b) Mean inter-dim |r|
    ax = axes[0, 1]
    corrs = [fq_data[c].get("mean_interdim_corr",
             fq_data[c].get("mean_abs_correlation", 0)) for c in conditions]
    ax.bar(x, corrs, color=colors, alpha=0.85)
    ax.axhline(y=0.30, color="red", linestyle="--", alpha=0.5, label="Threshold (0.30)")
    ax.set_ylabel("Mean |r|")
    ax.set_title("(b) Mean Inter-Dimension Correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=7)

    # (c) PCA explained variance at 10/50/100 dims
    ax = axes[1, 0]
    dims = [10, 50, 100]
    dim_colors = ["#56B4E9", "#009E73", "#D55E00"]
    bar_width = 0.25
    for i, (d, dc) in enumerate(zip(dims, dim_colors)):
        values = []
        for c in conditions:
            pca = fq_data[c].get("pca_explained_variance", {})
            values.append(pca.get(str(d), pca.get(d, 0)))
        ax.bar(x + i * bar_width, values, bar_width, label=f"{d} dims", color=dc, alpha=0.85)
    ax.set_ylabel("Cumulative Variance")
    ax.set_title("(c) PCA Explained Variance")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=7)

    # (d) Collapsed dimensions
    ax = axes[1, 1]
    collapsed = [fq_data[c].get("collapsed_dims",
                  fq_data[c].get("n_collapsed", 0)) for c in conditions]
    ax.bar(x, collapsed, color=colors, alpha=0.85)
    ax.set_ylabel("Collapsed Dims (var < 0.01)")
    ax.set_title("(d) Dimensional Collapse")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)

    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig3_feature_quality")


# ============================================================================
# Fig 4: Probe R^2 Comparison
# ============================================================================

def fig4_probe_r2(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """Probe R^2 Comparison (1x3, 12x4).

    Volume, Location, Shape grouped bars (Linear vs MLP) with delta over
    baseline annotated.
    """
    probe_data = _load_json(cache_dir / "probe_metrics.json")
    if not probe_data:
        logger.warning("No probe metrics; skipping fig4.")
        return

    conditions = _ordered_conditions(probe_data)
    if not conditions:
        return

    feature_types = ["volume", "location", "shape"]
    titles = ["Volume R\u00b2", "Location R\u00b2", "Shape R\u00b2"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Get baseline values for delta annotation
    baseline_r2 = {}
    if "baseline" in probe_data:
        for ft in feature_types:
            baseline_r2[ft] = probe_data["baseline"].get(
                f"r2_{ft}_linear", probe_data["baseline"].get(f"r2_{ft}", 0)
            ) or 0

    for ax, feat, title in zip(axes, feature_types, titles):
        linear_r2 = []
        mlp_r2 = []
        for cond in conditions:
            m = probe_data[cond]
            linear_r2.append(m.get(f"r2_{feat}_linear", m.get(f"r2_{feat}", 0)) or 0)
            mlp_r2.append(m.get(f"r2_{feat}_mlp", 0) or 0)

        x = np.arange(len(conditions))
        width = 0.35

        ax.bar(x - width / 2, linear_r2, width, label="Linear",
               color=PROBE_COLORS["linear"], alpha=0.85)
        ax.bar(x + width / 2, mlp_r2, width, label="MLP",
               color=PROBE_COLORS["mlp"], alpha=0.85)

        # Delta annotation over baseline
        base_val = baseline_r2.get(feat, 0)
        for i, cond in enumerate(conditions):
            if cond != "baseline" and cond != "baseline_frozen":
                delta = linear_r2[i] - base_val
                if abs(delta) > 0.01:
                    sign = "+" if delta > 0 else ""
                    ax.annotate(f"{sign}{delta:.2f}",
                                xy=(x[i] - width / 2, linear_r2[i]),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", fontsize=6, color="#333333")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("R\u00b2")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([get_label(c) for c in conditions], rotation=35, ha="right",
                           fontsize=7)
        ax.legend(fontsize=7)

    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig4_probe_r2")


# ============================================================================
# Fig 5: Variance Spectrum
# ============================================================================

def fig5_variance_spectrum(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """Variance Spectrum (10x4) - log-scale sorted variance, one line per condition."""
    npz_path = cache_dir / "variance_spectrum.npz"
    if not npz_path.exists():
        logger.warning("No variance spectrum cache; skipping fig5.")
        return

    data = np.load(npz_path)
    conditions = [c for c in V3_CONDITIONS if c in data]
    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for cond in conditions:
        variance = data[cond]
        ax.plot(variance, label=get_label(cond), color=get_color(cond), alpha=0.8)

    # Collapse zone
    ax.axhspan(0, 0.01, color="red", alpha=0.08, label="Collapse zone (< 0.01)")
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.4)

    ax.set_xlabel("Dimension (sorted by variance)")
    ax.set_ylabel("Variance")
    ax.set_title("Feature Variance Spectrum")
    ax.set_yscale("log")
    ax.legend(fontsize=7)

    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig5_variance_spectrum")


# ============================================================================
# Fig 6: UMAP Latent Space
# ============================================================================

def fig6_umap_latent(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """UMAP Latent Space (1x3, 15x4.5).

    (a) By condition  (b) By tumor volume  (c) By sphericity
    """
    emb_path = cache_dir / "umap_embedding.npz"
    tgt_path = cache_dir / "umap_targets.npz"
    if not emb_path.exists() or not tgt_path.exists():
        logger.warning("No UMAP cache; skipping fig6.")
        return

    emb_data = np.load(emb_path)
    tgt_data = np.load(tgt_path, allow_pickle=True)

    embedding = emb_data["embedding"]
    conditions = tgt_data["conditions"]
    volumes = tgt_data["volume"]
    shape0 = tgt_data["shape_0"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) By condition
    ax = axes[0]
    unique_conds = [c for c in V3_CONDITIONS if c in set(conditions)]
    for cond in unique_conds:
        mask = conditions == cond
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=get_color(cond), label=get_label(cond), s=8, alpha=0.6)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("(a) By Condition")
    ax.legend(fontsize=6, markerscale=2)

    # (b) By tumor volume
    ax = axes[1]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=volumes, cmap="viridis", s=8, alpha=0.6)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("(b) By Tumor Volume")
    plt.colorbar(sc, ax=ax, label="log(Volume+1)")

    # (c) By sphericity
    ax = axes[2]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=shape0, cmap="plasma", s=8, alpha=0.6)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("(c) By Sphericity")
    plt.colorbar(sc, ax=ax, label="Sphericity")

    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig6_umap_latent")


# ============================================================================
# Fig 7: Prediction Scatter
# ============================================================================

def fig7_prediction_scatter(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """Prediction Scatter (2x3, 12x8).

    Best condition. Row 1: Linear. Row 2: MLP.
    Columns: [Sphericity, Enhancement Ratio, Infiltration Index]
    """
    pred_data = _load_json(cache_dir / "predictions.json")
    if not pred_data or not pred_data.get("predictions"):
        logger.warning("No predictions; skipping fig7.")
        return

    preds = pred_data["predictions"]
    best = pred_data.get("best_condition", "?")

    # Only plot shape features (3 dims)
    if "shape" not in preds:
        logger.warning("No shape predictions; skipping fig7.")
        return

    gt = np.array(preds["shape"]["ground_truth"])
    linear = np.array(preds["shape"].get("linear", preds["shape"].get("prediction", [])))
    mlp = np.array(preds["shape"].get("mlp", []))

    n_dims = min(gt.shape[1] if gt.ndim > 1 else 1, len(V3_SHAPE_LABELS))

    fig, axes = plt.subplots(2, n_dims, figsize=(12, 8))
    if n_dims == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_dims):
        gt_col = gt[:, col] if gt.ndim > 1 else gt
        label = V3_SHAPE_LABELS[col] if col < len(V3_SHAPE_LABELS) else f"Dim {col}"

        for row, (pred_arr, probe_type) in enumerate([(linear, "Linear"), (mlp, "MLP")]):
            ax = axes[row, col]
            if pred_arr is None or len(pred_arr) == 0:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            pred_col = pred_arr[:, col] if pred_arr.ndim > 1 else pred_arr

            ax.scatter(gt_col, pred_col, alpha=0.3, s=10,
                       color=PROBE_COLORS.get(probe_type.lower(), "#333333"))

            # Identity line
            lims = [min(gt_col.min(), pred_col.min()),
                    max(gt_col.max(), pred_col.max())]
            ax.plot(lims, lims, "k--", alpha=0.4)

            # R^2 and RMSE
            ss_res = np.sum((gt_col - pred_col) ** 2)
            ss_tot = np.sum((gt_col - gt_col.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((gt_col - pred_col) ** 2))
            ax.text(0.05, 0.92, f"R\u00b2={r2:.3f}\nRMSE={rmse:.3f}",
                    transform=ax.transAxes, fontsize=8, va="top")

            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{probe_type} Prediction")
            if row == 1:
                ax.set_xlabel("Ground Truth")

    fig.suptitle(f"Predictions vs Ground Truth ({get_label(best)})", y=1.01)
    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig7_prediction_scatter")


# ============================================================================
# Fig 8: DCI Disentanglement
# ============================================================================

def fig8_dci_scores(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """DCI Disentanglement (10x5) - grouped bars: D, C, I per condition."""
    fq_data = _load_json(cache_dir / "feature_quality_data.json")
    if not fq_data:
        logger.warning("No feature quality data; skipping fig8.")
        return

    conditions = _ordered_conditions(fq_data)
    if not conditions:
        return

    # Check if DCI data exists
    has_dci = any(
        "dci_disentanglement" in fq_data[c] or "dci_D" in fq_data[c]
        for c in conditions
    )
    if not has_dci:
        logger.warning("No DCI scores in feature quality data; skipping fig8.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(conditions))
    width = 0.25
    dci_keys = [
        ("D", "dci_disentanglement", "dci_D"),
        ("C", "dci_completeness", "dci_C"),
        ("I", "dci_informativeness", "dci_I"),
    ]

    for i, (label, key1, key2) in enumerate(dci_keys):
        values = []
        for c in conditions:
            v = fq_data[c].get(key1, fq_data[c].get(key2, 0)) or 0
            values.append(v)
        ax.bar(x + i * width, values, width, label=f"DCI-{label}",
               color=DCI_COLORS[label], alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("DCI Disentanglement Scores")
    ax.set_xticks(x + width)
    ax.set_xticklabels([get_label(c) for c in conditions], rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig8_dci_scores")


# ============================================================================
# Fig 9: Domain UMAP Grid
# ============================================================================

def fig9_domain_umap_grid(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """Domain UMAP Grid (1xN) - GLI vs MEN scatter in shared UMAP space.

    One panel per condition, with silhouette score annotation.
    """
    npz_path = cache_dir / "domain_umap_grid.npz"
    if not npz_path.exists():
        logger.warning("No domain UMAP grid cache; skipping fig9.")
        return

    data = np.load(npz_path, allow_pickle=True)
    embedding = data["embedding"]
    domains = data["domains"]
    conditions = data["conditions"]

    unique_conds = [c for c in V3_CONDITIONS if c in set(conditions)]
    if not unique_conds:
        return

    n_panels = len(unique_conds)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.5 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    panel_labels = "abcdefghijklmnop"

    for i, cond in enumerate(unique_conds):
        ax = axes[i]
        mask = conditions == cond

        cond_emb = embedding[mask]
        cond_domains = domains[mask]

        # GLI points
        gli_mask = cond_domains == "glioma"
        ax.scatter(
            cond_emb[gli_mask, 0], cond_emb[gli_mask, 1],
            c=DOMAIN_COLORS["glioma"], marker="o", s=12, alpha=0.5,
            linewidths=0.3, edgecolors="none", label="GLI",
        )
        # MEN points
        men_mask = cond_domains == "meningioma"
        ax.scatter(
            cond_emb[men_mask, 0], cond_emb[men_mask, 1],
            c=DOMAIN_COLORS["meningioma"], marker="^", s=12, alpha=0.5,
            linewidths=0.3, edgecolors="none", label="MEN",
        )

        # Silhouette annotation
        sil_key = f"silhouette_{cond}"
        if sil_key in data:
            sil = float(data[sil_key])
            ax.text(
                0.95, 0.05, f"S = {sil:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=PLOT_SETTINGS["annotation_fontsize"],
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
            )

        # Panel label
        label_char = panel_labels[i] if i < len(panel_labels) else str(i)
        ax.set_title(f"({label_char}) {get_label(cond)}", fontsize=PLOT_SETTINGS["axes_titlesize"])

        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.set_ylabel("UMAP 2")
        if i == n_panels - 1:
            ax.legend(fontsize=7, markerscale=1.5, loc="upper right")

    fig.text(0.5, 0.01, "UMAP 1", ha="center", fontsize=PLOT_SETTINGS["axes_labelsize"])
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save_fig(fig, figures_dir, "fig9_domain_umap_grid")


# ============================================================================
# Fig 10: Semantic Probe R² Summary
# ============================================================================

def fig10_semantic_summary(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """Semantic Probe R² Summary (7x4) - grouped bar chart across all conditions.

    vol/loc/shape R² with solid (Linear) and hatched (MLP) bars, delta-over-frozen
    annotation on LoRA conditions.
    """
    probe_data = _load_json(cache_dir / "probe_metrics.json")
    if not probe_data:
        logger.warning("No probe metrics; skipping fig10.")
        return

    conditions = _ordered_conditions(probe_data)
    if not conditions:
        return

    feature_types = ["volume", "location", "shape"]
    feat_colors = [SEMANTIC_COLORS["volume"], SEMANTIC_COLORS["location"], SEMANTIC_COLORS["shape"]]
    feat_labels = ["Volume", "Location", "Shape"]

    n_feat = len(feature_types)
    n_cond = len(conditions)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    # 6 bars per condition group: 3 linear (solid) + 3 MLP (hatched)
    bar_w = PLOT_SETTINGS["bar_width"]
    group_width = (2 * n_feat + 1) * bar_w  # gap between linear and MLP sub-groups

    x_centers = np.arange(n_cond) * group_width * 1.3

    # Get frozen baseline values for delta annotation
    frozen_r2 = {}
    frozen_key = "baseline_frozen" if "baseline_frozen" in probe_data else None
    if frozen_key:
        for ft in feature_types:
            frozen_r2[ft] = probe_data[frozen_key].get(
                f"r2_{ft}_linear", probe_data[frozen_key].get(f"r2_{ft}", 0)
            ) or 0

    for j, (ft, fc, fl) in enumerate(zip(feature_types, feat_colors, feat_labels)):
        linear_vals = []
        mlp_vals = []
        for cond in conditions:
            m = probe_data[cond]
            linear_vals.append(m.get(f"r2_{ft}_linear", m.get(f"r2_{ft}", 0)) or 0)
            mlp_vals.append(m.get(f"r2_{ft}_mlp", 0) or 0)

        # Linear bars (solid)
        offset_lin = (j - n_feat / 2 + 0.5) * bar_w
        bars_lin = ax.bar(
            x_centers + offset_lin, linear_vals, bar_w,
            color=fc, alpha=0.85, label=f"{fl} (Lin)" if j < n_feat else None,
        )

        # MLP bars (hatched, offset by n_feat*bar_w + gap)
        offset_mlp = offset_lin + n_feat * bar_w + bar_w * 0.5
        bars_mlp = ax.bar(
            x_centers + offset_mlp, mlp_vals, bar_w,
            color=fc, alpha=0.55, hatch="//",
            label=f"{fl} (MLP)" if j < n_feat else None,
        )

        # Delta-over-frozen annotation for LoRA conditions (linear only)
        if frozen_key:
            base_val = frozen_r2.get(ft, 0)
            for k, cond in enumerate(conditions):
                if cond in (frozen_key, "baseline"):
                    continue
                delta = linear_vals[k] - base_val
                if abs(delta) > 0.01:
                    sign = "+" if delta > 0 else ""
                    ax.annotate(
                        f"{sign}{delta:.2f}",
                        xy=(x_centers[k] + offset_lin, linear_vals[k]),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=5, color="#333333",
                    )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel(r"R$^2$")
    ax.set_title("Semantic Probe R\u00b2 (Linear + MLP)")
    ax.set_xticks(x_centers + n_feat * bar_w * 0.25)
    ax.set_xticklabels([get_label(c) for c in conditions], rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=7, ncol=2, loc="upper left")

    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig10_semantic_summary")


# ============================================================================
# Fig 11: Dice + Generalization
# ============================================================================

def fig11_dice_generalization(
    cache_dir: Path,
    figures_dir: Path,
) -> None:
    """Dice + Generalization (7x5.5) - two stacked panels.

    Top: MEN vs GLI Dice grouped bars.
    Bottom: Retention ratio (GLI/MEN) with dashed line at 1.0.
    """
    dice_data = _load_json(cache_dir / "dice_data.json")
    if not dice_data:
        logger.warning("No dice data; skipping fig11.")
        return

    # Need at least one condition with both MEN and GLI dice
    conditions = _ordered_conditions(dice_data)
    has_gli = [c for c in conditions if _get_gli_dice(dice_data, c) is not None]
    if not has_gli:
        logger.warning("No GLI dice data available; skipping fig11.")
        return

    conditions_to_plot = [c for c in conditions if c in has_gli or _get_men_dice(dice_data, c)]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.0, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    x = np.arange(len(conditions_to_plot))
    width = 0.35

    men_means = []
    men_stds = []
    gli_means = []
    gli_stds = []
    retention = []

    for cond in conditions_to_plot:
        men = _get_men_dice(dice_data, cond)
        gli = _get_gli_dice(dice_data, cond)

        m_mean = men.get("dice_mean", 0) if men else 0
        m_std = men.get("dice_std", 0) if men else 0
        g_mean = gli.get("dice_mean", 0) if gli else 0
        g_std = gli.get("dice_std", 0) if gli else 0

        men_means.append(m_mean)
        men_stds.append(m_std)
        gli_means.append(g_mean)
        gli_stds.append(g_std)
        retention.append(g_mean / m_mean if m_mean > 0 else 0)

    # Top panel: grouped bars
    ax_top.bar(
        x - width / 2, men_means, width, yerr=men_stds,
        label="BraTS-MEN", color=DOMAIN_COLORS["meningioma"],
        alpha=0.85, capsize=PLOT_SETTINGS["errorbar_capsize"],
    )
    ax_top.bar(
        x + width / 2, gli_means, width, yerr=gli_stds,
        label="BraTS-GLI", color=DOMAIN_COLORS["glioma"],
        alpha=0.85, capsize=PLOT_SETTINGS["errorbar_capsize"],
    )
    ax_top.set_ylabel("Mean Dice")
    ax_top.set_title("(a) In-Domain vs Out-of-Domain Dice")
    ax_top.set_ylim(0.5, 1.0)
    ax_top.legend(fontsize=8)

    # Bottom panel: retention ratio
    bar_colors = [get_color(c) for c in conditions_to_plot]
    ax_bot.bar(x, retention, width * 2, color=bar_colors, alpha=0.85)
    ax_bot.axhline(y=1.0, color="black", linestyle="--", alpha=0.6, linewidth=0.8)
    ax_bot.set_ylabel("Retention (GLI / MEN)")
    ax_bot.set_title("(b) Generalization Ratio")

    # Value annotations
    for i, (r, cond) in enumerate(zip(retention, conditions_to_plot)):
        if r > 0:
            ax_bot.text(
                x[i], r + 0.01, f"{r:.2f}",
                ha="center", va="bottom",
                fontsize=PLOT_SETTINGS["annotation_fontsize"],
            )

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([get_label(c) for c in conditions_to_plot], rotation=30, ha="right")

    plt.tight_layout()
    _save_fig(fig, figures_dir, "fig11_dice_generalization")


# ============================================================================
# Master function
# ============================================================================

def generate_all_v3_figures(
    cache_dir: Path,
    figures_dir: Path,
    config: Optional[dict] = None,
) -> None:
    """Generate all 11 thesis-quality figures from cached data.

    Args:
        cache_dir: Path to ``results/figure_cache/`` with precomputed data.
        figures_dir: Output directory for figures (PDF + PNG).
        config: Optional experiment config (unused, reserved for future use).
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available; cannot generate figures.")
        return

    apply_v3_style()
    cache_dir = Path(cache_dir)
    figures_dir = Path(figures_dir)

    logger.info("Generating 11 v3 figures...")

    fig1_training_dynamics(cache_dir, figures_dir)
    fig2_dice_comparison(cache_dir, figures_dir)
    fig3_feature_quality(cache_dir, figures_dir)
    fig4_probe_r2(cache_dir, figures_dir)
    fig5_variance_spectrum(cache_dir, figures_dir)
    fig6_umap_latent(cache_dir, figures_dir)
    fig7_prediction_scatter(cache_dir, figures_dir)
    fig8_dci_scores(cache_dir, figures_dir)
    fig9_domain_umap_grid(cache_dir, figures_dir)
    fig10_semantic_summary(cache_dir, figures_dir)
    fig11_dice_generalization(cache_dir, figures_dir)

    logger.info(f"All 11 figures saved to {figures_dir}")
