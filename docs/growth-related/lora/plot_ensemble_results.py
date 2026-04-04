#!/usr/bin/env python
"""Plotting suite for LoRA-Ensemble uncertainty segmentation results.

Generates publication-quality figures from the evaluation CSVs and JSONs
produced by the uncertainty_segmentation module.

Usage:
    python plot_ensemble_results.py /path/to/r8_M10_s42/ --output ./figures/
    python plot_ensemble_results.py /path/to/r8_M10_s42/ --dpi 300 --format pdf

Structure follows a population → individual hierarchy:
    Fig 1: Training dynamics (all members)
    Fig 2: Segmentation performance comparison (box + statistical annotations)
    Fig 3: Paired comparison — ensemble vs baseline (scatter + histogram)
    Fig 4: Per-member forest plot (individual CIs)
    Fig 5: LLN convergence (running mean ± CI)
    Fig 6: Calibration reliability diagram
    Fig 7: Best / worst case analysis
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Global style
# =============================================================================

# Subdued, print-friendly palette (colorblind-safe, Okabe-Ito inspired)
C_BASELINE = "#999999"    # grey
C_ENSEMBLE = "#0072B2"    # blue
C_MEMBERS  = "#E69F00"    # amber
C_BEST     = "#009E73"    # teal
C_DELTA_POS = "#0072B2"
C_DELTA_NEG = "#D55E00"   # vermillion
C_FILL     = "#0072B2"
MEMBER_CMAP = plt.cm.Set3

def _setup_style() -> None:
    """Configure matplotlib for publication figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,   # TrueType (editable in Illustrator)
        "ps.fonttype": 42,
    })


def _significance_label(p: float) -> str:
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def _add_bracket(ax, x1, x2, y, h, text, fontsize=7):
    """Draw a significance bracket between two positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c="k")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


# =============================================================================
# FIGURE 1: Training Dynamics
# =============================================================================

def plot_training_curves(curves_df: pd.DataFrame, ax_loss=None, ax_dice=None):
    """Training loss and validation Dice over epochs (mean ± std ribbon)."""
    if ax_loss is None or ax_dice is None:
        fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(7, 2.8))

    epochs = curves_df["epoch"].values

    # Loss
    ax_loss.fill_between(
        epochs,
        curves_df["train_loss_mean"] - curves_df["train_loss_std"],
        curves_df["train_loss_mean"] + curves_df["train_loss_std"],
        alpha=0.2, color=C_MEMBERS, label=None,
    )
    ax_loss.plot(epochs, curves_df["train_loss_mean"], color=C_MEMBERS,
                 lw=1.2, label="Train loss")
    ax_loss.fill_between(
        epochs,
        curves_df["val_loss_mean"] - curves_df["val_loss_std"],
        curves_df["val_loss_mean"] + curves_df["val_loss_std"],
        alpha=0.2, color=C_ENSEMBLE,
    )
    ax_loss.plot(epochs, curves_df["val_loss_mean"], color=C_ENSEMBLE,
                 lw=1.2, label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (Dice + CE)")
    ax_loss.legend(frameon=False)
    ax_loss.set_title("a) Training loss", loc="left", fontweight="bold")

    # Dice WT
    ax_dice.fill_between(
        epochs,
        curves_df["val_dice_wt_mean"] - curves_df["val_dice_wt_std"],
        curves_df["val_dice_wt_mean"] + curves_df["val_dice_wt_std"],
        alpha=0.2, color=C_ENSEMBLE,
    )
    ax_dice.plot(epochs, curves_df["val_dice_wt_mean"], color=C_ENSEMBLE,
                 lw=1.2, label="WT")
    ax_dice.fill_between(
        epochs,
        curves_df["val_dice_tc_mean"] - curves_df["val_dice_tc_std"],
        curves_df["val_dice_tc_mean"] + curves_df["val_dice_tc_std"],
        alpha=0.15, color=C_BEST,
    )
    ax_dice.plot(epochs, curves_df["val_dice_tc_mean"], color=C_BEST,
                 lw=1.0, ls="--", label="TC")
    ax_dice.fill_between(
        epochs,
        curves_df["val_dice_et_mean"] - curves_df["val_dice_et_std"],
        curves_df["val_dice_et_mean"] + curves_df["val_dice_et_std"],
        alpha=0.15, color=C_DELTA_NEG,
    )
    ax_dice.plot(epochs, curves_df["val_dice_et_mean"], color=C_DELTA_NEG,
                 lw=1.0, ls=":", label="ET")
    ax_dice.set_xlabel("Epoch")
    ax_dice.set_ylabel("Validation Dice")
    ax_dice.set_ylim(0, 1)
    ax_dice.legend(frameon=False, ncol=3)
    ax_dice.set_title("b) Validation Dice (mean ± std across M members)",
                      loc="left", fontweight="bold")


# =============================================================================
# FIGURE 2: Segmentation Performance Comparison
# =============================================================================

def plot_performance_comparison(
    per_member: pd.DataFrame,
    ensemble: pd.DataFrame,
    baseline: pd.DataFrame,
    stats: dict,
    ax=None,
):
    """Box plots: Baseline vs Members (pooled) vs Ensemble, for WT Dice.
    
    Annotated with statistical significance brackets.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Prepare data
    baseline_wt = baseline["dice_wt"].values
    member_wt = per_member["dice_wt"].values
    ensemble_wt = ensemble["dice_wt"].values

    data = [baseline_wt, member_wt, ensemble_wt]
    positions = [0, 1, 2]
    colors = [C_BASELINE, C_MEMBERS, C_ENSEMBLE]
    labels = [
        f"Frozen BSF\n(n={len(baseline_wt)})",
        f"Individual\nmembers\n(n={len(member_wt)})",
        f"Ensemble\n(n={len(ensemble_wt)})",
    ]

    bp = ax.boxplot(
        data, positions=positions, widths=0.5, patch_artist=True,
        showfliers=False, medianprops=dict(color="k", lw=1.2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Overlay individual points (jittered)
    rng = np.random.RandomState(42)
    for pos, d, c in zip(positions, data, colors):
        jitter = rng.uniform(-0.15, 0.15, size=len(d))
        ax.scatter(pos + jitter, d, s=6, alpha=0.3, color=c, edgecolors="none", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Dice (Whole Tumor)")
    ax.set_ylim(-0.05, 1.05)

    # Statistical brackets
    evb = stats["ensemble_vs_baseline"]["wt"]
    p_val = evb["p_value_wilcoxon"]
    d_val = evb["cohens_d"]
    sig = _significance_label(p_val)

    y_top = max(np.percentile(ensemble_wt, 95), np.percentile(baseline_wt, 95)) + 0.02
    _add_bracket(ax, 0, 2, y_top, 0.03,
                 f"{sig}  d={d_val:.2f}", fontsize=7)

    ax.set_title("Whole Tumor Dice: Baseline vs Ensemble", fontweight="bold", fontsize=10)


# =============================================================================
# FIGURE 3: Paired Comparison (Scatter + Histogram)
# =============================================================================

def plot_paired_comparison(
    ensemble: pd.DataFrame,
    baseline: pd.DataFrame,
    paired: pd.DataFrame,
    stats: dict,
    axes=None,
):
    """Panel A: Scatter (baseline vs ensemble). Panel B: ΔDice histogram."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    ax_scatter, ax_hist = axes

    # Merge on scan_id
    merged = baseline.merge(ensemble, on="scan_id", suffixes=("_bas", "_ens"))
    x = merged["dice_wt_bas"].values
    y = merged["dice_wt_ens"].values

    # --- Panel A: Scatter ---
    colors_sc = np.where(y > x, C_DELTA_POS, C_DELTA_NEG)
    ax_scatter.scatter(x, y, s=14, c=colors_sc, alpha=0.6, edgecolors="none", zorder=3)
    lims = [-0.05, 1.05]
    ax_scatter.plot(lims, lims, "k--", lw=0.7, alpha=0.5, label="Identity")
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel("Baseline Dice (WT)")
    ax_scatter.set_ylabel("Ensemble Dice (WT)")
    ax_scatter.set_aspect("equal")

    # Count improvements
    n_better = (y > x).sum()
    n_worse = (y < x).sum()
    n_equal = (y == x).sum()
    ax_scatter.text(
        0.05, 0.92,
        f"Improved: {n_better}/{len(x)}\nWorsened: {n_worse}/{len(x)}",
        transform=ax_scatter.transAxes, fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax_scatter.set_title("a) Per-subject comparison", loc="left", fontweight="bold")

    # --- Panel B: Paired differences histogram ---
    deltas = paired["dice_wt_delta"].values

    evb = stats["ensemble_vs_baseline"]["wt"]
    delta_mean = evb["delta"]
    ci_lo = evb["ci_95_lower"]
    ci_hi = evb["ci_95_upper"]
    p_val = evb["p_value_wilcoxon"]
    d_val = evb["cohens_d"]

    ax_hist.hist(deltas, bins=30, color=C_FILL, alpha=0.5, edgecolor="white", lw=0.5)
    ax_hist.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
    ax_hist.axvline(delta_mean, color=C_ENSEMBLE, lw=1.5, label=f"Mean Δ = {delta_mean:.3f}")

    # CI band
    ax_hist.axvspan(ci_lo, ci_hi, alpha=0.15, color=C_ENSEMBLE, label="95% CI")

    ax_hist.set_xlabel("ΔDice (Ensemble − Baseline)")
    ax_hist.set_ylabel("Count")

    sig = _significance_label(p_val)
    ax_hist.text(
        0.95, 0.92,
        f"p = {p_val:.1e} {sig}\nd = {d_val:.2f}",
        transform=ax_hist.transAxes, fontsize=7,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax_hist.legend(frameon=False, fontsize=7, loc="upper left")
    ax_hist.set_title("b) Paired differences (WT Dice)", loc="left", fontweight="bold")


# =============================================================================
# FIGURE 4: Per-Member Forest Plot
# =============================================================================

def plot_forest(stats: dict, ax=None):
    """Forest plot: per-member WT Dice mean ± 95% CI, ensemble + baseline refs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))

    members = stats["per_member_summary"]
    M = len(members)

    y_positions = list(range(M))
    means = [m["dice_wt_mean"] for m in members]
    ci_lo = [m["dice_wt_ci95"][0] for m in members]
    ci_hi = [m["dice_wt_ci95"][1] for m in members]
    labels = [f"Member {m['member_id']}" for m in members]

    # Per-member CIs
    for i, (mean, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
        ax.plot([lo, hi], [i, i], color=C_MEMBERS, lw=1.5, solid_capstyle="round")
        ax.plot(mean, i, "o", color=C_MEMBERS, ms=5, zorder=4)

    # Ensemble reference band
    evb = stats["ensemble_vs_baseline"]["wt"]
    ens_mean = evb["ensemble_mean"]
    ens_ci = evb.get("ensemble_ci95", [ens_mean, ens_mean])
    ax.axvspan(ens_ci[0], ens_ci[1], alpha=0.12, color=C_ENSEMBLE, zorder=1)
    ax.axvline(ens_mean, color=C_ENSEMBLE, lw=1.2, ls="-", label="Ensemble", zorder=2)

    # Baseline reference
    bas_mean = evb["baseline_mean"]
    ax.axvline(bas_mean, color=C_BASELINE, lw=1.2, ls="--", label="Baseline", zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Dice (WT)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # ICC annotation
    icc = stats["inter_member_agreement"]["icc_wt"]
    ax.text(
        0.02, 0.02,
        f"ICC(3,1) = {icc:.3f}",
        transform=ax.transAxes, fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )
    ax.set_title("Per-member WT Dice (mean ± 95% CI)", fontweight="bold")


# =============================================================================
# FIGURE 5: LLN Convergence
# =============================================================================

def plot_convergence(conv_df: pd.DataFrame, metric_name: str = "Dice (WT)", ax=None):
    """Running mean ± CI as function of ensemble size k, averaged over scans."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3))

    # Aggregate across scans: for each k, mean of running_mean and running_se
    grouped = conv_df[conv_df["k"] >= 2].groupby("k").agg(
        mean_of_means=("running_mean", "mean"),
        mean_se=("running_se", "mean"),
        std_se=("running_se", "std"),
    ).reset_index()

    k = grouped["k"].values
    y = grouped["mean_of_means"].values
    se = grouped["mean_se"].values

    ax.fill_between(k, y - 1.96 * se, y + 1.96 * se, alpha=0.2, color=C_ENSEMBLE)
    ax.plot(k, y, "o-", color=C_ENSEMBLE, ms=4, lw=1.2, label="Running mean ± 1.96 SE")

    # Theoretical 1/√k curve (normalized to match SE at k=2)
    if len(se) > 0:
        se_at_2 = se[0]
        k_theory = np.arange(2, k.max() + 1)
        se_theory = se_at_2 * np.sqrt(2) / np.sqrt(k_theory)
        ax.plot(k_theory, y[0] + 1.96 * se_theory, ":", color=C_BASELINE, lw=0.8)
        ax.plot(k_theory, y[0] - 1.96 * se_theory, ":", color=C_BASELINE, lw=0.8,
                label=r"Theoretical $\propto 1/\sqrt{k}$")

    ax.set_xlabel("Ensemble size (k)")
    ax.set_ylabel(metric_name)
    ax.set_xticks(range(2, int(k.max()) + 1))
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"Convergence of {metric_name}", fontweight="bold")


# =============================================================================
# FIGURE 6: Reliability Diagram
# =============================================================================

def plot_reliability(calibration: dict, ax=None):
    """Reliability diagram with gap shading and ECE annotation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

    rel = calibration["reliability"]
    bin_edges = np.array(rel["bin_edges"])
    bin_acc = np.array(rel["bin_accuracy"])
    bin_conf = np.array(rel["bin_confidence"])
    bin_count = np.array(rel["bin_count"])

    # Filter bins with > 50 samples (exclude near-empty bins)
    mask = bin_count > 50
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5, label="Perfect")

    # Bar chart
    width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers, bin_acc, width=width * 0.85, alpha=0.5, color=C_ENSEMBLE,
           edgecolor="white", lw=0.3, label="Observed")

    # Gap fill (miscalibration)
    for i in range(len(bin_acc)):
        if bin_count[i] > 50:
            color = C_DELTA_NEG if bin_acc[i] < bin_conf[i] else C_BEST
            ax.fill_between(
                [bin_centers[i] - width / 2, bin_centers[i] + width / 2],
                bin_acc[i], bin_conf[i],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ece = calibration["ece"]
    brier = calibration["brier_score"]
    ax.text(
        0.05, 0.92,
        f"ECE = {ece:.4f}\nBrier = {brier:.4f}",
        transform=ax.transAxes, fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Calibration (reliability diagram)", fontweight="bold")


# =============================================================================
# FIGURE 7: Best / Worst Case Analysis
# =============================================================================

def plot_best_worst(
    ensemble: pd.DataFrame,
    baseline: pd.DataFrame,
    per_member: pd.DataFrame,
    n_cases: int = 5,
    ax=None,
):
    """Horizontal bar chart showing N best and N worst improvement cases.
    
    For each case: baseline Dice (grey) and ensemble Dice (blue) side by side,
    with ΔDice annotation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    merged = baseline.merge(ensemble, on="scan_id", suffixes=("_bas", "_ens"))
    merged["delta_wt"] = merged["dice_wt_ens"] - merged["dice_wt_bas"]
    merged = merged.sort_values("delta_wt")

    worst = merged.head(n_cases)
    best = merged.tail(n_cases).iloc[::-1]
    cases = pd.concat([best, worst])

    y_pos = np.arange(len(cases))
    h = 0.35

    ax.barh(y_pos - h / 2, cases["dice_wt_bas"].values, h,
            color=C_BASELINE, alpha=0.6, label="Baseline")
    ax.barh(y_pos + h / 2, cases["dice_wt_ens"].values, h,
            color=C_ENSEMBLE, alpha=0.6, label="Ensemble")

    # Delta annotations
    for i, (_, row) in enumerate(cases.iterrows()):
        delta = row["delta_wt"]
        color = C_DELTA_POS if delta > 0 else C_DELTA_NEG
        sign = "+" if delta > 0 else ""
        x_pos = max(row["dice_wt_bas"], row["dice_wt_ens"]) + 0.02
        ax.text(x_pos, i, f"{sign}{delta:.3f}", va="center", fontsize=7, color=color)

    # Separator line between best and worst
    ax.axhline(n_cases - 0.5, color="k", ls=":", lw=0.5, alpha=0.5)
    ax.text(-0.02, n_cases / 2 - 0.5, "Best Δ", ha="right", va="center",
            fontsize=7, color=C_BEST, fontweight="bold", transform=ax.get_yaxis_transform())
    ax.text(-0.02, n_cases + n_cases / 2 - 0.5, "Worst Δ", ha="right", va="center",
            fontsize=7, color=C_DELTA_NEG, fontweight="bold", transform=ax.get_yaxis_transform())

    # Truncate scan IDs for readability
    short_ids = [s.replace("BraTS-MEN-", "") for s in cases["scan_id"].values]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_ids, fontsize=7)
    ax.set_xlabel("Dice (WT)")
    ax.set_xlim(0, 1.15)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.invert_yaxis()
    ax.set_title(f"Top-{n_cases} best and worst improvements", fontweight="bold")


# =============================================================================
# FIGURE 8: Multi-panel Dice by Sub-compartment
# =============================================================================

def plot_dice_by_compartment(
    ensemble: pd.DataFrame,
    baseline: pd.DataFrame,
    stats: dict,
    ax=None,
):
    """Grouped bar chart: Baseline vs Ensemble for TC, WT, ET with CI error bars."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3))

    compartments = ["tc", "wt", "et"]
    labels = ["TC", "WT", "ET"]
    x = np.arange(len(compartments))
    width = 0.3

    bas_means = []
    ens_means = []
    bas_cis = []
    ens_cis = []

    for comp in compartments:
        evb = stats["ensemble_vs_baseline"][comp]
        bas_means.append(evb["baseline_mean"])
        ens_means.append(evb["ensemble_mean"])
        bas_cis.append([
            evb["baseline_mean"] - evb["baseline_ci95"][0],
            evb["baseline_ci95"][1] - evb["baseline_mean"],
        ])
        ens_cis.append([
            evb["ensemble_mean"] - evb["ensemble_ci95"][0],
            evb["ensemble_ci95"][1] - evb["ensemble_mean"],
        ])

    bas_err = np.array(bas_cis).T
    ens_err = np.array(ens_cis).T

    ax.bar(x - width / 2, bas_means, width, yerr=bas_err, color=C_BASELINE,
           alpha=0.6, label="Baseline", capsize=3, error_kw=dict(lw=0.8))
    ax.bar(x + width / 2, ens_means, width, yerr=ens_err, color=C_ENSEMBLE,
           alpha=0.6, label="Ensemble", capsize=3, error_kw=dict(lw=0.8))

    # Significance annotations
    for i, comp in enumerate(compartments):
        evb = stats["ensemble_vs_baseline"][comp]
        p = evb["p_value_wilcoxon"]
        d = evb["cohens_d"]
        sig = _significance_label(p)
        y_max = max(ens_means[i], bas_means[i]) + max(ens_err[1][i], bas_err[1][i]) + 0.03
        _add_bracket(ax, i - width / 2, i + width / 2, y_max, 0.02,
                     f"{sig} d={d:.2f}", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Dice by sub-compartment (mean ± 95% CI)", fontweight="bold")


# =============================================================================
# COMPOSITE FIGURE GENERATOR
# =============================================================================

def generate_all_figures(
    eval_dir: Path,
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Load all data and generate every figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    curves = pd.read_csv(eval_dir / "aggregated_training_curves.csv")
    per_member = pd.read_csv(eval_dir / "per_member_test_dice.csv")
    ensemble = pd.read_csv(eval_dir / "ensemble_test_dice.csv")
    baseline = pd.read_csv(eval_dir / "baseline_test_dice.csv")
    paired = pd.read_csv(eval_dir / "paired_differences.csv")
    conv_wt = pd.read_csv(eval_dir / "convergence_dice_wt.csv")
    conv_tc = pd.read_csv(eval_dir / "convergence_dice_tc.csv")
    conv_et = pd.read_csv(eval_dir / "convergence_dice_et.csv")

    with open(eval_dir / "statistical_summary.json") as f:
        stats = json.load(f)
    with open(eval_dir / "calibration.json") as f:
        calibration = json.load(f)

    logger.info(f"Loaded data: {len(per_member)} per-member rows, "
                f"{len(ensemble)} ensemble rows, {len(baseline)} baseline rows")

    _setup_style()

    # --- Figure 1: Training dynamics ---
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(7, 2.8))
    plot_training_curves(curves, ax_loss=ax1a, ax_dice=ax1b)
    fig1.tight_layout()
    fig1.savefig(output_dir / f"fig1_training_curves.{fmt}", dpi=dpi)
    logger.info("Saved fig1_training_curves")

    # --- Figure 2: Performance comparison (box plot) ---
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
    plot_performance_comparison(per_member, ensemble, baseline, stats, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(output_dir / f"fig2_performance_comparison.{fmt}", dpi=dpi)
    logger.info("Saved fig2_performance_comparison")

    # --- Figure 3: Paired comparison ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(7, 3.2))
    plot_paired_comparison(ensemble, baseline, paired, stats, axes=axes3)
    fig3.tight_layout()
    fig3.savefig(output_dir / f"fig3_paired_comparison.{fmt}", dpi=dpi)
    logger.info("Saved fig3_paired_comparison")

    # --- Figure 4: Forest plot ---
    fig4, ax4 = plt.subplots(figsize=(4.5, 3.5))
    plot_forest(stats, ax=ax4)
    fig4.tight_layout()
    fig4.savefig(output_dir / f"fig4_forest_plot.{fmt}", dpi=dpi)
    logger.info("Saved fig4_forest_plot")

    # --- Figure 5: LLN Convergence (3-panel) ---
    fig5, axes5 = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, conv_data, name in zip(axes5, [conv_wt, conv_tc, conv_et],
                                    ["Dice (WT)", "Dice (TC)", "Dice (ET)"]):
        plot_convergence(conv_data, metric_name=name, ax=ax)
    fig5.tight_layout()
    fig5.savefig(output_dir / f"fig5_convergence.{fmt}", dpi=dpi)
    logger.info("Saved fig5_convergence")

    # --- Figure 6: Calibration ---
    fig6, ax6 = plt.subplots(figsize=(3.5, 3.5))
    plot_reliability(calibration, ax=ax6)
    fig6.tight_layout()
    fig6.savefig(output_dir / f"fig6_calibration.{fmt}", dpi=dpi)
    logger.info("Saved fig6_calibration")

    # --- Figure 7: Best/worst cases ---
    fig7, ax7 = plt.subplots(figsize=(5, 4.5))
    plot_best_worst(ensemble, baseline, per_member, n_cases=5, ax=ax7)
    fig7.tight_layout()
    fig7.savefig(output_dir / f"fig7_best_worst.{fmt}", dpi=dpi)
    logger.info("Saved fig7_best_worst")

    # --- Figure 8: Dice by compartment ---
    fig8, ax8 = plt.subplots(figsize=(4.5, 3))
    plot_dice_by_compartment(ensemble, baseline, stats, ax=ax8)
    fig8.tight_layout()
    fig8.savefig(output_dir / f"fig8_dice_compartments.{fmt}", dpi=dpi)
    logger.info("Saved fig8_dice_compartments")

    plt.close("all")
    logger.info(f"All figures saved to {output_dir}/")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot LoRA-Ensemble results.")
    parser.add_argument("run_dir", type=str, help="Path to run directory (e.g., r8_M10_s42/)")
    parser.add_argument("--output", type=str, default=None, help="Output dir for figures")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "evaluation"
    output_dir = Path(args.output) if args.output else run_dir / "figures"

    generate_all_figures(eval_dir, output_dir, fmt=args.format, dpi=args.dpi)
