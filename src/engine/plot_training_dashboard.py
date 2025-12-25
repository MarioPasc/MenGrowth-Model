#!/usr/bin/env python
"""Training dashboard visualization for VAE experiments.

This script generates a multi-panel dashboard figure from tidy CSV logs.

Usage:
    python src/engine/plot_training_dashboard.py --run_dir experiments/runs/exp1_baseline_vae_20250101_120000
    python src/engine/plot_training_dashboard.py --run_dir experiments/runs/exp2_tcvae_sbd_20250101_120000
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate training dashboard from tidy CSV logs"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory (containing logs/tidy/epoch_metrics.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for dashboard PNG (default: <run_dir>/logs/tidy/dashboard.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output figure (default: 150)",
    )
    return parser.parse_args()


def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Load epoch metrics CSV.

    Args:
        run_dir: Run directory path.

    Returns:
        DataFrame with epoch metrics.
    """
    epoch_csv = run_dir / "logs" / "tidy" / "epoch_metrics.csv"

    if not epoch_csv.exists():
        raise FileNotFoundError(
            f"Epoch metrics CSV not found: {epoch_csv}\n"
            f"Make sure you're pointing to a run directory with tidy logging enabled."
        )

    df = pd.read_csv(epoch_csv)
    logger.info(f"Loaded {len(df)} epochs from {epoch_csv}")

    return df


def detect_experiment_type(df: pd.DataFrame) -> str:
    """Detect experiment type from available columns.

    Args:
        df: Epoch metrics DataFrame.

    Returns:
        Experiment type: "exp1" (baseline VAE), "exp2_tc" (TC-VAE), or "exp2_dip" (DIP-VAE).
    """
    # Exp2b: DIP-VAE has covariance penalty metrics
    if "train_epoch/cov_penalty" in df.columns:
        return "exp2_dip"
    # Exp2a: TC-VAE has TC decomposition metrics
    elif "train_epoch/mi" in df.columns and "train_epoch/tc" in df.columns:
        return "exp2_tc"
    # Exp1: Baseline VAE
    else:
        return "exp1"


def plot_dashboard(df: pd.DataFrame, experiment_type: str, output_path: Path, dpi: int = 150):
    """Generate 3x3 dashboard figure.

    Args:
        df: Epoch metrics DataFrame.
        experiment_type: "exp1" or "exp2".
        output_path: Output path for PNG.
        dpi: DPI for output figure.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"Training Dashboard ({experiment_type.upper()})", fontsize=14, fontweight="bold")

    epoch = df["epoch"]

    # === Row 1: Loss components ===
    ax = axes[0, 0]
    ax.plot(epoch, df["train_epoch/loss"], label="Train", linewidth=1.5)
    ax.plot(epoch, df["val_epoch/loss"], label="Val", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epoch, df["train_epoch/recon"], label="Train", linewidth=1.5)
    ax.plot(epoch, df["val_epoch/recon"], label="Val", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_title("Reconstruction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if experiment_type == "exp1":
        # Exp1: KL loss
        ax.plot(epoch, df["train_epoch/kl"], label="Train KL", linewidth=1.5)
        if "val_epoch/kl" in df.columns:
            ax.plot(epoch, df["val_epoch/kl"], label="Val KL", linewidth=1.5, linestyle="--")
        ax.set_ylabel("KL Loss")
        ax.set_title("KL Regularization")
    elif experiment_type == "exp2_tc":
        # Exp2a: TC decomposition (MI, TC, DWKL)
        ax.plot(epoch, df["train_epoch/mi"], label="MI", linewidth=1.5, alpha=0.7)
        ax.plot(epoch, df["train_epoch/tc"], label="TC", linewidth=1.5, alpha=0.7)
        ax.plot(epoch, df["train_epoch/dwkl"], label="DWKL", linewidth=1.5, alpha=0.7)
        ax.set_ylabel("TC Decomposition")
        ax.set_title("TC-VAE Decomposition (MI, TC, DWKL)")
    elif experiment_type == "exp2_dip":
        # Exp2b: DIP-VAE covariance penalties
        ax.plot(epoch, df["train_epoch/cov_penalty"], label="Total Cov Penalty", linewidth=1.5)
        ax.plot(epoch, df["train_epoch/cov_penalty_od"], label="Off-diag", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.plot(epoch, df["train_epoch/cov_penalty_d"], label="Diag", linewidth=1.5, linestyle=":", alpha=0.7)
        ax.set_ylabel("Covariance Penalty")
        ax.set_title("DIP-VAE Covariance Penalties")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === Row 2: Schedules, LR, Grad Norm ===
    ax = axes[1, 0]
    if experiment_type == "exp1":
        # Exp1: Beta schedule
        if "sched/beta" in df.columns:
            ax.plot(epoch, df["sched/beta"], label="Beta", linewidth=1.5, color="C2")
            ax.set_ylabel("Beta")
            ax.set_title("KL Beta Schedule")
        # Add expected KL floor if available
        if "sched/expected_kl_floor" in df.columns:
            ax2 = ax.twinx()
            ax2.plot(epoch, df["sched/expected_kl_floor"], label="Expected KL Floor",
                    linewidth=1.5, color="C3", linestyle="--", alpha=0.7)
            ax2.set_ylabel("Expected KL Floor (nats)")
            ax2.legend(loc="upper right")
    elif experiment_type == "exp2_tc":
        # Exp2a: Beta_tc schedule
        if "sched/beta_tc" in df.columns:
            ax.plot(epoch, df["sched/beta_tc"], label="Beta TC", linewidth=1.5, color="C2")
            ax.set_ylabel("Beta TC")
            ax.set_title("Beta TC Schedule")
        # Add expected KL floor if available
        if "sched/expected_kl_floor" in df.columns:
            ax2 = ax.twinx()
            ax2.plot(epoch, df["sched/expected_kl_floor"], label="Expected KL Floor",
                    linewidth=1.5, color="C3", linestyle="--", alpha=0.7)
            ax2.set_ylabel("Expected KL Floor (nats)")
            ax2.legend(loc="upper right")
    elif experiment_type == "exp2_dip":
        # Exp2b: Lambda schedule
        if "sched/lambda_od" in df.columns:
            ax.plot(epoch, df["sched/lambda_od"], label="λ_od", linewidth=1.5, color="C2")
        if "sched/lambda_d" in df.columns:
            ax.plot(epoch, df["sched/lambda_d"], label="λ_d", linewidth=1.5, color="C3", linestyle="--")
        ax.set_ylabel("Lambda")
        ax.set_title("DIP-VAE Lambda Schedule")
        # Add expected KL floor if available
        if "sched/expected_kl_floor" in df.columns:
            ax2 = ax.twinx()
            ax2.plot(epoch, df["sched/expected_kl_floor"], label="Expected KL Floor",
                    linewidth=1.5, color="C4", linestyle=":", alpha=0.7)
            ax2.set_ylabel("Expected KL Floor (nats)")
            ax2.legend(loc="upper right")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "opt/lr" in df.columns:
        ax.plot(epoch, df["opt/lr"], linewidth=1.5, color="C1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "LR not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Learning Rate")

    ax = axes[1, 2]
    if "opt/grad_norm" in df.columns:
        ax.plot(epoch, df["opt/grad_norm"], linewidth=1.5, color="C4")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_title("Gradient Norm (Optimization Stability)")
        ax.grid(True, alpha=0.3)
        # Add median line for reference
        median_norm = df["opt/grad_norm"].median()
        ax.axhline(median_norm, color="red", linestyle="--", alpha=0.5, linewidth=1,
                  label=f"Median: {median_norm:.2f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Grad norm not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Gradient Norm")

    # === Row 3: Latent Diagnostics ===
    ax = axes[2, 0]
    # Active Units (AU) - canonical metric only
    # Note: Proxy metric (train_epoch/kl_active_frac_proxy) removed in favor of canonical AU
    if "diag/au_frac" in df.columns:
        # Only plot where available (every N epochs)
        diag_df = df[df["diag/au_frac"].notna()]
        ax.scatter(diag_df["epoch"], diag_df["diag/au_frac"], label="Canonical AU",
                  marker="o", s=50, color="C3", zorder=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Fraction")
    ax.set_title("Active Units (AU) - Canonical Dataset-Level Variance")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if "diag/corr_offdiag_meanabs" in df.columns:
        diag_df = df[df["diag/corr_offdiag_meanabs"].notna()]
        ax.plot(diag_df["epoch"], diag_df["diag/corr_offdiag_meanabs"],
               linewidth=1.5, marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean |Correlation|")
        ax.set_title("Latent Correlation (Off-diagonal)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Correlation not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Latent Correlation")

    ax = axes[2, 2]
    if "diag/shift_mu_l2_mean" in df.columns:
        diag_df = df[df["diag/shift_mu_l2_mean"].notna()]
        ax.plot(diag_df["epoch"], diag_df["diag/shift_mu_l2_mean"],
               linewidth=1.5, marker="o", markersize=4, label="L2 norm")
        if "diag/shift_mu_absmean" in df.columns:
            ax.plot(diag_df["epoch"], diag_df["diag/shift_mu_absmean"],
                   linewidth=1.5, marker="s", markersize=4, alpha=0.7, label="Mean |Δμ|")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Δμ Magnitude")
        ax.set_title("Spatial Shift Sensitivity")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Shift sensitivity not available",
               ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Shift Sensitivity")

    # Adjust layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved dashboard to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load metrics
    df = load_metrics(run_dir)

    # Detect experiment type
    experiment_type = detect_experiment_type(df)
    logger.info(f"Detected experiment type: {experiment_type}")

    # Determine output path
    if args.output is None:
        output_path = run_dir / "logs" / "tidy" / "dashboard.png"
    else:
        output_path = Path(args.output)

    # Generate dashboard
    plot_dashboard(df, experiment_type, output_path, dpi=args.dpi)

    logger.info("Dashboard generation complete!")


if __name__ == "__main__":
    main()
