"""Unified CSV logging callback for VAE training.

This module provides the unified CSV logging callback that consolidates:
- Epoch-level metrics (train/val/sched/opt/latent_diag/system)
- System metrics (GPU memory, throughput, batch time)
- Gradient norm tracking (optional)
- Run configuration metadata

All callbacks are DDP-safe (rank-zero only file writes).
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _convert_for_json(obj):
    """Recursively convert OmegaConf objects to plain Python types.

    Args:
        obj: Object to convert (may contain nested OmegaConf objects).

    Returns:
        JSON-serializable version of obj.
    """
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    return obj


class UnifiedCSVCallback(Callback):
    """Unified CSV callback that consolidates all epoch-level logging.

    Combines functionality from:
    - TidyEpochCSVCallback (epoch metrics collection)
    - SystemMetricsCallback (GPU memory, throughput, batch time)
    - GradNormCallback (gradient norm tracking)

    Writes one row per epoch containing:
    - Metadata: epoch, global_step
    - Training metrics: train_epoch/*
    - Validation metrics: val_epoch/* (including SSIM, PSNR)
    - Schedule metrics: sched/*
    - Optimizer metrics: opt/lr, opt/grad_norm, opt/grad_finite
    - Latent diagnostics: latent_diag/* (sparse, "-" placeholder on non-diagnostic epochs)
    - System metrics: system/* (GPU memory, throughput, execution time)

    Output: {run_dir}/logs/tidy/epoch_metrics.csv

    Args:
        run_dir: Root directory for outputs
        tidy_subdir: Subdirectory for tidy logs (default: "logs/tidy")
        log_grad_norm: Whether to track gradient norms (default: True)
        grad_norm_type: Norm type for gradient norm (default: 2.0 for L2)
    """

    # Define all possible latent_diag/* columns for placeholder initialization
    # These are populated by LatentDiagnosticsCallback and ActiveUnitsCallback
    LATENT_DIAG_COLUMNS = [
        # Active Units (from ActiveUnitsCallback)
        "latent_diag/au_count",
        "latent_diag/au_frac",
        "latent_diag/au_threshold",
        "latent_diag/au_subset_size",
        # Correlation (from LatentDiagnosticsCallback)
        "latent_diag/corr_offdiag_meanabs",
        # Shift sensitivity (from LatentDiagnosticsCallback)
        "latent_diag/shift_mu_l2_mean",
        "latent_diag/shift_mu_absmean",
        # DIP-VAE covariance (from LatentDiagnosticsCallback)
        "latent_diag/cov_q_offdiag_meanabs",
        "latent_diag/cov_q_offdiag_fro",
        "latent_diag/cov_q_diag_meanabs_error",
        "latent_diag/cov_q_diag_mean",
        # Ridge probe R² scores (from LatentDiagnosticsCallback)
        # Volume targets
        "latent_diag/r2_logV_ncr_mean",
        "latent_diag/r2_logV_ncr_std",
        "latent_diag/r2_logV_ed_mean",
        "latent_diag/r2_logV_ed_std",
        "latent_diag/r2_logV_et_mean",
        "latent_diag/r2_logV_et_std",
        "latent_diag/r2_logV_total_mean",
        "latent_diag/r2_logV_total_std",
        # Centroid targets
        "latent_diag/r2_cz_total_mean",
        "latent_diag/r2_cz_total_std",
        "latent_diag/r2_cy_total_mean",
        "latent_diag/r2_cy_total_std",
        "latent_diag/r2_cx_total_mean",
        "latent_diag/r2_cx_total_std",
        # Radius targets
        "latent_diag/r2_r_ncr_mean",
        "latent_diag/r2_r_ncr_std",
        "latent_diag/r2_r_ed_mean",
        "latent_diag/r2_r_ed_std",
        "latent_diag/r2_r_et_mean",
        "latent_diag/r2_r_et_std",
    ]

    def __init__(
        self,
        run_dir: Path,
        tidy_subdir: str = "logs/tidy",
        log_grad_norm: bool = True,
        grad_norm_type: float = 2.0,
    ):
        self.csv_path = Path(run_dir) / tidy_subdir / "epoch_metrics.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False

        # Gradient norm tracking
        self.log_grad_norm = log_grad_norm
        self.grad_norm_type = grad_norm_type
        self._grad_norms = []
        self._grad_finite_flags = []

        # System metrics tracking
        self._epoch_start_time = None
        self._batch_start_time = None
        self._batch_times = []
        self._gpu_mem_samples = []
        self._samples_processed = 0

    def on_train_epoch_start(self, trainer, pl_module):
        """Reset accumulators and record epoch start time."""
        self._epoch_start_time = time.time()
        self._batch_times = []
        self._gpu_mem_samples = []
        self._samples_processed = 0
        self._grad_norms = []
        self._grad_finite_flags = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record batch start time for throughput calculation."""
        self._batch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Sample GPU memory and record batch times."""
        # Record batch time
        if self._batch_start_time is not None:
            batch_time = time.time() - self._batch_start_time
            self._batch_times.append(batch_time)

        # Count samples processed
        batch_size = batch["image"].size(0)
        self._samples_processed += batch_size

        # Sample GPU memory (every 50 batches to minimize overhead)
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            self._gpu_mem_samples.append((gpu_mem_allocated, gpu_mem_reserved))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Compute gradient norm before optimizer step (if enabled)."""
        if not self.log_grad_norm:
            return

        # Compute total gradient norm
        total_norm = 0.0
        all_finite = True

        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.grad_norm_type)
                total_norm += param_norm.item() ** self.grad_norm_type
                all_finite = all_finite and torch.isfinite(param_norm).all()

        total_norm = total_norm ** (1.0 / self.grad_norm_type)

        # Accumulate for epoch average
        self._grad_norms.append(total_norm)
        self._grad_finite_flags.append(float(all_finite))

    def on_train_epoch_end(self, trainer, pl_module):
        """Compute system metrics, collect all metrics, and write CSV row."""
        # Rank-zero safety
        if not trainer.is_global_zero:
            return

        # Skip sanity check epoch
        if trainer.sanity_checking:
            return

        # Compute system metrics
        epoch_time = time.time() - self._epoch_start_time if self._epoch_start_time else 0.0
        samples_per_sec = self._samples_processed / epoch_time if epoch_time > 0 else 0.0
        batch_time_avg = np.mean(self._batch_times) * 1000 if self._batch_times else 0.0  # ms

        # Average GPU memory
        if self._gpu_mem_samples:
            gpu_mem_allocated_avg = np.mean([x[0] for x in self._gpu_mem_samples])
            gpu_mem_reserved_avg = np.mean([x[1] for x in self._gpu_mem_samples])
        else:
            gpu_mem_allocated_avg = 0.0
            gpu_mem_reserved_avg = 0.0

        # Average gradient norm (if enabled)
        if self.log_grad_norm and self._grad_norms:
            grad_norm_avg = np.mean(self._grad_norms)
            grad_finite_avg = np.mean(self._grad_finite_flags)
        else:
            grad_norm_avg = None
            grad_finite_avg = None

        # Build row dict
        row = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }

        # Collect from trainer.callback_metrics (train/val/sched/opt/diag from lit_modules)
        # Note: We filter for train_epoch/, val_epoch/, sched/, opt/, diag/
        # diag/* from lit_modules are batch-level collapse diagnostics (e.g., diag/recon_mu_mse)
        # latent_diag/* from diagnostic callbacks are dataset-level diagnostics
        metrics = trainer.callback_metrics
        for key, val in metrics.items():
            if key.startswith(("train_epoch/", "val_epoch/", "sched/", "opt/", "diag/")):
                if torch.is_tensor(val):
                    row[key] = val.item()
                else:
                    row[key] = float(val) if isinstance(val, (int, float)) else val

        # Initialize latent_diag/* columns with "-" placeholder
        # (will be overwritten if diagnostics ran this epoch)
        for col in self.LATENT_DIAG_COLUMNS:
            row[col] = "-"

        # Merge diagnostic callback exports (if available)
        # Callbacks set last_epoch_metrics with latent_diag/* keys
        for callback in trainer.callbacks:
            if hasattr(callback, "last_epoch_metrics") and callback.last_epoch_metrics is not None:
                if callback.last_epoch_metrics.get("epoch") == trainer.current_epoch:
                    # Merge latent_diag/* metrics (overwrites "-" placeholders)
                    row.update(callback.last_epoch_metrics)

        # Add system metrics
        row["system/gpu_mem_allocated_gb"] = gpu_mem_allocated_avg
        row["system/gpu_mem_reserved_gb"] = gpu_mem_reserved_avg
        row["system/samples_per_sec"] = samples_per_sec
        row["system/batch_time_ms_avg"] = batch_time_avg
        row["system/epoch_time_sec"] = epoch_time

        # Add gradient metrics (if enabled)
        if self.log_grad_norm and grad_norm_avg is not None:
            row["opt/grad_norm"] = grad_norm_avg
            row["opt/grad_finite"] = grad_finite_avg

        # Write to CSV
        self._append_to_csv(row)

    def _append_to_csv(self, row: Dict[str, Any]):
        """Append row to CSV with atomic write."""
        df_new = pd.DataFrame([row])

        if self.csv_path.exists():
            df_existing = pd.read_csv(self.csv_path)

            # Remove duplicate epoch if exists (allows re-runs)
            df_existing = df_existing[df_existing["epoch"] != row["epoch"]]

            # Concatenate
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new

        # Sort by epoch
        df = df.sort_values("epoch").reset_index(drop=True)

        # Atomic write (tempfile + rename)
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=self.csv_path.parent, suffix=".tmp"
        ) as tmp_f:
            tmp_path = Path(tmp_f.name)
            df.to_csv(tmp_f, index=False)

        tmp_path.replace(self.csv_path)

        if not self.header_written:
            logger.info(f"Initialized unified epoch CSV: {self.csv_path}")
            self.header_written = True


class RunMetadataCallback(Callback):
    """Save run configuration and data signature to JSON.

    Writes metadata at training start for reproducibility and auditability.

    Output: {run_dir}/logs/tidy/run_meta.json

    Includes:
        - Experiment config (seed, epochs, batch_size, lr, z_dim, precision)
        - Data signature (input_shape, voxel_count, normalization, recon_likelihood)
        - Schedule config (beta, annealing, free_bits, capacity)

    Args:
        run_dir: Root directory for outputs
        cfg: OmegaConf configuration
        tidy_subdir: Subdirectory for tidy logs (default: "logs/tidy")
    """

    def __init__(
        self,
        run_dir: Path,
        cfg: DictConfig,
        tidy_subdir: str = "logs/tidy",
    ):
        self.meta_path = Path(run_dir) / tidy_subdir / "run_meta.json"
        self.cfg = cfg

    def on_train_start(self, trainer, pl_module):
        """Write metadata at training start."""
        # Rank-zero safety
        if not trainer.is_global_zero:
            return

        # Determine experiment type
        experiment = self.cfg.model.get("variant", "baseline_vae")

        # Base metadata
        meta = {
            "experiment": experiment,
            "run_dir": str(self.meta_path.parent.parent.parent),
            "seed": self.cfg.train.seed,
            "max_epochs": self.cfg.train.max_epochs,
            "batch_size": self.cfg.data.batch_size,
            "lr": self.cfg.train.lr,
            "z_dim": self.cfg.model.z_dim,
            "precision": self.cfg.train.precision,
        }

        # Data signature (for reproducibility and comparison)
        # Compute voxel count from roi_size
        roi_size = self.cfg.data.roi_size  # e.g., [128, 128, 128]
        voxel_count = roi_size[0] * roi_size[1] * roi_size[2]

        meta["data_signature"] = {
            "input_shape": roi_size,
            "input_channels": self.cfg.model.input_channels,
            "voxel_count": voxel_count,
            "spacing_mm": self.cfg.data.get("spacing", [1.875, 1.875, 1.875]),
            "modalities": self.cfg.data.get("modalities", ["t1c", "t1n", "t2f", "t2w"]),
            # Note: Intensity normalization is z-score per-subject (channel-wise)
            "intensity_normalization": "zscore_per_subject_per_channel",
            # Reconstruction likelihood
            "recon_likelihood_type": "gaussian_mse_surrogate",
            "loss_reduction": self.cfg.train.get("loss_reduction", "mean"),
        }

        # Schedule config
        meta["schedule"] = {}

        # Exp1-specific (ELBO VAE)
        if "kl_beta" in self.cfg.train:
            meta["schedule"]["kl_beta"] = self.cfg.train.kl_beta
            meta["schedule"]["kl_annealing_epochs"] = self.cfg.train.kl_annealing_epochs
            meta["schedule"]["kl_annealing_type"] = self.cfg.train.get(
                "kl_annealing_type", "linear"
            )
            meta["schedule"]["kl_annealing_cycles"] = self.cfg.train.get(
                "kl_annealing_cycles", 1
            )
            meta["schedule"]["kl_annealing_ratio"] = self.cfg.train.get(
                "kl_annealing_ratio", 1.0
            )
            meta["schedule"]["kl_free_bits"] = self.cfg.train.get("kl_free_bits", 0.0)
            meta["schedule"]["kl_free_bits_mode"] = self.cfg.train.get(
                "kl_free_bits_mode", "per_sample"
            )

            # Capacity control (if enabled)
            if self.cfg.train.get("kl_target_capacity") is not None:
                meta["schedule"]["kl_target_capacity"] = (
                    self.cfg.train.kl_target_capacity
                )
                meta["schedule"]["kl_capacity_anneal_epochs"] = (
                    self.cfg.train.kl_capacity_anneal_epochs
                )

        # Exp2a-specific (β-TCVAE)
        if "loss" in self.cfg and "beta_tc_target" in self.cfg.loss:
            meta["schedule"]["beta_tc_target"] = self.cfg.loss.beta_tc_target
            meta["schedule"]["beta_tc_annealing_epochs"] = (
                self.cfg.loss.beta_tc_annealing_epochs
            )
            meta["schedule"]["alpha"] = self.cfg.loss.alpha
            meta["schedule"]["gamma"] = self.cfg.loss.gamma
            meta["schedule"]["kl_free_bits"] = self.cfg.train.get("kl_free_bits", 0.0)
            meta["schedule"]["kl_free_bits_mode"] = self.cfg.train.get(
                "kl_free_bits_mode", "batch_mean"
            )

        # Exp2b-specific (DIP-VAE)
        if "loss" in self.cfg and "lambda_od" in self.cfg.loss:
            meta["schedule"]["lambda_od"] = self.cfg.loss.lambda_od
            meta["schedule"]["lambda_d"] = self.cfg.loss.lambda_d
            meta["schedule"]["lambda_cov_annealing_epochs"] = self.cfg.loss.get(
                "lambda_cov_annealing_epochs", 0
            )
            meta["schedule"]["use_ddp_gather"] = self.cfg.loss.get(
                "use_ddp_gather", True
            )
            meta["schedule"]["posterior_logvar_min"] = self.cfg.train.get(
                "posterior_logvar_min", -6.0
            )
            meta["schedule"]["sbd_upsample_mode"] = self.cfg.model.get(
                "sbd_upsample_mode", "resize_conv"
            )
            meta["schedule"]["kl_free_bits"] = self.cfg.train.get("kl_free_bits", 0.0)
            meta["schedule"]["kl_free_bits_mode"] = self.cfg.train.get(
                "kl_free_bits_mode", "batch_mean"
            )

        # Collapse proxy config
        meta["collapse_metrics"] = {
            "eps_kl_dim": self.cfg.logging.get("eps_kl_dim", 0.01),
            "note": "kl_active_frac_proxy is an online minibatch estimate; latent_diag/au_frac is the canonical dataset-level Active Units metric",
        }

        # Ensure directory exists
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert OmegaConf objects to plain Python types for JSON serialization
        meta_serializable = _convert_for_json(meta)

        # Write JSON
        with open(self.meta_path, "w") as f:
            json.dump(meta_serializable, f, indent=2)

        logger.info(f"Saved run metadata to {self.meta_path}")
