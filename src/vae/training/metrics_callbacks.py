"""Tidy CSV logging callbacks for VAE training.

This module provides callbacks for structured, analysis-ready CSV logging:
- TidyEpochCSVCallback: One row per epoch (no NaN fragmentation)
- TidyStepCSVCallback: Downsampled step-level logs
- GradNormCallback: Gradient norm monitoring for optimization stability
- RunMetadataCallback: Run configuration and data signature metadata

All callbacks are DDP-safe (rank-zero only file writes).
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

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


class TidyEpochCSVCallback(Callback):
    """Write epoch-level metrics to tidy CSV (one row per epoch).

    Reads trainer.callback_metrics after validation and writes a single row
    containing both train and val metrics, eliminating NaN fragmentation.

    Filters metrics by namespace prefix (train_epoch/, val_epoch/, sched/, opt/, diag/)
    and merges LatentDiagnosticsCallback metrics if available.

    Output: {run_dir}/logs/tidy/epoch_metrics.csv

    Args:
        run_dir: Root directory for outputs
        tidy_subdir: Subdirectory for tidy logs (default: "logs/tidy")
    """

    def __init__(
        self,
        run_dir: Path,
        tidy_subdir: str = "logs/tidy",
    ):
        self.csv_path = Path(run_dir) / tidy_subdir / "epoch_metrics.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False

    def on_train_epoch_end(self, trainer, pl_module):
        """Write epoch row after training epoch (and validation if it runs)."""
        # Rank-zero safety
        if not trainer.is_global_zero:
            return

        # Skip sanity check epoch
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics  # Dict[str, Tensor]

        # Collect row data
        row = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }

        # Filter by namespace prefix (NOT _epoch suffix)
        # Lightning only creates suffixes when on_step=True, on_epoch=True (which we removed)
        for key, val in metrics.items():
            if key.startswith(("train_epoch/", "val_epoch/", "sched/", "opt/", "diag/")):
                if torch.is_tensor(val):
                    row[key] = val.item()
                else:
                    row[key] = float(val) if isinstance(val, (int, float)) else val

        # Merge LatentDiagnosticsCallback metrics if available
        # Check if the callback exists and has exported last_epoch_metrics
        for callback in trainer.callbacks:
            if hasattr(callback, "last_epoch_metrics") and callback.last_epoch_metrics is not None:
                if callback.last_epoch_metrics.get("epoch") == trainer.current_epoch:
                    # Merge diag/* metrics
                    row.update(callback.last_epoch_metrics)
                    break

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
            logger.info(f"Initialized tidy epoch CSV: {self.csv_path}")
            self.header_written = True


class TidyStepCSVCallback(Callback):
    """Write downsampled step-level metrics to CSV.

    Buffers step metrics and flushes to CSV after each epoch.
    Uses global_step as primary x-axis (not batch_idx).

    Output: {run_dir}/logs/tidy/step_metrics.csv

    Args:
        run_dir: Root directory for outputs
        stride: Log every N steps (default: 50)
        tidy_subdir: Subdirectory for tidy logs (default: "logs/tidy")
    """

    def __init__(
        self,
        run_dir: Path,
        stride: int = 50,
        tidy_subdir: str = "logs/tidy",
    ):
        self.csv_path = Path(run_dir) / tidy_subdir / "step_metrics.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.stride = stride
        self._step_buffer = []
        self.header_written = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log step metrics with downsampling."""
        # Rank-zero safety (only collect on rank 0 to avoid memory leak on other ranks)
        if not trainer.is_global_zero:
            return

        # Downsample by stride
        if trainer.global_step % self.stride != 0:
            return

        metrics = trainer.callback_metrics
        row = self._collect_step_row(metrics, trainer, split="train")
        self._step_buffer.append(row)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Log validation step metrics with downsampling."""
        # Rank-zero safety (only collect on rank 0 to avoid memory leak on other ranks)
        if not trainer.is_global_zero:
            return

        # Downsample by stride
        if batch_idx % self.stride != 0:
            return

        metrics = trainer.callback_metrics
        row = self._collect_step_row(metrics, trainer, split="val")
        self._step_buffer.append(row)

    def on_train_epoch_end(self, trainer, pl_module):
        """Flush buffer to CSV after epoch."""
        # Rank-zero safety
        if not trainer.is_global_zero:
            return

        if not self._step_buffer:
            return

        self._flush_buffer()

    def _collect_step_row(self, metrics, trainer, split: str) -> Dict[str, Any]:
        """Extract step-level metrics."""
        row = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "split": split,
        }

        # Collect step-level metrics (not epoch-level)
        # Look for keys with step namespace (train_step/, val_step/, opt/)
        for key, val in metrics.items():
            # Keep step metrics matching the split
            if key.startswith(f"{split}_step/") or key.startswith("opt/"):
                if torch.is_tensor(val):
                    row[key] = val.item()
                else:
                    row[key] = float(val) if isinstance(val, (int, float)) else val

        return row

    def _flush_buffer(self):
        """Append buffer to CSV with atomic write."""
        df_new = pd.DataFrame(self._step_buffer)

        if self.csv_path.exists():
            df_existing = pd.read_csv(self.csv_path)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new

        # Atomic write
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=self.csv_path.parent, suffix=".tmp"
        ) as tmp_f:
            tmp_path = Path(tmp_f.name)
            df.to_csv(tmp_f, index=False)

        tmp_path.replace(self.csv_path)

        if not self.header_written:
            logger.info(
                f"Initialized tidy step CSV: {self.csv_path} (stride={self.stride})"
            )
            self.header_written = True

        # Clear buffer
        self._step_buffer = []


class GradNormCallback(Callback):
    """Log gradient norms for optimization stability monitoring.

    Computes total gradient norm (L2) across all parameters.
    Averages over optimizer steps (not batches) for epoch-level metric.

    Logs:
        - opt/grad_norm (epoch-level average)
        - opt/grad_finite (boolean check for NaN/Inf)

    Args:
        norm_type: Norm type (default: 2.0 for L2 norm)
    """

    def __init__(self, norm_type: float = 2.0):
        self.norm_type = norm_type
        self._grad_norms = []
        self._grad_finite_flags = []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Compute gradient norm before optimizer step."""
        # Compute total gradient norm
        total_norm = 0.0
        all_finite = True

        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
                all_finite = all_finite and torch.isfinite(param_norm).all()

        total_norm = total_norm ** (1.0 / self.norm_type)

        # Accumulate for epoch average (average over optimizer steps, not batches)
        self._grad_norms.append(total_norm)
        self._grad_finite_flags.append(float(all_finite))

        # Log step-level (for debugging)
        pl_module.log(
            "opt/grad_norm_step",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        pl_module.log(
            "opt/grad_finite_step",
            float(all_finite),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch-level average."""
        if not self._grad_norms:
            return

        # Average over optimizer steps
        avg_norm = sum(self._grad_norms) / len(self._grad_norms)
        avg_finite = sum(self._grad_finite_flags) / len(self._grad_finite_flags)

        # Log epoch-level
        pl_module.log(
            "opt/grad_norm", avg_norm, on_step=False, on_epoch=True, prog_bar=False
        )
        pl_module.log(
            "opt/grad_finite", avg_finite, on_step=False, on_epoch=True, prog_bar=False
        )

        # Reset accumulators
        self._grad_norms = []
        self._grad_finite_flags = []


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
            "note": "kl_active_frac_proxy is an online minibatch estimate; diag/au_frac is the canonical dataset-level Active Units metric",
        }

        # Ensure directory exists
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert OmegaConf objects to plain Python types for JSON serialization
        meta_serializable = _convert_for_json(meta)

        # Write JSON
        with open(self.meta_path, "w") as f:
            json.dump(meta_serializable, f, indent=2)

        logger.info(f"Saved run metadata to {self.meta_path}")
