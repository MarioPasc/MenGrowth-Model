"""Lightning callbacks for VAE training.

This module implements custom callbacks for:
- Saving reconstruction visualizations at regular intervals
- Custom logging to replace tqdm progress bars with informative console output
"""

import logging
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


logger = logging.getLogger(__name__)


class TrainingLoggingCallback(Callback):
    """Callback to log training progress to console, replacing tqdm.

    Logs loss values and metrics at regular intervals within each epoch,
    ensuring at least `min_logs_per_epoch` log entries per epoch.

    Also logs epoch summaries and timing information.
    """

    def __init__(
        self,
        min_logs_per_epoch: int = 3,
        log_val_every_n_batches: int = 1,
    ):
        """Initialize TrainingLoggingCallback.

        Args:
            min_logs_per_epoch: Minimum number of log entries per training epoch.
            log_val_every_n_batches: Log validation metrics every N batches.
        """
        super().__init__()
        self.min_logs_per_epoch = min_logs_per_epoch
        self.log_val_every_n_batches = log_val_every_n_batches

        # Epoch tracking
        self._epoch_start_time = None
        self._train_batch_count = 0
        self._log_interval = 1

        # Accumulate metrics for summary
        self._train_losses = []
        self._val_losses = []

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Record epoch start time and reset counters."""
        self._epoch_start_time = time.time()
        self._train_batch_count = 0
        self._train_losses = []

        # Calculate log interval to achieve min_logs_per_epoch
        total_batches = len(trainer.train_dataloader)
        self._log_interval = max(1, total_batches // self.min_logs_per_epoch)

        logger.info(
            f"Epoch {trainer.current_epoch}/{trainer.max_epochs - 1} started "
            f"({total_batches} batches, logging every {self._log_interval} batches)"
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Log training metrics at regular intervals."""
        self._train_batch_count += 1

        # Extract loss from outputs
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = outputs["loss"].item() if torch.is_tensor(outputs["loss"]) else outputs["loss"]
        elif torch.is_tensor(outputs):
            loss_val = outputs.item()
        else:
            loss_val = float(outputs) if outputs is not None else 0.0

        self._train_losses.append(loss_val)

        # Log at intervals
        if (batch_idx + 1) % self._log_interval == 0 or batch_idx == 0:
            total_batches = len(trainer.train_dataloader)
            progress = 100 * (batch_idx + 1) / total_batches

            # Get additional metrics from logged values
            metrics_str = self._format_logged_metrics(trainer, prefix="train/")

            logger.info(
                f"  [Train] Batch {batch_idx + 1}/{total_batches} ({progress:.0f}%) | "
                f"loss={loss_val:.4f}{metrics_str}"
            )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log epoch summary with average loss and timing."""
        epoch_time = time.time() - self._epoch_start_time

        avg_train_loss = sum(self._train_losses) / len(self._train_losses) if self._train_losses else 0.0

        logger.info(
            f"Epoch {trainer.current_epoch} train complete | "
            f"avg_loss={avg_train_loss:.4f} | time={epoch_time:.1f}s"
        )

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Reset validation tracking."""
        self._val_losses = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Track validation losses."""
        if torch.is_tensor(outputs):
            loss_val = outputs.item()
        elif isinstance(outputs, (int, float)):
            loss_val = float(outputs)
        else:
            loss_val = 0.0

        self._val_losses.append(loss_val)

        # Log validation progress
        if (batch_idx + 1) % self.log_val_every_n_batches == 0:
            total_batches = len(trainer.val_dataloaders)
            metrics_str = self._format_logged_metrics(trainer, prefix="val/")
            logger.info(
                f"  [Val] Batch {batch_idx + 1}/{total_batches} | "
                f"loss={loss_val:.4f}{metrics_str}"
            )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log validation epoch summary."""
        avg_val_loss = sum(self._val_losses) / len(self._val_losses) if self._val_losses else 0.0

        # Get all validation metrics
        metrics_str = self._format_logged_metrics(trainer, prefix="val/", include_loss=False)

        logger.info(
            f"Epoch {trainer.current_epoch} validation complete | "
            f"val_loss={avg_val_loss:.4f}{metrics_str}"
        )

    def _format_logged_metrics(
        self,
        trainer: pl.Trainer,
        prefix: str,
        include_loss: bool = False,
    ) -> str:
        """Format logged metrics as string.

        Args:
            trainer: PyTorch Lightning trainer.
            prefix: Metric prefix to filter (e.g., "train/", "val/").
            include_loss: Whether to include loss in output (already shown separately).

        Returns:
            Formatted string of metrics.
        """
        metrics = trainer.callback_metrics
        parts = []

        for key, value in metrics.items():
            if key.startswith(prefix):
                short_key = key.replace(prefix, "")
                if not include_loss and short_key == "loss":
                    continue
                if torch.is_tensor(value):
                    value = value.item()
                parts.append(f"{short_key}={value:.4f}")

        if parts:
            return " | " + " | ".join(parts)
        return ""


class ReconstructionCallback(Callback):
    """Callback to save reconstruction visualizations during validation.

    Saves central slices (axial, coronal, sagittal) for each modality channel
    comparing input vs reconstruction vs absolute difference.

    Visualizations are saved to:
        <run_dir>/recon/epoch_<E>/sample_<S>_mod<M>_<view>.png
    """

    def __init__(
        self,
        run_dir: str,
        recon_every_n_epochs: int = 5,
        num_recon_samples: int = 2,
        modality_names: Optional[list] = None,
    ):
        """Initialize ReconstructionCallback.

        Args:
            run_dir: Path to run directory for saving outputs.
            recon_every_n_epochs: Save reconstructions every N epochs.
            num_recon_samples: Number of samples to visualize.
            modality_names: Names of modality channels for labeling.
        """
        super().__init__()
        self.run_dir = Path(run_dir)
        self.recon_every_n_epochs = recon_every_n_epochs
        self.num_recon_samples = num_recon_samples
        self.modality_names = modality_names or ["T1c", "T1n", "T2f", "T2w"]

        self._val_outputs = []

        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available, reconstruction visualizations disabled")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Store batch data for visualization at epoch end."""
        # Only store from first few batches to get required samples
        if batch_idx == 0:
            self._val_outputs = []

        current_samples = sum(len(o["x"]) for o in self._val_outputs)
        if current_samples < self.num_recon_samples:
            x = batch["image"].detach().cpu()
            with torch.no_grad():
                # Get reconstruction (first element) - works for both BaselineVAE (3 returns) and TCVAESBD (4 returns)
                x_hat = pl_module.model(batch["image"])[0]
                x_hat = x_hat.detach().cpu()

            self._val_outputs.append({"x": x, "x_hat": x_hat})

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Save reconstruction visualizations at end of validation epoch."""
        if not HAS_MATPLOTLIB:
            return

        epoch = trainer.current_epoch
        if epoch % self.recon_every_n_epochs != 0:
            self._val_outputs = []
            return

        if not self._val_outputs:
            return

        # Collect samples
        all_x = torch.cat([o["x"] for o in self._val_outputs], dim=0)
        all_x_hat = torch.cat([o["x_hat"] for o in self._val_outputs], dim=0)

        num_samples = min(self.num_recon_samples, len(all_x))

        # Create output directory
        epoch_dir = self.run_dir / "recon" / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        for sample_idx in range(num_samples):
            x = all_x[sample_idx].numpy()      # [C, D, H, W]
            x_hat = all_x_hat[sample_idx].numpy()

            self._save_sample_visualizations(
                x, x_hat, sample_idx, epoch_dir
            )

        logger.info(f"Saved {num_samples} reconstruction samples to {epoch_dir}")
        self._val_outputs = []

    def _save_sample_visualizations(
        self,
        x: np.ndarray,
        x_hat: np.ndarray,
        sample_idx: int,
        output_dir: Path,
    ) -> None:
        """Save visualization for a single sample.

        Args:
            x: Original input [C, D, H, W].
            x_hat: Reconstruction [C, D, H, W].
            sample_idx: Sample index for filename.
            output_dir: Directory to save images.
        """
        C, D, H, W = x.shape
        center_d, center_h, center_w = D // 2, H // 2, W // 2

        for mod_idx, mod_name in enumerate(self.modality_names):
            # Get modality channel
            x_mod = x[mod_idx]
            x_hat_mod = x_hat[mod_idx]
            diff = np.abs(x_hat_mod - x_mod)

            # Extract central slices
            slices = {
                "axial": (x_mod[center_d, :, :], x_hat_mod[center_d, :, :], diff[center_d, :, :]),
                "coronal": (x_mod[:, center_h, :], x_hat_mod[:, center_h, :], diff[:, center_h, :]),
                "sagittal": (x_mod[:, :, center_w], x_hat_mod[:, :, center_w], diff[:, :, center_w]),
            }

            for view_name, (orig, recon, d) in slices.items():
                self._save_comparison_figure(
                    orig, recon, d,
                    mod_name, view_name,
                    sample_idx, output_dir,
                )

    def _save_comparison_figure(
        self,
        original: np.ndarray,
        reconstruction: np.ndarray,
        difference: np.ndarray,
        modality: str,
        view: str,
        sample_idx: int,
        output_dir: Path,
    ) -> None:
        """Save a comparison figure showing input, reconstruction, and difference.

        Args:
            original: Original slice [H, W].
            reconstruction: Reconstructed slice [H, W].
            difference: Absolute difference [H, W].
            modality: Modality name for title.
            view: View name (axial/coronal/sagittal).
            sample_idx: Sample index.
            output_dir: Output directory.
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Determine color limits from original
        vmin, vmax = original.min(), original.max()

        axes[0].imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Input ({modality})")
        axes[0].axis('off')

        axes[1].imshow(reconstruction, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1].set_title("Reconstruction")
        axes[1].axis('off')

        im = axes[2].imshow(difference, cmap='hot')
        axes[2].set_title("Absolute Difference")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"{modality} - {view.capitalize()}", fontsize=12)
        plt.tight_layout()

        filename = f"sample_{sample_idx:02d}_{modality.lower()}_{view}.png"
        fig.savefig(output_dir / filename, dpi=100, bbox_inches='tight')
        plt.close(fig)


class LatentDiagnosticsCallback(Callback):
    """Callback to compute latent space diagnostics on a fixed validation subset.

    This callback implements lightweight disentanglement diagnostics computed every N epochs
    on a deterministically selected fixed subset of validation samples. All statistics are
    computed in float32 on CPU to ensure numerical stability.

    Diagnostics:
    - Correlation: Mean absolute off-diagonal correlation of latent means
      (proxy for independence - high correlation indicates entangled factors)
    - Translation sensitivity: L2 change in μ under spatial shift
      (ODE-readiness check - low sensitivity indicates position-invariant encoding, as desired for SBD)
    - Segmentation probes: Ridge regression R² from latents to tumor characteristics
      (semantic alignment check - high R² indicates latent subspaces encode interpretable features)

    Metrics are logged to:
    1. Lightning logger (tensorboard/wandb) for real-time monitoring
    2. CSV file for post-hoc analysis and plotting

    Hard constraints:
    - DDP-safe: All filesystem writes only on rank 0
    - Deterministic: Fixed sample IDs persisted and reused across epochs
    - Lightweight: Computed only every N epochs on small subset (~32 samples)
    - Numerically stable: FP32 on CPU for all statistics

    Args:
        run_dir: Root directory for saving outputs.
        every_n_epochs: Compute diagnostics every N epochs (default: 10).
        num_samples: Number of validation samples to use (default: 32).
        shift_vox: Translation shift magnitude in voxels (default: 5).
        eps_au: Variance threshold (nats) - kept for backward compatibility but no longer used.
        csv_name: Relative path for CSV metrics file (default: "latent_diag/metrics.csv").
        ids_name: Relative path for sample IDs file (default: "latent_diag/ids.txt").
        image_key: Batch dict key for images (default: "image").
        seg_key: Batch dict key for segmentations (default: "seg").
        id_key: Batch dict key for sample IDs (default: "id").
        seg_labels: Segmentation label mapping dict (default: {"ncr": 1, "ed": 2, "et": 3} for BraTS).

    Note:
        Active Units (AU) metric is now computed by the dedicated ActiveUnitsCallback,
        which provides more flexible scheduling (dense-early/sparse-late) and uses a
        configurable subset size. This callback focuses on correlation, shift sensitivity,
        and segmentation probe diagnostics.
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        every_n_epochs: int = 10,
        num_samples: int = 32,
        shift_vox: int = 5,
        eps_au: float = 0.01,
        csv_name: str = "latent_diag/metrics.csv",
        ids_name: str = "latent_diag/ids.txt",
        image_key: str = "image",
        seg_key: str = "seg",
        id_key: str = "id",
        seg_labels: Optional[Dict[str, int]] = None,
    ):
        """Initialize LatentDiagnosticsCallback.

        Args:
            run_dir: Root directory for saving outputs.
            every_n_epochs: Compute diagnostics every N epochs.
            num_samples: Number of validation samples to use.
            shift_vox: Translation shift magnitude in voxels.
            eps_au: Variance threshold for active units (nats).
            csv_name: Relative path for CSV file.
            ids_name: Relative path for sample IDs file.
            image_key: Batch dict key for images.
            seg_key: Batch dict key for segmentations.
            id_key: Batch dict key for sample IDs.
            seg_labels: Segmentation label mapping (default: {"ncr": 1, "ed": 2, "et": 3} for BraTS).
        """
        super().__init__()
        self.run_dir = Path(run_dir)
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.shift_vox = shift_vox
        self.eps_au = eps_au
        self.csv_name = csv_name
        self.ids_name = ids_name
        self.image_key = image_key
        self.seg_key = seg_key
        self.id_key = id_key

        # Segmentation labels (configurable for different datasets)
        if seg_labels is None:
            self.seg_labels = {"ncr": 1, "ed": 2, "et": 3}
        else:
            self.seg_labels = seg_labels

        # Runtime state
        self._sample_ids: Optional[List[str]] = None
        self._val_data: List[Dict[str, Any]] = []
        self.last_epoch_metrics: Optional[Dict[str, Any]] = None

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Reset accumulator and initialize sample IDs if needed."""
        # Only run on diagnostic epochs
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Load or initialize sample IDs (rank 0 only)
        if self._sample_ids is None and trainer.is_global_zero:
            self._sample_ids = self._load_sample_ids()

            # If still None, initialize from validation dataloader
            if self._sample_ids is None:
                val_dataloader = (
                    trainer.val_dataloaders[0]
                    if isinstance(trainer.val_dataloaders, list)
                    else trainer.val_dataloaders
                )
                self._sample_ids = self._initialize_sample_ids(val_dataloader)

                # Write to disk
                ids_path = self.run_dir / self.ids_name
                ids_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ids_path, "w") as f:
                    f.write("\n".join(self._sample_ids))

                logger.info(
                    f"Initialized {len(self._sample_ids)} sample IDs for latent diagnostics"
                )

        # Broadcast to all ranks if DDP
        if trainer.world_size > 1 and self._sample_ids is not None:
            self._sample_ids = self._broadcast_sample_ids(self._sample_ids, trainer)

        # Reset accumulator
        self._val_data = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate μ and metadata for selected samples."""
        # Only run on diagnostic epochs
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Skip if IDs not initialized yet
        if self._sample_ids is None:
            return

        # Filter batch for selected samples
        batch_ids = batch[self.id_key]
        selected_set = set(self._sample_ids)

        # Extract μ for selected samples
        with torch.no_grad():
            pl_module.model.eval()
            mu, _ = pl_module.model.encode(batch[self.image_key])
            mu_cpu = mu.detach().cpu().float()  # Force FP32 on CPU

        # Accumulate data
        for i, bid in enumerate(batch_ids):
            if bid in selected_set:
                self._val_data.append(
                    {
                        "id": bid,
                        "mu": mu_cpu[i],  # [z_dim]
                        "x": batch[self.image_key][i].cpu(),  # [4, 128, 128, 128]
                        "seg": (
                            batch[self.seg_key][i].cpu()
                            if self.seg_key in batch
                            else None
                        ),  # [1, 128, 128, 128] or None
                    }
                )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Compute and log diagnostics on rank 0."""
        # Only run on diagnostic epochs
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Only process on rank 0
        if not trainer.is_global_zero:
            return

        # Check if we have data
        if len(self._val_data) == 0:
            logger.warning(
                f"No samples accumulated for diagnostics at epoch {trainer.current_epoch}"
            )
            return

        # Deduplicate by ID (in case of multi-GPU issues)
        data_dict = {d["id"]: d for d in self._val_data}
        unique_data = list(data_dict.values())

        if len(unique_data) < 2:
            logger.warning(
                f"Only {len(unique_data)} samples available, skipping diagnostics (need at least 2)"
            )
            return

        # Compute diagnostics
        try:
            metrics = self._compute_diagnostics(unique_data, pl_module, trainer)
        except Exception as e:
            logger.error(f"Failed to compute diagnostics: {e}", exc_info=True)
            return

        # Log and save
        self._log_metrics(metrics, pl_module)
        self._save_csv(metrics, trainer)

        # Export metrics for TidyEpochCSVCallback to merge
        # Include all scalar diagnostic metrics with diag/* namespace
        self.last_epoch_metrics = {
            "epoch": metrics["epoch"],
            "diag/corr_offdiag_meanabs": metrics["corr_offdiag_meanabs"],
            "diag/shift_mu_l2_mean": metrics.get("shift_mu_l2_mean", np.nan),
            "diag/shift_mu_absmean": metrics.get("shift_mu_absmean", np.nan),
            "diag/r2_logV_ncr": metrics.get("r2_logV_ncr", np.nan),
            "diag/r2_logV_ed": metrics.get("r2_logV_ed", np.nan),
            "diag/r2_logV_et": metrics.get("r2_logV_et", np.nan),
            "diag/r2_logV_total": metrics.get("r2_logV_total", np.nan),
            "diag/r2_cz_total": metrics.get("r2_cz_total", np.nan),
            "diag/r2_cy_total": metrics.get("r2_cy_total", np.nan),
            "diag/r2_cx_total": metrics.get("r2_cx_total", np.nan),
            "diag/r2_r_ncr": metrics.get("r2_r_ncr", np.nan),
            "diag/r2_r_ed": metrics.get("r2_r_ed", np.nan),
            "diag/r2_r_et": metrics.get("r2_r_et", np.nan),
        }

        # Clear accumulator
        self._val_data = []

    def _load_sample_ids(self) -> Optional[List[str]]:
        """Load previously persisted sample IDs.

        Returns:
            List of sample IDs if file exists, None otherwise.
        """
        ids_path = self.run_dir / self.ids_name
        if not ids_path.exists():
            return None

        with open(ids_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _initialize_sample_ids(self, dataloader) -> List[str]:
        """Select first N samples from dataloader deterministically.

        Args:
            dataloader: Validation dataloader.

        Returns:
            List of sample IDs (sorted for determinism).
        """
        ids = []
        for batch in dataloader:
            ids.extend(batch[self.id_key])
            if len(ids) >= self.num_samples:
                break

        # Take exactly num_samples, sort for determinism
        selected_ids = sorted(ids[: self.num_samples])
        return selected_ids

    def _broadcast_sample_ids(
        self, ids: Optional[List[str]], trainer: pl.Trainer
    ) -> List[str]:
        """Broadcast sample IDs from rank 0 to all ranks in DDP.

        Args:
            ids: Sample IDs on rank 0, None on other ranks.
            trainer: Lightning trainer.

        Returns:
            Sample IDs on all ranks.
        """
        if trainer.world_size == 1:
            return ids

        import torch.distributed as dist

        # Rank 0 prepares data
        if trainer.is_global_zero:
            id_str = "\n".join(ids)
            id_bytes = id_str.encode("utf-8")
            length = len(id_bytes)
        else:
            length = 0

        # Broadcast length
        length_tensor = torch.tensor(
            length, dtype=torch.long, device=trainer.strategy.root_device
        )
        dist.broadcast(length_tensor, src=0)
        length = length_tensor.item()

        # Broadcast string
        if trainer.is_global_zero:
            data_tensor = torch.frombuffer(id_bytes, dtype=torch.uint8).to(
                trainer.strategy.root_device
            )
        else:
            data_tensor = torch.zeros(
                length, dtype=torch.uint8, device=trainer.strategy.root_device
            )

        dist.broadcast(data_tensor, src=0)

        # Decode on all ranks
        id_str = data_tensor.cpu().numpy().tobytes().decode("utf-8")
        return id_str.split("\n")

    def _compute_diagnostics(
        self,
        data_list: List[Dict[str, Any]],
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
    ) -> Dict[str, Any]:
        """Run all diagnostic computations.

        Args:
            data_list: List of {id, mu, x, seg} dictionaries.
            pl_module: Lightning module with model.
            trainer: Lightning trainer.

        Returns:
            Dictionary with all metrics for CSV row.
        """
        # Build mu matrix
        mu_matrix, z_dim = self._build_mu_matrix(data_list)
        N = mu_matrix.shape[0]

        metrics = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "num_samples_planned": self.num_samples,
            "num_samples_used": N,
            "z_dim": z_dim,
        }

        # Correlation
        corr_metrics = self._compute_correlation(mu_matrix)
        metrics.update(corr_metrics)

        # Shift sensitivity
        try:
            shift_metrics = self._compute_shift_sensitivity(
                data_list, pl_module, self.shift_vox
            )
            metrics.update(shift_metrics)
        except Exception as e:
            logger.error(f"Shift sensitivity computation failed: {e}")
            metrics.update(
                {
                    "shift_vox": self.shift_vox,
                    "shift_mu_l2_mean": np.nan,
                    "shift_mu_absmean": np.nan,
                }
            )

        # Segmentation probes
        targets_df, seg_available = self._extract_seg_targets(data_list)
        metrics["seg_available"] = seg_available

        if seg_available:
            probe_results = self._ridge_probe(mu_matrix, targets_df)
            metrics.update(probe_results)
        else:
            # Fill with defaults
            for target in ["ncr", "ed", "et", "total"]:
                metrics[f"n_empty_{target}"] = 0

            for target_col in [
                "logV_ncr",
                "logV_ed",
                "logV_et",
                "logV_total",
                "cz_total",
                "cy_total",
                "cx_total",
                "r_ncr",
                "r_ed",
                "r_et",
            ]:
                metrics[f"r2_{target_col}"] = np.nan
                if target_col in ["logV_total", "cx_total", "cy_total", "cz_total"]:
                    metrics[f"top5dims_{target_col}"] = ""

        return metrics

    def _build_mu_matrix(
        self, data_list: List[Dict[str, Any]]
    ) -> tuple[torch.Tensor, int]:
        """Stack all mu vectors into matrix.

        Args:
            data_list: List of dictionaries with 'mu' key.

        Returns:
            mu_matrix: [N, z_dim] float32 CPU tensor.
            z_dim: Latent dimensionality.
        """
        mus = [d["mu"] for d in data_list]
        mu_matrix = torch.stack(mus, dim=0)  # [N, z_dim]
        assert mu_matrix.dtype == torch.float32
        return mu_matrix, mu_matrix.shape[1]

    def _compute_correlation(self, mu_matrix: torch.Tensor) -> Dict[str, float]:
        """Mean absolute off-diagonal correlation.

        Args:
            mu_matrix: [N, z_dim] latent means.

        Returns:
            Dictionary with 'corr_offdiag_meanabs'.
        """
        # Standardize
        mu_mean = mu_matrix.mean(dim=0)
        mu_std_val = mu_matrix.std(dim=0, unbiased=True)
        mu_std = (mu_matrix - mu_mean) / (mu_std_val + 1e-8)

        # Correlation matrix
        N = mu_std.shape[0]
        corr = torch.mm(mu_std.T, mu_std) / (N - 1)  # [z_dim, z_dim]

        # Off-diagonal elements
        z_dim = corr.shape[0]
        mask = ~torch.eye(z_dim, dtype=torch.bool)
        off_diag = corr[mask]

        return {"corr_offdiag_meanabs": off_diag.abs().mean().item()}

    def _compute_shift_sensitivity(
        self,
        data_list: List[Dict[str, Any]],
        pl_module: pl.LightningModule,
        shift_vox: int,
    ) -> Dict[str, float]:
        """Measure change in μ under random spatial shift.

        Args:
            data_list: List of {id, mu, x, seg} dictionaries.
            pl_module: Lightning module with model.
            shift_vox: Maximum shift magnitude per axis.

        Returns:
            Dictionary with shift_vox, shift_mu_l2_mean, shift_mu_absmean.
        """
        delta_norms = []
        delta_absmeans = []

        # Set generator for reproducible random shifts
        rng = torch.Generator().manual_seed(42)

        with torch.no_grad():
            pl_module.model.eval()

            for d in data_list:
                x = d["x"]  # [4, 128, 128, 128]
                mu_orig = d["mu"]  # [z_dim]

                # Random shift
                shift_d = torch.randint(-shift_vox, shift_vox + 1, (1,), generator=rng).item()
                shift_h = torch.randint(-shift_vox, shift_vox + 1, (1,), generator=rng).item()
                shift_w = torch.randint(-shift_vox, shift_vox + 1, (1,), generator=rng).item()

                x_shifted = self._shift_image_no_wrap(x, shift_d, shift_h, shift_w)

                # Re-encode (need to move to device)
                x_shifted_batch = x_shifted.unsqueeze(0).to(pl_module.device)
                mu_shifted, _ = pl_module.model.encode(x_shifted_batch)
                mu_shifted = mu_shifted.squeeze(0).cpu().float()

                # Compute change
                delta = mu_shifted - mu_orig
                delta_norms.append(delta.norm(p=2).item())
                delta_absmeans.append(delta.abs().mean().item())

        return {
            "shift_vox": shift_vox,
            "shift_mu_l2_mean": float(np.mean(delta_norms)),
            "shift_mu_absmean": float(np.mean(delta_absmeans)),
        }

    def _shift_image_no_wrap(
        self, x: torch.Tensor, shift_d: int, shift_h: int, shift_w: int
    ) -> torch.Tensor:
        """Shift image by (shift_d, shift_h, shift_w) voxels without wrap-around.

        Args:
            x: [C, D, H, W] image tensor.
            shift_d: Signed shift in depth dimension.
            shift_h: Signed shift in height dimension.
            shift_w: Signed shift in width dimension.

        Returns:
            x_shifted: [C, D, H, W] with zeros in vacated regions.
        """
        C, D, H, W = x.shape
        x_shifted = torch.zeros_like(x)

        # Source slice ranges
        d_src_start = max(0, -shift_d)
        d_src_end = min(D, D - shift_d)
        h_src_start = max(0, -shift_h)
        h_src_end = min(H, H - shift_h)
        w_src_start = max(0, -shift_w)
        w_src_end = min(W, W - shift_w)

        # Destination slice ranges
        d_dst_start = max(0, shift_d)
        d_dst_end = d_dst_start + (d_src_end - d_src_start)
        h_dst_start = max(0, shift_h)
        h_dst_end = h_dst_start + (h_src_end - h_src_start)
        w_dst_start = max(0, shift_w)
        w_dst_end = w_dst_start + (w_src_end - w_src_start)

        x_shifted[
            :, d_dst_start:d_dst_end, h_dst_start:h_dst_end, w_dst_start:w_dst_end
        ] = x[:, d_src_start:d_src_end, h_src_start:h_src_end, w_src_start:w_src_end]

        return x_shifted

    def _extract_seg_targets(
        self, data_list: List[Dict[str, Any]]
    ) -> tuple[Optional[pd.DataFrame], int]:
        """Extract volume, centroid, and ratio features from segmentations.

        Args:
            data_list: List of {id, mu, x, seg} dictionaries.

        Returns:
            df: DataFrame with volume/centroid/ratio columns, or None if no seg.
            seg_available: 1 if any seg exists, 0 otherwise.
        """
        # Check if any seg exists
        has_seg = any(d["seg"] is not None for d in data_list)
        if not has_seg:
            return None, 0

        targets = []

        for d in data_list:
            if d["seg"] is None:
                # Fill with NaN
                targets.append(
                    {
                        "logV_ncr": np.nan,
                        "logV_ed": np.nan,
                        "logV_et": np.nan,
                        "logV_total": np.nan,
                        "cz_total": np.nan,
                        "cy_total": np.nan,
                        "cx_total": np.nan,
                        "r_ncr": np.nan,
                        "r_ed": np.nan,
                        "r_et": np.nan,
                    }
                )
                continue

            seg = d["seg"].squeeze(0)  # [128, 128, 128]

            # Volumes (voxel counts)
            # Use configurable segmentation labels
            ncr_label = self.seg_labels["ncr"]
            ed_label = self.seg_labels["ed"]
            et_label = self.seg_labels["et"]

            V_ncr = (seg == ncr_label).sum().item()
            V_ed = (seg == ed_label).sum().item()
            V_et = (seg == et_label).sum().item()
            V_total = V_ncr + V_ed + V_et

            # Log volumes (with offset to handle zeros)
            logV_ncr = np.log1p(V_ncr)
            logV_ed = np.log1p(V_ed)
            logV_et = np.log1p(V_et)
            logV_total = np.log1p(V_total)

            # Centroid (only for total tumor mask)
            if V_total > 0:
                mask_total = seg > 0
                coords = torch.nonzero(mask_total, as_tuple=False).float()  # [N_voxels, 3]
                centroid = coords.mean(dim=0)  # [3]: (D_idx, H_idx, W_idx)
                cz, cy, cx = centroid.tolist()
            else:
                cz, cy, cx = np.nan, np.nan, np.nan

            # Ratios (handle division by zero)
            if V_total > 0:
                r_ncr = V_ncr / V_total
                r_ed = V_ed / V_total
                r_et = V_et / V_total
            else:
                r_ncr = r_ed = r_et = np.nan

            targets.append(
                {
                    "logV_ncr": logV_ncr,
                    "logV_ed": logV_ed,
                    "logV_et": logV_et,
                    "logV_total": logV_total,
                    "cz_total": cz,
                    "cy_total": cy,
                    "cx_total": cx,
                    "r_ncr": r_ncr,
                    "r_ed": r_ed,
                    "r_et": r_et,
                }
            )

        df = pd.DataFrame(targets)
        return df, 1

    def _ridge_probe(
        self, mu_matrix: torch.Tensor, targets_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Ridge regression from standardized μ to each target.

        Args:
            mu_matrix: [N, z_dim] float32 latent means.
            targets_df: DataFrame with target columns.

        Returns:
            Dictionary with r2_{target}, top5dims_{target}, n_empty_{target}.
        """
        # Standardize μ
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(mu_matrix.numpy())  # [N, z_dim]

        results = {}

        # Define targets to probe
        target_cols = [
            "logV_ncr",
            "logV_ed",
            "logV_et",
            "logV_total",
            "cz_total",
            "cy_total",
            "cx_total",
            "r_ncr",
            "r_ed",
            "r_et",
        ]

        # Track empty counts
        empty_counts = {"ncr": 0, "ed": 0, "et": 0, "total": 0}

        for target_col in target_cols:
            y = targets_df[target_col].values  # [N]

            # Count invalid samples
            valid_mask = np.isfinite(y)
            n_valid = valid_mask.sum()
            n_empty = len(y) - n_valid

            # Update empty counts for volume targets
            if target_col.startswith("logV_"):
                compartment = target_col.replace("logV_", "")
                empty_counts[compartment] = n_empty

            # Skip if too few valid samples
            if n_valid < 2:
                logger.warning(
                    f"Target {target_col} has only {n_valid} valid samples, skipping regression"
                )
                results[f"r2_{target_col}"] = np.nan
                # Only add top5dims for key targets
                if target_col in ["logV_total", "cx_total", "cy_total", "cz_total"]:
                    results[f"top5dims_{target_col}"] = ""
                continue

            # Filter to valid samples
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            # Standardize y
            scaler_y = StandardScaler()
            y_valid_std = scaler_y.fit_transform(y_valid.reshape(-1, 1)).ravel()

            # Ridge regression
            ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=0)
            ridge.fit(X_valid, y_valid_std)
            y_pred = ridge.predict(X_valid)

            # R² score
            r2 = r2_score(y_valid_std, y_pred)
            results[f"r2_{target_col}"] = r2

            # Top-5 coefficient dimensions (only for key targets)
            if target_col in ["logV_total", "cx_total", "cy_total", "cz_total"]:
                coef_abs = np.abs(ridge.coef_)
                top5_idx = np.argsort(coef_abs)[::-1][:5]
                results[f"top5dims_{target_col}"] = ";".join(map(str, top5_idx))

        # Add empty counts
        for compartment, count in empty_counts.items():
            results[f"n_empty_{compartment}"] = count

        return results

    def _log_metrics(
        self, metrics: Dict[str, Any], pl_module: pl.LightningModule
    ) -> None:
        """Log scalar metrics to Lightning loggers.

        Args:
            metrics: Dictionary with all metrics.
            pl_module: Lightning module for logging.
        """
        scalar_keys = [
            "corr_offdiag_meanabs",
            "shift_mu_l2_mean",
            "shift_mu_absmean",
            "r2_logV_ncr",
            "r2_logV_ed",
            "r2_logV_et",
            "r2_logV_total",
            "r2_cz_total",
            "r2_cy_total",
            "r2_cx_total",
            "r2_r_ncr",
            "r2_r_ed",
            "r2_r_et",
        ]

        log_dict = {}
        for k in scalar_keys:
            if k in metrics:
                val = metrics[k]
                # Lightning handles NaN gracefully
                log_dict[f"diag/{k}"] = val

        pl_module.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=True,
        )

        logger.info(
            f"Latent diagnostics at epoch {metrics['epoch']}: "
            f"corr={metrics['corr_offdiag_meanabs']:.4f}"
        )

    def _save_csv(
        self, metrics: Dict[str, Any], trainer: pl.Trainer
    ) -> None:
        """Save metrics to CSV with atomic write and duplicate epoch handling.

        Args:
            metrics: Dictionary with all metrics.
            trainer: Lightning trainer.
        """
        csv_path = self.run_dir / self.csv_name
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Define expected columns
        expected_cols = [
            "epoch",
            "global_step",
            "num_samples_planned",
            "num_samples_used",
            "z_dim",
            "corr_offdiag_meanabs",
            "shift_vox",
            "shift_mu_l2_mean",
            "shift_mu_absmean",
            "seg_available",
            "n_empty_ncr",
            "n_empty_ed",
            "n_empty_et",
            "n_empty_total",
            "r2_logV_ncr",
            "r2_logV_ed",
            "r2_logV_et",
            "r2_logV_total",
            "r2_cz_total",
            "r2_cy_total",
            "r2_cx_total",
            "r2_r_ncr",
            "r2_r_ed",
            "r2_r_et",
            "top5dims_logV_total",
            "top5dims_cx_total",
            "top5dims_cy_total",
            "top5dims_cz_total",
        ]

        # Convert metrics to DataFrame row
        row_df = pd.DataFrame([metrics])

        # Reindex to ensure column order
        for col in expected_cols:
            if col not in row_df.columns:
                if col.startswith("r2_") or col.startswith("n_empty"):
                    row_df[col] = np.nan
                else:
                    row_df[col] = ""

        row_df = row_df[expected_cols]

        # Load existing CSV if present
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)

            # Remove duplicate epoch if exists
            existing_df = existing_df[existing_df["epoch"] != metrics["epoch"]]

            # Append new row
            combined_df = pd.concat([existing_df, row_df], ignore_index=True)
        else:
            combined_df = row_df

        # Sort by epoch
        combined_df = combined_df.sort_values("epoch").reset_index(drop=True)

        # Atomic write
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=csv_path.parent, suffix=".tmp"
        ) as tmp_f:
            tmp_path = Path(tmp_f.name)
            combined_df.to_csv(tmp_f, index=False)

        # Rename
        tmp_path.replace(csv_path)

        logger.info(f"Saved latent diagnostics to {csv_path}")
