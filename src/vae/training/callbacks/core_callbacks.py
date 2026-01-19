"""Core training callbacks for VAE training.

This module implements essential training utilities:
- ReconstructionCallback: Saves reconstruction visualizations at regular intervals
- TrainingLoggingCallback: Custom console logging to replace tqdm progress bars
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

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
            # Get length of the first validation dataloader (handles both list and single dataloader cases)
            try:
                val_dl = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
                total_batches = len(val_dl)
            except (TypeError, AttributeError):
                total_batches = "?"
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
        <run_dir>/visualizations/reconstructions/epoch_<E>/sample_<S>_<mod>_grid.png
    """

    def __init__(
        self,
        run_dir: str,
        recon_every_n_epochs: int = 5,
        num_recon_samples: int = 2,
        modality_names: Optional[list] = None,
        log_to_wandb: bool = False,
    ):
        """Initialize ReconstructionCallback.

        Args:
            run_dir: Path to run directory for saving outputs.
            recon_every_n_epochs: Save reconstructions every N epochs.
            num_recon_samples: Number of samples to visualize.
            modality_names: Names of modality channels for labeling.
            log_to_wandb: Whether to log reconstructions to wandb.
        """
        super().__init__()
        self.run_dir = Path(run_dir)
        self.recon_base_dir = self.run_dir / "visualizations" / "reconstructions"
        self.recon_every_n_epochs = recon_every_n_epochs
        self.num_recon_samples = num_recon_samples
        self.modality_names = modality_names or ["T1c", "T1n", "T2f", "T2w"]
        self.log_to_wandb = log_to_wandb

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

        # Store trainer reference for wandb logging
        self._current_trainer = trainer

        # Collect samples
        all_x = torch.cat([o["x"] for o in self._val_outputs], dim=0)
        all_x_hat = torch.cat([o["x_hat"] for o in self._val_outputs], dim=0)

        num_samples = min(self.num_recon_samples, len(all_x))

        # Create output directory
        epoch_dir = self.recon_base_dir / f"epoch_{epoch:04d}"
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
        """Save visualization for a single sample as a 3x3 grid per modality.

        Each modality gets one image with:
        - Rows: Axial, Coronal, Sagittal views
        - Columns: Input, Reconstruction, Absolute Difference

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

            # Extract central slices for all three views
            # Each entry: (original, reconstruction, difference)
            slices = {
                "Axial": (x_mod[center_d, :, :], x_hat_mod[center_d, :, :], diff[center_d, :, :]),
                "Coronal": (x_mod[:, center_h, :], x_hat_mod[:, center_h, :], diff[:, center_h, :]),
                "Sagittal": (x_mod[:, :, center_w], x_hat_mod[:, :, center_w], diff[:, :, center_w]),
            }

            self._save_grid_figure(
                slices,
                mod_name,
                sample_idx,
                output_dir,
            )

    def _save_grid_figure(
        self,
        slices: dict,
        modality: str,
        sample_idx: int,
        output_dir: Path,
    ) -> None:
        """Save a 3x3 grid figure showing all views for a modality.

        Grid layout:
        - Rows: Axial, Coronal, Sagittal
        - Columns: Input, Reconstruction, Absolute Difference

        Args:
            slices: Dict mapping view name to (original, reconstruction, difference) tuples.
            modality: Modality name for title.
            sample_idx: Sample index.
            output_dir: Output directory.
        """
        view_names = ["Axial", "Coronal", "Sagittal"]
        col_names = ["Input", "Reconstruction", "Difference"]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)

        for row_idx, view_name in enumerate(view_names):
            orig, recon, diff = slices[view_name]

            # Determine color limits from original for consistent scaling
            vmin, vmax = orig.min(), orig.max()

            # Input
            axes[row_idx, 0].imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 0].axis('off')

            # Reconstruction
            axes[row_idx, 1].imshow(recon, cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 1].axis('off')

            # Difference
            im = axes[row_idx, 2].imshow(diff, cmap='hot')
            axes[row_idx, 2].axis('off')

            # Row labels on the left
            axes[row_idx, 0].set_ylabel(view_name, fontsize=12, rotation=90, labelpad=10)
            axes[row_idx, 0].yaxis.set_visible(True)
            axes[row_idx, 0].tick_params(left=False, labelleft=False)

        # Column titles
        for col_idx, col_name in enumerate(col_names):
            axes[0, col_idx].set_title(col_name, fontsize=12)

        # Add colorbar for difference column
        fig.colorbar(im, ax=axes[:, 2], fraction=0.02, pad=0.02, label='Abs. Diff.')

        fig.suptitle(f"{modality} - Sample {sample_idx}", fontsize=14)

        filename = f"sample_{sample_idx:02d}_{modality.lower()}_grid.png"
        save_path = output_dir / filename
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

        # Log to wandb if enabled
        if self.log_to_wandb:
            try:
                import wandb
                if hasattr(self, '_current_trainer') and self._current_trainer.logger and hasattr(self._current_trainer.logger, 'experiment'):
                    self._current_trainer.logger.experiment.log({
                        f"reconstructions/{modality.lower()}_sample_{sample_idx}": wandb.Image(str(save_path)),
                        "epoch": self._current_trainer.current_epoch,
                    })
            except Exception:
                pass  # Silently skip if wandb not available

        plt.close(fig)
