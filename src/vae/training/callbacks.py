"""Lightning callbacks for VAE training.

This module implements custom callbacks for:
- Saving reconstruction visualizations at regular intervals
- Custom logging to replace tqdm progress bars with informative console output
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
