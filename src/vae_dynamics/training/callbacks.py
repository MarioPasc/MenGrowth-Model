"""Lightning callbacks for VAE training.

This module implements custom callbacks for:
- Saving reconstruction visualizations at regular intervals
"""

import logging
from pathlib import Path
from typing import Optional

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
                x_hat, _, _ = pl_module.model(batch["image"])
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
