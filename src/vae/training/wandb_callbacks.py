"""Wandb-specific visualization callbacks.

This module provides callbacks for logging visual artifacts to Weights & Biases,
including training dashboards and latent space visualizations.
"""

import torch
import numpy as np
from pathlib import Path
from pytorch_lightning import Callback, LightningModule, Trainer
from typing import Optional

# Import wandb conditionally to avoid hard dependency
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbDashboardCallback(Callback):
    """Generates and logs training dashboard plots to wandb.

    Reuses the existing plot_training_dashboard.py logic to create
    a 3x3 dashboard figure and uploads it to wandb periodically.

    Args:
        run_dir: Path to run directory (contains logs/tidy/epoch_metrics.csv)
        every_n_epochs: Logging frequency (default: 10)

    Example:
        >>> callback = WandbDashboardCallback(run_dir=Path("experiments/runs/exp2"))
        >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(self, run_dir: Path, every_n_epochs: int = 10):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Generate and log dashboard plot.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: LightningModule being trained
        """
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        if trainer.global_rank != 0:
            return  # Only log from rank 0

        if not WANDB_AVAILABLE:
            return

        try:
            # Import plotting function
            from engine.plot_training_dashboard import load_metrics, detect_experiment_type

            # Path to tidy CSV
            csv_path = self.run_dir / "logs" / "tidy" / "epoch_metrics.csv"

            if not csv_path.exists():
                return

            # Load metrics and detect experiment type
            df = load_metrics(self.run_dir)
            experiment_type = detect_experiment_type(df)

            # Generate figure using existing plotting logic
            import matplotlib.pyplot as plt
            from engine.plot_training_dashboard import plot_dashboard

            # Create temporary output path (will be deleted after upload)
            temp_output = self.run_dir / "logs" / "tidy" / "temp_dashboard.png"

            # Generate dashboard
            plot_dashboard(df, experiment_type, temp_output, dpi=100)

            # Log to wandb
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                trainer.logger.experiment.log({
                    "dashboard": wandb.Image(str(temp_output)),
                    "epoch": trainer.current_epoch,
                })

            # Clean up temp file
            if temp_output.exists():
                temp_output.unlink()

        except Exception as e:
            # Silently skip if dashboard generation fails
            # This ensures training continues even if visualization has issues
            pass


class WandbLatentVizCallback(Callback):
    """Visualizes latent space using PCA projections.

    Logs 2D PCA scatter plots of latent embeddings to wandb,
    colored by relevant factors. Useful for monitoring disentanglement
    and latent space structure during training.

    Args:
        every_n_epochs: Logging frequency (default: 20)
        n_samples: Maximum number of samples to visualize (default: 100)

    Example:
        >>> callback = WandbLatentVizCallback(every_n_epochs=20, n_samples=100)
        >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(self, every_n_epochs: int = 20, n_samples: int = 100):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Generate and log latent space visualization.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: LightningModule being trained
        """
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        if trainer.global_rank != 0:
            return

        if not WANDB_AVAILABLE:
            return

        try:
            # Collect latent codes from validation set
            val_loader = trainer.val_dataloaders

            latents = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= self.n_samples:
                        break

                    # Move to device
                    x = batch["image"].to(pl_module.device)

                    # Encode (handle both model.encode and model.forward)
                    if hasattr(pl_module.model, 'encode'):
                        mu, _ = pl_module.model.encode(x)
                    else:
                        # Fallback: use full forward pass
                        outputs = pl_module.model(x)
                        if isinstance(outputs, tuple) and len(outputs) >= 3:
                            _, mu, _ = outputs[:3]
                        else:
                            return  # Can't extract latents

                    latents.append(mu.cpu())

            if len(latents) == 0:
                return

            latents = torch.cat(latents, dim=0)  # [N, z_dim]

            # PCA projection
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents.numpy())

            # Create scatter plot
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6, s=20, c='steelblue', edgecolors='white', linewidth=0.5)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax.set_title(f"Latent Space PCA (Epoch {trainer.current_epoch})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            # Log to wandb
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                trainer.logger.experiment.log({
                    "latent_pca": wandb.Image(fig),
                    "epoch": trainer.current_epoch,
                })

            plt.close(fig)

        except Exception as e:
            # Silently skip if visualization fails
            pass
