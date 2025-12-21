"""PyTorch Lightning module for VAE training.

This module implements the training logic for the baseline 3D VAE including:
- Training and validation steps with ELBO loss
- KL annealing schedule
- Logging of all loss components and current beta
- AdamW optimizer configuration
"""

import logging
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from ..models import BaselineVAE
from ..losses import compute_elbo
from ..losses.elbo import get_beta_schedule


logger = logging.getLogger(__name__)


class VAELitModule(pl.LightningModule):
    """PyTorch Lightning module for training 3D VAE.

    Handles training loop, validation, loss computation, logging,
    and optimizer configuration.

    Attributes:
        model: The BaselineVAE model.
        lr: Learning rate for optimizer.
        kl_beta: Target beta value for KL term.
        kl_annealing_epochs: Number of epochs for KL annealing.
        current_beta: Current beta value (updated each epoch).
    """

    def __init__(
        self,
        model: BaselineVAE,
        lr: float = 1e-4,
        kl_beta: float = 1.0,
        kl_annealing_epochs: int = 40,
    ):
        """Initialize VAELitModule.

        Args:
            model: BaselineVAE model instance.
            lr: Learning rate for AdamW optimizer.
            kl_beta: Target beta value after annealing.
            kl_annealing_epochs: Number of epochs for linear KL annealing.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.kl_beta = kl_beta
        self.kl_annealing_epochs = kl_annealing_epochs
        self.current_beta = 0.0

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "VAELitModule":
        """Create VAELitModule from configuration.

        Args:
            cfg: Configuration object with model and train parameters.

        Returns:
            Configured VAELitModule instance.
        """
        # Determine num_groups from norm config
        num_groups = 8  # default
        if cfg.model.norm == "GROUP":
            num_groups = 8

        model = BaselineVAE(
            input_channels=cfg.model.input_channels,
            z_dim=cfg.model.z_dim,
            base_filters=cfg.model.base_filters,
            num_groups=num_groups,
        )

        return cls(
            model=model,
            lr=cfg.train.lr,
            kl_beta=cfg.train.kl_beta,
            kl_annealing_epochs=cfg.train.kl_annealing_epochs,
        )

    def on_train_epoch_start(self) -> None:
        """Update beta at the start of each training epoch."""
        self.current_beta = get_beta_schedule(
            epoch=self.current_epoch,
            kl_beta=self.kl_beta,
            kl_annealing_epochs=self.kl_annealing_epochs,
        )
        logger.debug(f"Epoch {self.current_epoch}: beta = {self.current_beta:.4f}")

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through VAE.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            Tuple of (x_hat, mu, logvar).
        """
        return self.model(x)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Execute single training step.

        Args:
            batch: Dict containing "image" tensor [B, C, D, H, W].
            batch_idx: Index of current batch.

        Returns:
            Total loss for optimization.
        """
        x = batch["image"]

        # Forward pass
        x_hat, mu, logvar = self.model(x)

        # Compute ELBO loss
        loss_dict = compute_elbo(x, x_hat, mu, logvar, beta=self.current_beta)

        # Log metrics
        self.log("train/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recon", loss_dict["recon"], on_step=False, on_epoch=True)
        self.log("train/kl", loss_dict["kl"], on_step=False, on_epoch=True)
        self.log("train/beta", self.current_beta, on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Execute single validation step.

        Args:
            batch: Dict containing "image" tensor [B, C, D, H, W].
            batch_idx: Index of current batch.

        Returns:
            Total loss.
        """
        x = batch["image"]

        # Forward pass
        x_hat, mu, logvar = self.model(x)

        # Compute ELBO loss
        loss_dict = compute_elbo(x, x_hat, mu, logvar, beta=self.current_beta)

        # Log metrics
        self.log("val/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon", loss_dict["recon"], on_step=False, on_epoch=True)
        self.log("val/kl", loss_dict["kl"], on_step=False, on_epoch=True)
        self.log("val/beta", self.current_beta, on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer.

        Returns:
            AdamW optimizer instance.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
        )
        return optimizer
