"""PyTorch Lightning modules for VAE training.

This module implements training logic for:
- Exp1: Baseline 3D VAE with ELBO loss (VAELitModule)
- Exp2: β-TCVAE with SBD and TC loss decomposition (TCVAELitModule)

Both modules include:
- Training and validation steps with respective losses
- Annealing schedules for KL/TC weights
- Logging of all loss components
- AdamW optimizer configuration
"""

import logging
from typing import Any, Dict, Tuple

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from ..models import BaselineVAE, TCVAESBD
from ..losses import compute_elbo, compute_tcvae_loss
from ..losses.elbo import get_beta_schedule
from ..losses.tcvae import get_beta_tc_schedule


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


class TCVAELitModule(pl.LightningModule):
    """PyTorch Lightning module for training β-TCVAE with SBD.

    Handles training loop, validation, TC-decomposed loss computation,
    logging, and optimizer configuration.

    The loss decomposes into:
    - Reconstruction (MSE sum)
    - Mutual Information (MI)
    - Total Correlation (TC) - weighted by beta_tc
    - Dimension-wise KL (DWKL)

    Attributes:
        model: The TCVAESBD model.
        n_train: Number of training samples (for MWS estimator).
        lr: Learning rate for optimizer.
        alpha: Weight for MI term.
        beta_tc_target: Target beta_tc value after annealing.
        gamma: Weight for DWKL term.
        beta_tc_annealing_epochs: Number of epochs for TC annealing.
        compute_in_fp32: Whether to compute TC terms in fp32.
        current_beta_tc: Current beta_tc value (updated each epoch).
    """

    def __init__(
        self,
        model: TCVAESBD,
        n_train: int,
        lr: float = 1e-4,
        alpha: float = 1.0,
        beta_tc_target: float = 6.0,
        gamma: float = 1.0,
        beta_tc_annealing_epochs: int = 40,
        compute_in_fp32: bool = True,
    ):
        """Initialize TCVAELitModule.

        Args:
            model: TCVAESBD model instance.
            n_train: Number of training samples (N for MWS estimator).
            lr: Learning rate for AdamW optimizer.
            alpha: Weight for MI term (default 1.0).
            beta_tc_target: Target beta_tc value after annealing.
            gamma: Weight for DWKL term (default 1.0).
            beta_tc_annealing_epochs: Number of epochs for TC annealing.
            compute_in_fp32: Whether to compute TC terms in float32.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.n_train = n_train
        self.lr = lr
        self.alpha = alpha
        self.beta_tc_target = beta_tc_target
        self.gamma = gamma
        self.beta_tc_annealing_epochs = beta_tc_annealing_epochs
        self.compute_in_fp32 = compute_in_fp32
        self.current_beta_tc = 0.0

    @classmethod
    def from_config(cls, cfg: DictConfig, n_train: int) -> "TCVAELitModule":
        """Create TCVAELitModule from configuration.

        Args:
            cfg: Configuration object with model, loss, and train parameters.
            n_train: Number of training samples.

        Returns:
            Configured TCVAELitModule instance.
        """
        # Determine num_groups from norm config
        num_groups = 8
        if cfg.model.norm == "GROUP":
            num_groups = 8

        # Get SBD grid size
        sbd_grid_size = tuple(cfg.model.sbd_grid_size)

        # Get gradient checkpointing flag
        gradient_checkpointing = cfg.train.get("gradient_checkpointing", False)

        model = TCVAESBD(
            input_channels=cfg.model.input_channels,
            z_dim=cfg.model.z_dim,
            base_filters=cfg.model.base_filters,
            num_groups=num_groups,
            sbd_grid_size=sbd_grid_size,
            gradient_checkpointing=gradient_checkpointing,
        )

        return cls(
            model=model,
            n_train=n_train,
            lr=cfg.train.lr,
            alpha=cfg.loss.alpha,
            beta_tc_target=cfg.loss.beta_tc_target,
            gamma=cfg.loss.gamma,
            beta_tc_annealing_epochs=cfg.loss.beta_tc_annealing_epochs,
            compute_in_fp32=cfg.loss.get("compute_in_fp32", True),
        )

    def on_train_epoch_start(self) -> None:
        """Update beta_tc at the start of each training epoch."""
        self.current_beta_tc = get_beta_tc_schedule(
            epoch=self.current_epoch,
            beta_tc_target=self.beta_tc_target,
            beta_tc_annealing_epochs=self.beta_tc_annealing_epochs,
        )
        logger.debug(f"Epoch {self.current_epoch}: beta_tc = {self.current_beta_tc:.4f}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through TCVAE+SBD.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            Tuple of (x_hat, mu, logvar, z).
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
        x_hat, mu, logvar, z = self.model(x)

        # Compute β-TCVAE loss
        loss_dict = compute_tcvae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            n_data=self.n_train,
            alpha=self.alpha,
            beta_tc=self.current_beta_tc,
            gamma=self.gamma,
            compute_in_fp32=self.compute_in_fp32,
        )

        # Log metrics
        self.log("train/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recon", loss_dict["recon"], on_step=False, on_epoch=True)
        self.log("train/mi", loss_dict["mi"], on_step=False, on_epoch=True)
        self.log("train/tc", loss_dict["tc"], on_step=False, on_epoch=True)
        self.log("train/dwkl", loss_dict["dwkl"], on_step=False, on_epoch=True)
        self.log("train/beta_tc", self.current_beta_tc, on_step=False, on_epoch=True)

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
        x_hat, mu, logvar, z = self.model(x)

        # Compute β-TCVAE loss
        loss_dict = compute_tcvae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            n_data=self.n_train,
            alpha=self.alpha,
            beta_tc=self.current_beta_tc,
            gamma=self.gamma,
            compute_in_fp32=self.compute_in_fp32,
        )

        # Log metrics
        self.log("val/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon", loss_dict["recon"], on_step=False, on_epoch=True)
        self.log("val/mi", loss_dict["mi"], on_step=False, on_epoch=True)
        self.log("val/tc", loss_dict["tc"], on_step=False, on_epoch=True)
        self.log("val/dwkl", loss_dict["dwkl"], on_step=False, on_epoch=True)
        self.log("val/beta_tc", self.current_beta_tc, on_step=False, on_epoch=True)

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
