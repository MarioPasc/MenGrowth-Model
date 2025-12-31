"""PyTorch Lightning modules for VAE training.

This module implements training logic for:
- Exp1: Baseline 3D VAE with ELBO loss (VAELitModule)
- Exp2: DIP-VAE with configurable decoder (standard or SBD) and covariance regularization (DIPVAELitModule)

All modules include:
- Training and validation steps with respective losses
- Annealing schedules for KL/covariance weights
- Logging of all loss components
- AdamW optimizer configuration
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig

from ..models import BaselineVAE
from ..losses import (
    compute_elbo,
    compute_dipvae_loss,
    get_capacity_schedule,
    get_lambda_cov_schedule,
)
from ..losses.elbo import get_beta_schedule
from vae.metrics import compute_ssim_2d_slices, compute_psnr_3d


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
        loss_reduction: str = "mean",
        # New parameters for posterior collapse mitigation
        kl_annealing_type: str = "cyclical",
        kl_annealing_cycles: int = 4,
        kl_annealing_ratio: float = 0.5,
        kl_free_bits: float = 0.5,
        kl_free_bits_mode: str = "batch_mean",
        kl_target_capacity: Optional[float] = None,
        kl_capacity_anneal_epochs: int = 100,
        posterior_logvar_min: float = -6.0,
    ):
        """Initialize VAELitModule.

        Args:
            model: BaselineVAE model instance.
            lr: Learning rate for AdamW optimizer.
            kl_beta: Target beta value after annealing.
            kl_annealing_epochs: Number of epochs for KL annealing.
            loss_reduction: Loss reduction strategy ("mean" or "sum").
                           Default "mean" for numerical stability in FP16.
            kl_annealing_type: Annealing type ("linear" or "cyclical").
                              Default "cyclical" for posterior collapse mitigation.
            kl_annealing_cycles: Number of cycles for cyclical annealing. Default: 4.
            kl_annealing_ratio: Fraction of each cycle for annealing. Default: 0.5.
            kl_free_bits: Free bits threshold per dimension (nats). Default: 0.5.
            kl_free_bits_mode: Free bits clamping mode ("per_sample" or "batch_mean").
                              Default: "batch_mean" (recommended for small batches).
            kl_target_capacity: Target capacity (nats). None disables. Default: None.
            kl_capacity_anneal_epochs: Epochs to reach target capacity. Default: 100.
            posterior_logvar_min: Minimum value for logvar (clamping). Default: -6.0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.kl_beta = kl_beta
        self.kl_annealing_epochs = kl_annealing_epochs
        self.loss_reduction = loss_reduction
        self.current_beta = 0.0
        # New parameters
        self.kl_annealing_type = kl_annealing_type
        self.kl_annealing_cycles = kl_annealing_cycles
        self.kl_annealing_ratio = kl_annealing_ratio
        self.kl_free_bits = kl_free_bits
        self.kl_free_bits_mode = kl_free_bits_mode
        self.kl_target_capacity = kl_target_capacity
        self.kl_capacity_anneal_epochs = kl_capacity_anneal_epochs
        self.current_capacity = 0.0 if kl_target_capacity is not None else None
        self.posterior_logvar_min = posterior_logvar_min

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "VAELitModule":
        """Create VAELitModule from configuration.

        Uses model_factory to build model (consistent with DIPVAELitModule pattern).

        Args:
            cfg: Configuration object with model and train parameters.

        Returns:
            Configured VAELitModule instance.
        """
        from engine.model_factory import create_vae_model

        # Use factory to create model (respects use_sbd flag if present)
        model = create_vae_model(cfg)

        return cls(
            model=model,
            lr=cfg.train.lr,
            kl_beta=cfg.train.kl_beta,
            kl_annealing_epochs=cfg.train.kl_annealing_epochs,
            loss_reduction=cfg.train.get("loss_reduction", "mean"),
            # New parameters with backward-compatible defaults
            kl_annealing_type=cfg.train.get("kl_annealing_type", "cyclical"),
            kl_annealing_cycles=cfg.train.get("kl_annealing_cycles", 4),
            kl_annealing_ratio=cfg.train.get("kl_annealing_ratio", 0.5),
            kl_free_bits=cfg.train.get("kl_free_bits", 0.5),
            kl_free_bits_mode=cfg.train.get("kl_free_bits_mode", "batch_mean"),
            kl_target_capacity=cfg.train.get("kl_target_capacity", None),
            kl_capacity_anneal_epochs=cfg.train.get("kl_capacity_anneal_epochs", 100),
            posterior_logvar_min=cfg.train.get("posterior_logvar_min", -6.0),
        )

    def on_train_epoch_start(self) -> None:
        """Update beta and capacity at the start of each training epoch."""
        # Update beta schedule
        self.current_beta = get_beta_schedule(
            epoch=self.current_epoch,
            kl_beta=self.kl_beta,
            kl_annealing_epochs=self.kl_annealing_epochs,
            kl_annealing_type=self.kl_annealing_type,
            kl_annealing_cycles=self.kl_annealing_cycles,
            kl_annealing_ratio=self.kl_annealing_ratio,
        )

        # Update capacity schedule (if enabled)
        if self.kl_target_capacity is not None:
            self.current_capacity = get_capacity_schedule(
                epoch=self.current_epoch,
                kl_target_capacity=self.kl_target_capacity,
                kl_capacity_anneal_epochs=self.kl_capacity_anneal_epochs,
            )
            logger.debug(
                f"Epoch {self.current_epoch}: "
                f"beta = {self.current_beta:.4f}, "
                f"capacity = {self.current_capacity:.4f}"
            )
        else:
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

        # Forward pass (now returns 4 values for consistency with VAESBD)
        x_hat, mu, logvar, z = self.model(x)

        # NOTE: logvar is already clamped at the source in model.encode()

        # Compute ELBO loss
        loss_dict = compute_elbo(
            x,
            x_hat,
            mu,
            logvar,
            beta=self.current_beta,
            reduction=self.loss_reduction,
            kl_free_bits=self.kl_free_bits,
            kl_free_bits_mode=self.kl_free_bits_mode,
            kl_capacity=self.current_capacity,
        )

        # === STEP LOGGING (minimal) ===
        self.log("train_step/loss", loss_dict["loss"], on_step=True, on_epoch=False, prog_bar=False)

        # === EPOCH LOGGING (comprehensive) ===
        self.log("train_epoch/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_epoch/recon", loss_dict["recon"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl", loss_dict["kl"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl_raw", loss_dict["kl_raw"], on_step=False, on_epoch=True)

        # Normalized metrics (resolution-independent)
        z_dim = mu.shape[1]
        # Note: recon is already mean-reduced (per-element MSE), no additional normalization needed
        self.log("train_epoch/kl_per_dim", loss_dict["kl_raw"] / z_dim, on_step=False, on_epoch=True)

        # Latent statistics
        self.log("train_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)

        # === SCHEDULE LOGGING ===
        self.log("sched/beta", self.current_beta, on_step=False, on_epoch=True)
        if self.current_capacity is not None:
            self.log("sched/capacity", self.current_capacity, on_step=False, on_epoch=True)

        # Schedule state (for cyclical annealing)
        if self.kl_annealing_type == "cyclical":
            cycle_length = self.kl_annealing_epochs / self.kl_annealing_cycles
            cycle_idx = int(self.current_epoch / cycle_length)
            phase = (self.current_epoch % cycle_length) / cycle_length
            self.log("sched/cycle_idx", float(cycle_idx), on_step=False, on_epoch=True)
            self.log("sched/phase", phase, on_step=False, on_epoch=True)

        # === COLLAPSE PROXIES ===
        # Expected KL floor from free bits (cite: Kingma et al. 2016, arXiv:1606.04934)
        if self.kl_free_bits > 0.0:
            expected_kl_floor = z_dim * self.kl_free_bits
            self.log("sched/expected_kl_floor", expected_kl_floor, on_step=False, on_epoch=True)
            self.log("sched/free_bits", self.kl_free_bits, on_step=False, on_epoch=True)

        # === OPTIMIZER LOGGING ===
        opt = self.optimizers()
        if opt is not None:
            current_lr = opt.param_groups[0]["lr"]
            self.log("opt/lr", current_lr, on_step=False, on_epoch=True)

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

        # Forward pass (now returns 4 values for consistency with VAESBD)
        x_hat, mu, logvar, z = self.model(x)

        # NOTE: logvar is already clamped at the source in model.encode()

        # Compute ELBO loss
        loss_dict = compute_elbo(
            x,
            x_hat,
            mu,
            logvar,
            beta=self.current_beta,
            reduction=self.loss_reduction,
            kl_free_bits=self.kl_free_bits,
            kl_free_bits_mode=self.kl_free_bits_mode,
            kl_capacity=self.current_capacity,
        )

        # === STEP LOGGING (minimal) ===
        self.log("val_step/loss", loss_dict["loss"], on_step=True, on_epoch=False, prog_bar=False)

        # === EPOCH LOGGING (comprehensive) ===
        self.log("val_epoch/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_epoch/recon", loss_dict["recon"], on_step=False, on_epoch=True)
        self.log("val_epoch/kl", loss_dict["kl"], on_step=False, on_epoch=True)
        self.log("val_epoch/kl_raw", loss_dict["kl_raw"], on_step=False, on_epoch=True)

        # Normalized metrics (resolution-independent)
        z_dim = mu.shape[1]
        # Note: recon is already mean-reduced (per-element MSE), no additional normalization needed
        self.log("val_epoch/kl_per_dim", loss_dict["kl_raw"] / z_dim, on_step=False, on_epoch=True)

        # Latent statistics
        self.log("val_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)

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
