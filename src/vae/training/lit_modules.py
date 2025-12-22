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
from typing import Any, Dict, Optional, Tuple

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from ..models import BaselineVAE, TCVAESBD
from ..losses import compute_elbo, compute_tcvae_loss, get_capacity_schedule
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
        loss_reduction: str = "mean",
        # New parameters for posterior collapse mitigation
        kl_annealing_type: str = "cyclical",
        kl_annealing_cycles: int = 4,
        kl_annealing_ratio: float = 0.5,
        kl_free_bits: float = 0.5,
        kl_free_bits_mode: str = "batch_mean",
        kl_target_capacity: Optional[float] = None,
        kl_capacity_anneal_epochs: int = 100,
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
            loss_reduction=cfg.train.get("loss_reduction", "mean"),
            # New parameters with backward-compatible defaults
            kl_annealing_type=cfg.train.get("kl_annealing_type", "cyclical"),
            kl_annealing_cycles=cfg.train.get("kl_annealing_cycles", 4),
            kl_annealing_ratio=cfg.train.get("kl_annealing_ratio", 0.5),
            kl_free_bits=cfg.train.get("kl_free_bits", 0.5),
            kl_free_bits_mode=cfg.train.get("kl_free_bits_mode", "batch_mean"),
            kl_target_capacity=cfg.train.get("kl_target_capacity", None),
            kl_capacity_anneal_epochs=cfg.train.get("kl_capacity_anneal_epochs", 100),
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

        # Forward pass
        x_hat, mu, logvar = self.model(x)

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

        # Log metrics (on_step=True for intra-epoch logging)
        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/recon", loss_dict["recon"], on_step=True, on_epoch=True)
        self.log("train/kl", loss_dict["kl"], on_step=True, on_epoch=True)
        self.log("train/kl_raw", loss_dict["kl_raw"], on_step=True, on_epoch=True)
        self.log("train/beta", self.current_beta, on_step=True, on_epoch=True)
        if self.current_capacity is not None:
            self.log("train/capacity", self.current_capacity, on_step=True, on_epoch=True)

        # Log latent space statistics
        self.log("train/mu_mean", mu.mean(), on_step=True, on_epoch=True)
        self.log("train/mu_std", mu.std(), on_step=True, on_epoch=True)
        self.log("train/logvar_mean", logvar.mean(), on_step=True, on_epoch=True)

        # Log learning rate
        opt = self.optimizers()
        if opt is not None:
            current_lr = opt.param_groups[0]["lr"]
            self.log("train/lr", current_lr, on_step=True, on_epoch=False)

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

        # Log metrics (on_step=True for intra-epoch logging)
        self.log("val/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/recon", loss_dict["recon"], on_step=True, on_epoch=True)
        self.log("val/kl", loss_dict["kl"], on_step=True, on_epoch=True)
        self.log("val/kl_raw", loss_dict["kl_raw"], on_step=True, on_epoch=True)
        self.log("val/beta", self.current_beta, on_step=True, on_epoch=True)
        if self.current_capacity is not None:
            self.log("val/capacity", self.current_capacity, on_step=True, on_epoch=True)

        # Log latent space statistics
        self.log("val/mu_mean", mu.mean(), on_step=True, on_epoch=True)
        self.log("val/mu_std", mu.std(), on_step=True, on_epoch=True)
        self.log("val/logvar_mean", logvar.mean(), on_step=True, on_epoch=True)

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
        kl_free_bits: Minimum KL threshold per latent dimension.
        kl_free_bits_mode: Free Bits clamping mode.
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
        loss_reduction: str = "mean",
        kl_free_bits: float = 0.0,
        kl_free_bits_mode: str = "batch_mean",
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
            loss_reduction: Loss reduction strategy ("mean" or "sum").
                           Default "mean" for numerical stability in FP16.
            kl_free_bits: Minimum KL threshold per latent dimension (nats).
                         Set to 0.0 to disable Free Bits (default).
            kl_free_bits_mode: Free Bits clamping mode ("batch_mean" or "per_sample").
                              Default "batch_mean".
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
        self.loss_reduction = loss_reduction
        self.kl_free_bits = kl_free_bits
        self.kl_free_bits_mode = kl_free_bits_mode
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
            loss_reduction=cfg.loss.get("reduction", cfg.train.get("loss_reduction", "mean")),
            kl_free_bits=cfg.train.get("kl_free_bits", 0.0),
            kl_free_bits_mode=cfg.train.get("kl_free_bits_mode", "batch_mean"),
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
            reduction=self.loss_reduction,
            kl_free_bits=self.kl_free_bits,
            kl_free_bits_mode=self.kl_free_bits_mode,
        )

        # Log metrics (on_step=True for intra-epoch logging)
        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/recon", loss_dict["recon"], on_step=True, on_epoch=True)
        self.log("train/mi", loss_dict["mi"], on_step=True, on_epoch=True)
        self.log("train/tc", loss_dict["tc"], on_step=True, on_epoch=True)
        self.log("train/dwkl", loss_dict["dwkl"], on_step=True, on_epoch=True)
        self.log("train/beta_tc", self.current_beta_tc, on_step=True, on_epoch=True)
        self.log("train/kl_raw", loss_dict["kl_raw"], on_step=True, on_epoch=True)
        self.log("train/kl_constrained", loss_dict["kl_constrained"], on_step=True, on_epoch=True)
        self.log("train/kl_free_bits_penalty", loss_dict["kl_free_bits_penalty"], on_step=True, on_epoch=True)

        # Log latent space statistics
        self.log("train/mu_mean", mu.mean(), on_step=True, on_epoch=True)
        self.log("train/mu_std", mu.std(), on_step=True, on_epoch=True)
        self.log("train/logvar_mean", logvar.mean(), on_step=True, on_epoch=True)
        self.log("train/z_mean", z.mean(), on_step=True, on_epoch=True)
        self.log("train/z_std", z.std(), on_step=True, on_epoch=True)

        # Log learning rate
        opt = self.optimizers()
        if opt is not None:
            current_lr = opt.param_groups[0]["lr"]
            self.log("train/lr", current_lr, on_step=True, on_epoch=False)

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
            reduction=self.loss_reduction,
            kl_free_bits=self.kl_free_bits,
            kl_free_bits_mode=self.kl_free_bits_mode,
        )

        # Log metrics (on_step=True for intra-epoch logging)
        self.log("val/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/recon", loss_dict["recon"], on_step=True, on_epoch=True)
        self.log("val/mi", loss_dict["mi"], on_step=True, on_epoch=True)
        self.log("val/tc", loss_dict["tc"], on_step=True, on_epoch=True)
        self.log("val/dwkl", loss_dict["dwkl"], on_step=True, on_epoch=True)
        self.log("val/beta_tc", self.current_beta_tc, on_step=True, on_epoch=True)
        self.log("val/kl_raw", loss_dict["kl_raw"], on_step=True, on_epoch=True)
        self.log("val/kl_constrained", loss_dict["kl_constrained"], on_step=True, on_epoch=True)
        self.log("val/kl_free_bits_penalty", loss_dict["kl_free_bits_penalty"], on_step=True, on_epoch=True)

        # Log latent space statistics
        self.log("val/mu_mean", mu.mean(), on_step=True, on_epoch=True)
        self.log("val/mu_std", mu.std(), on_step=True, on_epoch=True)
        self.log("val/logvar_mean", logvar.mean(), on_step=True, on_epoch=True)
        self.log("val/z_mean", z.mean(), on_step=True, on_epoch=True)
        self.log("val/z_std", z.std(), on_step=True, on_epoch=True)

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
