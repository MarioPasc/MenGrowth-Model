"""PyTorch Lightning modules for VAE training.

This module implements training logic for:
- Exp1: Baseline 3D VAE with ELBO loss (VAELitModule)
- Exp2a: β-TCVAE with SBD and TC loss decomposition (TCVAELitModule)
- Exp2b: DIP-VAE with SBD and covariance regularization (DIPVAELitModule)

All modules include:
- Training and validation steps with respective losses
- Annealing schedules for KL/TC/covariance weights
- Logging of all loss components
- AdamW optimizer configuration
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from ..models import BaselineVAE, TCVAESBD
from ..losses import (
    compute_elbo,
    compute_tcvae_loss,
    compute_dipvae_loss,
    get_capacity_schedule,
    get_lambda_cov_schedule,
)
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

        # === STEP LOGGING (minimal) ===
        self.log("train_step/loss", loss_dict["loss"], on_step=True, on_epoch=False, prog_bar=False)

        # === EPOCH LOGGING (comprehensive) ===
        self.log("train_epoch/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_epoch/recon", loss_dict["recon"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl", loss_dict["kl"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl_raw", loss_dict["kl_raw"], on_step=False, on_epoch=True)

        # Normalized metrics (resolution-independent)
        num_voxels = x.shape[2] * x.shape[3] * x.shape[4]  # D * H * W
        z_dim = mu.shape[1]
        self.log("train_epoch/recon_per_voxel", loss_dict["recon"] / num_voxels, on_step=False, on_epoch=True)
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

        # === STEP LOGGING (minimal) ===
        self.log("val_step/loss", loss_dict["loss"], on_step=True, on_epoch=False, prog_bar=False)

        # === EPOCH LOGGING (comprehensive) ===
        self.log("val_epoch/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_epoch/recon", loss_dict["recon"], on_step=False, on_epoch=True)
        self.log("val_epoch/kl", loss_dict["kl"], on_step=False, on_epoch=True)
        self.log("val_epoch/kl_raw", loss_dict["kl_raw"], on_step=False, on_epoch=True)

        # Normalized metrics (resolution-independent)
        num_voxels = x.shape[2] * x.shape[3] * x.shape[4]  # D * H * W
        z_dim = mu.shape[1]
        self.log("val_epoch/recon_per_voxel", loss_dict["recon"] / num_voxels, on_step=False, on_epoch=True)
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

        # === STEP LOGGING (minimal) ===
        self.log("train_step/loss", loss_dict["loss"], on_step=True, on_epoch=False, prog_bar=False)

        # === EPOCH LOGGING (comprehensive) ===
        self.log("train_epoch/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_epoch/recon", loss_dict["recon"], on_step=False, on_epoch=True)

        # TC-VAE decomposition (MI, TC, DWKL are the primary disentanglement diagnostics)
        # Cite: Chen et al. 2018, arXiv:1802.04942
        self.log("train_epoch/mi", loss_dict["mi"], on_step=False, on_epoch=True)
        self.log("train_epoch/tc", loss_dict["tc"], on_step=False, on_epoch=True)
        self.log("train_epoch/dwkl", loss_dict["dwkl"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl_raw", loss_dict["kl_raw"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl_constrained", loss_dict["kl_constrained"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl_free_bits_penalty", loss_dict["kl_free_bits_penalty"], on_step=False, on_epoch=True)

        # Normalized metrics (resolution-independent)
        num_voxels = x.shape[2] * x.shape[3] * x.shape[4]  # D * H * W
        z_dim = mu.shape[1]
        self.log("train_epoch/recon_per_voxel", loss_dict["recon"] / num_voxels, on_step=False, on_epoch=True)
        self.log("train_epoch/kl_per_dim", loss_dict["kl_raw"] / z_dim, on_step=False, on_epoch=True)

        # Latent statistics
        self.log("train_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/z_mean", z.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/z_std", z.std(), on_step=False, on_epoch=True)

        # === SCHEDULE LOGGING ===
        self.log("sched/beta_tc", self.current_beta_tc, on_step=False, on_epoch=True)

        # Schedule state (linear warm-up for beta_tc)
        if self.beta_tc_annealing_epochs > 0:
            phase = min(1.0, self.current_epoch / self.beta_tc_annealing_epochs)
            self.log("sched/beta_tc_phase", phase, on_step=False, on_epoch=True)

        # === COLLAPSE PROXIES ===
        # Expected KL floor from free bits
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

        # === STEP LOGGING (minimal) ===
        self.log("val_step/loss", loss_dict["loss"], on_step=True, on_epoch=False, prog_bar=False)

        # === EPOCH LOGGING (comprehensive) ===
        self.log("val_epoch/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_epoch/recon", loss_dict["recon"], on_step=False, on_epoch=True)

        # TC-VAE decomposition
        self.log("val_epoch/mi", loss_dict["mi"], on_step=False, on_epoch=True)
        self.log("val_epoch/tc", loss_dict["tc"], on_step=False, on_epoch=True)
        self.log("val_epoch/dwkl", loss_dict["dwkl"], on_step=False, on_epoch=True)
        self.log("val_epoch/kl_raw", loss_dict["kl_raw"], on_step=False, on_epoch=True)
        self.log("val_epoch/kl_constrained", loss_dict["kl_constrained"], on_step=False, on_epoch=True)

        # Normalized metrics (resolution-independent)
        num_voxels = x.shape[2] * x.shape[3] * x.shape[4]  # D * H * W
        z_dim = mu.shape[1]
        self.log("val_epoch/recon_per_voxel", loss_dict["recon"] / num_voxels, on_step=False, on_epoch=True)
        self.log("val_epoch/kl_per_dim", loss_dict["kl_raw"] / z_dim, on_step=False, on_epoch=True)

        # Latent statistics
        self.log("val_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/z_mean", z.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/z_std", z.std(), on_step=False, on_epoch=True)

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


class DIPVAELitModule(pl.LightningModule):
    """PyTorch Lightning module for DIP-VAE training with SBD.

    DIP-VAE-II uses covariance matching to encourage disentanglement
    without requiring density estimation (unlike TC-VAE's MWS).

    Loss = recon + KL + λ_od × ||Cov_offdiag||_F² + λ_d × ||diag(Cov) - 1||_2²

    Reference:
        Kumar et al. "Variational Inference of Disentangled Latent Concepts
        from Unlabeled Observations" (ICLR 2018), arXiv:1711.00848

    Attributes:
        model: The TCVAESBD model instance.
        lr: Learning rate for optimizer.
        lambda_od: Weight for off-diagonal covariance penalty.
        lambda_d: Weight for diagonal covariance penalty.
        lambda_cov_annealing_epochs: Number of epochs for lambda warmup.
        current_lambda_od: Current lambda_od value (updated each epoch).
        current_lambda_d: Current lambda_d value (updated each epoch).
    """

    def __init__(
        self,
        model: TCVAESBD,
        lr: float = 1e-4,
        lambda_od: float = 10.0,
        lambda_d: float = 5.0,
        lambda_cov_annealing_epochs: int = 40,
        compute_in_fp32: bool = True,
        loss_reduction: str = "mean",
        kl_free_bits: float = 0.0,
        kl_free_bits_mode: str = "batch_mean",
        use_ddp_gather: bool = True,
    ):
        """Initialize DIPVAELitModule.

        Args:
            model: TCVAESBD model instance.
            lr: Learning rate for AdamW optimizer.
            lambda_od: Weight for off-diagonal covariance penalty (default: 10.0).
            lambda_d: Weight for diagonal covariance penalty (default: 5.0).
            lambda_cov_annealing_epochs: Number of epochs for linear lambda warmup.
                Default: 40 (prevents optimizer shock).
            compute_in_fp32: If True, compute covariance in FP32 for stability.
            loss_reduction: Loss reduction strategy ("mean" or "sum").
                Default "mean" for numerical stability with large volumes.
            kl_free_bits: Minimum KL threshold per latent dimension (nats).
            kl_free_bits_mode: Free Bits clamping mode ("batch_mean" or "per_sample").
            use_ddp_gather: If True, use all-gather when DDP active (default: True).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.lambda_od = lambda_od
        self.lambda_d = lambda_d
        self.lambda_cov_annealing_epochs = lambda_cov_annealing_epochs
        self.compute_in_fp32 = compute_in_fp32
        self.loss_reduction = loss_reduction
        self.kl_free_bits = kl_free_bits
        self.kl_free_bits_mode = kl_free_bits_mode
        self.use_ddp_gather = use_ddp_gather

        # Initialize current lambda values (updated via schedule)
        self.current_lambda_od = 0.0
        self.current_lambda_d = 0.0

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "DIPVAELitModule":
        """Create DIPVAELitModule from configuration.

        Args:
            cfg: OmegaConf configuration object.

        Returns:
            Configured DIPVAELitModule instance.
        """
        # Determine num_groups from norm config
        num_groups = 8
        if cfg.model.norm == "GROUP":
            num_groups = 8

        # Get SBD grid size
        sbd_grid_size = tuple(cfg.model.sbd_grid_size)

        # Get SBD upsample mode
        sbd_upsample_mode = cfg.model.get("sbd_upsample_mode", "resize_conv")

        # Get posterior variance floor
        posterior_logvar_min = cfg.train.get("posterior_logvar_min", -6.0)

        # Get gradient checkpointing flag
        gradient_checkpointing = cfg.train.get("gradient_checkpointing", False)

        model = TCVAESBD(
            input_channels=cfg.model.input_channels,
            z_dim=cfg.model.z_dim,
            base_filters=cfg.model.base_filters,
            num_groups=num_groups,
            sbd_grid_size=sbd_grid_size,
            sbd_upsample_mode=sbd_upsample_mode,
            posterior_logvar_min=posterior_logvar_min,
            gradient_checkpointing=gradient_checkpointing,
        )

        return cls(
            model=model,
            lr=cfg.train.lr,
            lambda_od=cfg.loss.lambda_od,
            lambda_d=cfg.loss.lambda_d,
            lambda_cov_annealing_epochs=cfg.loss.get("lambda_cov_annealing_epochs", 0),
            compute_in_fp32=cfg.loss.get("compute_in_fp32", True),
            loss_reduction=cfg.loss.get("reduction", cfg.train.get("loss_reduction", "mean")),
            kl_free_bits=cfg.train.get("kl_free_bits", 0.0),
            kl_free_bits_mode=cfg.train.get("kl_free_bits_mode", "batch_mean"),
            use_ddp_gather=cfg.loss.get("use_ddp_gather", True),
        )

    def on_train_epoch_start(self) -> None:
        """Update lambda_od and lambda_d at the start of each training epoch."""
        self.current_lambda_od = get_lambda_cov_schedule(
            epoch=self.current_epoch,
            lambda_target=self.lambda_od,
            lambda_annealing_epochs=self.lambda_cov_annealing_epochs,
        )
        self.current_lambda_d = get_lambda_cov_schedule(
            epoch=self.current_epoch,
            lambda_target=self.lambda_d,
            lambda_annealing_epochs=self.lambda_cov_annealing_epochs,
        )
        logger.debug(
            f"Epoch {self.current_epoch}: lambda_od = {self.current_lambda_od:.4f}, "
            f"lambda_d = {self.current_lambda_d:.4f}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through DIP-VAE+SBD.

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

        # Compute DIP-VAE loss
        loss_dict = compute_dipvae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            lambda_od=self.current_lambda_od,
            lambda_d=self.current_lambda_d,
            compute_in_fp32=self.compute_in_fp32,
            reduction=self.loss_reduction,
            kl_free_bits=self.kl_free_bits,
            kl_free_bits_mode=self.kl_free_bits_mode,
            use_ddp_gather=self.use_ddp_gather,
        )

        # === STEP LOGGING (minimal) ===
        self.log("train_step/loss", loss_dict["loss"], on_step=True, on_epoch=False, prog_bar=False)

        # === EPOCH LOGGING (comprehensive) ===
        self.log("train_epoch/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_epoch/recon", loss_dict["recon"], on_step=False, on_epoch=True)

        # DIP-VAE covariance penalties (weighted)
        self.log("train_epoch/cov_penalty",
                 loss_dict["cov_penalty_od"] + loss_dict["cov_penalty_d"],
                 on_step=False, on_epoch=True)
        self.log("train_epoch/cov_penalty_od", loss_dict["cov_penalty_od"], on_step=False, on_epoch=True)
        self.log("train_epoch/cov_penalty_d", loss_dict["cov_penalty_d"], on_step=False, on_epoch=True)

        # Covariance diagnostics (unweighted)
        self.log("train_epoch/cov_offdiag_fro", loss_dict["cov_offdiag_fro"], on_step=False, on_epoch=True)
        self.log("train_epoch/cov_diag_l2", loss_dict["cov_diag_l2"], on_step=False, on_epoch=True)

        # KL divergence
        self.log("train_epoch/kl_raw", loss_dict["kl_raw"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl_constrained", loss_dict["kl_constrained"], on_step=False, on_epoch=True)

        # Normalized metrics (resolution-independent)
        num_voxels = x.shape[2] * x.shape[3] * x.shape[4]  # D * H * W
        z_dim = mu.shape[1]
        self.log("train_epoch/recon_per_voxel", loss_dict["recon"] / num_voxels, on_step=False, on_epoch=True)
        self.log("train_epoch/recon_sum", loss_dict["recon_sum"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl_per_dim", loss_dict["kl_raw"] / z_dim, on_step=False, on_epoch=True)

        # Latent statistics
        self.log("train_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/logvar_min", logvar.min(), on_step=False, on_epoch=True)
        self.log("train_epoch/z_mean", z.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/z_std", z.std(), on_step=False, on_epoch=True)

        # === SCHEDULE LOGGING ===
        self.log("sched/lambda_od", self.current_lambda_od, on_step=False, on_epoch=True)
        self.log("sched/lambda_d", self.current_lambda_d, on_step=False, on_epoch=True)

        # Schedule state (linear warm-up for lambda)
        if self.lambda_cov_annealing_epochs > 0:
            phase = min(1.0, self.current_epoch / self.lambda_cov_annealing_epochs)
            self.log("sched/lambda_phase", phase, on_step=False, on_epoch=True)

        # === COLLAPSE PROXIES ===
        # Expected KL floor from free bits
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

        # Forward pass
        x_hat, mu, logvar, z = self.model(x)

        # Compute DIP-VAE loss
        loss_dict = compute_dipvae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            lambda_od=self.current_lambda_od,
            lambda_d=self.current_lambda_d,
            compute_in_fp32=self.compute_in_fp32,
            reduction=self.loss_reduction,
            kl_free_bits=self.kl_free_bits,
            kl_free_bits_mode=self.kl_free_bits_mode,
            use_ddp_gather=self.use_ddp_gather,
        )

        # === STEP LOGGING (minimal) ===
        self.log("val_step/loss", loss_dict["loss"], on_step=True, on_epoch=False, prog_bar=False)

        # === EPOCH LOGGING (comprehensive) ===
        self.log("val_epoch/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_epoch/recon", loss_dict["recon"], on_step=False, on_epoch=True)

        # DIP-VAE covariance penalties
        self.log("val_epoch/cov_penalty",
                 loss_dict["cov_penalty_od"] + loss_dict["cov_penalty_d"],
                 on_step=False, on_epoch=True)
        self.log("val_epoch/cov_penalty_od", loss_dict["cov_penalty_od"], on_step=False, on_epoch=True)
        self.log("val_epoch/cov_penalty_d", loss_dict["cov_penalty_d"], on_step=False, on_epoch=True)

        # Covariance diagnostics
        self.log("val_epoch/cov_offdiag_fro", loss_dict["cov_offdiag_fro"], on_step=False, on_epoch=True)
        self.log("val_epoch/cov_diag_l2", loss_dict["cov_diag_l2"], on_step=False, on_epoch=True)

        # KL divergence
        self.log("val_epoch/kl_raw", loss_dict["kl_raw"], on_step=False, on_epoch=True)
        self.log("val_epoch/kl_constrained", loss_dict["kl_constrained"], on_step=False, on_epoch=True)

        # Normalized metrics (resolution-independent)
        num_voxels = x.shape[2] * x.shape[3] * x.shape[4]  # D * H * W
        z_dim = mu.shape[1]
        self.log("val_epoch/recon_per_voxel", loss_dict["recon"] / num_voxels, on_step=False, on_epoch=True)
        self.log("val_epoch/kl_per_dim", loss_dict["kl_raw"] / z_dim, on_step=False, on_epoch=True)

        # Latent statistics
        self.log("val_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/logvar_min", logvar.min(), on_step=False, on_epoch=True)
        self.log("val_epoch/z_mean", z.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/z_std", z.std(), on_step=False, on_epoch=True)

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
