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

from ...losses import (
    compute_dipvae_loss,
    get_lambda_cov_schedule,
)
from ...losses.elbo import get_beta_schedule
from vae.metrics import compute_ssim_2d_slices, compute_psnr_3d


logger = logging.getLogger(__name__)


class DIPVAELitModule(pl.LightningModule):
    """PyTorch Lightning module for DIP-VAE training.

    DIP-VAE-II uses covariance matching to encourage disentanglement
    without requiring density estimation.

    Loss = recon + KL + λ_od × ||Cov_offdiag||_F² + λ_d × ||diag(Cov) - 1||_2²

    The decoder can be either:
    - Standard transposed-conv decoder (BaselineVAE, use_sbd=false)
    - Spatial Broadcast Decoder (VAESBD, use_sbd=true)

    Reference:
        Kumar et al. "Variational Inference of Disentangled Latent Concepts
        from Unlabeled Observations" (ICLR 2018), arXiv:1711.00848

    Attributes:
        model: VAE model instance (BaselineVAE or VAESBD).
        lr: Learning rate for optimizer.
        lambda_od: Weight for off-diagonal covariance penalty.
        lambda_d: Weight for diagonal covariance penalty.
        lambda_cov_annealing_epochs: Number of epochs for lambda warmup.
        current_lambda_od: Current lambda_od value (updated each epoch).
        current_lambda_d: Current lambda_d value (updated each epoch).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        lambda_od: float = 10.0,
        lambda_d: float = 5.0,
        lambda_cov_annealing_epochs: int = 40,
        lambda_start_epoch: int = 0,
        compute_in_fp32: bool = True,
        loss_reduction: str = "mean",
        kl_beta: float = 1.0,
        kl_annealing_epochs: int = 40,
        kl_annealing_type: str = "cyclical",
        kl_annealing_cycles: int = 4,
        kl_annealing_ratio: float = 0.5,
        kl_free_bits: float = 0.0,
        kl_free_bits_mode: str = "batch_mean",
        use_ddp_gather: bool = True,
        log_collapse_diagnostics: bool = True,
        modality_names: Optional[list] = None,
        posterior_logvar_min: float = -6.0,
        weight_decay: float = 0.01,
    ):
        """Initialize DIPVAELitModule.

        Args:
            model: VAE model instance (BaselineVAE or VAESBD).
            lr: Learning rate for AdamW optimizer.
            lambda_od: Weight for off-diagonal covariance penalty (default: 10.0).
            lambda_d: Weight for diagonal covariance penalty (default: 5.0).
            lambda_cov_annealing_epochs: Number of epochs for linear lambda warmup.
                Default: 40 (prevents optimizer shock).
            lambda_start_epoch: Epoch to start applying covariance penalties.
                Default: 0. Set higher to pre-train VAE before DIP regularization.
            compute_in_fp32: If True, compute covariance in FP32 for stability.
            loss_reduction: Loss reduction strategy ("mean" or "sum").
                Default "mean" for numerical stability with large volumes.
            kl_beta: Target beta value after annealing.
            kl_annealing_epochs: Number of epochs for KL annealing.
            kl_annealing_type: Annealing type ("linear" or "cyclical").
            kl_annealing_cycles: Number of cycles for cyclical annealing.
            kl_annealing_ratio: Fraction of each cycle for annealing.
            kl_free_bits: Minimum KL threshold per latent dimension (nats).
            kl_free_bits_mode: Free Bits clamping mode ("batch_mean" or "per_sample").
            use_ddp_gather: If True, use all-gather when DDP active (default: True).
            log_collapse_diagnostics: If True, log additional metrics to diagnose
                posterior collapse (deterministic recon, z=0 ablation, μ variance).
            modality_names: List of modality names for logging.
            posterior_logvar_min: Minimum value for logvar. Stored for hyperparameter
                tracking; actual clamping happens in model.encode(). Default: -6.0.
            weight_decay: AdamW weight decay (L2 regularization). Default: 0.01.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.lambda_od = lambda_od
        self.lambda_d = lambda_d
        self.lambda_cov_annealing_epochs = lambda_cov_annealing_epochs
        self.lambda_start_epoch = lambda_start_epoch
        self.compute_in_fp32 = compute_in_fp32
        self.loss_reduction = loss_reduction
        self.kl_beta = kl_beta
        self.kl_annealing_epochs = kl_annealing_epochs
        self.kl_annealing_type = kl_annealing_type
        self.kl_annealing_cycles = kl_annealing_cycles
        self.kl_annealing_ratio = kl_annealing_ratio
        self.kl_free_bits = kl_free_bits
        self.kl_free_bits_mode = kl_free_bits_mode
        self.use_ddp_gather = use_ddp_gather
        self.log_collapse_diagnostics = log_collapse_diagnostics
        self.modality_names = modality_names or ["t1c", "t1n", "t2f", "t2w"]
        self.weight_decay = weight_decay
        # NOTE: posterior_logvar_min stored in hparams for tracking, but actual
        # clamping happens in model.encode(). Use self.model.posterior_logvar_min
        # for runtime monitoring (e.g., logvar saturation tracking).

        # Initialize current lambda values (updated via schedule)
        self.current_lambda_od = 0.0
        self.current_lambda_d = 0.0
        self.current_beta = 0.0

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "DIPVAELitModule":
        """Create DIPVAELitModule from configuration.

        Uses model_factory to build either BaselineVAE or VAESBD based on cfg.model.use_sbd.

        Args:
            cfg: OmegaConf configuration object.

        Returns:
            Configured DIPVAELitModule instance.
        """
        from vae.training.engine.model_factory import create_vae_model

        # Use factory to create model (handles use_sbd logic)
        model = create_vae_model(cfg)

        return cls(
            model=model,
            lr=cfg.train.lr,
            lambda_od=cfg.loss.lambda_od,
            lambda_d=cfg.loss.lambda_d,
            lambda_cov_annealing_epochs=cfg.loss.get("lambda_cov_annealing_epochs", 0),
            lambda_start_epoch=cfg.loss.get("lambda_start_epoch", 0),
            compute_in_fp32=cfg.loss.get("compute_in_fp32", True),
            loss_reduction=cfg.loss.get("reduction", cfg.train.get("loss_reduction", "mean")),
            kl_beta=cfg.train.get("kl_beta", 1.0),
            kl_annealing_epochs=cfg.train.get("kl_annealing_epochs", 40),
            kl_annealing_type=cfg.train.get("kl_annealing_type", "cyclical"),
            kl_annealing_cycles=cfg.train.get("kl_annealing_cycles", 4),
            kl_annealing_ratio=cfg.train.get("kl_annealing_ratio", 0.5),
            kl_free_bits=cfg.train.get("kl_free_bits", 0.0),
            kl_free_bits_mode=cfg.train.get("kl_free_bits_mode", "batch_mean"),
            use_ddp_gather=cfg.loss.get("use_ddp_gather", True),
            log_collapse_diagnostics=cfg.train.get("log_collapse_diagnostics", True),
            modality_names=cfg.data.get("modalities", ["t1c", "t1n", "t2f", "t2w"]),
            posterior_logvar_min=cfg.train.get("posterior_logvar_min", -6.0),
            weight_decay=cfg.train.get("weight_decay", 0.01),
        )

    def on_train_epoch_start(self) -> None:
        """Update lambda_od, lambda_d, and beta at the start of each training epoch.

        Respects lambda_start_epoch for delayed DIP regularization (pre-train VAE first).
        """
        # Update beta schedule
        self.current_beta = get_beta_schedule(
            epoch=self.current_epoch,
            kl_beta=self.kl_beta,
            kl_annealing_epochs=self.kl_annealing_epochs,
            kl_annealing_type=self.kl_annealing_type,
            kl_annealing_cycles=self.kl_annealing_cycles,
            kl_annealing_ratio=self.kl_annealing_ratio,
        )

        # Apply delayed start: effective epoch is shifted by lambda_start_epoch
        effective_epoch = max(0, self.current_epoch - self.lambda_start_epoch)
        if self.current_epoch < self.lambda_start_epoch:
            # Pre-training phase: no covariance penalty
            self.current_lambda_od = 0.0
            self.current_lambda_d = 0.0
        else:
            self.current_lambda_od = get_lambda_cov_schedule(
                epoch=effective_epoch,
                lambda_target=self.lambda_od,
                lambda_annealing_epochs=self.lambda_cov_annealing_epochs,
            )
            self.current_lambda_d = get_lambda_cov_schedule(
                epoch=effective_epoch,
                lambda_target=self.lambda_d,
                lambda_annealing_epochs=self.lambda_cov_annealing_epochs,
            )
        logger.debug(
            f"Epoch {self.current_epoch}: beta = {self.current_beta:.4f}, "
            f"lambda_od = {self.current_lambda_od:.4f}, "
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

        # NOTE: logvar is already clamped at the source in model.encode()

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
            beta=self.current_beta,
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
        z_dim = mu.shape[1]
        # Note: recon is already mean-reduced (per-element MSE), no additional normalization needed
        self.log("train_epoch/recon_sum", loss_dict["recon_sum"], on_step=False, on_epoch=True)
        self.log("train_epoch/kl_per_dim", loss_dict["kl_raw"] / z_dim, on_step=False, on_epoch=True)

        # Per-modality reconstruction error
        for i, mod_name in enumerate(self.modality_names):
            if i >= x.shape[1]: break
            mod_recon = torch.nn.functional.mse_loss(
                x_hat[:, i:i+1],  # Single channel
                x[:, i:i+1],
                reduction="mean"
            )
            self.log(
                f"train_epoch/recon_{mod_name}",
                mod_recon,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        # Latent statistics
        self.log("train_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/logvar_min", logvar.min(), on_step=False, on_epoch=True)
        self.log("train_epoch/z_mean", z.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/z_std", z.std(), on_step=False, on_epoch=True)

        # === SCHEDULE LOGGING ===
        self.log("sched/beta", self.current_beta, on_step=False, on_epoch=True)
        self.log("sched/lambda_od", self.current_lambda_od, on_step=False, on_epoch=True)
        self.log("sched/lambda_d", self.current_lambda_d, on_step=False, on_epoch=True)

        # Schedule state (linear warm-up for lambda, accounting for delayed start)
        if self.lambda_cov_annealing_epochs > 0:
            effective_epoch = max(0, self.current_epoch - self.lambda_start_epoch)
            phase = min(1.0, effective_epoch / self.lambda_cov_annealing_epochs)
            self.log("sched/lambda_phase", phase, on_step=False, on_epoch=True)

        # Schedule state (for cyclical annealing)
        if self.kl_annealing_type == "cyclical":
            cycle_length = self.kl_annealing_epochs / self.kl_annealing_cycles
            cycle_idx = int(self.current_epoch / cycle_length)
            phase = (self.current_epoch % cycle_length) / cycle_length
            self.log("sched/cycle_idx", float(cycle_idx), on_step=False, on_epoch=True)
            self.log("sched/phase", phase, on_step=False, on_epoch=True)

        # === COLLAPSE PROXIES ===
        # Expected KL floor from free bits
        if self.kl_free_bits > 0.0:
            expected_kl_floor = z_dim * self.kl_free_bits
            self.log("sched/expected_kl_floor", expected_kl_floor, on_step=False, on_epoch=True)
            self.log("sched/free_bits", self.kl_free_bits, on_step=False, on_epoch=True)

        # === PER-DIMENSION KL STATISTICS (collapse detection) ===
        # Compute per-dimension KL: 0.5 * (exp(logvar) + mu^2 - 1 - logvar) [B, z_dim]
        kl_per_dim = 0.5 * (torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
        kl_per_dim = torch.clamp(kl_per_dim, min=0.0)  # Numerical stability

        # Per-sample statistics (mean across batch of min/max across dims)
        self.log("train_epoch/kl_per_dim_min", kl_per_dim.min(dim=1).values.mean(), on_step=False, on_epoch=True)
        self.log("train_epoch/kl_per_dim_max", kl_per_dim.max(dim=1).values.mean(), on_step=False, on_epoch=True)

        # Collapsed fraction: dims with batch-mean KL < 0.01 nats
        kl_batch_mean_per_dim = kl_per_dim.mean(dim=0)  # [z_dim]
        collapsed_frac = (kl_batch_mean_per_dim < 0.01).float().mean()
        self.log("train_epoch/kl_collapsed_frac", collapsed_frac, on_step=False, on_epoch=True)

        # === LOGVAR SATURATION TRACKING ===
        # Fraction of dimensions hitting the posterior_logvar_min floor
        logvar_min = self.model.posterior_logvar_min  # -6.0
        logvar_at_min_frac = (logvar <= logvar_min + 1e-3).float().mean()
        self.log("train_epoch/logvar_at_min_frac", logvar_at_min_frac, on_step=False, on_epoch=True)

        # === KL-TO-RECONSTRUCTION RATIO ===
        # If << 1, KL is being ignored by the optimizer
        kl_to_recon_ratio = loss_dict["kl_raw"] / (loss_dict["recon"] + 1e-8)
        self.log("train_epoch/loss_ratio_kl_to_recon", kl_to_recon_ratio, on_step=False, on_epoch=True)

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

        # NOTE: logvar is already clamped at the source in model.encode()

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
            beta=self.current_beta,
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
        z_dim = mu.shape[1]
        # Note: recon is already mean-reduced (per-element MSE), no additional normalization needed
        self.log("val_epoch/kl_per_dim", loss_dict["kl_raw"] / z_dim, on_step=False, on_epoch=True)

        # Per-modality reconstruction error
        for i, mod_name in enumerate(self.modality_names):
            if i >= x.shape[1]: break
            mod_recon = torch.nn.functional.mse_loss(
                x_hat[:, i:i+1],  # Single channel
                x[:, i:i+1],
                reduction="mean"
            )
            self.log(
                f"val_epoch/recon_{mod_name}",
                mod_recon,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        # SSIM and PSNR (expensive, compute once per 10 batches on first sample)
        if batch_idx % 10 == 0:
            # Compute on first sample only to save time
            x_sample = x[0:1]  # [1, 4, 128, 128, 128]
            x_hat_sample = x_hat[0:1]

            # Per-modality SSIM/PSNR
            for i, mod_name in enumerate(self.modality_names):
                if i >= x_sample.shape[1]: break
                ssim_val = compute_ssim_2d_slices(
                    x_hat_sample[:, i:i+1],
                    x_sample[:, i:i+1],
                )
                psnr_val = compute_psnr_3d(
                    x_hat_sample[:, i:i+1],
                    x_sample[:, i:i+1],
                )

                self.log(f"val_epoch/ssim_{mod_name}", ssim_val, on_step=False, on_epoch=True)
                self.log(f"val_epoch/psnr_{mod_name}", psnr_val, on_step=False, on_epoch=True)

        # Latent statistics
        self.log("val_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/logvar_min", logvar.min(), on_step=False, on_epoch=True)
        self.log("val_epoch/z_mean", z.mean(), on_step=False, on_epoch=True)
        self.log("val_epoch/z_std", z.std(), on_step=False, on_epoch=True)

        # === COLLAPSE DIAGNOSTICS (expensive, run on first batch only) ===
        if self.log_collapse_diagnostics and batch_idx == 0:
            with torch.no_grad():
                # 1. Deterministic recon: decode z = μ (no sampling noise)
                x_hat_mu = self.model.decode(mu)
                recon_mu_mse = torch.nn.functional.mse_loss(x_hat_mu, x, reduction="mean")
                self.log("diag/recon_mu_mse", recon_mu_mse, on_step=False, on_epoch=True)

                # 2. z=0 ablation: test if decoder ignores z entirely
                z_zero = torch.zeros_like(mu)
                x_hat_z0 = self.model.decode(z_zero)
                recon_z0_mse = torch.nn.functional.mse_loss(x_hat_z0, x, reduction="mean")
                self.log("diag/recon_z0_mse", recon_z0_mse, on_step=False, on_epoch=True)

                # 3. Decoder z-dependence: ||x̂(μ) - x̂(z_sampled)||
                recon_delta = torch.nn.functional.mse_loss(x_hat_mu, x_hat, reduction="mean")
                self.log("diag/recon_delta_mu_vs_sampled", recon_delta, on_step=False, on_epoch=True)

                # 4. μ variance per-dim (dataset-level AU proxy for this batch)
                mu_var_per_dim = mu.var(dim=0, unbiased=False)  # [z_dim]
                mu_var_mean = mu_var_per_dim.mean()
                mu_var_max = mu_var_per_dim.max()
                self.log("diag/mu_var_mean", mu_var_mean, on_step=False, on_epoch=True)
                self.log("diag/mu_var_max", mu_var_max, on_step=False, on_epoch=True)

                # 5. Count batch-level "active" dims (var > 0.01)
                batch_au_count = (mu_var_per_dim > 0.01).sum().float()
                self.log("diag/batch_au_count", batch_au_count, on_step=False, on_epoch=True)

        # Per-dimension latent statistics (log to wandb as histograms)
        mu_per_dim = mu.mean(dim=0)  # [z_dim]
        logvar_per_dim = logvar.mean(dim=0)  # [z_dim]

        if self.logger and hasattr(self.logger, 'experiment'):
            try:
                import wandb
                self.logger.experiment.log({
                    "val/mu_histogram": wandb.Histogram(mu_per_dim.detach().cpu().numpy()),
                    "val/logvar_histogram": wandb.Histogram(logvar_per_dim.detach().cpu().numpy()),
                })
            except Exception as e:
                # Skip if not using wandb or histogram logging fails
                logger.debug(f"WandB histogram logging skipped: {e}")

        return loss_dict["loss"]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer.

        Returns:
            AdamW optimizer instance.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
