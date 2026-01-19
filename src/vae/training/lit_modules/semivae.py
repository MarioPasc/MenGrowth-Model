"""PyTorch Lightning module for Semi-Supervised VAE training.

This module implements training logic for Exp3: Semi-supervised VAE with
partitioned latent space for supervised disentanglement.

Key features:
- Semantic regression losses on supervised latent dimensions
- Total Correlation (TC) penalty on residual dimensions
- Delayed semantic supervision with annealing
- Per-partition loss tracking

Architecture:
    z = [z_vol | z_loc | z_shape | z_residual]
         ↓       ↓        ↓          ↓
      L_vol   L_loc   L_shape    KL + TC

References:
- Kingma et al., "Semi-Supervised Learning with Deep Generative Models", NeurIPS 2014
- Chen et al., "Isolating Sources of Disentanglement in VAEs", NeurIPS 2018
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig

from ...losses.elbo import get_beta_schedule
from ...losses.tc import compute_tc_loss_on_subset, compute_tc_ddp_aware
from ...losses.cross_partition import compute_cross_partition_loss
from vae.metrics import compute_ssim_2d_slices, compute_psnr_3d


logger = logging.getLogger(__name__)


def get_semantic_schedule(
    epoch: int,
    target_lambda: float,
    start_epoch: int,
    annealing_epochs: int,
) -> float:
    """Compute current semantic loss weight with delayed start and annealing.

    Args:
        epoch: Current epoch
        target_lambda: Target lambda value
        start_epoch: Epoch to start semantic supervision
        annealing_epochs: Number of epochs for linear warmup

    Returns:
        Current lambda value
    """
    if epoch < start_epoch:
        return 0.0

    effective_epoch = epoch - start_epoch
    if effective_epoch >= annealing_epochs:
        return target_lambda

    return target_lambda * (effective_epoch / annealing_epochs)


class SemiVAELitModule(pl.LightningModule):
    """PyTorch Lightning module for Semi-Supervised VAE training.

    Implements training with:
    - Reconstruction loss (MSE)
    - KL divergence on residual dimensions only
    - TC penalty on residual dimensions (optional)
    - Regression losses on semantic dimensions

    Attributes:
        model: SemiVAE model instance
        semantic_targets: Dictionary mapping partition names to target tensors
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        # Semantic supervision weights
        lambda_vol: float = 10.0,
        lambda_loc: float = 5.0,
        lambda_shape: float = 5.0,
        semantic_start_epoch: int = 10,
        semantic_annealing_epochs: int = 20,
        # TC regularization
        use_tc_residual: bool = True,
        lambda_tc: float = 2.0,
        tc_estimator: str = "minibatch_weighted",
        # Cross-partition independence
        lambda_cross_partition: float = 5.0,
        cross_partition_start_epoch: int = 10,
        # Manifold density regularization
        lambda_manifold: float = 1.0,
        manifold_start_epoch: int = 10,
        # KL regularization (on residual only)
        kl_beta: float = 1.0,
        kl_annealing_epochs: int = 200,
        kl_annealing_type: str = "cyclical",
        kl_annealing_cycles: int = 4,
        kl_annealing_ratio: float = 0.5,
        kl_free_bits: float = 0.2,
        kl_free_bits_mode: str = "batch_mean",
        # General settings
        loss_reduction: str = "mean",
        use_ddp_gather: bool = True,
        log_collapse_diagnostics: bool = True,
        modality_names: Optional[List[str]] = None,
        posterior_logvar_min: float = -6.0,
        weight_decay: float = 0.01,
        dataset_size: int = 1000,
    ):
        """Initialize SemiVAELitModule.

        Args:
            model: SemiVAE model instance
            lr: Learning rate for AdamW optimizer
            lambda_vol: Weight for volume regression loss
            lambda_loc: Weight for location regression loss
            lambda_shape: Weight for shape regression loss
            semantic_start_epoch: Epoch to start semantic supervision
            semantic_annealing_epochs: Epochs for semantic loss warmup
            use_tc_residual: Whether to apply TC penalty on residual
            lambda_tc: Weight for TC penalty
            tc_estimator: TC estimator type ("minibatch_weighted" or "stratified")
            kl_beta: Target beta value after annealing
            kl_annealing_epochs: Total epochs for KL annealing
            kl_annealing_type: "linear" or "cyclical"
            kl_annealing_cycles: Number of cycles for cyclical annealing
            kl_annealing_ratio: Fraction of cycle for annealing phase
            kl_free_bits: Per-dim KL floor (nats)
            kl_free_bits_mode: "per_sample" or "batch_mean"
            loss_reduction: "mean" or "sum"
            use_ddp_gather: Use all-gather for TC in DDP
            log_collapse_diagnostics: Log collapse detection metrics
            modality_names: Names for per-modality logging
            posterior_logvar_min: Stored for hparam tracking
            weight_decay: AdamW weight decay
            dataset_size: Total dataset size for TC estimation
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr

        # Semantic weights
        self.lambda_vol = lambda_vol
        self.lambda_loc = lambda_loc
        self.lambda_shape = lambda_shape
        self.semantic_start_epoch = semantic_start_epoch
        self.semantic_annealing_epochs = semantic_annealing_epochs

        # TC settings
        self.use_tc_residual = use_tc_residual
        self.lambda_tc = lambda_tc
        self.tc_estimator = tc_estimator

        # Cross-partition independence settings
        self.lambda_cross_partition = lambda_cross_partition
        self.cross_partition_start_epoch = cross_partition_start_epoch

        # Manifold density regularization settings
        self.lambda_manifold = lambda_manifold
        self.manifold_start_epoch = manifold_start_epoch

        # KL settings
        self.kl_beta = kl_beta
        self.kl_annealing_epochs = kl_annealing_epochs
        self.kl_annealing_type = kl_annealing_type
        self.kl_annealing_cycles = kl_annealing_cycles
        self.kl_annealing_ratio = kl_annealing_ratio
        self.kl_free_bits = kl_free_bits
        self.kl_free_bits_mode = kl_free_bits_mode

        # General settings
        self.loss_reduction = loss_reduction
        self.use_ddp_gather = use_ddp_gather
        self.log_collapse_diagnostics = log_collapse_diagnostics
        self.modality_names = modality_names or ["t1c", "t1n", "t2f", "t2w"]
        self.weight_decay = weight_decay
        self.dataset_size = dataset_size

        # Current schedule values (updated each epoch)
        self.current_beta = 0.0
        self.current_lambda_vol = 0.0
        self.current_lambda_loc = 0.0
        self.current_lambda_shape = 0.0
        self.current_lambda_cross_partition = 0.0
        self.current_lambda_manifold = 0.0

        # Get residual indices from model
        self.residual_start, self.residual_end = model.get_residual_indices()
        self.residual_dim = self.residual_end - self.residual_start

        # Get partition indices for cross-partition loss (supervised partitions only)
        self.partition_indices = {}
        for name, config in model.get_partition_info().items():
            if config["supervision"] == "regression":
                self.partition_indices[name] = (config["start_idx"], config["end_idx"])

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "SemiVAELitModule":
        """Create SemiVAELitModule from configuration.

        Args:
            cfg: OmegaConf configuration object

        Returns:
            Configured SemiVAELitModule instance
        """
        from ...training.engine.model_factory import create_semivae_model

        model = create_semivae_model(cfg)

        return cls(
            model=model,
            lr=cfg.train.lr,
            lambda_vol=cfg.loss.get("lambda_vol", 10.0),
            lambda_loc=cfg.loss.get("lambda_loc", 5.0),
            lambda_shape=cfg.loss.get("lambda_shape", 5.0),
            semantic_start_epoch=cfg.loss.get("semantic_start_epoch", 10),
            semantic_annealing_epochs=cfg.loss.get("semantic_annealing_epochs", 20),
            use_tc_residual=cfg.loss.get("use_tc_residual", True),
            lambda_tc=cfg.loss.get("lambda_tc", 2.0),
            tc_estimator=cfg.loss.get("tc_estimator", "minibatch_weighted"),
            lambda_cross_partition=cfg.loss.get("lambda_cross_partition", 5.0),
            cross_partition_start_epoch=cfg.loss.get("cross_partition_start_epoch", 10),
            lambda_manifold=cfg.loss.get("lambda_manifold", 1.0),
            manifold_start_epoch=cfg.loss.get("manifold_start_epoch", 10),
            kl_beta=cfg.train.kl_beta,
            kl_annealing_epochs=cfg.train.kl_annealing_epochs,
            kl_annealing_type=cfg.train.kl_annealing_type,
            kl_annealing_cycles=cfg.train.get("kl_annealing_cycles", 4),
            kl_annealing_ratio=cfg.train.get("kl_annealing_ratio", 0.5),
            kl_free_bits=cfg.train.get("kl_free_bits", 0.2),
            kl_free_bits_mode=cfg.train.get("kl_free_bits_mode", "batch_mean"),
            loss_reduction=cfg.train.get("loss_reduction", "mean"),
            use_ddp_gather=cfg.loss.get("use_ddp_gather", True),
            log_collapse_diagnostics=cfg.train.get("log_collapse_diagnostics", True),
            modality_names=cfg.data.modalities,
            posterior_logvar_min=cfg.train.get("posterior_logvar_min", -6.0),
            weight_decay=cfg.train.get("weight_decay", 0.01),
            dataset_size=1000,  # Will be updated in setup
        )

    def on_train_epoch_start(self) -> None:
        """Update schedule values at epoch start."""
        epoch = self.current_epoch

        # Beta schedule for KL
        self.current_beta = get_beta_schedule(
            epoch=epoch,
            kl_beta=self.kl_beta,
            kl_annealing_epochs=self.kl_annealing_epochs,
            kl_annealing_type=self.kl_annealing_type,
            kl_annealing_cycles=self.kl_annealing_cycles,
            kl_annealing_ratio=self.kl_annealing_ratio,
        )

        # Semantic schedules
        self.current_lambda_vol = get_semantic_schedule(
            epoch, self.lambda_vol,
            self.semantic_start_epoch, self.semantic_annealing_epochs
        )
        self.current_lambda_loc = get_semantic_schedule(
            epoch, self.lambda_loc,
            self.semantic_start_epoch, self.semantic_annealing_epochs
        )
        self.current_lambda_shape = get_semantic_schedule(
            epoch, self.lambda_shape,
            self.semantic_start_epoch, self.semantic_annealing_epochs
        )

        # Cross-partition independence schedule (uses same annealing as semantic)
        self.current_lambda_cross_partition = get_semantic_schedule(
            epoch, self.lambda_cross_partition,
            self.cross_partition_start_epoch, self.semantic_annealing_epochs
        )

        # Manifold density schedule (uses same annealing as semantic)
        self.current_lambda_manifold = get_semantic_schedule(
            epoch, self.lambda_manifold,
            self.manifold_start_epoch, self.semantic_annealing_epochs
        )

    def _compute_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            x: Original input [B, C, D, H, W]
            x_hat: Reconstruction [B, C, D, H, W]

        Returns:
            MSE loss (scalar)
        """
        if self.loss_reduction == "mean":
            return F.mse_loss(x_hat, x, reduction="mean")
        else:
            return F.mse_loss(x_hat, x, reduction="sum")

    def _compute_kl_residual(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute KL divergence on residual dimensions only.

        Args:
            mu: Full posterior mean [B, z_dim]
            logvar: Full posterior logvar [B, z_dim]

        Returns:
            Dictionary with kl_raw and kl_constrained
        """
        # Extract residual dimensions
        mu_res = mu[:, self.residual_start:self.residual_end]
        logvar_res = logvar[:, self.residual_start:self.residual_end]

        # KL per dimension: 0.5 * (exp(logvar) + mu² - 1 - logvar)
        kl_per_dim = 0.5 * (torch.exp(logvar_res) + mu_res ** 2 - 1 - logvar_res)

        # Sum over dimensions, mean over batch
        kl_per_sample = kl_per_dim.sum(dim=1)  # [B]
        kl_raw = kl_per_sample.mean()

        # Apply free bits
        if self.kl_free_bits > 0:
            if self.kl_free_bits_mode == "batch_mean":
                # Clamp batch mean per dimension
                kl_per_dim_mean = kl_per_dim.mean(dim=0)  # [D]
                kl_clamped = torch.clamp(kl_per_dim_mean, min=self.kl_free_bits)
                kl_constrained = kl_clamped.sum()
            else:  # per_sample
                kl_clamped = torch.clamp(kl_per_dim, min=self.kl_free_bits)
                kl_constrained = kl_clamped.sum(dim=1).mean()
        else:
            kl_constrained = kl_raw

        return {
            "kl_raw": kl_raw,
            "kl_constrained": kl_constrained,
        }

    def _compute_tc_residual(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TC penalty on residual dimensions with DDP-aware gathering.

        In multi-GPU training, latents are gathered across all GPUs for better
        TC estimation with larger effective batch size.

        Args:
            z: Sampled latent [B, z_dim]
            mu: Posterior mean [B, z_dim]
            logvar: Posterior logvar [B, z_dim]

        Returns:
            TC loss (scalar)
        """
        if not self.use_tc_residual:
            return torch.tensor(0.0, device=z.device)

        # Extract residual dimensions
        z_res = z[:, self.residual_start:self.residual_end]
        mu_res = mu[:, self.residual_start:self.residual_end]
        logvar_res = logvar[:, self.residual_start:self.residual_end]

        # Use DDP-aware TC computation for multi-GPU
        return compute_tc_ddp_aware(
            z=z_res,
            mu=mu_res,
            logvar=logvar_res,
            dataset_size=self.dataset_size,
            use_ddp_gather=self.use_ddp_gather,
            estimator=self.tc_estimator,
        )

    def _compute_semantic_losses(
        self,
        semantic_preds: Dict[str, torch.Tensor],
        semantic_targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute regression losses on semantic predictions.

        Args:
            semantic_preds: Model predictions per partition
            semantic_targets: Ground truth features per partition

        Returns:
            Dictionary with loss per partition
        """
        losses = {}

        for name, pred in semantic_preds.items():
            if name in semantic_targets:
                target = semantic_targets[name]
                mse = F.mse_loss(pred, target, reduction="mean")
                losses[f"{name}_mse"] = mse

        return losses

    def _compute_cross_partition_loss(
        self,
        mu: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute cross-partition independence loss.

        Penalizes correlations between supervised latent partitions (z_vol, z_loc, z_shape)
        to encourage statistical independence. This helps ensure volume, location, and
        shape encodings remain independent factors for interpretable Neural ODE dynamics.

        Args:
            mu: Posterior mean [B, z_dim]

        Returns:
            Dictionary with:
                - loss: Total cross-partition penalty
                - per_pair: Per-pair absolute correlations for logging
        """
        if len(self.partition_indices) < 2:
            # Need at least 2 partitions for cross-correlation
            return {
                "loss": torch.tensor(0.0, device=mu.device),
                "per_pair": {},
            }

        result = compute_cross_partition_loss(
            mu=mu,
            partition_indices=self.partition_indices,
            use_ddp_gather=self.use_ddp_gather,
            compute_in_fp32=True,
        )

        return {
            "loss": result["loss"],
            "per_pair": result["per_pair"],
        }

    def _compute_manifold_density_loss(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute manifold density regularization loss.

        Penalizes residual latents that drift far from the prior N(0, I).
        Under N(0, I), ||z||^2 ~ chi-squared(d), with E[||z||^2] = d.

        This loss = E[(||z_residual||^2 / d - 1)^2], which is 0 when
        the squared norm matches its expected value under the prior.

        This regularization is critical for Neural ODE training, where
        ODE predictions must stay on the learned manifold during integration.

        Args:
            z: Sampled latent [B, z_dim]

        Returns:
            Manifold density loss (scalar)
        """
        # Extract residual dimensions
        z_res = z[:, self.residual_start:self.residual_end]

        # Squared norm per sample: [B]
        z_norm_sq = (z_res ** 2).sum(dim=1)

        # Deviation from expected norm (E[||z||^2] = d under prior)
        # When z ~ N(0, I), ||z||^2/d should be ~1
        deviation = z_norm_sq / self.residual_dim - 1.0

        # Mean squared deviation
        return (deviation ** 2).mean()

    def _check_nan_in_latents(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        step_type: str = "train",
    ) -> bool:
        """Check for NaN/Inf in latent tensors and log diagnostics.

        Args:
            mu: Posterior mean [B, z_dim]
            logvar: Posterior logvar [B, z_dim]
            z: Sampled latent [B, z_dim]
            step_type: "train" or "val" for logging prefix

        Returns:
            True if NaN/Inf detected, False otherwise
        """
        has_nan = False

        # Check full tensors
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            nan_count = torch.isnan(mu).sum().item()
            inf_count = torch.isinf(mu).sum().item()
            logger.error(
                f"NaN/Inf in mu at {step_type} step {self.global_step}: "
                f"NaN={nan_count}, Inf={inf_count}"
            )
            has_nan = True

        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            nan_count = torch.isnan(logvar).sum().item()
            inf_count = torch.isinf(logvar).sum().item()
            logger.error(
                f"NaN/Inf in logvar at {step_type} step {self.global_step}: "
                f"NaN={nan_count}, Inf={inf_count}"
            )
            has_nan = True

        if torch.isnan(z).any() or torch.isinf(z).any():
            nan_count = torch.isnan(z).sum().item()
            inf_count = torch.isinf(z).sum().item()
            logger.error(
                f"NaN/Inf in z at {step_type} step {self.global_step}: "
                f"NaN={nan_count}, Inf={inf_count}"
            )
            has_nan = True

        # If NaN detected, identify which partition(s) are affected
        if has_nan:
            partition_info = self.model.get_partition_info()
            for name, config in partition_info.items():
                start, end = config["start_idx"], config["end_idx"]
                mu_part = mu[:, start:end]
                if torch.isnan(mu_part).any() or torch.isinf(mu_part).any():
                    nan_frac = (torch.isnan(mu_part) | torch.isinf(mu_part)).float().mean().item()
                    logger.error(
                        f"  Partition '{name}' (dims {start}:{end}) has NaN/Inf: "
                        f"{nan_frac*100:.1f}% affected"
                    )

            # Log to wandb/tensorboard for visibility
            self.log(f"{step_type}_nan/detected", 1.0, sync_dist=True)

        return has_nan

    def _check_nan_in_semantic_targets(
        self,
        semantic_targets: Dict[str, torch.Tensor],
    ) -> bool:
        """Check for NaN/Inf in semantic targets.

        Args:
            semantic_targets: Dictionary of semantic target tensors

        Returns:
            True if NaN/Inf detected, False otherwise
        """
        has_nan = False
        for name, target in semantic_targets.items():
            if torch.isnan(target).any() or torch.isinf(target).any():
                nan_count = torch.isnan(target).sum().item()
                inf_count = torch.isinf(target).sum().item()
                logger.error(
                    f"NaN/Inf in semantic target '{name}' at step {self.global_step}: "
                    f"NaN={nan_count}, Inf={inf_count}, shape={target.shape}"
                )
                has_nan = True
        return has_nan

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dictionary with "image", "seg", "semantic_features"
            batch_idx: Batch index

        Returns:
            Total loss
        """
        x = batch["image"]

        # Forward pass
        x_hat, mu, logvar, z, semantic_preds = self.model(x)

        # Check for NaN/Inf in latents (critical for diagnosing training issues)
        self._check_nan_in_latents(mu, logvar, z, "train")

        # Reconstruction loss
        recon_loss = self._compute_reconstruction_loss(x, x_hat)

        # KL on residual
        kl_dict = self._compute_kl_residual(mu, logvar)
        kl_loss = self.current_beta * kl_dict["kl_constrained"]

        # TC on residual
        tc_loss = self.lambda_tc * self._compute_tc_residual(z, mu, logvar)

        # Semantic losses
        semantic_losses = {}
        total_semantic_loss = torch.tensor(0.0, device=x.device)

        if "semantic_features" in batch:
            semantic_targets = batch["semantic_features"]

            # Check for NaN/Inf in semantic targets (data pipeline issue)
            self._check_nan_in_semantic_targets(semantic_targets)

            semantic_losses = self._compute_semantic_losses(semantic_preds, semantic_targets)

            # Weighted sum of semantic losses
            if "z_vol_mse" in semantic_losses:
                total_semantic_loss = total_semantic_loss + self.current_lambda_vol * semantic_losses["z_vol_mse"]
            if "z_loc_mse" in semantic_losses:
                total_semantic_loss = total_semantic_loss + self.current_lambda_loc * semantic_losses["z_loc_mse"]
            if "z_shape_mse" in semantic_losses:
                total_semantic_loss = total_semantic_loss + self.current_lambda_shape * semantic_losses["z_shape_mse"]

        # Cross-partition independence loss (penalize correlations between partitions)
        cross_part_result = self._compute_cross_partition_loss(mu)
        cross_partition_loss = self.current_lambda_cross_partition * cross_part_result["loss"]

        # Manifold density loss (keep residual latents near prior)
        manifold_loss_raw = self._compute_manifold_density_loss(z)
        manifold_loss = self.current_lambda_manifold * manifold_loss_raw

        # Total loss
        loss = recon_loss + kl_loss + tc_loss + total_semantic_loss + cross_partition_loss + manifold_loss

        # Logging
        self.log("train_step/loss", loss, prog_bar=True, sync_dist=True)

        # Epoch-level metrics
        self.log("train_epoch/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/recon", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/kl_raw", kl_dict["kl_raw"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/kl_constrained", kl_dict["kl_constrained"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/tc", tc_loss / max(self.lambda_tc, 1e-6), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/semantic_total", total_semantic_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/cross_partition", cross_part_result["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/manifold", manifold_loss_raw, on_step=False, on_epoch=True, sync_dist=True)

        # Per-semantic losses
        for name, mse in semantic_losses.items():
            self.log(f"train_epoch/{name}", mse, on_step=False, on_epoch=True, sync_dist=True)

        # Per-pair cross-partition correlations
        for pair_name, corr_val in cross_part_result["per_pair"].items():
            self.log(f"cross_part/{pair_name}", corr_val, on_step=False, on_epoch=True, sync_dist=True)

        # Schedules
        self.log("sched/beta", self.current_beta, on_step=False, on_epoch=True)
        self.log("sched/lambda_vol", self.current_lambda_vol, on_step=False, on_epoch=True)
        self.log("sched/lambda_loc", self.current_lambda_loc, on_step=False, on_epoch=True)
        self.log("sched/lambda_shape", self.current_lambda_shape, on_step=False, on_epoch=True)
        self.log("sched/lambda_cross_partition", self.current_lambda_cross_partition, on_step=False, on_epoch=True)
        self.log("sched/lambda_manifold", self.current_lambda_manifold, on_step=False, on_epoch=True)
        self.log("sched/free_bits", self.kl_free_bits, on_step=False, on_epoch=True)

        # Latent statistics
        self.log("train_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/z_std", z.std(), on_step=False, on_epoch=True, sync_dist=True)

        # Residual-specific stats
        mu_res = mu[:, self.residual_start:self.residual_end]
        self.log("train_epoch/mu_res_std", mu_res.std(), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Dictionary with "image", "seg", "semantic_features"
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        x = batch["image"]

        # Forward pass
        x_hat, mu, logvar, z, semantic_preds = self.model(x)

        # Check for NaN/Inf in latents
        self._check_nan_in_latents(mu, logvar, z, "val")

        # Reconstruction loss
        recon_loss = self._compute_reconstruction_loss(x, x_hat)

        # KL on residual
        kl_dict = self._compute_kl_residual(mu, logvar)
        kl_loss = self.current_beta * kl_dict["kl_constrained"]

        # TC on residual
        tc_loss = self.lambda_tc * self._compute_tc_residual(z, mu, logvar)

        # Semantic losses
        semantic_losses = {}
        total_semantic_loss = torch.tensor(0.0, device=x.device)

        if "semantic_features" in batch:
            semantic_targets = batch["semantic_features"]

            # Check for NaN/Inf in semantic targets
            self._check_nan_in_semantic_targets(semantic_targets)

            semantic_losses = self._compute_semantic_losses(semantic_preds, semantic_targets)

            if "z_vol_mse" in semantic_losses:
                total_semantic_loss = total_semantic_loss + self.current_lambda_vol * semantic_losses["z_vol_mse"]
            if "z_loc_mse" in semantic_losses:
                total_semantic_loss = total_semantic_loss + self.current_lambda_loc * semantic_losses["z_loc_mse"]
            if "z_shape_mse" in semantic_losses:
                total_semantic_loss = total_semantic_loss + self.current_lambda_shape * semantic_losses["z_shape_mse"]

        # Cross-partition independence loss
        cross_part_result = self._compute_cross_partition_loss(mu)
        cross_partition_loss = self.current_lambda_cross_partition * cross_part_result["loss"]

        # Manifold density loss
        manifold_loss_raw = self._compute_manifold_density_loss(z)
        manifold_loss = self.current_lambda_manifold * manifold_loss_raw

        # Total loss
        loss = recon_loss + kl_loss + tc_loss + total_semantic_loss + cross_partition_loss + manifold_loss

        # Logging
        self.log("val_epoch/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_epoch/recon", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/kl_raw", kl_dict["kl_raw"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/tc", tc_loss / max(self.lambda_tc, 1e-6), on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/semantic_total", total_semantic_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/cross_partition", cross_part_result["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/manifold", manifold_loss_raw, on_step=False, on_epoch=True, sync_dist=True)

        # Per-semantic losses
        for name, mse in semantic_losses.items():
            self.log(f"val_epoch/{name}", mse, on_step=False, on_epoch=True, sync_dist=True)

        # Per-pair cross-partition correlations
        for pair_name, corr_val in cross_part_result["per_pair"].items():
            self.log(f"val_cross_part/{pair_name}", corr_val, on_step=False, on_epoch=True, sync_dist=True)

        # Collapse diagnostics on first batch
        if batch_idx == 0 and self.log_collapse_diagnostics:
            self._log_collapse_diagnostics(x, mu, logvar)

        # Per-modality metrics (every 10 batches)
        if batch_idx % 10 == 0:
            self._log_modality_metrics(x, x_hat, batch_idx)

        return loss

    def _log_collapse_diagnostics(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> None:
        """Log diagnostic metrics for collapse detection.

        Args:
            x: Original input
            mu: Posterior mean
            logvar: Posterior logvar
        """
        with torch.no_grad():
            # Deterministic reconstruction (using mu, no sampling)
            x_hat_mu = self.model.decode(mu)
            recon_mu_mse = F.mse_loss(x_hat_mu, x)
            self.log("diag/recon_mu_mse", recon_mu_mse, sync_dist=True)

            # z=0 ablation (tests if decoder ignores z)
            z_zero = torch.zeros_like(mu)
            x_hat_z0 = self.model.decode(z_zero)
            recon_z0_mse = F.mse_loss(x_hat_z0, x)
            self.log("diag/recon_z0_mse", recon_z0_mse, sync_dist=True)

            # Active units on residual (variance > threshold)
            mu_res = mu[:, self.residual_start:self.residual_end]
            mu_var = mu_res.var(dim=0)  # [D_res]
            au_count = (mu_var > 0.01).sum().float()
            au_frac = au_count / self.residual_dim
            self.log("diag/au_count_residual", au_count, sync_dist=True)
            self.log("diag/au_frac_residual", au_frac, sync_dist=True)

            # Semantic partition activity
            for name in ["z_vol", "z_loc", "z_shape"]:
                if name in self.model.partitioning:
                    config = self.model.partitioning[name]
                    mu_part = mu[:, config["start_idx"]:config["end_idx"]]
                    part_var = mu_part.var(dim=0).mean()
                    self.log(f"diag/{name}_var", part_var, sync_dist=True)

    def _log_modality_metrics(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Log per-modality reconstruction metrics.

        Args:
            x: Original input [B, C, D, H, W]
            x_hat: Reconstruction [B, C, D, H, W]
            batch_idx: Batch index
        """
        with torch.no_grad():
            for i, mod_name in enumerate(self.modality_names):
                if i < x.shape[1]:
                    # Per-modality MSE
                    mod_mse = F.mse_loss(x_hat[:, i], x[:, i])
                    self.log(f"val_epoch/recon_{mod_name}", mod_mse,
                             on_step=False, on_epoch=True, sync_dist=True)

                    # SSIM and PSNR (first sample only)
                    if batch_idx == 0:
                        ssim = compute_ssim_2d_slices(
                            x_hat[0:1, i:i+1], x[0:1, i:i+1]
                        )
                        psnr = compute_psnr_3d(
                            x_hat[0:1, i:i+1], x[0:1, i:i+1]
                        )
                        self.log(f"val_epoch/ssim_{mod_name}", ssim, sync_dist=True)
                        self.log(f"val_epoch/psnr_{mod_name}", psnr, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer.

        Returns:
            AdamW optimizer
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Test step for final evaluation on held-out test set.

        This step is called during trainer.test() after training completes.
        Computes all losses and metrics with test_epoch/ prefix.

        Args:
            batch: Dictionary with "image", "seg", "semantic_features"
            batch_idx: Batch index

        Returns:
            Test loss
        """
        x = batch["image"]

        # Forward pass
        x_hat, mu, logvar, z, semantic_preds = self.model(x)

        # Reconstruction loss
        recon_loss = self._compute_reconstruction_loss(x, x_hat)

        # KL on residual
        kl_dict = self._compute_kl_residual(mu, logvar)
        kl_loss = self.current_beta * kl_dict["kl_constrained"]

        # TC on residual
        tc_loss = self.lambda_tc * self._compute_tc_residual(z, mu, logvar)

        # Semantic losses
        semantic_losses = {}
        total_semantic_loss = torch.tensor(0.0, device=x.device)

        if "semantic_features" in batch:
            semantic_targets = batch["semantic_features"]
            semantic_losses = self._compute_semantic_losses(semantic_preds, semantic_targets)

            if "z_vol_mse" in semantic_losses:
                total_semantic_loss = total_semantic_loss + self.current_lambda_vol * semantic_losses["z_vol_mse"]
            if "z_loc_mse" in semantic_losses:
                total_semantic_loss = total_semantic_loss + self.current_lambda_loc * semantic_losses["z_loc_mse"]
            if "z_shape_mse" in semantic_losses:
                total_semantic_loss = total_semantic_loss + self.current_lambda_shape * semantic_losses["z_shape_mse"]

        # Cross-partition independence loss
        cross_part_result = self._compute_cross_partition_loss(mu)
        cross_partition_loss = self.current_lambda_cross_partition * cross_part_result["loss"]

        # Manifold density loss
        manifold_loss_raw = self._compute_manifold_density_loss(z)
        manifold_loss = self.current_lambda_manifold * manifold_loss_raw

        # Total loss
        loss = recon_loss + kl_loss + tc_loss + total_semantic_loss + cross_partition_loss + manifold_loss

        # Log test metrics with test_epoch/ prefix
        self.log("test_epoch/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_epoch/recon", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_epoch/kl_raw", kl_dict["kl_raw"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_epoch/tc", tc_loss / max(self.lambda_tc, 1e-6), on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_epoch/semantic_total", total_semantic_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_epoch/cross_partition", cross_part_result["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_epoch/manifold", manifold_loss_raw, on_step=False, on_epoch=True, sync_dist=True)

        # Per-semantic losses
        for name, mse in semantic_losses.items():
            self.log(f"test_epoch/{name}", mse, on_step=False, on_epoch=True, sync_dist=True)

        # Per-pair cross-partition correlations
        for pair_name, corr_val in cross_part_result["per_pair"].items():
            self.log(f"test_cross_part/{pair_name}", corr_val, on_step=False, on_epoch=True, sync_dist=True)

        # Per-modality metrics (compute for all batches in test)
        with torch.no_grad():
            for i, mod_name in enumerate(self.modality_names):
                if i < x.shape[1]:
                    mod_mse = F.mse_loss(x_hat[:, i], x[:, i])
                    self.log(f"test_epoch/recon_{mod_name}", mod_mse,
                             on_step=False, on_epoch=True, sync_dist=True)

                    # SSIM and PSNR for test set
                    ssim = compute_ssim_2d_slices(x_hat[:, i:i+1], x[:, i:i+1])
                    psnr = compute_psnr_3d(x_hat[:, i:i+1], x[:, i:i+1])
                    self.log(f"test_epoch/ssim_{mod_name}", ssim,
                             on_step=False, on_epoch=True, sync_dist=True)
                    self.log(f"test_epoch/psnr_{mod_name}", psnr,
                             on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_fit_start(self) -> None:
        """Called at the start of fit. Update dataset size."""
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            # Check if datamodule has train_dataset attribute
            if hasattr(self.trainer.datamodule, "train_dataset"):
                try:
                    self.dataset_size = len(self.trainer.datamodule.train_dataset)
                except Exception:
                    pass
        if self.dataset_size == 1000 and hasattr(self.trainer, "train_dataloader"):
            # Fallback: try to get size from dataloader
            try:
                self.dataset_size = len(self.trainer.train_dataloader.dataset)
            except Exception:
                pass
        logger.info(f"Dataset size for TC estimation: {self.dataset_size}")
