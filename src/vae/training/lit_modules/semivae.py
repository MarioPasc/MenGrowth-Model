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
from ...losses.dcor import compute_dcor_loss, PartitionDCorBuffer
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


def get_semantic_schedule_with_decay(
    epoch: int,
    target_lambda: float,
    start_epoch: int,
    annealing_epochs: int,
    decay_start_epoch: int = -1,
    decay_target_fraction: float = 0.5,
    decay_epochs: int = 200,
) -> float:
    """Three-phase schedule: ramp-up -> plateau -> decay.

    Phase 1 (ramp): Linear 0->target over [start, start+annealing]
    Phase 2 (plateau): Constant at target
    Phase 3 (decay): Linear target->(target*decay_fraction) over
                     [decay_start, decay_start+decay_epochs]

    The decay phase releases encoder capacity in late training, allowing
    the residual partition to reclaim information after semantic learning
    has plateaued.

    Args:
        epoch: Current epoch
        target_lambda: Target lambda value at plateau
        start_epoch: Epoch to start ramp-up
        annealing_epochs: Epochs for linear warmup
        decay_start_epoch: Epoch to start decay (-1 = no decay)
        decay_target_fraction: Final lambda as fraction of target (e.g. 0.5 = 50%)
        decay_epochs: Number of epochs for decay phase

    Returns:
        Current lambda value
    """
    # Phase 1: ramp-up
    if epoch < start_epoch:
        return 0.0
    effective = epoch - start_epoch
    if effective < annealing_epochs:
        return target_lambda * (effective / annealing_epochs)

    # Phase 2: plateau (or no decay configured)
    if decay_start_epoch < 0 or epoch < decay_start_epoch:
        return target_lambda

    # Phase 3: decay
    decay_progress = min(1.0, (epoch - decay_start_epoch) / max(decay_epochs, 1))
    return target_lambda * (1.0 - (1.0 - decay_target_fraction) * decay_progress)


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
        # Legacy shared schedule (used if per-partition schedules not specified)
        semantic_start_epoch: int = 10,
        semantic_annealing_epochs: int = 20,
        # Per-partition curriculum schedules (override shared schedule if > 0)
        vol_start_epoch: int = -1,  # -1 means use semantic_start_epoch
        vol_annealing_epochs: int = -1,
        loc_start_epoch: int = -1,
        loc_annealing_epochs: int = -1,
        shape_start_epoch: int = -1,
        shape_annealing_epochs: int = -1,
        # TC regularization
        use_tc_residual: bool = True,
        lambda_tc: float = 2.0,
        tc_estimator: str = "minibatch_weighted",
        tc_start_epoch: int = 10,
        tc_annealing_epochs: int = 20,
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
        # KL regularization on supervised partitions (prevents posterior collapse)
        # A small KL weight on supervised dims prevents logvar from collapsing to min
        kl_beta_supervised: float = 0.1,
        kl_supervised_free_bits: float = 0.05,
        # Distance Correlation (replaces cross-partition)
        lambda_dcor: float = 0.0,
        dcor_start_epoch: int = 150,
        dcor_annealing_epochs: int = 50,
        dcor_buffer_size: int = 256,
        # Gradient isolation
        gradient_isolation: bool = False,
        # Auxiliary residual reconstruction
        lambda_aux_recon: float = 0.0,
        aux_recon_target_size: int = 64,
        # Lambda decay phase
        sem_decay_start_epoch: int = -1,
        sem_decay_target_fraction: float = 0.5,
        sem_decay_epochs: int = 200,
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
            semantic_start_epoch: Default epoch to start semantic supervision
            semantic_annealing_epochs: Default epochs for semantic loss warmup
            vol_start_epoch: Epoch to start volume supervision (-1 = use default)
            vol_annealing_epochs: Epochs for volume warmup (-1 = use default)
            loc_start_epoch: Epoch to start location supervision (-1 = use default)
            loc_annealing_epochs: Epochs for location warmup (-1 = use default)
            shape_start_epoch: Epoch to start shape supervision (-1 = use default)
            shape_annealing_epochs: Epochs for shape warmup (-1 = use default)
            use_tc_residual: Whether to apply TC penalty on residual
            lambda_tc: Weight for TC penalty
            tc_estimator: TC estimator type ("minibatch_weighted" or "stratified")
            tc_start_epoch: Epoch to start TC penalty (delayed for stability)
            tc_annealing_epochs: Epochs for TC penalty warmup
            kl_beta: Target beta value after annealing
            kl_annealing_epochs: Total epochs for KL annealing
            kl_annealing_type: "linear" or "cyclical"
            kl_annealing_cycles: Number of cycles for cyclical annealing
            kl_annealing_ratio: Fraction of cycle for annealing phase
            kl_free_bits: Per-dim KL floor (nats)
            kl_free_bits_mode: "per_sample" or "batch_mean"
            kl_beta_supervised: Beta weight for KL on supervised partitions (prevents
                               posterior variance collapse). Set to 0 to disable.
            kl_supervised_free_bits: Per-dim KL floor for supervised partitions
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

        # Per-partition curriculum schedules (use defaults if -1)
        self.vol_start_epoch = vol_start_epoch if vol_start_epoch >= 0 else semantic_start_epoch
        self.vol_annealing_epochs = vol_annealing_epochs if vol_annealing_epochs >= 0 else semantic_annealing_epochs
        self.loc_start_epoch = loc_start_epoch if loc_start_epoch >= 0 else semantic_start_epoch
        self.loc_annealing_epochs = loc_annealing_epochs if loc_annealing_epochs >= 0 else semantic_annealing_epochs
        self.shape_start_epoch = shape_start_epoch if shape_start_epoch >= 0 else semantic_start_epoch
        self.shape_annealing_epochs = shape_annealing_epochs if shape_annealing_epochs >= 0 else semantic_annealing_epochs

        # TC settings
        self.use_tc_residual = use_tc_residual
        self.lambda_tc = lambda_tc
        self.tc_estimator = tc_estimator
        self.tc_start_epoch = tc_start_epoch
        self.tc_annealing_epochs = tc_annealing_epochs

        # Cross-partition independence settings
        self.lambda_cross_partition = lambda_cross_partition
        self.cross_partition_start_epoch = cross_partition_start_epoch

        # Manifold density regularization settings
        self.lambda_manifold = lambda_manifold
        self.manifold_start_epoch = manifold_start_epoch

        # Distance Correlation settings (replaces cross-partition)
        self.lambda_dcor = lambda_dcor
        self.dcor_start_epoch = dcor_start_epoch
        self.dcor_annealing_epochs = dcor_annealing_epochs
        self.dcor_buffer_size = dcor_buffer_size

        # Gradient isolation
        self.gradient_isolation = gradient_isolation

        # Auxiliary residual reconstruction
        self.lambda_aux_recon = lambda_aux_recon
        self.aux_recon_target_size = aux_recon_target_size

        # Lambda decay phase
        self.sem_decay_start_epoch = sem_decay_start_epoch
        self.sem_decay_target_fraction = sem_decay_target_fraction
        self.sem_decay_epochs = sem_decay_epochs

        # KL settings
        self.kl_beta = kl_beta
        self.kl_annealing_epochs = kl_annealing_epochs
        self.kl_annealing_type = kl_annealing_type
        self.kl_annealing_cycles = kl_annealing_cycles
        self.kl_annealing_ratio = kl_annealing_ratio
        self.kl_free_bits = kl_free_bits
        self.kl_free_bits_mode = kl_free_bits_mode

        # KL on supervised partitions (prevents posterior collapse)
        self.kl_beta_supervised = kl_beta_supervised
        self.kl_supervised_free_bits = kl_supervised_free_bits

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
        self.current_lambda_tc = 0.0
        self.current_lambda_cross_partition = 0.0
        self.current_lambda_manifold = 0.0
        self.current_lambda_dcor = 0.0

        # Distance Correlation buffer
        self.dcor_buffer: Optional[PartitionDCorBuffer] = None
        if dcor_buffer_size > 0 and lambda_dcor > 0:
            self.dcor_buffer = PartitionDCorBuffer(buffer_size=dcor_buffer_size)

        # Get residual indices from model
        self.residual_start, self.residual_end = model.get_residual_indices()
        self.residual_dim = self.residual_end - self.residual_start

        # Get partition indices for cross-partition loss (supervised partitions only)
        self.partition_indices = {}
        self.supervised_dim = 0
        for name, config in model.get_partition_info().items():
            if config["supervision"] == "regression":
                self.partition_indices[name] = (config["start_idx"], config["end_idx"])
                self.supervised_dim += config["end_idx"] - config["start_idx"]

        # Build target feature index mappings for partitions where
        # target_features is a subset of the full feature group.
        # The data pipeline always extracts the full feature group; this mapping
        # lets us select only the configured features at loss computation time.
        from ...data.semantic_features import get_feature_groups
        feature_groups = get_feature_groups()
        group_key_map = {"z_vol": "volume", "z_loc": "location", "z_shape": "shape"}
        self._target_feature_indices: Dict[str, List[int]] = {}
        for name, config in model.get_partition_info().items():
            if config["supervision"] != "regression":
                continue
            group_key = group_key_map.get(name)
            if group_key is None:
                continue
            full_features = feature_groups[group_key]
            target_features = config.get("target_features", full_features)
            if len(target_features) < len(full_features):
                self._target_feature_indices[name] = [
                    full_features.index(f) for f in target_features
                ]

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
            # Per-partition curriculum schedules (-1 = use shared schedule)
            vol_start_epoch=cfg.loss.get("vol_start_epoch", -1),
            vol_annealing_epochs=cfg.loss.get("vol_annealing_epochs", -1),
            loc_start_epoch=cfg.loss.get("loc_start_epoch", -1),
            loc_annealing_epochs=cfg.loss.get("loc_annealing_epochs", -1),
            shape_start_epoch=cfg.loss.get("shape_start_epoch", -1),
            shape_annealing_epochs=cfg.loss.get("shape_annealing_epochs", -1),
            use_tc_residual=cfg.loss.get("use_tc_residual", True),
            lambda_tc=cfg.loss.get("lambda_tc", 2.0),
            tc_estimator=cfg.loss.get("tc_estimator", "minibatch_weighted"),
            tc_start_epoch=cfg.loss.get("tc_start_epoch", 10),
            tc_annealing_epochs=cfg.loss.get("tc_annealing_epochs", 20),
            lambda_cross_partition=cfg.loss.get("lambda_cross_partition", 5.0),
            cross_partition_start_epoch=cfg.loss.get("cross_partition_start_epoch", 10),
            lambda_manifold=cfg.loss.get("lambda_manifold", 1.0),
            manifold_start_epoch=cfg.loss.get("manifold_start_epoch", 10),
            # Distance Correlation (replaces cross-partition)
            lambda_dcor=cfg.loss.get("lambda_dcor", 0.0),
            dcor_start_epoch=cfg.loss.get("dcor_start_epoch", 150),
            dcor_annealing_epochs=cfg.loss.get("dcor_annealing_epochs", 50),
            dcor_buffer_size=cfg.loss.get("dcor_buffer_size", 256),
            # Gradient isolation
            gradient_isolation=cfg.loss.get("gradient_isolation", False),
            # Auxiliary residual reconstruction
            lambda_aux_recon=cfg.loss.get("lambda_aux_recon", 0.0),
            aux_recon_target_size=cfg.loss.get("aux_recon_target_size", 64),
            # Lambda decay
            sem_decay_start_epoch=cfg.loss.get("sem_decay_start_epoch", -1),
            sem_decay_target_fraction=cfg.loss.get("sem_decay_target_fraction", 0.5),
            sem_decay_epochs=cfg.loss.get("sem_decay_epochs", 200),
            # General
            kl_beta=cfg.train.kl_beta,
            kl_annealing_epochs=cfg.train.kl_annealing_epochs,
            kl_annealing_type=cfg.train.kl_annealing_type,
            kl_annealing_cycles=cfg.train.get("kl_annealing_cycles", 4),
            kl_annealing_ratio=cfg.train.get("kl_annealing_ratio", 0.5),
            kl_free_bits=cfg.train.get("kl_free_bits", 0.2),
            kl_free_bits_mode=cfg.train.get("kl_free_bits_mode", "batch_mean"),
            kl_beta_supervised=cfg.train.get("kl_beta_supervised", 0.1),
            kl_supervised_free_bits=cfg.train.get("kl_supervised_free_bits", 0.05),
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

        # Semantic schedules with optional decay (per-partition curriculum learning)
        self.current_lambda_vol = get_semantic_schedule_with_decay(
            epoch, self.lambda_vol,
            self.vol_start_epoch, self.vol_annealing_epochs,
            self.sem_decay_start_epoch, self.sem_decay_target_fraction,
            self.sem_decay_epochs,
        )
        self.current_lambda_loc = get_semantic_schedule_with_decay(
            epoch, self.lambda_loc,
            self.loc_start_epoch, self.loc_annealing_epochs,
            self.sem_decay_start_epoch, self.sem_decay_target_fraction,
            self.sem_decay_epochs,
        )
        self.current_lambda_shape = get_semantic_schedule_with_decay(
            epoch, self.lambda_shape,
            self.shape_start_epoch, self.shape_annealing_epochs,
            self.sem_decay_start_epoch, self.sem_decay_target_fraction,
            self.sem_decay_epochs,
        )

        # TC schedule (delayed start for numerical stability)
        self.current_lambda_tc = get_semantic_schedule(
            epoch, self.lambda_tc,
            self.tc_start_epoch, self.tc_annealing_epochs
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

        # Distance Correlation schedule (delayed start, replaces cross-partition)
        self.current_lambda_dcor = get_semantic_schedule(
            epoch, self.lambda_dcor,
            self.dcor_start_epoch, self.dcor_annealing_epochs
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

    def _compute_kl_supervised(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute KL divergence on supervised dimensions.

        This adds light KL regularization to supervised partitions to prevent
        posterior variance collapse. Without this, the semantic regression losses
        can push logvar to the minimum allowed value.

        Args:
            mu: Full posterior mean [B, z_dim]
            logvar: Full posterior logvar [B, z_dim]

        Returns:
            Dictionary with kl_supervised_raw and kl_supervised
        """
        if self.kl_beta_supervised <= 0:
            return {
                "kl_supervised_raw": torch.tensor(0.0, device=mu.device),
                "kl_supervised": torch.tensor(0.0, device=mu.device),
            }

        # Collect KL across all supervised partitions
        kl_total = torch.tensor(0.0, device=mu.device)
        kl_raw_total = torch.tensor(0.0, device=mu.device)

        for name, (start_idx, end_idx) in self.partition_indices.items():
            mu_part = mu[:, start_idx:end_idx]
            logvar_part = logvar[:, start_idx:end_idx]

            # KL per dimension: 0.5 * (exp(logvar) + mu² - 1 - logvar)
            kl_per_dim = 0.5 * (torch.exp(logvar_part) + mu_part ** 2 - 1 - logvar_part)
            kl_per_sample = kl_per_dim.sum(dim=1)
            kl_raw_total = kl_raw_total + kl_per_sample.mean()

            # Apply free bits (lighter than residual)
            if self.kl_supervised_free_bits > 0:
                kl_per_dim_mean = kl_per_dim.mean(dim=0)
                kl_clamped = torch.clamp(kl_per_dim_mean, min=self.kl_supervised_free_bits)
                kl_total = kl_total + kl_clamped.sum()
            else:
                kl_total = kl_total + kl_per_sample.mean()

        return {
            "kl_supervised_raw": kl_raw_total,
            "kl_supervised": kl_total,
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
                # Filter target to match configured target_features subset
                if name in self._target_feature_indices:
                    indices = self._target_feature_indices[name]
                    target = target[:, indices]
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

    def _compute_dcor_loss(
        self,
        mu: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute distance correlation between supervised partitions.

        Uses Székely's dCor² on full multivariate partition vectors (not
        dimension means like the broken cross-partition loss). With the
        EMA buffer, effective sample size reaches 256 despite N=8 per batch.

        Args:
            mu: Posterior mean [B, z_dim]

        Returns:
            Dictionary with:
                - loss: Total dCor² penalty (scalar)
                - per_pair: Per-pair dCor² values for logging
                - n_effective: Effective sample size used
        """
        if len(self.partition_indices) < 2 or self.current_lambda_dcor == 0:
            return {
                "loss": torch.tensor(0.0, device=mu.device),
                "per_pair": {},
                "n_effective": 0,
            }

        result = compute_dcor_loss(
            mu=mu,
            partition_indices=self.partition_indices,
            buffer=self.dcor_buffer,
            use_ddp_gather=self.use_ddp_gather,
            compute_in_fp32=True,
        )

        # Update buffer with current batch (detached)
        if self.dcor_buffer is not None:
            partition_data = {}
            for name, (start, end) in self.partition_indices.items():
                partition_data[name] = mu[:, start:end].detach()
            self.dcor_buffer.update(partition_data)

        return result

    def _compute_aux_recon_loss(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary residual reconstruction loss.

        Decodes from z_residual dims only through a lightweight 64^3 decoder.
        This provides gradient signal to keep residual dims active.

        Args:
            z: Full sampled latent [B, z_dim]
            x: Original input [B, C, D, H, W]

        Returns:
            MSE loss between aux decoder output and downsampled input
        """
        if self.lambda_aux_recon <= 0 or self.model.aux_decoder is None:
            return torch.tensor(0.0, device=x.device)

        # Extract residual dims
        z_residual = z[:, self.residual_start:self.residual_end]

        # Downsample target to 64^3
        target_size = self.aux_recon_target_size
        x_lowres = F.interpolate(
            x, size=(target_size, target_size, target_size),
            mode="trilinear", align_corners=False
        )

        # Decode from residual only
        x_hat_aux = self.model.decode_residual(z_residual)

        return F.mse_loss(x_hat_aux, x_lowres, reduction="mean")

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

        # Forward pass (with gradient isolation if configured)
        x_hat, mu, logvar, z, semantic_preds = self.model(
            x, gradient_isolation=self.gradient_isolation
        )

        # Check for NaN/Inf in latents (critical for diagnosing training issues)
        self._check_nan_in_latents(mu, logvar, z, "train")

        # Reconstruction loss
        recon_loss = self._compute_reconstruction_loss(x, x_hat)

        # KL on residual
        kl_dict = self._compute_kl_residual(mu, logvar)
        kl_loss = self.current_beta * kl_dict["kl_constrained"]

        # KL on supervised partitions (prevents posterior collapse)
        kl_sup_dict = self._compute_kl_supervised(mu, logvar)
        kl_supervised_loss = self.kl_beta_supervised * kl_sup_dict["kl_supervised"]

        # TC on residual
        tc_loss = self.current_lambda_tc * self._compute_tc_residual(z, mu, logvar)

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

        # Cross-partition independence loss (legacy, can be disabled via lambda=0)
        cross_part_result = self._compute_cross_partition_loss(mu)
        cross_partition_loss = self.current_lambda_cross_partition * cross_part_result["loss"]

        # Distance Correlation loss (replaces cross-partition when lambda_dcor > 0)
        dcor_result = self._compute_dcor_loss(mu)
        dcor_loss = self.current_lambda_dcor * dcor_result["loss"]

        # Auxiliary residual reconstruction (prevents z_residual deflation)
        aux_recon_loss = self.lambda_aux_recon * self._compute_aux_recon_loss(z, x)

        # Manifold density loss (keep residual latents near prior)
        manifold_loss_raw = self._compute_manifold_density_loss(z)
        manifold_loss = self.current_lambda_manifold * manifold_loss_raw

        # Total loss
        loss = (
            recon_loss + kl_loss + kl_supervised_loss + tc_loss +
            total_semantic_loss + cross_partition_loss + dcor_loss +
            aux_recon_loss + manifold_loss
        )

        # Logging
        self.log("train_step/loss", loss, prog_bar=True, sync_dist=True)

        # Epoch-level metrics
        self.log("train_epoch/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/recon", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/kl_raw", kl_dict["kl_raw"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/kl_constrained", kl_dict["kl_constrained"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/kl_supervised", kl_sup_dict["kl_supervised"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/tc", tc_loss / max(self.lambda_tc, 1e-6), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/semantic_total", total_semantic_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/cross_partition", cross_part_result["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/dcor", dcor_result["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/aux_recon", aux_recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/manifold", manifold_loss_raw, on_step=False, on_epoch=True, sync_dist=True)

        # Per-semantic losses
        for name, mse in semantic_losses.items():
            self.log(f"train_epoch/{name}", mse, on_step=False, on_epoch=True, sync_dist=True)

        # Per-pair cross-partition correlations (legacy)
        for pair_name, corr_val in cross_part_result["per_pair"].items():
            self.log(f"cross_part/{pair_name}", corr_val, on_step=False, on_epoch=True, sync_dist=True)

        # Per-pair distance correlations
        for pair_name, dcor_val in dcor_result.get("per_pair", {}).items():
            self.log(f"dcor/{pair_name}", dcor_val, on_step=False, on_epoch=True, sync_dist=True)

        # Schedules (current effective lambda values)
        self.log("sched/beta", self.current_beta, on_step=False, on_epoch=True)
        self.log("sched/lambda_vol", self.current_lambda_vol, on_step=False, on_epoch=True)
        self.log("sched/lambda_loc", self.current_lambda_loc, on_step=False, on_epoch=True)
        self.log("sched/lambda_shape", self.current_lambda_shape, on_step=False, on_epoch=True)
        self.log("sched/lambda_tc", self.current_lambda_tc, on_step=False, on_epoch=True)
        self.log("sched/lambda_cross_partition", self.current_lambda_cross_partition, on_step=False, on_epoch=True)
        self.log("sched/lambda_dcor", self.current_lambda_dcor, on_step=False, on_epoch=True)
        self.log("sched/lambda_manifold", self.current_lambda_manifold, on_step=False, on_epoch=True)
        self.log("sched/free_bits", self.kl_free_bits, on_step=False, on_epoch=True)
        self.log("sched/kl_beta_supervised", self.kl_beta_supervised, on_step=False, on_epoch=True)

        # Log curriculum phase indicators (1 if active, 0 if not yet started)
        self.log("curriculum/vol_active", float(self.current_lambda_vol > 0), on_step=False, on_epoch=True)
        self.log("curriculum/loc_active", float(self.current_lambda_loc > 0), on_step=False, on_epoch=True)
        self.log("curriculum/shape_active", float(self.current_lambda_shape > 0), on_step=False, on_epoch=True)
        self.log("curriculum/tc_active", float(self.current_lambda_tc > 0), on_step=False, on_epoch=True)
        self.log("curriculum/dcor_active", float(self.current_lambda_dcor > 0), on_step=False, on_epoch=True)

        # Latent statistics
        self.log("train_epoch/mu_mean", mu.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/logvar_mean", logvar.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_epoch/z_std", z.std(), on_step=False, on_epoch=True, sync_dist=True)

        # Residual-specific stats
        mu_res = mu[:, self.residual_start:self.residual_end]
        self.log("train_epoch/mu_res_std", mu_res.std(), on_step=False, on_epoch=True, sync_dist=True)

        # dCor buffer stats
        if self.dcor_buffer is not None:
            self.log("dcor/buffer_size", float(self.dcor_buffer.size), on_step=False, on_epoch=True)
            self.log("dcor/n_effective", float(dcor_result.get("n_effective", 0)), on_step=False, on_epoch=True)

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

        # Forward pass (no gradient isolation in validation)
        x_hat, mu, logvar, z, semantic_preds = self.model(x)

        # Check for NaN/Inf in latents
        self._check_nan_in_latents(mu, logvar, z, "val")

        # Reconstruction loss
        recon_loss = self._compute_reconstruction_loss(x, x_hat)

        # KL on residual
        kl_dict = self._compute_kl_residual(mu, logvar)
        kl_loss = self.current_beta * kl_dict["kl_constrained"]

        # KL on supervised partitions (prevents posterior collapse)
        kl_sup_dict = self._compute_kl_supervised(mu, logvar)
        kl_supervised_loss = self.kl_beta_supervised * kl_sup_dict["kl_supervised"]

        # TC on residual
        tc_loss = self.current_lambda_tc * self._compute_tc_residual(z, mu, logvar)

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

        # Cross-partition independence loss (legacy)
        cross_part_result = self._compute_cross_partition_loss(mu)
        cross_partition_loss = self.current_lambda_cross_partition * cross_part_result["loss"]

        # Distance Correlation loss
        dcor_result = self._compute_dcor_loss(mu)
        dcor_loss = self.current_lambda_dcor * dcor_result["loss"]

        # Auxiliary residual reconstruction
        aux_recon_loss = self.lambda_aux_recon * self._compute_aux_recon_loss(z, x)

        # Manifold density loss
        manifold_loss_raw = self._compute_manifold_density_loss(z)
        manifold_loss = self.current_lambda_manifold * manifold_loss_raw

        # Total loss
        loss = (
            recon_loss + kl_loss + kl_supervised_loss + tc_loss +
            total_semantic_loss + cross_partition_loss + dcor_loss +
            aux_recon_loss + manifold_loss
        )

        # Compute ODE readiness score for checkpoint selection
        if "semantic_features" in batch:
            # Use dCor for independence scoring if available, else fall back to Pearson
            independence_corrs = dcor_result.get("per_pair", {}) if self.lambda_dcor > 0 else cross_part_result["per_pair"]
            ode_readiness = self._compute_ode_readiness(
                semantic_losses=semantic_losses,
                semantic_targets=semantic_targets,
                cross_partition_corrs=independence_corrs,
            )
        else:
            ode_readiness = torch.tensor(0.0, device=x.device)

        # Logging
        self.log("val_epoch/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_epoch/recon", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/kl_raw", kl_dict["kl_raw"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/kl_supervised", kl_sup_dict["kl_supervised"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/tc", tc_loss / max(self.lambda_tc, 1e-6), on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/semantic_total", total_semantic_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/cross_partition", cross_part_result["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/dcor", dcor_result["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/aux_recon", aux_recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_epoch/manifold", manifold_loss_raw, on_step=False, on_epoch=True, sync_dist=True)
        # ODE readiness score for checkpoint selection (higher = better for Neural ODE)
        self.log("val_epoch/ode_readiness", ode_readiness, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        # Per-semantic losses
        for name, mse in semantic_losses.items():
            self.log(f"val_epoch/{name}", mse, on_step=False, on_epoch=True, sync_dist=True)

        # Per-pair cross-partition correlations (legacy)
        for pair_name, corr_val in cross_part_result["per_pair"].items():
            self.log(f"val_cross_part/{pair_name}", corr_val, on_step=False, on_epoch=True, sync_dist=True)

        # Per-pair distance correlations
        for pair_name, dcor_val in dcor_result.get("per_pair", {}).items():
            self.log(f"val_dcor/{pair_name}", dcor_val, on_step=False, on_epoch=True, sync_dist=True)

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

        # KL on supervised partitions (prevents posterior collapse)
        kl_sup_dict = self._compute_kl_supervised(mu, logvar)
        kl_supervised_loss = self.kl_beta_supervised * kl_sup_dict["kl_supervised"]

        # TC on residual
        tc_loss = self.current_lambda_tc * self._compute_tc_residual(z, mu, logvar)

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
        loss = recon_loss + kl_loss + kl_supervised_loss + tc_loss + total_semantic_loss + cross_partition_loss + manifold_loss

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

    def _compute_ode_readiness(
        self,
        semantic_losses: Dict[str, torch.Tensor],
        semantic_targets: Dict[str, torch.Tensor],
        cross_partition_corrs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute composite ODE readiness score for checkpoint selection.

        This metric combines semantic encoding quality and factor independence
        to select models optimal for downstream Neural ODE training.

        Score = 0.5 * vol_r2_proxy + 0.25 * loc_r2_proxy + 0.25 * independence_score

        Where:
        - vol_r2_proxy: Approximate R² for volume encoding (1 - MSE/Var)
        - loc_r2_proxy: Approximate R² for location encoding
        - independence_score: 1 - max(|cross_partition_correlation|)

        Args:
            semantic_losses: Dictionary with z_vol_mse, z_loc_mse, z_shape_mse
            semantic_targets: Dictionary with target tensors for variance estimation
            cross_partition_corrs: Dictionary with per-pair absolute correlations

        Returns:
            ODE readiness score in [0, 1], higher is better
        """
        device = next(iter(semantic_losses.values())).device if semantic_losses else torch.device("cpu")

        # Compute R² proxies: R² ≈ 1 - MSE/Var(target)
        # Use target variance from batch (this is an approximation)
        vol_r2_proxy = torch.tensor(0.0, device=device)
        loc_r2_proxy = torch.tensor(0.0, device=device)

        # Volume R² proxy
        if "z_vol_mse" in semantic_losses and "z_vol" in semantic_targets:
            vol_mse = semantic_losses["z_vol_mse"]
            vol_target = semantic_targets["z_vol"]
            vol_var = vol_target.var() + 1e-8  # Prevent division by zero
            vol_r2_proxy = torch.clamp(1.0 - vol_mse / vol_var, min=0.0, max=1.0)

        # Location R² proxy
        if "z_loc_mse" in semantic_losses and "z_loc" in semantic_targets:
            loc_mse = semantic_losses["z_loc_mse"]
            loc_target = semantic_targets["z_loc"]
            loc_var = loc_target.var() + 1e-8
            loc_r2_proxy = torch.clamp(1.0 - loc_mse / loc_var, min=0.0, max=1.0)

        # Independence score: 1 - max absolute cross-partition correlation
        # Higher is better (less correlation = more independent)
        independence_score = torch.tensor(1.0, device=device)
        if cross_partition_corrs:
            # Only consider correlations between supervised partitions (not with residual)
            supervised_pairs = [k for k in cross_partition_corrs.keys()
                               if "residual" not in k]
            if supervised_pairs:
                max_corr = max(cross_partition_corrs[k].abs()
                              for k in supervised_pairs)
                independence_score = torch.clamp(1.0 - max_corr, min=0.0, max=1.0)

        # Composite score: weighted combination
        # Volume is most critical for Gompertz ODE (50%)
        # Location is secondary (25%)
        # Independence ensures factors can evolve separately (25%)
        ode_readiness = (
            0.50 * vol_r2_proxy +
            0.25 * loc_r2_proxy +
            0.25 * independence_score
        )

        return ode_readiness

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
