# src/growth/training/lit_modules/sdp_module.py
"""LightningModule for Phase 2: SDP training.

Manages frozen encoder, SDP network, semantic heads, and disentanglement losses.
Optionally supports curriculum scheduling for loss terms.
"""

import logging
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

from growth.losses.dcor import distance_correlation as dcor_fn
from growth.losses.sdp_loss import SDPLoss
from growth.models.projection.partition import LatentPartition
from growth.models.projection.sdp import SDP, SDPWithHeads
from growth.models.projection.semantic_heads import SemanticHeads

logger = logging.getLogger(__name__)


class SDPLitModule(pl.LightningModule):
    """Lightning module for SDP training on precomputed features.

    Operates on precomputed encoder features (full-batch).
    Manages normalization statistics as registered buffers.

    Args:
        config: OmegaConf config dict with sdp, partition, targets,
                loss, curriculum, and training sections.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.cfg = config

        # Build SDP model
        partition = LatentPartition.from_config(
            vol_dim=config.partition.vol_dim,
            loc_dim=config.partition.loc_dim,
            shape_dim=config.partition.shape_dim,
            residual_dim=config.partition.residual_dim,
            n_vol=config.targets.n_vol,
            n_loc=config.targets.n_loc,
            n_shape=config.targets.n_shape,
        )
        sdp = SDP(
            in_dim=config.sdp.in_dim,
            hidden_dim=config.sdp.hidden_dim,
            out_dim=config.sdp.out_dim,
            dropout=config.sdp.dropout,
        )
        heads = SemanticHeads(
            vol_in=config.partition.vol_dim,
            vol_out=config.targets.n_vol,
            loc_in=config.partition.loc_dim,
            loc_out=config.targets.n_loc,
            shape_in=config.partition.shape_dim,
            shape_out=config.targets.n_shape,
        )
        self.model = SDPWithHeads(sdp=sdp, partition=partition, heads=heads)

        # Build loss
        curriculum_cfg = config.get("curriculum", {})
        self.loss_fn = SDPLoss(
            lambda_vol=config.loss.lambda_vol,
            lambda_loc=config.loss.lambda_loc,
            lambda_shape=config.loss.lambda_shape,
            lambda_cov=config.loss.lambda_cov,
            lambda_var=config.loss.lambda_var,
            lambda_dcor=config.loss.lambda_dcor,
            gamma_var=config.loss.gamma_var,
            use_curriculum=curriculum_cfg.get("enabled", True),
            warmup_end=curriculum_cfg.get("warmup_end", 10),
            semantic_end=curriculum_cfg.get("semantic_end", 40),
            independence_end=curriculum_cfg.get("independence_end", 60),
        )

        # Normalization buffers (populated by setup_data)
        self.register_buffer("h_mean", torch.zeros(config.sdp.in_dim))
        self.register_buffer("h_std", torch.ones(config.sdp.in_dim))
        self.register_buffer("target_means", torch.zeros(1))
        self.register_buffer("target_stds", torch.ones(1))

        # Data holders (set by setup_data)
        self._train_dataset: TensorDataset | None = None
        self._val_dataset: TensorDataset | None = None
        self._target_keys = ["vol", "loc", "shape"]

        # Per-target normalization stats
        self._target_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def setup_data(
        self,
        h_train: torch.Tensor,
        targets_train: dict[str, torch.Tensor],
        h_val: torch.Tensor | None = None,
        targets_val: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Set up data with normalization statistics from training set.

        Computes μ, σ on train set only (D14). Stores as registered buffers.

        Args:
            h_train: Training encoder features [N_train, 768].
            targets_train: Dict with "vol", "loc", "shape" target tensors.
            h_val: Optional validation encoder features.
            targets_val: Optional validation targets dict.
        """
        # Compute feature normalization (train only)
        self.h_mean = h_train.mean(dim=0)
        self.h_std = h_train.std(dim=0).clamp(min=1e-8)

        # Compute per-target normalization (train only)
        for key in self._target_keys:
            t = targets_train[key]
            mean = t.mean(dim=0)
            std = t.std(dim=0).clamp(min=1e-8)
            self._target_stats[key] = (mean, std)
            # Register as buffers for checkpoint saving
            self.register_buffer(f"target_mean_{key}", mean)
            self.register_buffer(f"target_std_{key}", std)

        # Normalize and create train dataset
        h_train_norm = self._normalize_features(h_train)
        targets_train_norm = self._normalize_targets(targets_train)
        self._train_dataset = self._build_dataset(h_train_norm, targets_train_norm)

        logger.info(
            f"Train data: {h_train.shape[0]} samples, "
            f"feature mean range: [{self.h_mean.min():.3f}, {self.h_mean.max():.3f}]"
        )

        # Normalize and create val dataset
        if h_val is not None and targets_val is not None:
            h_val_norm = self._normalize_features(h_val)
            targets_val_norm = self._normalize_targets(targets_val)
            self._val_dataset = self._build_dataset(h_val_norm, targets_val_norm)
            logger.info(f"Val data: {h_val.shape[0]} samples")

    def _normalize_features(self, h: torch.Tensor) -> torch.Tensor:
        """Normalize features using train statistics."""
        return (h - self.h_mean) / self.h_std

    def _normalize_targets(self, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize targets using train statistics."""
        normalized = {}
        for key in self._target_keys:
            mean, std = self._target_stats[key]
            normalized[key] = (targets[key] - mean) / std
        return normalized

    def _build_dataset(self, h: torch.Tensor, targets: dict[str, torch.Tensor]) -> TensorDataset:
        """Build TensorDataset from features and target dict."""
        return TensorDataset(
            h,
            targets["vol"],
            targets["loc"],
            targets["shape"],
        )

    def _unpack_batch(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Unpack a batch from TensorDataset into (h, targets_dict)."""
        h, vol, loc, shape = batch
        targets = {"vol": vol, "loc": loc, "shape": shape}
        return h, targets

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """SDP training step.

        Args:
            batch: Tuple from TensorDataset.
            batch_idx: Batch index.

        Returns:
            Total loss for backpropagation.
        """
        h, targets = self._unpack_batch(batch)
        z, partitions, predictions = self.model(h)
        loss, details = self.loss_fn(z, partitions, predictions, targets)

        # Log all loss terms
        for key, value in details.items():
            self.log(f"train/{key}", value, prog_bar=(key == "loss_total"))

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """SDP validation step — computes R^2 per partition + disentanglement metrics.

        Logs 11 inline metrics per epoch alongside losses:
        - R^2 per partition and mean
        - dCor between all supervised partition pairs
        - Max cross-partition Pearson correlation
        - Variance health: pct_dims_std_gt_03/05, mean/min dim std, effective rank
        - Curriculum phase indicator

        Args:
            batch: Tuple from TensorDataset.
            batch_idx: Batch index.
        """
        h, targets = self._unpack_batch(batch)
        z, partitions, predictions = self.model(h)
        loss, details = self.loss_fn(z, partitions, predictions, targets)

        # Log loss
        self.log("val/loss_total", details["loss_total"])

        # R^2 per partition (on normalized predictions)
        r2_values = {}
        for key in self._target_keys:
            pred = predictions[key]
            target = targets[key]
            ss_res = ((pred - target) ** 2).sum()
            ss_tot = ((target - target.mean(dim=0)) ** 2).sum()
            r2 = 1.0 - ss_res / (ss_tot + 1e-8)
            self.log(f"val/r2_{key}", r2, prog_bar=True)
            r2_values[key] = r2

        # --- Inline disentanglement metrics ---

        # R² mean
        r2_mean = sum(r2_values.values()) / len(r2_values)
        self.log("val/r2_mean", r2_mean)

        # dCor between supervised partition pairs
        pairs = [("vol", "loc"), ("vol", "shape"), ("loc", "shape")]
        for name_i, name_j in pairs:
            dcor_val = dcor_fn(partitions[name_i], partitions[name_j])
            self.log(f"val/dcor_{name_i}_{name_j}", dcor_val)

        # Max cross-partition Pearson correlation
        max_corr = torch.tensor(0.0, device=z.device)
        for name_i, name_j in pairs:
            zi = partitions[name_i]
            zj = partitions[name_j]
            corr = torch.corrcoef(torch.cat([zi.T, zj.T], dim=0))
            di = zi.shape[1]
            cross_block = corr[:di, di:]
            pair_max = cross_block.abs().max()
            max_corr = torch.max(max_corr, pair_max)
        self.log("val/max_cross_partition_corr", max_corr)

        # Variance health metrics
        z_std = z.std(dim=0)
        self.log("val/pct_dims_std_gt_03", (z_std > 0.3).float().mean())
        self.log("val/pct_dims_std_gt_05", (z_std > 0.5).float().mean())
        self.log("val/mean_dim_std", z_std.mean())
        self.log("val/min_dim_std", z_std.min())

        # Effective rank via SVD entropy
        z_centered = z - z.mean(dim=0)
        s = torch.linalg.svdvals(z_centered)
        s_norm = s / (s.sum() + 1e-10)
        s_norm = s_norm[s_norm > 1e-10]
        entropy = -(s_norm * torch.log(s_norm)).sum()
        effective_rank = torch.exp(entropy)
        self.log("val/effective_rank", effective_rank)

        # Curriculum phase indicator
        epoch = self.current_epoch
        curriculum_cfg = self.cfg.get("curriculum", {})
        warmup_end = curriculum_cfg.get("warmup_end", 10)
        semantic_end = curriculum_cfg.get("semantic_end", 40)
        independence_end = curriculum_cfg.get("independence_end", 60)
        if epoch < warmup_end:
            phase = 0
        elif epoch < semantic_end:
            phase = 1
        elif epoch < independence_end:
            phase = 2
        else:
            phase = 3
        self.log("val/curriculum_phase", float(phase))

    def on_train_epoch_start(self) -> None:
        """Update curriculum epoch."""
        self.loss_fn.set_epoch(self.current_epoch)

    def configure_optimizers(self) -> dict:
        """Configure AdamW with cosine annealing + linear warmup."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        warmup_epochs = self.cfg.training.scheduler.warmup_epochs
        max_epochs = self.cfg.training.max_epochs

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=self.cfg.training.scheduler.min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def train_dataloader(self) -> DataLoader:
        """Training dataloader with configurable batch size."""
        assert self._train_dataset is not None, "Call setup_data() first"
        batch_size_cfg = self.cfg.training.get("batch_size", "full")
        if batch_size_cfg == "full" or batch_size_cfg is None:
            batch_size = len(self._train_dataset)
            shuffle = False
        else:
            batch_size = int(batch_size_cfg)
            shuffle = True
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Full-batch validation dataloader."""
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=len(self._val_dataset),
            shuffle=False,
            num_workers=0,
        )

    def save_sdp_checkpoint(self, path: str) -> None:
        """Save SDP checkpoint with model, normalization stats, and config.

        Args:
            path: Output file path.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "h_mean": self.h_mean,
            "h_std": self.h_std,
            "target_stats": {
                key: {
                    "mean": getattr(self, f"target_mean_{key}"),
                    "std": getattr(self, f"target_std_{key}"),
                }
                for key in self._target_keys
            },
            "partition_config": {
                "vol_dim": self.cfg.partition.vol_dim,
                "loc_dim": self.cfg.partition.loc_dim,
                "shape_dim": self.cfg.partition.shape_dim,
                "residual_dim": self.cfg.partition.residual_dim,
            },
            "sdp_config": {
                "in_dim": self.cfg.sdp.in_dim,
                "hidden_dim": self.cfg.sdp.hidden_dim,
                "out_dim": self.cfg.sdp.out_dim,
                "dropout": self.cfg.sdp.dropout,
            },
            "target_config": {
                "n_vol": self.cfg.targets.n_vol,
                "n_loc": self.cfg.targets.n_loc,
                "n_shape": self.cfg.targets.n_shape,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved SDP checkpoint to {path}")
