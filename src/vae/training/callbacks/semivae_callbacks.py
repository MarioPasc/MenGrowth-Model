"""SemiVAE-specific callbacks for interpretability and analysis.

This module provides callbacks tailored for Semi-Supervised VAE training:
- Partition-level statistics and disentanglement metrics
- Semantic prediction quality tracking (R², correlation)
- Cross-partition correlation analysis
- Latent space visualization for interpretability

These metrics are critical for:
1. Verifying semantic supervision is working
2. Ensuring residual dimensions remain decorrelated
3. Monitoring partition activity for Neural ODE downstream use
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import csv
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


class SemiVAEDiagnosticsCallback(Callback):
    """Comprehensive diagnostics callback for Semi-Supervised VAE.

    Logs partition-level metrics for interpretability analysis:
    - Per-partition variance and activity
    - Semantic prediction R² and Pearson correlation
    - Cross-partition correlation matrix
    - Partition-wise active units
    - Semantic feature prediction accuracy

    Outputs:
    - {run_dir}/diagnostics/semivae/partition_stats.csv
    - {run_dir}/diagnostics/semivae/semantic_quality.csv
    - {run_dir}/diagnostics/semivae/cross_correlation.csv
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        val_dataset: Any,
        every_n_epochs: int = 5,
        num_samples: int = 256,
        batch_size: int = 32,
        eps_au: float = 0.01,
        seed: int = 42,
    ):
        """Initialize SemiVAEDiagnosticsCallback.

        Args:
            run_dir: Run directory for output files
            val_dataset: Validation dataset for computing metrics
            every_n_epochs: Frequency of diagnostic computation
            num_samples: Number of samples for diagnostics
            batch_size: Batch size for inference
            eps_au: Variance threshold for active units
            seed: Random seed for sample selection
        """
        super().__init__()
        self.run_dir = Path(run_dir)
        self.val_dataset = val_dataset
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.eps_au = eps_au
        self.seed = seed

        # Create output directory
        self.output_dir = self.run_dir / "diagnostics" / "semivae"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output file paths
        self.partition_stats_path = self.output_dir / "partition_stats.csv"
        self.semantic_quality_path = self.output_dir / "semantic_quality.csv"
        self.cross_correlation_path = self.output_dir / "cross_correlation.csv"

        # Initialize CSV files
        self._init_csv_files()

        # Sample indices
        self._sample_indices = None

    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Partition stats
        with open(self.partition_stats_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "partition", "dim", "mu_mean", "mu_std", "mu_var_mean",
                "logvar_mean", "au_count", "au_frac", "kl_mean"
            ])

        # Semantic quality
        with open(self.semantic_quality_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "partition", "mse", "rmse", "r2", "pearson_corr",
                "target_mean", "target_std", "pred_mean", "pred_std"
            ])

        # Cross-correlation (matrix format)
        with open(self.cross_correlation_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "partition_i", "partition_j", "correlation", "abs_correlation"
            ])

    def _get_sample_indices(self) -> List[int]:
        """Get deterministic sample indices."""
        if self._sample_indices is None:
            n_total = len(self.val_dataset)
            n_samples = min(self.num_samples, n_total)

            rng = np.random.RandomState(self.seed)
            self._sample_indices = rng.choice(n_total, size=n_samples, replace=False).tolist()

        return self._sample_indices

    def _should_run(self, trainer: Trainer) -> bool:
        """Check if diagnostics should run this epoch."""
        return (trainer.current_epoch + 1) % self.every_n_epochs == 0

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log SemiVAE diagnostics at end of validation."""
        if not self._should_run(trainer):
            return

        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch
        device = pl_module.device

        # Check if this is a SemiVAE model
        if not hasattr(pl_module.model, "get_partition_info"):
            return

        logger.info(f"Computing SemiVAE diagnostics for epoch {epoch}...")

        # Create subset dataloader
        sample_indices = self._get_sample_indices()
        subset = Subset(self.val_dataset, sample_indices)

        # Use safe_collate from datasets
        from vae.data.datasets import safe_collate
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=safe_collate,
        )

        # Collect latents and predictions
        all_mu = []
        all_logvar = []
        all_semantic_preds = {k: [] for k in pl_module.model.semantic_heads.keys()}
        all_semantic_targets = {k: [] for k in pl_module.model.semantic_heads.keys()}

        pl_module.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device)
                mu, logvar = pl_module.model.encode(x)

                all_mu.append(mu.cpu())
                all_logvar.append(logvar.cpu())

                # Get semantic predictions
                semantic_preds = pl_module.model.predict_semantic_features(mu)
                for k, v in semantic_preds.items():
                    all_semantic_preds[k].append(v.cpu())

                # Get semantic targets if available
                if "semantic_features" in batch:
                    for k in all_semantic_targets.keys():
                        if k in batch["semantic_features"]:
                            all_semantic_targets[k].append(batch["semantic_features"][k].cpu())

        # Concatenate
        mu = torch.cat(all_mu, dim=0)  # [N, z_dim]
        logvar = torch.cat(all_logvar, dim=0)  # [N, z_dim]

        semantic_preds = {k: torch.cat(v, dim=0) for k, v in all_semantic_preds.items() if v}
        semantic_targets = {k: torch.cat(v, dim=0) for k, v in all_semantic_targets.items() if v}

        # Compute partition statistics
        partition_info = pl_module.model.get_partition_info()
        self._log_partition_stats(epoch, mu, logvar, partition_info, trainer)

        # Compute semantic quality metrics
        if semantic_targets:
            self._log_semantic_quality(epoch, semantic_preds, semantic_targets, trainer)

        # Compute cross-partition correlations
        self._log_cross_correlations(epoch, mu, partition_info, trainer)

        logger.info(f"SemiVAE diagnostics complete for epoch {epoch}")

    def _log_partition_stats(
        self,
        epoch: int,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        partition_info: Dict[str, Dict],
        trainer: Trainer,
    ) -> None:
        """Log per-partition statistics."""
        rows = []

        for name, config in partition_info.items():
            start_idx = config["start_idx"]
            end_idx = config["end_idx"]
            dim = config["dim"]

            # Extract partition
            mu_part = mu[:, start_idx:end_idx]
            logvar_part = logvar[:, start_idx:end_idx]

            # Statistics
            mu_mean = mu_part.mean().item()
            mu_std = mu_part.std().item()
            mu_var_mean = mu_part.var(dim=0).mean().item()
            logvar_mean = logvar_part.mean().item()

            # Active units
            mu_var_per_dim = mu_part.var(dim=0)
            au_count = (mu_var_per_dim > self.eps_au).sum().item()
            au_frac = au_count / dim

            # KL per partition (mean over samples and dimensions)
            kl_per_dim = 0.5 * (torch.exp(logvar_part) + mu_part ** 2 - 1 - logvar_part)
            kl_mean = kl_per_dim.mean().item()

            # Log to trainer
            trainer.logger.log_metrics({
                f"semivae/{name}_mu_std": mu_std,
                f"semivae/{name}_au_frac": au_frac,
                f"semivae/{name}_kl": kl_mean,
            }, step=epoch)

            rows.append([
                epoch, name, dim, mu_mean, mu_std, mu_var_mean,
                logvar_mean, au_count, au_frac, kl_mean
            ])

        # Write to CSV
        with open(self.partition_stats_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _log_semantic_quality(
        self,
        epoch: int,
        semantic_preds: Dict[str, torch.Tensor],
        semantic_targets: Dict[str, torch.Tensor],
        trainer: Trainer,
    ) -> None:
        """Log semantic prediction quality metrics."""
        rows = []

        for name in semantic_preds.keys():
            if name not in semantic_targets:
                continue

            pred = semantic_preds[name]  # [N, F]
            target = semantic_targets[name]  # [N, F]

            # Flatten for overall metrics
            pred_flat = pred.flatten()
            target_flat = target.flatten()

            # MSE and RMSE
            mse = F.mse_loss(pred, target).item()
            rmse = np.sqrt(mse)

            # R² score
            ss_res = ((target_flat - pred_flat) ** 2).sum()
            ss_tot = ((target_flat - target_flat.mean()) ** 2).sum()
            r2 = 1 - (ss_res / (ss_tot + 1e-8)).item()

            # Pearson correlation
            pred_centered = pred_flat - pred_flat.mean()
            target_centered = target_flat - target_flat.mean()
            numerator = (pred_centered * target_centered).sum()
            denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum() + 1e-8)
            pearson = (numerator / denominator).item()

            # Statistics
            target_mean = target_flat.mean().item()
            target_std = target_flat.std().item()
            pred_mean = pred_flat.mean().item()
            pred_std = pred_flat.std().item()

            # Log to trainer
            trainer.logger.log_metrics({
                f"sem/{name}_r2": r2,
                f"sem/{name}_pearson": pearson,
                f"sem/{name}_rmse": rmse,
            }, step=epoch)

            rows.append([
                epoch, name, mse, rmse, r2, pearson,
                target_mean, target_std, pred_mean, pred_std
            ])

        # Write to CSV
        with open(self.semantic_quality_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _log_cross_correlations(
        self,
        epoch: int,
        mu: torch.Tensor,
        partition_info: Dict[str, Dict],
        trainer: Trainer,
    ) -> None:
        """Log cross-partition correlation matrix.

        For interpretability, we want to verify:
        - Supervised partitions (z_vol, z_loc, z_shape) have low correlation with each other
        - Residual (z_residual) is decorrelated from supervised partitions
        """
        rows = []
        partition_names = list(partition_info.keys())

        # Compute mean latent per partition
        partition_means = {}
        for name, config in partition_info.items():
            mu_part = mu[:, config["start_idx"]:config["end_idx"]]
            partition_means[name] = mu_part.mean(dim=1)  # [N]

        # Compute pairwise correlations
        for i, name_i in enumerate(partition_names):
            for j, name_j in enumerate(partition_names):
                if i > j:  # Only upper triangle
                    continue

                vec_i = partition_means[name_i]
                vec_j = partition_means[name_j]

                # Pearson correlation
                vec_i_centered = vec_i - vec_i.mean()
                vec_j_centered = vec_j - vec_j.mean()
                numerator = (vec_i_centered * vec_j_centered).sum()
                denominator = torch.sqrt(
                    (vec_i_centered ** 2).sum() * (vec_j_centered ** 2).sum() + 1e-8
                )
                corr = (numerator / denominator).item()

                rows.append([epoch, name_i, name_j, corr, abs(corr)])

                # Log to trainer (only off-diagonal)
                if i != j:
                    trainer.logger.log_metrics({
                        f"semivae/corr_{name_i}_{name_j}": corr,
                    }, step=epoch)

        # Write to CSV
        with open(self.cross_correlation_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


class SemiVAELatentVisualizationCallback(Callback):
    """Callback for latent space visualization of SemiVAE partitions.

    Generates:
    - t-SNE/UMAP plots of each partition colored by semantic features
    - Latent traversal visualizations for supervised dimensions
    - Partition activity heatmaps
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        val_dataset: Any,
        every_n_epochs: int = 20,
        num_samples: int = 512,
        batch_size: int = 32,
        seed: int = 42,
    ):
        """Initialize visualization callback.

        Args:
            run_dir: Run directory for output files
            val_dataset: Validation dataset
            every_n_epochs: Frequency of visualization
            num_samples: Number of samples for visualization
            batch_size: Batch size for inference
            seed: Random seed
        """
        super().__init__()
        self.run_dir = Path(run_dir)
        self.val_dataset = val_dataset
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed

        # Create output directory
        self.output_dir = self.run_dir / "visualizations" / "latent"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._sample_indices = None

    def _should_run(self, trainer: Trainer) -> bool:
        """Check if visualization should run this epoch."""
        return (trainer.current_epoch + 1) % self.every_n_epochs == 0

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Generate latent visualizations."""
        if not self._should_run(trainer):
            return

        if not trainer.is_global_zero:
            return

        # Check for required libraries
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            logger.warning("matplotlib or sklearn not available, skipping visualization")
            return

        epoch = trainer.current_epoch
        device = pl_module.device

        if not hasattr(pl_module.model, "get_partition_info"):
            return

        logger.info(f"Generating SemiVAE latent visualizations for epoch {epoch}...")

        # Collect latents
        sample_indices = self._get_sample_indices()
        subset = Subset(self.val_dataset, sample_indices)

        from vae.data.datasets import safe_collate
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=safe_collate,
        )

        all_mu = []
        all_vol_targets = []

        pl_module.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device)
                mu, _ = pl_module.model.encode(x)
                all_mu.append(mu.cpu())

                # Get volume target for coloring
                if "semantic_features" in batch and "z_vol" in batch["semantic_features"]:
                    all_vol_targets.append(batch["semantic_features"]["z_vol"][:, 0].cpu())

        mu = torch.cat(all_mu, dim=0).numpy()
        vol_targets = torch.cat(all_vol_targets, dim=0).numpy() if all_vol_targets else None

        # Generate partition activity heatmap
        self._plot_partition_activity(epoch, mu, pl_module, plt)

        # Generate PCA visualizations for each partition
        partition_info = pl_module.model.get_partition_info()
        self._plot_partition_pca(epoch, mu, vol_targets, partition_info, plt, PCA)

        plt.close("all")
        logger.info(f"Latent visualizations saved to {self.output_dir}")

    def _get_sample_indices(self) -> List[int]:
        """Get deterministic sample indices."""
        if self._sample_indices is None:
            n_total = len(self.val_dataset)
            n_samples = min(self.num_samples, n_total)

            rng = np.random.RandomState(self.seed)
            self._sample_indices = rng.choice(n_total, size=n_samples, replace=False).tolist()

        return self._sample_indices

    def _plot_partition_activity(
        self,
        epoch: int,
        mu: np.ndarray,
        pl_module: LightningModule,
        plt,
    ) -> None:
        """Plot partition activity heatmap."""
        partition_info = pl_module.model.get_partition_info()

        # Compute variance per dimension
        var_per_dim = np.var(mu, axis=0)

        fig, ax = plt.subplots(figsize=(12, 3))

        # Create colored bars for each partition
        colors = {"z_vol": "red", "z_loc": "green", "z_shape": "blue", "z_residual": "gray"}
        x = np.arange(len(var_per_dim))

        for name, config in partition_info.items():
            start = config["start_idx"]
            end = config["end_idx"]
            color = colors.get(name, "gray")
            ax.bar(x[start:end], var_per_dim[start:end], color=color, alpha=0.7, label=name)

        ax.axhline(y=0.01, color="black", linestyle="--", label="AU threshold")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Variance")
        ax.set_title(f"Partition Activity (Epoch {epoch})")
        ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(self.output_dir / f"partition_activity_epoch{epoch:04d}.png", dpi=150)
        plt.close(fig)

    def _plot_partition_pca(
        self,
        epoch: int,
        mu: np.ndarray,
        vol_targets: Optional[np.ndarray],
        partition_info: Dict[str, Dict],
        plt,
        PCA,
    ) -> None:
        """Plot PCA of each partition."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        partition_names = ["z_vol", "z_loc", "z_shape", "z_residual"]

        for i, name in enumerate(partition_names):
            if name not in partition_info:
                continue

            ax = axes[i]
            config = partition_info[name]
            mu_part = mu[:, config["start_idx"]:config["end_idx"]]

            if mu_part.shape[1] < 2:
                ax.set_title(f"{name} (dim < 2, skipped)")
                continue

            # Check for NaN/Inf values before PCA
            if not np.isfinite(mu_part).all():
                ax.set_title(f"{name} (contains NaN/Inf, skipped)")
                logger.warning(f"Partition {name} contains NaN/Inf values, skipping PCA")
                continue

            # PCA to 2D
            pca = PCA(n_components=2)
            mu_2d = pca.fit_transform(mu_part)

            # Color by volume if available
            if vol_targets is not None:
                scatter = ax.scatter(mu_2d[:, 0], mu_2d[:, 1], c=vol_targets, cmap="viridis", alpha=0.6, s=10)
                plt.colorbar(scatter, ax=ax, label="Volume")
            else:
                ax.scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.6, s=10)

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax.set_title(f"{name} ({config['dim']} dims)")

        fig.suptitle(f"Partition PCA (Epoch {epoch})")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"partition_pca_epoch{epoch:04d}.png", dpi=150)
        plt.close(fig)


class SemiVAESemanticTrackingCallback(Callback):
    """Lightweight callback for tracking semantic supervision progress.

    Logs at every epoch (low overhead):
    - Semantic loss components
    - Prediction/target correlation
    - Phase information (warmup vs active)
    """

    def __init__(self, run_dir: Union[str, Path]):
        """Initialize callback.

        Args:
            run_dir: Run directory for output
        """
        super().__init__()
        self.run_dir = Path(run_dir)
        self.output_dir = self.run_dir / "diagnostics" / "semivae"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tracking_path = self.output_dir / "semantic_tracking.csv"
        self._init_csv()

    def _init_csv(self):
        """Initialize tracking CSV."""
        with open(self.tracking_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "phase", "lambda_vol", "lambda_loc", "lambda_shape",
                "loss_vol", "loss_loc", "loss_shape", "loss_semantic_total"
            ])

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log semantic tracking at end of each epoch."""
        if not trainer.is_global_zero:
            return

        if not hasattr(pl_module, "current_lambda_vol"):
            return

        epoch = trainer.current_epoch

        # Determine phase
        if epoch < pl_module.semantic_start_epoch:
            phase = "warmup"
        elif epoch < pl_module.semantic_start_epoch + pl_module.semantic_annealing_epochs:
            phase = "annealing"
        else:
            phase = "active"

        # Get logged metrics
        metrics = trainer.callback_metrics

        loss_vol = metrics.get("train_epoch/z_vol_mse", 0)
        loss_loc = metrics.get("train_epoch/z_loc_mse", 0)
        loss_shape = metrics.get("train_epoch/z_shape_mse", 0)
        loss_total = metrics.get("train_epoch/semantic_total", 0)

        # Convert tensors to floats
        if isinstance(loss_vol, torch.Tensor):
            loss_vol = loss_vol.item()
        if isinstance(loss_loc, torch.Tensor):
            loss_loc = loss_loc.item()
        if isinstance(loss_shape, torch.Tensor):
            loss_shape = loss_shape.item()
        if isinstance(loss_total, torch.Tensor):
            loss_total = loss_total.item()

        # Write row
        with open(self.tracking_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, phase,
                pl_module.current_lambda_vol,
                pl_module.current_lambda_loc,
                pl_module.current_lambda_shape,
                loss_vol, loss_loc, loss_shape, loss_total
            ])
