"""Active Units (AU) callback for canonical latent activity tracking.

Implements canonical AU metric via dataset-level variance computation
on a fixed validation subset with adaptive scheduling.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from math import ceil

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


class ActiveUnitsCallback(Callback):
    """Canonical Active Units callback with adaptive scheduling.

    Computes AU = count(Var(μ_i) > δ) on fixed validation subset.

    Schedule:
        - Dense: every epoch for epochs 0..au_dense_until
        - Sparse: every au_sparse_interval epochs after that

    Args:
        run_dir: Output directory.
        val_dataset: Validation dataset to subsample.
        au_dense_until: Dense phase end epoch (default 15).
        au_sparse_interval: Sparse phase interval (default 5).
        au_subset_fraction: Subset fraction (default 0.25).
        au_batch_size: Batch size for computation (default 64).
        eps_au: Variance threshold in nats (default 0.01).
        au_subset_seed: RNG seed for subset (default 42).
        au_ids_path: Relative path for indices file (default "latent_diag/au_ids.txt").
        image_key: Batch dict key for images (default "image").
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        val_dataset,
        au_dense_until: int = 15,
        au_sparse_interval: int = 5,
        au_subset_fraction: float = 0.25,
        au_batch_size: int = 64,
        eps_au: float = 0.01,
        au_subset_seed: int = 42,
        au_ids_path: str = "latent_diag/au_ids.txt",
        image_key: str = "image",
    ):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.val_dataset = val_dataset
        self.au_dense_until = au_dense_until
        self.au_sparse_interval = au_sparse_interval
        self.au_subset_fraction = au_subset_fraction
        self.au_batch_size = au_batch_size
        self.eps_au = eps_au
        self.au_subset_seed = au_subset_seed
        self.au_ids_path = au_ids_path
        self.image_key = image_key

        self._subset_indices: Optional[List[int]] = None
        self._subset_dataloader: Optional[DataLoader] = None
        self.last_epoch_metrics: Optional[Dict[str, Any]] = None

    def _should_compute(self, epoch: int) -> bool:
        """Check if AU should be computed this epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            True if AU should be computed, False otherwise.
        """
        if epoch <= self.au_dense_until:
            return True  # Dense phase: compute every epoch
        else:
            # Sparse phase: compute every au_sparse_interval epochs
            return (epoch - self.au_dense_until) % self.au_sparse_interval == 0

    def _load_or_initialize_subset(self, trainer: pl.Trainer) -> List[int]:
        """Load existing indices or create new subset.

        Args:
            trainer: PyTorch Lightning trainer.

        Returns:
            List of dataset indices for AU computation.
        """
        ids_path = self.run_dir / self.au_ids_path
        ids_path.parent.mkdir(parents=True, exist_ok=True)

        if ids_path.exists():
            # Reuse existing subset for reproducibility
            with open(ids_path, "r") as f:
                indices = [int(line.strip()) for line in f if line.strip()]
            logger.info(f"Loaded {len(indices)} AU subset indices from {ids_path}")
            return indices

        # Initialize new subset
        n_total = len(self.val_dataset)
        n_subset = ceil(self.au_subset_fraction * n_total)

        rng = np.random.default_rng(self.au_subset_seed)
        indices = rng.choice(n_total, size=n_subset, replace=False).tolist()
        indices.sort()  # Deterministic ordering

        # Save to file (rank 0 only)
        with open(ids_path, "w") as f:
            f.write("\n".join(map(str, indices)))

        logger.info(
            f"Initialized AU subset: {n_subset}/{n_total} samples "
            f"({self.au_subset_fraction:.1%}), seed={self.au_subset_seed}"
        )
        return indices

    def _build_subset_dataloader(self) -> DataLoader:
        """Build DataLoader for subset.

        Returns:
            DataLoader for the AU subset.
        """
        subset_dataset = Subset(self.val_dataset, self._subset_indices)
        return DataLoader(
            subset_dataset,
            batch_size=self.au_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize subset at training start (rank 0 only).

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module.
        """
        if not trainer.is_global_zero:
            return

        self._subset_indices = self._load_or_initialize_subset(trainer)
        self._subset_dataloader = self._build_subset_dataloader()

        logger.info(
            f"AU callback initialized: dense until epoch {self.au_dense_until}, "
            f"then every {self.au_sparse_interval} epochs, "
            f"subset_size={len(self._subset_indices)}, eps_au={self.eps_au}"
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Compute AU at validation end (rank 0 only).

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module.
        """
        if not trainer.is_global_zero:
            return

        if not self._should_compute(trainer.current_epoch):
            return

        try:
            au_count, au_frac, z_dim = self._compute_au(pl_module)
        except Exception as e:
            logger.error(
                f"AU computation failed at epoch {trainer.current_epoch}: {e}",
                exc_info=True
            )
            return

        # Log to Lightning logger
        pl_module.log_dict(
            {
                "diag/au_count": au_count,
                "diag/au_frac": au_frac,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=True,
        )

        # Export for TidyEpochCSVCallback merge
        self.last_epoch_metrics = {
            "epoch": trainer.current_epoch,
            "diag/au_count": au_count,
            "diag/au_frac": au_frac,
            "diag/au_threshold": self.eps_au,
            "diag/au_subset_size": len(self._subset_indices),
        }

        logger.info(
            f"AU at epoch {trainer.current_epoch}: "
            f"{au_count}/{z_dim} active ({au_frac:.3f})"
        )

    def _compute_au(self, pl_module: pl.LightningModule) -> tuple:
        """Compute AU on subset.

        Args:
            pl_module: Lightning module.

        Returns:
            Tuple of (au_count, au_frac, z_dim).
        """
        pl_module.eval()
        device = pl_module.device

        mu_list = []

        with torch.no_grad():
            for batch in self._subset_dataloader:
                x = batch[self.image_key].to(device)
                mu, _ = pl_module.model.encode(x)
                mu_cpu = mu.detach().cpu().float()  # FP32 on CPU
                mu_list.append(mu_cpu)

        mu_mat = torch.cat(mu_list, dim=0)  # [N, z_dim]
        var_per_dim = mu_mat.var(dim=0, unbiased=False)  # [z_dim]

        active_mask = var_per_dim > self.eps_au
        au_count = active_mask.sum().item()
        z_dim = mu_mat.shape[1]
        au_frac = au_count / z_dim

        return float(au_count), float(au_frac), int(z_dim)
