"""Latent space diagnostic callbacks for VAE training.

This module provides callbacks for latent space analysis:
- LatentDiagnosticsCallback: Correlation, shift sensitivity, DIP-VAE covariance, regression probes
- ActiveUnitsCallback: Canonical Active Units metric with adaptive scheduling

All diagnostics use latent_diag/* namespace and export to UnifiedCSVCallback.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from math import ceil

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, Subset

from vae.data.datasets import safe_collate
from vae.metrics import (
    compute_correlation,
    compute_dipvae_covariance,
    compute_shift_sensitivity,
    extract_segmentation_targets,
    ridge_probe_cv,
    compute_active_units,
)

logger = logging.getLogger(__name__)


class LatentDiagnosticsCallback(Callback):
    """Callback to compute latent space diagnostics on a fixed validation subset.

    This callback implements lightweight disentanglement diagnostics computed every N epochs
    on a deterministically selected fixed subset of validation samples. All statistics are
    computed in float32 on CPU to ensure numerical stability.

    Diagnostics:
    - Correlation: Mean absolute off-diagonal correlation of latent means
      (proxy for independence - high correlation indicates entangled factors)
    - Translation sensitivity: L2 change in μ under spatial shift
      (ODE-readiness check - low sensitivity indicates position-invariant encoding, as desired for SBD)
    - Segmentation probes: Ridge regression R² from latents to tumor characteristics
      (semantic alignment check - high R² indicates latent subspaces encode interpretable features)
    - DIP-VAE covariance: Covariance matching metrics (Cov_q vs identity)

    Metrics are logged to Lightning logger for real-time monitoring and exported
    to UnifiedCSVCallback for consolidated CSV logging.

    Hard constraints:
    - DDP-safe: All filesystem writes only on rank 0
    - Deterministic: Fixed sample IDs persisted and reused across epochs
    - Lightweight: Computed only every N epochs on small subset (~32 samples)
    - Numerically stable: FP32 on CPU for all statistics

    Args:
        run_dir: Root directory for saving outputs.
        seg_labels: Segmentation label mapping dict (REQUIRED, no default). Example for BraTS: {"ncr": 1, "ed": 2, "et": 3}.
        every_n_epochs: Compute diagnostics every N epochs (default: 10).
        num_samples: Number of validation samples to use (default: 32).
        shift_vox: Translation shift magnitude in voxels (default: 5).
        eps_au: Variance threshold (nats) - kept for backward compatibility but no longer used.
        ids_name: Relative path for sample IDs file (default: "latent_diag/ids.txt").
        image_key: Batch dict key for images (default: "image").
        seg_key: Batch dict key for segmentations (default: "seg").
        id_key: Batch dict key for sample IDs (default: "id").

    Note:
        Active Units (AU) metric is now computed by the dedicated ActiveUnitsCallback,
        which provides more flexible scheduling (dense-early/sparse-late) and uses a
        configurable subset size. This callback focuses on correlation, shift sensitivity,
        and segmentation probe diagnostics.
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        seg_labels: Dict[str, int],
        spacing: tuple,
        every_n_epochs: int = 10,
        num_samples: int = 32,
        shift_vox: int = 5,
        eps_au: float = 0.01,
        ids_name: str = "latent_diag/ids.txt",
        image_key: str = "image",
        seg_key: str = "seg",
        id_key: str = "id",
    ):
        """Initialize LatentDiagnosticsCallback.

        Args:
            run_dir: Root directory for saving outputs.
            seg_labels: Segmentation label mapping (REQUIRED). Example for BraTS: {"ncr": 1, "ed": 2, "et": 3}.
            spacing: Voxel spacing in mm (D, H, W).
            every_n_epochs: Compute diagnostics every N epochs.
            num_samples: Number of validation samples to use.
            shift_vox: Translation shift magnitude in voxels.
            eps_au: Variance threshold for active units (nats) - kept for backward compatibility.
            ids_name: Relative path for sample IDs file.
            image_key: Batch dict key for images.
            seg_key: Batch dict key for segmentations.
            id_key: Batch dict key for sample IDs.
        """
        super().__init__()

        # Validate seg_labels
        if seg_labels is None:
            raise ValueError(
                "seg_labels must be explicitly provided in config. "
                "Example for BraTS: {'ncr': 1, 'ed': 2, 'et': 3}"
            )
        if not isinstance(seg_labels, dict) or len(seg_labels) == 0:
            raise ValueError(
                f"seg_labels must be a non-empty dict, got {type(seg_labels)}"
            )

        self.run_dir = Path(run_dir)
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.shift_vox = shift_vox
        self.eps_au = eps_au
        self.ids_name = ids_name
        self.image_key = image_key
        self.seg_key = seg_key
        self.id_key = id_key
        self.seg_labels = seg_labels
        self.spacing = spacing

        # Runtime state
        self._sample_ids: Optional[List[str]] = None
        self._val_data: List[Dict[str, Any]] = []
        self.last_epoch_metrics: Optional[Dict[str, Any]] = None

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Reset accumulator and initialize sample IDs if needed."""
        # Only run on diagnostic epochs
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Load or initialize sample IDs (rank 0 only - file I/O is inside this guard)
        if self._sample_ids is None and trainer.is_global_zero:
            self._sample_ids = self._load_sample_ids()

            # If loaded successfully, verify and log
            if self._sample_ids is not None:
                ids_path = self.run_dir / self.ids_name
                logger.info(
                    f"Latent diagnostics: loaded {len(self._sample_ids)} sample IDs from {ids_path} "
                    f"(epoch {trainer.current_epoch}: reusing same subset for consistency)"
                )

            # If still None, initialize from validation dataloader
            if self._sample_ids is None:
                val_dataloader = (
                    trainer.val_dataloaders[0]
                    if isinstance(trainer.val_dataloaders, list)
                    else trainer.val_dataloaders
                )
                self._sample_ids = self._initialize_sample_ids(val_dataloader)

                # Write to disk
                ids_path = self.run_dir / self.ids_name
                ids_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ids_path, "w") as f:
                    f.write("\n".join(self._sample_ids))

                logger.info(
                    f"Latent diagnostics: initialized {len(self._sample_ids)} sample IDs "
                    f"(deterministic selection from first {len(self._sample_ids)} validation samples)"
                )
                logger.info(f"Sample IDs saved to: {ids_path}")

        # Broadcast to all ranks if DDP
        if trainer.world_size > 1 and self._sample_ids is not None:
            self._sample_ids = self._broadcast_sample_ids(self._sample_ids, trainer)

        # Reset accumulator
        self._val_data = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate μ and metadata for selected samples."""
        # Only run on diagnostic epochs
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Skip if IDs not initialized yet
        if self._sample_ids is None:
            return

        # Filter batch for selected samples
        batch_ids = batch[self.id_key]
        selected_set = set(self._sample_ids)

        # Extract μ and logvar for selected samples
        with torch.no_grad():
            pl_module.model.eval()
            mu, logvar = pl_module.model.encode(batch[self.image_key])
            mu_cpu = mu.detach().cpu().float()  # Force FP32 on CPU
            logvar_cpu = logvar.detach().cpu().float()  # Force FP32 on CPU

        # Accumulate data
        for i, bid in enumerate(batch_ids):
            if bid in selected_set:
                self._val_data.append(
                    {
                        "id": bid,
                        "mu": mu_cpu[i],  # [z_dim]
                        "logvar": logvar_cpu[i],  # [z_dim]
                        "x": batch[self.image_key][i].cpu(),  # [4, 128, 128, 128]
                        "seg": (
                            batch[self.seg_key][i].cpu()
                            if self.seg_key in batch
                            else None
                        ),  # [1, 128, 128, 128] or None
                    }
                )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Compute and log diagnostics on rank 0."""
        # Only run on diagnostic epochs
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Only process on rank 0
        if not trainer.is_global_zero:
            return

        # Check if we have data
        if len(self._val_data) == 0:
            logger.warning(
                f"No samples accumulated for diagnostics at epoch {trainer.current_epoch}"
            )
            return

        # Deduplicate by ID (in case of multi-GPU issues)
        data_dict = {d["id"]: d for d in self._val_data}
        unique_data = list(data_dict.values())

        if len(unique_data) < 2:
            logger.warning(
                f"Only {len(unique_data)} samples available, skipping diagnostics (need at least 2)"
            )
            return

        # Validate segmentation labels on epoch 0
        if trainer.current_epoch == 0:
            all_labels_seen = set()
            for d in unique_data:
                if d["seg"] is not None:
                    seg_labels_in_sample = torch.unique(d["seg"]).cpu().tolist()
                    all_labels_seen.update(seg_labels_in_sample)

            expected_labels = set(self.seg_labels.values())
            if not expected_labels.issubset(all_labels_seen):
                missing = expected_labels - all_labels_seen
                logger.warning(
                    f"Expected seg labels {expected_labels} but only found {all_labels_seen}. "
                    f"Missing: {missing}. This may indicate empty tumor compartments in the evaluation subset."
                )
            else:
                logger.info(
                    f"Segmentation labels validated: found {sorted(all_labels_seen)} "
                    f"(expected {sorted(expected_labels)})"
                )

        # Compute diagnostics
        try:
            metrics = self._compute_diagnostics(unique_data, pl_module, trainer)
        except Exception as e:
            logger.error(f"Failed to compute diagnostics: {e}", exc_info=True)
            return

        # Log to Lightning logger
        self._log_metrics(metrics, pl_module)

        # Export metrics for UnifiedCSVCallback to merge
        # Use latent_diag/* namespace (per user requirement)
        self.last_epoch_metrics = {
            "epoch": metrics["epoch"],
            "latent_diag/corr_offdiag_meanabs": metrics["corr_offdiag_meanabs"],
            "latent_diag/shift_mu_l2_mean": metrics.get("shift_mu_l2_mean", np.nan),
            "latent_diag/shift_mu_absmean": metrics.get("shift_mu_absmean", np.nan),
            # DIP-VAE-II covariance metrics
            "latent_diag/cov_q_offdiag_meanabs": metrics.get("cov_q_offdiag_meanabs", np.nan),
            "latent_diag/cov_q_offdiag_fro": metrics.get("cov_q_offdiag_fro", np.nan),
            "latent_diag/cov_q_diag_meanabs_error": metrics.get("cov_q_diag_meanabs_error", np.nan),
            "latent_diag/cov_q_diag_mean": metrics.get("cov_q_diag_mean", np.nan),
            # Ridge probe R² (cross-validated, mean ± std)
            "latent_diag/r2_logV_ncr_mean": metrics.get("r2_logV_ncr_mean", np.nan),
            "latent_diag/r2_logV_ncr_std": metrics.get("r2_logV_ncr_std", np.nan),
            "latent_diag/r2_logV_ed_mean": metrics.get("r2_logV_ed_mean", np.nan),
            "latent_diag/r2_logV_ed_std": metrics.get("r2_logV_ed_std", np.nan),
            "latent_diag/r2_logV_et_mean": metrics.get("r2_logV_et_mean", np.nan),
            "latent_diag/r2_logV_et_std": metrics.get("r2_logV_et_std", np.nan),
            "latent_diag/r2_logV_total_mean": metrics.get("r2_logV_total_mean", np.nan),
            "latent_diag/r2_logV_total_std": metrics.get("r2_logV_total_std", np.nan),
            "latent_diag/r2_cz_total_mean": metrics.get("r2_cz_total_mean", np.nan),
            "latent_diag/r2_cz_total_std": metrics.get("r2_cz_total_std", np.nan),
            "latent_diag/r2_cy_total_mean": metrics.get("r2_cy_total_mean", np.nan),
            "latent_diag/r2_cy_total_std": metrics.get("r2_cy_total_std", np.nan),
            "latent_diag/r2_cx_total_mean": metrics.get("r2_cx_total_mean", np.nan),
            "latent_diag/r2_cx_total_std": metrics.get("r2_cx_total_std", np.nan),
            "latent_diag/r2_r_ncr_mean": metrics.get("r2_r_ncr_mean", np.nan),
            "latent_diag/r2_r_ncr_std": metrics.get("r2_r_ncr_std", np.nan),
            "latent_diag/r2_r_ed_mean": metrics.get("r2_r_ed_mean", np.nan),
            "latent_diag/r2_r_ed_std": metrics.get("r2_r_ed_std", np.nan),
            "latent_diag/r2_r_et_mean": metrics.get("r2_r_et_mean", np.nan),
            "latent_diag/r2_r_et_std": metrics.get("r2_r_et_std", np.nan),
        }

        # Clear accumulator
        self._val_data = []

    def _load_sample_ids(self) -> Optional[List[str]]:
        """Load previously persisted sample IDs.

        Returns:
            List of sample IDs if file exists, None otherwise.
        """
        ids_path = self.run_dir / self.ids_name
        if not ids_path.exists():
            return None

        with open(ids_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _initialize_sample_ids(self, dataloader) -> List[str]:
        """Select first N samples from dataloader deterministically.

        Args:
            dataloader: Validation dataloader.

        Returns:
            List of sample IDs (sorted for determinism).
        """
        ids = []
        for batch in dataloader:
            ids.extend(batch[self.id_key])
            if len(ids) >= self.num_samples:
                break

        # Take exactly num_samples, sort for determinism
        selected_ids = sorted(ids[: self.num_samples])
        return selected_ids

    def _broadcast_sample_ids(
        self, ids: Optional[List[str]], trainer: pl.Trainer
    ) -> List[str]:
        """Broadcast sample IDs from rank 0 to all ranks in DDP.

        Args:
            ids: Sample IDs on rank 0, None on other ranks.
            trainer: Lightning trainer.

        Returns:
            Sample IDs on all ranks.
        """
        if trainer.world_size == 1:
            return ids

        import torch.distributed as dist

        # Rank 0 prepares data
        if trainer.is_global_zero:
            id_str = "\n".join(ids)
            id_bytes = id_str.encode("utf-8")
            length = len(id_bytes)
        else:
            length = 0

        # Broadcast length
        length_tensor = torch.tensor(
            length, dtype=torch.long, device=trainer.strategy.root_device
        )
        dist.broadcast(length_tensor, src=0)
        length = length_tensor.item()

        # Broadcast string
        if trainer.is_global_zero:
            data_tensor = torch.frombuffer(id_bytes, dtype=torch.uint8).to(
                trainer.strategy.root_device
            )
        else:
            data_tensor = torch.zeros(
                length, dtype=torch.uint8, device=trainer.strategy.root_device
            )

        dist.broadcast(data_tensor, src=0)

        # Decode on all ranks
        id_str = data_tensor.cpu().numpy().tobytes().decode("utf-8")
        return id_str.split("\n")

    def _compute_diagnostics(
        self,
        data_list: List[Dict[str, Any]],
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
    ) -> Dict[str, Any]:
        """Run all diagnostic computations.

        Args:
            data_list: List of {id, mu, x, seg} dictionaries.
            pl_module: Lightning module with model.
            trainer: Lightning trainer.

        Returns:
            Dictionary with all metrics for logging.
        """
        # Build mu and logvar matrices
        mu_matrix, z_dim = self._build_mu_matrix(data_list)
        logvar_matrix = self._build_logvar_matrix(data_list)
        N = mu_matrix.shape[0]

        metrics = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "num_samples_planned": self.num_samples,
            "num_samples_used": N,
            "z_dim": z_dim,
        }

        # Correlation (existing metric, kept for comparison)
        corr_metrics = compute_correlation(mu_matrix, return_matrix=False)
        metrics.update(corr_metrics)

        # DIP-VAE-II covariance metrics (matches training loss computation)
        dipvae_cov_metrics = compute_dipvae_covariance(mu_matrix, logvar_matrix, compute_in_fp32=True)
        # Remove cov_q_matrix from metrics (too large for logging)
        dipvae_cov_metrics.pop("cov_q_matrix", None)
        metrics.update(dipvae_cov_metrics)

        # Shift sensitivity
        try:
            shift_metrics = self._compute_shift_sensitivity(
                data_list, pl_module, self.shift_vox
            )
            metrics.update(shift_metrics)
        except Exception as e:
            logger.error(f"Shift sensitivity computation failed: {e}")
            metrics.update(
                {
                    "shift_vox": self.shift_vox,
                    "shift_mu_l2_mean": np.nan,
                    "shift_mu_absmean": np.nan,
                }
            )

        # Segmentation probes
        targets_df, seg_available = self._extract_seg_targets_from_list(data_list)
        metrics["seg_available"] = seg_available

        if seg_available:
            probe_results = ridge_probe_cv(
                z=mu_matrix.cpu().numpy(),
                targets_df=targets_df,
                n_folds=5,
                alpha=1.0,
                random_state=42
            )
            # Flatten r2 results (remove _mean and _std suffixes for backward compat in some cases)
            for target_col in ["logV_ncr", "logV_ed", "logV_et", "logV_total",
                               "cz_total", "cy_total", "cx_total", "r_ncr", "r_ed", "r_et"]:
                # Keep both _mean and _std for complete info
                if f"r2_{target_col}_mean" in probe_results:
                    metrics[f"r2_{target_col}"] = probe_results[f"r2_{target_col}_mean"]
            metrics.update(probe_results)
        else:
            # Fill with defaults
            for target in ["ncr", "ed", "et", "total"]:
                metrics[f"n_empty_{target}"] = 0

            for target_col in [
                "logV_ncr",
                "logV_ed",
                "logV_et",
                "logV_total",
                "cz_total",
                "cy_total",
                "cx_total",
                "r_ncr",
                "r_ed",
                "r_et",
            ]:
                metrics[f"r2_{target_col}"] = np.nan
                if target_col in ["logV_total", "cx_total", "cy_total", "cz_total"]:
                    metrics[f"top5dims_{target_col}"] = ""

        return metrics

    def _build_mu_matrix(
        self, data_list: List[Dict[str, Any]]
    ) -> tuple:
        """Stack all mu vectors into matrix.

        Args:
            data_list: List of dictionaries with 'mu' key.

        Returns:
            mu_matrix: [N, z_dim] float32 CPU tensor.
            z_dim: Latent dimensionality.
        """
        mus = [d["mu"] for d in data_list]
        mu_matrix = torch.stack(mus, dim=0)  # [N, z_dim]
        assert mu_matrix.dtype == torch.float32
        return mu_matrix, mu_matrix.shape[1]

    def _build_logvar_matrix(
        self, data_list: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Stack all logvar vectors into matrix.

        Args:
            data_list: List of dictionaries with 'logvar' key.

        Returns:
            logvar_matrix: [N, z_dim] float32 CPU tensor.
        """
        logvars = [d["logvar"] for d in data_list]
        logvar_matrix = torch.stack(logvars, dim=0)  # [N, z_dim]
        assert logvar_matrix.dtype == torch.float32
        return logvar_matrix

    def _compute_shift_sensitivity(
        self,
        data_list: List[Dict[str, Any]],
        pl_module: pl.LightningModule,
        shift_vox: int,
    ) -> Dict[str, float]:
        """Measure change in μ under random spatial shift.

        Args:
            data_list: List of {id, mu, x, seg} dictionaries.
            pl_module: Lightning module with model.
            shift_vox: Maximum shift magnitude per axis.

        Returns:
            Dictionary with shift_vox, shift_mu_l2_mean, shift_mu_absmean.
        """
        delta_norms = []
        delta_absmeans = []

        # Set generator for reproducible random shifts
        rng = torch.Generator().manual_seed(42)

        with torch.no_grad():
            pl_module.model.eval()

            for d in data_list:
                x = d["x"]  # [4, 128, 128, 128]
                mu_orig = d["mu"]  # [z_dim]

                # Random shift
                shift_d = torch.randint(-shift_vox, shift_vox + 1, (1,), generator=rng).item()
                shift_h = torch.randint(-shift_vox, shift_vox + 1, (1,), generator=rng).item()
                shift_w = torch.randint(-shift_vox, shift_vox + 1, (1,), generator=rng).item()

                x_shifted = self._shift_image_no_wrap(x, shift_d, shift_h, shift_w)

                # Re-encode (need to move to device)
                x_shifted_batch = x_shifted.unsqueeze(0).to(pl_module.device)
                mu_shifted, _ = pl_module.model.encode(x_shifted_batch)
                mu_shifted = mu_shifted.squeeze(0).cpu().float()

                # Compute change
                delta = mu_shifted - mu_orig
                delta_norms.append(delta.norm(p=2).item())
                delta_absmeans.append(delta.abs().mean().item())

        return {
            "shift_vox": shift_vox,
            "shift_mu_l2_mean": float(np.mean(delta_norms)),
            "shift_mu_absmean": float(np.mean(delta_absmeans)),
        }

    def _shift_image_no_wrap(
        self, x: torch.Tensor, shift_d: int, shift_h: int, shift_w: int
    ) -> torch.Tensor:
        """Shift image by (shift_d, shift_h, shift_w) voxels without wrap-around.

        Args:
            x: [C, D, H, W] image tensor.
            shift_d: Signed shift in depth dimension.
            shift_h: Signed shift in height dimension.
            shift_w: Signed shift in width dimension.

        Returns:
            x_shifted: [C, D, H, W] with zeros in vacated regions.
        """
        C, D, H, W = x.shape
        x_shifted = torch.zeros_like(x)

        # Source slice ranges
        d_src_start = max(0, -shift_d)
        d_src_end = min(D, D - shift_d)
        h_src_start = max(0, -shift_h)
        h_src_end = min(H, H - shift_h)
        w_src_start = max(0, -shift_w)
        w_src_end = min(W, W - shift_w)

        # Destination slice ranges
        d_dst_start = max(0, shift_d)
        d_dst_end = d_dst_start + (d_src_end - d_src_start)
        h_dst_start = max(0, shift_h)
        h_dst_end = h_dst_start + (h_src_end - h_src_start)
        w_dst_start = max(0, shift_w)
        w_dst_end = w_dst_start + (w_src_end - w_src_start)

        x_shifted[
            :, d_dst_start:d_dst_end, h_dst_start:h_dst_end, w_dst_start:w_dst_end
        ] = x[:, d_src_start:d_src_end, h_src_start:h_src_end, w_src_start:w_src_end]

        return x_shifted

    def _extract_seg_targets_from_list(
        self, data_list: List[Dict[str, Any]]
    ) -> tuple:
        """Extract volume, centroid, and ratio features from segmentations.

        Adapter method that converts data_list format to batch tensor and calls metrics function.

        Args:
            data_list: List of {id, mu, x, seg} dictionaries.

        Returns:
            df: DataFrame with volume/centroid/ratio columns, or None if no seg.
            seg_available: 1 if any seg exists, 0 otherwise.
        """
        # Check if any seg exists
        has_seg = any(d["seg"] is not None for d in data_list)
        if not has_seg:
            return None, 0

        # Collect segmentations into batch (handle None values)
        seg_list = []
        for d in data_list:
            if d["seg"] is not None:
                seg_list.append(d["seg"])  # [1, D, H, W]
            else:
                # Create dummy tensor filled with zeros
                # Assume shape from first non-None seg or default
                if seg_list:
                    dummy_shape = seg_list[0].shape
                else:
                    dummy_shape = (1, 128, 128, 128)
                seg_list.append(torch.zeros(dummy_shape))

        seg_batch = torch.stack(seg_list, dim=0)  # [B, 1, D, H, W]

        # Use metrics function
        df = extract_segmentation_targets(
            seg_batch=seg_batch,
            label_map=self.seg_labels,
            spacing=self.spacing
        )

        return df, 1

    def _log_metrics(
        self, metrics: Dict[str, Any], pl_module: pl.LightningModule
    ) -> None:
        """Log scalar metrics to Lightning loggers.

        Args:
            metrics: Dictionary with all metrics.
            pl_module: Lightning module for logging.
        """
        scalar_keys = [
            "corr_offdiag_meanabs",
            "shift_mu_l2_mean",
            "shift_mu_absmean",
            # DIP-VAE-II covariance metrics (Cov_q = Cov(mu) + E[diag(var)])
            "cov_q_offdiag_meanabs",
            "cov_q_offdiag_fro",
            "cov_q_diag_meanabs_error",
            "cov_q_diag_mean",
            # Ridge probe R² (cross-validated, mean ± std)
            "r2_logV_ncr_mean",
            "r2_logV_ncr_std",
            "r2_logV_ed_mean",
            "r2_logV_ed_std",
            "r2_logV_et_mean",
            "r2_logV_et_std",
            "r2_logV_total_mean",
            "r2_logV_total_std",
            "r2_cz_total_mean",
            "r2_cz_total_std",
            "r2_cy_total_mean",
            "r2_cy_total_std",
            "r2_cx_total_mean",
            "r2_cx_total_std",
            "r2_r_ncr_mean",
            "r2_r_ncr_std",
            "r2_r_ed_mean",
            "r2_r_ed_std",
            "r2_r_et_mean",
            "r2_r_et_std",
        ]

        log_dict = {}
        for k in scalar_keys:
            if k in metrics:
                val = metrics[k]
                # Lightning handles NaN gracefully
                # Use latent_diag/* namespace (changed from diag/*)
                log_dict[f"latent_diag/{k}"] = val

        pl_module.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=True,
        )

        logger.info(
            f"Latent diagnostics at epoch {metrics['epoch']}: "
            f"corr={metrics['corr_offdiag_meanabs']:.4f}"
        )


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
            collate_fn=safe_collate,
        )

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Initialize subset during setup phase (rank 0 only).

        Called before sanity validation to ensure dataloader is ready.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module.
            stage: Current stage ('fit', 'validate', 'test', or 'predict').
        """
        if stage != "fit":
            return

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

        # Log to Lightning logger with latent_diag/* namespace (changed from diag/*)
        pl_module.log_dict(
            {
                "latent_diag/au_count": au_count,
                "latent_diag/au_frac": au_frac,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            rank_zero_only=True,
        )

        # Export for UnifiedCSVCallback merge with latent_diag/* namespace (changed from diag/*)
        self.last_epoch_metrics = {
            "epoch": trainer.current_epoch,
            "latent_diag/au_count": au_count,
            "latent_diag/au_frac": au_frac,
            "latent_diag/au_threshold": self.eps_au,
            "latent_diag/au_subset_size": len(self._subset_indices),
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

        Raises:
            RuntimeError: If subset dataloader is not initialized.
        """
        if self._subset_dataloader is None:
            raise RuntimeError(
                "AU subset dataloader not initialized. "
                "This should not happen if setup() was called properly."
            )

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

        # Use metrics module for AU computation
        results = compute_active_units(mu_mat, eps_au=self.eps_au)

        au_count = results["au_count"]
        au_frac = results["au_frac"]
        z_dim = mu_mat.shape[1]

        return float(au_count), float(au_frac), int(z_dim)
