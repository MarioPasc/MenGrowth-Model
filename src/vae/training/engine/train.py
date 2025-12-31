#!/usr/bin/env python
"""Training script for VAE experiments.

This script orchestrates the training process for:
- Exp1: Baseline 3D VAE with ELBO loss
- Exp2: DIP-VAE with configurable decoder (standard or SBD)

Workflow:
1. Load configuration from YAML
2. Create run directory with timestamp
3. Setup logging
4. Build dataset index and create train/val splits
5. Build data loaders with persistent caching
6. Instantiate model and Lightning module (based on config variant)
7. Configure trainer with callbacks and logging
8. Run training

Usage:
    # Exp1
    python scripts/train.py --config src/vae/config/vae.yaml

    # Exp2
    python scripts/train.py --config src/vae/config/dipvae.yaml

    # Resume from checkpoint
    python scripts/train.py --config path/to/config.yaml --resume path/to/checkpoint.ckpt
"""

import argparse
import logging
import sys
from math import ceil
from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

# Import wandb conditionally
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae.data import build_subject_index, create_train_val_split, get_dataloaders
from vae.training import (
    VAELitModule,
    DIPVAELitModule,
)
from vae.training.callbacks import (
    ReconstructionCallback,
    TrainingLoggingCallback,
    ActiveUnitsCallback,
    LatentDiagnosticsCallback,
    UnifiedCSVCallback,
    RunMetadataCallback,
)
from vae.utils import set_seed, setup_logging, save_config, create_run_dir, save_split_csvs


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train 3D VAE for multi-modal MRI (Exp1 or Exp2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def get_experiment_info(cfg: OmegaConf) -> tuple:
    """Determine experiment type from configuration.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (experiment_name, variant_type).
        variant_type: "baseline" or "dipvae"
    """
    # Check for model variant
    variant = cfg.model.get("variant", None)

    if variant == "dipvae":
        return "exp2_dipvae", "dipvae"
    else:
        # Default to Exp1 baseline
        return "exp1_baseline_vae", "baseline"


def create_logger(cfg: OmegaConf, run_dir: Path, experiment_name: str):
    """Create logger based on config (wandb or csv).

    Args:
        cfg: Configuration object
        run_dir: Run directory path
        experiment_name: Experiment name for wandb run naming

    Returns:
        CSVLogger or WandbLogger instance
    """
    logger_type = cfg.logging.get("logger", {}).get("type", "csv")

    if logger_type == "wandb" and WANDB_AVAILABLE:
        wandb_cfg = cfg.logging.logger.wandb

        # Auto-generate run name if not provided
        run_name = wandb_cfg.get("name", None)
        if run_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{experiment_name}_{timestamp}"

        # Initialize WandbLogger
        wandb_logger = WandbLogger(
            project=wandb_cfg.project,
            entity=wandb_cfg.get("entity", None),
            name=run_name,
            tags=list(wandb_cfg.get("tags", [])),
            notes=wandb_cfg.get("notes", ""),
            offline=wandb_cfg.get("offline", True),
            save_dir=str(run_dir / "wandb"),
            save_code=wandb_cfg.get("save_code", True),
            log_model=wandb_cfg.get("log_model", False),
        )

        # Log config as hyperparameters
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

        logger.info(f"Wandb logger initialized: {wandb_cfg.project}/{run_name}")
        logger.info(f"  Offline mode: {wandb_cfg.get('offline', True)}")
        logger.info(f"  Run dir: {run_dir / 'wandb'}")

        return wandb_logger
    else:
        # Fallback to CSV logger
        if logger_type == "wandb" and not WANDB_AVAILABLE:
            logger.warning("Wandb requested but not available. Falling back to CSVLogger.")

        csv_logger = CSVLogger(
            save_dir=run_dir / "logs",
            name="",
            version="",
        )
        logger.info("CSV logger initialized")
        return csv_logger


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)

    # Determine experiment type
    experiment_name, variant_type = get_experiment_info(cfg)

    # Create run directory
    run_dir = create_run_dir(cfg.logging.save_dir, experiment_name=experiment_name)
    run_dir_str = str(run_dir)

    # Setup logging to console and file
    setup_logging(run_dir_str)
    logger.info(f"Starting {experiment_name} training")
    logger.info(f"Run directory: {run_dir}")

    # Set seed for reproducibility
    set_seed(cfg.train.seed, workers=True)
    logger.info(f"Set random seed: {cfg.train.seed}")

    # Save resolved configuration
    save_config(cfg, run_dir_str)

    # Build dataset index
    logger.info(f"Building subject index from {cfg.data.root_dir}")
    subjects = build_subject_index(cfg.data.root_dir, cfg.data.modalities)

    # Create train/val split
    train_subjects, val_subjects = create_train_val_split(
        subjects,
        val_split=cfg.data.val_split,
        seed=cfg.train.seed,
    )

    # Save split CSVs
    save_split_csvs(train_subjects, val_subjects, run_dir_str)

    # Build data loaders
    logger.info("Building data loaders with persistent caching...")
    train_loader, val_loader = get_dataloaders(
        cfg, run_dir_str, train_subjects, val_subjects
    )

    # Create model and Lightning module based on experiment type
    logger.info("Creating model...")

    if variant_type == "dipvae":
        # Exp2b: DIP-VAE with configurable decoder
        lit_module = DIPVAELitModule.from_config(cfg)

        # CRITICAL: Derive decoder type from ACTUAL instantiated module, not config
        # This is single source of truth - log cannot lie if wiring changes
        from vae.models.components.sbd import SpatialBroadcastDecoder

        model = lit_module.model
        has_sbd = isinstance(model.decoder, SpatialBroadcastDecoder)
        decoder_type = "SpatialBroadcastDecoder (SBD)" if has_sbd else "Standard Transposed-Conv"

        # Verify config matches reality (detect drift)
        config_use_sbd = cfg.model.get("use_sbd", True)
        if has_sbd != config_use_sbd:
            logger.warning(
                f"Config/model mismatch: cfg.model.use_sbd={config_use_sbd} but "
                f"instantiated decoder is {decoder_type}. Config may be stale."
            )

        logger.info("=" * 60)
        logger.info("DIP-VAE Model Configuration")
        logger.info("=" * 60)
        logger.info(f"Decoder type:         {decoder_type}")
        logger.info(f"Latent dim (z_dim):   {cfg.model.z_dim}")
        logger.info(f"Gradient checkpoint:  {cfg.train.get('gradient_checkpointing', False)}")
        logger.info(f"Posterior logvar_min: {cfg.train.get('posterior_logvar_min', -6.0)}")
        logger.info("")
        logger.info("DIP-VAE Loss Parameters:")
        logger.info(f"  λ_od (off-diagonal): {cfg.loss.lambda_od}")
        logger.info(f"  λ_d (diagonal):      {cfg.loss.lambda_d}")
        logger.info(f"  Lambda start epoch:  {cfg.loss.get('lambda_start_epoch', 0)}")
        logger.info(f"  Lambda warmup epochs: {cfg.loss.get('lambda_cov_annealing_epochs', 0)}")

        # Only log SBD params if ACTUALLY using SBD (based on introspection)
        if has_sbd:
            logger.info("")
            logger.info("SBD Decoder Parameters:")
            logger.info(f"  Grid size:       {cfg.model.sbd_grid_size}")
            logger.info(f"  Upsample mode:   {cfg.model.get('sbd_upsample_mode', 'resize_conv')}")

        logger.info("=" * 60)
    else:
        # Exp1: Baseline VAE
        lit_module = VAELitModule.from_config(cfg)
        logger.info("Created VAELitModule (baseline)")
        logger.info(f"  KL beta: {cfg.train.kl_beta}")

    # Log model info
    total_params = sum(p.numel() for p in lit_module.model.parameters())
    trainable_params = sum(p.numel() for p in lit_module.model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback (use config settings if available)
    ckpt_cfg = cfg.logging.get("checkpointing", {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename=ckpt_cfg.get("filename", "vae-{epoch:03d}"),
        monitor=ckpt_cfg.get("monitor", "val_epoch/loss"),
        mode=ckpt_cfg.get("mode", "min"),
        save_top_k=ckpt_cfg.get("save_top_k", 3),
        save_last=ckpt_cfg.get("save_last", True),
        every_n_epochs=ckpt_cfg.get("every_n_epochs", 1),
    )
    callbacks.append(checkpoint_callback)

    # Reconstruction visualization callback
    log_to_wandb = cfg.logging.get("logger", {}).get("type", "csv") == "wandb"
    recon_callback = ReconstructionCallback(
        run_dir=run_dir_str,
        recon_every_n_epochs=cfg.logging.visual.get("recon_every_n_epochs", cfg.logging.get("recon_every_n_epochs", 5)),
        num_recon_samples=cfg.logging.visual.get("num_recon_samples", cfg.logging.get("num_recon_samples", 2)),
        modality_names=cfg.data.modalities,
        log_to_wandb=log_to_wandb,
    )
    callbacks.append(recon_callback)

    # === Unified CSV logging ===

    # Unified CSV callback (consolidates epoch metrics, system metrics, gradient norms)
    unified_csv_callback = UnifiedCSVCallback(
        run_dir=run_dir,
        tidy_subdir=cfg.logging.get("tidy_dir", "logs/tidy"),
        log_grad_norm=cfg.logging.get("log_grad_norm", True),
        grad_norm_type=2.0,
    )
    callbacks.append(unified_csv_callback)
    logger.info("Configured unified CSV logging (all metrics in epoch_metrics.csv)")

    # Run metadata JSON (config + data signature)
    run_meta_callback = RunMetadataCallback(
        run_dir=run_dir,
        cfg=cfg,
        tidy_subdir=cfg.logging.get("tidy_dir", "logs/tidy"),
    )
    callbacks.append(run_meta_callback)
    logger.info("Configured run metadata JSON")

    # === END unified CSV logging ===

    if cfg.logging.get("latent_diag_every_n_epochs", 0) > 0:
        # Get seg_labels from config (REQUIRED, convert DictConfig to dict)
        seg_labels = cfg.logging.get("seg_labels", None)
        if seg_labels is None:
            raise ValueError(
                "logging.seg_labels must be defined in config for latent diagnostics. "
                "Example for BraTS: seg_labels: {ncr: 1, ed: 2, et: 3}"
            )
        seg_labels = dict(seg_labels)  # Convert DictConfig to dict

        diag_callback = LatentDiagnosticsCallback(
            run_dir=run_dir_str,
            seg_labels=seg_labels,  # REQUIRED parameter
            spacing=tuple(cfg.data.spacing),  # Pass spacing
            every_n_epochs=cfg.logging.latent_diag_every_n_epochs,
            num_samples=cfg.logging.latent_diag_num_samples,
            shift_vox=cfg.logging.latent_diag_shift_vox,
            ids_name=cfg.logging.latent_diag_ids_name,
        )
        callbacks.append(diag_callback)
        logger.info(f"Configured latent diagnostics every {cfg.logging.latent_diag_every_n_epochs} epochs")

    # Active Units (AU) callback - canonical latent activity tracking
    if cfg.logging.get("au_dense_until", -1) >= 0:
        val_dataset = val_loader.dataset  # Get underlying dataset from DataLoader

        au_callback = ActiveUnitsCallback(
            run_dir=run_dir_str,
            val_dataset=val_dataset,
            au_dense_until=cfg.logging.get("au_dense_until", 15),
            au_sparse_interval=cfg.logging.get("au_sparse_interval", 5),
            au_subset_fraction=cfg.logging.get("au_subset_fraction", 0.25),
            au_batch_size=cfg.logging.get("au_batch_size", 64),
            eps_au=cfg.logging.get("eps_au", 0.01),
            au_subset_seed=cfg.logging.get("au_subset_seed", 42),
        )
        callbacks.append(au_callback)

        n_val = len(val_dataset)
        n_subset = ceil(cfg.logging.get("au_subset_fraction", 0.25) * n_val)
        logger.info(
            f"Configured Active Units (AU) callback: "
            f"dense until epoch {cfg.logging.get('au_dense_until', 15)}, "
            f"then every {cfg.logging.get('au_sparse_interval', 5)} epochs, "
            f"subset={n_subset}/{n_val} samples ({cfg.logging.get('au_subset_fraction', 0.25):.1%})"
        )

    # Custom training logging callback (replaces tqdm)
    min_logs_per_epoch = cfg.logging.get("min_logs_per_epoch", 3)
    logging_callback = TrainingLoggingCallback(
        min_logs_per_epoch=min_logs_per_epoch,
        log_val_every_n_batches=1,
    )
    callbacks.append(logging_callback)
    logger.info(f"Configured logging: at least {min_logs_per_epoch} logs per training epoch")

    # Setup logger (wandb or csv based on config)
    pl_logger = create_logger(cfg, run_dir, experiment_name)

    # Calculate log_every_n_steps from config or calculate default
    log_every_n_steps = cfg.logging.get("log_every_n_steps", None)
    if log_every_n_steps is None:
        # Fall back to legacy calculation
        n_train_samples = len(train_subjects)
        batch_size = cfg.data.batch_size
        approx_batches_per_epoch = max(1, n_train_samples // batch_size)
        log_every_n_steps = max(1, approx_batches_per_epoch // min_logs_per_epoch)

    logger.info(f"Logging every {log_every_n_steps} steps")

    # Extract gradient clipping config
    gradient_clip_val = cfg.train.get("gradient_clip_val", None)
    gradient_clip_algorithm = cfg.train.get("gradient_clip_algorithm", "norm")

    # Create trainer (disable progress bar since we use custom logging)
    # PyTorch Lightning handles gradient clipping natively via these parameters
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.get("accelerator", "auto"),
        devices=cfg.train.get("devices", 1),
        precision=cfg.train.precision,
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=False,  # Disabled in favor of custom logging
        deterministic=True,
        gradient_clip_val=gradient_clip_val,  # Gradient clipping for numerical stability
        gradient_clip_algorithm=gradient_clip_algorithm,  # "norm" or "value"
    )

    # Log trainer info
    logger.info(f"Trainer configuration:")
    logger.info(f"  Max epochs: {cfg.train.max_epochs}")
    logger.info(f"  Precision: {cfg.train.precision}")
    logger.info(f"  Accelerator: {trainer.accelerator}")

    # Log gradient clipping info
    if gradient_clip_val is not None and gradient_clip_val > 0:
        logger.info(f"  Gradient clipping: {gradient_clip_algorithm} with value {gradient_clip_val}")
    else:
        logger.info("  Gradient clipping: disabled")

    # Start training
    logger.info("Starting training...")
    trainer.fit(
        lit_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )

    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Best val/loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
