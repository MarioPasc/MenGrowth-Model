#!/usr/bin/env python
"""Training script for VAE experiments.

This script orchestrates the training process for:
- Exp1: Baseline 3D VAE with ELBO loss
- Exp2: β-TCVAE with Spatial Broadcast Decoder

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
    python scripts/train.py --config src/vae/config/exp1_baseline_vae.yaml

    # Exp2
    python scripts/train.py --config src/vae/config/exp2_tcvae_sbd.yaml

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
from pytorch_lightning.loggers import CSVLogger

# Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae.data import build_subject_index, create_train_val_split, get_dataloaders
from vae.training import (
    VAELitModule,
    TCVAELitModule,
    ReconstructionCallback,
    TrainingLoggingCallback,
    ActiveUnitsCallback,
)
from vae.training.metrics_callbacks import (
    TidyEpochCSVCallback,
    TidyStepCSVCallback,
    GradNormCallback,
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
        Tuple of (experiment_name, is_tcvae).
    """
    # Check for model variant
    variant = cfg.model.get("variant", None)

    if variant == "tcvae_sbd":
        return "exp2_tcvae_sbd", True
    else:
        # Default to Exp1 baseline
        return "exp1_baseline_vae", False


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)

    # Determine experiment type
    experiment_name, is_tcvae = get_experiment_info(cfg)

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

    if is_tcvae:
        # Exp2: β-TCVAE with SBD
        n_train = len(train_subjects)
        lit_module = TCVAELitModule.from_config(cfg, n_train=n_train)
        logger.info(f"Created TCVAELitModule with N_train={n_train}")
        logger.info(f"  SBD grid size: {cfg.model.sbd_grid_size}")
        logger.info(f"  β_tc target: {cfg.loss.beta_tc_target}")
        logger.info(f"  Gradient checkpointing: {cfg.train.get('gradient_checkpointing', False)}")
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

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="vae-{epoch:03d}-{val_loss:.4f}",
        monitor="val_epoch/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Reconstruction visualization callback
    recon_callback = ReconstructionCallback(
        run_dir=run_dir_str,
        recon_every_n_epochs=cfg.logging.recon_every_n_epochs,
        num_recon_samples=cfg.logging.num_recon_samples,
        modality_names=cfg.data.modalities,
    )
    callbacks.append(recon_callback)

    # === NEW: Tidy CSV logging callbacks ===

    # Tidy epoch CSV (one row per epoch, no NaN fragmentation)
    tidy_epoch_callback = TidyEpochCSVCallback(
        run_dir=run_dir,
        tidy_subdir=cfg.logging.get("tidy_dir", "logs/tidy"),
    )
    callbacks.append(tidy_epoch_callback)
    logger.info("Configured tidy epoch CSV logging")

    # Tidy step CSV (downsampled step logs)
    step_stride = cfg.logging.get("step_log_stride", 50)
    tidy_step_callback = TidyStepCSVCallback(
        run_dir=run_dir,
        stride=step_stride,
        tidy_subdir=cfg.logging.get("tidy_dir", "logs/tidy"),
    )
    callbacks.append(tidy_step_callback)
    logger.info(f"Configured tidy step CSV logging (stride={step_stride})")

    # Gradient norm logging (optimization stability)
    if cfg.logging.get("log_grad_norm", False):
        grad_norm_callback = GradNormCallback(norm_type=2.0)
        callbacks.append(grad_norm_callback)
        logger.info("Configured gradient norm logging")

    # Run metadata JSON (config + data signature)
    run_meta_callback = RunMetadataCallback(
        run_dir=run_dir,
        cfg=cfg,
        tidy_subdir=cfg.logging.get("tidy_dir", "logs/tidy"),
    )
    callbacks.append(run_meta_callback)
    logger.info("Configured run metadata JSON")

    # === END NEW ===

    if cfg.logging.get("latent_diag_every_n_epochs", 0) > 0:
            from vae.training.callbacks import LatentDiagnosticsCallback

            # Get seg_labels from config (convert DictConfig to dict)
            seg_labels = cfg.logging.get("seg_labels", None)
            if seg_labels is not None:
                seg_labels = dict(seg_labels)  # Convert DictConfig to dict

            diag_callback = LatentDiagnosticsCallback(
                run_dir=run_dir_str,
                every_n_epochs=cfg.logging.latent_diag_every_n_epochs,
                num_samples=cfg.logging.latent_diag_num_samples,
                shift_vox=cfg.logging.latent_diag_shift_vox,
                csv_name=cfg.logging.latent_diag_csv_name,
                ids_name=cfg.logging.latent_diag_ids_name,
                seg_labels=seg_labels,  # Pass segmentation labels if provided
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

    # Setup logger
    csv_logger = CSVLogger(
        save_dir=run_dir / "logs",
        name="",
        version="",
    )

    # Calculate log_every_n_steps for at least 3 entries per epoch in CSV
    # Estimate number of training batches
    n_train_samples = len(train_subjects)
    batch_size = cfg.data.batch_size
    approx_batches_per_epoch = max(1, n_train_samples // batch_size)
    log_every_n_steps = max(1, approx_batches_per_epoch // min_logs_per_epoch)
    logger.info(f"CSV logging: every {log_every_n_steps} steps (~{approx_batches_per_epoch // log_every_n_steps} entries/epoch)")

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
        logger=csv_logger,
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
