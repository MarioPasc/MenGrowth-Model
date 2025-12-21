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
    python scripts/train.py --config src/vae_dynamics/config/exp1_baseline_vae.yaml

    # Exp2
    python scripts/train.py --config src/vae_dynamics/config/exp2_tcvae_sbd.yaml

    # Resume from checkpoint
    python scripts/train.py --config path/to/config.yaml --resume path/to/checkpoint.ckpt
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae_dynamics.data import build_subject_index, create_train_val_split, get_dataloaders
from vae_dynamics.training import VAELitModule, TCVAELitModule, ReconstructionCallback
from vae_dynamics.utils import set_seed, setup_logging, save_config, create_run_dir, save_split_csvs


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
        monitor="val/loss",
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

    # Setup logger
    csv_logger = CSVLogger(
        save_dir=run_dir / "logs",
        name="",
        version="",
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.get("accelerator", "auto"),
        devices=cfg.train.get("devices", 1),
        precision=cfg.train.precision,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        deterministic=True,
    )

    # Log trainer info
    logger.info(f"Trainer configuration:")
    logger.info(f"  Max epochs: {cfg.train.max_epochs}")
    logger.info(f"  Precision: {cfg.train.precision}")
    logger.info(f"  Accelerator: {trainer.accelerator}")

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
