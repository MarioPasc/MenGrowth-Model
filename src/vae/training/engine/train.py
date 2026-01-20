#!/usr/bin/env python
"""Training script for VAE experiments.

This script orchestrates the training process for:
- Exp1: Baseline 3D VAE with ELBO loss
- Exp2: DIP-VAE with configurable decoder (standard or SBD)
- Exp3: Semi-supervised VAE with partitioned latent space (multi-GPU DDP)

Workflow:
1. Load configuration from YAML
2. Create run directory with timestamp
3. Setup logging
4. Build dataset index and create train/val splits
5. Build data loaders with persistent caching
6. Instantiate model and Lightning module (based on config variant)
7. Configure trainer with callbacks and logging (including DDP for multi-GPU)
8. Run training

Usage:
    # Exp1
    python -m vae.training.engine.train --config src/vae/config/vae.yaml

    # Exp2
    python -m vae.training.engine.train --config src/vae/config/dipvae.yaml

    # Exp3 (Semi-supervised VAE with multi-GPU)
    python -m vae.training.engine.train --config src/vae/config/semivae.yaml

    # Resume from checkpoint
    python -m vae.training.engine.train --config path/to/config.yaml --resume path/to/checkpoint.ckpt
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

from vae.data import (
    build_subject_index,
    create_train_val_split,
    create_train_val_test_split,
    get_dataloaders,
    get_dataloaders_with_test,
)
from vae.training import (
    VAELitModule,
    DIPVAELitModule,
    SemiVAELitModule,
)
from vae.training.callbacks import (
    ReconstructionCallback,
    TrainingLoggingCallback,
    ActiveUnitsCallback,
    LatentDiagnosticsCallback,
    UnifiedCSVCallback,
    RunMetadataCallback,
    GradientStatsCallback,
    SemiVAEDiagnosticsCallback,
    SemiVAELatentVisualizationCallback,
    SemiVAESemanticTrackingCallback,
)
from vae.utils import set_seed, setup_logging, save_config, create_run_dir, save_split_csvs, update_runs_index


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
        variant_type: "baseline", "dipvae", or "semivae"
    """
    # Check for model variant
    variant = cfg.model.get("variant", None)

    if variant == "dipvae":
        return "exp2_dipvae", "dipvae"
    elif variant == "semivae":
        return "exp3_semivae", "semivae"
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

    # Create run directory with informative naming
    run_dir = create_run_dir(cfg.logging.save_dir, cfg=cfg, experiment_type=experiment_name)
    run_dir_str = str(run_dir)

    # Update runs index with running status
    update_runs_index(run_dir, cfg, experiment_name, status="running")

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

    # Create train/val/test split
    test_split = cfg.data.get("test_split", 0.0)
    test_subjects = None
    test_loader = None

    if test_split > 0:
        # 3-way split: train/val/test
        train_subjects, val_subjects, test_subjects = create_train_val_test_split(
            subjects,
            val_split=cfg.data.val_split,
            test_split=test_split,
            seed=cfg.train.seed,
        )
        logger.info(f"Created 3-way split: {len(train_subjects)} train, "
                   f"{len(val_subjects)} val, {len(test_subjects)} test")
    else:
        # 2-way split: train/val only (backward compatible)
        train_subjects, val_subjects = create_train_val_split(
            subjects,
            val_split=cfg.data.val_split,
            seed=cfg.train.seed,
        )

    # Save split CSVs
    save_split_csvs(train_subjects, val_subjects, run_dir_str, test_subjects)

    # Build data loaders
    logger.info("Building data loaders with persistent caching...")
    if test_subjects:
        train_loader, val_loader, test_loader = get_dataloaders_with_test(
            cfg, run_dir_str, train_subjects, val_subjects, test_subjects
        )
    else:
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
    elif variant_type == "semivae":
        # Exp3: Semi-supervised VAE
        lit_module = SemiVAELitModule.from_config(cfg)

        # Log SemiVAE configuration
        from vae.models.components.sbd import SpatialBroadcastDecoder

        model = lit_module.model
        has_sbd = isinstance(model.decoder, SpatialBroadcastDecoder)
        decoder_type = "SpatialBroadcastDecoder (SBD)" if has_sbd else "Standard Transposed-Conv"

        logger.info("=" * 60)
        logger.info("Semi-Supervised VAE Model Configuration")
        logger.info("=" * 60)
        logger.info(f"Decoder type:         {decoder_type}")
        logger.info(f"Latent dim (z_dim):   {cfg.model.z_dim}")
        logger.info(f"Gradient checkpoint:  {cfg.train.get('gradient_checkpointing', False)}")
        logger.info(f"Posterior logvar_min: {cfg.train.get('posterior_logvar_min', -6.0)}")
        logger.info("")
        logger.info("Latent Partitioning:")
        for name, config in model.get_partition_info().items():
            logger.info(f"  {name}: dims [{config['start_idx']}:{config['end_idx']}] "
                       f"({config['dim']} dims, {config['supervision']})")
        logger.info("")
        logger.info("Semantic Loss Weights:")
        logger.info(f"  λ_vol:   {cfg.loss.get('lambda_vol', 10.0)}")
        logger.info(f"  λ_loc:   {cfg.loss.get('lambda_loc', 5.0)}")
        logger.info(f"  λ_shape: {cfg.loss.get('lambda_shape', 5.0)}")
        logger.info(f"  λ_tc:    {cfg.loss.get('lambda_tc', 2.0)}")
        logger.info(f"  Semantic start epoch: {cfg.loss.get('semantic_start_epoch', 10)}")
        logger.info(f"  Semantic warmup:      {cfg.loss.get('semantic_annealing_epochs', 20)} epochs")
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
        log_grad_norm=cfg.logging.get("log_grad_norm", True),
        grad_norm_type=2.0,
    )
    callbacks.append(unified_csv_callback)
    logger.info("Configured unified CSV logging (all metrics in logs/metrics.csv)")

    # Run metadata JSON (config + data signature + hardware info)
    run_meta_callback = RunMetadataCallback(
        run_dir=run_dir,
        cfg=cfg,
    )
    callbacks.append(run_meta_callback)
    logger.info("Configured run metadata JSON (config/run_meta.json)")

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

    # Gradient stats callback for stability analysis
    grad_stats_callback = GradientStatsCallback(run_dir=run_dir)
    callbacks.append(grad_stats_callback)
    logger.info("Configured gradient stats logging (diagnostics/gradients/grad_stats.csv)")

    # === SemiVAE-specific callbacks ===
    if variant_type == "semivae":
        # Get validation dataset for callbacks
        val_dataset = val_loader.dataset

        # Get diagnostic frequency from config
        semivae_diag_every = cfg.logging.get("semivae_diag_every_n_epochs", 10)
        semivae_num_samples = cfg.logging.get("semivae_diag_num_samples", 100)

        # SemiVAE diagnostics (partition stats, semantic quality, cross-correlations)
        semivae_diag_callback = SemiVAEDiagnosticsCallback(
            run_dir=run_dir,
            val_dataset=val_dataset,
            every_n_epochs=semivae_diag_every,
            num_samples=semivae_num_samples,
        )
        callbacks.append(semivae_diag_callback)
        logger.info(f"Configured SemiVAE diagnostics every {semivae_diag_every} epochs")

        # Semantic tracking (lightweight per-epoch R² and correlation tracking)
        semantic_tracking_callback = SemiVAESemanticTrackingCallback(run_dir=run_dir)
        callbacks.append(semantic_tracking_callback)
        logger.info("Configured SemiVAE semantic tracking (diagnostics/semivae/semantic_tracking.csv)")

        # Latent visualization (PCA, partition activity heatmaps)
        if cfg.logging.get("semivae_visualize_latent", True):
            viz_every = cfg.logging.get("semivae_viz_every_n_epochs", 25)
            semivae_viz_callback = SemiVAELatentVisualizationCallback(
                run_dir=run_dir,
                val_dataset=val_dataset,
                every_n_epochs=viz_every,
                num_samples=semivae_num_samples,
            )
            callbacks.append(semivae_viz_callback)
            logger.info(f"Configured SemiVAE latent visualization every {viz_every} epochs")

    # === END SemiVAE callbacks ===

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

    # Configure multi-GPU strategy
    devices = cfg.train.get("devices", 1)
    strategy = cfg.train.get("strategy", "auto")

    # Auto-configure DDP strategy for multi-GPU
    if devices != 1 and strategy == "auto":
        # Use DDP for multi-GPU training
        strategy = "ddp"
        logger.info(f"Auto-configured DDP strategy for {devices} devices")

    # For SemiVAE, ensure DDP-aware TC computation is enabled when using multi-GPU
    if variant_type == "semivae" and devices != 1:
        # Ensure the LitModule knows to use DDP gathering
        if hasattr(lit_module, 'use_ddp_gather'):
            lit_module.use_ddp_gather = True
            logger.info("Enabled DDP-aware TC computation for SemiVAE")

    # Create trainer (disable progress bar since we use custom logging)
    # PyTorch Lightning handles gradient clipping natively via these parameters
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.get("accelerator", "auto"),
        devices=devices,
        strategy=strategy,
        precision=cfg.train.precision,
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=False,  # Disabled in favor of custom logging
        deterministic=cfg.train.get("deterministic", True),
        gradient_clip_val=gradient_clip_val,  # Gradient clipping for numerical stability
        gradient_clip_algorithm=gradient_clip_algorithm,  # "norm" or "value"
        sync_batchnorm=cfg.train.get("sync_batchnorm", False),  # Sync BN across GPUs
    )

    # Log trainer info
    logger.info(f"Trainer configuration:")
    logger.info(f"  Max epochs: {cfg.train.max_epochs}")
    logger.info(f"  Precision: {cfg.train.precision}")
    logger.info(f"  Accelerator: {trainer.accelerator}")
    logger.info(f"  Devices: {devices}")
    logger.info(f"  Strategy: {strategy}")

    # Log multi-GPU specific info
    if devices != 1:
        logger.info(f"  Multi-GPU training enabled:")
        logger.info(f"    - Strategy: {strategy}")
        logger.info(f"    - Sync BatchNorm: {cfg.train.get('sync_batchnorm', False)}")
        if variant_type == "semivae":
            logger.info(f"    - DDP-aware TC: enabled")

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

    # Run test evaluation if test set is available
    test_results = None
    if test_loader is not None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("RUNNING FINAL TEST EVALUATION")
        logger.info("=" * 60)
        logger.info(f"Test set: {len(test_loader.dataset)} samples")

        # Load best checkpoint for test evaluation
        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt:
            logger.info(f"Loading best checkpoint: {best_ckpt}")

        # Run test
        test_results = trainer.test(
            lit_module,
            dataloaders=test_loader,
            ckpt_path=best_ckpt,  # Use best checkpoint
            weights_only=False
        )

        # Log test results
        if test_results:
            logger.info("")
            logger.info("Test Results:")
            for key, value in test_results[0].items():
                logger.info(f"  {key}: {value:.4f}")

            # Save test results to CSV
            test_metrics_path = run_dir / "logs" / "test_metrics.csv"
            test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            import csv
            with open(test_metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for key, value in test_results[0].items():
                    writer.writerow([key, value])
            logger.info(f"Test metrics saved to: {test_metrics_path}")

        logger.info("=" * 60)

    # Get final AU metrics from callback if available
    final_au_count = None
    final_au_frac = None
    if cfg.logging.get("au_dense_until", -1) >= 0:
        if au_callback.last_epoch_metrics is not None:
            final_au_count = au_callback.last_epoch_metrics.get("latent_diag/au_count")
            final_au_frac = au_callback.last_epoch_metrics.get("latent_diag/au_frac")

    # Update runs index with completed status and final metrics
    best_val_loss = checkpoint_callback.best_model_score
    best_epoch = None
    if checkpoint_callback.best_model_path:
        # Extract epoch from checkpoint path (e.g., "vae-epoch=042.ckpt")
        import re
        match = re.search(r"epoch[=_](\d+)", checkpoint_callback.best_model_path)
        if match:
            best_epoch = int(match.group(1))

    # Extract test loss if available
    test_loss = None
    if test_results:
        test_loss = test_results[0].get("test_epoch/loss")

    update_runs_index(
        run_dir=run_dir,
        cfg=cfg,
        experiment_type=experiment_name,
        status="completed",
        best_val_loss=float(best_val_loss) if best_val_loss is not None else None,
        best_epoch=best_epoch,
        final_au_count=final_au_count,
        final_au_frac=final_au_frac,
        test_loss=float(test_loss) if test_loss is not None else None,
    )


if __name__ == "__main__":
    main()
