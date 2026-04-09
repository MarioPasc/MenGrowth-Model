# experiments/uncertainty_segmentation/engine/train_member.py
"""Training engine for a single LoRA ensemble member.

Each ensemble member is an independent LoRA adapter + decoder, trained from
a different random seed (controlling both weight initialization and data
augmentation order). The shared frozen BrainSegFounder backbone provides the
feature extraction foundation.

Usage:
    Called by run_train.py CLI or directly:
        train_single_member(config, member_id=0, device="cuda")

Reuses from the codebase:
    - load_full_swinunetr() for checkpoint loading
    - LoRASwinViT for LoRA adapter injection
    - LoRAOriginalDecoderModel for full segmentation model
    - create_dataloaders() for HDF5 data loading
    - SegmentationLoss3Ch, DiceMetric3Ch for training objectives
"""

import contextlib
import csv
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from growth.data.bratsmendata import create_dataloaders
from growth.losses.segmentation import DiceMetric3Ch, SegmentationLoss3Ch
from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_full_swinunetr
from growth.models.segmentation.original_decoder import LoRAOriginalDecoderModel
from growth.utils.reproducibility import save_reproducibility_artifacts
from growth.utils.seed import set_seed

from .paths import get_run_dir

_INTERACTIVE = sys.stderr.isatty()

logger = logging.getLogger(__name__)


# =============================================================================
# Model Creation
# =============================================================================


def create_ensemble_member_model(
    config: DictConfig,
    device: str = "cuda",
) -> LoRAOriginalDecoderModel:
    """Create a LoRA-adapted segmentation model for one ensemble member.

    Follows the same pattern as model_factory._create_original_decoder_model()
    but without the ablation experiment abstractions.

    Args:
        config: Full experiment configuration.
        device: Device to load model to.

    Returns:
        LoRAOriginalDecoderModel ready for training.
    """
    checkpoint_path = str(
        Path(config.paths.checkpoint_dir) / config.paths.checkpoint_filename
    )

    # Load full SwinUNETR with pretrained encoder + decoder weights
    full_model = load_full_swinunetr(
        checkpoint_path,
        freeze_encoder=True,
        freeze_decoder=False,
        out_channels=config.training.get("out_channels", 3),
        device=device,
    )

    # Wrap encoder with LoRA adapters (init seeded by current RNG state)
    lora_encoder = LoRASwinViT(
        full_model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_stages=list(config.lora.target_stages),
        use_dora=config.lora.get("use_dora", False),
    )

    # Create full segmentation model
    model = LoRAOriginalDecoderModel(
        lora_encoder=lora_encoder,
        freeze_decoder=config.training.freeze_decoder,
        out_channels=config.training.get("out_channels", 3),
        use_semantic_heads=False,
    )

    # Encoder-only mode: freeze decoder but unfreeze the output head
    # This forces LoRA to be the primary adaptation mechanism while allowing
    # the output head to remap channel semantics (GLI→MEN).
    if config.training.get("train_output_head", False) and config.training.freeze_decoder:
        for param in model.decoder.out.parameters():
            param.requires_grad = True
        out_params = sum(p.numel() for p in model.decoder.out.parameters())
        logger.info(f"Unfroze output head only: {out_params} params (decoder remains frozen)")

    return model.to(device)


# =============================================================================
# Optimizer
# =============================================================================


def create_optimizer(
    model: LoRAOriginalDecoderModel,
    config: DictConfig,
) -> tuple[AdamW, dict]:
    """Create AdamW optimizer with separate LR groups and LR schedulers.

    Args:
        model: LoRA segmentation model.
        config: Full experiment configuration.

    Returns:
        (optimizer, scheduler_dict) where scheduler_dict has 'warmup',
        'plateau', and 'warmup_epochs' keys.
    """
    encoder_params = model.get_encoder_params()
    decoder_params = model.get_decoder_params()

    # Use separate output_head LR when in encoder-only mode
    decoder_lr = config.training.learning_rate.get(
        "output_head", config.training.learning_rate.decoder
    ) if config.training.get("train_output_head", False) else config.training.learning_rate.decoder

    param_groups = [
        {
            "params": encoder_params,
            "lr": config.training.learning_rate.encoder,
            "name": "encoder_lora",
        },
        {
            "params": decoder_params,
            "lr": decoder_lr,
            "name": "output_head" if config.training.get("train_output_head", False) else "decoder",
        },
    ]

    optimizer = AdamW(param_groups, weight_decay=config.training.weight_decay)

    warmup_epochs = config.training.get("warmup_epochs", 5)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.training.get("plateau_factor", 0.5),
        patience=config.training.get("plateau_patience", 7),
        min_lr=config.training.get("plateau_min_lr", 1e-7),
    )

    return optimizer, {
        "warmup": warmup_scheduler,
        "plateau": plateau_scheduler,
        "warmup_epochs": warmup_epochs,
    }


# =============================================================================
# Training & Validation Loops
# =============================================================================


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    seg_loss_fn: nn.Module,
    dice_metric: nn.Module,
    device: str,
    use_amp: bool = False,
) -> dict[str, float]:
    """Validate model on MEN-domain data.

    Args:
        model: Model to validate.
        dataloader: Validation DataLoader.
        seg_loss_fn: Segmentation loss function.
        dice_metric: Dice metric module.
        device: Device string.
        use_amp: Whether to use bf16 autocast.

    Returns:
        Dict with loss, dice_mean, dice_tc, dice_wt, dice_et.
    """
    model.eval()
    total_loss = 0.0
    all_dice: list[torch.Tensor] = []
    num_batches = 0
    n_total = len(dataloader)
    log_interval = max(1, n_total // 3)  # Log ~3 times per val pass

    for step, batch in enumerate(
        tqdm(dataloader, desc="Val", leave=False, disable=not _INTERACTIVE)
    ):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            pred = model(images)
            loss = seg_loss_fn(pred, segs, domain="MEN")

        total_loss += loss.item()
        dice = dice_metric(pred.float(), segs, domain="MEN")
        all_dice.append(dice.cpu())
        num_batches += 1

        if not _INTERACTIVE and (step + 1) % log_interval == 0:
            logger.info(
                f"  [val] {step + 1}/{n_total} batches "
                f"({100 * (step + 1) / n_total:.0f}%)"
            )

    if num_batches == 0:
        return {
            "loss": 0.0,
            "dice_mean": 0.0,
            "dice_tc": 0.0,
            "dice_wt": 0.0,
            "dice_et": 0.0,
        }

    avg_loss = total_loss / num_batches
    dice_tensor = torch.cat(all_dice, dim=0).mean(dim=0)  # [3]

    return {
        "loss": avg_loss,
        "dice_mean": dice_tensor.mean().item(),
        "dice_tc": dice_tensor[0].item(),
        "dice_wt": dice_tensor[1].item(),
        "dice_et": dice_tensor[2].item(),
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    seg_loss_fn: nn.Module,
    optimizer: AdamW,
    device: str,
    gradient_clip: float = 1.0,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training DataLoader.
        seg_loss_fn: Segmentation loss function.
        optimizer: Optimizer.
        device: Device string.
        gradient_clip: Maximum gradient norm.
        use_amp: Whether to use bf16 autocast.
        grad_accum_steps: Gradient accumulation steps.

    Returns:
        Dict with 'loss' key.
    """
    model.train()

    total_loss = 0.0
    num_batches = 0
    n_total = len(dataloader)
    log_interval = max(1, n_total // 5)  # Log ~5 times per epoch
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for step, batch in enumerate(
        tqdm(dataloader, desc="Train", leave=False, disable=not _INTERACTIVE)
    ):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            pred = model(images)
            loss = seg_loss_fn(pred, segs, domain="MEN")

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            nn.utils.clip_grad_norm_(trainable_params, gradient_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        # Periodic logging for non-interactive environments (SLURM)
        if not _INTERACTIVE and (step + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"  [train] {step + 1}/{n_total} batches "
                f"({100 * (step + 1) / n_total:.0f}%) | "
                f"loss={avg_loss:.4f}"
            )

    return {"loss": total_loss / max(1, num_batches)}


# =============================================================================
# Checkpoint Saving
# =============================================================================


def save_member_checkpoint(
    model: LoRAOriginalDecoderModel,
    member_dir: Path,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Save LoRA adapter and decoder weights for one ensemble member.

    Saves:
        member_dir/adapter/   — PEFT adapter files (LoRA weights only)
        member_dir/decoder.pt — Decoder state dict (~40M params)

    Args:
        model: Trained LoRA segmentation model.
        member_dir: Output directory for this member.
        epoch: Best epoch number.
        metrics: Best validation metrics.
    """
    member_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter via PEFT
    adapter_dir = member_dir / "adapter"
    model.lora_encoder.save_lora(str(adapter_dir))

    # Save decoder state dict separately
    torch.save(model.decoder.state_dict(), member_dir / "decoder.pt")

    logger.info(
        f"Saved checkpoint at epoch {epoch}: "
        f"adapter → {adapter_dir}, decoder → {member_dir / 'decoder.pt'}"
    )


# =============================================================================
# Main Training Function
# =============================================================================


def train_single_member(
    config: DictConfig,
    member_id: int,
    device: str = "cuda",
    run_dir: Path | str | None = None,
) -> dict[str, float]:
    """Train a single LoRA ensemble member.

    Seeds all RNG sources with (base_seed + member_id) to ensure each member
    gets different random initialization and augmentation ordering.

    Args:
        config: Full experiment configuration (OmegaConf DictConfig).
        member_id: Ensemble member index (0-based).
        device: Device to train on.
        run_dir: Override run directory (from SLURM --run-dir). If None,
            derived from config via get_run_dir().

    Returns:
        Dict with best validation metrics.
    """
    # 1. Seed everything for this member
    seed_m = config.ensemble.base_seed + member_id
    set_seed(seed_m)
    logger.info(f"Training ensemble member {member_id} with seed {seed_m}")

    # 2. Output directory
    resolved_run_dir = get_run_dir(config, override=run_dir)
    member_dir = resolved_run_dir / "adapters" / f"member_{member_id}"
    member_dir.mkdir(parents=True, exist_ok=True)

    # Save reproducibility artifacts (once per run, idempotent)
    config_snapshot_path = resolved_run_dir / "config_snapshot.yaml"
    if not config_snapshot_path.exists():
        OmegaConf.save(config, config_snapshot_path, resolve=True)
        save_reproducibility_artifacts(
            output_dir=resolved_run_dir,
            config=OmegaConf.to_container(config, resolve=True),
            config_path="experiments/uncertainty_segmentation/config.yaml",
        )
        logger.info(f"Saved reproducibility artifacts to {resolved_run_dir}")

    # Set up file logging
    log_handler = logging.FileHandler(member_dir / "train.log", mode="w")
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(log_handler)

    try:
        return _train_member_inner(config, member_id, device, seed_m,
                                   resolved_run_dir, member_dir)
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


def _train_member_inner(
    config: DictConfig,
    member_id: int,
    device: str,
    seed_m: int,
    resolved_run_dir: Path,
    member_dir: Path,
) -> dict[str, float]:
    """Inner training logic (separated for try/finally handler cleanup)."""
    # 3. Create model (LoRA init is seeded by seed_m)
    logger.info("Creating model...")
    model = create_ensemble_member_model(config, device)

    param_counts = model.get_trainable_param_count()
    logger.info(f"Trainable parameters: {param_counts}")

    # 4. Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        h5_path=config.paths.men_h5_file,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        compute_semantic=False,
        augment_train=config.data.augment_train,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        roi_size=tuple(config.data.roi_size),
        val_roi_size=tuple(config.data.val_roi_size),
        val_batch_size=config.training.val_batch_size,
        persistent_workers=config.training.num_workers > 0,
        include_gaussian_noise=config.data.get("augment_gaussian_noise", False),
        include_gaussian_smooth=config.data.get("augment_gaussian_smooth", False),
    )

    logger.info(
        f"Data: {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val samples"
    )

    # 5. Optimizer & scheduler
    optimizer, scheduler_info = create_optimizer(model, config)

    # 6. Loss & metric
    seg_loss_fn = SegmentationLoss3Ch(
        lambda_dice=config.loss.lambda_dice,
        lambda_bce=config.loss.lambda_ce,
    )
    dice_metric = DiceMetric3Ch()

    # 7. Training configuration
    epochs = config.training.epochs
    patience = config.training.early_stopping.patience
    min_delta = config.training.early_stopping.get("min_delta", 0.0)
    gradient_clip = config.training.gradient_clip
    use_amp = config.training.get("use_amp", False)
    grad_accum_steps = config.training.get("grad_accum_steps", 1)

    # 8. CSV log
    log_path = member_dir / "training_log.csv"
    csv_columns = [
        "epoch", "train_loss",
        "val_loss", "val_dice_mean", "val_dice_tc", "val_dice_wt", "val_dice_et",
        "lr", "epoch_time_sec",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_columns)

    # 9. Training loop
    best_score = 0.0
    best_metrics: dict[str, float] = {}
    patience_counter = 0
    start_time = time.time()

    logger.info(
        f"\n{'=' * 60}\n"
        f"TRAINING CONFIG — member {member_id}\n"
        f"  Epochs: {epochs} (patience={patience}, min_delta={min_delta})\n"
        f"  Batch size: {config.training.batch_size} × {grad_accum_steps} accum "
        f"= effective {config.training.batch_size * grad_accum_steps}\n"
        f"  AMP: {use_amp} | LR: encoder={config.training.learning_rate.encoder}, "
        f"decoder={config.training.learning_rate.decoder}\n"
        f"  Warmup: {scheduler_info['warmup_epochs']} epochs → "
        f"Plateau patience: {config.training.get('plateau_patience', 7)}\n"
        f"  Train scans: {len(train_loader.dataset)} | "
        f"Val scans: {len(val_loader.dataset)}\n"
        f"{'=' * 60}"
    )

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        # Train
        train_metrics = train_epoch(
            model, train_loader, seg_loss_fn, optimizer, device,
            gradient_clip=gradient_clip,
            use_amp=use_amp,
            grad_accum_steps=grad_accum_steps,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, seg_loss_fn, dice_metric, device,
            use_amp=use_amp,
        )

        # LR scheduling
        if epoch < scheduler_info["warmup_epochs"]:
            scheduler_info["warmup"].step()
        else:
            scheduler_info["plateau"].step(val_metrics["dice_mean"])
        current_lr = optimizer.param_groups[0]["lr"]

        # Timing & ETA
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time
        avg_epoch_time = elapsed_total / (epoch + 1)
        remaining_epochs = epochs - (epoch + 1)
        eta_sec = avg_epoch_time * remaining_epochs

        # GPU memory (if available)
        gpu_mem_str = ""
        if torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            gpu_mem_str = f" | GPU: {allocated:.1f}/{reserved:.1f}GB"

        # Log
        logger.info(
            f"  Loss: {train_metrics['loss']:.4f} → "
            f"Dice: {val_metrics['dice_mean']:.4f} "
            f"(TC={val_metrics['dice_tc']:.3f} "
            f"WT={val_metrics['dice_wt']:.3f} "
            f"ET={val_metrics['dice_et']:.3f}) | "
            f"LR: {current_lr:.1e}{gpu_mem_str}"
        )
        logger.info(
            f"  Best: {best_score:.4f} | "
            f"Patience: {patience_counter}/{patience} | "
            f"Epoch: {epoch_time:.0f}s | "
            f"ETA: {eta_sec / 60:.0f}min"
        )
        csv_row = [
            epoch + 1, train_metrics["loss"],
            val_metrics["loss"], val_metrics["dice_mean"],
            val_metrics["dice_tc"], val_metrics["dice_wt"], val_metrics["dice_et"],
            current_lr, round(epoch_time, 1),
        ]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(csv_row)

        # Checkpoint & early stopping
        if val_metrics["dice_mean"] > best_score + min_delta:
            best_score = val_metrics["dice_mean"]
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                **val_metrics,
            }
            save_member_checkpoint(model, member_dir, epoch + 1, best_metrics)
            patience_counter = 0
            logger.info(f"  New best! dice_mean={best_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # 10. Save training summary
    total_time = time.time() - start_time
    logger.info(
        f"\n{'=' * 60}\n"
        f"TRAINING COMPLETE — member {member_id}\n"
        f"  Duration: {total_time / 60:.1f} min ({total_time / 3600:.1f} h)\n"
        f"  Best Dice: {best_score:.4f} at epoch {best_metrics.get('epoch', 0)}\n"
        f"    TC={best_metrics.get('dice_tc', 0):.4f}  "
        f"WT={best_metrics.get('dice_wt', 0):.4f}  "
        f"ET={best_metrics.get('dice_et', 0):.4f}\n"
        f"  Total epochs: {epoch + 1}/{epochs} "
        f"({'early stopped' if patience_counter >= patience else 'completed'})\n"
        f"  Output: {member_dir}\n"
        f"{'=' * 60}"
    )

    summary = {
        "member_id": member_id,
        "seed": seed_m,
        "param_counts": dict(param_counts),
        "best_epoch": best_metrics.get("epoch", 0),
        "best_dice_mean": best_metrics.get("dice_mean", 0),
        "best_dice_tc": best_metrics.get("dice_tc", 0),
        "best_dice_wt": best_metrics.get("dice_wt", 0),
        "best_dice_et": best_metrics.get("dice_et", 0),
        "total_epochs": epoch + 1,
        "training_time_minutes": round(total_time / 60, 1),
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
    }
    with open(member_dir / "training_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    return best_metrics
