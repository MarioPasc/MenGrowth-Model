#!/usr/bin/env python
# experiments/lora_ablation/train_condition.py
"""Train a single experimental condition (baseline or LoRA).

This script trains either:
- Baseline: Frozen encoder + trainable segmentation head
- LoRA: Frozen encoder + LoRA adapters + trainable segmentation head

Training uses Dice+CE loss for BraTS-style segmentation.
Early stopping based on validation Dice score.

Usage:
    python -m experiments.lora_ablation.train_condition \
        --config experiments/lora_ablation/config/ablation.yaml \
        --condition lora_r8

    # Quick test with 2 epochs:
    python -m experiments.lora_ablation.train_condition \
        --config experiments/lora_ablation/config/ablation.yaml \
        --condition baseline \
        --max-epochs 2
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from growth.data.bratsmendata import BraTSMENDataset, create_dataloaders
from growth.losses.segmentation import SegmentationLoss, DiceMetric
from growth.models.encoder.lora_adapter import LoRASwinViT, create_lora_encoder
from growth.models.encoder.swin_loader import load_swin_encoder
from growth.models.segmentation.seg_head import SegmentationHead, LoRASegmentationModel
from growth.utils.seed import set_seed
from growth.utils.model_card import LoRAModelCardConfig, model_card_from_training

from .data_splits import load_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaselineSegmentationModel(nn.Module):
    """Frozen encoder with trainable segmentation head.

    Used for baseline condition (no LoRA).

    Note: BraTS has 4 classes (0=background, 1=NCR, 2=ED, 3=ET),
    so out_channels must be 4 for proper one-hot encoding in the loss.
    """

    def __init__(self, encoder: nn.Module, out_channels: int = 4):
        super().__init__()
        self.encoder = encoder
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.decoder = SegmentationHead(out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden_states = self.encoder.swinViT(x, self.encoder.normalize)
        return self.decoder(hidden_states)

    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get encoder hidden states (for feature extraction)."""
        with torch.no_grad():
            return self.encoder.swinViT(x, self.encoder.normalize)

    def get_trainable_param_count(self) -> dict:
        return {
            "encoder": 0,
            "decoder": self.decoder.get_param_count(),
            "total": self.decoder.get_param_count(),
        }


def get_condition_config(config: dict, condition_name: str) -> dict:
    """Get configuration for a specific condition."""
    for cond in config["conditions"]:
        if cond["name"] == condition_name:
            return cond
    raise ValueError(
        f"Unknown condition: {condition_name}. "
        f"Available: {[c['name'] for c in config['conditions']]}"
    )


def create_model(
    condition_config: dict,
    checkpoint_path: str,
    device: str,
) -> nn.Module:
    """Create model based on condition configuration.

    Args:
        condition_config: Condition config dict with 'lora_rank' key.
        checkpoint_path: Path to encoder checkpoint.
        device: Device to load model to.

    Returns:
        Either BaselineSegmentationModel or LoRASegmentationModel.
    """
    lora_rank = condition_config.get("lora_rank")

    if lora_rank is None:
        # Baseline: frozen encoder
        logger.info("Creating baseline model (frozen encoder)")
        encoder = load_swin_encoder(
            checkpoint_path,
            freeze=True,
            device=device,
        )
        model = BaselineSegmentationModel(encoder)
    else:
        # LoRA: frozen encoder + LoRA adapters
        lora_alpha = condition_config.get("lora_alpha", lora_rank * 2)
        logger.info(f"Creating LoRA model (rank={lora_rank}, alpha={lora_alpha})")
        lora_encoder = create_lora_encoder(
            checkpoint_path,
            rank=lora_rank,
            alpha=lora_alpha,
            device=device,
        )
        model = LoRASegmentationModel(lora_encoder)

    model = model.to(device)
    return model


def create_optimizer(
    model: nn.Module,
    config: dict,
    is_baseline: bool,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and scheduler.

    For baseline: only decoder params
    For LoRA: encoder LoRA params (low lr) + decoder params (high lr)
    """
    training_config = config["training"]

    if is_baseline:
        # Only decoder parameters
        params = [{"params": model.decoder.parameters(), "lr": training_config["lr_decoder"]}]
    else:
        # Separate param groups for LoRA and decoder
        encoder_params = [p for p in model.encoder.model.parameters() if p.requires_grad]
        decoder_params = list(model.decoder.parameters())

        params = [
            {"params": encoder_params, "lr": training_config["lr_encoder"]},
            {"params": decoder_params, "lr": training_config["lr_decoder"]},
        ]

    optimizer = AdamW(
        params,
        weight_decay=training_config["weight_decay"],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_config["max_epochs"],
        eta_min=1e-7,
    )

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_clip: float = 1.0,
) -> float:
    """Train for one epoch.

    Returns:
        Average training loss.
    """
    model.train()

    # For baseline, keep encoder in eval mode
    if hasattr(model, "encoder") and not hasattr(model.encoder, "model"):
        model.encoder.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(images)

        # Compute loss
        loss = loss_fn(pred, segs)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                gradient_clip
            )

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    dice_metric: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Validate model.

    Returns:
        Dict with 'loss', 'dice_mean', 'dice_0', 'dice_1', 'dice_2'.
    """
    model.eval()

    total_loss = 0.0
    all_dice_scores = []
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        # Forward pass
        pred = model(images)

        # Compute loss
        loss = loss_fn(pred, segs)
        total_loss += loss.item()

        # Compute Dice per class
        dice_scores = dice_metric(pred, segs)
        all_dice_scores.append(dice_scores.cpu())

        num_batches += 1

    # Aggregate metrics
    avg_loss = total_loss / num_batches
    dice_tensor = torch.stack(all_dice_scores).mean(dim=0)

    return {
        "loss": avg_loss,
        "dice_mean": dice_tensor.mean().item(),
        "dice_0": dice_tensor[0].item(),  # NCR
        "dice_1": dice_tensor[1].item(),  # ED
        "dice_2": dice_tensor[2].item(),  # ET
    }


def save_checkpoint(
    model: nn.Module,
    condition_dir: Path,
    is_baseline: bool,
    epoch: int,
    metrics: dict,
) -> None:
    """Save model checkpoint.

    For baseline: saves full model state
    For LoRA: saves only LoRA adapter
    """
    condition_dir.mkdir(parents=True, exist_ok=True)

    if is_baseline:
        # Save decoder state dict
        checkpoint = {
            "epoch": epoch,
            "decoder_state_dict": model.decoder.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, condition_dir / "checkpoint.pt")
    else:
        # Save LoRA adapter
        adapter_dir = condition_dir / "adapter"
        model.encoder.save_lora(adapter_dir)

        # Also save decoder and metadata
        checkpoint = {
            "epoch": epoch,
            "decoder_state_dict": model.decoder.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, condition_dir / "checkpoint.pt")

    logger.info(f"Saved checkpoint at epoch {epoch}")


def train_condition(
    condition_name: str,
    config: dict,
    splits: dict,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Train a single experimental condition.

    Args:
        condition_name: Name of condition ('baseline', 'lora_r4', etc.)
        config: Full experiment configuration.
        splits: Data splits dictionary.
        max_epochs: Override max epochs (for testing).
        device: Device to train on.

    Returns:
        Dict with best validation metrics.
    """
    # Get condition-specific config
    condition_config = get_condition_config(config, condition_name)
    is_baseline = condition_config.get("lora_rank") is None

    logger.info(f"Training condition: {condition_name}")
    logger.info(f"Description: {condition_config.get('description', 'N/A')}")

    # Set up output directory
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_root=config["paths"]["data_root"],
        train_ids=splits["lora_train"],
        val_ids=splits["lora_val"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        compute_semantic=False,  # Not needed for segmentation training
        augment_train=True,
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(condition_config, config["paths"]["checkpoint"], device)

    # Log parameter counts
    param_counts = model.get_trainable_param_count()
    logger.info(f"Trainable parameters: {param_counts}")

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config, is_baseline)

    # Create loss and metric
    loss_fn = SegmentationLoss(
        lambda_dice=config["loss"]["lambda_dice"],
        lambda_ce=config["loss"]["lambda_ce"],
    )
    dice_metric = DiceMetric()

    # Training configuration
    training_config = config["training"]
    epochs = max_epochs or training_config["max_epochs"]
    patience = training_config["early_stopping_patience"]
    gradient_clip = training_config.get("gradient_clip", 1.0)

    # Training log
    log_path = condition_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice_mean",
                        "val_dice_0", "val_dice_1", "val_dice_2", "lr"])

    # Training loop
    best_dice = 0.0
    best_metrics = {}
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, gradient_clip
        )

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, dice_metric, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics
        logger.info(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice_mean']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                val_metrics["loss"],
                val_metrics["dice_mean"],
                val_metrics["dice_0"],
                val_metrics["dice_1"],
                val_metrics["dice_2"],
                current_lr,
            ])

        # Early stopping check
        if val_metrics["dice_mean"] > best_dice:
            best_dice = val_metrics["dice_mean"]
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics,
            }
            save_checkpoint(model, condition_dir, is_baseline, epoch + 1, best_metrics)
            patience_counter = 0
            logger.info(f"  New best! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Training complete
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time / 60:.1f} minutes")
    logger.info(f"Best validation Dice: {best_dice:.4f} at epoch {best_metrics['epoch']}")

    # Generate model card for LoRA adapters (if enabled)
    if not is_baseline and config.get("model_card", {}).get("enabled", True):
        adapter_dir = condition_dir / "adapter"
        model_card_cfg = model_card_from_training(
            lora_rank=condition_config["lora_rank"],
            lora_alpha=condition_config.get("lora_alpha", condition_config["lora_rank"] * 2),
            lora_dropout=training_config.get("lora_dropout", 0.1),
            train_samples=len(splits["lora_train"]),
            val_samples=len(splits["lora_val"]),
            epochs=best_metrics["epoch"],
            batch_size=training_config["batch_size"],
            lr_encoder=training_config["lr_encoder"],
            lr_decoder=training_config["lr_decoder"],
            best_val_dice=best_dice,
            final_train_loss=best_metrics.get("train_loss", 0.0),
            training_time_seconds=total_time,
            device=device,
            seed=config["experiment"]["seed"],
            trainable_params=param_counts.get("encoder", 0),
            base_model_path=config["paths"]["checkpoint"],
            condition_name=condition_name,
        )
        # Re-save adapter with model card
        model.encoder.save_lora(adapter_dir, model_card_config=model_card_cfg)
        logger.info("Model card generated for LoRA adapter")

    # Save final metrics
    final_metrics_path = condition_dir / "training_summary.yaml"
    summary = {
        "condition": condition_name,
        "is_baseline": is_baseline,
        "param_counts": param_counts,
        "best_epoch": best_metrics["epoch"],
        "best_val_dice": best_dice,
        "total_epochs": epoch + 1,
        "training_time_minutes": total_time / 60,
    }
    with open(final_metrics_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    return best_metrics


def main(
    config_path: str,
    condition: str,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
) -> None:
    """Main entry point for training a condition.

    Args:
        config_path: Path to ablation.yaml.
        condition: Condition name to train.
        max_epochs: Override max epochs (for testing).
        device: Device to train on.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Load splits
    splits = load_splits(config_path)

    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Train condition
    train_condition(
        condition_name=condition,
        config=config,
        splits=splits,
        max_epochs=max_epochs,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a single LoRA ablation condition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["baseline", "lora_r2", "lora_r4", "lora_r8", "lora_r16", "lora_r32"],
        help="Condition to train",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on",
    )

    args = parser.parse_args()
    main(args.config, args.condition, args.max_epochs, args.device)
