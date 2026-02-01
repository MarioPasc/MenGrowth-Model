#!/usr/bin/env python
# experiments/lora_ablation/train_condition_v2.py
"""Enhanced training with original decoder and auxiliary semantic losses.

Key improvements over v1:
1. Uses original SwinUNETR decoder for stronger gradient signal
2. Optional auxiliary semantic prediction losses during training
3. Multi-scale feature extraction support
4. Better logging and diagnostics

Usage:
    python -m experiments.lora_ablation.train_condition_v2 \
        --config experiments/lora_ablation/config/ablation_v2.yaml \
        --condition lora_r8
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from growth.data.bratsmendata import BraTSMENDataset, create_dataloaders
from growth.losses.segmentation import SegmentationLoss, DiceMetric
from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_full_swinunetr
from growth.models.segmentation.original_decoder import (
    LoRAOriginalDecoderModel,
    OriginalDecoderSegmentationModel,
)
from growth.models.segmentation.semantic_heads import AuxiliarySemanticLoss
from growth.utils.seed import set_seed
from growth.utils.model_card import LoRAModelCardConfig, model_card_from_training

from .data_splits import load_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaselineOriginalDecoderModel(nn.Module):
    """Frozen encoder with original SwinUNETR decoder (pretrained weights).

    Used for baseline condition with the full decoder capacity.
    IMPORTANT: Uses load_full_swinunetr to load pretrained decoder weights.
    """

    def __init__(self, full_model: nn.Module, out_channels: int = 4):
        super().__init__()

        # full_model already has pretrained decoder weights from load_full_swinunetr
        # Wrap in OriginalDecoderSegmentationModel
        self.model = OriginalDecoderSegmentationModel(
            encoder=full_model,
            freeze_decoder=False,  # Train decoder (fine-tune on meningiomas)
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.model.get_hidden_states(x)

    def get_trainable_param_count(self) -> dict:
        return self.model.get_trainable_param_count()


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
    use_semantic_heads: bool = False,
    freeze_decoder: bool = False,
) -> nn.Module:
    """Create model with original decoder and PRETRAINED weights.

    IMPORTANT: Uses load_full_swinunetr() to load ALL pretrained weights
    including the decoder, which is essential for good performance (~0.85 Dice).

    Args:
        condition_config: Condition config dict.
        checkpoint_path: Path to BrainSegFounder checkpoint.
        device: Device to load model to.
        use_semantic_heads: If True, add auxiliary semantic heads.
        freeze_decoder: If True, freeze decoder weights.

    Returns:
        Model with original decoder architecture and pretrained weights.
    """
    lora_rank = condition_config.get("lora_rank")

    if lora_rank is None:
        # Baseline: frozen encoder + trainable original decoder (pretrained)
        logger.info("Creating baseline model with ORIGINAL decoder (pretrained weights)")
        full_model = load_full_swinunetr(
            checkpoint_path,
            freeze_encoder=True,   # Freeze swinViT
            freeze_decoder=False,  # Train decoder (fine-tune for meningiomas)
            out_channels=4,
            device=device,
        )
        model = BaselineOriginalDecoderModel(full_model, out_channels=4)
    else:
        # LoRA: frozen encoder + LoRA adapters + pretrained original decoder
        lora_alpha = condition_config.get("lora_alpha", lora_rank * 2)
        logger.info(f"Creating LoRA model with ORIGINAL decoder (rank={lora_rank}, pretrained weights)")

        # Load full model with pretrained decoder weights
        full_model = load_full_swinunetr(
            checkpoint_path,
            freeze_encoder=True,   # Will be unfrozen for LoRA layers
            freeze_decoder=freeze_decoder,
            out_channels=4,
            device=device,
        )

        # Wrap with LoRA adapters
        lora_encoder = LoRASwinViT(
            full_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=0.1,
            target_stages=[3, 4],
        )

        model = LoRAOriginalDecoderModel(
            lora_encoder=lora_encoder,
            freeze_decoder=freeze_decoder,
            out_channels=4,
            use_semantic_heads=use_semantic_heads,
        )

    model = model.to(device)
    return model


def create_optimizer(
    model: nn.Module,
    config: dict,
    is_baseline: bool,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer with separate param groups."""
    training_config = config["training"]

    if is_baseline:
        # Only decoder parameters
        params = [{"params": model.model.decoder.parameters(),
                  "lr": training_config["lr_decoder"]}]
    else:
        # Separate groups for LoRA, decoder, and semantic heads
        encoder_params = model.get_encoder_params()
        decoder_params = model.get_decoder_params()

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


def compute_target_statistics(dataloader: DataLoader) -> Dict[str, torch.Tensor]:
    """Compute mean and std of semantic targets for normalization."""
    all_volumes = []
    all_locations = []
    all_shapes = []

    for batch in dataloader:
        if "semantic_features" in batch:
            all_volumes.append(batch["semantic_features"]["volume"])
            all_locations.append(batch["semantic_features"]["location"])
            all_shapes.append(batch["semantic_features"]["shape"])

    if not all_volumes:
        return {}

    volumes = torch.cat(all_volumes, dim=0)
    locations = torch.cat(all_locations, dim=0)
    shapes = torch.cat(all_shapes, dim=0)

    return {
        'volume_mean': volumes.mean(dim=0),
        'volume_std': volumes.std(dim=0).clamp(min=1e-6),
        'location_mean': locations.mean(dim=0),
        'location_std': locations.std(dim=0).clamp(min=1e-6),
        'shape_mean': shapes.mean(dim=0),
        'shape_std': shapes.std(dim=0).clamp(min=1e-6),
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    seg_loss_fn: nn.Module,
    aux_loss_fn: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_clip: float = 1.0,
    lambda_aux: float = 0.1,
    use_semantic_heads: bool = False,
) -> Dict[str, float]:
    """Train for one epoch with optional auxiliary losses.

    Returns:
        Dict with loss components.
    """
    model.train()

    # For baseline, keep encoder in eval mode
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        model.model.encoder.eval()

    total_seg_loss = 0.0
    total_aux_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_semantic_heads and hasattr(model, 'forward_with_semantics'):
            outputs = model.forward_with_semantics(images)
            pred = outputs['logits']
        else:
            pred = model(images)

        # Segmentation loss
        seg_loss = seg_loss_fn(pred, segs)
        loss = seg_loss

        # Auxiliary semantic loss (if enabled)
        aux_loss = torch.tensor(0.0, device=device)
        if use_semantic_heads and aux_loss_fn is not None and "semantic_features" in batch:
            semantic_targets = {
                'volume': batch["semantic_features"]["volume"].to(device),
                'location': batch["semantic_features"]["location"].to(device),
                'shape': batch["semantic_features"]["shape"].to(device),
            }
            aux_loss, _ = aux_loss_fn(outputs, semantic_targets)
            loss = seg_loss + lambda_aux * aux_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                gradient_clip
            )

        optimizer.step()

        total_seg_loss += seg_loss.item()
        total_aux_loss += aux_loss.item()
        total_loss += loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "seg_loss": total_seg_loss / num_batches,
        "aux_loss": total_aux_loss / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    seg_loss_fn: nn.Module,
    dice_metric: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    all_dice_scores = []
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        # Forward pass
        if hasattr(model, 'forward_with_semantics'):
            outputs = model.forward_with_semantics(images)
            pred = outputs['logits']
        else:
            pred = model(images)

        # Compute loss
        loss = seg_loss_fn(pred, segs)
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
    """Save model checkpoint."""
    condition_dir.mkdir(parents=True, exist_ok=True)

    if is_baseline:
        # Save decoder state dict
        checkpoint = {
            "epoch": epoch,
            "decoder_state_dict": model.model.decoder.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, condition_dir / "checkpoint.pt")
        # Also save as best_model.pt for compatibility
        torch.save(model.state_dict(), condition_dir / "best_model.pt")
    else:
        # Save LoRA adapter
        adapter_dir = condition_dir / "adapter"
        model.lora_encoder.save_lora(adapter_dir)

        # Also save decoder and metadata
        checkpoint = {
            "epoch": epoch,
            "decoder_state_dict": model.decoder.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, condition_dir / "checkpoint.pt")

        # Save full model state for evaluation
        torch.save(model.state_dict(), condition_dir / "best_model.pt")

    logger.info(f"Saved checkpoint at epoch {epoch}")


def train_condition(
    condition_name: str,
    config: dict,
    splits: dict,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Train a single experimental condition with enhanced setup."""
    # Get condition-specific config
    condition_config = get_condition_config(config, condition_name)
    is_baseline = condition_config.get("lora_rank") is None

    logger.info(f"Training condition: {condition_name}")
    logger.info(f"Description: {condition_config.get('description', 'N/A')}")

    # Get training options
    training_config = config["training"]
    use_semantic_heads = training_config.get("use_semantic_heads", False)
    freeze_decoder = training_config.get("freeze_decoder", False)
    lambda_aux = training_config.get("lambda_aux", 0.1)

    logger.info(f"Use semantic heads: {use_semantic_heads}")
    logger.info(f"Freeze decoder: {freeze_decoder}")
    logger.info(f"Lambda aux: {lambda_aux}")

    # Set up output directory
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders (with semantic features if using aux loss)
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_root=config["paths"]["data_root"],
        train_ids=splits["lora_train"],
        val_ids=splits["lora_val"],
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        compute_semantic=use_semantic_heads,
        augment_train=True,
    )

    # Create model with original decoder
    logger.info("Creating model with ORIGINAL decoder...")
    model = create_model(
        condition_config,
        config["paths"]["checkpoint"],
        device,
        use_semantic_heads=use_semantic_heads,
        freeze_decoder=freeze_decoder,
    )

    # Log parameter counts
    param_counts = model.get_trainable_param_count()
    logger.info(f"Trainable parameters: {param_counts}")

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config, is_baseline)

    # Create losses
    seg_loss_fn = SegmentationLoss(
        lambda_dice=config["loss"]["lambda_dice"],
        lambda_ce=config["loss"]["lambda_ce"],
    )
    dice_metric = DiceMetric()

    # Auxiliary semantic loss (if enabled)
    aux_loss_fn = None
    if use_semantic_heads and not is_baseline:
        aux_loss_fn = AuxiliarySemanticLoss(
            lambda_volume=config["loss"].get("lambda_volume", 1.0),
            lambda_location=config["loss"].get("lambda_location", 1.0),
            lambda_shape=config["loss"].get("lambda_shape", 1.0),
            normalize_targets=True,
        )

        # Compute target statistics for normalization
        logger.info("Computing target statistics...")
        stats = compute_target_statistics(train_loader)
        if stats:
            aux_loss_fn.update_statistics(
                stats['volume_mean'].unsqueeze(0).expand(100, -1),
                stats['location_mean'].unsqueeze(0).expand(100, -1),
                stats['shape_mean'].unsqueeze(0).expand(100, -1),
            )

    # Training configuration
    epochs = max_epochs or training_config["max_epochs"]
    patience = training_config["early_stopping_patience"]
    gradient_clip = training_config.get("gradient_clip", 1.0)

    # Training log
    log_path = condition_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_seg_loss", "train_aux_loss",
                        "val_loss", "val_dice_mean", "val_dice_0", "val_dice_1",
                        "val_dice_2", "lr"])

    # Training loop
    best_dice = 0.0
    best_metrics = {}
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, seg_loss_fn, aux_loss_fn,
            optimizer, device, gradient_clip, lambda_aux,
            use_semantic_heads=use_semantic_heads and not is_baseline,
        )

        # Validate
        val_metrics = validate(model, val_loader, seg_loss_fn, dice_metric, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics
        logger.info(
            f"  Train Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f}, aux={train_metrics['aux_loss']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice_mean']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics["loss"],
                train_metrics["seg_loss"],
                train_metrics["aux_loss"],
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
                "train_loss": train_metrics["loss"],
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

    # Save final summary
    summary = {
        "condition": condition_name,
        "is_baseline": is_baseline,
        "param_counts": param_counts,
        "best_epoch": best_metrics["epoch"],
        "best_val_dice": best_dice,
        "total_epochs": epoch + 1,
        "training_time_minutes": total_time / 60,
        "use_original_decoder": True,
        "use_semantic_heads": use_semantic_heads,
        "freeze_decoder": freeze_decoder,
    }
    with open(condition_dir / "training_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    return best_metrics


def main(
    config_path: str,
    condition: str,
    max_epochs: Optional[int] = None,
    device: str = "cuda",
) -> None:
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])
    splits = load_splits(config_path)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    train_condition(
        condition_name=condition,
        config=config,
        splits=splits,
        max_epochs=max_epochs,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LoRA ablation with original decoder"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.max_epochs, args.device)
