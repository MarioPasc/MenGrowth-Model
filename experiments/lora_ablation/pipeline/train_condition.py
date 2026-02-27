#!/usr/bin/env python
# experiments/lora_ablation/train_condition.py
"""Unified training script for LoRA ablation with configurable decoder.

Supports both decoder types via the `decoder_type` config parameter:
- "lightweight": Custom SegmentationHead (~2M params) - v1 behavior
- "original": Full SwinUNETR decoder (~30M params) - v2 behavior (recommended)

Key features:
1. Configurable decoder architecture via decoder_type
2. Optional auxiliary semantic prediction losses during training
3. Auxiliary loss warmup for stable multi-task learning
4. Per-component loss logging for monitoring
5. Optional gradient diagnostics (magnitude and conflict detection)
6. Multi-scale feature extraction support

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
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from growth.data.bratsmendata import create_dataloaders
from growth.losses.encoder_vicreg import EncoderVICRegLoss
from growth.losses.segmentation import DiceMetric3Ch, SegmentationLoss3Ch
from growth.models.segmentation.semantic_heads import AuxiliarySemanticLoss
from growth.utils.model_card import model_card_from_training
from growth.utils.seed import set_seed

from .data_splits import load_splits
from .model_factory import create_ablation_model, get_condition_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Track whether file logging has been set up
_file_logging_initialized = False


def setup_file_logging(output_dir: Path, condition_name: str) -> Path:
    """Set up file logging for a training condition.

    Creates a log file in the condition directory that captures all
    log messages. Useful for post-hoc debugging:
        grep -E "WARNING|ERROR" experiment.log

    Args:
        output_dir: Experiment output directory.
        condition_name: Name of the condition being trained.

    Returns:
        Path to the log file.
    """
    global _file_logging_initialized

    if _file_logging_initialized:
        return output_dir / "conditions" / condition_name / "train.log"

    condition_dir = output_dir / "conditions" / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{timestamp}.log"
    log_path = condition_dir / log_filename

    # Create file handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Create symlink to latest log
    latest_link = condition_dir / "train.log"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(log_path.name)
    except OSError:
        pass

    _file_logging_initialized = True

    logger.info(f"Logging to file: {log_path}")
    logger.info(f"To check for errors: grep -E 'WARNING|ERROR' {log_path}")

    return log_path


# =============================================================================
# Skip Training (Validation Only)
# =============================================================================


def _run_validation_only(
    condition_name: str,
    config: dict,
    splits: dict,
    device: str,
) -> dict[str, float]:
    """Run validation only for skip_training conditions.

    This is used for completely frozen baselines that don't need training.
    Simply loads the model, runs a single validation pass, and saves metrics.

    Args:
        condition_name: Name of condition.
        config: Full experiment configuration.
        splits: Data splits dictionary.
        device: Device to run on.

    Returns:
        Dict with validation metrics.
    """
    condition_config = get_condition_config(config, condition_name)
    training_config = config["training"]

    logger.info("=" * 60)
    logger.info("VALIDATION ONLY MODE (skip_training=True)")
    logger.info("=" * 60)

    # Set up output directory
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Create validation dataloader
    # H5 backend: val uses 192³ (100% tumor containment) at batch_size=1.
    logger.info("Creating validation dataloader...")
    h5_path = config.get("paths", {}).get("h5_file")
    _, val_loader = create_dataloaders(
        data_root=config["paths"]["data_root"],
        train_ids=splits["lora_train"][:10],  # Minimal train set (not used)
        val_ids=splits["lora_val"],
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        compute_semantic=False,
        augment_train=False,
        h5_path=h5_path,
        val_batch_size=training_config.get("val_batch_size", 1),
    )

    # Create model (completely frozen)
    logger.info("Creating completely frozen model...")
    model = create_ablation_model(
        condition_config=condition_config,
        training_config=training_config,
        checkpoint_path=config["paths"]["checkpoint"],
        device=device,
    )

    # Log parameter counts (should all be 0)
    param_counts = model.get_trainable_param_count()
    logger.info(f"Trainable parameters: {param_counts}")

    # Create losses for validation
    seg_loss_fn = SegmentationLoss3Ch(
        lambda_dice=config["loss"]["lambda_dice"],
        lambda_bce=config["loss"]["lambda_ce"],  # BCE for 3-channel sigmoid
    )
    dice_metric = DiceMetric3Ch()

    # Run single validation pass
    logger.info("Running validation pass...")
    val_metrics = validate(model, val_loader, seg_loss_fn, dice_metric, device)

    logger.info(f"Validation Dice: {val_metrics['dice_mean']:.4f}")
    logger.info(f"  TC: {val_metrics['dice_0']:.4f}")
    logger.info(f"  WT: {val_metrics['dice_1']:.4f}")
    logger.info(f"  ET: {val_metrics['dice_2']:.4f}")

    # Save summary
    summary = {
        "condition": condition_name,
        "is_baseline": True,
        "skip_training": True,
        "param_counts": param_counts,
        "best_epoch": 0,
        "best_val_dice": val_metrics["dice_mean"],
        "total_epochs": 0,
        "training_time_minutes": 0.0,
        "decoder_type": training_config.get("decoder_type", "original"),
    }
    with open(condition_dir / "training_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    # Save validation log (single entry)
    log_path = condition_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_seg_loss",
                "train_aux_loss",
                "val_loss",
                "val_dice_mean",
                "val_dice_0",
                "val_dice_1",
                "val_dice_2",
            ]
        )
        writer.writerow(
            [
                0,
                0.0,
                0.0,
                0.0,
                val_metrics["loss"],
                val_metrics["dice_mean"],
                val_metrics["dice_0"],
                val_metrics["dice_1"],
                val_metrics["dice_2"],
            ]
        )

    logger.info(f"Saved validation results to {condition_dir}")

    return {
        "epoch": 0,
        "train_loss": 0.0,
        **val_metrics,
    }


# =============================================================================
# Auxiliary Loss Warmup
# =============================================================================


def compute_lambda_aux_effective(
    epoch: int,
    lambda_aux: float,
    warmup_start: int,
    warmup_duration: int,
) -> float:
    """Compute effective lambda_aux with warmup schedule.

    The auxiliary loss is:
    - 0 for epochs < warmup_start
    - Linearly ramped from 0 to lambda_aux over warmup_duration epochs
    - lambda_aux after warmup completes

    Args:
        epoch: Current epoch (0-indexed).
        lambda_aux: Target lambda_aux value.
        warmup_start: Epoch to start auxiliary loss (0-indexed).
        warmup_duration: Number of epochs to ramp from 0 to lambda_aux.

    Returns:
        Effective lambda_aux for this epoch.
    """
    if epoch < warmup_start:
        return 0.0

    epochs_since_start = epoch - warmup_start
    if warmup_duration <= 0:
        return lambda_aux

    ramp_factor = min(1.0, epochs_since_start / warmup_duration)
    return lambda_aux * ramp_factor


# =============================================================================
# Gradient Diagnostics
# =============================================================================


def compute_gradient_norms(
    model: nn.Module,
    decoder_type: str,
    is_baseline: bool,
) -> dict[str, float]:
    """Compute gradient norms for different parameter groups.

    Args:
        model: The model after backward pass.
        decoder_type: "lightweight" or "original".
        is_baseline: Whether this is baseline condition.

    Returns:
        Dict with gradient norms for encoder, decoder, semantic heads.
    """
    norms = {}

    # Collect gradients by group
    encoder_grads = []
    decoder_grads = []
    semantic_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_flat = param.grad.flatten()

            if (
                "semantic_heads" in name
                or "volume_head" in name
                or "location_head" in name
                or "shape_head" in name
            ):
                semantic_grads.append(grad_flat)
            elif "decoder" in name:
                decoder_grads.append(grad_flat)
            elif "lora" in name.lower() or "encoder" in name:
                encoder_grads.append(grad_flat)

    # Compute norms
    if encoder_grads:
        norms["encoder_grad_norm"] = torch.norm(torch.cat(encoder_grads)).item()
    else:
        norms["encoder_grad_norm"] = 0.0

    if decoder_grads:
        norms["decoder_grad_norm"] = torch.norm(torch.cat(decoder_grads)).item()
    else:
        norms["decoder_grad_norm"] = 0.0

    if semantic_grads:
        norms["semantic_grad_norm"] = torch.norm(torch.cat(semantic_grads)).item()
    else:
        norms["semantic_grad_norm"] = 0.0

    return norms


def compute_gradient_conflict(
    model: nn.Module,
    seg_loss: torch.Tensor,
    aux_loss: torch.Tensor,
    device: str,
) -> float:
    """Compute cosine similarity between segmentation and auxiliary gradients.

    A negative value indicates conflicting gradients (tasks pulling in opposite directions).

    Args:
        model: The model.
        seg_loss: Segmentation loss tensor (requires grad).
        aux_loss: Auxiliary loss tensor (requires grad).
        device: Device.

    Returns:
        Cosine similarity between gradients (-1 to 1).
    """
    # This is expensive, so only call periodically
    # Get parameters that are affected by both losses
    params = [p for p in model.parameters() if p.requires_grad]

    # Compute gradients for each loss separately
    seg_grads = torch.autograd.grad(seg_loss, params, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(aux_loss, params, retain_graph=True, allow_unused=True)

    # Flatten and concatenate
    seg_flat = []
    aux_flat = []
    for sg, ag in zip(seg_grads, aux_grads):
        if sg is not None and ag is not None:
            seg_flat.append(sg.flatten())
            aux_flat.append(ag.flatten())

    if not seg_flat:
        return 0.0

    seg_vec = torch.cat(seg_flat)
    aux_vec = torch.cat(aux_flat)

    # Cosine similarity
    cos_sim = F.cosine_similarity(seg_vec.unsqueeze(0), aux_vec.unsqueeze(0)).item()

    return cos_sim


# =============================================================================
# Optimizer Creation
# =============================================================================


def create_optimizer(
    model: nn.Module,
    config: dict,
    is_baseline: bool,
    decoder_type: str,
) -> tuple[torch.optim.Optimizer, dict]:
    """Create optimizer with separate param groups and warmup+plateau scheduler.

    Args:
        model: The model to optimize.
        config: Full experiment configuration.
        is_baseline: Whether this is the baseline condition.
        decoder_type: "lightweight" or "original".

    Returns:
        Tuple of (optimizer, scheduler_info) where scheduler_info is a dict
        with 'warmup', 'plateau', and 'warmup_epochs' keys.
    """
    training_config = config["training"]

    if is_baseline:
        # Only decoder parameters for baseline (includes semantic heads if enabled)
        if decoder_type == "lightweight":
            params = [{"params": model.decoder.parameters(), "lr": training_config["lr_decoder"]}]
        else:  # original - use get_decoder_params() to include semantic heads
            params = [{"params": model.get_decoder_params(), "lr": training_config["lr_decoder"]}]
    else:
        # Separate groups for LoRA and decoder
        if decoder_type == "lightweight":
            encoder_params = [p for p in model.encoder.model.parameters() if p.requires_grad]
            decoder_params = list(model.decoder.parameters())
        else:  # original
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

    warmup_epochs = training_config.get("lr_warmup_epochs", 5)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        factor=training_config.get("lr_reduce_factor", 0.5),
        patience=training_config.get("lr_reduce_patience", 10),
        min_lr=1e-7,
    )

    scheduler_info = {
        'warmup': warmup_scheduler,
        'plateau': plateau_scheduler,
        'warmup_epochs': warmup_epochs,
    }

    return optimizer, scheduler_info


# =============================================================================
# Target Statistics
# =============================================================================


def compute_target_statistics(dataloader: DataLoader) -> dict[str, torch.Tensor]:
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
        "volume_mean": volumes.mean(dim=0),
        "volume_std": volumes.std(dim=0).clamp(min=1e-6),
        "location_mean": locations.mean(dim=0),
        "location_std": locations.std(dim=0).clamp(min=1e-6),
        "shape_mean": shapes.mean(dim=0),
        "shape_std": shapes.std(dim=0).clamp(min=1e-6),
    }


# =============================================================================
# Training Epoch
# =============================================================================


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    seg_loss_fn: nn.Module,
    aux_loss_fn: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_clip: float = 1.0,
    lambda_aux: float = 0.1,
    use_semantic_heads: bool = False,
    decoder_type: str = "original",
    is_baseline: bool = False,
    enable_gradient_monitoring: bool = False,
    gradient_monitor_freq: int = 50,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    vicreg_loss_fn: nn.Module | None = None,
) -> dict[str, float]:
    """Train for one epoch with optional auxiliary losses.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        seg_loss_fn: Segmentation loss function.
        aux_loss_fn: Auxiliary semantic loss function (optional).
        optimizer: Optimizer.
        device: Device to train on.
        gradient_clip: Gradient clipping value.
        lambda_aux: Weight for auxiliary loss (already adjusted for warmup).
        use_semantic_heads: Whether to use semantic heads.
        decoder_type: "lightweight" or "original".
        is_baseline: Whether this is baseline condition.
        enable_gradient_monitoring: Whether to compute gradient diagnostics.
        gradient_monitor_freq: How often to compute gradient diagnostics (batches).
        use_amp: Whether to use bf16 automatic mixed precision.
        grad_accum_steps: Number of batches to accumulate gradients over.
        vicreg_loss_fn: Optional encoder VICReg loss for feature richness.

    Returns:
        Dict with loss components and optional gradient diagnostics.
    """
    model.train()

    # For baseline, keep encoder in eval mode
    if decoder_type == "lightweight":
        if hasattr(model, "encoder") and not hasattr(model.encoder, "model"):
            model.encoder.eval()
    else:  # original
        if hasattr(model, "model") and hasattr(model.model, "encoder"):
            model.model.encoder.eval()

    # Accumulators
    total_seg_loss = 0.0
    total_aux_loss = 0.0
    total_vol_loss = 0.0
    total_loc_loss = 0.0
    total_shape_loss = 0.0
    total_vicreg_loss = 0.0
    total_vicreg_var = 0.0
    total_vicreg_cov = 0.0
    total_loss = 0.0
    total_variance_hinge = 0.0
    variance_hinge_count = 0
    num_batches = 0
    num_steps = 0

    # VICReg feature buffer for accumulation across micro-batches
    vicreg_feature_buffer: list[torch.Tensor] = []

    # Gradient monitoring accumulators
    grad_norms_sum = {"encoder_grad_norm": 0.0, "decoder_grad_norm": 0.0, "semantic_grad_norm": 0.0}
    grad_conflict_sum = 0.0
    grad_monitor_count = 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        # Zero gradients at accumulation boundaries only
        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        # Forward pass under bf16 autocast (no-op when use_amp=False)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # Use forward_with_semantics when we need features (for aux or vicreg)
            need_features = (use_semantic_heads or vicreg_loss_fn is not None)
            if need_features and hasattr(model, "forward_with_semantics"):
                outputs = model.forward_with_semantics(images)
                pred = outputs["logits"]
            else:
                pred = model(images)
                outputs = {}

            # Segmentation loss
            seg_loss = seg_loss_fn(pred, segs)
            loss = seg_loss

            # Auxiliary semantic loss (if enabled)
            aux_loss = torch.tensor(0.0, device=device)
            vol_loss = 0.0
            loc_loss = 0.0
            shape_loss = 0.0

            if (
                use_semantic_heads
                and aux_loss_fn is not None
                and "semantic_features" in batch
                and lambda_aux > 0
            ):
                semantic_targets = {
                    "volume": batch["semantic_features"]["volume"].to(device),
                    "location": batch["semantic_features"]["location"].to(device),
                    "shape": batch["semantic_features"]["shape"].to(device),
                }
                aux_loss, aux_components = aux_loss_fn(outputs, semantic_targets)
                loss = seg_loss + lambda_aux * aux_loss

                # Extract individual components
                vol_loss = aux_components.get("vol_loss", 0.0)
                loc_loss = aux_components.get("loc_loss", 0.0)
                shape_loss = aux_components.get("shape_loss", 0.0)

            # Buffer features for VICReg (computed at step boundary for better statistics)
            if vicreg_loss_fn is not None and "features" in outputs:
                vicreg_feature_buffer.append(outputs["features"])

        # Backward pass for seg+aux only (outside autocast — gradients in fp32)
        (loss / grad_accum_steps).backward()

        # Step at accumulation boundaries or last batch
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Compute VICReg on accumulated features (better statistics)
            step_vicreg_loss = 0.0
            step_vicreg_components: dict[str, float] = {}
            if vicreg_loss_fn is not None and vicreg_feature_buffer:
                # Keep gradients for last micro-batch only (standard approach)
                detached = [f.detach() for f in vicreg_feature_buffer[:-1]]
                last_with_grad = vicreg_feature_buffer[-1]
                all_features = torch.cat(detached + [last_with_grad], dim=0)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    vicreg_loss_t, step_vicreg_components = vicreg_loss_fn(all_features)

                (vicreg_loss_t / grad_accum_steps).backward()
                step_vicreg_loss = vicreg_loss_t.item()

                # Track feature dimensional health (variance hinge)
                with torch.no_grad():
                    feat_std = all_features.float().std(dim=0)
                    vh = torch.clamp(1.0 - feat_std, min=0.0).mean().item()
                    total_variance_hinge += vh
                    variance_hinge_count += 1

                vicreg_feature_buffer = []

            # Gradient monitoring (periodic)
            if enable_gradient_monitoring and (batch_idx + 1) % gradient_monitor_freq == 0:
                grad_norms = compute_gradient_norms(model, decoder_type, is_baseline)
                for key in grad_norms_sum:
                    grad_norms_sum[key] += grad_norms.get(key, 0.0)
                grad_monitor_count += 1

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

            optimizer.step()
            num_steps += 1

            # Track VICReg at step level
            total_vicreg_loss += step_vicreg_loss
            total_vicreg_var += step_vicreg_components.get("vicreg_var_loss", 0.0)
            total_vicreg_cov += step_vicreg_components.get("vicreg_cov_loss", 0.0)

        # Track UNSCALED loss (seg + aux per batch)
        total_seg_loss += seg_loss.item()
        total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        total_vol_loss += vol_loss
        total_loc_loss += loc_loss
        total_shape_loss += shape_loss
        total_loss += loss.item()
        num_batches += 1

    # Compute averages (VICReg averaged per step, not per batch)
    num_steps = max(1, num_steps)
    metrics = {
        "loss": total_loss / num_batches,
        "seg_loss": total_seg_loss / num_batches,
        "aux_loss": total_aux_loss / num_batches,
        "vol_loss": total_vol_loss / num_batches,
        "loc_loss": total_loc_loss / num_batches,
        "shape_loss": total_shape_loss / num_batches,
        "vicreg_loss": total_vicreg_loss / num_steps,
        "vicreg_var_loss": total_vicreg_var / num_steps,
        "vicreg_cov_loss": total_vicreg_cov / num_steps,
        "variance_hinge": total_variance_hinge / max(1, variance_hinge_count),
    }

    # Add gradient monitoring results
    if enable_gradient_monitoring and grad_monitor_count > 0:
        metrics["encoder_grad_norm"] = grad_norms_sum["encoder_grad_norm"] / grad_monitor_count
        metrics["decoder_grad_norm"] = grad_norms_sum["decoder_grad_norm"] / grad_monitor_count
        metrics["semantic_grad_norm"] = grad_norms_sum["semantic_grad_norm"] / grad_monitor_count

    return metrics


# =============================================================================
# Validation
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
    """Validate model.

    Args:
        model: Model to validate.
        dataloader: Validation data loader.
        seg_loss_fn: Segmentation loss function.
        dice_metric: Dice metric function.
        device: Device to validate on.
        use_amp: Whether to use bf16 automatic mixed precision.

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

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # Forward pass
            if hasattr(model, "forward_with_semantics"):
                outputs = model.forward_with_semantics(images)
                pred = outputs["logits"]
            else:
                pred = model(images)

            # Compute loss
            loss = seg_loss_fn(pred, segs)

        total_loss += loss.item()

        # Compute Dice per class (in fp32 for accuracy)
        dice_scores = dice_metric(pred.float(), segs)
        all_dice_scores.append(dice_scores.cpu())

        num_batches += 1

    # Aggregate metrics
    avg_loss = total_loss / num_batches
    dice_tensor = torch.cat(all_dice_scores, dim=0).mean(dim=0)  # [B1+B2+..., 3] -> [3]

    return {
        "loss": avg_loss,
        "dice_mean": dice_tensor.mean().item(),
        "dice_0": dice_tensor[0].item(),  # TC (Tumor Core)
        "dice_1": dice_tensor[1].item(),  # WT (Whole Tumor)
        "dice_2": dice_tensor[2].item(),  # ET (Enhancing Tumor)
    }


# =============================================================================
# Inline Feature Quality Evaluation
# =============================================================================


@torch.no_grad()
def evaluate_feature_quality_inline(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    use_amp: bool = False,
) -> dict[str, float]:
    """Quick Ridge-probe R² on validation features. ~30s overhead.

    Args:
        model: Model with forward_with_semantics method.
        dataloader: Validation data loader (must include semantic_features).
        device: Device string.
        use_amp: Whether to use bf16 autocast.

    Returns:
        Dict with probe_vol_r2, probe_loc_r2, probe_shape_r2, probe_mean_r2.
        Empty dict if insufficient data.
    """
    model.eval()
    all_features, all_vol, all_loc, all_shape = [], [], [], []

    for batch in dataloader:
        images = batch["image"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            if hasattr(model, "forward_with_semantics"):
                outputs = model.forward_with_semantics(images)
                all_features.append(outputs["features"].float().cpu().numpy())

        if "semantic_features" in batch:
            all_vol.append(batch["semantic_features"]["volume"].numpy())
            all_loc.append(batch["semantic_features"]["location"].numpy())
            all_shape.append(batch["semantic_features"]["shape"].numpy())

    if not all_features or not all_vol:
        return {}

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(np.concatenate(all_features))
    results: dict[str, float] = {}
    for name, y_list in [("vol", all_vol), ("loc", all_loc), ("shape", all_shape)]:
        Y = np.concatenate(y_list)
        n_cv = min(5, max(2, len(X)))
        scores = cross_val_score(Ridge(alpha=1.0), X, Y, cv=n_cv, scoring='r2')
        results[f"probe_{name}_r2"] = max(0.0, float(scores.mean()))

    results["probe_mean_r2"] = float(np.mean([
        results["probe_vol_r2"], results["probe_loc_r2"], results["probe_shape_r2"]
    ]))
    return results


# =============================================================================
# Checkpoint Saving
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    condition_dir: Path,
    is_baseline: bool,
    epoch: int,
    metrics: dict,
    decoder_type: str = "original",
) -> None:
    """Save model checkpoint.

    For baseline: saves decoder state dict
    For LoRA: saves LoRA adapter + decoder state
    """
    condition_dir.mkdir(parents=True, exist_ok=True)

    if is_baseline:
        # Save decoder state dict
        if decoder_type == "lightweight":
            decoder_state = model.decoder.state_dict()
        else:
            decoder_state = model.model.decoder.state_dict()

        checkpoint = {
            "epoch": epoch,
            "decoder_state_dict": decoder_state,
            "metrics": metrics,
        }
        torch.save(checkpoint, condition_dir / "checkpoint.pt")
        # Also save as best_model.pt for compatibility
        torch.save(model.state_dict(), condition_dir / "best_model.pt")
    else:
        # Save LoRA adapter
        adapter_dir = condition_dir / "adapter"

        if decoder_type == "lightweight":
            model.encoder.save_lora(adapter_dir)
            decoder_state = model.decoder.state_dict()
        else:
            model.lora_encoder.save_lora(adapter_dir)
            decoder_state = model.decoder.state_dict()

        # Save decoder and metadata
        checkpoint = {
            "epoch": epoch,
            "decoder_state_dict": decoder_state,
            "metrics": metrics,
        }
        torch.save(checkpoint, condition_dir / "checkpoint.pt")

        # Save full model state for evaluation
        torch.save(model.state_dict(), condition_dir / "best_model.pt")

    logger.info(f"Saved checkpoint at epoch {epoch}")


# =============================================================================
# Main Training Function
# =============================================================================


def train_condition(
    condition_name: str,
    config: dict,
    splits: dict,
    max_epochs: int | None = None,
    device: str = "cuda",
) -> dict[str, float]:
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
    skip_training = condition_config.get("skip_training", False)

    logger.info(f"Training condition: {condition_name}")
    logger.info(f"Description: {condition_config.get('description', 'N/A')}")

    # Handle skip_training conditions (completely frozen baseline)
    if skip_training:
        return _run_validation_only(condition_name, config, splits, device)

    # Get training options
    training_config = config["training"]
    decoder_type = training_config.get("decoder_type", "original")
    use_semantic_heads = training_config.get("use_semantic_heads", False)
    freeze_decoder = training_config.get("freeze_decoder", False)

    # Per-condition lambda_aux override (for v3 ablation)
    lambda_aux = condition_config.get(
        "lambda_aux_override", training_config.get("lambda_aux", 0.1)
    )

    # Auxiliary loss warmup parameters
    aux_warmup_start = training_config.get("aux_warmup_epochs", 0)
    aux_warmup_duration = training_config.get("aux_warmup_duration", 10)

    # Gradient monitoring options
    enable_gradient_monitoring = training_config.get("enable_gradient_monitoring", False)
    gradient_monitor_freq = training_config.get("gradient_monitor_freq", 50)

    # Mixed precision and gradient accumulation (backward-compatible defaults)
    use_amp = training_config.get("use_amp", False)
    grad_accum_steps = training_config.get("grad_accum_steps", 1)

    logger.info(f"Decoder type: {decoder_type}")
    logger.info(f"Use semantic heads: {use_semantic_heads}")
    logger.info(f"Freeze decoder: {freeze_decoder}")
    logger.info(f"Lambda aux: {lambda_aux}")
    if use_amp:
        logger.info("Mixed precision: bf16 (torch.autocast)")
    if grad_accum_steps > 1:
        logger.info(f"Gradient accumulation: {grad_accum_steps} steps")
    if use_semantic_heads:
        logger.info(f"Aux warmup: start={aux_warmup_start}, duration={aux_warmup_duration}")
    if enable_gradient_monitoring:
        logger.info(f"Gradient monitoring enabled (freq={gradient_monitor_freq})")

    # Set up output directory
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders (with semantic features if using aux loss)
    # H5 backend: val uses 192³ (100% tumor containment) at batch_size=1.
    # Training uses 128³ random crops (stochastic coverage, matches BSF).
    logger.info("Creating dataloaders...")
    h5_path = config.get("paths", {}).get("h5_file")
    train_loader, val_loader = create_dataloaders(
        data_root=config["paths"]["data_root"],
        train_ids=splits["lora_train"],
        val_ids=splits["lora_val"],
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        compute_semantic=use_semantic_heads,
        augment_train=True,
        h5_path=h5_path,
        val_batch_size=training_config.get("val_batch_size", 1),
    )

    # Create model using the unified factory
    logger.info(f"Creating model with {decoder_type.upper()} decoder...")
    model = create_ablation_model(
        condition_config=condition_config,
        training_config=training_config,
        checkpoint_path=config["paths"]["checkpoint"],
        device=device,
    )

    # Log parameter counts
    param_counts = model.get_trainable_param_count()
    if is_baseline:
        logger.info(
            f"Trainable parameters (baseline - encoder frozen): "
            f"encoder={param_counts.get('encoder', 0):,}, "
            f"decoder={param_counts.get('decoder', 0):,}, "
            f"semantic_heads={param_counts.get('semantic_heads', 0):,}, "
            f"total={param_counts.get('total', 0):,}"
        )
    else:
        logger.info(
            f"Trainable parameters (LoRA): "
            f"encoder_lora={param_counts.get('encoder_lora', param_counts.get('encoder', 0)):,}, "
            f"decoder={param_counts.get('decoder', 0):,}, "
            f"semantic_heads={param_counts.get('semantic_heads', 0):,}, "
            f"total={param_counts.get('total', 0):,}"
        )

    # Create optimizer and scheduler
    optimizer, scheduler_info = create_optimizer(model, config, is_baseline, decoder_type)

    # Create losses
    seg_loss_fn = SegmentationLoss3Ch(
        lambda_dice=config["loss"]["lambda_dice"],
        lambda_bce=config["loss"]["lambda_ce"],  # BCE for 3-channel sigmoid
    )
    dice_metric = DiceMetric3Ch()

    # Auxiliary semantic loss (if enabled, for original decoder - both baseline and LoRA)
    aux_loss_fn = None
    if use_semantic_heads and decoder_type == "original":
        aux_loss_fn = AuxiliarySemanticLoss(
            lambda_volume=config["loss"].get("lambda_volume", 1.0),
            lambda_location=config["loss"].get("lambda_location", 1.0),
            lambda_shape=config["loss"].get("lambda_shape", 0.5),  # Reduced for harder task
            normalize_targets=True,
        )

        # Compute target statistics for normalization
        logger.info("Computing target statistics...")
        stats = compute_target_statistics(train_loader)
        if stats:
            # Use .copy_() to preserve registered buffer status
            aux_loss_fn.volume_mean.copy_(stats["volume_mean"])
            aux_loss_fn.volume_std.copy_(stats["volume_std"])
            aux_loss_fn.location_mean.copy_(stats["location_mean"])
            aux_loss_fn.location_std.copy_(stats["location_std"])
            aux_loss_fn.shape_mean.copy_(stats["shape_mean"])
            aux_loss_fn.shape_std.copy_(stats["shape_std"])
            aux_loss_fn._stats_initialized = True
            logger.info(
                f"Target normalization stats: "
                f"vol_std={stats['volume_std'].mean():.2f}, "
                f"loc_std={stats['location_std'].mean():.2f}, "
                f"shape_std={stats['shape_std'].mean():.2f}"
            )

    # VICReg encoder feature richness loss (optional, per-condition toggle)
    vicreg_loss_fn = None
    use_vicreg = condition_config.get("use_vicreg", False)
    lambda_var_enc = training_config.get("lambda_var_enc", 5.0) if use_vicreg else 0.0
    lambda_cov_enc = training_config.get("lambda_cov_enc", 1.0) if use_vicreg else 0.0
    if use_vicreg:
        vicreg_gamma = training_config.get("vicreg_gamma", 1.0)
        vicreg_loss_fn = EncoderVICRegLoss(
            lambda_var=lambda_var_enc,
            lambda_cov=lambda_cov_enc,
            gamma=vicreg_gamma,
        )
        logger.info(
            f"Encoder VICReg loss enabled: lambda_var={lambda_var_enc}, "
            f"lambda_cov={lambda_cov_enc}, gamma={vicreg_gamma}"
        )

    # Training configuration
    epochs = max_epochs or training_config["max_epochs"]
    patience = training_config["early_stopping_patience"]
    gradient_clip = training_config.get("gradient_clip", 1.0)

    # Training log with extended columns
    log_path = condition_dir / "training_log.csv"
    csv_columns = [
        "epoch",
        "train_loss",
        "train_seg_loss",
        "train_aux_loss",
        "train_vol_loss",
        "train_loc_loss",
        "train_shape_loss",
        "train_vicreg_loss",
        "train_vicreg_var",
        "train_vicreg_cov",
        "val_loss",
        "val_dice_mean",
        "val_dice_0",
        "val_dice_1",
        "val_dice_2",
        "lr",
        "lambda_aux_eff",
        "variance_hinge",
        "probe_vol_r2",
        "probe_loc_r2",
        "probe_shape_r2",
        "probe_mean_r2",
        "checkpoint_score",
    ]
    if enable_gradient_monitoring:
        csv_columns.extend(["encoder_grad_norm", "decoder_grad_norm", "semantic_grad_norm"])

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)

    # Training loop
    best_score = 0.0
    best_dice = 0.0
    best_metrics = {}
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Compute effective lambda_aux with warmup
        lambda_aux_eff = compute_lambda_aux_effective(
            epoch, lambda_aux, aux_warmup_start, aux_warmup_duration
        )

        if use_semantic_heads and epoch < aux_warmup_start + aux_warmup_duration:
            logger.info(f"  Aux loss warmup: lambda_aux_eff = {lambda_aux_eff:.4f}")

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            seg_loss_fn,
            aux_loss_fn,
            optimizer,
            device,
            gradient_clip,
            lambda_aux_eff,
            use_semantic_heads=use_semantic_heads and decoder_type == "original",
            decoder_type=decoder_type,
            is_baseline=is_baseline,
            enable_gradient_monitoring=enable_gradient_monitoring,
            gradient_monitor_freq=gradient_monitor_freq,
            use_amp=use_amp,
            grad_accum_steps=grad_accum_steps,
            vicreg_loss_fn=vicreg_loss_fn,
        )

        # Validate
        val_metrics = validate(model, val_loader, seg_loss_fn, dice_metric, device, use_amp=use_amp)

        # Update scheduler (warmup then plateau)
        if epoch < scheduler_info['warmup_epochs']:
            scheduler_info['warmup'].step()
        else:
            scheduler_info['plateau'].step(val_metrics['dice_mean'])
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics with per-component losses
        log_msg = (
            f"  Train Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f}, aux={train_metrics['aux_loss']:.4f}"
        )
        if use_semantic_heads and train_metrics["aux_loss"] > 0:
            log_msg += (
                f" [vol={train_metrics['vol_loss']:.4f}, "
                f"loc={train_metrics['loc_loss']:.4f}, "
                f"shape={train_metrics['shape_loss']:.4f}]"
            )
        log_msg += ")"
        if use_vicreg and train_metrics["vicreg_loss"] > 0:
            log_msg += (
                f" | VICReg: {train_metrics['vicreg_loss']:.4f} "
                f"(var={train_metrics['vicreg_var_loss']:.4f}, "
                f"cov={train_metrics['vicreg_cov_loss']:.4f})"
            )
        log_msg += (
            f" | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice_mean']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        logger.info(log_msg)

        # Log gradient norms if monitoring
        if enable_gradient_monitoring and "encoder_grad_norm" in train_metrics:
            logger.info(
                f"  Grad norms: encoder={train_metrics['encoder_grad_norm']:.2e}, "
                f"decoder={train_metrics['decoder_grad_norm']:.2e}, "
                f"semantic={train_metrics['semantic_grad_norm']:.2e}"
            )

        # Inline feature probes (every epoch — ~30s overhead)
        probe_metrics: dict[str, float] = {}
        if use_semantic_heads:
            probe_metrics = evaluate_feature_quality_inline(model, val_loader, device, use_amp)
            model.train()  # Restore training mode
            if probe_metrics:
                logger.info(
                    f"  Probe R²: vol={probe_metrics['probe_vol_r2']:.3f}, "
                    f"loc={probe_metrics['probe_loc_r2']:.3f}, "
                    f"shape={probe_metrics['probe_shape_r2']:.3f}"
                )

        # Checkpoint score: probe_mean_r2 when available, else fall back to dice
        checkpoint_score = probe_metrics.get("probe_mean_r2", val_metrics["dice_mean"])

        # Save to CSV
        csv_row = [
            epoch + 1,
            train_metrics["loss"],
            train_metrics["seg_loss"],
            train_metrics["aux_loss"],
            train_metrics["vol_loss"],
            train_metrics["loc_loss"],
            train_metrics["shape_loss"],
            train_metrics["vicreg_loss"],
            train_metrics["vicreg_var_loss"],
            train_metrics["vicreg_cov_loss"],
            val_metrics["loss"],
            val_metrics["dice_mean"],
            val_metrics["dice_0"],
            val_metrics["dice_1"],
            val_metrics["dice_2"],
            current_lr,
            lambda_aux_eff,
            train_metrics.get("variance_hinge", ""),
            probe_metrics.get("probe_vol_r2", ""),
            probe_metrics.get("probe_loc_r2", ""),
            probe_metrics.get("probe_shape_r2", ""),
            probe_metrics.get("probe_mean_r2", ""),
            checkpoint_score,
        ]
        if enable_gradient_monitoring:
            csv_row.extend(
                [
                    train_metrics.get("encoder_grad_norm", 0.0),
                    train_metrics.get("decoder_grad_norm", 0.0),
                    train_metrics.get("semantic_grad_norm", 0.0),
                ]
            )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)

        # Checkpoint selection (probe R² only; dice fallback for non-semantic conditions)
        if checkpoint_score > best_score:
            best_score = checkpoint_score
            best_dice = val_metrics["dice_mean"]
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                **val_metrics,
                **probe_metrics,
            }
            save_checkpoint(
                model, condition_dir, is_baseline, epoch + 1, best_metrics, decoder_type
            )
            patience_counter = 0
            logger.info(
                f"  New best! probe_mean_r2={checkpoint_score:.4f} "
                f"(dice={best_dice:.4f})"
            )
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
        if decoder_type == "lightweight":
            model.encoder.save_lora(adapter_dir, model_card_config=model_card_cfg)
        else:
            model.lora_encoder.save_lora(adapter_dir, model_card_config=model_card_cfg)
        logger.info("Model card generated for LoRA adapter")

    # Save final summary
    summary = {
        "condition": condition_name,
        "is_baseline": is_baseline,
        "param_counts": param_counts,
        "best_epoch": best_metrics["epoch"],
        "best_val_dice": best_dice,
        "total_epochs": epoch + 1,
        "training_time_minutes": total_time / 60,
        "decoder_type": decoder_type,
        "use_semantic_heads": use_semantic_heads,
        "freeze_decoder": freeze_decoder,
        "aux_warmup_start": aux_warmup_start,
        "aux_warmup_duration": aux_warmup_duration,
        "use_amp": use_amp,
        "grad_accum_steps": grad_accum_steps,
    }
    with open(condition_dir / "training_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    return best_metrics


# =============================================================================
# Entry Point
# =============================================================================


def main(
    config_path: str,
    condition: str,
    max_epochs: int | None = None,
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

    # Set up file logging
    output_dir = Path(config["experiment"]["output_dir"])
    setup_file_logging(output_dir, condition)

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
    parser = argparse.ArgumentParser(description="Train a single LoRA ablation condition")
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
        help="Condition to train (e.g., baseline_frozen, baseline, lora_r2, lora_r4, lora_r8, lora_r16, lora_r32, dora_r4, dora_r8, dora_r16)",
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
