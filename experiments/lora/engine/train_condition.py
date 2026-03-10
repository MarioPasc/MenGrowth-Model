#!/usr/bin/env python
# experiments/lora/engine/train_condition.py
"""Unified training loop for LoRA adaptation (single-domain and dual-domain).

Handles both single-domain (MEN-only) and dual-domain (MEN+GLI) training via
the `dual_domain` flag in the condition config. Supports:
- Domain-aware label conversion (MEN vs GLI)
- Dual-domain DataLoader (ConcatDataset + WeightedRandomSampler)
- Per-domain validation and training loss tracking
- Combined early stopping metric
- Optional VICReg encoder regularization
- Optional auxiliary semantic prediction losses with warmup
- Gradient monitoring and diagnostics
- Mixed precision (bf16) and gradient accumulation

Usage:
    python -m experiments.lora.run --config <yaml> train --condition <name>
"""

import argparse
import contextlib
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

_INTERACTIVE = sys.stderr.isatty()

from growth.data.bratsmendata import create_dataloaders
from growth.data.dual_domain import (
    create_dual_domain_train_loader,
    create_per_domain_val_loaders,
)
from growth.data.transforms import DEFAULT_ROI_SIZE
from growth.losses.encoder_vicreg import EncoderVICRegLoss
from growth.losses.segmentation import DiceMetric3Ch, SegmentationLoss3Ch
from growth.models.segmentation.semantic_heads import AuxiliarySemanticLoss
from growth.utils.seed import set_seed

from .model_factory import create_ablation_model, get_condition_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_file_logging_initialized = False


# =============================================================================
# File Logging
# =============================================================================


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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{timestamp}.log"
    log_path = condition_dir / log_filename

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

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
    norms: dict[str, float] = {}

    encoder_grads: list[torch.Tensor] = []
    decoder_grads: list[torch.Tensor] = []
    semantic_grads: list[torch.Tensor] = []

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

    A negative value indicates conflicting gradients (tasks pulling in
    opposite directions).

    Args:
        model: The model.
        seg_loss: Segmentation loss tensor (requires grad).
        aux_loss: Auxiliary loss tensor (requires grad).
        device: Device.

    Returns:
        Cosine similarity between gradients (-1 to 1).
    """
    params = [p for p in model.parameters() if p.requires_grad]

    seg_grads = torch.autograd.grad(seg_loss, params, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(aux_loss, params, retain_graph=True, allow_unused=True)

    seg_flat: list[torch.Tensor] = []
    aux_flat: list[torch.Tensor] = []
    for sg, ag in zip(seg_grads, aux_grads):
        if sg is not None and ag is not None:
            seg_flat.append(sg.flatten())
            aux_flat.append(ag.flatten())

    if not seg_flat:
        return 0.0

    seg_vec = torch.cat(seg_flat)
    aux_vec = torch.cat(aux_flat)

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
        if decoder_type == "lightweight":
            params = [{"params": model.decoder.parameters(), "lr": training_config["lr_decoder"]}]
        else:
            params = [{"params": model.get_decoder_params(), "lr": training_config["lr_decoder"]}]
    else:
        if decoder_type == "lightweight":
            encoder_params = [p for p in model.encoder.model.parameters() if p.requires_grad]
            decoder_params = list(model.decoder.parameters())
        else:
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
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training_config.get("lr_reduce_factor", 0.5),
        patience=training_config.get("lr_reduce_patience", 10),
        min_lr=1e-7,
    )

    scheduler_info = {
        "warmup": warmup_scheduler,
        "plateau": plateau_scheduler,
        "warmup_epochs": warmup_epochs,
    }

    return optimizer, scheduler_info


# =============================================================================
# Target Statistics
# =============================================================================


def compute_target_statistics(dataloader: DataLoader) -> dict[str, torch.Tensor]:
    """Compute mean and std of volume target for normalization."""
    all_volumes: list[torch.Tensor] = []

    for batch in dataloader:
        if "semantic_features" in batch:
            all_volumes.append(batch["semantic_features"]["volume"])

    if not all_volumes:
        return {}

    volumes = torch.cat(all_volumes, dim=0)

    return {
        "volume_mean": volumes.mean(dim=0),
        "volume_std": volumes.std(dim=0).clamp(min=1e-6),
    }


# =============================================================================
# Inline Feature Quality Evaluation
# =============================================================================


def evaluate_feature_quality_inline(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    use_amp: bool = False,
) -> dict[str, float]:
    """Quick GP-Linear probe R² on validation features. ~30s overhead.

    Uses a lightweight GP with linear kernel (mathematically equivalent to
    Ridge regression but consistent with our GP-based evaluation pipeline).

    Args:
        model: Model with forward_with_semantics method.
        dataloader: Validation data loader (must include semantic_features).
        device: Device string.
        use_amp: Whether to use bf16 autocast.

    Returns:
        Dict with probe_vol_r2, probe_mean_r2.
        Empty dict if insufficient data.
    """
    model.eval()
    all_features: list[np.ndarray] = []
    all_vol: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                if hasattr(model, "forward_with_semantics"):
                    outputs = model.forward_with_semantics(images)
                    all_features.append(outputs["features"].float().cpu().numpy())

            if "semantic_features" in batch:
                all_vol.append(batch["semantic_features"]["volume"].numpy())

    if not all_features or not all_vol:
        return {}

    from growth.evaluation.gp_probes import GPProbe

    X = np.concatenate(all_features)
    results: dict[str, float] = {}
    Y = np.concatenate(all_vol)
    probe = GPProbe(
        kernel_type="linear",
        normalize_features=True,
        normalize_targets=True,
        r2_ci_samples=0,
    )
    probe.fit(X, Y)
    gp_results = probe.evaluate(X, Y)
    results["probe_vol_r2"] = max(0.0, gp_results.r2)
    results["probe_mean_r2"] = results["probe_vol_r2"]
    return results


# =============================================================================
# Per-Epoch Diagnostics (domain gap + extended probes)
# =============================================================================


@torch.no_grad()
def _extract_val_features(
    model: nn.Module,
    val_loaders: dict[str, DataLoader],
    device: str,
    use_amp: bool = False,
) -> dict[str, np.ndarray]:
    """Extract encoder10 features from all validation loaders.

    Runs the encoder through each val loader in eval mode and returns
    GAP-pooled 768-dim features per domain. Used for per-epoch domain
    gap tracking — lightweight, no disk I/O.

    Args:
        model: Model instance.
        val_loaders: Dict of domain → DataLoader (e.g. {"men": ..., "gli": ...}).
        device: Device string.
        use_amp: Whether to use bf16 autocast.

    Returns:
        Dict of domain → features array [N, 768].
    """
    model.eval()
    domain_features: dict[str, np.ndarray] = {}

    for domain, loader in val_loaders.items():
        batched: list[np.ndarray] = []
        for batch in loader:
            images = batch["image"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Get encoder10 features via hidden states → encoder10 → GAP
                raw = model
                if hasattr(raw, "model") and hasattr(raw.model, "get_hidden_states"):
                    hidden_states = raw.model.get_hidden_states(images)
                    enc10 = raw.model.decoder.encoder10(hidden_states[4])
                elif hasattr(raw, "lora_encoder"):
                    hidden_states = raw.lora_encoder.get_hidden_states(images)
                    enc10 = raw.decoder.encoder10(hidden_states[4])
                elif hasattr(raw, "encoder"):
                    hidden_states = raw.encoder.swinViT(images, raw.encoder.normalize)
                    enc10 = raw.encoder.encoder10(hidden_states[4])
                else:
                    logger.debug("Cannot extract features: unknown model type")
                    return {}

                features = F.adaptive_avg_pool3d(enc10, 1).flatten(1)  # [B, 768]
            batched.append(features.float().cpu().numpy())

        if batched:
            domain_features[domain] = np.concatenate(batched, axis=0)

    return domain_features


def compute_epoch_diagnostics(
    model: nn.Module,
    val_loaders: dict[str, DataLoader],
    device: str,
    use_amp: bool = False,
) -> dict[str, float]:
    """Compute domain gap and feature quality metrics for the current epoch.

    Extracts encoder10 features from all val domains and computes:
    - MMD² between MEN and GLI (if both available)
    - PAD (Proxy A-Distance)
    - Per-domain effective rank
    - Domain classifier accuracy

    Args:
        model: Model instance.
        val_loaders: Dict of domain → DataLoader.
        device: Device string.
        use_amp: Whether to use bf16 autocast.

    Returns:
        Dict with diagnostic metrics (prefixed with "diag_").
    """
    from growth.evaluation.latent_quality import (
        compute_domain_classifier_accuracy,
        compute_effective_rank,
        compute_proxy_a_distance,
        mmd_permutation_test,
    )

    domain_features = _extract_val_features(model, val_loaders, device, use_amp)

    if not domain_features:
        return {}

    results: dict[str, float] = {}

    # Per-domain effective rank
    for domain, feats in domain_features.items():
        results[f"diag_{domain}_effective_rank"] = compute_effective_rank(feats)

    # Domain gap (MEN vs GLI)
    men_feats = domain_features.get("men")
    gli_feats = domain_features.get("gli")

    if men_feats is not None and gli_feats is not None:
        # MMD² with fewer permutations for speed (50 vs 200 in final eval)
        mmd_val, mmd_pval = mmd_permutation_test(men_feats, gli_feats, n_perm=50)
        results["diag_mmd_squared"] = mmd_val
        results["diag_mmd_pvalue"] = mmd_pval

        # PAD
        domain_acc = compute_domain_classifier_accuracy(men_feats, gli_feats)
        pad = compute_proxy_a_distance(men_feats, gli_feats)
        results["diag_domain_classifier_acc"] = domain_acc
        results["diag_pad"] = pad

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

    For baseline: saves decoder state dict.
    For LoRA: saves LoRA adapter + decoder state.

    Args:
        model: Model to save.
        condition_dir: Condition output directory.
        is_baseline: Whether this is a baseline condition.
        epoch: Current epoch number.
        metrics: Current best metrics.
        decoder_type: "lightweight" or "original".
    """
    condition_dir.mkdir(parents=True, exist_ok=True)

    if is_baseline:
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
        torch.save(model.state_dict(), condition_dir / "best_model.pt")
    else:
        adapter_dir = condition_dir / "adapter"

        if decoder_type == "lightweight":
            model.encoder.save_lora(adapter_dir)
            decoder_state = model.decoder.state_dict()
        else:
            model.lora_encoder.save_lora(adapter_dir)
            decoder_state = model.decoder.state_dict()

        checkpoint = {
            "epoch": epoch,
            "decoder_state_dict": decoder_state,
            "metrics": metrics,
        }
        torch.save(checkpoint, condition_dir / "checkpoint.pt")
        torch.save(model.state_dict(), condition_dir / "best_model.pt")

    logger.info(f"Saved checkpoint at epoch {epoch}")


# =============================================================================
# Validation Only (Frozen Baseline)
# =============================================================================


def _run_validation_only(
    condition_name: str,
    config: dict,
    device: str,
) -> dict[str, float]:
    """Run validation only for skip_training conditions (frozen baseline).

    Supports both single-domain (MEN-only via h5_file) and dual-domain
    (MEN+GLI via men_h5_file/gli_h5_file) configs.

    Args:
        condition_name: Condition name.
        config: Full experiment configuration.
        device: Device to run on.

    Returns:
        Dict with validation metrics.
    """
    condition_config = get_condition_config(config, condition_name)
    training_config = config["training"]

    logger.info("=" * 60)
    logger.info("VALIDATION ONLY MODE (skip_training=True)")
    logger.info("=" * 60)

    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_ablation_model(
        condition_config=condition_config,
        training_config=training_config,
        checkpoint_path=config["paths"]["checkpoint"],
        device=device,
    )

    seg_loss_fn = SegmentationLoss3Ch(
        lambda_dice=config["loss"]["lambda_dice"],
        lambda_bce=config["loss"]["lambda_ce"],
    )
    dice_metric = DiceMetric3Ch()

    # Determine if dual-domain config
    is_dual = "men_h5_file" in config["paths"] and "gli_h5_file" in config["paths"]

    if is_dual:
        val_loaders = create_per_domain_val_loaders(
            men_h5_path=config["paths"]["men_h5_file"],
            gli_h5_path=config["paths"]["gli_h5_file"],
            batch_size=training_config.get("val_batch_size", 1),
            num_workers=training_config["num_workers"],
            roi_size=DEFAULT_ROI_SIZE,
            compute_semantic=False,
        )
    else:
        # Single-domain: create MEN-only val loader
        _, val_loader = create_dataloaders(
            h5_path=config["paths"]["h5_file"],
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
            compute_semantic=False,
            augment_train=False,
            val_batch_size=training_config.get("val_batch_size", 1),
            val_roi_size=DEFAULT_ROI_SIZE,
        )
        val_loaders = {"men": val_loader}

    all_metrics: dict[str, float] = {}
    for domain_name, loader in val_loaders.items():
        domain_metrics = validate_single_domain(
            model,
            loader,
            seg_loss_fn,
            dice_metric,
            device,
            domain=domain_name.upper(),
        )
        for k, v in domain_metrics.items():
            all_metrics[f"{domain_name}_{k}"] = v
        logger.info(
            f"  {domain_name.upper()} Dice: {domain_metrics['dice_mean']:.4f} "
            f"(TC={domain_metrics['dice_tc']:.4f}, "
            f"WT={domain_metrics['dice_wt']:.4f}, "
            f"ET={domain_metrics['dice_et']:.4f})"
        )

    all_metrics["combined_dice_mean"] = float(
        np.mean(
            [all_metrics[f"{d}_dice_mean"] for d in val_loaders if f"{d}_dice_mean" in all_metrics]
        )
    )

    summary = {
        "condition": condition_name,
        "skip_training": True,
        "best_epoch": 0,
        "total_epochs": 0,
        **{f"val_{k}": v for k, v in all_metrics.items()},
    }
    with open(condition_dir / "training_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    return all_metrics


# =============================================================================
# Per-Domain Validation
# =============================================================================


@torch.no_grad()
def validate_single_domain(
    model: nn.Module,
    dataloader: DataLoader,
    seg_loss_fn: nn.Module,
    dice_metric: nn.Module,
    device: str,
    domain: str = "MEN",
    use_amp: bool = False,
) -> dict[str, float]:
    """Validate model on a single-domain dataset.

    Args:
        model: Model to validate.
        dataloader: Single-domain validation DataLoader.
        seg_loss_fn: Segmentation loss function.
        dice_metric: Dice metric.
        device: Device.
        domain: Domain string ("MEN" or "GLI").
        use_amp: Whether to use bf16 autocast.

    Returns:
        Dict with loss, dice_mean, dice_tc, dice_wt, dice_et.
    """
    model.eval()
    total_loss = 0.0
    all_dice_scores: list[torch.Tensor] = []
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Val-{domain}", leave=False, disable=not _INTERACTIVE):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(images)

            loss = seg_loss_fn(pred, segs, domain=domain)

        total_loss += loss.item()
        dice_scores = dice_metric(pred.float(), segs, domain=domain)
        all_dice_scores.append(dice_scores.cpu())
        num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "dice_mean": 0.0, "dice_tc": 0.0, "dice_wt": 0.0, "dice_et": 0.0}

    avg_loss = total_loss / num_batches
    dice_tensor = torch.cat(all_dice_scores, dim=0).mean(dim=0)  # [3]

    return {
        "loss": avg_loss,
        "dice_mean": dice_tensor.mean().item(),
        "dice_tc": dice_tensor[0].item(),
        "dice_wt": dice_tensor[1].item(),
        "dice_et": dice_tensor[2].item(),
    }


def validate_dual_domain(
    model: nn.Module,
    val_loaders: dict[str, DataLoader],
    seg_loss_fn: nn.Module,
    dice_metric: nn.Module,
    device: str,
    use_amp: bool = False,
) -> dict[str, float]:
    """Validate on all available domains and compute combined metric.

    Args:
        model: Model to validate.
        val_loaders: Dict of domain -> DataLoader.
        seg_loss_fn: Segmentation loss function.
        dice_metric: Dice metric.
        device: Device.
        use_amp: Whether to use bf16 autocast.

    Returns:
        Dict with per-domain and combined metrics.
    """
    all_metrics: dict[str, float] = {}

    for domain_name, loader in val_loaders.items():
        domain_metrics = validate_single_domain(
            model,
            loader,
            seg_loss_fn,
            dice_metric,
            device,
            domain=domain_name.upper(),
            use_amp=use_amp,
        )
        for k, v in domain_metrics.items():
            all_metrics[f"{domain_name}_{k}"] = v

    domain_dices = [
        all_metrics[f"{d}_dice_mean"] for d in val_loaders.keys() if f"{d}_dice_mean" in all_metrics
    ]
    all_metrics["combined_dice_mean"] = float(np.mean(domain_dices)) if domain_dices else 0.0

    return all_metrics


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
    show_progress: bool = True,
) -> dict[str, float]:
    """Train for one epoch with domain-aware loss.

    Handles both single-domain and dual-domain batches. When domain info
    is present in the batch, per-domain loss tracking is enabled.

    Args:
        model: Model to train.
        dataloader: Training DataLoader (mixed or single domain).
        seg_loss_fn: Segmentation loss (domain-aware).
        aux_loss_fn: Auxiliary semantic loss (optional).
        optimizer: Optimizer.
        device: Device.
        gradient_clip: Gradient clipping value.
        lambda_aux: Effective lambda_aux (after warmup).
        use_semantic_heads: Whether semantic heads are active.
        decoder_type: "lightweight" or "original".
        is_baseline: Whether this is baseline condition.
        enable_gradient_monitoring: Compute gradient norms.
        gradient_monitor_freq: Frequency of gradient monitoring.
        use_amp: bf16 autocast.
        grad_accum_steps: Gradient accumulation steps.
        vicreg_loss_fn: Optional VICReg loss.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        Dict with per-component loss metrics including per-domain breakdowns.
    """
    model.train()

    # Keep encoder in eval for baseline
    raw = model
    if decoder_type == "original":
        if hasattr(raw, "model") and hasattr(raw.model, "encoder"):
            raw.model.encoder.eval()

    # Accumulators
    total_seg_loss = 0.0
    total_aux_loss = 0.0
    total_vicreg_loss = 0.0
    total_vicreg_var = 0.0
    total_vicreg_cov = 0.0
    total_loss = 0.0
    total_variance_hinge = 0.0
    variance_hinge_count = 0
    num_batches = 0
    num_steps = 0

    # Per-domain loss tracking
    domain_seg_loss: dict[str, float] = {"MEN": 0.0, "GLI": 0.0}
    domain_count: dict[str, int] = {"MEN": 0, "GLI": 0}

    vicreg_feature_buffer: list[torch.Tensor] = []

    # Gradient monitoring
    grad_norms_sum = {"encoder_grad_norm": 0.0, "decoder_grad_norm": 0.0, "semantic_grad_norm": 0.0}
    grad_monitor_count = 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_batches = len(dataloader)
    log_interval = max(1, n_batches // 4)

    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="Training", leave=False, disable=not show_progress)
    ):
        images = batch["image"].to(device)
        segs = batch["seg"].to(device)
        domains = batch.get("domain")  # str, list[str], or None

        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        is_step_boundary = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == n_batches

        with contextlib.nullcontext():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                need_features = use_semantic_heads or vicreg_loss_fn is not None
                if need_features and hasattr(model, "forward_with_semantics"):
                    outputs = model(images, return_semantics=True)
                    pred = outputs["logits"]
                else:
                    pred = model(images)
                    outputs = {}

                # Domain-aware segmentation loss
                seg_loss = seg_loss_fn(pred, segs, domain=domains)
                loss = seg_loss

                # Track per-domain seg loss
                if isinstance(domains, (list, tuple)):
                    for i, d in enumerate(domains):
                        d_key = d if d in domain_seg_loss else "MEN"
                        domain_seg_loss[d_key] += seg_loss.item()
                        domain_count[d_key] += 1
                elif isinstance(domains, str):
                    domain_seg_loss[domains] += seg_loss.item() * images.shape[0]
                    domain_count[domains] += images.shape[0]

                # Auxiliary semantic loss
                aux_loss = torch.tensor(0.0, device=device)
                if (
                    use_semantic_heads
                    and aux_loss_fn is not None
                    and "semantic_features" in batch
                    and lambda_aux > 0
                ):
                    semantic_targets = {
                        "volume": batch["semantic_features"]["volume"].to(device),
                    }
                    aux_loss, _ = aux_loss_fn(outputs, semantic_targets)
                    loss = seg_loss + lambda_aux * aux_loss

                # Buffer detached features for VICReg variance stats
                if vicreg_loss_fn is not None and "features" in outputs:
                    vicreg_feature_buffer.append(outputs["features"].detach())

                # On step boundary, compute VICReg and add to total loss.
                # Only the current micro-batch features retain grad; earlier
                # ones are detached (VICReg regularizes batch statistics but
                # backprops through the last micro-batch only).
                step_vicreg_loss = 0.0
                step_vicreg_components: dict[str, float] = {}
                if vicreg_loss_fn is not None and is_step_boundary and "features" in outputs:
                    current_features = outputs["features"]  # retains grad
                    if len(vicreg_feature_buffer) > 1:
                        all_features = torch.cat(
                            vicreg_feature_buffer[:-1] + [current_features], dim=0
                        )
                    else:
                        all_features = current_features

                    vicreg_loss_t, step_vicreg_components = vicreg_loss_fn(all_features)
                    loss = loss + vicreg_loss_t
                    step_vicreg_loss = vicreg_loss_t.item()

            # Single backward per micro-batch
            (loss / grad_accum_steps).backward()

        if is_step_boundary:
            # VICReg variance tracking
            if vicreg_loss_fn is not None and vicreg_feature_buffer:
                with torch.no_grad():
                    all_feats_for_stats = torch.cat(vicreg_feature_buffer, dim=0)
                    feat_std = all_feats_for_stats.float().std(dim=0)
                    vh = torch.clamp(1.0 - feat_std, min=0.0).mean().item()
                    total_variance_hinge += vh
                    variance_hinge_count += 1

                vicreg_feature_buffer = []

            # Gradient monitoring
            if enable_gradient_monitoring and (batch_idx + 1) % gradient_monitor_freq == 0:
                grad_norms = compute_gradient_norms(model, decoder_type, is_baseline)
                for key in grad_norms_sum:
                    grad_norms_sum[key] += grad_norms.get(key, 0.0)
                grad_monitor_count += 1

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

            optimizer.step()
            num_steps += 1

            total_vicreg_loss += step_vicreg_loss
            total_vicreg_var += step_vicreg_components.get("vicreg_var_loss", 0.0)
            total_vicreg_cov += step_vicreg_components.get("vicreg_cov_loss", 0.0)

        total_seg_loss += seg_loss.item()
        total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        total_loss += loss.item()
        num_batches += 1

        if not _INTERACTIVE and (batch_idx + 1) % log_interval == 0:
            pct = 100 * (batch_idx + 1) / n_batches
            avg_loss = total_loss / num_batches
            logger.info(f"  [batch {batch_idx + 1}/{n_batches} ({pct:.0f}%)] loss={avg_loss:.4f}")

    num_steps = max(1, num_steps)
    metrics: dict[str, float] = {
        "loss": total_loss / max(1, num_batches),
        "seg_loss": total_seg_loss / max(1, num_batches),
        "aux_loss": total_aux_loss / max(1, num_batches),
        "vicreg_loss": total_vicreg_loss / num_steps,
        "vicreg_var_loss": total_vicreg_var / num_steps,
        "vicreg_cov_loss": total_vicreg_cov / num_steps,
        "variance_hinge": total_variance_hinge / max(1, variance_hinge_count),
    }

    # Per-domain seg loss
    for d in ("MEN", "GLI"):
        if domain_count[d] > 0:
            metrics[f"{d.lower()}_seg_loss"] = domain_seg_loss[d] / domain_count[d]
        else:
            metrics[f"{d.lower()}_seg_loss"] = 0.0

    if enable_gradient_monitoring and grad_monitor_count > 0:
        for key in grad_norms_sum:
            metrics[key] = grad_norms_sum[key] / grad_monitor_count

    return metrics


# =============================================================================
# Main Training Function
# =============================================================================


def train_condition(
    condition_name: str,
    config: dict,
    max_epochs: int | None = None,
    device: str = "cuda",
) -> dict[str, float]:
    """Train a single experimental condition (single or dual domain).

    Args:
        condition_name: Condition name.
        config: Full experiment configuration.
        max_epochs: Override max epochs.
        device: Device.

    Returns:
        Dict with best validation metrics.
    """

    condition_config = get_condition_config(config, condition_name)
    is_baseline = condition_config.get("lora_rank") is None
    skip_training = condition_config.get("skip_training", False)

    logger.info(f"Training condition: {condition_name}")
    logger.info(f"Description: {condition_config.get('description', 'N/A')}")

    if skip_training:
        return _run_validation_only(condition_name, config, device)

    training_config = config["training"]
    decoder_type = training_config.get("decoder_type", "original")
    use_semantic_heads = training_config.get("use_semantic_heads", False)
    is_dual = condition_config.get("dual_domain", False)

    # Detect dual-domain from config paths if not explicitly set
    if (
        not is_dual
        and "men_h5_file" in config.get("paths", {})
        and "gli_h5_file" in config.get("paths", {})
    ):
        is_dual = condition_config.get("dual_domain", False)

    lambda_aux = condition_config.get("lambda_aux_override", training_config.get("lambda_aux", 0.1))
    aux_warmup_start = training_config.get("aux_warmup_epochs", 0)
    aux_warmup_duration = training_config.get("aux_warmup_duration", 10)
    enable_gradient_monitoring = training_config.get("enable_gradient_monitoring", False)
    gradient_monitor_freq = training_config.get("gradient_monitor_freq", 50)
    use_amp = training_config.get("use_amp", False)
    grad_accum_steps = training_config.get("grad_accum_steps", 1)
    diagnostic_interval = training_config.get("diagnostic_interval", 10)

    logger.info(f"Dual-domain: {is_dual}")
    logger.info(f"Decoder type: {decoder_type}, semantic heads: {use_semantic_heads}")
    if use_amp:
        logger.info("Mixed precision: bf16")
    if grad_accum_steps > 1:
        logger.info(f"Gradient accumulation: {grad_accum_steps} steps")

    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    if is_dual:
        men_h5 = config["paths"]["men_h5_file"]
        gli_h5 = config["paths"]["gli_h5_file"]

        logger.info("Creating DUAL-DOMAIN train loader...")
        train_loader = create_dual_domain_train_loader(
            men_h5_path=men_h5,
            gli_h5_path=gli_h5,
            domain_ratio=config.get("data", {}).get("domain_ratio", 0.5),
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
            compute_semantic=use_semantic_heads,
            augment=True,
            seed=config["experiment"]["seed"],
            persistent_workers=True,
        )
        val_loaders = create_per_domain_val_loaders(
            men_h5_path=men_h5,
            gli_h5_path=gli_h5,
            batch_size=training_config.get("val_batch_size", 1),
            num_workers=training_config["num_workers"],
            roi_size=DEFAULT_ROI_SIZE,
            compute_semantic=use_semantic_heads,
            persistent_workers=True,
        )
    else:
        # Single-domain (MEN only)
        h5_path = config["paths"].get("h5_file") or config["paths"].get("men_h5_file")
        logger.info("Creating MEN-only train loader...")
        train_loader, val_loader = create_dataloaders(
            h5_path=h5_path,
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
            compute_semantic=use_semantic_heads,
            augment_train=True,
            val_batch_size=training_config.get("val_batch_size", 1),
            val_roi_size=DEFAULT_ROI_SIZE,
            persistent_workers=True,
        )
        val_loaders = {"men": val_loader}

    # Create model
    model = create_ablation_model(
        condition_config=condition_config,
        training_config=training_config,
        checkpoint_path=config["paths"]["checkpoint"],
        device=device,
    )

    param_counts = model.get_trainable_param_count()
    logger.info(f"Trainable parameters: {param_counts}")

    optimizer, scheduler_info = create_optimizer(model, config, is_baseline, decoder_type)

    # Losses
    seg_loss_fn = SegmentationLoss3Ch(
        lambda_dice=config["loss"]["lambda_dice"],
        lambda_bce=config["loss"]["lambda_ce"],
    )
    dice_metric = DiceMetric3Ch()

    # Auxiliary semantic loss
    aux_loss_fn = None
    if use_semantic_heads and decoder_type == "original":
        aux_loss_fn = AuxiliarySemanticLoss(
            lambda_volume=config["loss"].get("lambda_volume", 1.0),
            normalize_targets=True,
        )
        logger.info("Computing target statistics...")
        stats = compute_target_statistics(train_loader)
        if stats:
            aux_loss_fn.volume_mean.copy_(stats["volume_mean"])
            aux_loss_fn.volume_std.copy_(stats["volume_std"])
            aux_loss_fn._stats_initialized = True

    # VICReg
    vicreg_loss_fn = None
    use_vicreg = condition_config.get("use_vicreg", False)
    if use_vicreg:
        vicreg_loss_fn = EncoderVICRegLoss(
            lambda_var=training_config.get("lambda_var_enc", 5.0),
            lambda_cov=training_config.get("lambda_cov_enc", 1.0),
            gamma=training_config.get("vicreg_gamma", 1.0),
        )
        logger.info("VICReg loss enabled")

    # Training config
    epochs = max_epochs or training_config["max_epochs"]
    patience = training_config["early_stopping_patience"]
    gradient_clip = training_config.get("gradient_clip", 1.0)

    # CSV log with extended dual-domain columns
    log_path = condition_dir / "training_log.csv"
    csv_columns = [
        "epoch",
        "train_loss",
        "train_seg_loss",
        "train_men_seg_loss",
        "train_gli_seg_loss",
        "train_aux_loss",
        "train_vicreg_loss",
        "train_vicreg_var",
        "train_vicreg_cov",
        "val_men_loss",
        "val_men_dice_mean",
        "val_men_dice_tc",
        "val_men_dice_wt",
        "val_men_dice_et",
        "val_gli_loss",
        "val_gli_dice_mean",
        "val_gli_dice_tc",
        "val_gli_dice_wt",
        "val_gli_dice_et",
        "val_combined_dice_mean",
        "lr",
        "lambda_aux_eff",
        "variance_hinge",
        "probe_vol_r2",
        "probe_mean_r2",
        # Per-N-epoch diagnostics (empty on non-diagnostic epochs)
        "diag_mmd_squared",
        "diag_pad",
        "diag_domain_classifier_acc",
        "diag_men_effective_rank",
        "diag_gli_effective_rank",
    ]
    if enable_gradient_monitoring:
        csv_columns.extend(["encoder_grad_norm", "decoder_grad_norm", "semantic_grad_norm"])

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_columns)

    # Training loop
    best_score = 0.0
    best_metrics: dict[str, float] = {}
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        lambda_aux_eff = compute_lambda_aux_effective(
            epoch, lambda_aux, aux_warmup_start, aux_warmup_duration
        )

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
            show_progress=_INTERACTIVE,
        )

        # Validate per domain
        val_metrics = validate_dual_domain(
            model,
            val_loaders,
            seg_loss_fn,
            dice_metric,
            device,
            use_amp=use_amp,
        )

        # Scheduler
        combined_dice = val_metrics["combined_dice_mean"]
        if epoch < scheduler_info["warmup_epochs"]:
            scheduler_info["warmup"].step()
        else:
            scheduler_info["plateau"].step(combined_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log (all ranks log, but only main rank writes CSV/checkpoints)
        log_parts = [
            f"  Train: loss={train_metrics['loss']:.4f} (seg={train_metrics['seg_loss']:.4f}",
        ]
        if is_dual:
            log_parts.append(
                f" men={train_metrics['men_seg_loss']:.4f} gli={train_metrics['gli_seg_loss']:.4f}"
            )
        log_parts.append(f") aux={train_metrics['aux_loss']:.4f}")
        if use_vicreg:
            log_parts.append(f" | VICReg={train_metrics['vicreg_loss']:.4f}")
        logger.info("".join(log_parts))

        for d in val_loaders:
            logger.info(
                f"  Val-{d.upper()}: dice={val_metrics.get(f'{d}_dice_mean', 0):.4f} "
                f"(TC={val_metrics.get(f'{d}_dice_tc', 0):.4f} "
                f"WT={val_metrics.get(f'{d}_dice_wt', 0):.4f} "
                f"ET={val_metrics.get(f'{d}_dice_et', 0):.4f})"
            )
        logger.info(f"  Combined Dice: {combined_dice:.4f} | LR: {current_lr:.2e}")

        probe_metrics: dict[str, float] = {}
        if use_semantic_heads:
            men_val = val_loaders.get("men")
            if men_val is not None:
                probe_metrics = evaluate_feature_quality_inline(model, men_val, device, use_amp)
                model.train()
                if probe_metrics:
                    logger.info(f"  Probe R²: vol={probe_metrics['probe_vol_r2']:.3f}")

        # Per-N-epoch diagnostics (domain gap, effective rank)
        diag_metrics: dict[str, float] = {}
        is_diagnostic_epoch = (epoch + 1) % diagnostic_interval == 0 or epoch == 0
        if is_diagnostic_epoch:
            logger.info(f"  Running epoch diagnostics (interval={diagnostic_interval})...")
            diag_metrics = compute_epoch_diagnostics(model, val_loaders, device, use_amp)
            model.train()
            if diag_metrics:
                parts = []
                if "diag_mmd_squared" in diag_metrics:
                    parts.append(f"MMD²={diag_metrics['diag_mmd_squared']:.4f}")
                if "diag_pad" in diag_metrics:
                    parts.append(f"PAD={diag_metrics['diag_pad']:.4f}")
                for dom in ("men", "gli"):
                    key = f"diag_{dom}_effective_rank"
                    if key in diag_metrics:
                        parts.append(f"rank_{dom}={diag_metrics[key]:.1f}")
                logger.info(f"  Diagnostics: {' | '.join(parts)}")

        checkpoint_score = probe_metrics.get("probe_mean_r2", combined_dice)

        # Write CSV row
        csv_row = [
            epoch + 1,
            train_metrics["loss"],
            train_metrics["seg_loss"],
            train_metrics["men_seg_loss"],
            train_metrics["gli_seg_loss"],
            train_metrics["aux_loss"],
            train_metrics["vicreg_loss"],
            train_metrics["vicreg_var_loss"],
            train_metrics["vicreg_cov_loss"],
            val_metrics.get("men_loss", ""),
            val_metrics.get("men_dice_mean", ""),
            val_metrics.get("men_dice_tc", ""),
            val_metrics.get("men_dice_wt", ""),
            val_metrics.get("men_dice_et", ""),
            val_metrics.get("gli_loss", ""),
            val_metrics.get("gli_dice_mean", ""),
            val_metrics.get("gli_dice_tc", ""),
            val_metrics.get("gli_dice_wt", ""),
            val_metrics.get("gli_dice_et", ""),
            combined_dice,
            current_lr,
            lambda_aux_eff,
            train_metrics.get("variance_hinge", ""),
            probe_metrics.get("probe_vol_r2", ""),
            probe_metrics.get("probe_mean_r2", ""),
            # Diagnostic columns (empty on non-diagnostic epochs)
            diag_metrics.get("diag_mmd_squared", ""),
            diag_metrics.get("diag_pad", ""),
            diag_metrics.get("diag_domain_classifier_acc", ""),
            diag_metrics.get("diag_men_effective_rank", ""),
            diag_metrics.get("diag_gli_effective_rank", ""),
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
            csv.writer(f).writerow(csv_row)

        # Checkpoint
        if checkpoint_score > best_score:
            best_score = checkpoint_score
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
            logger.info(f"  New best! score={checkpoint_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time / 60:.1f} minutes")
    logger.info(
        f"Best combined Dice: {best_metrics.get('combined_dice_mean', 0):.4f} "
        f"at epoch {best_metrics.get('epoch', 0)}"
    )

    summary = {
        "condition": condition_name,
        "is_baseline": is_baseline,
        "dual_domain": is_dual,
        "param_counts": param_counts,
        "best_epoch": best_metrics.get("epoch", 0),
        "best_combined_dice": best_metrics.get("combined_dice_mean", 0),
        "total_epochs": epoch + 1,
        "training_time_minutes": total_time / 60,
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
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])
    setup_file_logging(output_dir, condition)
    set_seed(config["experiment"]["seed"])

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    train_condition(
        condition_name=condition,
        config=config,
        max_epochs=max_epochs,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA condition")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.max_epochs, args.device)
