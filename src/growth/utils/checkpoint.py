# src/growth/utils/checkpoint.py
"""Checkpoint loading and weight manipulation utilities.

This module provides:
- BrainSegFounder checkpoint loading
- Encoder weight extraction (filtering decoder keys)
- LoRA weight merging (for inference)
"""

import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Keys to extract from BrainSegFounder checkpoint (encoder only)
ENCODER_PREFIXES: FrozenSet[str] = frozenset([
    "swinViT",      # Swin Transformer backbone
    "encoder1",     # Skip connection processor (stage 1)
    "encoder2",     # Skip connection processor (stage 2)
    "encoder3",     # Skip connection processor (stage 3)
    "encoder4",     # Skip connection processor (stage 4)
    "encoder10",    # Bottleneck processor (768 channels)
])

# Keys to discard (decoder and output head)
DECODER_PREFIXES: FrozenSet[str] = frozenset([
    "decoder1",
    "decoder2",
    "decoder3",
    "decoder4",
    "decoder5",
    "out",
])


def load_checkpoint(
    ckpt_path: Union[str, Path],
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load checkpoint from disk.

    Handles BrainSegFounder's checkpoint format which includes
    numpy scalars requiring weights_only=False.

    Args:
        ckpt_path: Path to checkpoint file (.pt or .pth).
        map_location: Device to load weights to ("cpu", "cuda", etc.).

    Returns:
        Checkpoint dictionary with keys:
        - 'state_dict': Model weights
        - 'epoch': Training epoch (may be None)
        - 'best_acc': Best accuracy (may be None)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        RuntimeError: If checkpoint format is invalid.

    Example:
        >>> ckpt = load_checkpoint("model.pt")
        >>> state_dict = ckpt["state_dict"]
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # weights_only=False required for numpy scalars in BrainSegFounder checkpoints
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Standard format with metadata
        logger.info(
            f"Loaded checkpoint from {ckpt_path.name}: "
            f"epoch={checkpoint.get('epoch', 'N/A')}, "
            f"best_acc={checkpoint.get('best_acc', 'N/A')}"
        )
        return checkpoint
    elif isinstance(checkpoint, dict):
        # Raw state dict (no wrapper) - wrap it
        logger.info(f"Loaded raw state_dict from {ckpt_path.name}")
        return {"state_dict": checkpoint, "epoch": None, "best_acc": None}
    else:
        raise RuntimeError(
            f"Invalid checkpoint format: expected dict, got {type(checkpoint)}"
        )


def extract_encoder_weights(
    state_dict: Dict[str, torch.Tensor],
    include_encoder10: bool = True,
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """Extract encoder-only weights from full SwinUNETR state dict.

    Filters out decoder and output head weights, keeping only
    the encoder components needed for feature extraction.

    Args:
        state_dict: Full SwinUNETR state dictionary.
        include_encoder10: If True, include encoder10 (bottleneck processor).
            Set False to use raw swinViT.layers4 output (384-dim).
        strict: If True, raise error if no encoder keys found.

    Returns:
        Filtered state dictionary with encoder weights only.

    Raises:
        ValueError: If strict=True and no encoder keys found.

    Example:
        >>> encoder_weights = extract_encoder_weights(full_state_dict)
        >>> len(encoder_weights) < len(full_state_dict)
        True
    """
    prefixes = set(ENCODER_PREFIXES)
    if not include_encoder10:
        prefixes.discard("encoder10")

    encoder_state_dict = {}
    for key, value in state_dict.items():
        prefix = key.split(".")[0]
        if prefix in prefixes:
            encoder_state_dict[key] = value

    if strict and len(encoder_state_dict) == 0:
        available_prefixes = set(k.split(".")[0] for k in state_dict.keys())
        raise ValueError(
            f"No encoder keys found in state_dict. "
            f"Available prefixes: {available_prefixes}"
        )

    # Log extraction summary
    discarded = len(state_dict) - len(encoder_state_dict)
    logger.info(
        f"Extracted {len(encoder_state_dict)} encoder keys, "
        f"discarded {discarded} decoder/output keys"
    )

    return encoder_state_dict


def get_checkpoint_stats(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    """Get statistics about checkpoint weights.

    Useful for debugging and understanding checkpoint structure.

    Args:
        state_dict: State dictionary to analyze.

    Returns:
        Dictionary mapping prefixes to statistics:
        - count: Number of keys
        - params: Total parameters
        - params_m: Parameters in millions
        - shapes: List of tensor shapes

    Example:
        >>> stats = get_checkpoint_stats(state_dict)
        >>> print(f"swinViT: {stats['swinViT']['params_m']:.2f}M params")
    """
    stats: Dict[str, Dict[str, Any]] = {}

    for key, tensor in state_dict.items():
        prefix = key.split(".")[0]
        if prefix not in stats:
            stats[prefix] = {
                "count": 0,
                "params": 0,
                "shapes": [],
            }
        stats[prefix]["count"] += 1
        stats[prefix]["params"] += tensor.numel()
        stats[prefix]["shapes"].append(tuple(tensor.shape))

    # Convert to millions and add summary
    for prefix in stats:
        stats[prefix]["params_m"] = stats[prefix]["params"] / 1e6

    return stats


def merge_lora_weights(
    base_state_dict: Dict[str, torch.Tensor],
    lora_state_dict: Dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Merge LoRA adapter weights into base model weights.

    For each LoRA pair (lora_A, lora_B), computes:
        W_merged = W_base + alpha * (B @ A)

    This is used for inference to avoid the overhead of
    separate LoRA forward passes.

    Args:
        base_state_dict: Base model weights.
        lora_state_dict: LoRA adapter weights with keys like
            "module.lora_A", "module.lora_B".
        alpha: LoRA scaling factor (typically lora_alpha / lora_rank).

    Returns:
        Merged state dictionary with LoRA weights folded in.

    Raises:
        ValueError: If LoRA weight shapes are incompatible.

    Example:
        >>> merged = merge_lora_weights(base_weights, lora_weights, alpha=2.0)
    """
    merged = dict(base_state_dict)
    n_merged = 0

    # Find LoRA pairs
    lora_a_keys = [k for k in lora_state_dict if ".lora_A" in k]

    for lora_a_key in lora_a_keys:
        # Derive base key and lora_B key
        base_key = lora_a_key.replace(".lora_A", "")
        lora_b_key = lora_a_key.replace(".lora_A", ".lora_B")

        if base_key not in base_state_dict:
            logger.warning(f"Base key not found for LoRA: {base_key}")
            continue

        if lora_b_key not in lora_state_dict:
            logger.warning(f"LoRA B key not found: {lora_b_key}")
            continue

        # Get LoRA weights
        lora_a = lora_state_dict[lora_a_key]  # [rank, in_features]
        lora_b = lora_state_dict[lora_b_key]  # [out_features, rank]

        # Compute merged weights: W + alpha * (B @ A)
        delta = alpha * (lora_b @ lora_a)

        if delta.shape != base_state_dict[base_key].shape:
            raise ValueError(
                f"Shape mismatch for {base_key}: "
                f"base={base_state_dict[base_key].shape}, delta={delta.shape}"
            )

        merged[base_key] = base_state_dict[base_key] + delta
        n_merged += 1
        logger.debug(f"Merged LoRA weights for: {base_key}")

    logger.info(f"Merged {n_merged} LoRA weight pairs with alpha={alpha}")
    return merged


def print_checkpoint_summary(state_dict: Dict[str, torch.Tensor]) -> None:
    """Print a formatted summary of checkpoint contents.

    Args:
        state_dict: State dictionary to summarize.
    """
    stats = get_checkpoint_stats(state_dict)
    total_params = sum(s["params"] for s in stats.values())

    print("\n" + "=" * 60)
    print("Checkpoint Summary")
    print("=" * 60)
    print(f"{'Prefix':<15} {'Keys':>8} {'Params':>12} {'%':>8}")
    print("-" * 60)

    for prefix in sorted(stats.keys()):
        info = stats[prefix]
        pct = 100 * info["params"] / total_params
        print(f"{prefix:<15} {info['count']:>8} {info['params_m']:>10.2f}M {pct:>7.1f}%")

    print("-" * 60)
    print(f"{'TOTAL':<15} {sum(s['count'] for s in stats.values()):>8} {total_params/1e6:>10.2f}M {100.0:>7.1f}%")
    print("=" * 60 + "\n")
