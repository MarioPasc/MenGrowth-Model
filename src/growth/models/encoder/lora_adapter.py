# src/growth/models/encoder/lora_adapter.py
"""LoRA adapter for SwinViT encoder.

Applies Low-Rank Adaptation to Q, K, V projections in Stages 3-4.
Freezes Stages 0-2 to preserve low-level anatomy features.

Uses HuggingFace PEFT library for robust LoRA implementation.

Target modules in SwinUNETR/SwinViT:
    - swinViT.layers3.*.blocks.*.attn.qkv (192 -> 576)
    - swinViT.layers4.*.blocks.*.attn.qkv (384 -> 1152)

Optionally also targets projection layers:
    - swinViT.layers3.*.blocks.*.attn.proj (192 -> 192)
    - swinViT.layers4.*.blocks.*.attn.proj (384 -> 384)
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from .swin_loader import load_swin_encoder

logger = logging.getLogger(__name__)


def get_lora_target_modules(
    stages: List[int] = [3, 4],
    include_proj: bool = False,
) -> List[str]:
    """Get target module patterns for LoRA injection.

    Args:
        stages: Which stages to apply LoRA to (3 and/or 4).
        include_proj: Whether to include output projection layers.

    Returns:
        List of module name patterns (regex-compatible).

    Example:
        >>> modules = get_lora_target_modules(stages=[3, 4])
        >>> modules
        ['layers3.0.blocks.0.attn.qkv', 'layers3.0.blocks.1.attn.qkv', ...]
    """
    target_modules = []

    for stage in stages:
        # QKV projection
        target_modules.append(f"swinViT.layers{stage}")

        if include_proj:
            # Output projection
            pass  # PEFT will match patterns

    return target_modules


def _find_lora_targets(model: nn.Module, stages: List[int] = [3, 4]) -> List[str]:
    """Find exact module names for LoRA injection.

    PEFT works better with exact module names than regex patterns.

    Args:
        model: The SwinUNETR model.
        stages: Which stages to apply LoRA to.

    Returns:
        List of exact module names containing 'qkv'.
    """
    targets = []
    for name, module in model.named_modules():
        # Match pattern: swinViT.layers{stage}.*.blocks.*.attn.qkv
        for stage in stages:
            if f"layers{stage}" in name and "attn.qkv" in name:
                # Remove 'swinViT.' prefix since PEFT adds it
                if name.startswith("swinViT."):
                    targets.append(name[len("swinViT."):])
                else:
                    targets.append(name)
                break
    return targets


class LoRASwinViT(nn.Module):
    """SwinViT encoder with LoRA adapters.

    Wraps a SwinUNETR model and adds LoRA adapters to Q, K, V
    projections in stages 3 and 4. All base model parameters are
    frozen; only LoRA parameters are trainable.

    Args:
        base_encoder: SwinUNETR model (from load_swin_encoder).
        rank: LoRA rank (common values: 4, 8, 16).
        alpha: LoRA alpha for scaling. Effective scale is alpha/rank.
        dropout: LoRA dropout rate.
        target_stages: Which stages to apply LoRA to (default: [3, 4]).

    Attributes:
        model: The PEFT-wrapped model.
        rank: LoRA rank.
        alpha: LoRA alpha.
        lora_config: The LoRA configuration.

    Example:
        >>> base_encoder = load_swin_encoder(ckpt_path, freeze=False)
        >>> lora_encoder = LoRASwinViT(base_encoder, rank=8, alpha=16)
        >>> print(f"Trainable params: {lora_encoder.get_trainable_params():,}")
        Trainable params: 150,528
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_stages: List[int] = [3, 4],
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.target_stages = target_stages

        # Freeze all base parameters first
        for param in base_encoder.parameters():
            param.requires_grad = False

        # Find target modules for LoRA
        target_modules = _find_lora_targets(base_encoder, target_stages)
        logger.info(f"Found {len(target_modules)} target modules for LoRA")

        if not target_modules:
            raise ValueError(
                f"No target modules found for stages {target_stages}. "
                "Check model structure."
            )

        # Create LoRA config
        self.lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            modules_to_save=None,
        )

        # Apply PEFT
        self.model = get_peft_model(base_encoder, self.lora_config)

        # Log parameter counts
        trainable = self.get_trainable_params()
        total = self.get_total_params()
        logger.info(
            f"LoRA applied: {trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA-adapted encoder.

        Args:
            x: Input tensor [B, 4, D, H, W].

        Returns:
            Output tensor from full model.
        """
        return self.model(x)

    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get hidden states from all SwinViT stages.

        Args:
            x: Input tensor [B, 4, D, H, W].

        Returns:
            List of 5 tensors, one per stage.
        """
        return self.model.swinViT(x, self.model.normalize)

    def merge_lora(self) -> nn.Module:
        """Merge LoRA weights into base model for efficient inference.

        After merging, the model can be used without PEFT overhead.
        This is irreversible on the current instance.

        Returns:
            The merged model.
        """
        merged = self.model.merge_and_unload()
        logger.info("LoRA weights merged into base model")
        return merged

    def save_lora(self, path: Union[str, Path]) -> None:
        """Save only LoRA adapter weights.

        Args:
            path: Directory path to save adapter.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        logger.info(f"LoRA adapter saved to {path}")

    @classmethod
    def load_lora(
        cls,
        base_encoder: nn.Module,
        adapter_path: Union[str, Path],
        device: str = "cpu",
        trainable: bool = True,
    ) -> "LoRASwinViT":
        """Load LoRA adapter from saved checkpoint.

        Args:
            base_encoder: Fresh SwinUNETR encoder (unfrozen).
            adapter_path: Path to saved adapter directory.
            device: Device to load to.
            trainable: If True, make LoRA parameters trainable after loading.

        Returns:
            LoRASwinViT instance with loaded adapter.
        """
        adapter_path = Path(adapter_path)

        # Freeze base encoder
        for param in base_encoder.parameters():
            param.requires_grad = False

        # Load PEFT model
        model = PeftModel.from_pretrained(base_encoder, adapter_path)
        model.to(device)

        # PEFT loads parameters as non-trainable by default for inference
        # Re-enable training on LoRA parameters if requested
        if trainable:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

        # Create wrapper instance
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.model = model
        instance.rank = model.peft_config["default"].r
        instance.alpha = model.peft_config["default"].lora_alpha
        instance.target_stages = [3, 4]  # Default assumption
        instance.lora_config = model.peft_config["default"]

        logger.info(f"LoRA adapter loaded from {adapter_path}")
        return instance

    def get_trainable_params(self) -> int:
        """Count trainable LoRA parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters (including frozen)."""
        return sum(p.numel() for p in self.model.parameters())

    def get_lora_params(self) -> Dict[str, torch.Tensor]:
        """Get only LoRA parameter tensors.

        Returns:
            Dict mapping parameter names to tensors.
        """
        lora_params = {}
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                lora_params[name] = param
        return lora_params

    def print_trainable_parameters(self) -> None:
        """Print summary of trainable parameters."""
        self.model.print_trainable_parameters()


def create_lora_encoder(
    checkpoint_path: Union[str, Path],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_stages: List[int] = [3, 4],
    device: str = "cuda",
) -> LoRASwinViT:
    """Factory function to create LoRA-adapted encoder.

    Convenience function that:
    1. Loads base encoder from checkpoint
    2. Adds LoRA adapters
    3. Returns ready-to-train LoRASwinViT

    Args:
        checkpoint_path: Path to BrainSegFounder checkpoint.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        target_stages: Which stages to apply LoRA to.
        device: Device to load model to.

    Returns:
        LoRASwinViT instance ready for training.

    Example:
        >>> lora_encoder = create_lora_encoder(
        ...     checkpoint_path="checkpoints/fold_0.pt",
        ...     rank=8,
        ...     device="cuda",
        ... )
        >>> # Only LoRA params are trainable
        >>> optimizer = torch.optim.AdamW(
        ...     [p for p in lora_encoder.parameters() if p.requires_grad],
        ...     lr=1e-4,
        ... )
    """
    # Load base encoder (unfrozen initially, LoRASwinViT will freeze)
    base_encoder = load_swin_encoder(
        checkpoint_path,
        freeze=False,  # Don't freeze here, LoRASwinViT handles it
        device=device,
    )

    return LoRASwinViT(
        base_encoder,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_stages=target_stages,
    )


def count_lora_params(model: LoRASwinViT) -> Dict[str, int]:
    """Count LoRA parameters by layer.

    Args:
        model: LoRASwinViT instance.

    Returns:
        Dict with 'total', 'layers3', 'layers4' counts.
    """
    counts = {"total": 0, "layers3": 0, "layers4": 0}

    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue

        numel = param.numel()
        counts["total"] += numel

        if "layers3" in name:
            counts["layers3"] += numel
        elif "layers4" in name:
            counts["layers4"] += numel

    return counts
