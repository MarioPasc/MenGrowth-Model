# src/growth/models/encoder/lora_adapter.py
"""LoRA adapter for SwinViT encoder.

Applies Low-Rank Adaptation to selected linear layers within Swin Transformer
blocks of the SwinUNETR/SwinViT encoder. Which stages and which linear types
are adapted is controlled by ``target_stages`` and ``target_module_types``.

Supported module types (matched against the tail of ``named_modules()`` keys).
The user-facing keywords follow standard Swin/Transformer terminology, while
the suffixes below use MONAI's actual attribute names (``linear1``/``linear2``
rather than the literature's ``fc1``/``fc2``):

    qkv   -> ``.attn.qkv``       (combined Q, K, V projection)
    proj  -> ``.attn.proj``      (attention output projection)
    fc1   -> ``.mlp.linear1``    (MLP sublayer: dim -> 4*dim)
    fc2   -> ``.mlp.linear2``    (MLP sublayer: 4*dim -> dim)

Default behaviour (``target_module_types=["qkv"]``) preserves the historical
qkv-only configuration. Extended configurations (``["qkv","proj","fc1","fc2"]``)
let each ensemble member perturb the full Swin block, including the MLP
sublayer where nonlinear feature transformation happens.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from .swin_loader import load_swin_encoder

if TYPE_CHECKING:
    from growth.utils.model_card import LoRAModelCardConfig

logger = logging.getLogger(__name__)

# Map module-type keyword -> required suffix in dotted module name.
# The suffix is matched with str.endswith, so it must uniquely identify the
# linear layer within the block (e.g. ".attn.proj" does not collide with
# ".mlp.fc1").
MODULE_TYPE_SUFFIX: Dict[str, str] = {
    "qkv": ".attn.qkv",
    "proj": ".attn.proj",
    # MONAI's SwinUNETR names the MLP linears "linear1"/"linear2" rather than
    # the literature's "fc1"/"fc2" — we keep the user-facing keywords aligned
    # with the literature and map them to MONAI's actual attribute names.
    "fc1": ".mlp.linear1",
    "fc2": ".mlp.linear2",
}

SUPPORTED_MODULE_TYPES = tuple(MODULE_TYPE_SUFFIX.keys())


def _validate_module_types(module_types: List[str]) -> List[str]:
    """Normalise and validate a list of module-type keywords."""
    if not module_types:
        raise ValueError("target_module_types must be a non-empty list")
    unknown = [t for t in module_types if t not in MODULE_TYPE_SUFFIX]
    if unknown:
        raise ValueError(
            f"Unsupported LoRA target_module_types: {unknown}. "
            f"Expected a subset of {SUPPORTED_MODULE_TYPES}."
        )
    # Preserve caller ordering but deduplicate.
    seen: List[str] = []
    for t in module_types:
        if t not in seen:
            seen.append(t)
    return seen


def _infer_module_types_from_target_names(target_modules: List[str]) -> List[str]:
    """Infer the set of module-type keywords from a PEFT target_modules list.

    Used by ``LoRASwinViT.load_lora`` to reconstruct the wrapper attribute
    from a saved adapter's ``adapter_config.json`` (which stores the full
    ``target_modules`` list but not our type-keyword abstraction).

    Args:
        target_modules: List of dotted module names saved by PEFT.

    Returns:
        List of type keywords in deterministic order (qkv, proj, fc1, fc2).
    """
    inferred: List[str] = []
    for key, suffix in MODULE_TYPE_SUFFIX.items():
        if any(name.endswith(suffix) for name in target_modules):
            inferred.append(key)
    return inferred


def _find_lora_targets(
    model: nn.Module,
    stages: List[int] = [3, 4],
    module_types: List[str] = ["qkv"],
) -> List[str]:
    """Find exact module names for LoRA injection.

    PEFT matches ``target_modules`` using endswith semantics, so we return
    exact (relative) names rather than regex patterns.

    Args:
        model: The SwinUNETR model.
        stages: Which Swin stages to adapt (1, 2, 3, or 4).
        module_types: Which linear layers inside each selected block to
            adapt. Subset of ``SUPPORTED_MODULE_TYPES``. Default ``["qkv"]``
            preserves the historical behaviour.

    Returns:
        List of exact module names (with any ``swinViT.`` prefix stripped,
        because PEFT re-prefixes during injection).
    """
    module_types = _validate_module_types(module_types)
    suffixes = [MODULE_TYPE_SUFFIX[t] for t in module_types]

    targets: List[str] = []
    for name, _module in model.named_modules():
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        if not any(f"layers{stage}" in name for stage in stages):
            continue
        # Strip the 'swinViT.' prefix because PEFT re-adds it internally.
        if name.startswith("swinViT."):
            targets.append(name[len("swinViT."):])
        else:
            targets.append(name)
    return targets


class LoRASwinViT(nn.Module):
    """SwinViT encoder with LoRA adapters.

    Wraps a SwinUNETR model and adds LoRA adapters to the linear layers
    inside selected Swin Transformer blocks. All base model parameters
    are frozen; only LoRA parameters are trainable.

    Args:
        base_encoder: SwinUNETR model (from load_swin_encoder).
        rank: LoRA rank (common values: 2, 4, 8, 16, 32).
        alpha: LoRA alpha for scaling. Effective scale is alpha/rank.
        dropout: LoRA dropout rate.
        target_stages: Which Swin stages to apply LoRA to. Default ``[3, 4]``.
        target_module_types: Which linear layers inside each selected block
            to adapt. Subset of ``("qkv", "proj", "fc1", "fc2")``. Default
            ``["qkv"]`` preserves the historical qkv-only configuration.
        use_dora: If True, use DoRA (Weight-Decomposed LoRA) instead of
            standard LoRA. Default: False.

    Attributes:
        model: The PEFT-wrapped model.
        rank, alpha, use_dora, target_stages, target_module_types, lora_config.

    Example:
        >>> base_encoder = load_swin_encoder(ckpt_path, freeze=False)
        >>> lora_encoder = LoRASwinViT(
        ...     base_encoder, rank=8, alpha=16,
        ...     target_stages=[1, 2, 3],
        ...     target_module_types=["qkv", "proj", "fc1", "fc2"],
        ... )

    References:
        - DoRA: Liu et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation."
          arXiv:2402.09353.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_stages: List[int] = [3, 4],
        target_module_types: List[str] = ["qkv"],
        use_dora: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.target_stages = list(target_stages)
        self.target_module_types = _validate_module_types(list(target_module_types))
        self.use_dora = use_dora

        # Freeze all base parameters first
        for param in base_encoder.parameters():
            param.requires_grad = False

        # Find target modules for LoRA (stage + module-type filter).
        target_modules = _find_lora_targets(
            base_encoder,
            stages=self.target_stages,
            module_types=self.target_module_types,
        )
        logger.info(
            f"Found {len(target_modules)} target modules for LoRA "
            f"(stages={self.target_stages}, types={self.target_module_types})"
        )

        if not target_modules:
            raise ValueError(
                f"No target modules found for stages={self.target_stages}, "
                f"types={self.target_module_types}. Check model structure."
            )

        # Create LoRA config (with optional DoRA)
        adapter_type = "DoRA" if use_dora else "LoRA"
        logger.info(f"Using {adapter_type} with rank={rank}, alpha={alpha}")

        self.lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            modules_to_save=None,
            use_dora=use_dora,
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

    def save_lora(
        self,
        path: Union[str, Path],
        model_card_config: Optional["LoRAModelCardConfig"] = None,
    ) -> None:
        """Save only LoRA adapter weights.

        Args:
            path: Directory path to save adapter.
            model_card_config: Optional model card configuration. If provided,
                a README.md with training metadata will be generated. PEFT will
                then merge its own metadata into the card.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model card first (if configured) so PEFT can merge into it
        if model_card_config is not None:
            from growth.utils.model_card import save_lora_model_card

            save_lora_model_card(path, model_card_config)

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

        # Create wrapper instance (bypass __init__ since PEFT already wired the adapters)
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.model = model

        saved_config = model.peft_config["default"]
        instance.rank = saved_config.r
        instance.alpha = saved_config.lora_alpha
        instance.use_dora = bool(getattr(saved_config, "use_dora", False))
        instance.lora_config = saved_config

        # Reconstruct target_stages / target_module_types from the saved
        # target_modules list, so a reloaded adapter reports the same
        # configuration it was trained with (no caller-side duplication).
        saved_targets = list(getattr(saved_config, "target_modules", []) or [])
        instance.target_module_types = _infer_module_types_from_target_names(saved_targets)
        inferred_stages: List[int] = []
        for name in saved_targets:
            for stage in (1, 2, 3, 4):
                if f"layers{stage}" in name and stage not in inferred_stages:
                    inferred_stages.append(stage)
        instance.target_stages = sorted(inferred_stages) if inferred_stages else [3, 4]

        logger.info(
            f"LoRA adapter loaded from {adapter_path} "
            f"(stages={instance.target_stages}, types={instance.target_module_types})"
        )
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
    target_module_types: List[str] = ["qkv"],
    device: str = "cuda",
    use_dora: bool = False,
) -> LoRASwinViT:
    """Factory function to create LoRA-adapted encoder.

    Args:
        checkpoint_path: Path to BrainSegFounder checkpoint.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        target_stages: Which stages to apply LoRA to.
        target_module_types: Which linear layers inside each block
            (subset of ``("qkv","proj","fc1","fc2")``).
        device: Device to load model to.
        use_dora: If True, use DoRA.

    Returns:
        LoRASwinViT instance ready for training.
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
        target_module_types=target_module_types,
        use_dora=use_dora,
    )


def count_lora_params(model: LoRASwinViT) -> Dict[str, int]:
    """Count trainable LoRA parameters by Swin stage.

    Args:
        model: LoRASwinViT instance.

    Returns:
        Dict with keys ``total``, ``layers1``, ``layers2``, ``layers3``,
        ``layers4``. Stages not targeted by the configuration will report 0.
    """
    counts: Dict[str, int] = {"total": 0, "layers1": 0, "layers2": 0, "layers3": 0, "layers4": 0}

    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue

        numel = param.numel()
        counts["total"] += numel

        for stage in (1, 2, 3, 4):
            if f"layers{stage}" in name:
                counts[f"layers{stage}"] += numel
                break

    return counts
