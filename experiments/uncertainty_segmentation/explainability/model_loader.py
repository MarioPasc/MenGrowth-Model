"""Model loading for TSI analysis: frozen and LoRA-adapted conditions.

Reuses the existing infrastructure from swin_loader.py, lora_adapter.py,
and original_decoder.py. The loading pattern for the adapted model matches
EnsemblePredictor._load_member_model() exactly.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from omegaconf import DictConfig

from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_full_swinunetr
from growth.models.segmentation.original_decoder import LoRAOriginalDecoderModel

logger = logging.getLogger(__name__)


def get_checkpoint_path(config: DictConfig) -> Path:
    """Resolve BrainSegFounder checkpoint path from config.

    Args:
        config: Parent uncertainty_segmentation config.

    Returns:
        Absolute path to the .pt checkpoint file.
    """
    ckpt_dir = Path(config.paths.checkpoint_dir)
    ckpt_file = config.paths.checkpoint_filename
    return ckpt_dir / ckpt_file


def get_run_dir(config: DictConfig, tsi_config: DictConfig, rank: int) -> Path:
    """Resolve the run directory for a given LoRA rank.

    Args:
        config: Parent config (unused, kept for interface consistency).
        tsi_config: TSI analysis config with paths.base_results_dir and run_dir_pattern.
        rank: LoRA rank (4, 8, or 16).

    Returns:
        Path to the run directory (e.g., .../r8_M20_s42).
    """
    base = Path(tsi_config.paths.base_results_dir)
    pattern = tsi_config.paths.run_dir_pattern
    run_name = pattern.format(rank=rank)
    return base / run_name


def load_frozen_model(
    config: DictConfig,
    device: str = "cuda",
) -> SwinUNETR:
    """Load original BrainSegFounder without any LoRA adaptation.

    The frozen model serves as the baseline condition for TSI comparison.
    Both encoder and decoder are frozen (decoder weights loaded but unused
    for TSI — only encoder hidden states matter).

    Args:
        config: Parent config with paths.checkpoint_dir/filename.
        device: Target device.

    Returns:
        SwinUNETR model in eval mode.
    """
    ckpt_path = get_checkpoint_path(config)
    logger.info(f"Loading frozen BrainSegFounder from {ckpt_path}")

    model = load_full_swinunetr(
        ckpt_path,
        freeze_encoder=True,
        freeze_decoder=True,
        out_channels=3,
        device=device,
    )
    model.eval()
    return model


def load_adapted_model(
    config: DictConfig,
    tsi_config: DictConfig,
    rank: int,
    member_id: int = 0,
    device: str = "cuda",
) -> LoRAOriginalDecoderModel:
    """Load LoRA-adapted model (one ensemble member).

    Replicates the loading pattern from EnsemblePredictor._load_member_model()
    (ensemble_inference.py:159). Fresh base model loaded each time because
    PeftModel.from_pretrained() mutates the base model in-place.

    Args:
        config: Parent config with checkpoint paths.
        tsi_config: TSI config with run directory paths.
        rank: LoRA rank (determines which run directory to use).
        member_id: Which ensemble member to load (default 0).
        device: Target device.

    Returns:
        LoRAOriginalDecoderModel in eval mode.
    """
    ckpt_path = get_checkpoint_path(config)
    run_dir = get_run_dir(config, tsi_config, rank)
    member_dir = run_dir / "adapters" / f"member_{member_id}"
    adapter_path = member_dir / "adapter"
    decoder_path = member_dir / "decoder.pt"

    logger.info(
        f"Loading LoRA-adapted model: rank={rank}, member={member_id} "
        f"from {member_dir}"
    )

    # Verify files exist
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder not found: {decoder_path}")

    # 1. Fresh base model (PeftModel mutates base in-place)
    full_model = load_full_swinunetr(
        ckpt_path,
        freeze_encoder=True,
        freeze_decoder=True,
        out_channels=3,
        device="cpu",
    )

    # 2. Load LoRA adapter
    lora_encoder = LoRASwinViT.load_lora(
        base_encoder=full_model,
        adapter_path=str(adapter_path),
        device="cpu",
        trainable=False,
    )

    # 3. Create segmentation model wrapper
    model = LoRAOriginalDecoderModel(
        lora_encoder=lora_encoder,
        freeze_decoder=True,
        out_channels=3,
        use_semantic_heads=False,
    )

    # 4. Load trained decoder weights
    decoder_state = torch.load(decoder_path, map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(decoder_state)

    # Move to device and set eval
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded adapted model (rank={rank}, member={member_id})")
    return model
