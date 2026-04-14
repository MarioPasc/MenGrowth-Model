"""Model loading for explainability analysis: frozen and LoRA-adapted variants.

Reuses the existing infrastructure from ``swin_loader.py``, ``lora_adapter.py``
and ``original_decoder.py``. The loading pattern for the adapted model
matches ``EnsemblePredictor._load_member_model`` exactly so that captured
attention weights correspond to the production segmentation model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from monai.networks.nets import SwinUNETR
from omegaconf import DictConfig

from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_full_swinunetr
from growth.models.segmentation.original_decoder import LoRAOriginalDecoderModel

logger = logging.getLogger(__name__)


def get_checkpoint_path(config: DictConfig) -> Path:
    """Resolve the BrainSegFounder checkpoint file from the parent config."""
    return Path(config.paths.checkpoint_dir) / config.paths.checkpoint_filename


def get_run_dir(config: DictConfig, analysis_config: DictConfig, rank: int) -> Path:
    """Resolve the LoRA run directory for a given rank.

    Parameters
    ----------
    config : DictConfig
        Parent config (kept for interface symmetry; unused here).
    analysis_config : DictConfig
        Explainability config with ``paths.base_results_dir`` and
        ``paths.run_dir_pattern``.
    rank : int
        LoRA rank (e.g. 4, 8, 16) that selects the run via the pattern.

    Returns
    -------
    pathlib.Path
        The run directory ``.../{run_dir_pattern}``.
    """
    base = Path(analysis_config.paths.base_results_dir)
    pattern = analysis_config.paths.run_dir_pattern
    return base / pattern.format(rank=rank)


def load_frozen_model(
    config: DictConfig,
    device: str = "cuda",
) -> SwinUNETR:
    """Load the original BrainSegFounder without any LoRA adaptation.

    Parameters
    ----------
    config : DictConfig
        Parent config containing ``paths.checkpoint_dir`` /
        ``paths.checkpoint_filename``.
    device : str
        Target device (``"cuda"``, ``"cuda:0"``, ``"cpu"``).

    Returns
    -------
    SwinUNETR
        Frozen, eval-mode model.
    """
    ckpt_path = get_checkpoint_path(config)
    logger.info("Loading frozen BrainSegFounder from %s", ckpt_path)

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
    analysis_config: DictConfig,
    rank: int,
    member_id: int = 0,
    device: str = "cuda",
    run_dir: Path | str | None = None,
) -> LoRAOriginalDecoderModel:
    """Load one LoRA-adapted ensemble member.

    Replicates the pattern from ``EnsemblePredictor._load_member_model``.
    Each call rebuilds the base SwinUNETR because ``PeftModel.from_pretrained``
    mutates the base model in-place.

    Parameters
    ----------
    config : DictConfig
        Parent config (used for the BSF base checkpoint).
    analysis_config : DictConfig
        Explainability config (used to resolve the run directory).
    rank : int
        LoRA rank; only used when ``run_dir`` is ``None``.
    member_id : int
        Which ensemble member to load.
    device : str
        Target device.
    run_dir : pathlib.Path | str | None
        Optional explicit run directory; overrides the rank-based lookup
        when the CLI passes ``--run-dir`` directly.

    Returns
    -------
    LoRAOriginalDecoderModel
        Eval-mode model wrapping a LoRA-adapted SwinViT and a fine-tuned
        original decoder.
    """
    ckpt_path = get_checkpoint_path(config)
    if run_dir is None:
        run_dir = get_run_dir(config, analysis_config, rank)
    else:
        run_dir = Path(run_dir)
    member_dir = run_dir / "adapters" / f"member_{member_id}"
    adapter_path = member_dir / "adapter"
    decoder_path = member_dir / "decoder.pt"

    logger.info(
        "Loading LoRA-adapted model: rank=%d member=%d from %s",
        rank, member_id, member_dir,
    )

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder not found: {decoder_path}")

    full_model = load_full_swinunetr(
        ckpt_path,
        freeze_encoder=True,
        freeze_decoder=True,
        out_channels=3,
        device="cpu",
    )

    lora_encoder = LoRASwinViT.load_lora(
        base_encoder=full_model,
        adapter_path=str(adapter_path),
        device="cpu",
        trainable=False,
    )

    model = LoRAOriginalDecoderModel(
        lora_encoder=lora_encoder,
        freeze_decoder=True,
        out_channels=3,
        use_semantic_heads=False,
    )

    decoder_state = torch.load(decoder_path, map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(decoder_state)

    model = model.to(device)
    model.eval()
    return model
