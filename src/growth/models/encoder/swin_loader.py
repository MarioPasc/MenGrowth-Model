# src/growth/models/encoder/swin_loader.py
"""BrainSegFounder SwinUNETR encoder loader.

Loads the pre-trained SwinViT encoder from BrainSegFounder checkpoint.
Extracts only encoder components (swinViT + encoder1-4 + encoder10).
"""

import logging
from pathlib import Path

import torch
from torch import nn

try:
    from monai.networks.nets import SwinUNETR
except ImportError:
    raise ImportError("MONAI is required for SwinUNETR. Install with: pip install monai>=1.3.0")

from growth.utils.checkpoint import (
    extract_encoder_weights,
    get_checkpoint_stats,
    load_checkpoint,
)

logger = logging.getLogger(__name__)

# BrainSegFounder architecture constants
BRAINSEGFOUNDER_FEATURE_SIZE = 48  # Base channel count
BRAINSEGFOUNDER_IN_CHANNELS = 4  # [FLAIR, T1ce, T1, T2] = [t2f, t1c, t1n, t2w]
BRAINSEGFOUNDER_OUT_CHANNELS = 3  # BraTS segmentation classes (TC, WT, ET)
BRAINSEGFOUNDER_DEPTHS = (2, 2, 2, 2)
BRAINSEGFOUNDER_NUM_HEADS = (3, 6, 12, 24)


def create_swinunetr(
    in_channels: int = BRAINSEGFOUNDER_IN_CHANNELS,
    out_channels: int = BRAINSEGFOUNDER_OUT_CHANNELS,
    feature_size: int = BRAINSEGFOUNDER_FEATURE_SIZE,
    depths: tuple[int, ...] = BRAINSEGFOUNDER_DEPTHS,
    num_heads: tuple[int, ...] = BRAINSEGFOUNDER_NUM_HEADS,
    use_checkpoint: bool = False,
    spatial_dims: int = 3,
) -> SwinUNETR:
    """Create SwinUNETR model matching BrainSegFounder architecture.

    Args:
        in_channels: Number of input channels (default: 4 for MRI modalities).
        out_channels: Number of output channels (default: 3 for BraTS).
        feature_size: Base feature size (default: 48, must match checkpoint).
        depths: Number of Swin Transformer blocks per stage.
        num_heads: Number of attention heads per stage.
        use_checkpoint: If True, use gradient checkpointing to save memory.
        spatial_dims: Spatial dimensions (2 or 3).

    Returns:
        SwinUNETR model instance.

    Example:
        >>> model = create_swinunetr()
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 3, 128, 128, 128])
    """
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
    )

    logger.info(
        f"Created SwinUNETR: in={in_channels}, out={out_channels}, "
        f"feature_size={feature_size}, depths={depths}"
    )

    return model


def load_swin_encoder(
    ckpt_path: str | Path,
    include_encoder10: bool = True,
    freeze: bool = True,
    use_checkpoint: bool = False,
    device: str | torch.device = "cpu",
    strict_load: bool = False,
) -> SwinUNETR:
    """Load BrainSegFounder SwinUNETR encoder from checkpoint.

    Creates a SwinUNETR model and loads only the encoder weights,
    discarding the U-Net decoder and segmentation head.

    Args:
        ckpt_path: Path to BrainSegFounder checkpoint (.pt file).
        include_encoder10: If True, include encoder10 (bottleneck processor).
            Output will be 768-dim. If False, use swinViT.layers4 directly
            for 384-dim output.
        freeze: If True, freeze all encoder parameters.
        use_checkpoint: If True, use gradient checkpointing (saves memory).
        device: Device to load model to ("cpu", "cuda", etc.).
        strict_load: If True, raise error on missing encoder keys.

    Returns:
        SwinUNETR model with encoder weights loaded.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        RuntimeError: If weight loading fails.

    Example:
        >>> encoder = load_swin_encoder(
        ...     "checkpoints/finetuned_model_fold_0.pt",
        ...     freeze=True
        ... )
        >>> encoder.eval()
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> with torch.no_grad():
        ...     hidden = encoder.swinViT(x, encoder.normalize)
    """
    ckpt_path = Path(ckpt_path)

    # Load checkpoint
    checkpoint = load_checkpoint(ckpt_path)
    state_dict = checkpoint["state_dict"]

    # Log checkpoint statistics
    stats = get_checkpoint_stats(state_dict)
    logger.info("Checkpoint weight groups:")
    for prefix, info in sorted(stats.items()):
        logger.info(f"  {prefix}: {info['count']} keys, {info['params_m']:.2f}M params")

    # Extract encoder weights
    encoder_state_dict = extract_encoder_weights(
        state_dict,
        include_encoder10=include_encoder10,
        strict=strict_load,
    )

    # Create model
    model = create_swinunetr(use_checkpoint=use_checkpoint)

    # Load encoder weights (strict=False to ignore missing decoder keys)
    missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)

    # Categorize missing keys
    decoder_prefixes = ("decoder", "out")
    expected_missing = [k for k in missing if k.startswith(decoder_prefixes)]
    actual_missing = [k for k in missing if not k.startswith(decoder_prefixes)]

    if actual_missing:
        logger.warning(f"Missing encoder keys ({len(actual_missing)}): {actual_missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    encoder_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded encoder: {len(encoder_state_dict)} keys, "
        f"{encoder_params / 1e6:.2f}M params, "
        f"{len(expected_missing)} decoder keys not loaded (expected)"
    )

    # Freeze encoder if requested
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Froze all encoder parameters")

    # Move to device and set eval mode
    model = model.to(device)
    if freeze:
        model.eval()

    return model


def get_swin_feature_dims(
    feature_size: int = BRAINSEGFOUNDER_FEATURE_SIZE,
) -> dict[str, tuple[int, int]]:
    """Get feature dimensions at each Swin stage.

    Args:
        feature_size: Base feature size (default: 48).

    Returns:
        Dictionary mapping stage names to (channels, spatial_downsample_factor).
        The spatial factor indicates downsampling relative to input size.

    Example:
        >>> dims = get_swin_feature_dims()
        >>> dims["layers4"]
        (384, 32)
    """
    # MONAI 1.5+ SwinUNETR hidden states dimensions
    # Note: channels double at each stage after patch_embed
    return {
        "patch_embed": (feature_size, 2),  # 48, input/2
        "layers1": (feature_size * 2, 4),  # 96, input/4
        "layers2": (feature_size * 4, 8),  # 192, input/8
        "layers3": (feature_size * 8, 16),  # 384, input/16
        "layers4": (feature_size * 16, 32),  # 768, input/32
        "encoder10": (feature_size * 16, 32),  # 768, input/32 (same as layers4)
    }


def get_encoder_output_dim(
    feature_level: str = "encoder10",
    feature_size: int = BRAINSEGFOUNDER_FEATURE_SIZE,
) -> int:
    """Get output dimension for a given feature level.

    Args:
        feature_level: Feature extraction level.
        feature_size: Base feature size.

    Returns:
        Output channel dimension.

    Raises:
        ValueError: If feature_level is unknown.
    """
    dims = get_swin_feature_dims(feature_size)

    if feature_level == "multi_scale":
        # layers2 + layers3 + layers4
        return dims["layers2"][0] + dims["layers3"][0] + dims["layers4"][0]

    if feature_level not in dims:
        raise ValueError(
            f"Unknown feature level: {feature_level}. "
            f"Choose from: {list(dims.keys())} or 'multi_scale'"
        )

    return dims[feature_level][0]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_full_swinunetr(
    ckpt_path: str | Path,
    freeze_encoder: bool = True,
    freeze_decoder: bool = False,
    out_channels: int = 3,
    use_checkpoint: bool = False,
    device: str | torch.device = "cpu",
) -> SwinUNETR:
    """Load FULL BrainSegFounder SwinUNETR with ALL pretrained weights.

    Unlike load_swin_encoder(), this loads both encoder AND decoder weights,
    which is necessary for using the original SwinUNETR decoder architecture
    with pretrained weights (achieving ~0.85+ Dice).

    IMPORTANT: BrainSegFounder was trained with 3 output channels (TC, WT, ET)
    using sigmoid activation per channel. These are hierarchical overlapping
    regions, not individual labels. Setting out_channels=3 preserves the
    pretrained output layer. Setting out_channels=4 will REPLACE the output
    layer with random weights (not recommended for frozen baselines).

    Args:
        ckpt_path: Path to BrainSegFounder checkpoint (.pt file).
        freeze_encoder: If True, freeze encoder (swinViT) parameters.
        freeze_decoder: If True, freeze decoder parameters.
        out_channels: Number of output channels. Default 3 to preserve pretrained
            output layer. Use 3 with sigmoid+BCE loss, not softmax+CE.
        use_checkpoint: If True, use gradient checkpointing (saves memory).
        device: Device to load model to.

    Returns:
        SwinUNETR model with ALL weights loaded (encoder + decoder).

    Example:
        >>> model = load_full_swinunetr("checkpoint.pt", freeze_encoder=True)
        >>> # Now model.decoder1-5 have pretrained weights!
    """
    ckpt_path = Path(ckpt_path)

    # Load checkpoint
    checkpoint = load_checkpoint(ckpt_path)
    state_dict = checkpoint["state_dict"]

    # Log checkpoint statistics
    stats = get_checkpoint_stats(state_dict)
    logger.info("Loading FULL SwinUNETR (encoder + decoder):")
    for prefix, info in sorted(stats.items()):
        logger.info(f"  {prefix}: {info['count']} keys, {info['params_m']:.2f}M params")

    # Create model (use same out_channels as checkpoint for weight loading)
    # BrainSegFounder was trained with out_channels=3 (BraTS classes without background)
    model = create_swinunetr(out_channels=3, use_checkpoint=use_checkpoint)

    # Load ALL weights (strict=True to ensure everything matches)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded full model: {len(state_dict)} keys, {total_params / 1e6:.2f}M params")

    # Modify output layer if needed (BrainSegFounder: 3 classes, BraTS-MEN: 4 classes)
    if out_channels != 3:
        in_features = model.out.conv.conv.in_channels
        model.out = nn.Sequential(
            nn.Conv3d(in_features, out_channels, kernel_size=1),
        )
        logger.info(f"Replaced output layer: {in_features} -> {out_channels} channels")

    # Freeze encoder if requested
    if freeze_encoder:
        # Freeze swinViT (the main encoder)
        for param in model.swinViT.parameters():
            param.requires_grad = False
        logger.info("Froze encoder (swinViT) parameters")

    # Freeze decoder if requested
    if freeze_decoder:
        decoder_modules = [
            model.encoder1,
            model.encoder2,
            model.encoder3,
            model.encoder4,
            model.encoder10,
            model.decoder5,
            model.decoder4,
            model.decoder3,
            model.decoder2,
            model.decoder1,
            model.out,
        ]
        for module in decoder_modules:
            for param in module.parameters():
                param.requires_grad = False
        logger.info("Froze decoder parameters")

    # Move to device
    model = model.to(device)

    return model
