"""Curriculum learning visualization and utilities.

This module provides ASCII visualization of training schedules for
semi-supervised VAE with staged semantic supervision.
"""

import logging
from typing import Dict, Optional
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_schedule_value(epoch: int, target: float, start: int, anneal: int) -> float:
    """Compute schedule value at given epoch.

    Args:
        epoch: Current epoch
        target: Target lambda value
        start: Start epoch
        anneal: Annealing epochs

    Returns:
        Current effective lambda value
    """
    if epoch < start:
        return 0.0
    effective = epoch - start
    if effective >= anneal:
        return target
    return target * (effective / anneal)


def print_curriculum_schedule(cfg: DictConfig, width: int = 80) -> None:
    """Print ASCII visualization of curriculum learning schedule.

    Args:
        cfg: Configuration with loss schedule parameters
        width: Total width of the visualization
    """
    max_epochs = cfg.train.max_epochs

    # Extract schedule parameters
    schedules = {}

    # Volume schedule
    vol_start = cfg.loss.get("vol_start_epoch", -1)
    if vol_start < 0:
        vol_start = cfg.loss.get("semantic_start_epoch", 10)
    vol_anneal = cfg.loss.get("vol_annealing_epochs", -1)
    if vol_anneal < 0:
        vol_anneal = cfg.loss.get("semantic_annealing_epochs", 20)
    vol_lambda = cfg.loss.get("lambda_vol", 10.0)
    schedules["Volume"] = (vol_start, vol_start + vol_anneal, vol_lambda, "vol")

    # Location schedule
    loc_start = cfg.loss.get("loc_start_epoch", -1)
    if loc_start < 0:
        loc_start = cfg.loss.get("semantic_start_epoch", 10)
    loc_anneal = cfg.loss.get("loc_annealing_epochs", -1)
    if loc_anneal < 0:
        loc_anneal = cfg.loss.get("semantic_annealing_epochs", 20)
    loc_lambda = cfg.loss.get("lambda_loc", 5.0)
    schedules["Location"] = (loc_start, loc_start + loc_anneal, loc_lambda, "loc")

    # Shape schedule
    shape_start = cfg.loss.get("shape_start_epoch", -1)
    if shape_start < 0:
        shape_start = cfg.loss.get("semantic_start_epoch", 10)
    shape_anneal = cfg.loss.get("shape_annealing_epochs", -1)
    if shape_anneal < 0:
        shape_anneal = cfg.loss.get("semantic_annealing_epochs", 20)
    shape_lambda = cfg.loss.get("lambda_shape", 5.0)
    schedules["Shape"] = (shape_start, shape_start + shape_anneal, shape_lambda, "shape")

    # Cross-partition schedule
    cross_start = cfg.loss.get("cross_partition_start_epoch", 10)
    cross_anneal = cfg.loss.get("semantic_annealing_epochs", 20)
    cross_lambda = cfg.loss.get("lambda_cross_partition", 5.0)
    schedules["Cross-part"] = (cross_start, cross_start + cross_anneal, cross_lambda, "cross")

    # TC schedule
    tc_start = cfg.loss.get("tc_start_epoch", 10)
    tc_anneal = cfg.loss.get("tc_annealing_epochs", 20)
    tc_lambda = cfg.loss.get("lambda_tc", 2.0)
    schedules["TC"] = (tc_start, tc_start + tc_anneal, tc_lambda, "tc")

    # Manifold schedule
    manifold_start = cfg.loss.get("manifold_start_epoch", 10)
    manifold_lambda = cfg.loss.get("lambda_manifold", 1.0)
    if manifold_lambda > 0:
        schedules["Manifold"] = (manifold_start, manifold_start + 20, manifold_lambda, "manifold")

    # Print header
    print()
    print("=" * width)
    print("CURRICULUM LEARNING SCHEDULE".center(width))
    print("=" * width)

    # Print epoch scale
    label_width = 12
    bar_width = width - label_width - 2

    # Create epoch markers
    markers = [0]
    step = max_epochs // 6
    for i in range(1, 6):
        markers.append(i * step)
    markers.append(max_epochs)

    # Print epoch header
    header = " " * label_width + "|"
    for m in markers:
        pos = int((m / max_epochs) * bar_width)
        header = header[:label_width + 1 + pos] + str(m) + header[label_width + 1 + pos + len(str(m)):]

    # Build clean header
    epoch_line = " " * label_width + "|"
    for i, m in enumerate(markers):
        pos = int((m / max_epochs) * bar_width)
        marker_str = str(m)
        if i == 0:
            epoch_line += marker_str
        else:
            current_len = len(epoch_line) - label_width - 1
            padding = pos - current_len
            if padding > 0:
                epoch_line += " " * (padding - len(marker_str)) + marker_str

    print(epoch_line)
    print(" " * label_width + "|" + "-" * bar_width)

    # Sort schedules by start epoch
    sorted_schedules = sorted(schedules.items(), key=lambda x: x[1][0])

    # Print each schedule
    for name, (start, full, lam, key) in sorted_schedules:
        label = f"{name:>10} |"

        # Build the bar
        bar = []
        for i in range(bar_width):
            epoch = int((i / bar_width) * max_epochs)
            if epoch < start:
                bar.append(" ")
            elif epoch < full:
                # Annealing phase
                bar.append("~")
            else:
                # Full strength
                bar.append("#")

        bar_str = "".join(bar)

        # Add lambda annotation at the end
        annotation = f" [start:{start}, full:{full}, l={lam}]"

        print(f"{label}{bar_str}")

    print(" " * label_width + "|" + "-" * bar_width)
    print()

    # Print legend
    print("Legend: [ ] = inactive  [~] = annealing  [#] = full strength")
    print()

    # Print schedule summary table
    print("Schedule Summary:")
    print("-" * 60)
    print(f"{'Component':<12} {'Start':>8} {'Full':>8} {'Lambda':>10}")
    print("-" * 60)
    for name, (start, full, lam, _) in sorted_schedules:
        print(f"{name:<12} {start:>8} {full:>8} {lam:>10.1f}")
    print("-" * 60)
    print()


def print_latent_partitioning(cfg: DictConfig, width: int = 80) -> None:
    """Print ASCII visualization of latent space partitioning.

    Args:
        cfg: Configuration with latent partitioning
        width: Total width
    """
    if not cfg.model.get("latent_partitioning", {}).get("enabled", False):
        return

    z_dim = cfg.model.z_dim
    partitions = cfg.model.latent_partitioning

    print("=" * width)
    print("LATENT SPACE PARTITIONING".center(width))
    print("=" * width)

    # Collect partition info
    parts = []
    for name in ["z_vol", "z_loc", "z_shape", "z_residual"]:
        if name in partitions:
            p = partitions[name]
            start = p.get("start_idx", 0)
            dim = p.get("dim", 0)
            sup = p.get("supervision", "none")
            features = p.get("target_features", [])
            n_features = len(features) if features else 0
            parts.append((name, start, dim, sup, n_features))

    # Sort by start index
    parts.sort(key=lambda x: x[1])

    # Print visual bar
    bar_width = 64
    print()
    print(f"z_dim = {z_dim}")
    print()

    # Build the partition bar
    bar = [" "] * bar_width
    labels_above = []

    for name, start, dim, sup, n_feat in parts:
        start_pos = int((start / z_dim) * bar_width)
        end_pos = int(((start + dim) / z_dim) * bar_width)

        # Choose character based on supervision
        if sup == "regression":
            char = "#"
        else:
            char = "."

        for i in range(start_pos, min(end_pos, bar_width)):
            bar[i] = char

        # Center label
        mid_pos = (start_pos + end_pos) // 2
        labels_above.append((mid_pos, name, dim))

    # Print labels
    label_line = [" "] * bar_width
    for pos, name, dim in labels_above:
        label = f"{name}({dim})"
        start = max(0, pos - len(label) // 2)
        for i, c in enumerate(label):
            if start + i < bar_width:
                label_line[start + i] = c

    print("".join(label_line))
    print("|" + "".join(bar) + "|")
    print("0" + " " * (bar_width - len(str(z_dim))) + str(z_dim))
    print()

    # Print partition table
    print(f"{'Partition':<12} {'Dims':>6} {'Range':>12} {'Supervision':>12} {'Features':>10}")
    print("-" * 60)
    for name, start, dim, sup, n_feat in parts:
        range_str = f"[{start}:{start+dim}]"
        feat_str = str(n_feat) if n_feat > 0 else "-"
        print(f"{name:<12} {dim:>6} {range_str:>12} {sup:>12} {feat_str:>10}")
    print("-" * 60)
    print()
    print("Legend: [#] = supervised (regression)  [.] = unsupervised (KL prior)")
    print()


def print_model_summary(cfg: DictConfig, model, width: int = 80) -> None:
    """Print comprehensive model configuration summary.

    Args:
        cfg: Configuration object
        model: The model instance
        width: Total width
    """
    print()
    print("=" * width)
    print("MODEL CONFIGURATION SUMMARY".center(width))
    print("=" * width)
    print()

    # Data configuration
    print("DATA:")
    print(f"  Resolution:     {cfg.data.spacing[0]}mm isotropic")
    print(f"  ROI size:       {cfg.data.roi_size[0]}x{cfg.data.roi_size[1]}x{cfg.data.roi_size[2]}")
    print(f"  Modalities:     {', '.join(cfg.data.modalities)}")
    print(f"  Batch size:     {cfg.data.batch_size}")
    print()

    # Architecture
    print("ARCHITECTURE:")
    from vae.models.components.sbd import SpatialBroadcastDecoder
    has_sbd = isinstance(model.decoder, SpatialBroadcastDecoder)
    decoder_type = "SpatialBroadcastDecoder" if has_sbd else "TransposedConv"
    print(f"  Encoder:        3D ResNet-style")
    print(f"  Decoder:        {decoder_type}")
    print(f"  z_dim:          {cfg.model.z_dim}")
    print(f"  Base filters:   {cfg.model.base_filters}")
    print(f"  Spectral norm:  {cfg.model.get('use_spectral_norm', False)}")
    print()

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print("PARAMETERS:")
    print(f"  Total:          {total_params:,}")
    print(f"  Trainable:      {trainable_params:,}")
    print(f"  Encoder:        {encoder_params:,} ({100*encoder_params/total_params:.1f}%)")
    print(f"  Decoder:        {decoder_params:,} ({100*decoder_params/total_params:.1f}%)")
    print()

    # Training configuration
    print("TRAINING:")
    print(f"  Max epochs:     {cfg.train.max_epochs}")
    print(f"  Learning rate:  {cfg.train.lr}")
    print(f"  Weight decay:   {cfg.train.weight_decay}")
    print(f"  Precision:      {cfg.train.get('precision', 'fp32')}")
    print(f"  Grad clip:      {cfg.train.get('gradient_clip_val', 'none')}")
    print(f"  GPUs:           {cfg.train.get('devices', 1)}")
    print()

    # KL configuration
    print("KL REGULARIZATION:")
    print(f"  Beta target:    {cfg.train.kl_beta}")
    print(f"  Annealing:      {cfg.train.kl_annealing_type} ({cfg.train.kl_annealing_epochs} epochs)")
    print(f"  Free bits:      {cfg.train.get('kl_free_bits', 0)}")
    print()


def log_curriculum_schedule(cfg: DictConfig) -> None:
    """Log curriculum schedule to the logger.

    This is a logger-friendly version that doesn't use print().

    Args:
        cfg: Configuration with loss schedule parameters
    """
    max_epochs = cfg.train.max_epochs

    # Extract schedules
    schedules = []

    # Volume
    vol_start = cfg.loss.get("vol_start_epoch", cfg.loss.get("semantic_start_epoch", 10))
    if vol_start < 0:
        vol_start = cfg.loss.get("semantic_start_epoch", 10)
    vol_anneal = cfg.loss.get("vol_annealing_epochs", cfg.loss.get("semantic_annealing_epochs", 20))
    if vol_anneal < 0:
        vol_anneal = cfg.loss.get("semantic_annealing_epochs", 20)
    schedules.append(("Volume", vol_start, vol_start + vol_anneal, cfg.loss.get("lambda_vol", 10.0)))

    # Location
    loc_start = cfg.loss.get("loc_start_epoch", cfg.loss.get("semantic_start_epoch", 10))
    if loc_start < 0:
        loc_start = cfg.loss.get("semantic_start_epoch", 10)
    loc_anneal = cfg.loss.get("loc_annealing_epochs", cfg.loss.get("semantic_annealing_epochs", 20))
    if loc_anneal < 0:
        loc_anneal = cfg.loss.get("semantic_annealing_epochs", 20)
    schedules.append(("Location", loc_start, loc_start + loc_anneal, cfg.loss.get("lambda_loc", 5.0)))

    # Shape
    shape_start = cfg.loss.get("shape_start_epoch", cfg.loss.get("semantic_start_epoch", 10))
    if shape_start < 0:
        shape_start = cfg.loss.get("semantic_start_epoch", 10)
    shape_anneal = cfg.loss.get("shape_annealing_epochs", cfg.loss.get("semantic_annealing_epochs", 20))
    if shape_anneal < 0:
        shape_anneal = cfg.loss.get("semantic_annealing_epochs", 20)
    schedules.append(("Shape", shape_start, shape_start + shape_anneal, cfg.loss.get("lambda_shape", 5.0)))

    # TC
    tc_start = cfg.loss.get("tc_start_epoch", 10)
    tc_anneal = cfg.loss.get("tc_annealing_epochs", 20)
    schedules.append(("TC", tc_start, tc_start + tc_anneal, cfg.loss.get("lambda_tc", 2.0)))

    # Cross-partition
    cross_start = cfg.loss.get("cross_partition_start_epoch", 10)
    schedules.append(("Cross-part", cross_start, cross_start + 20, cfg.loss.get("lambda_cross_partition", 5.0)))

    # Sort and log
    schedules.sort(key=lambda x: x[1])

    logger.info("")
    logger.info("Curriculum Learning Schedule:")
    logger.info("-" * 50)
    logger.info(f"{'Component':<12} {'Start':>8} {'Full':>8} {'Lambda':>10}")
    logger.info("-" * 50)
    for name, start, full, lam in schedules:
        logger.info(f"{name:<12} {start:>8} {full:>8} {lam:>10.1f}")
    logger.info("-" * 50)
    logger.info(f"Max epochs: {max_epochs}")
    logger.info("")
