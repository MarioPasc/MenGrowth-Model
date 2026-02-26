#!/usr/bin/env python
# experiments/sdp/train_sdp.py
"""Train the Supervised Disentangled Projection (SDP) network.

Loads precomputed features from HDF5 files, trains the SDP with
curriculum scheduling, and outputs a self-contained run directory.

Outputs (structured run directory):
    outputs/sdp/{run_name}/
    ├── meta/              — config snapshot + data manifest
    ├── checkpoints/       — phase2_sdp.pt
    ├── training/          — CSV logs with losses + inline eval metrics
    ├── latent/            — z, partitions, predictions, targets per split
    ├── evaluation/        — quality_report.json + full eval (if --skip-eval not set)
    ├── figures/           — publication figures (if --skip-eval not set)
    └── tables/            — CSV + LaTeX tables (if --skip-eval not set)

Usage:
    python -m experiments.sdp.train_sdp \
        --config experiments/sdp/config/sdp_default.yaml

    # Skip post-training evaluation for fast iteration:
    python -m experiments.sdp.train_sdp \
        --config experiments/sdp/config/sdp_default.yaml --skip-eval

    # Override splits:
    python -m experiments.sdp.train_sdp \
        --config experiments/sdp/config/sdp_default.yaml \
        --train-splits lora_train sdp_train --val-split lora_val
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from experiments.sdp.output_paths import (
    create_run_dir,
    save_data_manifest,
    save_run_config,
)
from growth.training.train_sdp import (
    build_sdp_module,
    generate_quality_report,
    load_and_combine_splits,
    load_precomputed_features,
)
from growth.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_latent_vectors_h5(
    module: "SDPLitModule",
    features_dir: str,
    output_dir: str,
    split_names: list[str],
    targets_by_split: dict[str, dict[str, torch.Tensor]] | None = None,
    shape_indices: list[int] | None = None,
) -> None:
    """Project all splits through trained SDP and save latent vectors + targets.

    Saves one .h5 file per split: {output_dir}/latent_{split}.h5
    containing: z, partitions, predictions, targets (unnormalized), and subject_ids.

    Args:
        module: Trained SDPLitModule.
        features_dir: Directory with per-split feature .h5 files.
        output_dir: Directory for output latent .h5 files.
        split_names: List of split names to process.
        targets_by_split: Optional dict of {split: {key: tensor}} with
            unnormalized targets to save alongside latent vectors.
        shape_indices: Optional list of shape target column indices to keep.
    """
    module.eval()
    device = next(module.parameters()).device
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in split_names:
        h5_in = Path(features_dir) / f"{split}.h5"
        if not h5_in.exists():
            logger.warning(f"Skipping {split}: {h5_in} not found")
            continue

        # Load features and targets from source H5
        h, raw_targets = load_precomputed_features(
            str(h5_in), shape_indices=shape_indices
        )

        # Load subject IDs
        with h5py.File(h5_in, "r") as f:
            subject_ids = list(f["subject_ids"][()])

        # Normalize and project
        h_norm = module._normalize_features(h.to(device))
        with torch.no_grad():
            z, partitions, predictions = module.model(h_norm)

        # Use provided targets if available, otherwise use raw from source
        targets_to_save = (
            targets_by_split.get(split, raw_targets)
            if targets_by_split is not None
            else raw_targets
        )

        # Save as HDF5
        h5_out = output_path / f"latent_{split}.h5"
        with h5py.File(h5_out, "w") as f:
            f.create_dataset("z", data=z.cpu().float().numpy(), compression="gzip")

            part_grp = f.create_group("partitions")
            for name, tensor in partitions.items():
                part_grp.create_dataset(name, data=tensor.cpu().float().numpy(), compression="gzip")

            pred_grp = f.create_group("predictions")
            for name, tensor in predictions.items():
                pred_grp.create_dataset(name, data=tensor.cpu().float().numpy(), compression="gzip")

            # Save unnormalized targets for downstream evaluation
            # Use long-form keys matching evaluate_sdp.py expectations
            target_key_map = {"vol": "volume", "loc": "location", "shape": "shape"}
            tgt_grp = f.create_group("targets")
            for name, tensor in targets_to_save.items():
                data = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
                save_key = target_key_map.get(name, name)
                tgt_grp.create_dataset(save_key, data=data, compression="gzip")

            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("subject_ids", data=subject_ids, dtype=dt)

        logger.info(f"Saved latent vectors for {split}: {h5_out}")


def main(
    config_path: str,
    train_splits: list[str] | None = None,
    val_split: str | None = None,
    run_name: str | None = None,
    skip_eval: bool = False,
) -> str:
    """Main training entry point.

    Args:
        config_path: Path to experiment config.
        train_splits: Override training split names.
        val_split: Override validation split name.
        run_name: Optional name for the run directory.
        skip_eval: If True, skip post-training evaluation and visualization.

    Returns:
        Path to the run directory.
    """
    cfg = OmegaConf.load(config_path)
    set_seed(cfg.training.seed)

    features_dir = cfg.paths.features_dir

    # Determine splits
    train_split_names = train_splits or list(cfg.data.train_splits)
    val_split_name = val_split or cfg.data.val_split
    test_split_name = cfg.data.get("test_split", "test")

    logger.info(f"Train splits: {train_split_names}")
    logger.info(f"Val split: {val_split_name}")

    # Create structured run directory
    base_dir = cfg.paths.get("output_dir", "outputs/sdp")
    paths = create_run_dir(base_dir=base_dir, run_name=run_name)
    save_run_config(paths, cfg)

    # Shape indices filtering (drop redundant features like surface_area_log)
    shape_indices = list(cfg.targets.shape_indices) if cfg.targets.get("shape_indices") else None

    # Load features
    h_train, targets_train = load_and_combine_splits(
        features_dir, train_split_names, shape_indices=shape_indices
    )
    h_val, targets_val = load_precomputed_features(
        str(Path(features_dir) / f"{val_split_name}.h5"), shape_indices=shape_indices
    )

    logger.info(f"Train: {h_train.shape[0]} samples, Val: {h_val.shape[0]} samples")

    # Save data manifest
    n_test = 0
    if test_split_name:
        test_h5 = Path(features_dir) / f"{test_split_name}.h5"
        if test_h5.exists():
            with h5py.File(test_h5, "r") as f:
                n_test = f["features/encoder10"].shape[0]

    save_data_manifest(
        paths,
        n_train=h_train.shape[0],
        n_val=h_val.shape[0],
        n_test=n_test,
        feature_dim=h_train.shape[1],
        target_dims={
            "vol": targets_train["vol"].shape[1],
            "loc": targets_train["loc"].shape[1],
            "shape": targets_train["shape"].shape[1],
        },
    )

    # Build module
    module = build_sdp_module(cfg)
    module.setup_data(h_train, targets_train, h_val, targets_val)

    # CSV logger into structured run dir
    csv_logger = pl.loggers.CSVLogger(
        save_dir=str(paths.training),
        name="csv_log",
    )

    # Lightning trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.get("accelerator", "auto"),
        devices=cfg.training.get("devices", 1),
        precision=cfg.training.get("precision", "32-true"),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 1),
        default_root_dir=str(paths.training),
        logger=[csv_logger],
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(module)

    # Save SDP checkpoint
    module.save_sdp_checkpoint(str(paths.checkpoint_path))

    # Generate quality report
    report = generate_quality_report(module, h_val, targets_val)

    with open(paths.quality_report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Quality report saved to {paths.quality_report_path}")

    # Print report
    logger.info("\n" + "=" * 50)
    logger.info("Phase 2 Quality Report")
    logger.info("=" * 50)
    for key, value in report.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("=" * 50)

    # Check BLOCKING thresholds
    blocking_ok = True
    thresholds = [
        ("r2_vol", 0.80, ">="),
        ("r2_loc", 0.85, ">="),
        ("r2_shape", 0.30, ">="),
        ("max_cross_partition_corr", 0.30, "<="),
    ]
    for key, threshold, direction in thresholds:
        val = report[key]
        if direction == ">=" and val < threshold:
            logger.warning(f"BLOCKING: {key}={val:.4f} < {threshold}")
            blocking_ok = False
        elif direction == "<=" and val > threshold:
            logger.warning(f"BLOCKING: {key}={val:.4f} > {threshold}")
            blocking_ok = False

    if blocking_ok:
        logger.info("All BLOCKING thresholds passed!")
    else:
        logger.warning("Some BLOCKING thresholds failed. See recovery steps in module_3_sdp.md")

    # Save latent vectors with targets for all splits
    all_splits = train_split_names + [val_split_name]
    if test_split_name:
        all_splits.append(test_split_name)

    save_latent_vectors_h5(
        module=module,
        features_dir=features_dir,
        output_dir=str(paths.latent),
        split_names=all_splits,
        shape_indices=shape_indices,
    )

    # Auto-trigger evaluation and visualization (skippable)
    if not skip_eval:
        try:
            from experiments.sdp.evaluate_sdp import main as evaluate_main

            logger.info("Running post-training evaluation...")
            evaluate_main(str(paths.root))
        except Exception as e:
            logger.warning(f"Post-training evaluation failed: {e}")

        try:
            from experiments.sdp.visualize_sdp import main as visualize_main

            logger.info("Generating publication figures...")
            visualize_main(str(paths.root))
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

    logger.info(f"Run complete. All outputs in: {paths.root}")
    return str(paths.root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SDP (Phase 2)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-splits", nargs="+", default=None)
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training evaluation and visualization",
    )

    args = parser.parse_args()
    main(
        args.config,
        args.train_splits,
        args.val_split,
        args.run_name,
        args.skip_eval,
    )
