#!/usr/bin/env python
# experiments/sdp/extract_all_features.py
"""Extract encoder features + semantic targets for all subjects.

Saves per-split HDF5 files: {features_dir}/{split_name}.h5
Each file contains:
    features/encoder10  [N, 768]
    targets/volume      [N, 4]
    targets/location    [N, 3]
    targets/shape       [N, 3]
    subject_ids         [N] (string dataset)

Splits are loaded from:
    1. H5 file (if paths.h5_file is set)  — preferred, no external JSON needed
    2. JSON file via data.splits_config    — auto-generated if missing

Encoder is loaded from:
    1. paths.lora_checkpoint directly      — preferred, no LoRA config needed
    2. data.splits_config LoRA config      — fallback

Usage:
    python -m experiments.sdp.extract_all_features \
        --config experiments/sdp/config/sdp_default.yaml
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from experiments.lora_ablation.extract_features import (
    extract_features_for_split,
)
from growth.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_splits(cfg: OmegaConf) -> dict[str, list[str]]:
    """Load data splits, preferring H5 then JSON (auto-generated if needed).

    Args:
        cfg: Full SDP experiment config.

    Returns:
        Dict mapping split names to subject ID lists.
    """
    h5_path = cfg.get("paths", {}).get("h5_file", None)

    # 1. Try H5 splits (embedded in the dataset file)
    if h5_path and Path(str(h5_path)).exists():
        from experiments.lora_ablation.data_splits import load_splits_h5

        logger.info(f"Loading splits from H5: {h5_path}")
        return load_splits_h5(str(h5_path))

    # 2. Try JSON splits from LoRA ablation output
    splits_config = cfg.data.get("splits_config", None)
    if splits_config:
        from experiments.lora_ablation.data_splits import load_splits

        try:
            return load_splits(splits_config)
        except FileNotFoundError:
            logger.warning("Splits JSON not found. Auto-generating...")
            from experiments.lora_ablation.data_splits import main as generate_splits

            generate_splits(splits_config)
            return load_splits(splits_config)

    raise RuntimeError("Cannot load splits: no paths.h5_file and no data.splits_config in config.")


def _load_encoder(cfg: OmegaConf, device: str) -> torch.nn.Module:
    """Load LoRA-adapted encoder, preferring direct path then LoRA config.

    Args:
        cfg: Full SDP experiment config.
        device: Torch device string.

    Returns:
        Merged SwinUNETR encoder in eval mode.
    """
    lora_ckpt = cfg.get("paths", {}).get("lora_checkpoint", None)
    ckpt_dir = cfg.get("paths", {}).get("checkpoint_dir", None)

    # 1. Direct loading from SDP config paths
    if lora_ckpt and ckpt_dir:
        lora_path = Path(str(lora_ckpt))
        ckpt_path = Path(str(ckpt_dir))

        # Resolve BrainSegFounder checkpoint file
        if ckpt_path.is_dir():
            ckpt_file = ckpt_path / "finetuned_model_fold_0.pt"
            if not ckpt_file.exists():
                # Try any .pt file
                pt_files = sorted(ckpt_path.glob("*.pt"))
                if pt_files:
                    ckpt_file = pt_files[0]
                else:
                    raise FileNotFoundError(f"No .pt checkpoint found in {ckpt_path}")
        else:
            ckpt_file = ckpt_path

        # Resolve LoRA adapter directory
        adapter_path = lora_path / "adapter" if (lora_path / "adapter").exists() else lora_path

        if ckpt_file.exists() and adapter_path.exists():
            from growth.models.encoder.lora_adapter import LoRASwinViT
            from growth.models.encoder.swin_loader import load_swin_encoder

            logger.info(f"Loading base encoder from {ckpt_file}")
            base_encoder = load_swin_encoder(ckpt_file, freeze=True, device=device)

            logger.info(f"Loading LoRA adapter from {adapter_path}")
            lora_encoder = LoRASwinViT.load_lora(
                base_encoder,
                adapter_path,
                device=device,
                trainable=False,
            )
            encoder = lora_encoder.merge_lora()
            logger.info("Merged LoRA weights into base encoder")
            encoder.eval()
            return encoder

    # 2. Fallback: load via LoRA ablation config
    splits_config = cfg.data.get("splits_config", None)
    if splits_config:
        from experiments.lora_ablation.extract_features import load_encoder_for_condition

        logger.info("Loading encoder via LoRA ablation config (fallback)")
        with open(splits_config) as f:
            lora_config = yaml.safe_load(f)
        return load_encoder_for_condition("lora_r8", lora_config, device)

    raise RuntimeError(
        "Cannot load encoder: need either (paths.checkpoint_dir + paths.lora_checkpoint) "
        "or data.splits_config in config."
    )


def save_features_h5(
    features_dict: dict[str, np.ndarray],
    targets_dict: dict[str, np.ndarray],
    subject_ids: list[str],
    save_path: Path,
) -> None:
    """Save features and targets to a single HDF5 file.

    Args:
        features_dict: Dict of feature arrays {level: [N, D]}.
        targets_dict: Dict of target arrays {name: [N, K]}.
        subject_ids: List of subject ID strings.
        save_path: Output .h5 file path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, "w") as f:
        # Features group
        feat_grp = f.create_group("features")
        for key, arr in features_dict.items():
            feat_grp.create_dataset(key, data=arr, compression="gzip")

        # Targets group
        tgt_grp = f.create_group("targets")
        for key, arr in targets_dict.items():
            tgt_grp.create_dataset(key, data=arr, compression="gzip")

        # Subject IDs (variable-length strings)
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("subject_ids", data=subject_ids, dtype=dt)

    logger.info(f"Saved {len(subject_ids)} subjects to {save_path}")


def main(config_path: str, device: str = "cuda") -> None:
    """Extract features for all splits and save as HDF5.

    Args:
        config_path: Path to SDP experiment config.
        device: Device for inference.
    """
    cfg = OmegaConf.load(config_path)

    set_seed(cfg.training.seed)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Load data splits (H5 → JSON → auto-generate)
    splits = _load_splits(cfg)

    # Load merged LoRA encoder (direct paths → LoRA config fallback)
    encoder = _load_encoder(cfg, device)

    # Feature extraction config
    fe_cfg = cfg.get("feature_extraction", {})
    batch_size = fe_cfg.get("batch_size", 1)
    num_workers = fe_cfg.get("num_workers", 2)

    features_dir = Path(OmegaConf.to_container(cfg, resolve=True)["paths"]["features_dir"])

    # H5 path for data backend (optional, faster I/O)
    data_h5_path = cfg.get("paths", {}).get("h5_file", None)
    if data_h5_path:
        logger.info(f"Using H5 backend: {data_h5_path}")

    # Extract for each split
    all_splits = list(splits.keys())
    logger.info(f"Extracting features for splits: {all_splits}")

    for split_name in all_splits:
        subject_ids = splits[split_name]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Split: {split_name} ({len(subject_ids)} subjects)")
        logger.info(f"{'=' * 60}")

        features_dict, targets_dict, extracted_ids = extract_features_for_split(
            encoder=encoder,
            subject_ids=subject_ids,
            data_root=cfg.paths.data_root,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            feature_level="encoder10",
            h5_path=data_h5_path,
            h5_split=split_name if data_h5_path else None,
        )

        # Save as HDF5
        out_path = features_dir / f"{split_name}.h5"
        save_features_h5(features_dict, targets_dict, extracted_ids, out_path)

    logger.info(f"\nAll features saved to {features_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for SDP training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.device)
