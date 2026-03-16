#!/usr/bin/env python
# experiments/lora/engine/extract_features.py
"""Feature extraction with multi-scale support (single-domain and per-domain).

Supports:
- Multi-scale features: encoder10 (768-dim), layers2+3+4 (1344-dim concat)
- Tumor-Aware Pooling (TAP): seg-mask-weighted features
- Per-domain extraction: separate MEN and GLI feature sets
- Single-domain extraction: MEN-only (lora_ablation compat)
- H5 backend with 192³ ROI for feature extraction

Usage:
    python -m experiments.lora.run --config <yaml> extract --condition <name>
"""

import argparse
import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from growth.data.bratsmendata import BraTSDatasetH5
from growth.data.transforms import FEATURE_ROI_SIZE, get_h5_val_transforms
from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_swin_encoder
from growth.utils.seed import set_seed

from .data_splits import load_splits_h5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Feature Extractors
# =============================================================================


class MultiScaleFeatureExtractor:
    """Extract features at multiple scales from SwinUNETR.

    Extracts GAP features from:
    - layers2: 192-dim
    - layers3: 384-dim
    - layers4: 768-dim
    - encoder10: 768-dim (bottleneck)

    The multi-scale representation (1344-dim) captures both
    local and global information.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        device: str = "cuda",
    ):
        self.encoder = encoder
        self.device = device
        self.encoder.eval()

    @torch.no_grad()
    def extract(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor [B, 4, D, H, W] (e.g. [B, 4, 192, 192, 192]).

        Returns:
            Dict with 'layers2', 'layers3', 'layers4', 'encoder10', 'multi_scale'.
        """
        hidden_states = self.encoder.swinViT(x, self.encoder.normalize)

        features: dict[str, torch.Tensor] = {}

        # Stage outputs with GAP
        features["layers2"] = F.adaptive_avg_pool3d(hidden_states[2], 1).flatten(1)  # [B, 192]
        features["layers3"] = F.adaptive_avg_pool3d(hidden_states[3], 1).flatten(1)  # [B, 384]
        features["layers4"] = F.adaptive_avg_pool3d(hidden_states[4], 1).flatten(1)  # [B, 768]

        # encoder10: bottleneck
        enc10 = self.encoder.encoder10(hidden_states[4])
        features["encoder10"] = F.adaptive_avg_pool3d(enc10, 1).flatten(1)  # [B, 768]

        # Multi-scale concatenation
        features["multi_scale"] = torch.cat(
            [features["layers2"], features["layers3"], features["layers4"]],
            dim=1,
        )  # [B, 1344]

        return features


class TumorAwarePoolExtractor:
    """Extract tumor-aware pooled features using segmentation mask weighting.

    Instead of GAP (Global Average Pooling) which pools equally over all
    spatial positions, TAP uses the predicted WT (Whole Tumor) segmentation
    probability to weight encoder features.

    This addresses the signal dilution problem: tumors occupy 1-5% of the
    192^3 volume, so 95%+ of GAP signal is non-tumor tissue.

    Requires the full model (encoder + decoder) to produce segmentation.

    Args:
        full_model: Full model with get_hidden_states() and forward() methods.
        device: Device to use.
        floor: Minimum weight for non-tumor regions (prevents zero-weight).
    """

    def __init__(
        self,
        full_model: torch.nn.Module,
        device: str = "cuda",
        floor: float = 0.01,
    ):
        self.full_model = full_model
        self.device = device
        self.floor = floor
        self.full_model.eval()

        self.encoder10 = self._find_encoder10(full_model)

    @staticmethod
    def _find_encoder10(model: torch.nn.Module) -> torch.nn.Module:
        """Find encoder10 module across different model architectures.

        Args:
            model: Any ablation model type.

        Returns:
            The encoder10 module.
        """
        if hasattr(model, "decoder") and hasattr(model.decoder, "encoder10"):
            return model.decoder.encoder10
        if hasattr(model, "model"):
            if hasattr(model.model, "encoder10"):
                return model.model.encoder10
            if hasattr(model.model, "decoder") and hasattr(model.model.decoder, "encoder10"):
                return model.model.decoder.encoder10
        raise AttributeError(f"Cannot find encoder10 on model type {type(model).__name__}")

    @torch.no_grad()
    def extract(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract TAP features.

        Args:
            x: Input tensor [B, 4, D, H, W].

        Returns:
            Dict with 'encoder10_tap' (768-dim TAP features) and
            'encoder10_gap' (768-dim GAP features for comparison).
        """
        hidden_states = self.full_model.get_hidden_states(x)

        enc10 = self.encoder10(hidden_states[4])

        # GAP baseline
        gap_features = F.adaptive_avg_pool3d(enc10, 1).flatten(1)  # [B, 768]

        # Get segmentation prediction for WT mask
        pred = self.full_model(x)  # [B, 3, D, H, W]
        wt_prob = torch.sigmoid(pred[:, 1:2])  # WT channel [B, 1, D, H, W]

        # Downsample WT probability to encoder10 spatial size
        enc_spatial = enc10.shape[2:]
        mask = F.adaptive_avg_pool3d(wt_prob, enc_spatial)

        # Apply floor and normalize
        mask = mask + self.floor
        mask = mask / mask.sum(dim=(2, 3, 4), keepdim=True)

        # Tumor-aware pooling: weighted sum over spatial dims
        tap_features = (enc10 * mask).sum(dim=(2, 3, 4))  # [B, 768]

        return {
            "encoder10_gap": gap_features,
            "encoder10_tap": tap_features,
        }


# =============================================================================
# Model Loading
# =============================================================================


def load_encoder_for_condition(
    condition_name: str,
    config: dict,
    device: str,
) -> torch.nn.Module:
    """Load encoder for a condition with proper weight merging.

    Args:
        condition_name: Condition name.
        config: Full experiment configuration.
        device: Device.

    Returns:
        SwinUNETR encoder in eval mode.
    """
    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name

    condition_config = None
    for cond in config["conditions"]:
        if cond["name"] == condition_name:
            condition_config = cond
            break

    if condition_config is None:
        raise ValueError(f"Unknown condition: {condition_name}")

    is_baseline = condition_config.get("lora_rank") is None

    base_encoder = load_swin_encoder(
        config["paths"]["checkpoint"],
        freeze=True,
        device=device,
    )

    if is_baseline:
        logger.info("Loaded baseline encoder (no LoRA)")
        encoder = base_encoder
    else:
        adapter_path = condition_dir / "adapter"
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

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


def load_full_model_for_condition(
    condition_name: str,
    config: dict,
    device: str,
) -> torch.nn.Module | None:
    """Load full model (encoder + decoder) for TAP extraction.

    Returns None if no checkpoint is available.

    Args:
        condition_name: Condition name.
        config: Full experiment config.
        device: Device.

    Returns:
        Full model in eval mode, or None.
    """
    from .model_factory import create_ablation_model, get_condition_config

    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name
    best_model_path = condition_dir / "best_model.pt"

    if not best_model_path.exists():
        logger.warning(f"No best_model.pt found for {condition_name}, skipping TAP")
        return None

    condition_config = get_condition_config(config, condition_name)
    training_config = config["training"]

    model = create_ablation_model(
        condition_config=condition_config,
        training_config=training_config,
        checkpoint_path=config["paths"]["checkpoint"],
        device=device,
    )

    state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logger.info(f"Loaded full model for TAP extraction: {condition_name}")
    return model


# =============================================================================
# Per-Domain Feature Extraction
# =============================================================================


@torch.no_grad()
def extract_features_for_domain(
    encoder: torch.nn.Module,
    h5_path: str,
    split: str,
    device: str,
    batch_size: int = 2,
    num_workers: int = 4,
    feature_level: str = "encoder10",
    use_amp: bool = False,
    tap_extractor: TumorAwarePoolExtractor | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Extract features for a single domain/split.

    Args:
        encoder: SwinUNETR encoder.
        h5_path: Path to H5 file.
        split: Split name (e.g. "test", "lora_train").
        device: Device.
        batch_size: Batch size for extraction.
        num_workers: DataLoader workers.
        feature_level: Feature level to extract.
        use_amp: Use bf16 autocast.
        tap_extractor: Optional TAP extractor.

    Returns:
        Tuple of (features_dict, targets_dict, subject_ids).
    """
    dataset = BraTSDatasetH5(
        h5_path=h5_path,
        split=split,
        transform=get_h5_val_transforms(roi_size=FEATURE_ROI_SIZE),
        compute_semantic=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    extractor = MultiScaleFeatureExtractor(encoder, device)

    all_features: dict[str, list[np.ndarray]] = {
        "encoder10": [],
        "multi_scale": [],
        "layers2": [],
        "layers3": [],
        "layers4": [],
    }
    if tap_extractor is not None:
        all_features["encoder10_tap"] = []

    all_volumes: list[np.ndarray] = []
    all_locations: list[np.ndarray] = []
    all_ids: list[str] = []

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    for batch in tqdm(dataloader, desc=f"Extracting [{split}]"):
        images = batch["image"].to(device)

        with amp_ctx:
            features = extractor.extract(images)

        for key in all_features:
            if key == "encoder10_tap":
                continue
            all_features[key].append(features[key].cpu().float().numpy())

        # TAP extraction
        if tap_extractor is not None:
            try:
                with amp_ctx:
                    tap_features = tap_extractor.extract(images)
                all_features["encoder10_tap"].append(
                    tap_features["encoder10_tap"].cpu().float().numpy()
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("TAP extraction OOM — disabling TAP")
                torch.cuda.empty_cache()
                all_features.pop("encoder10_tap", None)
                tap_extractor = None

        # Semantic targets (R1: volume + location only, shape removed)
        all_volumes.append(batch["semantic_features"]["volume"].numpy())
        all_locations.append(batch["semantic_features"]["location"].numpy())

        ids = batch["subject_id"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        all_ids.extend(ids)

    features_dict = {
        key: np.concatenate(vals, axis=0) for key, vals in all_features.items() if vals
    }
    targets_dict = {
        "volume": np.concatenate(all_volumes, axis=0),
        "location": np.concatenate(all_locations, axis=0),
    }

    logger.info(f"  Features: {', '.join(f'{k}={v.shape}' for k, v in features_dict.items())}")
    logger.info(f"  Targets: {', '.join(f'{k}={v.shape}' for k, v in targets_dict.items())}")

    return features_dict, targets_dict, all_ids


# =============================================================================
# Save Features
# =============================================================================


def save_domain_features(
    features_dict: dict[str, np.ndarray],
    targets_dict: dict[str, np.ndarray],
    subject_ids: list[str],
    save_dir: Path,
    domain: str,
    split: str,
) -> None:
    """Save per-domain features and targets.

    Args:
        features_dict: Feature arrays by level.
        targets_dict: Semantic target arrays.
        subject_ids: Subject IDs.
        save_dir: Directory to save to.
        domain: Domain name ("men" or "gli").
        split: Split name ("test" or "probe").
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{domain}_{split}"

    for key, arr in features_dict.items():
        path = save_dir / f"features_{prefix}_{key}.pt"
        torch.save(torch.from_numpy(arr), path)

    # Save primary features (encoder10) with simple name
    if "encoder10" in features_dict:
        torch.save(
            torch.from_numpy(features_dict["encoder10"]),
            save_dir / f"features_{prefix}.pt",
        )

    # Save targets
    torch.save(
        {k: torch.from_numpy(v) for k, v in targets_dict.items()},
        save_dir / f"targets_{prefix}.pt",
    )

    # Save subject IDs
    with open(save_dir / f"subject_ids_{prefix}.txt", "w") as f:
        f.write("\n".join(subject_ids))

    logger.info(f"  Saved {prefix} features ({len(subject_ids)} samples) to {save_dir}")


def save_features(
    features_dict: dict[str, np.ndarray],
    targets_dict: dict[str, np.ndarray],
    subject_ids: list[str],
    save_dir: Path,
    split_name: str,
) -> None:
    """Save features and targets (single-domain compat).

    Args:
        features_dict: Feature arrays by level.
        targets_dict: Semantic target arrays.
        subject_ids: Subject IDs.
        save_dir: Directory to save to.
        split_name: Split name ("test" or "probe").
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for key, arr in features_dict.items():
        path = save_dir / f"features_{split_name}_{key}.pt"
        torch.save(torch.from_numpy(arr), path)
        logger.info(f"Saved {key} features to {path}")

    # Save primary features (for backwards compatibility)
    torch.save(torch.from_numpy(features_dict["encoder10"]), save_dir / f"features_{split_name}.pt")

    # Save targets
    targets_path = save_dir / f"targets_{split_name}.pt"
    torch.save({k: torch.from_numpy(v) for k, v in targets_dict.items()}, targets_path)
    logger.info(f"Saved targets to {targets_path}")

    # Save subject IDs
    ids_path = save_dir / f"subject_ids_{split_name}.txt"
    with open(ids_path, "w") as f:
        f.write("\n".join(subject_ids))


# =============================================================================
# Main Extraction (Per-Domain)
# =============================================================================


def extract_features(
    condition_name: str,
    config: dict,
    device: str = "cuda",
) -> dict[str, Path]:
    """Extract features for a condition.

    Automatically detects dual-domain vs single-domain from config.
    For dual-domain: extracts MEN and GLI separately (test + probe splits).
    For single-domain: extracts MEN only.

    Args:
        condition_name: Condition name.
        config: Full experiment configuration.
        device: Device.

    Returns:
        Dict with paths to saved feature files.
    """
    logger.info(f"Extracting features for condition: {condition_name}")

    output_dir = Path(config["experiment"]["output_dir"])
    condition_dir = output_dir / "conditions" / condition_name
    features_dir = condition_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    encoder = load_encoder_for_condition(condition_name, config, device)

    fe_config = config.get("feature_extraction", {})
    batch_size = fe_config.get("batch_size", 2)
    num_workers = config["training"].get("num_workers", 4)
    feature_level = fe_config.get("level", "encoder10")
    pooling_mode = fe_config.get("pooling_mode", "gap")
    use_amp = config["training"].get("use_amp", False)

    # Optional TAP
    tap_extractor = None
    if pooling_mode in ("tap", "both"):
        full_model = load_full_model_for_condition(condition_name, config, device)
        if full_model is not None:
            tap_extractor = TumorAwarePoolExtractor(full_model, device)
            logger.info("TAP extraction enabled")

    result_paths: dict[str, Path] = {}

    # Detect dual-domain
    is_dual = "men_h5_file" in config.get("paths", {}) and "gli_h5_file" in config.get("paths", {})

    if is_dual:
        men_h5 = config["paths"]["men_h5_file"]
        gli_h5 = config["paths"]["gli_h5_file"]

        for domain, h5_path in [("men", men_h5), ("gli", gli_h5)]:
            for split_name, h5_split in [("test", "test"), ("probe", "lora_train")]:
                logger.info(f"\nExtracting {domain.upper()} {split_name}...")
                try:
                    features_dict, targets_dict, subject_ids = extract_features_for_domain(
                        encoder=encoder,
                        h5_path=h5_path,
                        split=h5_split,
                        device=device,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        feature_level=feature_level,
                        use_amp=use_amp,
                        tap_extractor=tap_extractor,
                    )
                    save_domain_features(
                        features_dict,
                        targets_dict,
                        subject_ids,
                        features_dir,
                        domain,
                        split_name,
                    )
                    result_paths[f"{domain}_{split_name}"] = (
                        features_dir / f"features_{domain}_{split_name}.pt"
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract {domain}/{split_name}: {e}")
    else:
        # Single-domain extraction (lora_ablation compat)
        h5_path = config["paths"].get("h5_file")
        if h5_path:
            logger.info(f"Using H5 backend: {h5_path}")

        # Load splits from H5
        splits = load_splits_h5(h5_path)

        # Determine probe split
        if "sdp_train" in splits and len(splits["sdp_train"]) > 0:
            sdp_split_key = "sdp_train"
        elif "probe_train" in splits:
            sdp_split_key = "probe_train"
        else:
            sdp_split_key = "lora_train"

        logger.info(f"\nExtracting probe features from '{sdp_split_key}'")
        probe_features, probe_targets, probe_ids = extract_features_for_domain(
            encoder=encoder,
            h5_path=h5_path,
            split=sdp_split_key,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            feature_level=feature_level,
            use_amp=use_amp,
            tap_extractor=tap_extractor,
        )
        save_features(probe_features, probe_targets, probe_ids, condition_dir, "probe")

        logger.info("\nExtracting test features")
        test_features, test_targets, test_ids = extract_features_for_domain(
            encoder=encoder,
            h5_path=h5_path,
            split="test",
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            feature_level=feature_level,
            use_amp=use_amp,
            tap_extractor=tap_extractor,
        )
        save_features(test_features, test_targets, test_ids, condition_dir, "test")

        result_paths = {
            "probe_features": condition_dir / "features_probe.pt",
            "test_features": condition_dir / "features_test.pt",
        }

    return result_paths


def main(
    config_path: str,
    condition: str,
    device: str = "cuda",
) -> None:
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    extract_features(condition, config, device)


# =============================================================================
# Backward-compat alias for SDP module
# =============================================================================


@torch.no_grad()
def extract_features_for_split(
    encoder: torch.nn.Module,
    subject_ids: list[str] | None = None,
    data_root: str | None = None,
    device: str = "cuda",
    batch_size: int = 2,
    num_workers: int = 4,
    feature_level: str = "encoder10",
    h5_path: str | None = None,
    h5_split: str | None = None,
    use_amp: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Extract features for a split (backward-compat wrapper).

    Delegates to extract_features_for_domain when h5_path is provided.

    Args:
        encoder: SwinUNETR encoder.
        subject_ids: Subject IDs (unused when h5_path is set).
        data_root: NIfTI data root (legacy, unused).
        device: Device.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        feature_level: Feature level.
        h5_path: Path to H5 file.
        h5_split: Split name in H5.
        use_amp: Use bf16 autocast.

    Returns:
        Tuple of (features_dict, targets_dict, subject_ids).
    """
    if h5_path is None:
        raise ValueError("h5_path is required for feature extraction")
    split = h5_split or "test"
    return extract_features_for_domain(
        encoder=encoder,
        h5_path=h5_path,
        split=split,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_level=feature_level,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.config, args.condition, args.device)
