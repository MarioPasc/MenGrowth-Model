# experiments/segment_based_approach/segment.py
"""Segmentation-based volume extraction for the A0 baseline.

Loads one or more segmentation models, runs sliding-window segmentation on each
MenGrowth H5 scan, thresholds the TC/WT/ET channels, and computes volumes (mm^3).
Also extracts manual volumes and per-region Dice scores for comparison.

Predicted segmentations (probabilities + reconstructed labels) are saved back
into the H5 file under ``predicted_segs/{model_name}/`` for future comparison.

Supports **multi-model** configs: each enabled model is run (or loaded from H5
cache) independently, and all results are stored in ``ScanVolumes.model_results``.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from monai.transforms.intensity.array import NormalizeIntensity
from scipy.ndimage import center_of_mass

from growth.inference.sliding_window import sliding_window_segment
from growth.models.encoder.swin_loader import load_full_swinunetr
from growth.models.growth.base import PatientTrajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PerModelResult:
    """Segmentation results from one model on one scan."""

    model_name: str
    wt_vol_mm3: float
    tc_vol_mm3: float
    et_vol_mm3: float
    wt_dice: float
    tc_dice: float
    et_dice: float
    is_empty: bool


@dataclass
class ScanVolumes:
    """Volume extraction and segmentation comparison results for a single scan.

    Manual volumes are always present.  Per-model predicted volumes live in
    ``model_results``: a dict mapping ``model_name -> PerModelResult``.

    Backward-compatible properties (``predicted_wt_vol_mm3``, ``wt_dice``,
    ``is_empty_predicted``, etc.) delegate to the **first** model in
    ``model_results`` so that existing code keeps working.
    """

    scan_id: str
    patient_id: str
    timepoint_idx: int
    # Manual volumes (mm^3)
    manual_wt_vol_mm3: float
    manual_tc_vol_mm3: float
    manual_et_vol_mm3: float
    is_empty_manual: bool
    # Centroid of WT mask (normalized [0,1] per axis), or None if empty
    centroid_xyz: tuple[float, float, float] | None = None
    # Per-model predictions
    model_results: dict[str, PerModelResult] = field(default_factory=dict)

    # -- Backward-compat helpers (delegate to first model) -----------------

    @property
    def _first_model(self) -> PerModelResult | None:
        if self.model_results:
            return next(iter(self.model_results.values()))
        return None

    @property
    def predicted_wt_vol_mm3(self) -> float:
        m = self._first_model
        return m.wt_vol_mm3 if m else 0.0

    @property
    def predicted_tc_vol_mm3(self) -> float:
        m = self._first_model
        return m.tc_vol_mm3 if m else 0.0

    @property
    def predicted_et_vol_mm3(self) -> float:
        m = self._first_model
        return m.et_vol_mm3 if m else 0.0

    @property
    def wt_dice(self) -> float:
        m = self._first_model
        return m.wt_dice if m else 0.0

    @property
    def tc_dice(self) -> float:
        m = self._first_model
        return m.tc_dice if m else 0.0

    @property
    def et_dice(self) -> float:
        m = self._first_model
        return m.et_dice if m else 0.0

    @property
    def is_empty_predicted(self) -> bool:
        m = self._first_model
        return m.is_empty if m else True


# ---------------------------------------------------------------------------
# Segmentation model config
# ---------------------------------------------------------------------------


@dataclass
class SegModelConfig:
    """Configuration for one segmentation model."""

    model_name: str
    model_type: str  # "BrainSegFounder" (extensible)
    checkpoint: str
    save_to_h5: bool = True
    enabled: bool = True
    lora_alpha: int | None = None  # LoRA scaling factor (for PEFT checkpoints)
    lora_rank: int | None = None  # LoRA rank (for PEFT checkpoints)


def parse_seg_config(cfg: dict) -> tuple[list[SegModelConfig], bool]:
    """Parse segmentation config, supporting old and new formats.

    New format: ``segmentation.models_to_use`` list.
    Old format: ``paths.checkpoint`` + ``segmentation.model_name``.

    Args:
        cfg: Full experiment config dict.

    Returns:
        (enabled_models, use_manual_segmentation).
    """
    seg_cfg = cfg.get("segmentation", {})

    # New format: models_to_use list
    models_list = seg_cfg.get("models_to_use")
    if models_list is not None:
        use_manual = seg_cfg.get("use_manual_segmentation", True)
        models = []
        for m in models_list:
            smc = SegModelConfig(
                model_name=m["model_name"],
                model_type=m.get("type", "BrainSegFounder"),
                checkpoint=m["checkpoints"],
                save_to_h5=m.get("save_to_h5", True),
                enabled=m.get("enabled", True),
                lora_alpha=m.get("lora_alpha"),
                lora_rank=m.get("lora_rank"),
            )
            if smc.enabled:
                models.append(smc)
        return models, use_manual

    # Old format fallback: paths.checkpoint + segmentation.model_name
    ckpt = cfg.get("paths", {}).get("checkpoint", "")
    model_name = seg_cfg.get("model_name", "brainsegfounder")
    save_to_h5 = seg_cfg.get("save_to_h5", True)
    if ckpt:
        models = [
            SegModelConfig(
                model_name=model_name,
                model_type="BrainSegFounder",
                checkpoint=ckpt,
                save_to_h5=save_to_h5,
                enabled=True,
            )
        ]
    else:
        models = []
    return models, True  # old format always used manual


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Prefix mapping for training checkpoints (model.encoder.* / model.decoder.*)
_TRAINING_CKPT_PREFIX_MAP = {
    # Standard training checkpoint prefixes
    "model.encoder.swinViT.": "swinViT.",
    "model.encoder.encoder10.": "encoder10.",
    "model.decoder.encoder1.": "encoder1.",
    "model.decoder.encoder2.": "encoder2.",
    "model.decoder.encoder3.": "encoder3.",
    "model.decoder.encoder4.": "encoder4.",
    "model.decoder.decoder5.": "decoder5.",
    "model.decoder.decoder4.": "decoder4.",
    "model.decoder.decoder3.": "decoder3.",
    "model.decoder.decoder2.": "decoder2.",
    "model.decoder.decoder1.": "decoder1.",
    "model.decoder.out.": "out.",
    # PEFT (LoRA) checkpoint prefixes — keys wrapped by peft library
    "lora_encoder.model.base_model.model.swinViT.": "swinViT.",
    "lora_encoder.model.base_model.model.encoder10.": "encoder10.",
    "lora_encoder.model.base_model.model.encoder1.": "encoder1.",
    "lora_encoder.model.base_model.model.encoder2.": "encoder2.",
    "lora_encoder.model.base_model.model.encoder3.": "encoder3.",
    "lora_encoder.model.base_model.model.encoder4.": "encoder4.",
    "lora_encoder.model.base_model.model.decoder5.": "decoder5.",
    "lora_encoder.model.base_model.model.decoder4.": "decoder4.",
    "lora_encoder.model.base_model.model.decoder3.": "decoder3.",
    "lora_encoder.model.base_model.model.decoder2.": "decoder2.",
    "lora_encoder.model.base_model.model.decoder1.": "decoder1.",
    "lora_encoder.model.base_model.model.out.": "out.",
}


def _strip_training_checkpoint_prefix(state_dict: dict) -> dict:
    """Strip ``model.encoder.*`` / ``model.decoder.*`` prefixes from a
    training checkpoint to produce bare SwinUNETR keys.

    Keys not matching any known prefix are silently skipped (e.g.
    ``semantic_heads.*``, optimizer state).

    Args:
        state_dict: Raw state dict from a training checkpoint.

    Returns:
        New dict with stripped keys.
    """
    stripped: dict = {}
    for key, val in state_dict.items():
        matched = False
        for prefix, replacement in _TRAINING_CKPT_PREFIX_MAP.items():
            if key.startswith(prefix):
                new_key = replacement + key[len(prefix) :]
                stripped[new_key] = val
                matched = True
                break
        if not matched:
            logger.debug(f"Skipping unrecognized key: {key}")
    return stripped


def _merge_lora_weights(
    state_dict: dict,
    lora_alpha: int = 16,
    lora_rank: int = 8,
) -> dict:
    """Merge LoRA adapter weights into base weights in-place.

    For each base weight ``W`` with corresponding ``lora_A`` and ``lora_B``,
    computes ``W_merged = W + (alpha / r) * B @ A`` and removes the LoRA keys.

    Args:
        state_dict: State dict with base weights and LoRA adapter weights.
            LoRA keys follow the pattern ``*.lora_A.default.weight`` and
            ``*.lora_B.default.weight``.
        lora_alpha: LoRA scaling factor.
        lora_rank: LoRA rank (used to compute scaling = alpha / rank).

    Returns:
        State dict with merged weights (LoRA keys removed).
    """
    scaling = lora_alpha / lora_rank

    # Collect LoRA A/B pairs keyed by the base parameter path
    lora_a_keys: dict[str, str] = {}
    lora_b_keys: dict[str, str] = {}
    for key in list(state_dict.keys()):
        if "lora_A" in key:
            # e.g. "swinViT.layers3.0.blocks.0.attn.qkv.lora_A.default.weight"
            base = key.split(".lora_A")[0]
            lora_a_keys[base] = key
        elif "lora_B" in key:
            base = key.split(".lora_B")[0]
            lora_b_keys[base] = key

    n_merged = 0
    for base in lora_a_keys:
        if base not in lora_b_keys:
            continue

        a_key = lora_a_keys[base]
        b_key = lora_b_keys[base]
        base_key = base + ".weight"

        A = state_dict[a_key]  # [r, in_features]
        B = state_dict[b_key]  # [out_features, r]

        if base_key in state_dict:
            W = state_dict[base_key]
            state_dict[base_key] = W + scaling * (B @ A)
            n_merged += 1
        else:
            logger.warning(f"LoRA base weight not found: {base_key}")

        # Remove LoRA keys
        del state_dict[a_key]
        del state_dict[b_key]

    # Remove any remaining LoRA metadata keys
    lora_meta_keys = [k for k in state_dict if "lora_" in k]
    for k in lora_meta_keys:
        del state_dict[k]

    logger.info(f"Merged {n_merged} LoRA adapters (alpha={lora_alpha}, r={lora_rank})")
    return state_dict


def load_segmentation_model(
    model_config: SegModelConfig,
    device: torch.device,
) -> "torch.nn.Module":
    """Load a segmentation model from checkpoint.

    Auto-detects checkpoint format:

    1. **Standard BSF** (``finetuned_model_fold_0.pt``): has ``state_dict``
       key with bare SwinUNETR keys -> delegates to ``load_full_swinunetr()``.
    2. **Training checkpoint** (``best_model.pt``): bare dict with
       ``model.encoder.*`` / ``model.decoder.*`` prefixes -> strips prefixes,
       creates SwinUNETR, loads ``strict=False``.

    Args:
        model_config: Model configuration.
        device: Target device.

    Returns:
        Loaded SwinUNETR model in eval mode.
    """
    from growth.models.encoder.swin_loader import create_swinunetr

    path = model_config.checkpoint
    logger.info(f"Loading model '{model_config.model_name}' from {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Standard BSF format
        logger.info("  Detected standard BSF checkpoint format")
        model = load_full_swinunetr(
            path,
            freeze_encoder=True,
            freeze_decoder=True,
            device=device,
        )
    else:
        # Training checkpoint — strip prefixes
        logger.info("  Detected training checkpoint format, stripping prefixes")
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unexpected checkpoint type {type(ckpt)} for {path}")
        stripped = _strip_training_checkpoint_prefix(ckpt)
        logger.info(f"  Stripped {len(ckpt)} -> {len(stripped)} keys")

        # Detect and merge LoRA weights if present
        has_lora = any("lora_A" in k or "lora_B" in k for k in stripped)
        if has_lora:
            alpha = model_config.lora_alpha or 16
            rank = model_config.lora_rank or 8
            logger.info(
                f"  Detected LoRA weights, merging with alpha={alpha}, r={rank}"
            )
            stripped = _merge_lora_weights(stripped, lora_alpha=alpha, lora_rank=rank)

        model = create_swinunetr()
        missing, unexpected = model.load_state_dict(stripped, strict=False)

        # Classify missing keys
        decoder_prefixes = ("decoder", "out")
        actual_missing = [k for k in missing if not k.startswith(decoder_prefixes)]
        if actual_missing:
            logger.warning(
                f"  Missing encoder keys ({len(actual_missing)}): {actual_missing[:5]}..."
            )
        if unexpected:
            logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False
        model = model.to(device)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks.

    Args:
        pred_mask: Predicted binary mask.
        gt_mask: Ground-truth binary mask.

    Returns:
        Dice score in [0, 1]. Returns 1.0 if both masks are empty.
    """
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    intersection = (pred_mask & gt_mask).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def _checkpoint_hash(path: str) -> str:
    """Short hash of checkpoint file for cache validation."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(4096))
    return h.hexdigest()[:8]


def _reconstruct_labels(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Reconstruct integer labels from 3-channel sigmoid probabilities.

    BrainSegFounder outputs overlapping regions:
        Ch0 = TC (Tumor Core = NCR + ET)
        Ch1 = WT (Whole Tumor = NCR + ED + ET)
        Ch2 = ET (Enhancing Tumor)

    Reconstructed labels: 0=BG, 1=NCR, 2=ED, 3=ET.

    Args:
        probs: Sigmoid probabilities, shape ``[3, D, H, W]``.
        threshold: Binarization threshold.

    Returns:
        Integer label volume, shape ``[D, H, W]``, dtype int8.
    """
    tc = probs[0] > threshold
    wt = probs[1] > threshold
    et = probs[2] > threshold

    labels = np.zeros(probs.shape[1:], dtype=np.int8)
    # ED = inside WT but outside TC
    labels[wt & ~tc] = 2
    # NCR = inside TC but outside ET
    labels[tc & ~et] = 1
    # ET = enhancing
    labels[et] = 3
    return labels


def list_prediction_models(h5_path: str) -> list[str]:
    """List all segmentation model names stored in the H5 file.

    Args:
        h5_path: Path to MenGrowth.h5.

    Returns:
        List of model name strings (e.g. ``["brainsegfounder"]``).
    """
    with h5py.File(h5_path, "r") as f:
        if "predicted_segs" not in f:
            return []
        return list(f["predicted_segs"].keys())


# ---------------------------------------------------------------------------
# Segmentation report
# ---------------------------------------------------------------------------


def generate_segmentation_report(volumes: list[ScanVolumes]) -> dict:
    """Generate a comprehensive segmentation comparison report.

    When multiple models are present in ``model_results``, produces a
    ``per_model`` section with per-region stats for each model.

    Args:
        volumes: List of ScanVolumes from extraction.

    Returns:
        Dict with per-model, per-region, per-scan, and aggregate metrics.
    """
    non_empty = [v for v in volumes if not v.is_empty_manual]

    # Discover all model names
    all_model_names: list[str] = []
    for v in volumes:
        for mn in v.model_results:
            if mn not in all_model_names:
                all_model_names.append(mn)

    report: dict = {
        "n_total_scans": len(volumes),
        "n_non_empty_manual": len(non_empty),
        "n_empty_manual": sum(1 for v in volumes if v.is_empty_manual),
    }

    per_model: dict[str, dict] = {}

    for model_name in all_model_names:
        n_empty_pred = sum(
            1
            for v in volumes
            if model_name in v.model_results and v.model_results[model_name].is_empty
        )

        # Per-region stats
        region_stats: dict[str, dict] = {}
        for region in ["wt", "tc", "et"]:
            dices = []
            manual_vols = []
            pred_vols = []
            for v in non_empty:
                mr = v.model_results.get(model_name)
                if mr is None:
                    continue
                dices.append(getattr(mr, f"{region}_dice"))
                manual_vols.append(getattr(v, f"manual_{region}_vol_mm3"))
                pred_vols.append(getattr(mr, f"{region}_vol_mm3"))

            if not dices:
                region_stats[region] = {}
                continue

            dice_arr = np.array(dices)
            manual_arr = np.array(manual_vols)
            pred_arr = np.array(pred_vols)

            # Volume correlation
            if len(manual_arr) > 2 and np.std(manual_arr) > 0 and np.std(pred_arr) > 0:
                from scipy.stats import pearsonr

                vol_r, vol_p = pearsonr(manual_arr, pred_arr)
            else:
                vol_r, vol_p = 0.0, 1.0

            # Volume R^2
            ss_res = np.sum((manual_arr - pred_arr) ** 2)
            ss_tot = np.sum((manual_arr - np.mean(manual_arr)) ** 2)
            vol_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0

            region_stats[region] = {
                "dice_mean": float(np.mean(dice_arr)),
                "dice_std": float(np.std(dice_arr)),
                "dice_median": float(np.median(dice_arr)),
                "dice_q25": float(np.percentile(dice_arr, 25)),
                "dice_q75": float(np.percentile(dice_arr, 75)),
                "dice_min": float(np.min(dice_arr)),
                "dice_max": float(np.max(dice_arr)),
                "volume_pearson_r": float(vol_r),
                "volume_pearson_p": float(vol_p),
                "volume_r2": vol_r2,
                "volume_mae_mm3": float(np.mean(np.abs(manual_arr - pred_arr))),
                "volume_rmse_mm3": float(np.sqrt(np.mean((manual_arr - pred_arr) ** 2))),
                "volume_mean_bias_mm3": float(np.mean(pred_arr - manual_arr)),
            }

        # Per-patient WT Dice
        patient_dices: dict[str, list[float]] = {}
        for v in non_empty:
            mr = v.model_results.get(model_name)
            if mr is not None:
                patient_dices.setdefault(v.patient_id, []).append(mr.wt_dice)

        per_model[model_name] = {
            "n_empty_predicted": n_empty_pred,
            "per_region": region_stats,
            "per_patient_wt_dice": {
                pid: {"mean": float(np.mean(d)), "n_scans": len(d)}
                for pid, d in sorted(patient_dices.items())
            },
        }

    report["per_model"] = per_model

    # Backward compat: top-level per_region from first model
    if all_model_names:
        first = all_model_names[0]
        report["per_region"] = per_model[first]["per_region"]
        report["per_patient_wt_dice"] = per_model[first]["per_patient_wt_dice"]
        report["n_empty_predicted"] = per_model[first]["n_empty_predicted"]

    # Per-scan details (serialize model_results)
    per_scan = []
    for v in volumes:
        scan_dict: dict = {
            "scan_id": v.scan_id,
            "patient_id": v.patient_id,
            "timepoint_idx": v.timepoint_idx,
            "manual_wt_vol_mm3": v.manual_wt_vol_mm3,
            "manual_tc_vol_mm3": v.manual_tc_vol_mm3,
            "manual_et_vol_mm3": v.manual_et_vol_mm3,
            "is_empty_manual": v.is_empty_manual,
            "model_results": {mn: asdict(mr) for mn, mr in v.model_results.items()},
        }
        per_scan.append(scan_dict)
    report["per_scan"] = per_scan

    return report


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class SegmentationVolumeExtractor:
    """Extract volumes from MenGrowth H5 using one or more segmentation models.

    Computes TC/WT/ET Dice and volumes for both manual and model-predicted
    segmentations on each scan.  Supports multi-model configs with H5
    caching of predictions (skip inference if already stored).

    Args:
        cfg: OmegaConf config dict (from config.yaml).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.h5_path: str = cfg["paths"]["mengrowth_h5"]
        self.output_dir = Path(cfg["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sw_roi = tuple(cfg["segmentation"]["sw_roi_size"])
        self.sw_overlap: float = cfg["segmentation"]["sw_overlap"]
        self.sw_mode: str = cfg["segmentation"]["sw_mode"]
        self.wt_threshold: float = cfg["segmentation"]["wt_threshold"]
        self.exclude_patients: list[str] = cfg["patients"].get("exclude", [])

        # Parse multi-model config
        self.seg_models, self.use_manual = parse_seg_config(cfg)

        self._cache_path = self.output_dir / "volume_cache.json"

    def extract_all(self, force_recompute: bool = False) -> list[ScanVolumes]:
        """Extract volumes for all scans in the H5 file.

        For each enabled segmentation model:

        1. Check if ``predicted_segs/{model_name}/labels`` exists in H5.
        2. If yes (and not ``force_recompute``): read labels, compute volumes
           + Dice from stored data (CPU only, no GPU needed).
        3. If no: load model, run sliding-window inference, save to H5,
           compute volumes + Dice, then unload model.

        Args:
            force_recompute: If True, ignore all caches and re-run inference.

        Returns:
            List of ScanVolumes for every scan.
        """
        # Try loading from JSON cache first
        if not force_recompute and self._cache_path.exists():
            try:
                cached = self._load_cache()
                if cached is not None:
                    logger.info(f"Loaded cached volumes from {self._cache_path}")
                    return cached
            except (ValueError, KeyError) as e:
                logger.warning(f"Cache load failed ({e}), recomputing")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)

        any_save = any(mc.save_to_h5 for mc in self.seg_models)
        h5_mode = "a" if any_save else "r"

        with h5py.File(self.h5_path, h5_mode) as f:
            n_scans = f.attrs["n_scans"]
            spacing = tuple(f.attrs.get("spacing", [1.0, 1.0, 1.0]))
            roi_size = tuple(f.attrs.get("roi_size", [192, 192, 192]))
            voxel_vol = float(np.prod(spacing))

            scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
            patient_ids = [s.decode() if isinstance(s, bytes) else s for s in f["patient_ids"][:]]
            timepoint_idx = f["timepoint_idx"][:].astype(int)

            # --- First pass: manual volumes ---
            logger.info(f"Computing manual volumes for {n_scans} scans...")
            results: list[ScanVolumes] = []
            gt_masks_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

            for i in range(n_scans):
                seg = f["segs"][i]  # [1, D, H, W] int8
                seg_np = seg[0]  # [D, H, W]
                gt_tc = ((seg_np == 1) | (seg_np == 3)).astype(np.uint8)
                gt_wt = (seg_np > 0).astype(np.uint8)
                gt_et = (seg_np == 3).astype(np.uint8)

                manual_wt_vol = float(gt_wt.sum() * voxel_vol)
                manual_tc_vol = float(gt_tc.sum() * voxel_vol)
                manual_et_vol = float(gt_et.sum() * voxel_vol)

                gt_masks_cache.append((gt_tc, gt_wt, gt_et))

                # Compute centroid from WT mask (normalized to [0, 1])
                centroid: tuple[float, float, float] | None = None
                if gt_wt.sum() > 0:
                    com = center_of_mass(gt_wt)
                    shape = gt_wt.shape
                    centroid = (
                        float(com[0]) / shape[0],
                        float(com[1]) / shape[1],
                        float(com[2]) / shape[2],
                    )

                results.append(
                    ScanVolumes(
                        scan_id=scan_ids[i],
                        patient_id=patient_ids[i],
                        timepoint_idx=int(timepoint_idx[i]),
                        manual_wt_vol_mm3=manual_wt_vol,
                        manual_tc_vol_mm3=manual_tc_vol,
                        manual_et_vol_mm3=manual_et_vol,
                        is_empty_manual=(manual_wt_vol == 0.0),
                        centroid_xyz=centroid,
                        model_results={},
                    )
                )

            # --- Per-model pass ---
            for mc in self.seg_models:
                h5_label_key = f"predicted_segs/{mc.model_name}/labels"
                h5_exists = h5_label_key in f

                if h5_exists and not force_recompute:
                    # Read from H5 — no GPU needed
                    logger.info(
                        f"Loading '{mc.model_name}' predictions from H5 (skipping inference)"
                    )
                    self._populate_from_h5_labels(f, mc, results, gt_masks_cache, voxel_vol)
                else:
                    # Run inference
                    logger.info(
                        f"Running inference for '{mc.model_name}' "
                        f"({n_scans} scans, device={device})"
                    )
                    model = load_segmentation_model(mc, device)

                    # Prepare H5 group if saving
                    pred_group = None
                    if mc.save_to_h5:
                        pred_group = self._prepare_h5_prediction_group(f, mc, n_scans, roi_size)

                    for i in range(n_scans):
                        image = f["images"][i]  # [4, D, H, W] float32
                        image_t = torch.from_numpy(image.astype(np.float32))
                        image_t = normalizer(image_t)
                        image_t = image_t.unsqueeze(0).to(device)

                        with torch.no_grad():
                            logits = sliding_window_segment(
                                model=model,
                                images=image_t,
                                roi_size=self.sw_roi,
                                overlap=self.sw_overlap,
                                mode=self.sw_mode,
                            )
                            probs = torch.sigmoid(logits[0]).cpu().numpy()

                        threshold = self.wt_threshold
                        pred_tc = (probs[0] > threshold).astype(np.uint8)
                        pred_wt = (probs[1] > threshold).astype(np.uint8)
                        pred_et = (probs[2] > threshold).astype(np.uint8)

                        # Save to H5
                        if pred_group is not None:
                            pred_group["probabilities"][i] = probs.astype(np.float16)
                            recon_labels = _reconstruct_labels(probs, threshold)
                            pred_group["labels"][i] = recon_labels[np.newaxis]

                        # Compute volumes + Dice
                        gt_tc, gt_wt, gt_et = gt_masks_cache[i]
                        pmr = PerModelResult(
                            model_name=mc.model_name,
                            wt_vol_mm3=float(pred_wt.sum() * voxel_vol),
                            tc_vol_mm3=float(pred_tc.sum() * voxel_vol),
                            et_vol_mm3=float(pred_et.sum() * voxel_vol),
                            wt_dice=_compute_dice(pred_wt, gt_wt),
                            tc_dice=_compute_dice(pred_tc, gt_tc),
                            et_dice=_compute_dice(pred_et, gt_et),
                            is_empty=(float(pred_wt.sum() * voxel_vol) == 0.0),
                        )
                        results[i].model_results[mc.model_name] = pmr

                        logger.info(
                            f"  [{i + 1}/{n_scans}] {scan_ids[i]} ({mc.model_name}): "
                            f"WT dice={pmr.wt_dice:.3f} TC dice={pmr.tc_dice:.3f} "
                            f"ET dice={pmr.et_dice:.3f}"
                        )

                    if pred_group is not None:
                        logger.info(f"Saved '{mc.model_name}' predictions to H5 (probs + labels)")

                    # Unload model to free GPU memory
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        self._save_cache(results)
        return results

    def _populate_from_h5_labels(
        self,
        f: h5py.File,
        mc: SegModelConfig,
        results: list[ScanVolumes],
        gt_masks_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        voxel_vol: float,
    ) -> None:
        """Populate model results by reading stored labels from H5.

        Args:
            f: Open h5py File.
            mc: Model configuration.
            results: ScanVolumes list to populate.
            gt_masks_cache: Cached (gt_tc, gt_wt, gt_et) per scan.
            voxel_vol: Voxel volume in mm^3.
        """
        label_ds = f[f"predicted_segs/{mc.model_name}/labels"]
        n_scans = len(results)

        for i in range(n_scans):
            labels = label_ds[i, 0]  # [D, H, W] int8
            pred_tc = ((labels == 1) | (labels == 3)).astype(np.uint8)
            pred_wt = (labels > 0).astype(np.uint8)
            pred_et = (labels == 3).astype(np.uint8)

            gt_tc, gt_wt, gt_et = gt_masks_cache[i]
            pmr = PerModelResult(
                model_name=mc.model_name,
                wt_vol_mm3=float(pred_wt.sum() * voxel_vol),
                tc_vol_mm3=float(pred_tc.sum() * voxel_vol),
                et_vol_mm3=float(pred_et.sum() * voxel_vol),
                wt_dice=_compute_dice(pred_wt, gt_wt),
                tc_dice=_compute_dice(pred_tc, gt_tc),
                et_dice=_compute_dice(pred_et, gt_et),
                is_empty=(float(pred_wt.sum() * voxel_vol) == 0.0),
            )
            results[i].model_results[mc.model_name] = pmr

        logger.info(f"  Loaded {n_scans} scans for '{mc.model_name}' from H5 labels")

    def _prepare_h5_prediction_group(
        self,
        f: h5py.File,
        mc: SegModelConfig,
        n_scans: int,
        roi_size: tuple,
    ) -> h5py.Group:
        """Create or replace the prediction group in H5.

        Schema::

            predicted_segs/{model_name}/
                attrs: model_name, checkpoint_path, checkpoint_hash,
                       timestamp, threshold, channel_order
                probabilities  [N, 3, D, H, W] float16  (raw sigmoid)
                labels         [N, 1, D, H, W] int8      (reconstructed)

        Args:
            f: Open h5py File in append mode.
            mc: Model configuration.
            n_scans: Number of scans.
            roi_size: Spatial dimensions (D, H, W).

        Returns:
            The h5py Group for this model's predictions.
        """
        group_path = f"predicted_segs/{mc.model_name}"

        # Remove existing group to overwrite cleanly
        if group_path in f:
            logger.info(f"Overwriting existing predictions at {group_path}")
            del f[group_path]

        pred_group = f.create_group(group_path)

        # Metadata
        ckpt_hash = _checkpoint_hash(mc.checkpoint)
        pred_group.attrs["model_name"] = mc.model_name
        pred_group.attrs["checkpoint_path"] = mc.checkpoint
        pred_group.attrs["checkpoint_hash"] = ckpt_hash
        pred_group.attrs["timestamp"] = datetime.now(UTC).isoformat()
        pred_group.attrs["threshold"] = self.wt_threshold
        pred_group.attrs["channel_order"] = ["TC", "WT", "ET"]
        pred_group.attrs["sw_roi_size"] = list(self.sw_roi)
        pred_group.attrs["sw_overlap"] = self.sw_overlap

        D, H, W = roi_size

        pred_group.create_dataset(
            "probabilities",
            shape=(n_scans, 3, D, H, W),
            dtype=np.float16,
            chunks=(1, 3, D, H, W),
            compression="gzip",
            compression_opts=4,
        )

        pred_group.create_dataset(
            "labels",
            shape=(n_scans, 1, D, H, W),
            dtype=np.int8,
            chunks=(1, 1, D, H, W),
            compression="gzip",
            compression_opts=4,
        )

        logger.info(
            f"Created H5 prediction group: {group_path} "
            f"(probs: float16, labels: int8, gzip-4, ckpt: {ckpt_hash})"
        )
        return pred_group

    def build_trajectories(
        self,
        volumes: list[ScanVolumes],
        source: str = "manual",
    ) -> list[PatientTrajectory]:
        """Build PatientTrajectory objects from extracted WT volumes.

        The time variable is controlled by ``config["time"]["variable"]``:

        - ``"ordinal"`` (default): timepoint index (0, 1, 2, ...).
        - ``"temporal"``: months from first scan, read from the H5
          ``time_delta_months`` dataset.  Falls back to ordinal with a
          warning if the dataset is absent.

        Args:
            volumes: List of ScanVolumes from extract_all().
            source: ``"manual"`` or any model name present in
                ``ScanVolumes.model_results``.

        Returns:
            List of PatientTrajectory (sorted by patient_id, then time).
        """
        time_variable = self.cfg.get("time", {}).get("variable", "ordinal")

        # Load real-time metadata from H5 when requested
        scan_id_to_months: dict[str, float] = {}
        if time_variable == "temporal":
            scan_id_to_months = self._load_temporal_metadata()
            if not scan_id_to_months:
                logger.warning(
                    "time.variable='temporal' but no time_delta_months in H5. "
                    "Falling back to ordinal."
                )
                time_variable = "ordinal"

        # Group by patient
        patient_scans: dict[str, list[ScanVolumes]] = {}
        for sv in volumes:
            if sv.patient_id in self.exclude_patients:
                continue
            patient_scans.setdefault(sv.patient_id, []).append(sv)

        trajectories: list[PatientTrajectory] = []
        min_tp = self.cfg["patients"].get("min_timepoints", 2)

        for pid, scans in sorted(patient_scans.items()):
            scans_sorted = sorted(scans, key=lambda s: s.timepoint_idx)

            if len(scans_sorted) < min_tp:
                logger.debug(f"Skipping {pid}: only {len(scans_sorted)} timepoints")
                continue

            if time_variable == "temporal":
                times = np.array(
                    [scan_id_to_months[s.scan_id] for s in scans_sorted],
                    dtype=np.float64,
                )
            else:
                times = np.array(
                    [s.timepoint_idx for s in scans_sorted],
                    dtype=np.float64,
                )

            if source == "manual":
                obs = np.array([s.manual_wt_vol_mm3 for s in scans_sorted])
            else:
                # Look up model name in model_results
                if source not in scans_sorted[0].model_results:
                    available = ["manual"] + list(scans_sorted[0].model_results.keys())
                    raise ValueError(f"Unknown source '{source}'. Available: {available}")
                obs = np.array([s.model_results[source].wt_vol_mm3 for s in scans_sorted])

            # Apply log1p transform
            obs_log = np.log1p(obs)

            # Skip patients where all volumes are zero
            if np.all(obs == 0.0):
                logger.debug(f"Skipping {pid}: all volumes are zero ({source})")
                continue

            # Attach centroid from first timepoint as static covariate
            covariates: dict[str, float] | None = None
            first_centroid = scans_sorted[0].centroid_xyz
            if first_centroid is not None:
                covariates = {
                    "centroid_x": first_centroid[0],
                    "centroid_y": first_centroid[1],
                    "centroid_z": first_centroid[2],
                }

            trajectories.append(
                PatientTrajectory(
                    patient_id=pid,
                    times=times,
                    observations=obs_log,
                    covariates=covariates,
                )
            )

        logger.info(
            f"Built {len(trajectories)} trajectories from {source} volumes "
            f"(time={time_variable}, excluded: {self.exclude_patients})"
        )
        return trajectories

    def build_delta_trajectories(
        self,
        volumes: list[ScanVolumes],
        source: str = "manual",
    ) -> list[PatientTrajectory]:
        """Build delta-V trajectories: observations are log-volume changes.

        For each patient with n timepoints, produces n-1 delta observations:
            delta_j = log1p(V_{j+1}) - log1p(V_j)
        at midpoint times:
            t_mid_j = (t_j + t_{j+1}) / 2

        Requires min_timepoints >= 3 (to get >= 2 delta observations per patient).

        Args:
            volumes: List of ScanVolumes from extract_all().
            source: ``"manual"`` or model name.

        Returns:
            List of PatientTrajectory with delta-V observations.
        """
        time_variable = self.cfg.get("time", {}).get("variable", "ordinal")

        scan_id_to_months: dict[str, float] = {}
        if time_variable == "temporal":
            scan_id_to_months = self._load_temporal_metadata()
            if not scan_id_to_months:
                logger.warning(
                    "time.variable='temporal' but no time_delta_months in H5. "
                    "Falling back to ordinal."
                )
                time_variable = "ordinal"

        patient_scans: dict[str, list[ScanVolumes]] = {}
        for sv in volumes:
            if sv.patient_id in self.exclude_patients:
                continue
            patient_scans.setdefault(sv.patient_id, []).append(sv)

        trajectories: list[PatientTrajectory] = []
        # Delta-V needs at least 3 timepoints to get 2 deltas for GP
        min_tp = max(self.cfg["patients"].get("min_timepoints", 2), 3)

        for pid, scans in sorted(patient_scans.items()):
            scans_sorted = sorted(scans, key=lambda s: s.timepoint_idx)

            if len(scans_sorted) < min_tp:
                logger.debug(
                    f"Skipping {pid}: only {len(scans_sorted)} timepoints "
                    f"(need {min_tp} for delta-V)"
                )
                continue

            if time_variable == "temporal":
                raw_times = np.array(
                    [scan_id_to_months[s.scan_id] for s in scans_sorted],
                    dtype=np.float64,
                )
            else:
                raw_times = np.array(
                    [s.timepoint_idx for s in scans_sorted],
                    dtype=np.float64,
                )

            if source == "manual":
                raw_vols = np.array([s.manual_wt_vol_mm3 for s in scans_sorted])
            else:
                if source not in scans_sorted[0].model_results:
                    available = ["manual"] + list(scans_sorted[0].model_results.keys())
                    raise ValueError(f"Unknown source '{source}'. Available: {available}")
                raw_vols = np.array(
                    [s.model_results[source].wt_vol_mm3 for s in scans_sorted]
                )

            log_vols = np.log1p(raw_vols)

            if np.all(raw_vols == 0.0):
                continue

            # Compute deltas and midpoint times
            deltas = np.diff(log_vols)  # [n-1]
            mid_times = (raw_times[:-1] + raw_times[1:]) / 2.0  # [n-1]

            # Attach centroid from first timepoint
            covariates: dict[str, float] | None = None
            first_centroid = scans_sorted[0].centroid_xyz
            if first_centroid is not None:
                covariates = {
                    "centroid_x": first_centroid[0],
                    "centroid_y": first_centroid[1],
                    "centroid_z": first_centroid[2],
                }

            trajectories.append(
                PatientTrajectory(
                    patient_id=pid,
                    times=mid_times,
                    observations=deltas,
                    covariates=covariates,
                )
            )

        logger.info(
            f"Built {len(trajectories)} delta-V trajectories from {source} "
            f"(time={time_variable}, min_tp={min_tp})"
        )
        return trajectories

    def _load_temporal_metadata(self) -> dict[str, float]:
        """Load ``time_delta_months`` from the H5 file.

        Returns:
            Mapping from scan_id to months-from-first-scan, or empty dict
            if the dataset is not present in the H5 file.
        """
        try:
            with h5py.File(self.h5_path, "r") as f:
                if "time_delta_months" not in f:
                    return {}
                months = f["time_delta_months"][:]
                scan_ids = [
                    sid.decode() if isinstance(sid, bytes) else str(sid) for sid in f["scan_ids"][:]
                ]
                return dict(zip(scan_ids, months.tolist()))
        except Exception as e:
            logger.warning(f"Failed to load temporal metadata from H5: {e}")
            return {}

    def _save_cache(self, results: list[ScanVolumes]) -> None:
        """Save volume results to JSON cache (v3 format)."""
        # Collect model names
        model_names = []
        for r in results:
            for mn in r.model_results:
                if mn not in model_names:
                    model_names.append(mn)

        scans_data = []
        for sv in results:
            scan_dict: dict = {
                "scan_id": sv.scan_id,
                "patient_id": sv.patient_id,
                "timepoint_idx": sv.timepoint_idx,
                "manual_wt_vol_mm3": sv.manual_wt_vol_mm3,
                "manual_tc_vol_mm3": sv.manual_tc_vol_mm3,
                "manual_et_vol_mm3": sv.manual_et_vol_mm3,
                "is_empty_manual": sv.is_empty_manual,
                "centroid_xyz": list(sv.centroid_xyz) if sv.centroid_xyz else None,
                "model_results": {mn: asdict(mr) for mn, mr in sv.model_results.items()},
            }
            scans_data.append(scan_dict)

        cache = {
            "cache_version": 3,
            "h5_path": self.h5_path,
            "models": model_names,
            "n_scans": len(results),
            "scans": scans_data,
        }
        with open(self._cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        logger.info(f"Saved volume cache (v3) to {self._cache_path}")

    def _load_cache(self) -> list[ScanVolumes] | None:
        """Load volume results from JSON cache.

        Returns:
            List of ScanVolumes, or None if cache is stale/incompatible.
        """
        with open(self._cache_path) as f:
            cache = json.load(f)

        version = cache.get("cache_version", 1)

        if version < 3:
            logger.warning(f"Cache is v{version}, need v3. Will recompute.")
            return None

        # Validate that cached models match current config
        cached_models = set(cache.get("models", []))
        config_models = {mc.model_name for mc in self.seg_models}
        if cached_models != config_models:
            logger.info(
                f"Cache models {cached_models} != config models {config_models}. "
                f"Will recompute (H5 predictions may still be reused)."
            )
            return None

        results = []
        for s in cache["scans"]:
            model_results = {}
            for mn, mr_dict in s.get("model_results", {}).items():
                model_results[mn] = PerModelResult(
                    model_name=mr_dict["model_name"],
                    wt_vol_mm3=mr_dict["wt_vol_mm3"],
                    tc_vol_mm3=mr_dict["tc_vol_mm3"],
                    et_vol_mm3=mr_dict["et_vol_mm3"],
                    wt_dice=mr_dict["wt_dice"],
                    tc_dice=mr_dict["tc_dice"],
                    et_dice=mr_dict["et_dice"],
                    is_empty=mr_dict["is_empty"],
                )

            centroid_raw = s.get("centroid_xyz")
            centroid = tuple(centroid_raw) if centroid_raw else None

            results.append(
                ScanVolumes(
                    scan_id=s["scan_id"],
                    patient_id=s["patient_id"],
                    timepoint_idx=s["timepoint_idx"],
                    manual_wt_vol_mm3=s["manual_wt_vol_mm3"],
                    manual_tc_vol_mm3=s["manual_tc_vol_mm3"],
                    manual_et_vol_mm3=s["manual_et_vol_mm3"],
                    is_empty_manual=s["is_empty_manual"],
                    centroid_xyz=centroid,
                    model_results=model_results,
                )
            )

        return results
