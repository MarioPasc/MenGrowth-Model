# experiments/segment_based_approach/segment.py
"""Segmentation-based volume extraction for the A0 baseline.

Loads the frozen BrainSegFounder, runs sliding-window segmentation on each
MenGrowth H5 scan, thresholds the TC/WT/ET channels, and computes volumes (mm^3).
Also extracts manual volumes and per-region Dice scores for comparison.

Predicted segmentations (probabilities + reconstructed labels) are saved back
into the H5 file under ``predicted_segs/{model_name}/`` for future comparison.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime, timezone

import h5py
import numpy as np
import torch
from monai.transforms.intensity.array import NormalizeIntensity

from growth.inference.sliding_window import sliding_window_segment
from growth.models.encoder.swin_loader import load_full_swinunetr
from growth.models.growth.base import PatientTrajectory

logger = logging.getLogger(__name__)


@dataclass
class ScanVolumes:
    """Volume extraction and segmentation comparison results for a single scan."""

    scan_id: str
    patient_id: str
    timepoint_idx: int
    # Per-region volumes (mm^3)
    manual_wt_vol_mm3: float
    predicted_wt_vol_mm3: float
    manual_tc_vol_mm3: float
    predicted_tc_vol_mm3: float
    manual_et_vol_mm3: float
    predicted_et_vol_mm3: float
    # Per-region Dice
    wt_dice: float
    tc_dice: float
    et_dice: float
    # Flags
    is_empty_manual: bool
    is_empty_predicted: bool


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


def generate_segmentation_report(volumes: list[ScanVolumes]) -> dict:
    """Generate a comprehensive segmentation comparison report.

    Compares BSF-predicted segmentations against manual ground truth
    across TC/WT/ET regions with per-scan and aggregate statistics.

    Args:
        volumes: List of ScanVolumes from extraction.

    Returns:
        Dict with per-region, per-scan, and aggregate comparison metrics.
    """
    non_empty = [v for v in volumes if not v.is_empty_manual]

    report: dict = {
        "n_total_scans": len(volumes),
        "n_non_empty_manual": len(non_empty),
        "n_empty_manual": sum(1 for v in volumes if v.is_empty_manual),
        "n_empty_predicted": sum(1 for v in volumes if v.is_empty_predicted),
    }

    # Per-region aggregate statistics
    region_stats: dict[str, dict] = {}
    for region in ["wt", "tc", "et"]:
        dices = [getattr(v, f"{region}_dice") for v in non_empty]
        manual_vols = [getattr(v, f"manual_{region}_vol_mm3") for v in non_empty]
        pred_vols = [getattr(v, f"predicted_{region}_vol_mm3") for v in non_empty]

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

    report["per_region"] = region_stats

    # Per-patient aggregated Dice (mean across timepoints)
    patient_dices: dict[str, list[float]] = {}
    for v in non_empty:
        patient_dices.setdefault(v.patient_id, []).append(v.wt_dice)

    report["per_patient_wt_dice"] = {
        pid: {
            "mean": float(np.mean(dices)),
            "n_scans": len(dices),
        }
        for pid, dices in sorted(patient_dices.items())
    }

    # Per-scan details
    report["per_scan"] = [asdict(v) for v in volumes]

    return report


class SegmentationVolumeExtractor:
    """Extract volumes from MenGrowth H5 using frozen BSF segmentation.

    Computes TC/WT/ET Dice and volumes for both manual and BSF-predicted
    segmentations on each scan.

    Args:
        cfg: OmegaConf config dict (from config.yaml).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.h5_path: str = cfg["paths"]["mengrowth_h5"]
        self.ckpt_path: str = cfg["paths"]["checkpoint"]
        self.output_dir = Path(cfg["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sw_roi = tuple(cfg["segmentation"]["sw_roi_size"])
        self.sw_overlap: float = cfg["segmentation"]["sw_overlap"]
        self.sw_mode: str = cfg["segmentation"]["sw_mode"]
        self.wt_threshold: float = cfg["segmentation"]["wt_threshold"]
        self.exclude_patients: list[str] = cfg["patients"].get("exclude", [])

        seg_cfg = cfg.get("segmentation", {})
        self.model_name: str = seg_cfg.get("model_name", "brainsegfounder")
        self.save_to_h5: bool = seg_cfg.get("save_to_h5", True)

        self._cache_path = self.output_dir / "volume_cache.json"

    def extract_all(self, force_recompute: bool = False) -> list[ScanVolumes]:
        """Extract volumes for all scans in the H5 file.

        When running BSF inference (not from cache), predicted segmentations
        are also saved into the H5 file under
        ``predicted_segs/{model_name}/probabilities`` (float16) and
        ``predicted_segs/{model_name}/labels`` (int8) for future comparison
        with other segmentation models.

        Args:
            force_recompute: If True, ignore cache and recompute.

        Returns:
            List of ScanVolumes for every scan.
        """
        # Check cache
        if not force_recompute and self._cache_path.exists():
            logger.info(f"Loading cached volumes from {self._cache_path}")
            return self._load_cache()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading BSF model from {self.ckpt_path} on {device}")

        model = load_full_swinunetr(
            self.ckpt_path,
            freeze_encoder=True,
            freeze_decoder=True,
            device=device,
        )
        model.eval()

        normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)

        results: list[ScanVolumes] = []

        # Open H5 in append mode: read images/segs, write predicted_segs
        h5_mode = "a" if self.save_to_h5 else "r"
        with h5py.File(self.h5_path, h5_mode) as f:
            n_scans = f.attrs["n_scans"]
            spacing = tuple(f.attrs.get("spacing", [1.0, 1.0, 1.0]))
            roi_size = tuple(f.attrs.get("roi_size", [192, 192, 192]))
            voxel_vol = float(np.prod(spacing))

            scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
            patient_ids = [s.decode() if isinstance(s, bytes) else s for s in f["patient_ids"][:]]
            timepoint_idx = f["timepoint_idx"][:].astype(int)

            # --- Prepare H5 prediction datasets ---
            pred_group = None
            if self.save_to_h5:
                pred_group = self._prepare_h5_prediction_group(f, n_scans, roi_size)

            logger.info(
                f"Processing {n_scans} scans from {self.h5_path} "
                f"(save_to_h5={self.save_to_h5}, model={self.model_name})"
            )

            for i in range(n_scans):
                scan_id = scan_ids[i]
                pid = patient_ids[i]
                tp = int(timepoint_idx[i])

                # --- Manual masks from ground truth labels ---
                # Labels: 0=bg, 1=NCR, 2=ED, 3=ET
                seg = f["segs"][i]  # [1, D, H, W] int8
                seg_np = seg[0]  # [D, H, W]
                gt_tc = ((seg_np == 1) | (seg_np == 3)).astype(np.uint8)
                gt_wt = (seg_np > 0).astype(np.uint8)
                gt_et = (seg_np == 3).astype(np.uint8)

                manual_wt_vol = float(gt_wt.sum() * voxel_vol)
                manual_tc_vol = float(gt_tc.sum() * voxel_vol)
                manual_et_vol = float(gt_et.sum() * voxel_vol)
                is_empty_manual = manual_wt_vol == 0.0

                # --- Predicted masks from BSF 3-channel sigmoid ---
                image = f["images"][i]  # [4, D, H, W] float32
                image_t = torch.from_numpy(image.astype(np.float32))
                image_t = normalizer(image_t)  # z-score normalize
                image_t = image_t.unsqueeze(0).to(device)  # [1, 4, D, H, W]

                with torch.no_grad():
                    logits = sliding_window_segment(
                        model=model,
                        images=image_t,
                        roi_size=self.sw_roi,
                        overlap=self.sw_overlap,
                        mode=self.sw_mode,
                    )
                    # 3-channel sigmoid: Ch0=TC, Ch1=WT, Ch2=ET
                    probs = torch.sigmoid(logits[0]).cpu().numpy()  # [3, D, H, W]

                threshold = self.wt_threshold
                pred_tc = (probs[0] > threshold).astype(np.uint8)
                pred_wt = (probs[1] > threshold).astype(np.uint8)
                pred_et = (probs[2] > threshold).astype(np.uint8)

                predicted_wt_vol = float(pred_wt.sum() * voxel_vol)
                predicted_tc_vol = float(pred_tc.sum() * voxel_vol)
                predicted_et_vol = float(pred_et.sum() * voxel_vol)
                is_empty_predicted = predicted_wt_vol == 0.0

                # Dice per region
                wt_dice = _compute_dice(pred_wt, gt_wt)
                tc_dice = _compute_dice(pred_tc, gt_tc)
                et_dice = _compute_dice(pred_et, gt_et)

                # --- Save predictions to H5 (scan-by-scan, no accumulation) ---
                if pred_group is not None:
                    pred_group["probabilities"][i] = probs.astype(np.float16)
                    recon_labels = _reconstruct_labels(probs, threshold)
                    pred_group["labels"][i] = recon_labels[np.newaxis]  # [1,D,H,W]

                sv = ScanVolumes(
                    scan_id=scan_id,
                    patient_id=pid,
                    timepoint_idx=tp,
                    manual_wt_vol_mm3=manual_wt_vol,
                    predicted_wt_vol_mm3=predicted_wt_vol,
                    manual_tc_vol_mm3=manual_tc_vol,
                    predicted_tc_vol_mm3=predicted_tc_vol,
                    manual_et_vol_mm3=manual_et_vol,
                    predicted_et_vol_mm3=predicted_et_vol,
                    wt_dice=wt_dice,
                    tc_dice=tc_dice,
                    et_dice=et_dice,
                    is_empty_manual=is_empty_manual,
                    is_empty_predicted=is_empty_predicted,
                )
                results.append(sv)

                logger.info(
                    f"  [{i + 1}/{n_scans}] {scan_id}: "
                    f"WT dice={wt_dice:.3f} TC dice={tc_dice:.3f} ET dice={et_dice:.3f} "
                    f"manual_wt={manual_wt_vol:.0f}mm3 pred_wt={predicted_wt_vol:.0f}mm3"
                    + (" [EMPTY]" if is_empty_manual else "")
                )

            if pred_group is not None:
                logger.info(
                    f"Saved predicted segmentations to H5: "
                    f"predicted_segs/{self.model_name}/ "
                    f"(probabilities: {pred_group['probabilities'].shape}, "
                    f"labels: {pred_group['labels'].shape})"
                )

        self._save_cache(results)
        return results

    def _prepare_h5_prediction_group(
        self,
        f: h5py.File,
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
            n_scans: Number of scans.
            roi_size: Spatial dimensions (D, H, W).

        Returns:
            The h5py Group for this model's predictions.
        """
        group_path = f"predicted_segs/{self.model_name}"

        # Remove existing group to overwrite cleanly
        if group_path in f:
            logger.info(f"Overwriting existing predictions at {group_path}")
            del f[group_path]

        pred_group = f.create_group(group_path)

        # Metadata
        ckpt_hash = _checkpoint_hash(self.ckpt_path)
        pred_group.attrs["model_name"] = self.model_name
        pred_group.attrs["checkpoint_path"] = self.ckpt_path
        pred_group.attrs["checkpoint_hash"] = ckpt_hash
        pred_group.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()
        pred_group.attrs["threshold"] = self.wt_threshold
        pred_group.attrs["channel_order"] = ["TC", "WT", "ET"]
        pred_group.attrs["sw_roi_size"] = list(self.sw_roi)
        pred_group.attrs["sw_overlap"] = self.sw_overlap

        D, H, W = roi_size

        # Probabilities: [N, 3, D, H, W] float16 — raw sigmoid output
        pred_group.create_dataset(
            "probabilities",
            shape=(n_scans, 3, D, H, W),
            dtype=np.float16,
            chunks=(1, 3, D, H, W),
            compression="gzip",
            compression_opts=4,
        )

        # Labels: [N, 1, D, H, W] int8 — reconstructed {0,1,2,3}
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

        Args:
            volumes: List of ScanVolumes from extract_all().
            source: ``"manual"`` or ``"predicted"`` — which WT volume to use.

        Returns:
            List of PatientTrajectory (sorted by patient_id, then time).
        """
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

            times = np.array([s.timepoint_idx for s in scans_sorted], dtype=np.float64)

            if source == "manual":
                obs = np.array([s.manual_wt_vol_mm3 for s in scans_sorted])
            elif source == "predicted":
                obs = np.array([s.predicted_wt_vol_mm3 for s in scans_sorted])
            else:
                raise ValueError(f"Unknown source: {source}. Use 'manual' or 'predicted'.")

            # Apply log1p transform
            obs_log = np.log1p(obs)

            # Skip patients where all volumes are zero
            if np.all(obs == 0.0):
                logger.debug(f"Skipping {pid}: all volumes are zero ({source})")
                continue

            trajectories.append(
                PatientTrajectory(patient_id=pid, times=times, observations=obs_log)
            )

        logger.info(
            f"Built {len(trajectories)} trajectories from {source} volumes "
            f"(excluded: {self.exclude_patients})"
        )
        return trajectories

    def _save_cache(self, results: list[ScanVolumes]) -> None:
        """Save volume results to JSON cache."""
        cache = {
            "checkpoint_hash": _checkpoint_hash(self.ckpt_path),
            "h5_path": self.h5_path,
            "n_scans": len(results),
            "cache_version": 2,  # v2: includes TC/ET Dice and volumes
            "scans": [asdict(sv) for sv in results],
        }
        with open(self._cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        logger.info(f"Saved volume cache to {self._cache_path}")

    def _load_cache(self) -> list[ScanVolumes]:
        """Load volume results from JSON cache."""
        with open(self._cache_path) as f:
            cache = json.load(f)

        # Check cache version — v1 caches lack TC/ET fields
        if cache.get("cache_version", 1) < 2:
            logger.warning("Cache is v1 (missing TC/ET). Re-running with --force-recompute.")
            raise ValueError("Stale v1 cache")

        return [
            ScanVolumes(
                scan_id=s["scan_id"],
                patient_id=s["patient_id"],
                timepoint_idx=s["timepoint_idx"],
                manual_wt_vol_mm3=s["manual_wt_vol_mm3"],
                predicted_wt_vol_mm3=s["predicted_wt_vol_mm3"],
                manual_tc_vol_mm3=s["manual_tc_vol_mm3"],
                predicted_tc_vol_mm3=s["predicted_tc_vol_mm3"],
                manual_et_vol_mm3=s["manual_et_vol_mm3"],
                predicted_et_vol_mm3=s["predicted_et_vol_mm3"],
                wt_dice=s["wt_dice"],
                tc_dice=s["tc_dice"],
                et_dice=s["et_dice"],
                is_empty_manual=s["is_empty_manual"],
                is_empty_predicted=s["is_empty_predicted"],
            )
            for s in cache["scans"]
        ]
