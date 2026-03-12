# experiments/segment_based_approach/segment.py
"""Segmentation-based volume extraction for the A0 baseline.

Loads the frozen BrainSegFounder, runs sliding-window segmentation on each
MenGrowth H5 scan, thresholds the WT channel, and computes volumes (mm^3).
Also extracts manual WT volumes and Dice scores for comparison.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

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
    """Volume extraction results for a single scan."""

    scan_id: str
    patient_id: str
    timepoint_idx: int
    manual_vol_mm3: float
    predicted_vol_mm3: float
    wt_dice: float
    is_empty_manual: bool
    is_empty_predicted: bool


def _compute_wt_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks.

    Args:
        pred_mask: Predicted binary WT mask.
        gt_mask: Ground-truth binary WT mask.

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


class SegmentationVolumeExtractor:
    """Extract WT volumes from MenGrowth H5 using frozen BSF segmentation.

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

        self._cache_path = self.output_dir / "volume_cache.json"

    def extract_all(self, force_recompute: bool = False) -> list[ScanVolumes]:
        """Extract volumes for all scans in the H5 file.

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

        with h5py.File(self.h5_path, "r") as f:
            n_scans = f.attrs["n_scans"]
            spacing = tuple(f.attrs.get("spacing", [1.0, 1.0, 1.0]))
            voxel_vol = float(np.prod(spacing))

            scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
            patient_ids = [s.decode() if isinstance(s, bytes) else s for s in f["patient_ids"][:]]
            timepoint_idx = f["timepoint_idx"][:].astype(int)

            logger.info(f"Processing {n_scans} scans from {self.h5_path}")

            for i in range(n_scans):
                scan_id = scan_ids[i]
                pid = patient_ids[i]
                tp = int(timepoint_idx[i])

                # Manual WT volume from ground truth seg
                seg = f["segs"][i]  # [1, D, H, W] int8
                seg_np = seg[0]  # [D, H, W]
                gt_wt = (seg_np > 0).astype(np.uint8)
                manual_vol = float(gt_wt.sum() * voxel_vol)
                is_empty_manual = manual_vol == 0.0

                # Predicted WT volume from BSF
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
                    # WT = channel 1, sigmoid threshold
                    wt_prob = torch.sigmoid(logits[0, 1]).cpu().numpy()  # [D, H, W]

                pred_wt = (wt_prob > self.wt_threshold).astype(np.uint8)
                predicted_vol = float(pred_wt.sum() * voxel_vol)
                is_empty_predicted = predicted_vol == 0.0

                dice = _compute_wt_dice(pred_wt, gt_wt)

                sv = ScanVolumes(
                    scan_id=scan_id,
                    patient_id=pid,
                    timepoint_idx=tp,
                    manual_vol_mm3=manual_vol,
                    predicted_vol_mm3=predicted_vol,
                    wt_dice=dice,
                    is_empty_manual=is_empty_manual,
                    is_empty_predicted=is_empty_predicted,
                )
                results.append(sv)

                logger.info(
                    f"  [{i + 1}/{n_scans}] {scan_id}: manual={manual_vol:.0f}mm3, "
                    f"pred={predicted_vol:.0f}mm3, dice={dice:.3f}"
                    + (" [EMPTY]" if is_empty_manual else "")
                )

        self._save_cache(results)
        return results

    def build_trajectories(
        self,
        volumes: list[ScanVolumes],
        source: str = "manual",
    ) -> list[PatientTrajectory]:
        """Build PatientTrajectory objects from extracted volumes.

        Args:
            volumes: List of ScanVolumes from extract_all().
            source: ``"manual"`` or ``"predicted"`` — which volume to use.

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
                obs = np.array([s.manual_vol_mm3 for s in scans_sorted])
            elif source == "predicted":
                obs = np.array([s.predicted_vol_mm3 for s in scans_sorted])
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
            "scans": [
                {
                    "scan_id": sv.scan_id,
                    "patient_id": sv.patient_id,
                    "timepoint_idx": sv.timepoint_idx,
                    "manual_vol_mm3": sv.manual_vol_mm3,
                    "predicted_vol_mm3": sv.predicted_vol_mm3,
                    "wt_dice": sv.wt_dice,
                    "is_empty_manual": sv.is_empty_manual,
                    "is_empty_predicted": sv.is_empty_predicted,
                }
                for sv in results
            ],
        }
        with open(self._cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        logger.info(f"Saved volume cache to {self._cache_path}")

    def _load_cache(self) -> list[ScanVolumes]:
        """Load volume results from JSON cache."""
        with open(self._cache_path) as f:
            cache = json.load(f)
        return [
            ScanVolumes(
                scan_id=s["scan_id"],
                patient_id=s["patient_id"],
                timepoint_idx=s["timepoint_idx"],
                manual_vol_mm3=s["manual_vol_mm3"],
                predicted_vol_mm3=s["predicted_vol_mm3"],
                wt_dice=s["wt_dice"],
                is_empty_manual=s["is_empty_manual"],
                is_empty_predicted=s["is_empty_predicted"],
            )
            for s in cache["scans"]
        ]
