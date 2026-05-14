"""Rank discovery, per-rank data loading, validation, and NIfTI helpers."""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.uncertainty_segmentation.plotting.inter_lora.errors import (
    BaselineMismatchError,
    MissingArtefactError,
    RankDiscoveryError,
)

logger = logging.getLogger(__name__)

RUN_DIR_RE = re.compile(r"^r(?P<rank>\d+)_M(?P<members>\d+)_s(?P<seed>\d+)$")

MIN_RANKS: int = 3
BASELINE_TOL: float = 1e-6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclasses.dataclass(slots=True, frozen=True)
class RankRun:
    """All evaluation artefacts for a single LoRA rank."""

    rank: int
    run_dir: Path
    ensemble_dice: pd.DataFrame
    per_member_dice: pd.DataFrame
    baseline_dice: pd.DataFrame
    calibration: dict[str, Any]
    calibration_coverage: pd.DataFrame
    bias_diagnostics: pd.DataFrame
    bias_dominance: pd.DataFrame
    epistemic_taxonomy: dict[str, Any]
    statistical_summary: dict[str, Any]
    paired_differences: pd.DataFrame
    predictions_dir: Path | None


@dataclasses.dataclass(slots=True, frozen=True)
class InterLoraData:
    """Aggregated data across all discovered ranks."""

    root_dir: Path
    out_root: Path
    ranks: tuple[RankRun, ...]
    compiled_metrics: pd.DataFrame
    git_sha: str
    run_timestamp: str
    selected_slices: dict[str, Any]

    @property
    def rank_values(self) -> list[int]:
        """Sorted list of non-zero rank integers."""
        return [r.rank for r in self.ranks if r.rank > 0]

    @property
    def all_rank_values(self) -> list[int]:
        """Sorted list of all rank integers including baseline (0)."""
        return [r.rank for r in self.ranks]

    def get_rank(self, rank: int) -> RankRun:
        """Retrieve RankRun by rank value."""
        for r in self.ranks:
            if r.rank == rank:
                return r
        msg = f"Rank {rank} not found in {self.all_rank_values}"
        raise KeyError(msg)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_ranks(
    root: Path,
    expected: frozenset[int] | None = None,
) -> list[Path]:
    """Discover rank directories under root, sorted by rank ascending.

    Args:
        root: Root directory containing r*_M*_s* subdirectories.
        expected: If provided, filter to only these ranks.

    Returns:
        Sorted list of rank directory paths.

    Raises:
        RankDiscoveryError: If fewer than MIN_RANKS ranks found.
    """
    dirs: list[tuple[int, Path]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if m is None:
            continue
        rank = int(m.group("rank"))
        if expected is not None and rank not in expected:
            continue
        dirs.append((rank, d))

    dirs.sort(key=lambda t: t[0])

    if len(dirs) < MIN_RANKS:
        msg = (
            f"Found {len(dirs)} rank(s) under {root}, need >= {MIN_RANKS}. "
            f"Discovered: {[d.name for _, d in dirs]}"
        )
        raise RankDiscoveryError(msg)

    logger.info(
        "Discovered %d ranks: %s",
        len(dirs),
        [r for r, _ in dirs],
    )
    return [d for _, d in dirs]


def discover_baseline(root: Path) -> Path | None:
    """Return baseline_frozen_bsf/ directory if present."""
    candidate = root / "baseline_frozen_bsf"
    if candidate.is_dir():
        return candidate
    return None


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        msg = f"Required file not found: {path}"
        raise MissingArtefactError(msg)
    return pd.read_csv(path)


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.debug("Optional file missing: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Required file not found: {path}"
        raise MissingArtefactError(msg)
    with open(path) as f:
        return json.load(f)


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.debug("Optional file missing: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-rank loading
# ---------------------------------------------------------------------------
def load_rank_run(rank_dir: Path, *, strict: bool = False) -> RankRun:
    """Load all evaluation artefacts for one rank directory.

    Args:
        rank_dir: Path to e.g. r8_M20_s42/.
        strict: If True, raise MissingArtefactError on any missing file.

    Returns:
        Populated RankRun dataclass.
    """
    m = RUN_DIR_RE.match(rank_dir.name)
    if m is None:
        msg = f"Cannot parse rank from directory name: {rank_dir.name}"
        raise RankDiscoveryError(msg)
    rank = int(m.group("rank"))

    ev = rank_dir / "evaluation"
    read_csv = _read_csv if strict else _read_csv_optional
    read_json = _read_json if strict else _read_json_optional

    ensemble_dice = _read_csv(ev / "ensemble_test_dice.csv")
    per_member_dice = _read_csv(ev / "per_member_test_dice.csv")
    baseline_dice = _read_csv(ev / "baseline_test_dice.csv")
    statistical_summary = _read_json(ev / "statistical_summary.json")

    calibration = read_json(ev / "calibration.json")
    calibration_coverage = read_csv(ev / "calibration_coverage.csv")
    bias_diagnostics = read_csv(ev / "bias_diagnostics.csv")
    bias_dominance = read_csv(ev / "bias_dominance_threshold.csv")
    epistemic_taxonomy = read_json(ev / "epistemic_taxonomy.json")
    paired_differences = read_csv(ev / "paired_differences.csv")

    pred_dir = rank_dir / "predictions"
    predictions_dir = pred_dir if pred_dir.is_dir() else None

    logger.info("Loaded rank r=%d from %s", rank, rank_dir.name)

    return RankRun(
        rank=rank,
        run_dir=rank_dir,
        ensemble_dice=ensemble_dice,
        per_member_dice=per_member_dice,
        baseline_dice=baseline_dice,
        calibration=calibration,
        calibration_coverage=calibration_coverage,
        bias_diagnostics=bias_diagnostics,
        bias_dominance=bias_dominance,
        epistemic_taxonomy=epistemic_taxonomy,
        statistical_summary=statistical_summary,
        paired_differences=paired_differences,
        predictions_dir=predictions_dir,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_baseline_consistency(runs: list[RankRun]) -> None:
    """Assert that baseline_test_dice.csv is identical across all ranks.

    Raises:
        BaselineMismatchError: If max absolute difference exceeds BASELINE_TOL.
    """
    if len(runs) < 2:
        return

    ref = runs[0].baseline_dice
    dice_cols = ["dice_tc", "dice_wt", "dice_et"]

    for run in runs[1:]:
        other = run.baseline_dice
        if len(ref) != len(other):
            msg = (
                f"Baseline row count mismatch: r={runs[0].rank} has {len(ref)}, "
                f"r={run.rank} has {len(other)}"
            )
            raise BaselineMismatchError(msg)

        ref_merged = ref.set_index("scan_id")[dice_cols]
        other_merged = other.set_index("scan_id")[dice_cols]
        common = ref_merged.index.intersection(other_merged.index)
        if len(common) < len(ref_merged):
            msg = f"Baseline scan_id mismatch between r={runs[0].rank} and r={run.rank}"
            raise BaselineMismatchError(msg)

        max_diff = (ref_merged.loc[common] - other_merged.loc[common]).abs().max().max()
        if max_diff > BASELINE_TOL:
            msg = (
                f"Baseline differs between r={runs[0].rank} and r={run.rank}: "
                f"max abs diff = {max_diff:.2e} > {BASELINE_TOL:.0e}"
            )
            raise BaselineMismatchError(msg)

    logger.info("Baseline consistency validated across %d ranks", len(runs))


# ---------------------------------------------------------------------------
# Subject selection for Qual1
# ---------------------------------------------------------------------------
def select_subjects_auto(
    ranks: tuple[RankRun, ...],
) -> dict[str, Any]:
    """Auto-select subjects for Qual1 slice grid.

    Returns dict with keys:
        brats_men: {scan_id, slice_idx}
        mengrowth: {scan_id, study_pattern, slice_idx}
    """
    max_rank_run = max(
        (r for r in ranks if r.rank > 0),
        key=lambda r: r.rank,
    )

    # BraTS-MEN: scan nearest to median ensemble Dice (WT)
    ens = max_rank_run.ensemble_dice.copy()
    median_dice = ens["dice_wt"].median()
    ens["dist_to_median"] = (ens["dice_wt"] - median_dice).abs()
    brats_scans = ens[ens["scan_id"].str.startswith("BraTS-MEN")]
    if brats_scans.empty:
        brats_scan_id = ens.sort_values("dist_to_median").iloc[0]["scan_id"]
    else:
        brats_scans = brats_scans.sort_values(
            ["dist_to_median", "volume_ensemble"],
            ascending=[True, False],
        )
        brats_scan_id = brats_scans.iloc[0]["scan_id"]

    # MenGrowth: highest inter-member std on volume
    pm = max_rank_run.per_member_dice
    mengrowth_members = pm[pm["scan_id"].str.startswith("MenGrowth")]
    if mengrowth_members.empty:
        mengrowth_id = None
    else:
        vol_std = (
            mengrowth_members.groupby("scan_id")["volume_pred"].std().sort_values(ascending=False)
        )
        mengrowth_id = vol_std.index[0]

    result: dict[str, Any] = {
        "brats_men": {"scan_id": brats_scan_id, "slice_idx": None},
    }
    if mengrowth_id is not None:
        result["mengrowth"] = {"scan_id": mengrowth_id, "slice_idx": None}

    logger.info(
        "Auto-selected subjects: BraTS-MEN=%s, MenGrowth=%s",
        brats_scan_id,
        mengrowth_id,
    )
    return result


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------
def _find_scan_dir(predictions_dir: Path, scan_id: str) -> Path | None:
    """Probe both predictions/{scan_id} and predictions/brats_men_test/{scan_id}."""
    for subdir in [
        predictions_dir / scan_id,
        predictions_dir / "brats_men_test" / scan_id,
    ]:
        if subdir.is_dir():
            return subdir
    parts = scan_id.split("-")
    if len(parts) >= 3:
        for d in predictions_dir.iterdir():
            if d.is_dir() and d.name.startswith(scan_id[:15]):
                return d
    return None


def load_nifti_slice(
    path: Path,
    slice_idx: int,
    axis: int = 2,
) -> np.ndarray:
    """Load a single axial slice from a NIfTI file, RAS-canonicalized.

    Args:
        path: Path to .nii.gz file.
        slice_idx: Slice index along the specified axis.
        axis: Axis to slice along (default 2 = axial).

    Returns:
        2D numpy array for the requested slice.
    """
    import nibabel as nib

    img = nib.load(str(path))
    img_canonical = nib.as_closest_canonical(img)
    data = img_canonical.get_fdata()

    slc = [slice(None)] * data.ndim
    slc[axis] = slice_idx
    result = data[tuple(slc)].copy()

    del data, img_canonical, img
    gc.collect()

    return result


def get_nifti_affine(path: Path) -> np.ndarray:
    """Return the affine matrix from a NIfTI file (RAS-canonical)."""
    import nibabel as nib

    img = nib.load(str(path))
    img_canonical = nib.as_closest_canonical(img)
    affine = img_canonical.affine.copy()
    del img_canonical, img
    return affine


def find_largest_tumor_slice(
    mask_path: Path,
    axis: int = 2,
) -> int:
    """Find the axial slice with largest tumour area in a mask NIfTI."""
    import nibabel as nib

    img = nib.load(str(mask_path))
    img_canonical = nib.as_closest_canonical(img)
    data = img_canonical.get_fdata()

    if data.ndim == 4:
        data = data.sum(axis=-1)
    binary = (data > 0.5).astype(np.float32)

    area_per_slice = (
        binary.sum(axis=(0, 1))
        if axis == 2
        else binary.sum(axis=tuple(i for i in range(binary.ndim) if i != axis))
    )
    best_slice = int(np.argmax(area_per_slice))

    del data, binary, img_canonical, img
    gc.collect()
    return best_slice


def compute_voxelwise_variance_slice(
    scan_dir: Path,
    slice_idx: int,
    n_members: int = 20,
    axis: int = 2,
) -> np.ndarray:
    """Online Welford variance of member probs for a single slice.

    Returns the std map (sqrt of variance) summed over foreground channels.
    """
    mean = None
    m2 = None
    count = 0

    for m_idx in range(n_members):
        prob_path = scan_dir / f"member_{m_idx}_probs.nii.gz"
        if not prob_path.exists():
            continue
        slc = load_nifti_slice(prob_path, slice_idx, axis)
        if slc.ndim == 2:
            slc = slc[..., np.newaxis]

        count += 1
        if mean is None:
            mean = slc.copy().astype(np.float64)
            m2 = np.zeros_like(mean, dtype=np.float64)
        else:
            delta = slc - mean
            mean += delta / count
            delta2 = slc - mean
            m2 += delta * delta2

        del slc
        gc.collect()

    if count < 2 or m2 is None:
        return np.zeros((1, 1), dtype=np.float32)

    variance = m2 / (count - 1)
    std_map = np.sqrt(variance)
    if std_map.ndim == 3:
        std_map = std_map.sum(axis=-1)
    return std_map.astype(np.float32)


def binary_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel binary entropy of sigmoid probabilities.

    For each channel independently: ``H = -(p log p + (1-p) log(1-p))``.
    Maximum entropy is ``ln(2) ~= 0.693`` at ``p = 0.5``. Mirrors
    ``engine.uncertainty_metrics.compute_binary_entropy`` so the plotting
    layer reports the same predictive-entropy definition as the pipeline.

    Args:
        probs: Sigmoid probabilities, any shape. Values in ``[0, 1]``.
        eps: Clamp applied before the logarithms for numerical stability.

    Returns:
        Binary entropy, same shape as ``probs``.
    """
    p = np.clip(probs.astype(np.float64), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def compute_voxelwise_entropy_slice(
    scan_dir: Path,
    slice_idx: int,
    channel: int | None = 0,
    n_members: int = 20,
    axis: int = 2,
) -> np.ndarray:
    """Predictive binary entropy of the ensemble-mean probability for one slice.

    The predictive entropy is ``H[mean_m p_m]`` evaluated per channel with the
    binary-entropy functional. The ensemble-mean probability is read from
    ``ensemble_probs.nii.gz`` when available (the per-voxel mean is already
    persisted for sample scans); otherwise it is recomputed by averaging the
    per-member ``member_*_probs.nii.gz`` maps. The precomputed
    ``entropy.nii.gz`` artefact is intentionally not used — it is stored as
    all-NaN by the current inference pipeline.

    Args:
        scan_dir: Prediction directory for one scan.
        slice_idx: Slice index along ``axis``.
        channel: Channel to return (``0`` = BSF ch0 = meningioma mass). If
            ``None``, the per-channel entropy is summed over all channels.
        n_members: Number of ensemble members to look for in the fallback path.
        axis: Axis to slice along (default 2 = axial).

    Returns:
        2D entropy map for the requested slice (float32). Returns a ``(1, 1)``
        zero array when no probability maps are available.
    """
    ensemble_path = scan_dir / "ensemble_probs.nii.gz"
    if ensemble_path.exists():
        slc = load_nifti_slice(ensemble_path, slice_idx, axis)
        if slc.ndim == 2:
            slc = slc[..., np.newaxis]
        entropy = binary_entropy(slc)
    else:
        prob_sum = None
        count = 0
        for m_idx in range(n_members):
            prob_path = scan_dir / f"member_{m_idx}_probs.nii.gz"
            if not prob_path.exists():
                continue
            slc = load_nifti_slice(prob_path, slice_idx, axis)
            if slc.ndim == 2:
                slc = slc[..., np.newaxis]
            count += 1
            if prob_sum is None:
                prob_sum = slc.astype(np.float64)
            else:
                prob_sum += slc
            del slc
            gc.collect()

        if count == 0 or prob_sum is None:
            return np.zeros((1, 1), dtype=np.float32)
        entropy = binary_entropy(prob_sum / count)

    if entropy.ndim == 3:
        if channel is None or channel >= entropy.shape[-1]:
            entropy = entropy.sum(axis=-1)
        else:
            entropy = entropy[..., channel]
    entropy = np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    return entropy.astype(np.float32)


def compute_scan_mean_entropy(
    scan_dir: Path,
    channel: int = 0,
    n_members: int = 20,
) -> float:
    """Mean predictive binary entropy within the predicted meningioma mask.

    Whole-volume scan-level summary of the qual1 voxelwise entropy map:
    predictive binary entropy of the ensemble-mean probability on ``channel``
    (BSF ch0 = meningioma), averaged over voxels inside
    ``ensemble_mask.nii.gz``. The ensemble-mean probability comes from
    ``ensemble_probs.nii.gz`` when present, otherwise the per-member maps are
    averaged. Falls back to the whole-volume mean when no mask is available.

    Args:
        scan_dir: Prediction directory for one scan.
        channel: Probability channel to evaluate (default 0 = meningioma).
        n_members: Number of ensemble members for the fallback averaging path.

    Returns:
        Mean entropy in nats, or ``nan`` when no probability map is available.
    """
    import nibabel as nib

    ensemble_path = scan_dir / "ensemble_probs.nii.gz"
    if ensemble_path.exists():
        probs = np.asarray(nib.load(str(ensemble_path)).dataobj, dtype=np.float64)
    else:
        prob_sum = None
        count = 0
        for m_idx in range(n_members):
            prob_path = scan_dir / f"member_{m_idx}_probs.nii.gz"
            if not prob_path.exists():
                continue
            arr = np.asarray(nib.load(str(prob_path)).dataobj, dtype=np.float64)
            prob_sum = arr if prob_sum is None else prob_sum + arr
            count += 1
            del arr
            gc.collect()
        if count == 0 or prob_sum is None:
            return float("nan")
        probs = prob_sum / count

    if probs.ndim == 4:
        ch = channel if channel < probs.shape[-1] else 0
        entropy = binary_entropy(probs[..., ch])
    else:
        entropy = binary_entropy(probs)

    mask_path = scan_dir / "ensemble_mask.nii.gz"
    if mask_path.exists():
        mask = np.asarray(nib.load(str(mask_path)).dataobj) > 0.5
        if mask.ndim == 4:
            mask = mask[..., 0]
        if mask.any():
            return float(np.nan_to_num(entropy[mask]).mean())
    return float(np.nan_to_num(entropy).mean())
