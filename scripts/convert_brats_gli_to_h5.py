#!/usr/bin/env python
# scripts/convert_brats_gli_to_h5.py
"""Convert BraTS-GLI (Glioma) NIfTI files to a single HDF5 file with longitudinal hierarchy.

Pre-applies spatial preprocessing (Orient -> Resample -> CropForeground -> SpatialPad ->
CenterCrop at 192^3) but does NOT apply z-score normalization.

GLI has 613 patients, 1350 scans (1-10 timepoints each), with labels {0,1,2,3,4}
where label 4 = Resection Cavity (GLI-specific). Raw labels are stored as-is;
label 4 -> 1 remapping is applied only for semantic features.

H5 Schema:
    brats_gli.h5
    |-- attrs: {n_scans, n_patients, roi_size, spacing, channel_order, version,
    |           dataset_type="longitudinal", domain="GLI"}
    |-- images           [N_scans, 4, 192, 192, 192] float32
    |-- segs             [N_scans, 1, 192, 192, 192] int8  (raw labels 0-4)
    |-- scan_ids         [N_scans] str
    |-- patient_ids      [N_scans] str
    |-- timepoint_idx    [N_scans] int32
    |-- semantic/
    |   |-- volume       [N_scans, 4] float32  (label 4 merged into NCR)
    |   |-- location     [N_scans, 3] float32
    |   +-- shape        [N_scans, 3] float32
    |-- longitudinal/
    |   |-- patient_offsets  [N_patients+1] int32  (CSR format)
    |   +-- patient_list     [N_patients] str
    |-- splits/  (patient-level indices into patient_list)
    |   |-- lora_train   int32
    |   |-- lora_val     int32
    |   +-- test         int32
    +-- metadata/
        |-- grade        [N_scans] int8
        |-- age          [N_scans] float32
        +-- sex          [N_scans] str

Usage:
    # Dry-run with 5 patients
    python scripts/convert_brats_gli_to_h5.py \
        --data-root /media/mpascual/PortableSSD/BraTS_GLI/source/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2 \
        --output /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/BraTS_GLI.h5 \
        --max-patients 5

    # Full conversion
    python scripts/convert_brats_gli_to_h5.py \
        --data-root /media/mpascual/PortableSSD/BraTS_GLI/source/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2 \
        --output /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/BraTS_GLI.h5
    
    python scripts/convert_brats_gli_to_h5.py \
        --data-root /media/mpascual/PortableSSD/BraTS_GLI/source/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2 \
        --output /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/source/BraTS_GLI.h5 ; python scripts/convert_brats_men_to_h5.py \
        --data-root /media/mpascual/PortableSSD/Meningiomas/BraTS/BraTS_Men_Train \
        --output /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/source/BraTS_MEN.h5 ; python scripts/convert_mengrowth_to_h5.py \
        --data-root /media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/MenGrowth-2025 \
        --output /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/source/MenGrowth.h5
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
    SpatialPadd,
)
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from growth.data.bratsmendata import split_subjects_multi
from growth.data.semantic_features import extract_semantic_features
from growth.data.transforms import (
    DEFAULT_SPACING,
    FEATURE_ROI_SIZE,
    MODALITY_KEYS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# H5 file version
H5_VERSION = "2.0"

# Modality suffixes for BraTS-GLI
_MODALITY_SUFFIXES = {
    "t1c": "-t1c.nii.gz",
    "t1n": "-t1n.nii.gz",
    "t2f": "-t2f.nii.gz",
    "t2w": "-t2w.nii.gz",
}
_SEG_SUFFIX = "-seg.nii.gz"

# Default split sizes (patient-level, ~70/10/20 split)
DEFAULT_SPLIT_SIZES = {
    "lora_train": 430,
    "lora_val": 60,
    "test": 123,
}

# Pattern for parsing scan directory names: BraTS-GLI-{patient}-{timepoint}
_SCAN_PATTERN = re.compile(r"^(BraTS-GLI-\d+)-(\d+)$")


# =========================================================================
# Discovery
# =========================================================================


def discover_gli_scans(
    data_root: Path,
) -> list[dict[str, str | Path]]:
    """Discover all BraTS-GLI scans and parse patient/timepoint structure.

    Args:
        data_root: Path containing BraTS-GLI-* scan directories.

    Returns:
        List of dicts with keys: 'scan_id', 'patient_id', 'timepoint', 'scan_dir'.
        Sorted by (patient_id, timepoint).
    """
    scans = []
    for item in sorted(data_root.iterdir()):
        if not item.is_dir():
            continue
        match = _SCAN_PATTERN.match(item.name)
        if match is None:
            continue
        patient_id = match.group(1)
        timepoint = int(match.group(2))
        scans.append({
            "scan_id": item.name,
            "patient_id": patient_id,
            "timepoint": timepoint,
            "scan_dir": item,
        })

    # Sort by patient then timepoint
    scans.sort(key=lambda s: (s["patient_id"], s["timepoint"]))
    return scans


def build_longitudinal_structure(
    scans: list[dict],
) -> tuple[list[str], np.ndarray, list[int]]:
    """Build CSR-style longitudinal indexing from scan list.

    Args:
        scans: Sorted list of scan dicts (from discover_gli_scans).

    Returns:
        Tuple of (patient_list, patient_offsets, timepoint_indices).
        - patient_list: Unique patient IDs in order.
        - patient_offsets: CSR array [N_patients+1] where patient i's scans
          are at indices [offsets[i]:offsets[i+1]].
        - timepoint_indices: Sequential 0-based timepoint index per scan.
    """
    patient_list = []
    patient_offsets = [0]
    timepoint_indices = []

    current_patient = None
    tp_counter = 0

    for scan in scans:
        pid = scan["patient_id"]
        if pid != current_patient:
            if current_patient is not None:
                patient_offsets.append(len(timepoint_indices))
            patient_list.append(pid)
            current_patient = pid
            tp_counter = 0

        timepoint_indices.append(tp_counter)
        tp_counter += 1

    # Final offset
    patient_offsets.append(len(timepoint_indices))

    return patient_list, np.array(patient_offsets, dtype=np.int32), timepoint_indices


def load_scan_paths(scan_dir: Path, scan_id: str) -> dict[str, Path]:
    """Get paths to all modalities and segmentation for a scan.

    Args:
        scan_dir: Path to scan directory.
        scan_id: Scan directory name.

    Returns:
        Dict with keys: 't1c', 't1n', 't2f', 't2w', 'seg'.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    paths: dict[str, Path] = {}
    for modality, suffix in _MODALITY_SUFFIXES.items():
        path = scan_dir / f"{scan_id}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Missing {modality} file: {path}")
        paths[modality] = path

    seg_path = scan_dir / f"{scan_id}{_SEG_SUFFIX}"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
    paths["seg"] = seg_path

    return paths


# =========================================================================
# Preprocessing
# =========================================================================


def build_preprocessing_transforms() -> Compose:
    """Build MONAI transforms for spatial preprocessing without normalization.

    Pipeline: Load -> EnsureChannelFirst -> Orient(RAS) -> Spacing(1mm) ->
              CropForeground -> SpatialPad(192^3) -> CenterCrop(192^3) ->
              Concat -> EnsureType

    Returns:
        MONAI Compose pipeline.
    """
    modality_keys = list(MODALITY_KEYS)
    seg_key = "seg"
    all_keys = modality_keys + [seg_key]
    roi_size = list(FEATURE_ROI_SIZE)

    transforms = [
        LoadImaged(keys=all_keys, image_only=False),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=modality_keys,
            pixdim=DEFAULT_SPACING,
            mode=["bilinear"] * len(modality_keys),
        ),
        Spacingd(keys=[seg_key], pixdim=DEFAULT_SPACING, mode=("nearest",)),
        CropForegroundd(
            keys=all_keys,
            source_key=modality_keys[0],
            k_divisible=roi_size,
        ),
        SpatialPadd(keys=all_keys, spatial_size=roi_size),
        ResizeWithPadOrCropd(keys=all_keys, spatial_size=roi_size),
        ConcatItemsd(keys=modality_keys, name="image", dim=0),
        EnsureTyped(keys=["image", seg_key], dtype="float32", track_meta=False),
    ]

    return Compose(transforms)


# =========================================================================
# Report Generation
# =========================================================================


def generate_longitudinal_report(h5_path: Path, output_dir: Path) -> None:
    """Generate a full longitudinal analysis report from a GLI H5 file.

    Creates plots and a CSV summarizing volume evolution, growth rates,
    shape changes, centroid drift, inter-scan Dice, and label composition.

    Args:
        h5_path: Path to the GLI H5 file.
        output_dir: Directory to write report files.
    """
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating longitudinal report in {output_dir}")

    with h5py.File(h5_path, "r") as f:
        n_scans = f.attrs["n_scans"]
        n_patients = f.attrs["n_patients"]
        patient_list = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["longitudinal/patient_list"][:]
        ]
        offsets = f["longitudinal/patient_offsets"][:].astype(int)
        scan_ids = [
            s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]
        ]
        patient_ids = [
            s.decode() if isinstance(s, bytes) else s for s in f["patient_ids"][:]
        ]
        tp_idx = f["timepoint_idx"][:].astype(int)

        volumes = f["semantic/volume"][:]  # [N, 4]
        locations = f["semantic/location"][:]  # [N, 3]
        shapes = f["semantic/shape"][:]  # [N, 3]

    # Compute total volume (exp(log_vol) - 1 to reverse log1p)
    total_vol = np.expm1(volumes[:, 0])

    # ---- 1. Summary text ----
    tp_counts = np.diff(offsets)
    summary_lines = [
        f"BraTS-GLI Longitudinal Report",
        f"{'=' * 50}",
        f"Total patients: {n_patients}",
        f"Total scans: {n_scans}",
        f"Timepoints per patient: min={tp_counts.min()}, max={tp_counts.max()}, "
        f"mean={tp_counts.mean():.1f}, median={np.median(tp_counts):.0f}",
        "",
        "Timepoint distribution:",
    ]
    for n_tp in sorted(set(tp_counts)):
        count = (tp_counts == n_tp).sum()
        summary_lines.append(f"  {n_tp} timepoint(s): {count} patients")

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    logger.info(f"  Written: {summary_path.name}")

    # ---- 2. Timepoint distribution histogram ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tp_counts, bins=range(1, tp_counts.max() + 2), align="left",
            edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of timepoints")
    ax.set_ylabel("Number of patients")
    ax.set_title("Timepoints per Patient")
    fig.tight_layout()
    fig.savefig(output_dir / "timepoint_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("  Written: timepoint_distribution.png")

    # ---- 3. Volume evolution (spaghetti plot) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_patients):
        start, end = offsets[i], offsets[i + 1]
        if end - start < 2:
            continue
        tps = tp_idx[start:end]
        vols = total_vol[start:end]
        ax.plot(tps, vols, alpha=0.15, color="steelblue", linewidth=0.8)

    # Mean trajectory
    max_tp = tp_idx.max() + 1
    mean_vol = np.full(max_tp, np.nan)
    std_vol = np.full(max_tp, np.nan)
    for t in range(max_tp):
        mask = tp_idx == t
        if mask.sum() > 0:
            mean_vol[t] = total_vol[mask].mean()
            std_vol[t] = total_vol[mask].std()

    valid = ~np.isnan(mean_vol)
    ax.plot(np.where(valid)[0], mean_vol[valid], color="darkred", linewidth=2,
            label="Mean")
    ax.fill_between(
        np.where(valid)[0],
        (mean_vol - std_vol)[valid],
        (mean_vol + std_vol)[valid],
        alpha=0.2, color="darkred",
    )
    ax.set_xlabel("Timepoint index")
    ax.set_ylabel("Total tumor volume (mm^3)")
    ax.set_title("Volume Evolution per Patient")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "volume_evolution.png", dpi=150)
    plt.close(fig)
    logger.info("  Written: volume_evolution.png")

    # ---- 4. Component volume evolution ----
    # volumes[:, 1:4] = log1p of [NCR, ED, ET]
    component_names = ["NCR", "ED", "ET"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c, (ax, name) in enumerate(zip(axes, component_names)):
        comp_vol = np.expm1(volumes[:, c + 1])
        for t in range(max_tp):
            mask = tp_idx == t
            if mask.sum() > 0:
                ax.boxplot(comp_vol[mask], positions=[t], widths=0.6, showfliers=False)
        ax.set_xlabel("Timepoint index")
        ax.set_ylabel("Volume (mm^3)")
        ax.set_title(f"{name} Volume")
    fig.tight_layout()
    fig.savefig(output_dir / "component_evolution.png", dpi=150)
    plt.close(fig)
    logger.info("  Written: component_evolution.png")

    # ---- 5. Volume growth rates ----
    growth_rates = []
    for i in range(n_patients):
        start, end = offsets[i], offsets[i + 1]
        if end - start < 2:
            continue
        vols = total_vol[start:end]
        for j in range(1, len(vols)):
            growth_rates.append(vols[j] - vols[j - 1])

    fig, ax = plt.subplots(figsize=(8, 5))
    if growth_rates:
        ax.hist(growth_rates, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Inter-scan volume change (mm^3)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Volume Growth Rates")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "volume_growth_rates.png", dpi=150)
    plt.close(fig)
    logger.info("  Written: volume_growth_rates.png")

    # ---- 6. Centroid drift ----
    centroid_drifts = []
    for i in range(n_patients):
        start, end = offsets[i], offsets[i + 1]
        if end - start < 2:
            continue
        locs = locations[start:end]  # [T, 3], normalized [0, 1]
        for j in range(1, len(locs)):
            drift = np.linalg.norm(locs[j] - locs[j - 1])
            centroid_drifts.append(drift)

    fig, ax = plt.subplots(figsize=(8, 5))
    if centroid_drifts:
        ax.hist(centroid_drifts, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Centroid displacement (normalized)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-scan Centroid Drift")
    fig.tight_layout()
    fig.savefig(output_dir / "centroid_drift.png", dpi=150)
    plt.close(fig)
    logger.info("  Written: centroid_drift.png")

    # ---- 7. Shape evolution ----
    shape_names = ["Sphericity", "Enhancement Ratio", "Infiltration Index"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c, (ax, name) in enumerate(zip(axes, shape_names)):
        vals = shapes[:, c]
        for t in range(max_tp):
            mask = tp_idx == t
            if mask.sum() > 0:
                ax.boxplot(vals[mask], positions=[t], widths=0.6, showfliers=False)
        ax.set_xlabel("Timepoint index")
        ax.set_ylabel(name)
        ax.set_title(f"{name} over Time")
    fig.tight_layout()
    fig.savefig(output_dir / "shape_evolution.png", dpi=150)
    plt.close(fig)
    logger.info("  Written: shape_evolution.png")

    # ---- 8. Label composition (stacked bar) ----
    label_names = ["Background", "NCR", "ED", "ET", "RC"]
    label_fracs_per_tp = {t: np.zeros(5) for t in range(max_tp)}
    label_counts_per_tp = {t: 0 for t in range(max_tp)}

    with h5py.File(h5_path, "r") as f:
        for idx in range(n_scans):
            seg = f["segs"][idx][0]  # [D, H, W]
            tp = tp_idx[idx]
            for lbl in range(5):
                label_fracs_per_tp[tp][lbl] += (seg == lbl).sum()
            label_counts_per_tp[tp] += seg.size

    fig, ax = plt.subplots(figsize=(10, 6))
    tps_to_plot = [t for t in range(max_tp) if label_counts_per_tp[t] > 0]
    bottom = np.zeros(len(tps_to_plot))
    colors = ["#cccccc", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for lbl in range(5):
        fracs = np.array([
            label_fracs_per_tp[t][lbl] / label_counts_per_tp[t]
            for t in tps_to_plot
        ])
        ax.bar(tps_to_plot, fracs, bottom=bottom, label=label_names[lbl],
               color=colors[lbl], edgecolor="white", linewidth=0.5)
        bottom += fracs

    ax.set_xlabel("Timepoint index")
    ax.set_ylabel("Fraction")
    ax.set_title("Label Composition across Timepoints")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "label_composition.png", dpi=150)
    plt.close(fig)
    logger.info("  Written: label_composition.png")

    # ---- 9. Inter-scan Dice (from stored segmentations) ----
    def _dice_binary(a: np.ndarray, b: np.ndarray) -> float:
        intersection = np.sum(a & b)
        total = np.sum(a) + np.sum(b)
        return 2.0 * intersection / total if total > 0 else 1.0

    dice_tc_list, dice_wt_list, dice_et_list = [], [], []

    with h5py.File(h5_path, "r") as f:
        for i in range(n_patients):
            start, end = offsets[i], offsets[i + 1]
            if end - start < 2:
                continue
            for j in range(start, end - 1):
                seg_t = f["segs"][j][0].astype(np.int8)
                seg_t1 = f["segs"][j + 1][0].astype(np.int8)

                # TC: labels 1, 3 (and 4 merged as NCR)
                tc_t = (seg_t == 1) | (seg_t == 3) | (seg_t == 4)
                tc_t1 = (seg_t1 == 1) | (seg_t1 == 3) | (seg_t1 == 4)
                dice_tc_list.append(_dice_binary(tc_t, tc_t1))

                # WT: labels 1, 2, 3, 4
                wt_t = seg_t > 0
                wt_t1 = seg_t1 > 0
                dice_wt_list.append(_dice_binary(wt_t, wt_t1))

                # ET: label 3
                et_t = seg_t == 3
                et_t1 = seg_t1 == 3
                dice_et_list.append(_dice_binary(et_t, et_t1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, name in zip(
        axes,
        [dice_tc_list, dice_wt_list, dice_et_list],
        ["TC", "WT", "ET"],
    ):
        if vals:
            ax.hist(vals, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Dice coefficient")
        ax.set_ylabel("Count")
        ax.set_title(f"Inter-scan Dice ({name})")
        ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "inter_scan_dice.png", dpi=150)
    plt.close(fig)
    logger.info("  Written: inter_scan_dice.png")

    # ---- 10. Patient trajectories CSV ----
    csv_path = output_dir / "patient_trajectories.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "patient_id", "scan_id", "timepoint", "total_volume",
            "ncr_volume", "ed_volume", "et_volume",
            "centroid_z", "centroid_y", "centroid_x",
            "sphericity", "enhancement_ratio", "infiltration_index",
            "growth_rate",
        ])
        for i in range(n_patients):
            start, end = offsets[i], offsets[i + 1]
            pid = patient_list[i]
            for j in range(start, end):
                vols_raw = np.expm1(volumes[j])  # [4]
                gr = ""
                if j > start:
                    gr = f"{vols_raw[0] - np.expm1(volumes[j - 1][0]):.1f}"
                writer.writerow([
                    pid, scan_ids[j], tp_idx[j],
                    f"{vols_raw[0]:.1f}", f"{vols_raw[1]:.1f}",
                    f"{vols_raw[2]:.1f}", f"{vols_raw[3]:.1f}",
                    f"{locations[j, 0]:.4f}", f"{locations[j, 1]:.4f}",
                    f"{locations[j, 2]:.4f}",
                    f"{shapes[j, 0]:.4f}", f"{shapes[j, 1]:.4f}",
                    f"{shapes[j, 2]:.4f}",
                    gr,
                ])
    logger.info(f"  Written: {csv_path.name}")

    logger.info("Longitudinal report generation complete.")


# =========================================================================
# Conversion
# =========================================================================


def convert(
    data_root: str,
    output_path: str,
    max_patients: int | None = None,
    seed: int = 42,
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    """Convert GLI NIfTI files to HDF5 with longitudinal hierarchy.

    Args:
        data_root: Path to BraTS-GLI scan directory.
        output_path: Output H5 file path.
        max_patients: Max patients for dry-run (None = all).
        seed: Random seed for split generation.
        compression: HDF5 compression algorithm.
        compression_level: Compression level (1-9).
    """
    data_root = Path(data_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover scans
    all_scans = discover_gli_scans(data_root)
    logger.info(f"Found {len(all_scans)} scans in {data_root}")

    # Build longitudinal structure
    patient_list, patient_offsets, timepoint_indices = build_longitudinal_structure(all_scans)
    n_patients = len(patient_list)
    logger.info(f"Found {n_patients} patients")

    # Optionally limit patients
    if max_patients is not None and max_patients < n_patients:
        # Keep first max_patients
        last_scan_idx = int(patient_offsets[max_patients])
        all_scans = all_scans[:last_scan_idx]
        patient_list = patient_list[:max_patients]
        patient_offsets = patient_offsets[:max_patients + 1]
        timepoint_indices = timepoint_indices[:last_scan_idx]
        n_patients = max_patients
        logger.info(f"Limiting to {max_patients} patients ({len(all_scans)} scans)")

    n_scans = len(all_scans)
    roi = list(FEATURE_ROI_SIZE)

    # Build preprocessing pipeline
    transforms = build_preprocessing_transforms()

    # Generate patient-level splits
    if max_patients is None or n_patients >= sum(DEFAULT_SPLIT_SIZES.values()):
        split_sizes = DEFAULT_SPLIT_SIZES
    else:
        total_default = sum(DEFAULT_SPLIT_SIZES.values())
        split_sizes = {}
        allocated = 0
        for name, size in DEFAULT_SPLIT_SIZES.items():
            if name == list(DEFAULT_SPLIT_SIZES.keys())[-1]:
                split_sizes[name] = n_patients - allocated
            else:
                split_sizes[name] = max(1, round(size * n_patients / total_default))
                allocated += split_sizes[name]

    patient_splits = split_subjects_multi(patient_list, split_sizes, seed=seed)

    # Build patient -> index mapping
    patient_to_idx = {pid: i for i, pid in enumerate(patient_list)}

    # Log timepoint distribution
    tp_counts = np.diff(patient_offsets)
    logger.info("Timepoint distribution:")
    for n_tp in sorted(set(tp_counts)):
        count = (tp_counts == n_tp).sum()
        logger.info(f"  {n_tp} tp: {count} patients")

    # Create H5 file
    logger.info(f"Creating H5 file: {output_path}")
    logger.info(f"  Scans: {n_scans}, Patients: {n_patients}")
    logger.info(f"  ROI size: {roi}")
    logger.info(f"  Compression: {compression} level={compression_level}")

    with h5py.File(output_path, "w") as f:
        # Global attributes
        f.attrs["n_scans"] = n_scans
        f.attrs["n_patients"] = n_patients
        f.attrs["roi_size"] = roi
        f.attrs["spacing"] = list(DEFAULT_SPACING)
        f.attrs["channel_order"] = list(MODALITY_KEYS)
        f.attrs["version"] = H5_VERSION
        f.attrs["dataset_type"] = "longitudinal"
        f.attrs["domain"] = "GLI"

        # Pre-allocate datasets
        images_ds = f.create_dataset(
            "images",
            shape=(n_scans, 4, *roi),
            dtype="float32",
            chunks=(1, 4, *roi),
            compression=compression,
            compression_opts=compression_level,
        )

        segs_ds = f.create_dataset(
            "segs",
            shape=(n_scans, 1, *roi),
            dtype="int8",
            chunks=(1, 1, *roi),
            compression=compression,
            compression_opts=compression_level,
        )

        # IDs
        dt = h5py.special_dtype(vlen=str)
        scan_id_list = [s["scan_id"] for s in all_scans]
        patient_id_list = [s["patient_id"] for s in all_scans]

        f.create_dataset("scan_ids", data=scan_id_list, dtype=dt)
        f.create_dataset("patient_ids", data=patient_id_list, dtype=dt)
        f.create_dataset("timepoint_idx", data=np.array(timepoint_indices, dtype=np.int32))

        # Longitudinal group
        long_grp = f.create_group("longitudinal")
        long_grp.create_dataset("patient_offsets", data=patient_offsets)
        long_grp.create_dataset("patient_list", data=patient_list, dtype=dt)

        # Metadata group (placeholder, no clinical metadata for GLI)
        meta_grp = f.create_group("metadata")
        meta_grp.create_dataset("grade", data=np.full(n_scans, -1, dtype=np.int8))
        meta_grp.create_dataset("age", data=np.full(n_scans, np.nan, dtype=np.float32))
        meta_grp.create_dataset("sex", data=["unknown"] * n_scans, dtype=dt)

        # Semantic features group
        sem_grp = f.create_group("semantic")
        vol_ds = sem_grp.create_dataset("volume", shape=(n_scans, 4), dtype="float32")
        loc_ds = sem_grp.create_dataset("location", shape=(n_scans, 3), dtype="float32")
        shape_ds = sem_grp.create_dataset("shape", shape=(n_scans, 3), dtype="float32")

        # Process each scan
        n_errors = 0
        for i, scan_info in enumerate(tqdm(all_scans, desc="Converting")):
            try:
                scan_id = scan_info["scan_id"]
                scan_dir = scan_info["scan_dir"]

                paths = load_scan_paths(scan_dir, scan_id)

                data = {
                    "t2f": str(paths["t2f"]),
                    "t1c": str(paths["t1c"]),
                    "t1n": str(paths["t1n"]),
                    "t2w": str(paths["t2w"]),
                    "seg": str(paths["seg"]),
                }
                result = transforms(data)

                image_np = result["image"].numpy()
                seg_np = result["seg"].numpy()

                assert image_np.shape == (4, *roi), f"Image shape mismatch: {image_np.shape}"
                assert seg_np.shape == (1, *roi), f"Seg shape mismatch: {seg_np.shape}"

                images_ds[i] = image_np
                segs_ds[i] = seg_np.astype(np.int8)

                # Semantic features (merge label 4 -> 1 for consistent features)
                seg_for_semantic = seg_np[0].astype(np.int32)
                sem = extract_semantic_features(
                    seg_for_semantic,
                    spacing=(1.0, 1.0, 1.0),
                    merge_rc_into_ncr=True,
                )
                vol_ds[i] = sem["volume"]
                loc_ds[i] = sem["location"]
                shape_ds[i] = sem["shape"]

            except Exception as e:
                logger.error(f"Failed to process {scan_info['scan_id']}: {e}")
                n_errors += 1
                continue

        # Splits group (patient-level indices into patient_list)
        splits_grp = f.create_group("splits")
        for split_name, pid_list in patient_splits.items():
            indices = np.array(
                [patient_to_idx[pid] for pid in pid_list], dtype=np.int32
            )
            splits_grp.create_dataset(split_name, data=indices)
            # Count scans in this split
            n_split_scans = sum(
                int(patient_offsets[idx + 1] - patient_offsets[idx]) for idx in indices
            )
            logger.info(
                f"  Split '{split_name}': {len(indices)} patients, "
                f"{n_split_scans} scans"
            )

    # Summary
    file_size_gb = output_path.stat().st_size / (1024**3)
    logger.info("\nConversion complete!")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  File size: {file_size_gb:.2f} GB")
    logger.info(f"  Scans: {n_scans} ({n_errors} errors)")
    logger.info(f"  Patients: {n_patients}")

    # Verification
    _verify_h5(output_path, n_scans, n_patients, roi)

    # Generate longitudinal report
    report_dir = output_path.parent / "gli_longitudinal_report"
    generate_longitudinal_report(output_path, report_dir)


def _verify_h5(
    h5_path: Path,
    n_scans: int,
    n_patients: int,
    roi: list[int],
) -> None:
    """Spot-check the H5 file for correctness."""
    logger.info("\nVerification:")

    with h5py.File(h5_path, "r") as f:
        # Check attributes
        assert f.attrs["n_scans"] == n_scans
        assert f.attrs["n_patients"] == n_patients
        assert list(f.attrs["roi_size"]) == roi
        assert list(f.attrs["channel_order"]) == list(MODALITY_KEYS)
        assert f.attrs["domain"] == "GLI"
        assert f.attrs["dataset_type"] == "longitudinal"

        # Check shapes
        assert f["images"].shape == (n_scans, 4, *roi)
        assert f["segs"].shape == (n_scans, 1, *roi)
        assert len(f["scan_ids"]) == n_scans
        assert len(f["patient_ids"]) == n_scans

        # Check longitudinal structure
        offsets = f["longitudinal/patient_offsets"][:]
        assert len(offsets) == n_patients + 1
        assert offsets[0] == 0
        assert offsets[-1] == n_scans

        # Check semantic shapes
        assert f["semantic/volume"].shape == (n_scans, 4)
        assert f["semantic/location"].shape == (n_scans, 3)
        assert f["semantic/shape"].shape == (n_scans, 3)

        # Spot-check 3 random scans
        rng = np.random.RandomState(42)
        check_indices = rng.choice(n_scans, min(3, n_scans), replace=False)

        for idx in check_indices:
            sid = f["scan_ids"][idx]
            if isinstance(sid, bytes):
                sid = sid.decode()

            img = f["images"][idx]
            seg = f["segs"][idx]

            assert not np.isnan(img).any(), f"NaN in image for {sid}"
            assert not np.isinf(img).any(), f"Inf in image for {sid}"
            assert img.max() > 0, f"All-zero image for {sid}"

            unique_labels = np.unique(seg)
            valid_labels = {0, 1, 2, 3, 4}
            assert set(unique_labels).issubset(valid_labels), (
                f"Invalid labels {unique_labels} for {sid}"
            )

            logger.info(
                f"  [{idx}] {sid}: "
                f"img range=[{img.min():.1f}, {img.max():.1f}], "
                f"seg labels={sorted(unique_labels)}"
            )

        # Check splits (patient-level)
        if "splits" in f:
            total_patients_in_splits = sum(len(f[f"splits/{s}"]) for s in f["splits"])
            logger.info(f"  Total patients in splits: {total_patients_in_splits}")

    logger.info("  Verification passed!")


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert BraTS-GLI NIfTI files to HDF5 with longitudinal hierarchy"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to BraTS-GLI scan directory (containing BraTS-GLI-* dirs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="brats_gli.h5",
        help="Output H5 file path",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Max patients for dry-run testing (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split generation",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level 1-9 (default: 4)",
    )

    args = parser.parse_args()

    convert(
        data_root=args.data_root,
        output_path=args.output,
        max_patients=args.max_patients,
        seed=args.seed,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    main()
