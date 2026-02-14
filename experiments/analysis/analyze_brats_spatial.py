#!/usr/bin/env python3
"""Analyze physical dimensions and tumor locations in BraTS-MEN vs BraTS-GLI.

Uses FLAIR modality for CropForeground (brain bounding box), seg for tumor analysis.
Matches production pipeline: Orientationd -> Spacingd(1mm) -> CropForegroundd(k_divisible=128).

Usage:
    python experiments/analysis/analyze_brats_spatial.py

Output:
    - Console report comparing MEN vs GLI spatial properties
    - /tmp/brats_spatial_analysis.json with per-sample details
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
)

MEN_DIR = Path("/media/mpascual/PortableSSD/Meningiomas/BraTS/BraTS_Men_Train")
GLI_DIR = Path(
    "/media/mpascual/PortableSSD/BraTS_GLI/source/"
    "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2"
)

CROP_SIZES = [128, 160, 192, 224]
MAX_WORKERS = min(os.cpu_count() or 1, 8)


def find_files(sample_dir: Path) -> dict:
    """Find modality and seg files in a BraTS sample directory."""
    files = {}
    for f in sample_dir.glob("*.nii.gz"):
        name = f.name.lower()
        if "seg" in name:
            files["seg"] = str(f)
        elif "t2f" in name or "flair" in name:
            files["t2f"] = str(f)
        elif "t1c" in name:
            files["t1c"] = str(f)
        elif "t1n" in name:
            files["t1n"] = str(f)
        elif "t2w" in name:
            files["t2w"] = str(f)
    return files


def analyze_sample(sample_dir: Path, dataset_name: str) -> dict | None:
    """Analyze one sample: volume dims, tumor bbox, crop containment."""
    sample_id = sample_dir.name
    files = find_files(sample_dir)

    if "seg" not in files:
        return None

    # Find a modality for CropForeground (prefer t2f/FLAIR)
    source_mod = None
    for mod in ["t2f", "t1c", "t1n", "t2w"]:
        if mod in files:
            source_mod = mod
            break
    if source_mod is None:
        return None

    # Raw info from header only (no data load)
    nii = nib.load(files["seg"])
    raw_shape = nii.shape
    raw_spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    del nii

    # --- Single load pipeline: load → orient → resample → crop ---
    # (Original loaded from disk TWICE: once for resampled shape, once for crop)
    keys = [source_mod, "seg"]
    data = {source_mod: files[source_mod], "seg": files["seg"]}

    transforms = Compose(
        [
            LoadImaged(keys=keys, image_only=True),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=[source_mod], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Spacingd(keys=["seg"], pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        ]
    )
    result = transforms(data)
    shape_resampled = tuple(int(s) for s in result[source_mod].shape[1:])

    # CropForeground on already-loaded data (no second disk read)
    crop_transform = CropForegroundd(
        keys=keys,
        source_key=source_mod,
        k_divisible=[128, 128, 128],
    )
    result = crop_transform(result)

    # Extract seg array, free modality data immediately
    seg_cropped = result["seg"][0].numpy()
    shape_cropped = seg_cropped.shape
    del result

    # Tumor analysis
    tumor_mask = seg_cropped > 0
    del seg_cropped
    tumor_voxels = int(tumor_mask.sum())

    if tumor_voxels == 0:
        return {
            "id": sample_id,
            "dataset": dataset_name,
            "raw_shape": raw_shape,
            "raw_spacing": raw_spacing,
            "shape_resampled": shape_resampled,
            "shape_cropped": shape_cropped,
            "has_tumor": False,
        }

    # Tumor voxel coordinates — used for all subsequent analysis
    coords = np.argwhere(tumor_mask)
    del tumor_mask  # Free full-volume boolean array

    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_extent = tuple(int(x) for x in (bbox_max - bbox_min + 1))

    # Center of mass from coordinates (faster than scipy for binary masks)
    com = tuple(float(x) for x in coords.mean(axis=0))

    # Peripherality: distance from volume center to tumor COM
    vol_center = np.array(shape_cropped, dtype=np.float64) / 2.0
    tumor_com = np.array(com)
    distance = float(np.linalg.norm(tumor_com - vol_center))
    max_distance = float(np.linalg.norm(vol_center))
    peripherality = distance / max_distance if max_distance > 0 else 0.0

    # Containment: coordinate-based checks (no full-volume mask allocation)
    containment = {}
    shape_arr = np.array(shape_cropped)
    center = shape_arr // 2
    com_arr = np.array(com)

    for cs in CROP_SIZES:
        crop_start = np.maximum(center - cs // 2, 0)
        crop_end = np.minimum(crop_start + cs, shape_arr)

        bbox_contained = bool(
            (bbox_min >= crop_start).all() and (bbox_max < crop_end).all()
        )
        com_contained = bool(
            (com_arr >= crop_start).all() and (com_arr < crop_end).all()
        )

        # Vectorized coord check — replaces np.zeros_like(tumor_mask) per crop
        in_crop = (
            (coords[:, 0] >= crop_start[0])
            & (coords[:, 0] < crop_end[0])
            & (coords[:, 1] >= crop_start[1])
            & (coords[:, 1] < crop_end[1])
            & (coords[:, 2] >= crop_start[2])
            & (coords[:, 2] < crop_end[2])
        )
        voxels_in_crop = float(in_crop.sum() / tumor_voxels)

        containment[cs] = {
            "bbox_contained": bbox_contained,
            "com_contained": com_contained,
            "voxel_pct": voxels_in_crop * 100,
        }

    return {
        "id": sample_id,
        "dataset": dataset_name,
        "raw_shape": raw_shape,
        "raw_spacing": raw_spacing,
        "shape_resampled": shape_resampled,
        "shape_cropped": shape_cropped,
        "has_tumor": True,
        "tumor_voxels": tumor_voxels,
        "bbox_extent": bbox_extent,
        "com": com,
        "peripherality": peripherality,
        "containment": containment,
    }


def _worker(args: tuple) -> dict | None:
    """Top-level wrapper for ProcessPoolExecutor."""
    sample_dir, dataset_name = args
    try:
        return analyze_sample(sample_dir, dataset_name)
    except Exception as e:
        print(f"  ERROR {sample_dir.name}: {e}")
        return None


def summarize(results: list[dict], name: str) -> dict:
    """Compute summary stats."""
    valid = [r for r in results if r is not None and r.get("has_tumor", False)]
    n = len(valid)
    if n == 0:
        return {"n": 0}

    cropped = np.array([r["shape_cropped"] for r in valid])
    resampled = np.array([r["shape_resampled"] for r in valid])
    peripherality = np.array([r["peripherality"] for r in valid])
    tumor_vol = np.array([r["tumor_voxels"] for r in valid])
    bbox_ext = np.array([r["bbox_extent"] for r in valid])

    stats = {
        "n": n,
        "name": name,
        "resampled_mean": resampled.mean(0).tolist(),
        "resampled_min": resampled.min(0).tolist(),
        "resampled_max": resampled.max(0).tolist(),
        "cropped_mean": cropped.mean(0).tolist(),
        "cropped_min": cropped.min(0).tolist(),
        "cropped_max": cropped.max(0).tolist(),
        "peripherality_mean": float(peripherality.mean()),
        "peripherality_std": float(peripherality.std()),
        "tumor_vol_mean": float(tumor_vol.mean()),
        "tumor_vol_std": float(tumor_vol.std()),
        "bbox_extent_mean": bbox_ext.mean(0).tolist(),
        "bbox_extent_max": bbox_ext.max(0).tolist(),
    }

    for cs in CROP_SIZES:
        bbox_ok = sum(1 for r in valid if r["containment"][cs]["bbox_contained"])
        com_ok = sum(1 for r in valid if r["containment"][cs]["com_contained"])
        voxel_pcts = [r["containment"][cs]["voxel_pct"] for r in valid]
        stats[f"crop{cs}_bbox_pct"] = 100 * bbox_ok / n
        stats[f"crop{cs}_com_pct"] = 100 * com_ok / n
        stats[f"crop{cs}_voxel_mean"] = float(np.mean(voxel_pcts))
        stats[f"crop{cs}_voxel_min"] = float(np.min(voxel_pcts))

    return stats


def print_report(men: dict, gli: dict) -> None:
    """Print comparison report."""
    print("\n" + "=" * 90)
    print("BraTS SPATIAL ANALYSIS: MEN (meningioma) vs GLI (glioma)")
    print("=" * 90)

    print(f"\nSamples: MEN={men['n']}, GLI={gli['n']}")

    print(f"\n{'─' * 90}")
    print("VOLUME DIMENSIONS (after resampling to 1mm isotropic)")
    print(f"{'─' * 90}")
    for label, s in [("MEN", men), ("GLI", gli)]:
        m, mn, mx = s["resampled_mean"], s["resampled_min"], s["resampled_max"]
        print(
            f"  {label}: mean ({m[0]:.0f}, {m[1]:.0f}, {m[2]:.0f})  "
            f"min ({mn[0]}, {mn[1]}, {mn[2]})  max ({mx[0]}, {mx[1]}, {mx[2]})"
        )

    print(f"\n{'─' * 90}")
    print("AFTER CropForeground (brain bbox, k_divisible=128)")
    print(f"{'─' * 90}")
    for label, s in [("MEN", men), ("GLI", gli)]:
        m, mn, mx = s["cropped_mean"], s["cropped_min"], s["cropped_max"]
        print(
            f"  {label}: mean ({m[0]:.0f}, {m[1]:.0f}, {m[2]:.0f})  "
            f"min ({mn[0]}, {mn[1]}, {mn[2]})  max ({mx[0]}, {mx[1]}, {mx[2]})"
        )

    print(f"\n{'─' * 90}")
    print("TUMOR PROPERTIES")
    print(f"{'─' * 90}")
    for label, s in [("MEN", men), ("GLI", gli)]:
        print(
            f"  {label}: volume {s['tumor_vol_mean']:.0f}+/-{s['tumor_vol_std']:.0f} voxels, "
            f"bbox mean ({s['bbox_extent_mean'][0]:.0f}, {s['bbox_extent_mean'][1]:.0f}, "
            f"{s['bbox_extent_mean'][2]:.0f}), "
            f"max ({s['bbox_extent_max'][0]}, {s['bbox_extent_max'][1]}, "
            f"{s['bbox_extent_max'][2]})"
        )

    print(f"\n{'─' * 90}")
    print("PERIPHERALITY (0=center, 1=corner)")
    print(f"{'─' * 90}")
    for label, s in [("MEN", men), ("GLI", gli)]:
        print(f"  {label}: {s['peripherality_mean']:.3f} +/- {s['peripherality_std']:.3f}")

    print(f"\n{'─' * 90}")
    print("CENTER CROP TUMOR CONTAINMENT")
    print(f"{'─' * 90}")
    header = (
        f"  {'Size':<8} {'Dataset':<6} {'BBox 100%':<14} "
        f"{'COM in crop':<14} {'Voxel mean%':<14} {'Voxel min%':<14}"
    )
    print(header)
    print(f"  {'─' * 76}")
    for cs in CROP_SIZES:
        for label, s in [("MEN", men), ("GLI", gli)]:
            print(
                f"  {cs}^3{'':<4} {label:<6} "
                f"{s[f'crop{cs}_bbox_pct']:>6.1f}%{'':<6} "
                f"{s[f'crop{cs}_com_pct']:>6.1f}%{'':<6} "
                f"{s[f'crop{cs}_voxel_mean']:>6.1f}%{'':<6} "
                f"{s[f'crop{cs}_voxel_min']:>6.1f}%"
            )
        print()

    print("=" * 90)


def main() -> None:
    """Run spatial analysis on BraTS-MEN and BraTS-GLI datasets."""
    num_samples = 50
    t0 = time.time()

    men_dirs = [(d, "MEN") for d in sorted(MEN_DIR.iterdir()) if d.is_dir()][:num_samples]
    gli_dirs = [(d, "GLI") for d in sorted(GLI_DIR.iterdir()) if d.is_dir()][:num_samples]
    all_tasks = men_dirs + gli_dirs

    print(
        f"Processing {len(men_dirs)} MEN + {len(gli_dirs)} GLI samples "
        f"({MAX_WORKERS} workers)..."
    )

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_worker, task): task for task in all_tasks}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                r = future.result()
                if r is not None:
                    results.append(r)
                status = "ok" if r is not None else "skip"
            except Exception as e:
                status = f"ERROR: {e}"
            print(
                f"  [{i}/{len(all_tasks)}] {task[0].name} ({task[1]}) - {status}",
                flush=True,
            )

    men_results = [r for r in results if r["dataset"] == "MEN"]
    gli_results = [r for r in results if r["dataset"] == "GLI"]

    men_stats = summarize(men_results, "MEN")
    gli_stats = summarize(gli_results, "GLI")

    print_report(men_stats, gli_stats)

    # Save JSON
    output = {
        "men": men_stats,
        "gli": gli_stats,
        "men_raw": men_results,
        "gli_raw": gli_results,
    }
    with open("/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/brats_spatial_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print("Detailed results: /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/brats_spatial_analysis.json")


if __name__ == "__main__":
    main()
