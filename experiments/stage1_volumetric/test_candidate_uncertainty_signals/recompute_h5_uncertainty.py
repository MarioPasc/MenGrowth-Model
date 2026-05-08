"""Stage 0 — recompute the broken /uncertainty/ entropy + MI scalars.

The v5 H5 ``uncertainty/{mean,men_mean}_{entropy,mi}`` datasets are mostly
NaN (163/179 for entropy, 176/179 for MI). The per-member soft-prob NIfTI
maps are still on disk under
``r32_M20_s42/predictions/{patient_id}-{tp:03d}/`` (~1.7 GB per scan,
20 members × 192³ × 3-channel float32 + the ensemble mean), so we can
re-aggregate the scalars on CPU without re-running ensemble inference.

Per-voxel formulae (binary, applied to each of the C output channels):

    H[p]        = -p log(p+ε) - (1-p) log(1-p+ε)               (predictive entropy)
    mean_H_m    = (1/M) Σ_m H[p_m]                              (aleatoric proxy)
    MI(p)       = H[mean_p] - mean_H_m                          (BALD; clamp ≥ 0 only at scalar level)

ROI / MEN / boundary aggregation:
    ROI      = entire 192³ volume × 3 channels (matches existing convention).
    MEN      = ``probs[mean_p, channel 0] > 0.5``, CC-cleaned (≥ 64 voxels), all 3 channels.
    boundary = (dilated_MEN \\ MEN) ∩ ROI, all 3 channels.

This script produces one CSV row per scan (patient_id, timepoint_idx,
scan_idx_in_h5, mean_entropy, mean_mi, men_mean_entropy, men_mean_mi,
men_boundary_entropy, men_boundary_mi, mean_var_check), to be patched
into the H5 by ``patch_h5_uncertainty.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label

from experiments.stage1_volumetric.engine.data import load_config

logger = logging.getLogger(__name__)

EPS = 1e-8
MIN_CC_VOXELS = 64


# ---------------------------------------------------------------------------
# IO + scan-ID mapping
# ---------------------------------------------------------------------------


def _read_h5_keys(h5_path: Path) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as f:
        offsets = f["longitudinal"]["patient_offsets"][:].astype(int)
        plist = f["longitudinal"]["patient_list"][:]
        plist = [p.decode() if isinstance(p, bytes) else str(p) for p in plist]
        timepoint_idx = f["timepoint_idx"][:].astype(int)
        n_scans = int(f.attrs.get("n_scans", offsets[-1]))

    pid_per_scan: list[str] = []
    for i, pid in enumerate(plist):
        pid_per_scan.extend([pid] * (int(offsets[i + 1]) - int(offsets[i])))
    return pd.DataFrame(
        {
            "scan_idx_in_h5": np.arange(n_scans, dtype=int),
            "patient_id": pid_per_scan,
            "timepoint_idx": timepoint_idx,
        }
    )


def _scan_dirname(patient_id: str, timepoint_idx: int) -> str:
    """Reconstruct the prediction-folder name from H5 keys.

    Convention observed in r32_M20_s42/predictions/: ``{patient_id}-{tp:03d}``,
    e.g. ``MenGrowth-0001-000``.
    """
    return f"{patient_id}-{int(timepoint_idx):03d}"


def _load_prob_nii(path: Path) -> np.ndarray:
    """Load a NIfTI prob map; return float32 array shape ``[..., C]`` or ``[..., ]``."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj).astype(np.float32)
    return arr


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """H[p] = -p log(p+ε) - (1-p) log(1-p+ε), per voxel (any shape)."""
    p = np.clip(p, 0.0, 1.0)
    return -(p * np.log(p + EPS) + (1.0 - p) * np.log(1.0 - p + EPS))


def _men_mask_from_mean_p(mean_p: np.ndarray, channel: int = 0) -> np.ndarray:
    """probs[..., channel] > 0.5 with 26-connected CC cleanup ≥ MIN_CC_VOXELS."""
    if mean_p.ndim == 4:
        mask = mean_p[..., channel] > 0.5
    elif mean_p.ndim == 3:
        mask = mean_p > 0.5
    else:
        raise ValueError(f"mean_p has unexpected shape {mean_p.shape}")
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n = label(mask.astype(np.uint8), structure=structure)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[1:] = sizes[1:] >= MIN_CC_VOXELS
    return keep[labeled]


def _boundary_band(men_mask: np.ndarray, dilation_iter: int = 2) -> np.ndarray:
    """(dilated_MEN \\ MEN). Empty if MEN is empty."""
    if not men_mask.any():
        return np.zeros_like(men_mask, dtype=bool)
    dil = binary_dilation(men_mask, iterations=int(dilation_iter))
    return dil & (~men_mask)


def _ensure_volume_axes(arr: np.ndarray) -> np.ndarray:
    """Standardise to ``[D, H, W, C]`` (volumetric + channel)."""
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[..., None]
    raise ValueError(f"Unexpected array shape {arr.shape}")


def _safe_mean(arr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Average over voxels (and channels). 0 if mask empty."""
    if mask is not None:
        if not mask.any():
            return 0.0
        # Broadcast mask over channel axis
        if arr.ndim == 4 and mask.ndim == 3:
            mask = mask[..., None]
        return float(arr[np.broadcast_to(mask, arr.shape)].mean())
    return float(arr.mean())


def recompute_one_scan(scan_dir: Path, channel_for_men: int = 0) -> dict[str, float]:
    """Compute all corrected scalars for one scan from disk.

    Args:
        scan_dir: predictions/<scan_id>/.
        channel_for_men: prob channel to threshold for the MEN mask
            (0 = BraTS-TC = labels 1∣3 = MEN per project convention).

    Returns:
        Dict with the 6 corrected scalars + ``mean_var_check`` for sanity
        comparison against the existing ``uncertainty/mean_var`` value.
    """
    mean_p_path = scan_dir / "ensemble_probs.nii.gz"
    if not mean_p_path.exists():
        raise FileNotFoundError(f"missing ensemble_probs at {mean_p_path}")
    mean_p = _ensure_volume_axes(_load_prob_nii(mean_p_path))

    # Predictive entropy (single pass over mean_p)
    H_pred = _binary_entropy(mean_p)

    # MEN + boundary masks
    men_mask = _men_mask_from_mean_p(mean_p, channel=channel_for_men)
    boundary_mask = _boundary_band(men_mask, dilation_iter=2)

    # Iterate members, accumulate running mean of H[p_m] and Var_m[p_m]
    member_paths = sorted(scan_dir.glob("member_*_probs.nii.gz"))
    if not member_paths:
        raise FileNotFoundError(f"no member_*_probs.nii.gz under {scan_dir}")

    mean_H_member = np.zeros_like(H_pred, dtype=np.float64)
    sum_p = np.zeros_like(mean_p, dtype=np.float64)
    sum_p2 = np.zeros_like(mean_p, dtype=np.float64)
    M = 0
    for mp in member_paths:
        p_m = _ensure_volume_axes(_load_prob_nii(mp))
        if p_m.shape != mean_p.shape:
            raise ValueError(f"shape mismatch for {mp}: {p_m.shape} != {mean_p.shape}")
        mean_H_member += _binary_entropy(p_m)
        sum_p += p_m
        sum_p2 += p_m.astype(np.float64) ** 2
        M += 1

    mean_H_member /= float(M)
    var_member_p = sum_p2 / float(M) - (sum_p / float(M)) ** 2  # population variance per voxel

    # MI = H[mean_p] - mean_H_member ; clamp ≥ 0 only after aggregation.
    mi_per_voxel = H_pred - mean_H_member

    out = {
        "M": int(M),
        "mean_entropy": _safe_mean(H_pred),
        "mean_mi": float(max(0.0, _safe_mean(mi_per_voxel))),
        "men_mean_entropy": _safe_mean(H_pred, men_mask),
        "men_mean_mi": float(max(0.0, _safe_mean(mi_per_voxel, men_mask))),
        "men_boundary_entropy": _safe_mean(H_pred, boundary_mask),
        "men_boundary_mi": float(max(0.0, _safe_mean(mi_per_voxel, boundary_mask))),
        "mean_var_check": _safe_mean(var_member_p),
    }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _iter_scan_indices(arg: str | None, n_scans: int) -> Iterable[int]:
    if arg is None or arg == "all":
        return range(n_scans)
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    out = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--scan-indices",
        default=None,
        help="Comma-separated indices or ranges (e.g. '0-9,15'). Default: all scans.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Override cfg.stage0.recomputed_csv")
    parser.add_argument(
        "--append", action="store_true", help="Append to existing CSV (for SLURM-array merging)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    h5_path = Path(cfg["paths"]["mengrowth_h5"])
    pred_root = Path(cfg["stage0"]["predictions_root"])
    channel = int(cfg["stage0"].get("channel_index", 0))
    out_csv = Path(args.out) if args.out else Path(cfg["stage0"]["recomputed_csv"])

    keys_df = _read_h5_keys(h5_path)
    n_scans = len(keys_df)
    indices = list(_iter_scan_indices(args.scan_indices, n_scans))
    logger.info("Recomputing %d / %d scans from %s", len(indices), n_scans, pred_root)

    rows: list[dict] = []
    for k in indices:
        row = keys_df.iloc[k].to_dict()
        scan_dir = pred_root / _scan_dirname(row["patient_id"], row["timepoint_idx"])
        if not scan_dir.exists():
            logger.warning(
                "scan_idx=%d %s — predictions dir missing: %s", k, row["patient_id"], scan_dir
            )
            row.update(
                {
                    "M": 0,
                    "mean_entropy": np.nan,
                    "mean_mi": np.nan,
                    "men_mean_entropy": np.nan,
                    "men_mean_mi": np.nan,
                    "men_boundary_entropy": np.nan,
                    "men_boundary_mi": np.nan,
                    "mean_var_check": np.nan,
                }
            )
        else:
            try:
                metrics = recompute_one_scan(scan_dir, channel_for_men=channel)
                row.update(metrics)
                logger.info(
                    "  scan %3d %-22s tp=%d  ent=%.4e  mi=%.4e  men_ent=%.4e  bnd_ent=%.4e",
                    k,
                    row["patient_id"],
                    row["timepoint_idx"],
                    metrics["mean_entropy"],
                    metrics["mean_mi"],
                    metrics["men_mean_entropy"],
                    metrics["men_boundary_entropy"],
                )
            except Exception as exc:
                logger.error("scan_idx=%d %s — recompute failed: %s", k, row["patient_id"], exc)
                row.update({"M": 0, "error": str(exc)})
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.append and out_csv.exists():
        prior = pd.read_csv(out_csv)
        df = pd.concat([prior, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows → %s", len(df), out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
