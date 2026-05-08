"""Re-run the LoRA ensemble inference on the MenGrowth H5 to repair the broken
``/uncertainty/`` group.

The ensemble is M **independently trained** LoRA adapters on a shared frozen
BSF backbone (e.g. M=20 at rank=32). Per-member checkpoints live at
``<lora_run_dir>/adapters/member_{0..M-1}/{adapter, decoder.pt}``. The wrapped
:class:`EnsemblePredictor` loads each member once and runs a single forward
pass — not M stochastic / dropout passes through one checkpoint.

For each scan in the H5:

1. Reads the 4-channel image tensor (channel order ``["t2f", "t1c", "t1n", "t2w"]``).
2. Calls ``predict_scan(images, save_per_member=True)``.
3. Aggregates the per-voxel ``predictive_entropy`` + ``mutual_information``
   tensors on three masks:

   * full ROI (entire 192³ × 3-channel volume),
   * MEN region (``mean_probs[0] > 0.5`` after CC cleanup),
   * boundary band (dilated MEN minus MEN).

4. Saves per-member hard masks + ensemble mask to disk (cheap, ~32 KB / mask
   → ~110 MB total for the cohort) so alternative metrics
   (Hausdorff95, ASSD) can be computed later without re-inference.
5. Appends one row per scan to ``recomputed_uncertainty.csv`` with the schema
   ``patch_h5_uncertainty.py`` expects.

Sharded mode (``--shard K/N``) runs every Nth scan starting at K, useful for
SLURM-array parallelism (writes ``shards/shard_<K>.csv``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

# Channel order baked into BSF / the H5 — never reorder.
H5_CHANNEL_ORDER = ("t2f", "t1c", "t1n", "t2w")
EPS = 1e-8


# ---------------------------------------------------------------------------
# H5 keys + scan IDs
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
    if len(pid_per_scan) != n_scans:
        raise RuntimeError(
            f"patient_offsets imply {len(pid_per_scan)} scans, but n_scans={n_scans}"
        )
    return pd.DataFrame(
        {
            "scan_idx_in_h5": np.arange(n_scans, dtype=int),
            "patient_id": pid_per_scan,
            "timepoint_idx": timepoint_idx,
        }
    )


def _scan_dirname(patient_id: str, timepoint_idx: int) -> str:
    return f"{patient_id}-{int(timepoint_idx):03d}"


def _load_image(h5_path: Path, scan_idx: int) -> torch.Tensor:
    """Return [1, 4, D, H, W] float32 tensor for one H5 row."""
    with h5py.File(h5_path, "r") as f:
        arr = f["images"][scan_idx][...]  # [4, D, H, W] float32
    if arr.shape[0] != 4:
        raise RuntimeError(f"unexpected channel count {arr.shape[0]} (expected 4)")
    return torch.from_numpy(arr.astype(np.float32))[None]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _boundary_band(mask_3d: torch.Tensor, dilation_iter: int = 2) -> torch.Tensor:
    """(dilated_mask \\ mask). All on CPU as np.bool_, returned as torch.bool."""
    arr = mask_3d.cpu().numpy().astype(bool)
    if not arr.any():
        return torch.zeros_like(mask_3d, dtype=torch.bool)
    dil = binary_dilation(arr, iterations=int(dilation_iter))
    return torch.from_numpy(dil & (~arr))


def _safe_mean(tensor_chw: torch.Tensor, mask_3d: torch.Tensor | None) -> float:
    """Average a [C, D, H, W] tensor over voxels (and channels), optionally
    restricted to a [D, H, W] boolean mask. Returns 0.0 if the mask is empty.
    """
    if mask_3d is None:
        return float(tensor_chw.mean().item())
    if not bool(mask_3d.any().item()):
        return 0.0
    masked = tensor_chw[:, mask_3d]  # [C, n_voxels_in_mask]
    return float(masked.mean().item())


def aggregate_metrics(prediction, channel_for_men: int = 0) -> dict[str, float]:
    """Compute corrected scalars from one EnsemblePrediction."""
    pe = prediction.predictive_entropy  # [C, D, H, W]
    mi_voxel = prediction.mutual_information  # [C, D, H, W]
    var_p = prediction.var_probs  # [C, D, H, W]
    men_mask = prediction.ensemble_mask.bool()  # [D, H, W]
    boundary_mask = _boundary_band(men_mask, dilation_iter=2)

    out = {
        "M": int(prediction.n_members),
        "mean_entropy": _safe_mean(pe, None),
        "mean_mi": float(max(0.0, _safe_mean(mi_voxel, None))),
        "men_mean_entropy": _safe_mean(pe, men_mask),
        "men_mean_mi": float(max(0.0, _safe_mean(mi_voxel, men_mask))),
        "men_boundary_entropy": _safe_mean(pe, boundary_mask),
        "men_boundary_mi": float(max(0.0, _safe_mean(mi_voxel, boundary_mask))),
        "mean_var_check": _safe_mean(var_p, None),
        "logvol_mean_check": float(prediction.log_volume_mean),
        "logvol_std_check": float(prediction.log_volume_std),
        "vol_mean_check": float(prediction.volume_mean),
        "vol_std_check": float(prediction.volume_std),
    }
    return out


# ---------------------------------------------------------------------------
# Disk I/O for per-member masks
# ---------------------------------------------------------------------------


def _save_masks(
    pred,
    scan_dir: Path,
    *,
    save_member_masks: bool,
    save_ensemble_mask: bool,
) -> None:
    scan_dir.mkdir(parents=True, exist_ok=True)
    if save_ensemble_mask:
        nib.save(
            nib.Nifti1Image(pred.ensemble_mask.cpu().numpy().astype(np.uint8), affine=np.eye(4)),
            scan_dir / "ensemble_mask.nii.gz",
        )
    if save_member_masks and pred.per_member_masks is not None:
        for m, mask in enumerate(pred.per_member_masks):
            nib.save(
                nib.Nifti1Image(mask.cpu().numpy().astype(np.uint8), affine=np.eye(4)),
                scan_dir / f"member_{m}_mask.nii.gz",
            )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _iter_scan_indices(arg: str | None, n_scans: int, shard: tuple[int, int] | None) -> list[int]:
    if shard is not None:
        k, n = shard
        return [i for i in range(n_scans) if i % n == k]
    if arg is None or arg == "all":
        return list(range(n_scans))
    out: list[int] = []
    for p in (s.strip() for s in arg.split(",") if s.strip()):
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return out


def _parse_shard(arg: str | None) -> tuple[int, int] | None:
    if arg is None:
        return None
    k, n = arg.split("/", 1)
    k_i, n_i = int(k), int(n)
    if not (0 <= k_i < n_i):
        raise ValueError(f"--shard K/N requires 0 ≤ K < N (got {arg})")
    return k_i, n_i


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora-config",
        required=True,
        type=Path,
        help="LoRA-ensemble run config YAML. Must contain paths.checkpoint_dir / "
        "paths.checkpoint_filename and ensemble.n_members.",
    )
    parser.add_argument(
        "--lora-run-dir",
        required=True,
        type=Path,
        help="Directory containing adapters/member_*/. Typically the run output dir.",
    )
    parser.add_argument("--h5", required=True, type=Path, help="Path to the v5 MenGrowth H5.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output dir for recomputed_uncertainty.csv + per-scan masks.",
    )
    parser.add_argument(
        "--scan-indices", default=None, help="Subset (e.g. '0,1' or '0-9'). Default: all scans."
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="K/N: process scan_idx where (scan_idx % N == K). Used for SLURM array sharding.",
    )
    parser.add_argument("--device", default="cuda", help="cuda | cpu (cpu is for debugging only).")
    parser.add_argument(
        "--save-member-masks",
        action="store_true",
        help="Save per-member hard masks to per_scan/<id>/member_*_mask.nii.gz "
        "(~32KB/mask × M × N_scans).",
    )
    parser.add_argument(
        "--save-ensemble-mask",
        action="store_true",
        help="Save the ensemble hard mask to per_scan/<id>/ensemble_mask.nii.gz.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.h5.exists():
        logger.error("H5 not found: %s", args.h5)
        return 2
    if not args.lora_config.exists():
        logger.error("LoRA config not found: %s", args.lora_config)
        return 2
    if not args.lora_run_dir.exists():
        logger.error("LoRA run dir not found: %s", args.lora_run_dir)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_scan_root = args.output_dir / "per_scan"
    if args.save_member_masks or args.save_ensemble_mask:
        per_scan_root.mkdir(parents=True, exist_ok=True)

    # Lazy import to keep argparse cheap when --help'ing on a CPU-only box.
    from experiments.uncertainty_segmentation.engine.ensemble_inference import (
        EnsemblePredictor,
    )

    cfg = OmegaConf.load(args.lora_config)
    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and device != "cuda":
        logger.warning("CUDA unavailable; falling back to CPU (slow).")
    predictor = EnsemblePredictor(cfg, device=device, run_dir=args.lora_run_dir)
    logger.info(
        "EnsemblePredictor ready: M=%d members, run_dir=%s, device=%s",
        len(predictor.available_members),
        args.lora_run_dir,
        device,
    )

    keys_df = _read_h5_keys(args.h5)
    n_scans = len(keys_df)
    shard = _parse_shard(args.shard)
    indices = _iter_scan_indices(args.scan_indices, n_scans, shard)
    logger.info("Re-inferring %d / %d scans (shard=%s)", len(indices), n_scans, shard)

    if shard is not None:
        out_csv = args.output_dir / "shards" / f"shard_{shard[0]:03d}_of_{shard[1]:03d}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_csv = args.output_dir / "recomputed_uncertainty.csv"

    rows: list[dict] = []
    t_start = time.time()
    for k in indices:
        meta = keys_df.iloc[k].to_dict()
        scan_id = _scan_dirname(meta["patient_id"], meta["timepoint_idx"])
        scan_dir = per_scan_root / scan_id
        try:
            images = _load_image(args.h5, k)
            t0 = time.time()
            pred = predictor.predict_scan(images, save_per_member=bool(args.save_member_masks))
            metrics = aggregate_metrics(pred)
            metrics["inference_time_sec"] = float(time.time() - t0)
            metrics["per_member_volumes"] = list(pred.per_member_volumes)

            if args.save_member_masks or args.save_ensemble_mask:
                _save_masks(
                    pred,
                    scan_dir,
                    save_member_masks=args.save_member_masks,
                    save_ensemble_mask=args.save_ensemble_mask,
                )
                with open(scan_dir / "metrics.json", "w") as f:
                    json.dump({**meta, **metrics, "scan_id": scan_id}, f, indent=2)

            row = {**meta, **{k: v for k, v in metrics.items() if k != "per_member_volumes"}}
            rows.append(row)

            elapsed = time.time() - t_start
            logger.info(
                "  scan %3d %-22s tp=%d  ent=%.4e  mi=%.4e  men_ent=%.4e  bnd_ent=%.4e  "
                "M=%d  per-scan=%.1fs  total=%.0fs",
                k,
                meta["patient_id"],
                meta["timepoint_idx"],
                metrics["mean_entropy"],
                metrics["mean_mi"],
                metrics["men_mean_entropy"],
                metrics["men_boundary_entropy"],
                metrics["M"],
                metrics["inference_time_sec"],
                elapsed,
            )
        except Exception as exc:
            logger.exception(
                "scan_idx=%d %s — re-inference FAILED: %s", k, meta.get("patient_id", "?"), exc
            )
            rows.append({**meta, "error": str(exc)})
        # checkpoint after every scan so a crash mid-run doesn't lose progress
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(
        "Wrote %d rows to %s (total %.1f min)", len(rows), out_csv, (time.time() - t_start) / 60.0
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
