"""One-shot enrichment of the MenGrowth H5 with derived σ²_v signals.

Adds the four CSV-derived candidate signals (``logvol_var``,
``logvol_mad_var``, ``vol_cv2``, ``composite_logvol_x_boundary_entropy``)
as new datasets under ``/uncertainty/`` so the main experiment can
select any of them via ``uncertainty.signal`` in its config. All four
signals are derived directly from datasets already in the H5; the CSV
produced by ``test_candidate_uncertainty_signals/extract_candidates.py``
is used only as an optional cross-check.

Behaviour:
  * Pattern mirrors ``test_candidate_uncertainty_signals/patch_h5_uncertainty.py``:
    timestamped ``.bak`` backup before any write, ``r+`` open afterward.
  * Idempotent: datasets that already exist are skipped unless ``--force``
    is passed. When every targeted dataset is already present and matches
    the recomputed values, the H5 is left untouched and no backup is made.
  * If a CSV is provided via ``--csv``, the recomputed values are
    compared against the CSV row-by-row (tolerance 1e-5). A mismatch
    fails the run (the H5 has diverged from the CSV's source state).

Run (with backup)::

    python -m experiments.stage1_volumetric.main_experiment.enrich_h5_uncertainty \\
        --h5 /media/mpascual/MeningD2/MENINGIOMAS/MENGROWTH/050526/h5_format/MenGrowth.h5

Dry-run / force re-add / skip backup::

    ... --dry-run
    ... --force                     # overwrite existing datasets
    ... --no-backup                 # DANGEROUS — skip .bak snapshot
    ... --csv .../candidate_signals.csv   # cross-validate against CSV
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Each entry maps an output dataset name to a (formula string, builder fn).
# builder fn takes the open ``/uncertainty/`` group and returns a 1-D float64
# array of length n_scans. CSV column names match the registry keys in
# test_candidate_uncertainty_signals/modules/candidates.py.
def _build_logvol_var(uq: h5py.Group) -> np.ndarray:
    return np.asarray(uq["logvol_std"][:], dtype=np.float64) ** 2


def _build_logvol_mad_var(uq: h5py.Group) -> np.ndarray:
    return np.asarray(uq["logvol_mad_scaled"][:], dtype=np.float64) ** 2


def _build_vol_cv2(uq: h5py.Group) -> np.ndarray:
    vmean = np.asarray(uq["vol_mean"][:], dtype=np.float64)
    vstd = np.asarray(uq["vol_std"][:], dtype=np.float64)
    cv = vstd / np.maximum(vmean, 1.0)
    return cv**2


def _build_composite(uq: h5py.Group) -> np.ndarray:
    base = _build_logvol_var(uq)
    bh = np.nan_to_num(
        np.asarray(uq["men_boundary_entropy"][:], dtype=np.float64), nan=0.0
    )
    return base * (1.0 + bh)


DERIVED_SIGNALS: dict[str, tuple[str, Callable[[h5py.Group], np.ndarray]]] = {
    "logvol_var": ("logvol_std ** 2", _build_logvol_var),
    "logvol_mad_var": ("logvol_mad_scaled ** 2", _build_logvol_mad_var),
    "vol_cv2": ("(vol_std / max(vol_mean, 1)) ** 2", _build_vol_cv2),
    "composite_logvol_x_boundary_entropy": (
        "logvol_std**2 * (1 + men_boundary_entropy)",
        _build_composite,
    ),
}


def _required_inputs(uq: h5py.Group) -> None:
    needed = {
        "logvol_std",
        "logvol_mad_scaled",
        "vol_mean",
        "vol_std",
        "men_boundary_entropy",
    }
    missing = needed - set(uq.keys())
    if missing:
        raise RuntimeError(
            f"H5 /uncertainty/ missing required datasets for enrichment: {sorted(missing)}"
        )


def _validate_csv(
    df: pd.DataFrame, expected_keys: list[str], n_scans: int
) -> None:
    required = {"scan_idx_in_h5", *expected_keys}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    if df["scan_idx_in_h5"].duplicated().any():
        raise ValueError("CSV has duplicate scan_idx_in_h5 entries")
    out_of_range = df[(df["scan_idx_in_h5"] < 0) | (df["scan_idx_in_h5"] >= n_scans)]
    if len(out_of_range):
        raise ValueError(f"CSV has {len(out_of_range)} scan_idx_in_h5 out of range")


def _cross_check_csv(
    csv_df: pd.DataFrame, key: str, computed: np.ndarray, tol: float = 1e-5
) -> None:
    if key not in csv_df.columns:
        return
    aligned = (
        csv_df[["scan_idx_in_h5", key]]
        .dropna()
        .sort_values("scan_idx_in_h5")
        .reset_index(drop=True)
    )
    if aligned.empty:
        return
    idx = aligned["scan_idx_in_h5"].to_numpy(dtype=np.int64)
    csv_vals = aligned[key].to_numpy(dtype=np.float64)
    h5_vals = computed[idx]
    diff = np.abs(h5_vals - csv_vals)
    max_diff = float(diff.max())
    if max_diff > tol:
        bad = int(np.sum(diff > tol))
        raise RuntimeError(
            f"CSV/H5 cross-check failed for '{key}': "
            f"max |diff|={max_diff:.3e} > tol={tol:.0e}, {bad}/{len(diff)} rows mismatched"
        )
    logger.info("  CSV cross-check %s: max |diff| = %.3e (n=%d)", key, max_diff, len(diff))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path, help="Path to MenGrowth H5")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional candidate_signals.csv for cross-validation",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default=None,
        help="Comma-separated subset of signals to enrich (default: all derived)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing datasets instead of skipping them",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the .bak.<timestamp> snapshot (DANGEROUS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without modifying the H5",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.h5.exists():
        logger.error("H5 not found: %s", args.h5)
        return 2

    targets = list(DERIVED_SIGNALS.keys())
    if args.keys is not None:
        requested = [k.strip() for k in args.keys.split(",") if k.strip()]
        unknown = set(requested) - set(DERIVED_SIGNALS.keys())
        if unknown:
            logger.error("Unknown --keys entries: %s", sorted(unknown))
            return 2
        targets = requested

    csv_df: pd.DataFrame | None = None
    if args.csv is not None:
        if not args.csv.exists():
            logger.error("CSV not found: %s", args.csv)
            return 2
        csv_df = pd.read_csv(args.csv)

    with h5py.File(args.h5, "r") as f:
        if "n_scans" in f.attrs:
            n_scans = int(f.attrs["n_scans"])
        elif "images" in f:
            n_scans = int(f["images"].shape[0])
        elif "uncertainty" in f and "logvol_mean" in f["uncertainty"]:
            n_scans = int(f["uncertainty"]["logvol_mean"].shape[0])
        else:
            logger.error("Cannot infer n_scans from H5: %s", args.h5)
            return 2
        if "uncertainty" not in f:
            logger.error("H5 has no /uncertainty group: %s", args.h5)
            return 2
        uq = f["uncertainty"]
        _required_inputs(uq)
        if csv_df is not None:
            _validate_csv(csv_df, targets, n_scans)
        existing = {k: (k in uq) for k in targets}
        computed: dict[str, np.ndarray] = {}
        for key in targets:
            _formula, builder = DERIVED_SIGNALS[key]
            arr = builder(uq).astype(np.float64)
            if arr.shape != (n_scans,):
                raise RuntimeError(
                    f"Builder for '{key}' returned shape {arr.shape}, expected ({n_scans},)"
                )
            computed[key] = arr
            if csv_df is not None:
                _cross_check_csv(csv_df, key, arr)

    todo: list[str] = []
    for key, present in existing.items():
        if present and not args.force:
            logger.info("  /uncertainty/%-40s — already present, skipping", key)
        else:
            todo.append(key)
            status = "overwriting" if present else "adding"
            logger.info(
                "  /uncertainty/%-40s — %s (formula: %s)",
                key,
                status,
                DERIVED_SIGNALS[key][0],
            )

    if not todo:
        logger.info("Nothing to do. H5 already enriched with: %s", sorted(targets))
        return 0

    if args.dry_run:
        logger.info("Dry-run: would write %d dataset(s) and finish.", len(todo))
        return 0

    if not args.no_backup:
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        bak = args.h5.with_suffix(args.h5.suffix + f".bak.{ts}")
        logger.info("Backing up H5 to %s", bak)
        shutil.copy2(args.h5, bak)

    with h5py.File(args.h5, "r+") as f:
        uq = f["uncertainty"]
        for key in todo:
            arr = computed[key]
            formula, _ = DERIVED_SIGNALS[key]
            if key in uq and args.force:
                del uq[key]
            ds = uq.create_dataset(key, data=arr, dtype="float64")
            ds.attrs["source"] = "main_experiment/enrich_h5_uncertainty"
            ds.attrs["formula"] = formula
            ds.attrs["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")
            logger.info("  wrote /uncertainty/%s (n=%d)", key, arr.size)

    logger.info("H5 enrichment complete. Added/updated %d dataset(s).", len(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
