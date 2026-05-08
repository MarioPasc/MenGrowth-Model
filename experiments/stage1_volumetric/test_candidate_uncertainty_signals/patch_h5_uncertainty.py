"""Stage 0 — apply recomputed uncertainty scalars to the v5 H5.

Reads ``recomputed_uncertainty.csv`` and overwrites the targeted
``/uncertainty/<key>`` datasets. A timestamped backup of the H5 file is
created before any in-place modification.

Run with ``--dry-run`` first to inspect what would change.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PATCHED_KEYS: tuple[str, ...] = (
    "mean_entropy",
    "mean_mi",
    "men_mean_entropy",
    "men_mean_mi",
    "men_boundary_entropy",
    "men_boundary_mi",
)


def _validate_csv(df: pd.DataFrame, n_scans: int) -> None:
    required = {"scan_idx_in_h5", "patient_id", "timepoint_idx", *PATCHED_KEYS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    if df["scan_idx_in_h5"].duplicated().any():
        dups = df.loc[df["scan_idx_in_h5"].duplicated(), "scan_idx_in_h5"].tolist()
        raise ValueError(f"CSV has duplicate scan_idx_in_h5 entries: {dups}")

    out_of_range = df[(df["scan_idx_in_h5"] < 0) | (df["scan_idx_in_h5"] >= n_scans)]
    if len(out_of_range):
        raise ValueError(f"CSV has {len(out_of_range)} scan_idx_in_h5 out of range")

    nan_counts = {k: int(df[k].isna().sum()) for k in PATCHED_KEYS}
    if any(v > 0 for v in nan_counts.values()):
        logger.warning("CSV still contains NaN scalars: %s — those rows are skipped.", nan_counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", required=True, type=Path, help="recomputed_uncertainty.csv from Stage 0 array"
    )
    parser.add_argument("--h5", required=True, type=Path, help="Path to the v5 MenGrowth H5 file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report planned changes without modifying H5"
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip the .bak backup (DANGEROUS)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.csv.exists():
        logger.error("CSV not found: %s", args.csv)
        return 2
    if not args.h5.exists():
        logger.error("H5 not found: %s", args.h5)
        return 2

    df = pd.read_csv(args.csv)
    with h5py.File(args.h5, "r") as f:
        n_scans = int(f.attrs.get("n_scans", f["images"].shape[0]))
    _validate_csv(df, n_scans)

    df = df.sort_values("scan_idx_in_h5").reset_index(drop=True)
    logger.info(
        "CSV: %d rows, %d unique scans (cohort n_scans=%d)",
        len(df),
        df["scan_idx_in_h5"].nunique(),
        n_scans,
    )
    for k in PATCHED_KEYS:
        finite = df[k].dropna()
        if len(finite):
            logger.info(
                "  %s: %d finite, range [%.4e, %.4e], mean %.4e",
                k,
                len(finite),
                finite.min(),
                finite.max(),
                finite.mean(),
            )

    if args.dry_run:
        logger.info(
            "[dry-run] would overwrite %d datasets in %s for %d scans",
            len(PATCHED_KEYS),
            args.h5,
            len(df),
        )
        return 0

    if not args.no_backup:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = args.h5.with_suffix(args.h5.suffix + f".bak.{ts}")
        logger.info("Backing up %s → %s", args.h5, backup)
        shutil.copy2(args.h5, backup)

    with h5py.File(args.h5, "r+") as f:
        u = f["uncertainty"]
        for k in PATCHED_KEYS:
            if k not in u:
                logger.warning("uncertainty/%s not in H5 — skipping", k)
                continue
            existing = u[k][:].copy()
            new = existing.copy().astype(existing.dtype, copy=False)
            for _, row in df.iterrows():
                v = row[k]
                if pd.isna(v):
                    continue
                new[int(row["scan_idx_in_h5"])] = v
            n_changed = int(np.sum(~np.isclose(new, existing, equal_nan=True)))
            u[k][...] = new
            logger.info("  uncertainty/%s ← %d / %d entries changed", k, n_changed, n_scans)

    logger.info("H5 patch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
