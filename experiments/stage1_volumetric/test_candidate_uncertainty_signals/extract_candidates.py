"""Extract per-scan candidate uncertainty signals from the v5 H5.

Reads ``/uncertainty/*`` and ``/longitudinal/*`` from
``cfg.paths.mengrowth_h5`` and writes a single CSV with one row per scan
containing the join keys (patient_id, timepoint_idx, scan_idx_in_h5) and
all candidate columns. The runner reads this CSV to assemble the per-scan
σ²_v vector for a given (candidate, scaling) pair.

This is a one-shot extraction. Run it once locally, then scp the CSV to
Picasso for the diagnostic sweep.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from experiments.stage1_volumetric.engine.data import load_config

from .modules.candidates import CANDIDATE_REGISTRY, candidate_value_per_scan

logger = logging.getLogger(__name__)


def _read_uncertainty_group(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        u = f["uncertainty"]
        out: dict[str, np.ndarray] = {}
        for k in u.keys():
            ds = u[k]
            # Keep 1-D scalars; drop higher-rank like per_member_volumes.
            if ds.ndim == 1:
                out[k] = ds[:].astype(np.float64)
        return out


def _read_join_keys(h5_path: Path) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as f:
        n_scans = int(f.attrs.get("n_scans", f["images"].shape[0]))
        offsets = f["longitudinal"]["patient_offsets"][:].astype(int)
        plist = f["longitudinal"]["patient_list"][:]
        plist = [p.decode() if isinstance(p, bytes) else str(p) for p in plist]
        timepoint_idx = f["timepoint_idx"][:].astype(int)

    pid_per_scan: list[str] = []
    for i, pid in enumerate(plist):
        i0, i1 = int(offsets[i]), int(offsets[i + 1])
        pid_per_scan.extend([pid] * (i1 - i0))
    if len(pid_per_scan) != n_scans:
        raise RuntimeError(f"patient_offsets imply {len(pid_per_scan)} scans but n_scans={n_scans}")

    return pd.DataFrame(
        {
            "scan_idx_in_h5": np.arange(n_scans, dtype=int),
            "patient_id": pid_per_scan,
            "timepoint_idx": timepoint_idx,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path, default=None, help="Override cfg.paths.candidate_signals_csv"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    h5_path = Path(cfg["paths"]["mengrowth_h5"])
    out_csv = Path(args.out) if args.out else Path(cfg["paths"]["candidate_signals_csv"])

    logger.info("Reading H5: %s", h5_path)
    if not h5_path.exists():
        logger.error("H5 not found: %s", h5_path)
        return 2

    uncertainty = _read_uncertainty_group(h5_path)
    keys_df = _read_join_keys(h5_path)
    n_scans = len(keys_df)
    logger.info("n_scans=%d  uncertainty datasets=%d", n_scans, len(uncertainty))

    df = keys_df.copy()
    # Always include the y target (for downstream join with cohort)
    df["logvol_mean"] = uncertainty["logvol_mean"]

    nan_summary: dict[str, int] = {}
    for name, spec in CANDIDATE_REGISTRY.items():
        try:
            vec = candidate_value_per_scan(spec, uncertainty)
            if vec.shape != (n_scans,):
                raise ValueError(f"{name}: shape {vec.shape} != ({n_scans},)")
        except Exception as exc:
            logger.warning("Candidate %s could not be built: %s", name, exc)
            vec = np.full(n_scans, np.nan, dtype=np.float64)
        df[name] = vec
        nan_summary[name] = int(np.isnan(vec).sum())

    # Empirical σ²_v reference (for mean_matched scaling and as a sanity column).
    df["empirical_sigma_v_sq"] = df["logvol_var"].fillna(0.0)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows × %d cols)", out_csv, len(df), df.shape[1])
    for name, n_nan in nan_summary.items():
        if n_nan:
            logger.warning(
                "  %s: %d / %d NaN — candidate may need Stage-0 repair", name, n_nan, n_scans
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
