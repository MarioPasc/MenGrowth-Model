"""Walk the runs/ tree and assemble a long-form table of cell metrics.

Output columns (long-form):

    family, level, level_value, seed, scope, tertile, metric, value

`scope` ∈ {"marginal", "tertile"}; `tertile` ∈ {"all", "low", "mid", "high"};
`metric` is one of the calibration-battery keys (r2_log, cov_95, is_95, ...).

The aggregator is intentionally agnostic to the parquet engine: it falls back
to CSV if pyarrow / fastparquet are unavailable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_METRIC_KEYS = (
    "n",
    "r2_log",
    "ci_width_mean",
    "cov_50",
    "cov_80",
    "cov_90",
    "cov_95",
    "crps",
    "is_95",
    "nlpd",
    "dss",
    "pred_var_mean",
    "pit_ks_stat",
    "pit_ks_p",
    "sigma_v_sq_mean",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _emit_marginal_rows(
    family: str,
    level: str,
    level_value: float,
    seed: int,
    marginal: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in _METRIC_KEYS:
        if k in marginal:
            rows.append(
                {
                    "family": family,
                    "level": level,
                    "level_value": level_value,
                    "seed": seed,
                    "scope": "marginal",
                    "tertile": "all",
                    "metric": k,
                    "value": float(marginal[k]) if marginal[k] is not None else np.nan,
                }
            )
    return rows


def _emit_tertile_rows(
    family: str,
    level: str,
    level_value: float,
    seed: int,
    tertile_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strata = tertile_payload.get("strata", {})
    for tname in ("low", "mid", "high"):
        battery = strata.get(tname, {})
        for k in _METRIC_KEYS:
            if k in battery:
                rows.append(
                    {
                        "family": family,
                        "level": level,
                        "level_value": level_value,
                        "seed": seed,
                        "scope": "tertile",
                        "tertile": tname,
                        "metric": k,
                        "value": float(battery[k]) if battery[k] is not None else np.nan,
                    }
                )
    return rows


def collect_runs(output_root: Path) -> pd.DataFrame:
    """Walk runs/, baseline/, build the long-form metric table."""
    rows: list[dict[str, Any]] = []
    runs_dir = output_root / "runs"

    if runs_dir.exists():
        for cell_dir in sorted(runs_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            family, _, level = cell_dir.name.partition("_")
            # cell name is "{family}_{level}" but family may contain '_'
            # so split on first known prefix
            for known_family in ("empirical_shift", "beta_alpha"):
                if cell_dir.name.startswith(known_family + "_"):
                    family = known_family
                    level = cell_dir.name[len(known_family) + 1 :]
                    break

            try:
                if level.startswith("tau_"):
                    level_value = float(level[len("tau_") :])
                elif level.startswith("alpha_"):
                    level_value = float(level[len("alpha_") :])
                else:
                    level_value = float("nan")
            except ValueError:
                level_value = float("nan")

            for seed_dir in sorted(cell_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                seed = int(seed_dir.name.split("_")[1])

                marginal = _read_json(seed_dir / "marginal_metrics.json")
                tertile = _read_json(seed_dir / "tertile_metrics.json")

                if marginal:
                    rows.extend(_emit_marginal_rows(family, level, level_value, seed, marginal))
                if tertile:
                    rows.extend(_emit_tertile_rows(family, level, level_value, seed, tertile))

    # Baselines
    for base_name in ("LME_baseline", "LMEHetero_Zero_baseline"):
        bdir = output_root / base_name
        marginal = _read_json(bdir / "marginal_metrics.json")
        tertile = _read_json(bdir / "tertile_metrics.json")
        if marginal:
            rows.extend(
                _emit_marginal_rows(
                    family="baseline",
                    level=base_name,
                    level_value=float("nan"),
                    seed=-1,
                    marginal=marginal,
                )
            )
        if tertile:
            rows.extend(
                _emit_tertile_rows(
                    family="baseline",
                    level=base_name,
                    level_value=float("nan"),
                    seed=-1,
                    tertile_payload=tertile,
                )
            )

    df = pd.DataFrame(rows)
    logger.info("Aggregator: %d rows from %s", len(df), output_root)
    return df


def write_table(df: pd.DataFrame, output_root: Path) -> Path:
    """Persist the long-form table; prefer parquet, fall back to CSV."""
    agg_dir = output_root / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = agg_dir / "results_table.parquet"
    csv_path = agg_dir / "results_table.csv"
    try:
        df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s (%d rows)", parquet_path, len(df))
        return parquet_path
    except Exception as exc:  # pragma: no cover - depends on optional engine
        logger.warning("Parquet write failed (%s); falling back to CSV", exc)
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %s (%d rows)", csv_path, len(df))
        return csv_path
