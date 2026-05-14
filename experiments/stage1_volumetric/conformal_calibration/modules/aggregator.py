"""Walk the runs/ tree and assemble a long-form table of per-layer metrics.

Output columns (long-form):

    base_model, layer, seed, scope, tertile, metric, value

``scope`` ∈ {"marginal", "tertile"}; ``tertile`` ∈ {"all", "low", "mid", "high"};
``metric`` is one of the calibration-battery keys (r2_log, is_95, coverage_95, ...).

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
    "is_95",
    "coverage_95",
    "coverage_95_ci_low",
    "coverage_95_ci_high",
    "mean_width",
    "crps",
    "cov_50",
    "cov_80",
    "cov_90",
    "cov_95_base",
    "sigma_v_sq_mean",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _emit_marginal_rows(
    base_model: str,
    layer: str,
    seed: int,
    marginal: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in _METRIC_KEYS:
        if k in marginal:
            rows.append(
                {
                    "base_model": base_model,
                    "layer": layer,
                    "seed": seed,
                    "scope": "marginal",
                    "tertile": "all",
                    "metric": k,
                    "value": float(marginal[k]) if marginal[k] is not None else np.nan,
                }
            )
    # Also capture top-level r2_log that lives outside the per-layer dict.
    if "r2_log" not in marginal and "r2_log" in marginal:
        pass  # already handled above
    return rows


def _emit_tertile_rows(
    base_model: str,
    layer: str,
    seed: int,
    tertile_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strata_by_layer = tertile_payload.get("strata_by_layer", {})
    strata = strata_by_layer.get(layer, {})
    for tname in ("low", "mid", "high"):
        battery = strata.get(tname, {})
        for k in _METRIC_KEYS:
            if k in battery:
                rows.append(
                    {
                        "base_model": base_model,
                        "layer": layer,
                        "seed": seed,
                        "scope": "tertile",
                        "tertile": tname,
                        "metric": k,
                        "value": float(battery[k]) if battery[k] is not None else np.nan,
                    }
                )
    return rows


def collect_runs(output_root: Path) -> pd.DataFrame:
    """Walk ``runs/{base_model}/seed_{NNN}/`` and build the long-form metric table.

    Args:
        output_root: Root output directory of the conformal calibration experiment.

    Returns:
        Long-form :class:`pandas.DataFrame` with columns
        ``[base_model, layer, seed, scope, tertile, metric, value]``.
    """
    rows: list[dict[str, Any]] = []
    runs_dir = output_root / "runs"

    if not runs_dir.exists():
        logger.warning("runs/ directory not found under %s", output_root)
        return pd.DataFrame(rows)

    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        base_model = model_dir.name

        for seed_dir in sorted(model_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed = int(seed_dir.name.split("_")[1])
            except (ValueError, IndexError):
                seed = -1

            marginal_path = seed_dir / "marginal_metrics.json"
            tertile_path = seed_dir / "tertile_metrics.json"

            marginal = _read_json(marginal_path)
            tertile = _read_json(tertile_path)

            if marginal is None:
                continue

            # marginal_metrics.json has {layer: {metric: value, ...}, r2_log: float}
            # Emit per-layer rows.
            for layer, layer_metrics in marginal.items():
                if not isinstance(layer_metrics, dict):
                    # top-level r2_log key — skip; it is inside each layer dict too
                    continue
                rows.extend(_emit_marginal_rows(base_model, layer, seed, layer_metrics))
                if tertile is not None:
                    rows.extend(_emit_tertile_rows(base_model, layer, seed, tertile))

            # Emit the top-level r2_log as a "base" layer metric for convenience.
            if "r2_log" in marginal and isinstance(marginal["r2_log"], (int, float)):
                rows.append(
                    {
                        "base_model": base_model,
                        "layer": "parametric",
                        "seed": seed,
                        "scope": "marginal",
                        "tertile": "all",
                        "metric": "r2_log_overall",
                        "value": float(marginal["r2_log"]),
                    }
                )

    df = pd.DataFrame(rows)
    logger.info("Aggregator: %d rows from %s", len(df), output_root)
    return df


def write_table(df: pd.DataFrame, output_root: Path) -> Path:
    """Persist the long-form table; prefer parquet, fall back to CSV.

    Args:
        df: Long-form metric table.
        output_root: Root output directory.

    Returns:
        Path to the written file.
    """
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
