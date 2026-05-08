"""Stage 2 aggregation: collect per-task LOPO outputs into one ranking table.

For each (candidate, scaling) cell:

* mean IS, sharpness, miss, cov95 (marginal + per tertile)
* ΔIS vs LME-homo (with paired BCa-style percentile bootstrap CI, B from cfg)
* BH-FDR-adjusted p-values across all tested cells

Reuses the patient-level paired bootstrap from
``main_experiment.modules.statistics`` so methods agree.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from experiments.stage1_volumetric.engine.data import load_config
from experiments.stage1_volumetric.main_experiment.modules.cohort import load_cohort

logger = logging.getLogger(__name__)


def _flatten_lopo(json_path: Path) -> pd.DataFrame:
    with open(json_path) as f:
        data = json.load(f)
    rows = []
    for fr in data["fold_results"]:
        pid = fr["patient_id"]
        for p in fr.get("predictions", {}).get("last_from_rest", []):
            mu, y = float(p["pred_mean"]), float(p["actual"])
            L, U = float(p["lower_95"]), float(p["upper_95"])
            width = U - L
            miss_lo = max(0.0, L - y)
            miss_hi = max(0.0, y - U)
            rows.append(
                {
                    "patient_id": pid,
                    "mu": mu,
                    "y": y,
                    "L": L,
                    "U": U,
                    "abs_resid": abs(mu - y),
                    "width": width,
                    "miss_lo": miss_lo,
                    "miss_hi": miss_hi,
                    "miss": (2.0 / 0.05) * (miss_lo + miss_hi),
                    "is_95": width + (2.0 / 0.05) * (miss_lo + miss_hi),
                    "covered_95": int((y >= L) and (y <= U)),
                    "sigma_v_sq_target": float(p.get("sigma_v_sq_target", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def _paired_delta_bootstrap(
    cell: pd.DataFrame,
    homo: pd.DataFrame,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Patient-level paired percentile bootstrap of mean(cell - homo)."""
    merged = cell[["patient_id", metric]].merge(
        homo[["patient_id", metric]], on="patient_id", suffixes=("_cell", "_homo")
    )
    if len(merged) < 3:
        return {
            "delta": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_value": float("nan"),
            "n_paired": int(len(merged)),
        }
    delta = merged[f"{metric}_cell"].to_numpy() - merged[f"{metric}_homo"].to_numpy()
    obs = float(delta.mean())

    rng = np.random.default_rng(seed)
    n = delta.size
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = delta[idx].mean()
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    # Two-sided p-value: 2 × min(P(boot ≤ 0), P(boot ≥ 0)).
    p = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    p = max(p, 1.0 / n_bootstrap)
    return {"delta": obs, "ci_lo": lo, "ci_hi": hi, "p_value": p, "n_paired": int(n)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    out_root = Path(cfg["paths"]["output_dir"])
    runs_dir = out_root / "runs"
    if not runs_dir.exists():
        logger.error("runs dir missing: %s", runs_dir)
        return 2

    homo_path_cfg = Path(cfg["paths"]["homo_baseline_lopo"])
    homo_sanity_path = runs_dir / "candidate_homo_sanity_scaling_raw" / "lopo_results.json"
    homo_path = homo_sanity_path if homo_sanity_path.exists() else homo_path_cfg
    if not homo_path.exists():
        logger.error("LME-homo baseline not found at %s nor %s", homo_sanity_path, homo_path_cfg)
        return 2
    homo_df = _flatten_lopo(homo_path)
    logger.info("Loaded LME-homo baseline (%d predictions) from %s", len(homo_df), homo_path)

    cohort = load_cohort(cfg)
    sv_emp = cohort.empirical_sigma_v_sq_flat
    cuts = (float(np.quantile(sv_emp, 1 / 3)), float(np.quantile(sv_emp, 2 / 3)))

    n_boot = int(cfg["statistics"]["bootstrap"]["n_samples"])
    seed = int(cfg["statistics"]["bootstrap"]["seed"])

    rows = []
    bootstrap_payload = []
    for cell_dir in sorted(runs_dir.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("candidate_"):
            continue
        lopo = cell_dir / "lopo_results.json"
        if not lopo.exists():
            logger.warning("missing %s", lopo)
            continue
        cell_df = _flatten_lopo(lopo)
        if cell_df.empty:
            continue

        # Marginal calibration battery
        mean_is = float(cell_df["is_95"].mean())
        mean_w = float(cell_df["width"].mean())
        mean_m = float(cell_df["miss"].mean())
        cov95 = float(cell_df["covered_95"].mean())

        # Tertile by held-out σ²_v
        tertile_metrics = {}
        for name, mask in (
            ("low", cell_df["sigma_v_sq_target"] <= cuts[0]),
            (
                "mid",
                (cell_df["sigma_v_sq_target"] > cuts[0])
                & (cell_df["sigma_v_sq_target"] <= cuts[1]),
            ),
            ("high", cell_df["sigma_v_sq_target"] > cuts[1]),
        ):
            sub = cell_df[mask]
            if sub.empty:
                tertile_metrics[name] = {"n": 0}
            else:
                tertile_metrics[name] = {
                    "n": int(len(sub)),
                    "is_95_mean": float(sub["is_95"].mean()),
                    "width_mean": float(sub["width"].mean()),
                    "miss_mean": float(sub["miss"].mean()),
                    "cov_95": float(sub["covered_95"].mean()),
                }

        # Paired bootstrap vs LME-homo
        delta_is = _paired_delta_bootstrap(cell_df, homo_df, "is_95", n_boot, seed)
        delta_w = _paired_delta_bootstrap(cell_df, homo_df, "width", n_boot, seed + 1)
        delta_m = _paired_delta_bootstrap(cell_df, homo_df, "miss", n_boot, seed + 2)

        rows.append(
            {
                "cell": cell_dir.name,
                "is_95_mean": mean_is,
                "width_mean": mean_w,
                "miss_mean": mean_m,
                "cov_95": cov95,
                "delta_is": delta_is["delta"],
                "delta_is_lo": delta_is["ci_lo"],
                "delta_is_hi": delta_is["ci_hi"],
                "p_delta_is": delta_is["p_value"],
                "delta_width": delta_w["delta"],
                "delta_miss": delta_m["delta"],
                "tertile_low_is": tertile_metrics["low"].get("is_95_mean", float("nan")),
                "tertile_mid_is": tertile_metrics["mid"].get("is_95_mean", float("nan")),
                "tertile_high_is": tertile_metrics["high"].get("is_95_mean", float("nan")),
            }
        )
        bootstrap_payload.append(
            {
                "cell": cell_dir.name,
                "delta_is": delta_is,
                "delta_width": delta_w,
                "delta_miss": delta_m,
                "tertile_metrics": tertile_metrics,
            }
        )

    table = pd.DataFrame(rows).sort_values("delta_is")
    pvals = table["p_delta_is"].to_numpy()
    valid = ~np.isnan(pvals)
    adj = np.full_like(pvals, np.nan, dtype=np.float64)
    if valid.any():
        clipped = np.clip(pvals[valid], 0.0, 1.0)
        adj[valid] = false_discovery_control(clipped, method="bh")
    table["p_delta_is_bh"] = adj

    out_dir = out_root / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "candidate_ranking.csv", index=False)
    with open(out_dir / "bootstrap_paired_BCa.json", "w") as f:
        json.dump(bootstrap_payload, f, indent=2)
    with open(out_dir / "bh_fdr_results.json", "w") as f:
        json.dump(
            {
                "q": cfg["statistics"].get("bh_fdr_q", 0.05),
                "n_tests": int(valid.sum()),
                "n_rejected": int(((adj < cfg["statistics"].get("bh_fdr_q", 0.05)) & valid).sum())
                if valid.any()
                else 0,
            },
            f,
            indent=2,
        )

    logger.info("Wrote candidate_ranking.csv (%d cells) → %s", len(table), out_dir)
    print("\nBest 5 cells by ΔIS (most negative = biggest improvement vs homo):")
    print(
        table.head(5)[
            ["cell", "is_95_mean", "delta_is", "delta_is_lo", "delta_is_hi", "p_delta_is"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
