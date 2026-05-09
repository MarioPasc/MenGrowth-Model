"""CLI driver for the Stage 1 information-content diagnostic.

Reads:
  * ``cfg.paths.candidate_signals_csv`` (from extract_candidates.py)
  * ``cfg.paths.homo_baseline_lopo`` (from main_experiment LME_baseline)

Writes:
  * ``stage1_diagnostic/homo_residuals.csv``
  * ``stage1_diagnostic/correlations.csv``
  * ``stage1_diagnostic/correlations.json`` (with metadata + bootstrap params)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from experiments.stage1_volumetric.engine.data import load_config

from ..modules.candidates import CANDIDATE_REGISTRY
from ..modules.diagnostic import correlate_all, join_with_candidates, load_homo_residuals

logger = logging.getLogger(__name__)


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
    out_root = Path(cfg["paths"]["output_dir"]) / "stage1_diagnostic"
    out_root.mkdir(parents=True, exist_ok=True)

    homo_path_cfg = Path(cfg["paths"]["homo_baseline_lopo"])
    homo_sanity_path = (
        Path(cfg["paths"]["output_dir"])
        / "runs"
        / "candidate_homo_sanity_scaling_raw"
        / "lopo_results.json"
    )
    homo_path = homo_path_cfg if homo_path_cfg.exists() else homo_sanity_path
    cand_path = Path(cfg["paths"]["candidate_signals_csv"])
    if not homo_path.exists():
        logger.error(
            "No homo baseline LOPO found. Configured: %s. Fallback (homo_sanity cell): %s",
            homo_path_cfg,
            homo_sanity_path,
        )
        return 2
    if homo_path != homo_path_cfg:
        logger.info("Using homo_sanity fallback: %s", homo_path)
    if not cand_path.exists():
        logger.error(
            "Candidate signals CSV not found: %s — run extract_candidates first", cand_path
        )
        return 2

    homo_df = load_homo_residuals(homo_path)
    cand_df = pd.read_csv(cand_path)
    logger.info("Loaded %d homo residuals, %d candidate rows", len(homo_df), len(cand_df))

    candidate_names = [
        c
        for c in cfg.get("diagnostic", {}).get("candidates", list(CANDIDATE_REGISTRY))
        if c in CANDIDATE_REGISTRY
    ]
    if not candidate_names:
        candidate_names = list(CANDIDATE_REGISTRY)
    logger.info("Diagnosing %d candidates: %s", len(candidate_names), candidate_names)

    joined = join_with_candidates(homo_df, cand_df, candidate_names)
    joined.to_csv(out_root / "homo_residuals.csv", index=False)

    n_boot = int(cfg["statistics"]["bootstrap"]["n_samples"])
    seed = int(cfg["statistics"]["bootstrap"]["seed"])
    table = correlate_all(joined, candidate_names, n_bootstrap=n_boot, seed=seed)

    table.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False, inplace=True)
    out_csv = out_root / "correlations.csv"
    table.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(table))

    summary = {
        "n_homo_predictions": int(len(joined)),
        "n_bootstrap": n_boot,
        "bootstrap_seed": seed,
        "candidates": candidate_names,
        "ranked_by_spearman": table[
            ["candidate", "spearman_rho", "spearman_lo", "spearman_hi", "r2_linear"]
        ].to_dict(orient="records"),
    }
    out_json = out_root / "correlations.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", out_json)

    print("\nTop 5 by |Spearman ρ|:")
    print(table.head(5).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
