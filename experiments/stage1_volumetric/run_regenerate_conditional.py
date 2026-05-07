"""Re-aggregate conditional calibration + per-tertile bootstrap from disk.

Reads ``<output_dir>/<model>/lopo_results.json`` for every model
sub-directory present, then regenerates
``conditional_calibration_*.{json,md}`` (now with per-tertile R²) and
``tertile_bootstrap_*.{json,md}`` (per-tertile paired-bootstrap CIs +
p-values for ΔR², ΔIS@95, Δcov_95).

The enabled pair list comes from a small YAML or, in the absence of a
config, defaults to the four canonical homo↔hetero pairs plus a
``LME ↔ LMEHetero_Zero`` and ``LMEHetero_Zero ↔ LMEHetero`` pair to
expose the implementation drift / clean propagation decomposition
introduced by ``run_lme_hetero_zero.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from experiments.stage1_volumetric.stats.conditional_calibration import (
    run_conditional_calibration,
)
from experiments.stage1_volumetric.stats.tertile_bootstrap import (
    run_tertile_bootstrap_for_pairs,
)
from growth.shared.lopo import LOPOResults

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = (
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction"
)

# Pairs probed for the per-tertile statistical tests. The first four are
# the canonical homo→hetero pairs; the last two add the LMEHetero_Zero
# ablation that isolates the genuine σ²_v propagation effect.
DEFAULT_PAIRS = [
    ["ScalarGP", "ScalarGPHetero"],
    ["LME", "LMEHetero"],
    ["HGP", "HGPHetero"],
    ["HGP_Gompertz", "HGP_Gompertz_Hetero"],
    ["LME", "LMEHetero_Zero"],
    ["LMEHetero_Zero", "LMEHetero"],
]


def _load_all_lopo(output_dir: Path) -> dict[str, LOPOResults]:
    out: dict[str, LOPOResults] = {}
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        cached = child / "lopo_results.json"
        if not cached.exists():
            continue
        try:
            with open(cached) as f:
                payload = json.load(f)
            out[child.name] = LOPOResults.from_dict(payload)
            logger.info("loaded %s", child.name)
        except Exception as exc:
            logger.warning("could not load %s: %s", cached, exc)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-model",
        default="LMEHetero",
        help="Model used to anchor σ²_v tertile cuts.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"output dir does not exist: {output_dir}")

    lopo_results = _load_all_lopo(output_dir)
    if not lopo_results:
        raise SystemExit("no lopo_results.json found under output_dir")
    logger.info("loaded %d models: %s", len(lopo_results), sorted(lopo_results.keys()))

    # 1. Conditional calibration (now includes per-tertile R²).
    run_conditional_calibration(
        lopo_results,
        output_dir,
        protocols=("last_from_rest", "all_from_first"),
        reference_model=args.reference_model,
    )

    # 2. Per-tertile paired bootstrap.
    pairs = [pair for pair in DEFAULT_PAIRS if pair[0] in lopo_results and pair[1] in lopo_results]
    skipped = [pair for pair in DEFAULT_PAIRS if pair not in pairs]
    if skipped:
        logger.warning("skipping pairs (missing models): %s", skipped)

    run_tertile_bootstrap_for_pairs(
        lopo_results,
        pairs,
        output_dir,
        protocol="last_from_rest",
        reference_model=args.reference_model,
        n_bootstrap=args.n_bootstrap,
        confidence_level=0.95,
        seed=args.seed,
        filename="tertile_bootstrap_last_from_rest",
    )


if __name__ == "__main__":
    main()
