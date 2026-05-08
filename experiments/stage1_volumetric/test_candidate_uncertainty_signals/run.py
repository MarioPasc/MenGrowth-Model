"""CLI entry point for the candidate-uncertainty-signal Stage 2 sweep.

Run modes
---------

Per-task (SLURM array)::

    python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.run \\
        --config <CONFIG.yaml> --task-index <K>

Smoke (one-shot, all tasks sequentially, optionally subset folds)::

    python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.run \\
        --config <CONFIG.yaml> --smoke --max-folds 2

Manifest only::

    python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.run \\
        --config <CONFIG.yaml> --write-manifest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from experiments.stage1_volumetric.engine.data import load_config
from experiments.stage1_volumetric.main_experiment.modules.cohort import (
    load_cohort,
    write_cohort_meta,
)

from .modules.runner import (
    DiagnosticManifest,
    run_task,
    subset_cohort,
)

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_candidates(cfg: dict) -> pd.DataFrame | None:
    csv_path = Path(cfg["paths"]["candidate_signals_csv"])
    if not csv_path.exists():
        logger.warning(
            "candidate_signals_csv not found: %s — only controls + homo_sanity will run",
            csv_path,
        )
        return None
    return pd.read_csv(csv_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument(
        "--max-folds", type=int, default=None, help="Subset to first N patients (smoke)."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    cfg = load_config(args.config)
    output_root = Path(cfg["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(cfg)
    cohort = subset_cohort(cohort, args.max_folds)
    write_cohort_meta(cohort, output_root / "cohort_meta.json")

    candidates_df = _load_candidates(cfg)

    manifest = DiagnosticManifest.from_cfg(cfg)
    manifest_path = output_root / "manifest.json"

    if args.write_manifest:
        manifest.to_json(manifest_path)
        print(f"Wrote {len(manifest.tasks)} tasks to {manifest_path}")
        return 0

    if not manifest_path.exists():
        manifest.to_json(manifest_path)

    if args.smoke:
        for i, t in enumerate(manifest.tasks):
            logger.info("[%d/%d] %s", i + 1, len(manifest.tasks), t.cell_dirname)
            run_task(t, cohort, candidates_df, cfg, output_root, force=args.force)
        return 0

    if args.task_index is None:
        parser.error("must pass --task-index, --smoke, or --write-manifest")

    if args.task_index < 0 or args.task_index >= len(manifest.tasks):
        raise IndexError(
            f"task-index {args.task_index} out of range (n_tasks={len(manifest.tasks)})"
        )
    spec = manifest.tasks[args.task_index]
    run_task(spec, cohort, candidates_df, cfg, output_root, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
