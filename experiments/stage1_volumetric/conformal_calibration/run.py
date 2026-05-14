"""CLI entry point for the conformal calibration experiment.

Run modes
---------

Per-task (SLURM array)::

    python -m experiments.stage1_volumetric.conformal_calibration.run \\
        --config <CONFIG.yaml> --task-index <K>

Write manifest only::

    python -m experiments.stage1_volumetric.conformal_calibration.run \\
        --config <CONFIG.yaml> --write-manifest

Smoke (all tasks sequentially, then analyze)::

    python -m experiments.stage1_volumetric.conformal_calibration.run \\
        --config <CONFIG.yaml> --smoke

Analysis (after all array tasks finished)::

    python -m experiments.stage1_volumetric.conformal_calibration.run \\
        --config <CONFIG.yaml> --analyze
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.stage1_volumetric.engine.data import load_config

from .modules import aggregator, figures, statistics
from .modules.cohort import Cohort, load_cohort, write_cohort_meta
from .modules.runner import TaskSpec, iter_task_specs, run_task


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskEntry:
    """One executable unit for the SLURM array."""

    kind: str  # "task"
    spec: dict[str, Any] | None = None  # TaskSpec fields


def build_manifest(cfg: dict) -> list[TaskEntry]:
    """Build the full task manifest from config.

    Args:
        cfg: Full experiment config dict.

    Returns:
        List of :class:`TaskEntry` objects in deterministic order.
    """
    tasks: list[TaskEntry] = []
    for spec in iter_task_specs(cfg):
        tasks.append(
            TaskEntry(kind="task", spec={"base_model": spec.base_model, "seed": spec.seed})
        )
    return tasks


def write_manifest(tasks: list[TaskEntry], path: Path) -> None:
    """Serialise the manifest as JSON.

    Args:
        tasks: List of task entries.
        path: Output JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([{"kind": t.kind, "spec": t.spec} for t in tasks], f, indent=2)


def read_manifest(path: Path) -> list[TaskEntry]:
    """Deserialise a manifest from JSON.

    Args:
        path: Path to manifest JSON.

    Returns:
        List of :class:`TaskEntry`.
    """
    with open(path) as f:
        data = json.load(f)
    return [TaskEntry(kind=d["kind"], spec=d.get("spec")) for d in data]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _prepare(cfg: dict) -> tuple[Cohort, Path]:
    """Load cohort and create output directory.

    Args:
        cfg: Full experiment config dict.

    Returns:
        Tuple of (cohort, output_root).
    """
    output_root = Path(cfg["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(cfg)
    write_cohort_meta(cohort, output_root / "cohort_meta.json")
    return cohort, output_root


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------


def execute_task(
    task: TaskEntry, cohort: Cohort, cfg: dict, output_root: Path, force: bool = False
) -> dict[str, Any]:
    """Execute one manifest task.

    Args:
        task: Task to execute.
        cohort: Pre-loaded cohort.
        cfg: Full experiment config dict.
        output_root: Root output directory.
        force: Re-run even if cached results exist.

    Returns:
        Dict with output paths and metrics.

    Raises:
        ValueError: If task kind is not recognised.
    """
    if task.kind == "task":
        spec = TaskSpec(**task.spec)
        return run_task(spec, cohort, cfg, output_root, force=force)
    raise ValueError(f"Unknown task kind: {task.kind!r}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def run_analysis(cfg: dict) -> None:
    """Aggregate results, compute statistics, and produce figures.

    Args:
        cfg: Full experiment config dict.
    """
    cohort, output_root = _prepare(cfg)

    df = aggregator.collect_runs(output_root)
    aggregator.write_table(df, output_root)

    # Load per-task LOPO JSON files keyed by "{base_model}/seed_{NNN}".
    results_by_task: dict[str, dict] = {}
    runs_dir = output_root / "runs"
    if runs_dir.exists():
        for model_dir in sorted(runs_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for seed_dir in sorted(model_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                lopo_path = seed_dir / "conformal_lopo_results.json"
                if not lopo_path.exists():
                    continue
                key = f"{model_dir.name}/{seed_dir.name}"
                with open(lopo_path) as fh:
                    results_by_task[key] = json.load(fh)

    if results_by_task:
        stat_payload = statistics.run_statistics(
            results_by_task,
            cohort.sigma_v_sq_flat,
            cfg,
        )
        statistics.write_results(
            stat_payload,
            output_root / "aggregated" / "statistics.json",
        )

    figures.make_all_figures(df, output_root, cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(description="Conformal calibration experiment runner.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    parser.add_argument("--task-index", type=int, default=None, help="Array task index.")
    parser.add_argument("--smoke", action="store_true", help="Run all tasks sequentially.")
    parser.add_argument("--analyze", action="store_true", help="Run analysis phase only.")
    parser.add_argument("--write-manifest", action="store_true", help="Write manifest and exit.")
    parser.add_argument("--force", action="store_true", help="Overwrite cached results.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    cfg = load_config(args.config)

    cohort, output_root = _prepare(cfg)
    manifest_path = output_root / "manifest.json"

    if args.write_manifest:
        tasks = build_manifest(cfg)
        write_manifest(tasks, manifest_path)
        print(f"Wrote {len(tasks)} tasks to {manifest_path}")
        return 0

    if args.analyze:
        run_analysis(cfg)
        return 0

    if args.smoke:
        tasks = build_manifest(cfg)
        write_manifest(tasks, manifest_path)
        log = logging.getLogger(__name__)
        for i, task in enumerate(tasks):
            log.info("[%d/%d] %s %s", i + 1, len(tasks), task.kind, task.spec or "")
            execute_task(task, cohort, cfg, output_root, force=args.force)
        run_analysis(cfg)
        return 0

    if args.task_index is None:
        parser.error("must pass --task-index, --smoke, --analyze, or --write-manifest")

    if not manifest_path.exists():
        tasks = build_manifest(cfg)
        write_manifest(tasks, manifest_path)
    else:
        tasks = read_manifest(manifest_path)

    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError(f"task-index {args.task_index} out of range (n_tasks={len(tasks)})")
    execute_task(tasks[args.task_index], cohort, cfg, output_root, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
