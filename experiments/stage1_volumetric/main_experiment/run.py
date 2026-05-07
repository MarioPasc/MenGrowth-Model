"""CLI entry point for the main σ²_v propagation experiment.

Run modes
---------

Per-cell::

    python -m experiments.stage1_volumetric.main_experiment.run \\
        --config <CONFIG.yaml> --task-index <K>

This executes the K-th task from the manifest produced by ``--write-manifest``.
That keeps SLURM array jobs simple: ``--array=0-N`` maps each task index to a
(family, level, seed) cell or a baseline.

Smoke (one-shot, all cells sequentially)::

    python -m experiments.stage1_volumetric.main_experiment.run \\
        --config <CONFIG.yaml> --smoke

Analysis (after all cells finished)::

    python -m experiments.stage1_volumetric.main_experiment.run \\
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

import numpy as np

from experiments.stage1_volumetric.engine.data import load_config

from .modules import aggregator, figures, statistics
from .modules.cohort import Cohort, load_cohort, write_cohort_meta
from .modules.runner import (
    CellSpec,
    iter_ablation_specs,
    iter_primary_specs,
    resolve_tau_grid,
    run_baseline_lme,
    run_baseline_lme_hetero_zero,
    run_cell,
)


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

    kind: str  # "baseline_lme" | "baseline_lme_hetero_zero" | "cell"
    spec: dict[str, Any] | None = None  # CellSpec fields when kind == "cell"


def build_manifest(cfg: dict, cohort: Cohort) -> list[TaskEntry]:
    tasks: list[TaskEntry] = []
    if cfg["evaluation"]["models"].get("fit_lme", True):
        tasks.append(TaskEntry(kind="baseline_lme"))
    if cfg["evaluation"]["models"].get("fit_lme_hetero_zero", True):
        tasks.append(TaskEntry(kind="baseline_lme_hetero_zero"))

    for spec in iter_primary_specs(cfg, cohort):
        tasks.append(TaskEntry(kind="cell", spec=spec.__dict__.copy()))
    for spec in iter_ablation_specs(cfg):
        tasks.append(TaskEntry(kind="cell", spec=spec.__dict__.copy()))
    return tasks


def write_manifest(tasks: list[TaskEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([{"kind": t.kind, "spec": t.spec} for t in tasks], f, indent=2)


def read_manifest(path: Path) -> list[TaskEntry]:
    with open(path) as f:
        data = json.load(f)
    return [TaskEntry(kind=d["kind"], spec=d.get("spec")) for d in data]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _prepare(cfg: dict) -> tuple[Cohort, Path]:
    output_root = Path(cfg["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(cfg)
    write_cohort_meta(cohort, output_root / "cohort_meta.json")

    # Persist the resolved τ-grid + saturation parameters for traceability.
    tau_grid = resolve_tau_grid(cfg, cohort)
    sweep_meta_path = output_root / "tau_grid.json"
    with open(sweep_meta_path, "w") as f:
        json.dump(
            {
                "tau_grid": [float(t) for t in tau_grid],
                "saturation": cfg["sweep"]["primary"].get("saturation", {}),
                "n_tau": int(tau_grid.size),
                "tau_min": float(tau_grid.min()),
                "tau_max": float(tau_grid.max()),
                "contains_zero": bool(np.any(np.isclose(tau_grid, 0.0))),
            },
            f,
            indent=2,
        )

    return cohort, output_root


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------


def execute_task(task: TaskEntry, cfg: dict, force: bool = False) -> dict[str, Any]:
    cohort, output_root = _prepare(cfg)

    if task.kind == "baseline_lme":
        return run_baseline_lme(cohort, cfg, output_root, force=force)
    if task.kind == "baseline_lme_hetero_zero":
        return run_baseline_lme_hetero_zero(cohort, cfg, output_root, force=force)
    if task.kind == "cell":
        spec = CellSpec(**task.spec)
        return run_cell(spec, cohort, cfg, output_root, force=force)
    raise ValueError(f"Unknown task kind: {task.kind}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _load_lopo(path: Path):
    from growth.shared.lopo import LOPOResults

    with open(path) as f:
        return LOPOResults.from_dict(json.load(f))


def run_analysis(cfg: dict) -> None:
    cohort, output_root = _prepare(cfg)

    df = aggregator.collect_runs(output_root)
    aggregator.write_table(df, output_root)

    # Bootstrap + Wilcoxon per cell against each transition pair
    transition_pairs = cfg["reporting"]["transition_pairs"]
    bootstrap_seed = int(cfg["statistics"]["bootstrap"]["seed"])
    n_bootstrap = int(cfg["statistics"]["bootstrap"]["n_samples"])
    confidence = float(cfg["statistics"]["bootstrap"]["confidence_level"])

    cuts = (
        float(np.quantile(cohort.empirical_sigma_v_sq_flat, 1 / 3)),
        float(np.quantile(cohort.empirical_sigma_v_sq_flat, 2 / 3)),
    )

    # Pre-load baselines
    baselines = {
        "LME": output_root / "LME_baseline" / "lopo_results.json",
        "LMEHetero_Zero": output_root / "LMEHetero_Zero_baseline" / "lopo_results.json",
    }
    baseline_results = {k: _load_lopo(p) if p.exists() else None for k, p in baselines.items()}

    # Iterate cells
    bootstrap_payload: list[dict[str, Any]] = []
    wilcoxon_payload: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    runs_dir = output_root / "runs"
    if runs_dir.exists():
        for cell_dir in sorted(runs_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            for seed_dir in sorted(cell_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                lopo_path = seed_dir / "lopo_results.json"
                if not lopo_path.exists():
                    continue
                cell_results = _load_lopo(lopo_path)
                seed_idx = int(seed_dir.name.split("_")[1])

                family, level, level_value = _parse_cell_name(cell_dir.name)

                for arm_a, arm_b_label in transition_pairs:
                    if arm_a not in baseline_results or baseline_results[arm_a] is None:
                        continue
                    transition_label = f"{arm_a}__{arm_b_label}"
                    try:
                        bp = statistics.run_bootstrap_for_pair(
                            baseline_results[arm_a],
                            cell_results,
                            edges=cuts,
                            n_bootstrap=n_bootstrap,
                            confidence_level=confidence,
                            seed=bootstrap_seed + seed_idx,
                        )
                    except Exception as exc:
                        logging.getLogger(__name__).warning(
                            "Bootstrap failed for %s seed %d: %s", cell_dir.name, seed_idx, exc
                        )
                        bp = {}

                    for metric, entries in bp.items():
                        for e in entries:
                            row = {
                                "family": family,
                                "level": level,
                                "level_value": level_value,
                                "seed": seed_idx,
                                "transition": transition_label,
                                **e,
                            }
                            bootstrap_payload.append(row)
                            delta_rows.append(
                                {
                                    "family": family,
                                    "level": level,
                                    "level_value": level_value,
                                    "seed": seed_idx,
                                    "transition": transition_label,
                                    "scope": "marginal"
                                    if e["tertile"] == "marginal"
                                    else "tertile",
                                    "tertile": e["tertile"]
                                    if e["tertile"] != "marginal"
                                    else "all",
                                    "metric": e["metric"],
                                    "delta": e["delta"],
                                    "p_value": e["p_value"],
                                }
                            )

                    try:
                        wd = statistics.wilcoxon_cohen_d(
                            baseline_results[arm_a], cell_results, protocol="last_from_rest"
                        )
                        wilcoxon_payload.append(
                            {
                                "family": family,
                                "level": level,
                                "level_value": level_value,
                                "seed": seed_idx,
                                "transition": transition_label,
                                **wd,
                            }
                        )
                    except Exception as exc:
                        logging.getLogger(__name__).warning(
                            "Wilcoxon failed for %s seed %d: %s", cell_dir.name, seed_idx, exc
                        )

    # BH-FDR
    p_marginal = np.asarray(
        [r["p_value"] for r in bootstrap_payload if r["tertile"] == "marginal"], dtype=np.float64
    )
    rej_m, padj_m = statistics.bh_fdr(p_marginal, q=cfg["statistics"]["bh_fdr_q"])
    p_tertile = np.asarray(
        [r["p_value"] for r in bootstrap_payload if r["tertile"] != "marginal"], dtype=np.float64
    )
    rej_t, padj_t = statistics.bh_fdr(p_tertile, q=cfg["statistics"]["bh_fdr_q"])

    # Stamp
    mi = ti = 0
    for r in bootstrap_payload:
        if r["tertile"] == "marginal":
            r["p_adj_bh"] = float(padj_m[mi]) if mi < len(padj_m) else float("nan")
            r["rejected_bh"] = bool(rej_m[mi]) if mi < len(rej_m) else False
            mi += 1
        else:
            r["p_adj_bh"] = float(padj_t[ti]) if ti < len(padj_t) else float("nan")
            r["rejected_bh"] = bool(rej_t[ti]) if ti < len(rej_t) else False
            ti += 1

    # Spearman trend
    import pandas as pd

    delta_df = pd.DataFrame(delta_rows)
    spearman_results: list[dict[str, Any]] = []
    if not delta_df.empty:
        for transition_label in delta_df["transition"].unique():
            for metric in ("is_95", "cov_95", "r2_log", "coverage_95"):
                for scope, tert in (
                    ("marginal", "all"),
                    ("tertile", "low"),
                    ("tertile", "mid"),
                    ("tertile", "high"),
                ):
                    res = statistics.spearman_across_sweep(
                        delta_df,
                        metric=metric,
                        scope=scope,
                        tertile=tert,
                        transition_label=transition_label,
                    )
                    spearman_results.append(
                        {
                            "transition": transition_label,
                            "metric": metric,
                            "scope": scope,
                            "tertile": tert,
                            **res,
                        }
                    )

    statistics.write_results(
        {"results": bootstrap_payload}, output_root / "aggregated" / "bootstrap_results.json"
    )
    statistics.write_results(
        {"results": wilcoxon_payload}, output_root / "aggregated" / "wilcoxon_results.json"
    )
    statistics.write_results(
        {"results": spearman_results}, output_root / "aggregated" / "spearman_results.json"
    )

    figures.make_all_figures(df, output_root, cfg)


def _parse_cell_name(cell_name: str) -> tuple[str, str, float]:
    for fam in ("empirical_shift", "beta_alpha"):
        if cell_name.startswith(fam + "_"):
            level = cell_name[len(fam) + 1 :]
            try:
                if level.startswith("tau_"):
                    return fam, level, float(level[len("tau_") :])
                if level.startswith("alpha_"):
                    return fam, level, float(level[len("alpha_") :])
                return fam, level, float("nan")
            except ValueError:
                return fam, level, float("nan")
    return "unknown", cell_name, float("nan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    cfg = load_config(args.config)

    cohort, output_root = _prepare(cfg)
    manifest_path = output_root / "manifest.json"

    if args.write_manifest:
        tasks = build_manifest(cfg, cohort)
        write_manifest(tasks, manifest_path)
        print(f"Wrote {len(tasks)} tasks to {manifest_path}")
        return 0

    if args.analyze:
        run_analysis(cfg)
        return 0

    if args.smoke:
        # Build manifest, run all sequentially, then analyze
        tasks = build_manifest(cfg, cohort)
        write_manifest(tasks, manifest_path)
        for i, task in enumerate(tasks):
            logging.getLogger(__name__).info(
                "[%d/%d] %s %s", i + 1, len(tasks), task.kind, task.spec or ""
            )
            execute_task(task, cfg, force=args.force)
        run_analysis(cfg)
        return 0

    if args.task_index is None:
        parser.error("must pass --task-index, --smoke, --analyze, or --write-manifest")

    if not manifest_path.exists():
        tasks = build_manifest(cfg, cohort)
        write_manifest(tasks, manifest_path)
    else:
        tasks = read_manifest(manifest_path)

    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError(f"task-index {args.task_index} out of range (n_tasks={len(tasks)})")
    execute_task(tasks[args.task_index], cfg, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
