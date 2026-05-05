"""Top-level compute step: per-(model, case) metrics with a manifest cache.

Outputs (under ``analysis_root/cache/``):
    - ``per_case_metrics.parquet``: long-format DataFrame
        columns = [model, case_id, label, dice, hd95, lesion_recall]
    - ``manifest.json``: tracks per-model dir mtime so re-runs are incremental.
    - ``case_t1n_slices.json``: chosen axial slice per case for qualitative panels.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_GT_ROOT,
    DEFAULT_MODELS_ROOT,
    DEFAULT_OUR_RUN,
    ModelEntry,
    OURS_MODEL_ID,
    discover_models,
    load_ground_truth,
    load_prediction,
)
from .metrics import compute_case_metrics, label_mask

logger = logging.getLogger(__name__)

PER_CASE_CSV = "per_case_metrics.csv"
MANIFEST_JSON = "manifest.json"
SLICES_JSON = "case_t1n_slices.json"


def _cache_dir(analysis_root: Path) -> Path:
    out = analysis_root / "cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_manifest(analysis_root: Path) -> dict[str, dict]:
    p = _cache_dir(analysis_root) / MANIFEST_JSON
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _save_manifest(manifest: dict[str, dict], analysis_root: Path) -> None:
    p = _cache_dir(analysis_root) / MANIFEST_JSON
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _load_existing_metrics(analysis_root: Path) -> pd.DataFrame:
    p = _cache_dir(analysis_root) / PER_CASE_CSV
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _models_to_recompute(
    entries: list[ModelEntry],
    manifest: dict[str, dict],
    force: bool,
) -> list[ModelEntry]:
    """Decide which model entries need (re)computation."""
    if force:
        return entries
    out: list[ModelEntry] = []
    for entry in entries:
        meta = manifest.get(entry.model_id)
        if meta is None:
            out.append(entry)
            continue
        if entry.dir_mtime() > meta.get("dir_mtime", 0.0) + 1.0:
            out.append(entry)
    return out


def _eval_one_case(
    args: tuple[str, str, str, str, str, str],
) -> tuple[str, str, list[dict[str, float | str]] | None, str | None]:
    """Worker: load GT + prediction and compute metrics for one (model, case)."""
    model_id, schema, pred_dir, case_id, gt_root, _placeholder = args
    try:
        # Reconstruct an in-process ModelEntry.
        entry = ModelEntry(model_id=model_id, pred_dir=Path(pred_dir), schema=schema)
        pred, frame = load_prediction(entry, case_id)
        gt, spacing = load_ground_truth(case_id, frame=frame, gt_root=Path(gt_root))
        if pred.shape != gt.shape:
            return model_id, case_id, None, f"shape mismatch ({frame}): pred={pred.shape} gt={gt.shape}"
        rows = compute_case_metrics(gt, pred, spacing=spacing)
        return model_id, case_id, rows, None
    except FileNotFoundError as exc:
        return model_id, case_id, None, f"missing file: {exc}"
    except (OSError, ValueError) as exc:  # pragma: no cover (defensive)
        return model_id, case_id, None, f"{type(exc).__name__}: {exc}"


def _select_slice_for_case(case_id: str, gt_root: Path) -> dict[str, int]:
    """Pick the axial slice with the largest GT-TC area, per evaluation frame."""
    out: dict[str, int] = {}
    for frame in ("subject", "h5_192"):
        try:
            gt, _ = load_ground_truth(case_id, frame=frame, gt_root=gt_root)
        except FileNotFoundError:
            continue
        tc = label_mask(gt, "TC")
        if tc.any():
            out[frame] = int(np.argmax(tc.sum(axis=(0, 1))))
        else:
            out[frame] = int(gt.shape[2] // 2)
    return out


def compute_all(
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    models_root: Path = DEFAULT_MODELS_ROOT,
    our_run: Path = DEFAULT_OUR_RUN,
    gt_root: Path = DEFAULT_GT_ROOT,
    force: bool = False,
    workers: int | None = None,
) -> pd.DataFrame:
    """Compute (or refresh) ``per_case_metrics.parquet``.

    Returns the resulting long-format DataFrame.
    """
    cache = _cache_dir(analysis_root)
    manifest = _load_manifest(analysis_root)
    existing = _load_existing_metrics(analysis_root)

    entries = discover_models(models_root=models_root, our_run=our_run)
    if not entries:
        raise RuntimeError("No models discovered.")

    todo = _models_to_recompute(entries, manifest, force=force)
    if not todo:
        logger.info("compute: cache hit for %d models, nothing to recompute", len(entries))
        # still rebuild slices file if missing
        _ensure_slices_cache(analysis_root, entries, gt_root)
        return existing

    # Build the work list across all (model, case) we need.
    # Drop stale rows for models that will be recomputed.
    if not existing.empty:
        keep_mask = ~existing["model"].isin([e.model_id for e in todo])
        existing = existing[keep_mask].copy()

    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    n_iso = (1.0, 1.0, 1.0)  # dummy; per-case spacing comes from the GT NIfTI

    new_rows: list[dict] = []
    skipped: list[tuple[str, str, str]] = []

    work_items: list[tuple[str, str, str, str, str, str]] = []
    for entry in todo:
        for case_id in entry.case_ids():
            work_items.append(
                (entry.model_id, entry.schema, str(entry.pred_dir), case_id, str(gt_root), "")
            )
    logger.info(
        "compute: %d (model, case) tasks across %d models with %d workers",
        len(work_items),
        len(todo),
        workers,
    )

    t0 = time.time()
    if workers <= 1:
        results = (_eval_one_case(item) for item in work_items)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = [ex.submit(_eval_one_case, item) for item in work_items]
            results = (f.result() for f in as_completed(futures))
        finally:
            pass  # context handled below

    def _drain(results: Iterable[tuple]) -> None:
        for model_id, case_id, rows, err in results:
            if err is not None:
                skipped.append((model_id, case_id, err))
                continue
            if rows is None:
                continue
            for row in rows:
                new_rows.append({"model": model_id, "case_id": case_id, **row})

    if workers <= 1:
        _drain(results)
    else:
        try:
            _drain(results)
        finally:
            ex.shutdown(wait=True)

    if skipped:
        logger.warning("compute: %d (model, case) skipped (showing first 5): %s", len(skipped), skipped[:5])

    new_df = pd.DataFrame(new_rows)
    out_df = pd.concat([existing, new_df], ignore_index=True) if not new_df.empty else existing

    # Persist
    out_path = cache / PER_CASE_CSV
    out_df.to_csv(out_path, index=False)
    logger.info(
        "compute: wrote %d rows to %s (added %d, %.1fs)",
        len(out_df),
        out_path,
        len(new_df),
        time.time() - t0,
    )

    # Update manifest
    for entry in todo:
        manifest[entry.model_id] = {
            "dir_mtime": entry.dir_mtime(),
            "n_cases": len(entry.case_ids()),
            "schema": entry.schema,
            "pred_dir": str(entry.pred_dir),
            "completed_at": time.time(),
        }
    _save_manifest(manifest, analysis_root)

    _ensure_slices_cache(analysis_root, entries, gt_root)
    return out_df


def _ensure_slices_cache(analysis_root: Path, entries: list[ModelEntry], gt_root: Path) -> None:
    """Populate ``case_t1n_slices.json`` for the union of cases across models."""
    slices_path = _cache_dir(analysis_root) / SLICES_JSON
    existing: dict[str, int] = {}
    if slices_path.exists():
        existing = json.loads(slices_path.read_text())

    union: set[str] = set()
    for e in entries:
        union.update(e.case_ids())

    missing = sorted(union - set(existing.keys()))
    if not missing:
        return
    logger.info("compute: selecting axial slice for %d new cases", len(missing))
    for case_id in missing:
        try:
            existing[case_id] = _select_slice_for_case(case_id, gt_root)
        except FileNotFoundError as exc:
            logger.warning("Skipping slice for %s: %s", case_id, exc)
    slices_path.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str))
