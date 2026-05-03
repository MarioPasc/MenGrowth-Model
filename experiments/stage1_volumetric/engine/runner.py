# experiments/stage1_volumetric/engine/runner.py
"""Single-model LOPO-CV runner with resume support and timing."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from experiments.utils.experiment_output import save_stage_results
from growth.shared.bootstrap import BootstrapResult
from growth.shared.growth_models import PatientTrajectory
from growth.shared.lopo import LOPOEvaluator, LOPOResults

logger = logging.getLogger(__name__)


def run_single_model(
    model_name: str,
    model_cls: type,
    model_kwargs: dict,
    trajectories: list[PatientTrajectory],
    output_dir: Path,
    bootstrap_n: int = 2000,
    bootstrap_seed: int = 42,
    force: bool = False,
    cfg: dict | None = None,
) -> tuple[LOPOResults | None, dict[str, BootstrapResult] | None, bool]:
    """Run LOPO-CV for one model with resume support.

    Checks for cached results at output_dir/model_name/lopo_results.json.
    If found (and force=False), loads and returns cached results.
    Otherwise runs fresh LOPO-CV, saves results, and returns them.

    Args:
        model_name: Display name for logging and directory naming.
        model_cls: GrowthModel subclass to instantiate per fold.
        model_kwargs: Keyword arguments passed to model_cls constructor.
        trajectories: Patient trajectories for LOPO-CV.
        output_dir: Root output directory (model gets a subdirectory).
        bootstrap_n: Number of bootstrap samples for CIs.
        bootstrap_seed: Seed for bootstrap.
        force: If True, re-run even if cached results exist.
        cfg: Full config dict (passed through to save_stage_results).

    Returns:
        (lopo_results, bootstrap_cis, was_cached) tuple.
        lopo_results is None if the model failed.
    """
    cached_path = output_dir / model_name / "lopo_results.json"

    if not force and cached_path.exists():
        try:
            with open(cached_path) as f:
                cached_data = json.load(f)
            results = LOPOResults.from_dict(cached_data)
            m = results.aggregate_metrics
            r2 = m.get("last_from_rest/r2_log", float("nan"))
            crps = m.get("last_from_rest/crps", float("nan"))
            logger.info(
                f"  CACHED (R2={r2:.4f}, CRPS={crps:.4f}, "
                f"folds={len(results.fold_results)}). "
                "Use --force to re-run."
            )
            return results, None, True
        except Exception as e:
            logger.warning(f"  Failed to load cache, re-running: {e}")

    model_start = time.monotonic()
    evaluator = LOPOEvaluator()

    try:
        results = evaluator.evaluate(model_cls, trajectories, **model_kwargs)
        elapsed = time.monotonic() - model_start

        m = results.aggregate_metrics
        r2 = m.get("last_from_rest/r2_log", float("nan"))
        mae = m.get("last_from_rest/mae_log", float("nan"))
        cal = m.get("last_from_rest/calibration_95", float("nan"))
        crps = m.get("last_from_rest/crps", float("nan"))

        logger.info(
            f"  R2={r2:.4f}, MAE={mae:.4f}, Cal95={cal:.3f}, CRPS={crps:.4f}, "
            f"folds={len(results.fold_results)}/"
            f"{len(results.fold_results) + len(results.failed_folds)} "
            f"[{elapsed:.1f}s]"
        )

        ci_results = save_stage_results(
            output_dir=output_dir,
            model_name=model_name,
            lopo_results=results,
            trajectories=trajectories,
            config=cfg or {},
            stage_name="stage1_volumetric_uq",
            bootstrap_n=bootstrap_n,
            bootstrap_seed=bootstrap_seed,
        )

        return results, ci_results, False

    except Exception as e:
        elapsed = time.monotonic() - model_start
        logger.error(f"  {model_name} FAILED after {elapsed:.1f}s: {e}", exc_info=True)
        return None, None, False
