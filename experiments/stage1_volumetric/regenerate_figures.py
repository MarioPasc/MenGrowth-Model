# experiments/segment_based_approach/regenerate_figures.py
"""Re-generate all figures from cached results (no GPU required).

Loads volume_cache.json and lopo_results_*.json, reconstructs the data
structures, and calls the figure generation pipeline.

Usage:
    python -m experiments.stage1_volumetric.regenerate_figures \
        --results-dir /path/to/results \
        --config experiments/segment_based_approach/config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from growth.evaluation.lopo_evaluator import LOPOFoldResult, LOPOResults
from growth.models.growth.base import FitResult, PatientTrajectory
from growth.models.growth.hgp_model import HierarchicalGPModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.models.growth.scalar_gp import ScalarGP

from .run_baseline import generate_all_figures, _build_model_configs
from .segment import PerModelResult, ScanVolumes

logger = logging.getLogger(__name__)


def load_volume_cache(cache_path: Path) -> list[ScanVolumes]:
    """Load ScanVolumes from volume_cache.json."""
    with open(cache_path) as f:
        data = json.load(f)

    volumes = []
    for scan in data["scans"]:
        model_results = {}
        for mn, mr in scan.get("model_results", {}).items():
            model_results[mn] = PerModelResult(
                model_name=mr["model_name"],
                wt_vol_mm3=mr["wt_vol_mm3"],
                tc_vol_mm3=mr["tc_vol_mm3"],
                et_vol_mm3=mr["et_vol_mm3"],
                wt_dice=mr["wt_dice"],
                tc_dice=mr["tc_dice"],
                et_dice=mr["et_dice"],
                is_empty=mr["is_empty"],
            )
        centroid_raw = scan.get("centroid_xyz")
        centroid = tuple(centroid_raw) if centroid_raw else None

        volumes.append(
            ScanVolumes(
                scan_id=scan["scan_id"],
                patient_id=scan["patient_id"],
                timepoint_idx=scan["timepoint_idx"],
                manual_wt_vol_mm3=scan["manual_wt_vol_mm3"],
                manual_tc_vol_mm3=scan["manual_tc_vol_mm3"],
                manual_et_vol_mm3=scan["manual_et_vol_mm3"],
                is_empty_manual=scan["is_empty_manual"],
                centroid_xyz=centroid,
                model_results=model_results,
            )
        )
    logger.info(f"Loaded {len(volumes)} scans from {cache_path}")
    return volumes


def load_lopo_results(gp_dir: Path, sources: list[str]) -> dict[str, LOPOResults]:
    """Load all LOPO result JSONs from the growth_prediction directory."""
    lopo_results: dict[str, LOPOResults] = {}

    for source_name in sources:
        source_dir = gp_dir / source_name
        if not source_dir.is_dir():
            logger.warning(f"Source dir not found: {source_dir}")
            continue

        for json_file in sorted(source_dir.glob("lopo_results_*.json")):
            gp_name = json_file.stem.replace("lopo_results_", "")
            key = f"{gp_name}_{source_name}"

            with open(json_file) as f:
                data = json.load(f)

            fold_results = []
            for fr in data.get("fold_results", []):
                fit_data = fr.get("fit_result", {})
                fold_results.append(
                    LOPOFoldResult(
                        patient_id=fr["patient_id"],
                        n_timepoints=fr["n_timepoints"],
                        n_train_patients=fr["n_train_patients"],
                        n_train_observations=fr["n_train_observations"],
                        fit_result=FitResult(
                            log_marginal_likelihood=fit_data.get(
                                "log_marginal_likelihood", 0.0
                            ),
                            hyperparameters=fit_data.get("hyperparameters", {}),
                            condition_number=fit_data.get("condition_number", 0.0),
                        ),
                        predictions=fr.get("predictions", {}),
                        fit_time_s=fr.get("fit_time_s", 0.0),
                    )
                )

            lopo_results[key] = LOPOResults(
                model_name=data["model_name"],
                fold_results=fold_results,
                aggregate_metrics=data.get("aggregate_metrics", {}),
                failed_folds=data.get("failed_folds", []),
            )
            logger.info(f"Loaded {key}: {len(fold_results)} folds")

    return lopo_results


def build_trajectories_from_volumes(
    volumes: list[ScanVolumes],
    source: str,
    exclude_patients: list[str],
    min_timepoints: int = 2,
) -> list[PatientTrajectory]:
    """Build PatientTrajectory objects from ScanVolumes (ordinal time)."""
    patient_scans: dict[str, list[ScanVolumes]] = {}
    for sv in volumes:
        if sv.patient_id in exclude_patients:
            continue
        patient_scans.setdefault(sv.patient_id, []).append(sv)

    trajectories: list[PatientTrajectory] = []
    for pid, scans in sorted(patient_scans.items()):
        scans_sorted = sorted(scans, key=lambda s: s.timepoint_idx)
        if len(scans_sorted) < min_timepoints:
            continue

        times = np.array([s.timepoint_idx for s in scans_sorted], dtype=np.float64)

        if source == "manual":
            obs = np.array([s.manual_wt_vol_mm3 for s in scans_sorted])
        else:
            if source not in scans_sorted[0].model_results:
                continue
            obs = np.array([s.model_results[source].wt_vol_mm3 for s in scans_sorted])

        obs_log = np.log1p(obs)

        if np.all(obs == 0.0):
            continue

        # Attach centroid from first timepoint as static covariate
        covariates: dict[str, float] | None = None
        first_centroid = scans_sorted[0].centroid_xyz
        if first_centroid is not None:
            covariates = {
                "centroid_x": first_centroid[0],
                "centroid_y": first_centroid[1],
                "centroid_z": first_centroid[2],
            }

        trajectories.append(
            PatientTrajectory(
                patient_id=pid,
                times=times,
                observations=obs_log,
                covariates=covariates,
            )
        )

    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures from cached results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to results directory containing volume_cache.json and growth_prediction/",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/segment_based_approach/config.yaml",
        help="Path to config YAML (for model configs)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1. Load volume cache
    volumes = load_volume_cache(results_dir / "volume_cache.json")

    # 2. Discover sources from growth_prediction subdirectories
    gp_dir = results_dir / "growth_prediction"
    sources_on_disk = [
        d.name for d in sorted(gp_dir.iterdir())
        if d.is_dir() and (d / "model_comparison.json").exists()
    ]
    logger.info(f"Discovered sources: {sources_on_disk}")

    # 3. Load LOPO results
    lopo_results = load_lopo_results(gp_dir, sources_on_disk)
    logger.info(f"Loaded {len(lopo_results)} LOPO result sets")

    # 4. Build trajectories from volumes
    exclude = cfg.get("patients", {}).get("exclude", [])
    min_tp = cfg.get("patients", {}).get("min_timepoints", 2)

    sources: dict[str, list] = {}
    for source_name in sources_on_disk:
        trajs = build_trajectories_from_volumes(volumes, source_name, exclude, min_tp)
        sources[source_name] = trajs
        logger.info(f"  {source_name}: {len(trajs)} patient trajectories")

    # 5. Build model configs (CPU only — no GPU needed)
    model_configs = _build_model_configs(cfg)

    # 6. Determine seg model names
    seg_model_names = [s for s in sources_on_disk if s != "manual"]

    # 7. Generate all figures
    h5_path = cfg["paths"].get("mengrowth_h5")
    if h5_path and not Path(h5_path).exists():
        logger.warning(f"H5 file not found ({h5_path}), skipping segmentation overlay")
        h5_path = None

    logger.info("=== Generating Figures ===")
    generate_all_figures(
        lopo_results,
        volumes,
        results_dir,
        sources=sources,
        model_configs=model_configs,
        h5_path=h5_path,
        seg_model_names=seg_model_names,
    )
    logger.info("Done!")


if __name__ == "__main__":
    main()
