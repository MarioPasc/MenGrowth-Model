"""Centralised data loading for the ensemble plotting suite.

Loads all CSVs/JSONs from a run directory into a single dataclass so that
figure modules receive one object instead of N separate DataFrames.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EnsembleResultsData:
    """All data from a single experiment run."""

    run_dir: Path

    # Training
    training_curves: pd.DataFrame          # aggregated_training_curves.csv

    # Evaluation
    per_member_dice: pd.DataFrame          # per_member_test_dice.csv
    ensemble_dice: pd.DataFrame            # ensemble_test_dice.csv
    baseline_dice: pd.DataFrame            # baseline_test_dice.csv
    paired_differences: pd.DataFrame       # paired_differences.csv
    convergence_wt: pd.DataFrame           # convergence_dice_wt.csv
    convergence_tc: pd.DataFrame           # convergence_dice_tc.csv
    convergence_et: pd.DataFrame           # convergence_dice_et.csv
    statistical_summary: dict[str, Any]    # statistical_summary.json
    calibration: dict[str, Any]            # calibration.json

    # Volumes (may be None if inference hasn't run)
    mengrowth_volumes: pd.DataFrame | None  # mengrowth_ensemble_volumes.csv

    # Predictions directory (may be None)
    predictions_dir: Path | None           # predictions/

    # Epistemic-uncertainty diagnostics (populated by epistemic_metrics.py).
    # All optional: None if the diagnostic pipeline has not run yet.
    bias_diagnostics: pd.DataFrame | None = None
    calibration_coverage: pd.DataFrame | None = None
    bias_dominance_threshold: pd.DataFrame | None = None
    epistemic_taxonomy: dict[str, Any] | None = None
    cross_rank_summary: pd.DataFrame | None = None

    # Ensemble-of-k Dice curves + threshold sensitivity (populated by
    # evaluate_ensemble_per_subject when per-member soft probs are saved).
    # Keys of ensemble_k_convergence: "wt", "tc", "et".
    ensemble_k_convergence: dict[str, pd.DataFrame] | None = None
    threshold_sensitivity: pd.DataFrame | None = None

    # Scans with full per-member predictions
    sample_scans: list[str] = dataclasses.field(default_factory=list)

    @property
    def n_members(self) -> int:
        """Number of ensemble members."""
        return self.per_member_dice["member_id"].nunique()

    @property
    def n_test_scans(self) -> int:
        """Number of test scans evaluated."""
        return len(self.ensemble_dice)

    @property
    def has_volumes(self) -> bool:
        """Whether MenGrowth volume data is available."""
        return self.mengrowth_volumes is not None

    @property
    def has_predictions(self) -> bool:
        """Whether NIfTI prediction directory exists."""
        return self.predictions_dir is not None and self.predictions_dir.exists()


def _read_csv_optional(path: Path) -> pd.DataFrame | None:
    """Read a CSV file, returning None if it doesn't exist."""
    if path.exists():
        return pd.read_csv(path)
    logger.warning("Missing file: %s", path)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def _find_sample_scans(predictions_dir: Path, n_members: int) -> list[str]:
    """Find scans that have full per-member predictions.

    A scan is "full" if it contains member_0_mask.nii.gz through
    member_{M-1}_mask.nii.gz.

    Args:
        predictions_dir: Path to the predictions directory.
        n_members: Expected number of ensemble members.

    Returns:
        Sorted list of scan IDs with complete per-member data.
    """
    sample_scans: list[str] = []
    if not predictions_dir.exists():
        return sample_scans

    for scan_dir in sorted(predictions_dir.iterdir()):
        if not scan_dir.is_dir():
            continue
        member_masks = list(scan_dir.glob("member_*_mask.nii.gz"))
        if len(member_masks) >= n_members:
            sample_scans.append(scan_dir.name)

    return sample_scans


def load_results(run_dir: Path) -> EnsembleResultsData:
    """Load all results from a run directory.

    Gracefully handles missing optional files (sets to None).

    Args:
        run_dir: Path to the experiment run directory
            (e.g., ``results/uncertainty_segmentation/r8_M10_s42/``).

    Returns:
        Populated EnsembleResultsData instance.

    Raises:
        FileNotFoundError: If required evaluation files are missing.
    """
    run_dir = Path(run_dir)
    eval_dir = run_dir / "evaluation"
    vol_dir = run_dir / "volumes"
    pred_dir = run_dir / "predictions"

    logger.info("Loading results from %s", run_dir)

    # --- Required files ---
    required_csvs = {
        "training_curves": eval_dir / "aggregated_training_curves.csv",
        "per_member_dice": eval_dir / "per_member_test_dice.csv",
        "ensemble_dice": eval_dir / "ensemble_test_dice.csv",
        "baseline_dice": eval_dir / "baseline_test_dice.csv",
        "paired_differences": eval_dir / "paired_differences.csv",
        "convergence_wt": eval_dir / "convergence_dice_wt.csv",
        "convergence_tc": eval_dir / "convergence_dice_tc.csv",
        "convergence_et": eval_dir / "convergence_dice_et.csv",
    }

    loaded: dict[str, pd.DataFrame] = {}
    for name, path in required_csvs.items():
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")
        loaded[name] = pd.read_csv(path)
        logger.info("  Loaded %s (%d rows)", name, len(loaded[name]))

    # JSON files
    stats_path = eval_dir / "statistical_summary.json"
    calib_path = eval_dir / "calibration.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Required file missing: {stats_path}")

    stats = _read_json(stats_path)
    calibration = _read_json(calib_path) if calib_path.exists() else stats.get(
        "calibration", {}
    )
    logger.info("  Loaded statistical_summary.json + calibration.json")

    # --- Optional files ---
    mengrowth_volumes = _read_csv_optional(
        vol_dir / "mengrowth_ensemble_volumes.csv"
    )
    if mengrowth_volumes is not None:
        logger.info("  Loaded mengrowth_volumes (%d rows)", len(mengrowth_volumes))
    else:
        logger.warning("  MenGrowth volumes not found — Figs 11-12 will be skipped")

    predictions_dir = pred_dir if pred_dir.exists() else None
    if predictions_dir is not None:
        logger.info("  Predictions directory found")
    else:
        logger.warning("  Predictions directory not found — Figs 13-14 will be skipped")

    # Determine number of members from per_member_dice
    n_members = loaded["per_member_dice"]["member_id"].nunique()

    # Find scans with full per-member predictions
    sample_scans: list[str] = []
    if predictions_dir is not None:
        sample_scans = _find_sample_scans(predictions_dir, n_members)
        logger.info("  Found %d scans with full per-member predictions: %s",
                     len(sample_scans), sample_scans)

    # Epistemic diagnostics (cached CSVs + JSON, optional).
    bias_diag = _read_csv_optional(eval_dir / "bias_diagnostics.csv")
    calib_cov = _read_csv_optional(eval_dir / "calibration_coverage.csv")
    k_star_df = _read_csv_optional(eval_dir / "bias_dominance_threshold.csv")
    taxonomy_path = eval_dir / "epistemic_taxonomy.json"
    taxonomy = _read_json(taxonomy_path) if taxonomy_path.exists() else None
    if bias_diag is not None:
        logger.info("  Loaded bias_diagnostics (%d rows)", len(bias_diag))
    if calib_cov is not None:
        logger.info("  Loaded calibration_coverage (%d rows)", len(calib_cov))
    if k_star_df is not None:
        logger.info("  Loaded bias_dominance_threshold (%d rows)", len(k_star_df))

    # Cross-rank summary lives next to sibling ranks, not inside run_dir.
    cross_rank_path = (
        run_dir.parent / "epistemic_summary" / "cross_rank_epistemic_summary.csv"
    )
    cross_rank = _read_csv_optional(cross_rank_path)
    if cross_rank is not None:
        logger.info("  Loaded cross_rank_summary (%d rows)", len(cross_rank))

    # Ensemble-of-k Dice curves (per channel) and threshold sensitivity.
    ensemble_k_convergence: dict[str, pd.DataFrame] = {}
    for ch in ("wt", "tc", "et"):
        ek_df = _read_csv_optional(eval_dir / f"convergence_ensemble_dice_{ch}.csv")
        if ek_df is not None:
            ensemble_k_convergence[ch] = ek_df
            logger.info("  Loaded ensemble-k Dice (%s): %d rows", ch, len(ek_df))
    threshold_df = _read_csv_optional(eval_dir / "threshold_sensitivity.csv")
    if threshold_df is not None:
        logger.info(
            "  Loaded threshold_sensitivity: %d rows, %d thresholds",
            len(threshold_df), threshold_df["threshold"].nunique(),
        )

    return EnsembleResultsData(
        run_dir=run_dir,
        training_curves=loaded["training_curves"],
        per_member_dice=loaded["per_member_dice"],
        ensemble_dice=loaded["ensemble_dice"],
        baseline_dice=loaded["baseline_dice"],
        paired_differences=loaded["paired_differences"],
        convergence_wt=loaded["convergence_wt"],
        convergence_tc=loaded["convergence_tc"],
        convergence_et=loaded["convergence_et"],
        statistical_summary=stats,
        calibration=calibration,
        mengrowth_volumes=mengrowth_volumes,
        predictions_dir=predictions_dir,
        bias_diagnostics=bias_diag,
        calibration_coverage=calib_cov,
        bias_dominance_threshold=k_star_df,
        epistemic_taxonomy=taxonomy,
        cross_rank_summary=cross_rank,
        ensemble_k_convergence=ensemble_k_convergence or None,
        threshold_sensitivity=threshold_df,
        sample_scans=sample_scans,
    )
