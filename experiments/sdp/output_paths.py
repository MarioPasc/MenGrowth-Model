#!/usr/bin/env python
# experiments/sdp/output_paths.py
"""Structured run directory setup and path helpers for SDP experiments.

Provides a consistent directory layout for all SDP training outputs,
enabling reproducibility and easy navigation.

Run folder structure:
    outputs/sdp/{run_name}/
    ├── meta/
    │   ├── run_config.yaml
    │   └── data_manifest.json
    ├── checkpoints/
    │   └── phase2_sdp.pt
    ├── training/
    │   └── csv_log/version_0/metrics.csv
    ├── latent/
    │   ├── latent_train.h5
    │   ├── latent_val.h5
    │   └── latent_test.h5
    ├── evaluation/
    ├── figures/
    └── tables/
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class SDPRunPaths:
    """Path container for a structured SDP run directory.

    Attributes:
        root: Run root directory.
        meta: Meta information directory.
        checkpoints: Model checkpoint directory.
        training: Training log directory.
        latent: Latent vector HDF5 directory.
        evaluation: Evaluation JSON output directory.
        figures: Publication figure directory.
        tables: Table output directory.
    """

    root: Path
    meta: Path
    checkpoints: Path
    training: Path
    latent: Path
    evaluation: Path
    figures: Path
    tables: Path

    @property
    def config_path(self) -> Path:
        """Frozen config snapshot path."""
        return self.meta / "run_config.yaml"

    @property
    def manifest_path(self) -> Path:
        """Data manifest JSON path."""
        return self.meta / "data_manifest.json"

    @property
    def checkpoint_path(self) -> Path:
        """SDP checkpoint path."""
        return self.checkpoints / "phase2_sdp.pt"

    @property
    def quality_report_path(self) -> Path:
        """BLOCKING quality report JSON path."""
        return self.evaluation / "quality_report.json"

    @property
    def full_metrics_path(self) -> Path:
        """Full evaluation metrics JSON path."""
        return self.evaluation / "full_metrics.json"

    @property
    def cross_probing_path(self) -> Path:
        """Cross-probing results JSON path."""
        return self.evaluation / "cross_probing.json"

    @property
    def dci_scores_path(self) -> Path:
        """DCI scores JSON path."""
        return self.evaluation / "dci_scores.json"

    @property
    def jacobian_analysis_path(self) -> Path:
        """Jacobian XAI analysis JSON path."""
        return self.evaluation / "jacobian_analysis.json"

    @property
    def variance_analysis_path(self) -> Path:
        """Variance analysis JSON path."""
        return self.evaluation / "variance_analysis.json"


def create_run_dir(
    base_dir: str = "outputs/sdp",
    run_name: str | None = None,
) -> SDPRunPaths:
    """Create a structured run directory with all subdirectories.

    Args:
        base_dir: Base directory for SDP runs.
        run_name: Optional run name. If None, generates timestamp-based name.

    Returns:
        SDPRunPaths with all paths created.
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    root = Path(base_dir) / run_name

    paths = SDPRunPaths(
        root=root,
        meta=root / "meta",
        checkpoints=root / "checkpoints",
        training=root / "training",
        latent=root / "latent",
        evaluation=root / "evaluation",
        figures=root / "figures",
        tables=root / "tables",
    )

    # Create all directories
    for field_name in [
        "meta",
        "checkpoints",
        "training",
        "latent",
        "evaluation",
        "figures",
        "tables",
    ]:
        getattr(paths, field_name).mkdir(parents=True, exist_ok=True)

    logger.info(f"Created run directory: {root}")
    return paths


def save_run_config(paths: SDPRunPaths, config: DictConfig) -> None:
    """Save frozen config snapshot to meta directory.

    Args:
        paths: Run paths.
        config: OmegaConf config to save.
    """
    OmegaConf.save(config, paths.config_path)
    logger.info(f"Saved config to {paths.config_path}")


def save_data_manifest(
    paths: SDPRunPaths,
    n_train: int,
    n_val: int,
    n_test: int,
    feature_dim: int,
    target_dims: dict[str, int],
    norm_stats: dict[str, Any] | None = None,
) -> None:
    """Save data manifest with sample counts, dimensions, and git hash.

    Args:
        paths: Run paths.
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        feature_dim: Input feature dimension.
        target_dims: Dict of target name to dimension.
        norm_stats: Optional normalization statistics summary.
    """
    git_hash = _get_git_hash()

    manifest = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "feature_dim": feature_dim,
        "target_dims": target_dims,
        "git_hash": git_hash,
        "timestamp": datetime.now().isoformat(),
    }
    if norm_stats is not None:
        manifest["norm_stats"] = norm_stats

    with open(paths.manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved data manifest to {paths.manifest_path}")


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def load_run_paths(run_dir: str) -> SDPRunPaths:
    """Load paths from an existing run directory.

    Args:
        run_dir: Path to existing run directory.

    Returns:
        SDPRunPaths pointing to existing directories.

    Raises:
        FileNotFoundError: If run directory doesn't exist.
    """
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"Run directory not found: {root}")

    return SDPRunPaths(
        root=root,
        meta=root / "meta",
        checkpoints=root / "checkpoints",
        training=root / "training",
        latent=root / "latent",
        evaluation=root / "evaluation",
        figures=root / "figures",
        tables=root / "tables",
    )
