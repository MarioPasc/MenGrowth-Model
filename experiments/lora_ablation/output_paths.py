# experiments/lora_ablation/output_paths.py
"""Centralized output path management for LoRA ablation experiments.

Provides a structured output directory layout:

    output_dir/
    ├── meta/                      # Reproducibility artifacts
    │   ├── run_manifest.json
    │   ├── config.yaml
    │   └── pip_freeze.txt
    ├── training/                  # Per-condition training outputs
    │   └── <condition>/
    │       ├── checkpoints/
    │       ├── logs/
    │       └── curves/
    ├── features/                  # Extracted representations
    │   └── <condition>/
    │       ├── probe/
    │       ├── test/
    │       └── domain/
    ├── evaluation/                # Evaluation results
    │   └── <condition>/
    │       ├── probes/
    │       ├── dice/
    │       └── domain_shift/
    ├── figures/                   # Visualizations
    │   ├── main/
    │   └── supplementary/
    ├── tables/                    # CSV and LaTeX tables
    └── reports/                   # Analysis reports
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConditionPaths:
    """Paths for a single experimental condition."""

    # Root
    condition_name: str
    output_dir: Path

    # Training
    checkpoints_dir: Path
    logs_dir: Path
    curves_dir: Path

    # Features
    features_probe_dir: Path
    features_test_dir: Path
    features_domain_dir: Path

    # Evaluation
    probes_dir: Path
    dice_dir: Path
    domain_shift_dir: Path

    # Key files (for backward compatibility)
    best_model: Path
    checkpoint: Path
    training_log: Path
    training_summary: Path

    @property
    def legacy_dir(self) -> Path:
        """Legacy flat directory for backward compatibility."""
        return self.output_dir / "conditions" / self.condition_name


@dataclass
class ExperimentPaths:
    """Paths for the entire experiment."""

    output_dir: Path
    meta_dir: Path
    figures_dir: Path
    figures_main_dir: Path
    figures_supplementary_dir: Path
    tables_dir: Path
    reports_dir: Path

    # Legacy paths
    data_splits: Path


class OutputPathManager:
    """Manages output paths for LoRA ablation experiments.

    Provides both the new structured layout and backward-compatible
    legacy paths to support incremental migration.

    Args:
        output_dir: Root output directory.
        use_structured_layout: If True, use new structured layout.
            If False, use flat legacy layout for compatibility.

    Example:
        >>> paths = OutputPathManager("/results/lora_ablation")
        >>> cond_paths = paths.get_condition_paths("lora_r8")
        >>> cond_paths.best_model
        PosixPath('/results/lora_ablation/training/lora_r8/checkpoints/best.pt')
    """

    def __init__(
        self,
        output_dir: str | Path,
        use_structured_layout: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.use_structured = use_structured_layout
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create top-level directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_structured:
            (self.output_dir / "meta").mkdir(exist_ok=True)
            (self.output_dir / "training").mkdir(exist_ok=True)
            (self.output_dir / "features").mkdir(exist_ok=True)
            (self.output_dir / "evaluation").mkdir(exist_ok=True)
            (self.output_dir / "figures" / "main").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "figures" / "supplementary").mkdir(exist_ok=True)
            (self.output_dir / "tables").mkdir(exist_ok=True)
            (self.output_dir / "reports").mkdir(exist_ok=True)
        else:
            # Legacy: just conditions folder
            (self.output_dir / "conditions").mkdir(exist_ok=True)
            (self.output_dir / "figures").mkdir(exist_ok=True)

    def get_experiment_paths(self) -> ExperimentPaths:
        """Get experiment-level paths."""
        if self.use_structured:
            return ExperimentPaths(
                output_dir=self.output_dir,
                meta_dir=self.output_dir / "meta",
                figures_dir=self.output_dir / "figures",
                figures_main_dir=self.output_dir / "figures" / "main",
                figures_supplementary_dir=self.output_dir / "figures" / "supplementary",
                tables_dir=self.output_dir / "tables",
                reports_dir=self.output_dir / "reports",
                data_splits=self.output_dir / "meta" / "data_splits.json",
            )
        else:
            # Legacy flat layout
            return ExperimentPaths(
                output_dir=self.output_dir,
                meta_dir=self.output_dir,
                figures_dir=self.output_dir / "figures",
                figures_main_dir=self.output_dir / "figures",
                figures_supplementary_dir=self.output_dir / "figures",
                tables_dir=self.output_dir,
                reports_dir=self.output_dir,
                data_splits=self.output_dir / "data_splits.json",
            )

    def get_condition_paths(self, condition_name: str) -> ConditionPaths:
        """Get paths for a specific condition.

        Args:
            condition_name: Name of the condition (e.g., "lora_r8").

        Returns:
            ConditionPaths with all relevant paths.
        """
        if self.use_structured:
            return self._get_structured_paths(condition_name)
        else:
            return self._get_legacy_paths(condition_name)

    def _get_structured_paths(self, condition_name: str) -> ConditionPaths:
        """Get paths using new structured layout."""
        train_dir = self.output_dir / "training" / condition_name
        feat_dir = self.output_dir / "features" / condition_name
        eval_dir = self.output_dir / "evaluation" / condition_name

        # Ensure directories exist
        for d in [
            train_dir / "checkpoints",
            train_dir / "logs",
            train_dir / "curves",
            feat_dir / "probe",
            feat_dir / "test",
            feat_dir / "domain",
            eval_dir / "probes",
            eval_dir / "dice",
            eval_dir / "domain_shift",
        ]:
            d.mkdir(parents=True, exist_ok=True)

        return ConditionPaths(
            condition_name=condition_name,
            output_dir=self.output_dir,
            # Training
            checkpoints_dir=train_dir / "checkpoints",
            logs_dir=train_dir / "logs",
            curves_dir=train_dir / "curves",
            # Features
            features_probe_dir=feat_dir / "probe",
            features_test_dir=feat_dir / "test",
            features_domain_dir=feat_dir / "domain",
            # Evaluation
            probes_dir=eval_dir / "probes",
            dice_dir=eval_dir / "dice",
            domain_shift_dir=eval_dir / "domain_shift",
            # Key files
            best_model=train_dir / "checkpoints" / "best.pt",
            checkpoint=train_dir / "checkpoints" / "checkpoint.pt",
            training_log=train_dir / "logs" / "training_log.csv",
            training_summary=train_dir / "logs" / "training_summary.yaml",
        )

    def _get_legacy_paths(self, condition_name: str) -> ConditionPaths:
        """Get paths using legacy flat layout."""
        cond_dir = self.output_dir / "conditions" / condition_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        return ConditionPaths(
            condition_name=condition_name,
            output_dir=self.output_dir,
            # All in same directory for legacy
            checkpoints_dir=cond_dir,
            logs_dir=cond_dir,
            curves_dir=cond_dir,
            features_probe_dir=cond_dir,
            features_test_dir=cond_dir,
            features_domain_dir=cond_dir,
            probes_dir=cond_dir,
            dice_dir=cond_dir,
            domain_shift_dir=cond_dir,
            # Key files (legacy names)
            best_model=cond_dir / "best_model.pt",
            checkpoint=cond_dir / "checkpoint.pt",
            training_log=cond_dir / "training_log.csv",
            training_summary=cond_dir / "training_summary.yaml",
        )


def get_path_manager(
    config: dict,
    use_structured: Optional[bool] = None,
) -> OutputPathManager:
    """Get path manager from config.

    Args:
        config: Experiment configuration.
        use_structured: Override structured layout setting.
            If None, reads from config['experiment'].get('structured_output', False).

    Returns:
        Configured OutputPathManager.
    """
    output_dir = config["experiment"]["output_dir"]

    if use_structured is None:
        use_structured = config.get("experiment", {}).get("structured_output", False)

    return OutputPathManager(output_dir, use_structured_layout=use_structured)


# Convenience functions for common paths

def get_features_path(
    condition_dir: Path,
    split: str,
    level: str = "multi_scale",
) -> Path:
    """Get path to features file.

    Args:
        condition_dir: Condition directory (legacy or structured).
        split: "probe" or "test".
        level: Feature level ("encoder10" or "multi_scale").

    Returns:
        Path to features .pt file.
    """
    # Try structured path first
    structured_path = condition_dir / split / f"features_{level}.pt"
    if structured_path.exists():
        return structured_path

    # Fall back to legacy paths
    legacy_path = condition_dir / f"features_{split}_{level}.pt"
    if legacy_path.exists():
        return legacy_path

    # Default legacy without level
    return condition_dir / f"features_{split}.pt"


def get_targets_path(condition_dir: Path, split: str) -> Path:
    """Get path to targets file."""
    # Try structured path first
    structured_path = condition_dir / split / "targets.pt"
    if structured_path.exists():
        return structured_path

    # Legacy path
    return condition_dir / f"targets_{split}.pt"


def get_metrics_path(condition_dir: Path, enhanced: bool = True) -> Path:
    """Get path to metrics JSON file."""
    # Try structured path first
    probes_dir = condition_dir / "probes"
    if probes_dir.exists():
        suffix = "_enhanced" if enhanced else ""
        return probes_dir / f"metrics{suffix}.json"

    # Legacy path
    suffix = "_enhanced" if enhanced else ""
    path = condition_dir / f"metrics{suffix}.json"
    if path.exists() or not enhanced:
        return path

    # Fall back to non-enhanced
    return condition_dir / "metrics.json"
