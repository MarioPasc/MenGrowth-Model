# src/growth/utils/paths.py
"""Path management utilities for experiments.

Provides structured output directory management with support for
both hierarchical layouts and legacy flat layouts.

Example:
    >>> from growth.utils.paths import ComponentPaths, ExperimentPaths, OutputPathManager
    >>> manager = OutputPathManager("/results/experiment")
    >>> paths = manager.get_component_paths("lora_r8")
    >>> print(paths.checkpoints_dir)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ComponentPaths:
    """Paths for a single experimental component (e.g., a condition or model variant).

    This is a generalized version that can be used for any experiment type.

    Attributes:
        component_name: Name of the component (e.g., "lora_r8", "baseline").
        output_dir: Root output directory.
        checkpoints_dir: Directory for model checkpoints.
        logs_dir: Directory for logs.
        curves_dir: Directory for learning curves.
        features_probe_dir: Directory for probe split features.
        features_test_dir: Directory for test split features.
        features_domain_dir: Directory for domain shift features.
        probes_dir: Directory for probe evaluation results.
        dice_dir: Directory for Dice evaluation results.
        domain_shift_dir: Directory for domain shift analysis.
        best_model: Path to best model checkpoint.
        checkpoint: Path to latest checkpoint.
        training_log: Path to training log CSV.
        training_summary: Path to training summary YAML.
    """

    component_name: str
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

    # Key files
    best_model: Path
    checkpoint: Path
    training_log: Path
    training_summary: Path

    @property
    def legacy_dir(self) -> Path:
        """Legacy flat directory for backward compatibility."""
        return self.output_dir / "conditions" / self.component_name


# Backward compatibility alias
ConditionPaths = ComponentPaths


@dataclass
class ExperimentPaths:
    """Paths for the entire experiment.

    Attributes:
        output_dir: Root output directory.
        meta_dir: Directory for reproducibility artifacts.
        figures_dir: Root directory for figures.
        figures_main_dir: Directory for main figures.
        figures_supplementary_dir: Directory for supplementary figures.
        tables_dir: Directory for CSV and LaTeX tables.
        reports_dir: Directory for analysis reports.
        data_splits: Path to data splits JSON file.
    """

    output_dir: Path
    meta_dir: Path
    figures_dir: Path
    figures_main_dir: Path
    figures_supplementary_dir: Path
    tables_dir: Path
    reports_dir: Path
    data_splits: Path


class OutputPathManager:
    """Manages output paths for experiments.

    Provides both the new structured layout and backward-compatible
    legacy paths to support incremental migration.

    Structured layout:
        output_dir/
        ├── meta/                      # Reproducibility artifacts
        ├── training/                  # Per-component training outputs
        │   └── <component>/
        │       ├── checkpoints/
        │       ├── logs/
        │       └── curves/
        ├── features/                  # Extracted representations
        │   └── <component>/
        ├── evaluation/                # Evaluation results
        │   └── <component>/
        ├── figures/                   # Visualizations
        ├── tables/                    # CSV and LaTeX tables
        └── reports/                   # Analysis reports

    Legacy layout:
        output_dir/
        ├── conditions/
        │   └── <component>/           # All files in one directory
        ├── figures/
        └── data_splits.json

    Args:
        output_dir: Root output directory.
        use_structured_layout: If True, use new structured layout.
            If False, use flat legacy layout for compatibility.

    Example:
        >>> manager = OutputPathManager("/results/lora_ablation")
        >>> paths = manager.get_component_paths("lora_r8")
        >>> paths.best_model
        PosixPath('/results/lora_ablation/training/lora_r8/checkpoints/best.pt')
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
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
        """Get experiment-level paths.

        Returns:
            ExperimentPaths with all experiment-wide directories.
        """
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

    def get_component_paths(self, component_name: str) -> ComponentPaths:
        """Get paths for a specific component.

        Args:
            component_name: Name of the component (e.g., "lora_r8").

        Returns:
            ComponentPaths with all relevant paths.
        """
        if self.use_structured:
            return self._get_structured_paths(component_name)
        else:
            return self._get_legacy_paths(component_name)

    # Backward compatibility alias
    def get_condition_paths(self, condition_name: str) -> ComponentPaths:
        """Alias for get_component_paths (backward compatibility)."""
        return self.get_component_paths(condition_name)

    def _get_structured_paths(self, component_name: str) -> ComponentPaths:
        """Get paths using new structured layout."""
        train_dir = self.output_dir / "training" / component_name
        feat_dir = self.output_dir / "features" / component_name
        eval_dir = self.output_dir / "evaluation" / component_name

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

        return ComponentPaths(
            component_name=component_name,
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

    def _get_legacy_paths(self, component_name: str) -> ComponentPaths:
        """Get paths using legacy flat layout."""
        comp_dir = self.output_dir / "conditions" / component_name
        comp_dir.mkdir(parents=True, exist_ok=True)

        return ComponentPaths(
            component_name=component_name,
            output_dir=self.output_dir,
            # All in same directory for legacy
            checkpoints_dir=comp_dir,
            logs_dir=comp_dir,
            curves_dir=comp_dir,
            features_probe_dir=comp_dir,
            features_test_dir=comp_dir,
            features_domain_dir=comp_dir,
            probes_dir=comp_dir,
            dice_dir=comp_dir,
            domain_shift_dir=comp_dir,
            # Key files (legacy names)
            best_model=comp_dir / "best_model.pt",
            checkpoint=comp_dir / "checkpoint.pt",
            training_log=comp_dir / "training_log.csv",
            training_summary=comp_dir / "training_summary.yaml",
        )


def get_path_manager(
    config: dict,
    use_structured: Optional[bool] = None,
) -> OutputPathManager:
    """Get path manager from config.

    Args:
        config: Experiment configuration with "experiment.output_dir" key.
        use_structured: Override structured layout setting.
            If None, reads from config['experiment'].get('structured_output', False).

    Returns:
        Configured OutputPathManager.

    Example:
        >>> config = {"experiment": {"output_dir": "/results", "structured_output": True}}
        >>> manager = get_path_manager(config)
    """
    output_dir = config["experiment"]["output_dir"]

    if use_structured is None:
        use_structured = config.get("experiment", {}).get("structured_output", False)

    return OutputPathManager(output_dir, use_structured_layout=use_structured)


def get_features_path(
    component_dir: Path,
    split: str,
    level: str = "multi_scale",
) -> Path:
    """Get path to features file.

    Handles both structured and legacy path formats.

    Args:
        component_dir: Component directory (legacy or structured).
        split: "probe" or "test".
        level: Feature level ("encoder10" or "multi_scale").

    Returns:
        Path to features .pt file.
    """
    # Try structured path first
    structured_path = component_dir / split / f"features_{level}.pt"
    if structured_path.exists():
        return structured_path

    # Fall back to legacy paths
    legacy_path = component_dir / f"features_{split}_{level}.pt"
    if legacy_path.exists():
        return legacy_path

    # Default legacy without level
    return component_dir / f"features_{split}.pt"


def get_targets_path(component_dir: Path, split: str) -> Path:
    """Get path to targets file.

    Args:
        component_dir: Component directory.
        split: "probe" or "test".

    Returns:
        Path to targets .pt file.
    """
    # Try structured path first
    structured_path = component_dir / split / "targets.pt"
    if structured_path.exists():
        return structured_path

    # Legacy path
    return component_dir / f"targets_{split}.pt"


def get_metrics_path(component_dir: Path, enhanced: bool = True) -> Path:
    """Get path to metrics JSON file.

    Args:
        component_dir: Component directory.
        enhanced: Whether to look for enhanced metrics file.

    Returns:
        Path to metrics JSON file.
    """
    # Try structured path first
    probes_dir = component_dir / "probes"
    if probes_dir.exists():
        suffix = "_enhanced" if enhanced else ""
        return probes_dir / f"metrics{suffix}.json"

    # Legacy path
    suffix = "_enhanced" if enhanced else ""
    path = component_dir / f"metrics{suffix}.json"
    if path.exists() or not enhanced:
        return path

    # Fall back to non-enhanced
    return component_dir / "metrics.json"


__all__ = [
    # Dataclasses
    "ComponentPaths",
    "ConditionPaths",  # Backward compatibility alias
    "ExperimentPaths",
    # Manager
    "OutputPathManager",
    "get_path_manager",
    # Utilities
    "get_features_path",
    "get_targets_path",
    "get_metrics_path",
]
