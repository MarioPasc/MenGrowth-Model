# src/growth/utils/__init__.py
"""
Utility functions for the growth forecasting pipeline.

Common utilities for config, checkpoints, logging, reproducibility, and paths.

Components:
- model_card: Model card generation for LoRA checkpoints
- reproducibility: Git info, environment capture, run manifests
- paths: Structured output directory management
"""

from growth.utils.model_card import (
    LoRAModelCardConfig,
    create_lora_model_card,
    model_card_from_training,
    save_lora_model_card,
)
from growth.utils.reproducibility import (
    get_git_info,
    get_environment_info,
    create_run_manifest,
    save_reproducibility_artifacts,
    check_reproducibility,
)
from growth.utils.paths import (
    ComponentPaths,
    ConditionPaths,
    ExperimentPaths,
    OutputPathManager,
    get_path_manager,
    get_features_path,
    get_targets_path,
    get_metrics_path,
)

__all__ = [
    # Model cards
    "LoRAModelCardConfig",
    "create_lora_model_card",
    "model_card_from_training",
    "save_lora_model_card",
    # Reproducibility
    "get_git_info",
    "get_environment_info",
    "create_run_manifest",
    "save_reproducibility_artifacts",
    "check_reproducibility",
    # Paths
    "ComponentPaths",
    "ConditionPaths",
    "ExperimentPaths",
    "OutputPathManager",
    "get_path_manager",
    "get_features_path",
    "get_targets_path",
    "get_metrics_path",
]
