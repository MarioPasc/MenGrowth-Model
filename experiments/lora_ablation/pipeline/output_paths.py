# experiments/lora_ablation/output_paths.py
"""Centralized output path management for LoRA ablation experiments.

This module provides experiment-specific path management by extending the
generic utilities from growth.utils.paths.

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
from pathlib import Path
from typing import Optional

# Import from growth library
from growth.utils.paths import (
    ComponentPaths,
    ExperimentPaths,
    OutputPathManager,
    get_path_manager,
    get_features_path,
    get_targets_path,
    get_metrics_path,
)

logger = logging.getLogger(__name__)

# Backward compatibility alias
ConditionPaths = ComponentPaths


class AblationPathManager(OutputPathManager):
    """Path manager specialized for LoRA ablation experiments.

    Extends the base OutputPathManager from growth.utils.paths.
    This class exists mainly for backward compatibility - new code should
    use the base OutputPathManager directly.

    Args:
        output_dir: Root output directory.
        use_structured_layout: If True, use new structured layout.
            If False, use flat legacy layout for compatibility.

    Example:
        >>> paths = AblationPathManager("/results/lora_ablation")
        >>> cond_paths = paths.get_condition_paths("lora_r8")
        >>> cond_paths.best_model
        PosixPath('/results/lora_ablation/training/lora_r8/checkpoints/best.pt')
    """
    pass  # All functionality inherited from base class


# Backward compatibility - keep the old name working
OutputPathManager = AblationPathManager


def get_ablation_path_manager(
    config: dict,
    use_structured: Optional[bool] = None,
) -> OutputPathManager:
    """Get path manager from ablation config.

    This is a convenience wrapper around get_path_manager from growth.utils.paths.

    Args:
        config: Experiment configuration.
        use_structured: Override structured layout setting.
            If None, reads from config['experiment'].get('structured_output', False).

    Returns:
        Configured OutputPathManager.
    """
    return get_path_manager(config, use_structured)
