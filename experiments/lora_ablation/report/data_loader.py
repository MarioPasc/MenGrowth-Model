"""Data loading for the LoRA ablation report.

Reads all result files from experiment output directories into
structured dataclasses for downstream figure and narrative generation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from experiments.lora_ablation.report.style import (
    CONDITION_ORDER_ALL,
    EXPERIMENT_LABELS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ConditionData:
    """All loaded data for a single experimental condition."""

    name: str
    training_summary: Dict = field(default_factory=dict)
    training_log: Optional[pd.DataFrame] = None
    dice_men: Dict = field(default_factory=dict)
    dice_gli: Dict = field(default_factory=dict)
    metrics_enhanced: Dict = field(default_factory=dict)
    domain_metrics: Dict = field(default_factory=dict)
    has_features: bool = False


@dataclass
class ExperimentData:
    """All loaded data for one experiment variant."""

    name: str
    adapter_type: str  # "lora" or "dora"
    semantic_heads: bool
    config: Dict = field(default_factory=dict)
    conditions: Dict[str, ConditionData] = field(default_factory=dict)
    statistical_comparisons: Optional[Dict] = None


# ─────────────────────────────────────────────────────────────────────
# File loading helpers
# ─────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, returning None if missing."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _load_yaml(path: Path) -> Optional[Dict]:
    """Load YAML file, returning None if missing."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV file into DataFrame, returning None if missing."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _infer_experiment_info(name: str) -> Tuple[str, bool]:
    """Infer adapter type and semantic heads from experiment directory name.

    Args:
        name: Experiment directory name.

    Returns:
        (adapter_type, semantic_heads) tuple.
    """
    adapter_type = "dora" if "dora" in name.lower() else "lora"
    semantic_heads = "no_semantic" not in name.lower()
    return adapter_type, semantic_heads


# ─────────────────────────────────────────────────────────────────────
# Condition loading
# ─────────────────────────────────────────────────────────────────────


def load_condition(condition_dir: Path) -> ConditionData:
    """Load all result files for a single condition.

    Args:
        condition_dir: Path to condition directory (e.g. .../conditions/lora_r8/).

    Returns:
        Populated ConditionData instance.
    """
    name = condition_dir.name
    cond = ConditionData(name=name)

    # Training summary
    summary = _load_yaml(condition_dir / "training_summary.yaml")
    if summary is not None:
        cond.training_summary = summary
    else:
        logger.warning("No training_summary.yaml in %s", name)

    # Training log
    cond.training_log = _load_csv(condition_dir / "training_log.csv")

    # Dice scores
    dice_men = _load_json(condition_dir / "test_dice_men.json")
    if dice_men is not None:
        cond.dice_men = dice_men
    else:
        logger.warning("No test_dice_men.json in %s", name)

    dice_gli = _load_json(condition_dir / "test_dice_gli.json")
    if dice_gli is not None:
        cond.dice_gli = dice_gli

    # Enhanced metrics (probes)
    enhanced = _load_json(condition_dir / "metrics_enhanced.json")
    if enhanced is None:
        enhanced = _load_json(condition_dir / "metrics.json")
    if enhanced is not None:
        cond.metrics_enhanced = enhanced

    # Domain metrics
    domain = _load_json(condition_dir / "domain_metrics.json")
    if domain is not None:
        cond.domain_metrics = domain

    # Check feature files for UMAP
    cond.has_features = (
        (condition_dir / "features_glioma.pt").exists()
        and (condition_dir / "features_meningioma_subset.pt").exists()
    )

    return cond


# ─────────────────────────────────────────────────────────────────────
# Experiment loading
# ─────────────────────────────────────────────────────────────────────


def _sort_conditions(conditions: Dict[str, ConditionData]) -> Dict[str, ConditionData]:
    """Sort conditions by canonical order."""

    def sort_key(name: str) -> int:
        try:
            return CONDITION_ORDER_ALL.index(name)
        except ValueError:
            return 999

    return dict(sorted(conditions.items(), key=lambda kv: sort_key(kv[0])))


def load_experiment(exp_dir: Path) -> ExperimentData:
    """Load all data for one experiment variant.

    Args:
        exp_dir: Path to experiment directory
            (e.g. .../lora_ablation_semantic_heads/).

    Returns:
        Populated ExperimentData instance.
    """
    name = exp_dir.name
    adapter_type, semantic_heads = _infer_experiment_info(name)

    exp = ExperimentData(
        name=name,
        adapter_type=adapter_type,
        semantic_heads=semantic_heads,
    )

    # Config
    config = _load_yaml(exp_dir / "config.yaml")
    if config is not None:
        exp.config = config

    # Statistical comparisons
    stats = _load_json(exp_dir / "statistical_comparisons.json")
    if stats is not None:
        exp.statistical_comparisons = stats

    # Load conditions
    conditions_dir = exp_dir / "conditions"
    if not conditions_dir.exists():
        logger.error("No conditions/ directory in %s", exp_dir)
        return exp

    conditions: Dict[str, ConditionData] = {}
    for cond_dir in sorted(conditions_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        cond = load_condition(cond_dir)
        conditions[cond.name] = cond
        logger.info(
            "Loaded condition: %s (dice_men=%s, domain_metrics=%s)",
            cond.name,
            bool(cond.dice_men),
            bool(cond.domain_metrics),
        )

    exp.conditions = _sort_conditions(conditions)
    logger.info(
        "Loaded experiment %s: %d conditions (%s, semantic=%s)",
        name,
        len(exp.conditions),
        adapter_type,
        semantic_heads,
    )
    return exp


# ─────────────────────────────────────────────────────────────────────
# Experiment discovery
# ─────────────────────────────────────────────────────────────────────

# Mapping from mode to expected directory name prefixes
_MODE_PREFIXES = {
    "lora": ["lora_ablation_"],
    "dora": ["dora_ablation_"],
    "both": ["lora_ablation_", "dora_ablation_"],
}


def resolve_experiment_dirs(
    results_dir: Path,
    mode: str = "both",
    compare_semantic: bool = False,
) -> List[Tuple[str, Path]]:
    """Discover experiment directories matching the requested mode.

    Args:
        results_dir: Top-level results directory containing experiment subdirs.
        mode: 'lora', 'dora', or 'both'.
        compare_semantic: If True, include both semantic and no-semantic variants.

    Returns:
        List of (experiment_name, path) tuples.
    """
    prefixes = _MODE_PREFIXES.get(mode, _MODE_PREFIXES["both"])
    found: List[Tuple[str, Path]] = []

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or not (child / "conditions").exists():
            continue
        if any(child.name.startswith(p) for p in prefixes):
            found.append((child.name, child))

    if not compare_semantic:
        # Keep only semantic_heads variants (the primary ones)
        filtered = [(n, p) for n, p in found if "no_semantic" not in n]
        if filtered:
            found = filtered

    logger.info("Discovered %d experiment(s): %s", len(found), [n for n, _ in found])
    return found


def load_all_experiments(
    results_dir: Path,
    mode: str = "both",
    compare_semantic: bool = False,
) -> List[ExperimentData]:
    """Load all matching experiments from a results directory.

    Args:
        results_dir: Top-level results directory.
        mode: 'lora', 'dora', or 'both'.
        compare_semantic: Include no-semantic variants.

    Returns:
        List of loaded ExperimentData.
    """
    dirs = resolve_experiment_dirs(results_dir, mode, compare_semantic)
    experiments = []
    for name, path in dirs:
        exp = load_experiment(path)
        experiments.append(exp)
    return experiments


# ─────────────────────────────────────────────────────────────────────
# Feature loading (for UMAP)
# ─────────────────────────────────────────────────────────────────────


def load_features(condition_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load pre-extracted GLI and MEN feature tensors.

    Args:
        condition_dir: Path to condition directory.

    Returns:
        (gli_features, men_features) as numpy arrays, or None if missing.
    """
    import torch

    gli_path = condition_dir / "features_glioma.pt"
    men_path = condition_dir / "features_meningioma_subset.pt"

    if not gli_path.exists() or not men_path.exists():
        return None

    gli_data = torch.load(gli_path, map_location="cpu", weights_only=False)
    men_data = torch.load(men_path, map_location="cpu", weights_only=False)

    gli_feat = gli_data["features"]
    men_feat = men_data["features"]

    if hasattr(gli_feat, "numpy"):
        gli_feat = gli_feat.numpy()
    if hasattr(men_feat, "numpy"):
        men_feat = men_feat.numpy()

    return gli_feat, men_feat
