# experiments/lora_ablation/__init__.py
"""LoRA ablation experiment for meningioma encoder adaptation.

This unified experiment compares:
- Baseline (frozen encoder with configurable decoder)
- LoRA rank 2, 4, 8, 16, 32 adaptations

Decoder types (via decoder_type config):
- "lightweight": Custom SegmentationHead (~2M params)
- "original": Full SwinUNETR decoder (~30M params, recommended)

Primary metric: Linear probe R² for semantic features.
Secondary metrics: MLP probe R², Segmentation Dice.

Usage:
    growth-exp-lora-ablation run-all --config path/to/ablation.yaml
"""

from __future__ import annotations

import importlib
import importlib.abc
import sys
import types

# ── Backward-compatibility shims ──────────────────────────────────
# SLURM scripts and external callers may use the old flat import paths
# e.g. ``from experiments.lora_ablation.model_factory import ...``
# We install a custom meta-path finder that lazily redirects these
# to the new subpackage locations on first access.

_PKG = "experiments.lora_ablation"

_MOVED: dict[str, str] = {}

for _name in (
    "data_splits", "evaluate_dice", "evaluate_feature_quality",
    "evaluate_probes", "extract_domain_features", "extract_features",
    "model_factory", "output_paths", "train_condition",
):
    _MOVED[f"{_PKG}.{_name}"] = f"{_PKG}.pipeline.{_name}"

for _name in (
    "analyze_results", "compute_domain_metrics", "domain_visualizations",
    "enhanced_diagnostics", "generate_tables", "regenerate_analysis",
    "statistical_analysis", "v3_cache", "v3_figures", "visualizations",
):
    _MOVED[f"{_PKG}.{_name}"] = f"{_PKG}.analysis.{_name}"

for _name in (
    "diagnose_frozen_gli", "merge_lora_checkpoint", "post_hoc_analysis",
):
    _MOVED[f"{_PKG}.{_name}"] = f"{_PKG}.scripts.{_name}"


class _CompatFinder(importlib.abc.MetaPathFinder):
    """Redirect old flat imports to new subpackage paths."""

    def find_module(
        self, fullname: str, path: object = None,
    ) -> _CompatFinder | None:
        if fullname in _MOVED:
            return self
        return None

    def load_module(self, fullname: str) -> types.ModuleType:
        if fullname in sys.modules:
            return sys.modules[fullname]
        real = importlib.import_module(_MOVED[fullname])
        sys.modules[fullname] = real
        return real


sys.meta_path.insert(0, _CompatFinder())
