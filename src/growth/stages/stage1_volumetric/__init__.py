# src/growth/stages/stage1_volumetric/__init__.py
"""Stage 1: Segmentation-based volumetric growth prediction (PRIMARY).

Facade re-exporting growth models configured for scalar (D=1) volume
trajectories, plus the Stage 1-specific Gompertz mean function.

The underlying models live in ``growth.models.growth`` and are stage-agnostic.
This package groups what Stage 1 needs and adds Stage 1-specific functionality.

Spec: ``docs/stages/stage_1_volumetric_baseline.md``
"""

from growth.stages.stage1_volumetric.gompertz import GompertzMeanFunction, fit_gompertz

from growth.models.growth.hgp_model import HierarchicalGPModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.models.growth.scalar_gp import ScalarGP

__all__ = [
    "GompertzMeanFunction",
    "HierarchicalGPModel",
    "LMEGrowthModel",
    "ScalarGP",
    "fit_gompertz",
]
