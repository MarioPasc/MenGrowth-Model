# experiments/stage1_volumetric/engine/model_registry.py
"""Build growth model configurations from YAML config.

Maps model display names to (model_class, kwargs) pairs. All model
class imports are centralised here so that runner.py and entrypoints
remain model-agnostic.
"""

from __future__ import annotations

import logging

from growth.models.growth.hgp_hetero import HGPHeteroModel
from growth.models.growth.hgp_model import HierarchicalGPModel
from growth.models.growth.lme_hetero import LMEHeteroGrowthModel
from growth.models.growth.lme_model import LMEGrowthModel
from growth.models.growth.nlme_analytical import (
    ExponentialNLME,
    GompertzNLME,
    LogisticNLME,
)
from growth.models.growth.scalar_gp import ScalarGP
from growth.models.growth.scalar_gp_hetero import ScalarGPHetero

logger = logging.getLogger(__name__)


def build_model_configs(cfg: dict) -> dict[str, tuple[type, dict]]:
    """Build model class + kwargs mapping from config.

    Args:
        cfg: Parsed YAML config dict.

    Returns:
        Dict mapping model display name to (model_class, kwargs).
    """
    gp_cfg = cfg["gp"]
    lme_cfg = cfg.get("lme", {})
    uq_cfg = cfg.get("uncertainty", {})
    seed = cfg["experiment"]["seed"]
    models_cfg = cfg.get("models", {})

    cov_cfg = cfg.get("covariates", {})
    cov_enabled = cov_cfg.get("enabled", False)
    cov_features = cov_cfg.get("features", [])
    cov_missing = cov_cfg.get("missing_strategy", "skip")

    cov_kwargs: dict = {}
    if cov_enabled:
        cov_kwargs = {
            "use_covariates": True,
            "covariate_names": cov_features,
            "missing_strategy": cov_missing,
        }

    floor_var = uq_cfg.get("floor_variance", 1e-6)

    shared_gp_kwargs = {
        "n_restarts": gp_cfg["n_restarts"],
        "max_iter": gp_cfg["max_iter"],
        "lengthscale_bounds": tuple(gp_cfg["lengthscale_bounds"]),
        "signal_var_bounds": tuple(gp_cfg["signal_var_bounds"]),
        "noise_var_bounds": tuple(gp_cfg["noise_var_bounds"]),
        "seed": seed,
        **cov_kwargs,
    }

    models: dict[str, tuple[type, dict]] = {}

    # --- Homoscedastic models ---
    if models_cfg.get("scalar_gp", True):
        models["ScalarGP"] = (
            ScalarGP,
            {
                "kernel_type": gp_cfg["kernel"],
                "mean_function": gp_cfg.get("mean_function", "linear"),
                **shared_gp_kwargs,
            },
        )

    if models_cfg.get("lme", True):
        models["LME"] = (
            LMEGrowthModel,
            {"method": lme_cfg.get("method", "reml"), **cov_kwargs},
        )

    if models_cfg.get("hgp", True):
        models["HGP"] = (
            HierarchicalGPModel,
            {
                "kernel_type": gp_cfg["kernel"],
                "mean_function": "linear",
                **shared_gp_kwargs,
            },
        )

    if models_cfg.get("hgp_gompertz", False):
        models["HGP_Gompertz"] = (
            HierarchicalGPModel,
            {
                "kernel_type": gp_cfg["kernel"],
                "mean_function": "gompertz",
                **shared_gp_kwargs,
            },
        )

    # --- Heteroscedastic models ---
    if models_cfg.get("scalar_gp_hetero", True):
        models["ScalarGPHetero"] = (
            ScalarGPHetero,
            {
                "mean_function": gp_cfg.get("mean_function", "linear"),
                "floor_variance": floor_var,
                **shared_gp_kwargs,
            },
        )

    if models_cfg.get("lme_hetero", True):
        models["LMEHetero"] = (
            LMEHeteroGrowthModel,
            {
                "method": lme_cfg.get("method", "reml"),
                "n_restarts": lme_cfg.get("n_restarts", 5),
                "max_iter": lme_cfg.get("max_iter", 1000),
                "seed": seed,
                "floor_variance": floor_var,
                **cov_kwargs,
            },
        )

    if models_cfg.get("hgp_hetero", True):
        models["HGPHetero"] = (
            HGPHeteroModel,
            {
                "mean_function": "linear",
                "floor_variance": floor_var,
                **shared_gp_kwargs,
            },
        )

    if models_cfg.get("hgp_gompertz_hetero", False):
        models["HGP_Gompertz_Hetero"] = (
            HGPHeteroModel,
            {
                "mean_function": "gompertz",
                "floor_variance": floor_var,
                **shared_gp_kwargs,
            },
        )

    # --- NLME analytical baselines ---
    analytical_cfg = cfg.get("analytical", {})
    if analytical_cfg.get("enabled", False):
        nlme_kwargs = {
            "n_restarts": analytical_cfg.get("n_restarts", 3),
            "max_iter": analytical_cfg.get("max_iter", 500),
            "seed": seed,
            "fallback_to_1re": analytical_cfg.get("fallback_to_1re", True),
        }
        nlme_models_cfg = analytical_cfg.get("models", {})
        if nlme_models_cfg.get("exponential", True):
            models["NLME_Exponential"] = (ExponentialNLME, {**nlme_kwargs})
        if nlme_models_cfg.get("logistic", True):
            models["NLME_Logistic"] = (LogisticNLME, {**nlme_kwargs})
        if nlme_models_cfg.get("gompertz", True):
            models["NLME_Gompertz"] = (GompertzNLME, {**nlme_kwargs})

    logger.info(f"Model registry: {len(models)} models configured — {list(models.keys())}")
    return models
