# src/growth/utils/config.py
"""OmegaConf configuration loading and validation utilities.

This module provides:
- Config loading from YAML with optional CLI overrides
- Schema validation for required fields
- Path resolution and environment variable expansion
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf, DictConfig, MissingMandatoryValue

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration validation error."""

    pass


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[List[str]] = None,
    resolve: bool = True,
) -> DictConfig:
    """Load configuration from YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML configuration file.
        overrides: List of CLI overrides in "key=value" format.
            Example: ["train.seed=123", "data.batch_size=8"]
        resolve: If True, resolve interpolations (${...}).

    Returns:
        OmegaConf DictConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ConfigError: If config loading fails.

    Example:
        >>> cfg = load_config("config.yaml", overrides=["train.seed=42"])
        >>> print(cfg.train.seed)
        42
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # Load base config
        cfg = OmegaConf.load(config_path)
        logger.info(f"Loaded config from: {config_path}")

        # Apply CLI overrides
        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)
            logger.info(f"Applied {len(overrides)} config overrides")

        # Resolve interpolations
        if resolve:
            OmegaConf.resolve(cfg)

        return cfg

    except Exception as e:
        raise ConfigError(f"Failed to load config from {config_path}: {e}") from e


def validate_config(
    cfg: DictConfig,
    schema: Optional[Dict[str, type]] = None,
) -> None:
    """Validate configuration against schema.

    Args:
        cfg: Configuration to validate.
        schema: Optional schema dict mapping dotted paths to expected types.
            If None, uses default foundation schema.

    Raises:
        ConfigError: If validation fails with detailed error messages.

    Example:
        >>> validate_config(cfg)  # Uses default schema
        >>> validate_config(cfg, {"custom.field": str})  # Custom schema
    """
    if schema is None:
        schema = _get_default_schema()

    errors = []
    for path, expected_type in schema.items():
        try:
            value = OmegaConf.select(cfg, path)
            if value is None:
                errors.append(f"Missing required field: {path}")
            elif expected_type is not None:
                # Handle list and tuple types specially
                if expected_type == list and not isinstance(value, (list, tuple)):
                    # OmegaConf returns ListConfig, check if it's list-like
                    if not hasattr(value, "__iter__") or isinstance(value, (str, dict)):
                        errors.append(
                            f"Invalid type for {path}: expected list, "
                            f"got {type(value).__name__}"
                        )
                elif expected_type not in (list,) and not isinstance(value, expected_type):
                    errors.append(
                        f"Invalid type for {path}: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        except MissingMandatoryValue:
            errors.append(f"Missing required field: {path}")
        except Exception as e:
            errors.append(f"Error validating {path}: {e}")

    if errors:
        error_msg = "Configuration validation failed:\n  " + "\n  ".join(errors)
        raise ConfigError(error_msg)

    logger.info("Configuration validation passed")


def _get_default_schema() -> Dict[str, type]:
    """Get default schema for foundation config.

    Returns:
        Dictionary mapping dotted paths to expected types.
    """
    return {
        # Required paths
        "paths.checkpoint_dir": str,
        "paths.data_root": str,
        # Encoder settings
        "encoder.fold": int,
        "encoder.feature_size": int,
        "encoder.feature_dim": int,
        # Data settings
        "data.modalities": list,
        "data.roi_size": list,
        "data.batch_size": int,
        # Train settings
        "train.seed": int,
    }


def get_checkpoint_path(
    cfg: DictConfig,
    fold: Optional[int] = None,
) -> Path:
    """Get full path to checkpoint for specified fold.

    Args:
        cfg: Configuration with paths.checkpoint_dir and encoder.fold.
        fold: Optional fold override. If None, uses cfg.encoder.fold.

    Returns:
        Path to checkpoint file.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist, with list of available folds.

    Example:
        >>> path = get_checkpoint_path(cfg)
        >>> path = get_checkpoint_path(cfg, fold=2)  # Override fold
    """
    fold = fold if fold is not None else cfg.encoder.fold
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_path = ckpt_dir / f"finetuned_model_fold_{fold}.pt"

    if not ckpt_path.exists():
        # Find available folds for helpful error message
        available = list(ckpt_dir.glob("finetuned_model_fold_*.pt"))
        available_folds = [p.stem.split("_")[-1] for p in available]
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Available folds in {ckpt_dir}: {available_folds}"
        )

    logger.info(f"Using checkpoint: {ckpt_path}")
    return ckpt_path


def to_dict(cfg: DictConfig, resolve: bool = True) -> Dict[str, Any]:
    """Convert OmegaConf DictConfig to plain Python dict.

    Args:
        cfg: OmegaConf DictConfig to convert.
        resolve: If True, resolve interpolations before converting.

    Returns:
        Plain Python dictionary.

    Example:
        >>> plain_dict = to_dict(cfg)
        >>> isinstance(plain_dict, dict)
        True
    """
    return OmegaConf.to_container(cfg, resolve=resolve)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configs with later configs taking precedence.

    Args:
        *configs: Variable number of DictConfig objects to merge.

    Returns:
        Merged DictConfig.

    Example:
        >>> base = load_config("base.yaml")
        >>> override = load_config("override.yaml")
        >>> merged = merge_configs(base, override)
    """
    return OmegaConf.merge(*configs)


def save_config(
    cfg: DictConfig,
    save_path: Union[str, Path],
    resolve: bool = True,
) -> None:
    """Save configuration to YAML file.

    Args:
        cfg: Configuration to save.
        save_path: Path to save YAML file.
        resolve: If True, resolve interpolations before saving.

    Example:
        >>> save_config(cfg, "saved_config.yaml")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        OmegaConf.save(cfg, f, resolve=resolve)

    logger.info(f"Saved config to: {save_path}")


def get_value(
    cfg: DictConfig,
    path: str,
    default: Any = None,
) -> Any:
    """Safely get a nested config value with default.

    Args:
        cfg: Configuration object.
        path: Dotted path to value (e.g., "train.lr").
        default: Default value if path doesn't exist.

    Returns:
        Config value or default.

    Example:
        >>> lr = get_value(cfg, "train.lr", default=1e-4)
    """
    try:
        value = OmegaConf.select(cfg, path)
        return value if value is not None else default
    except Exception:
        return default
