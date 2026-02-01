# tests/growth/test_config.py
"""Tests for configuration utilities."""

from pathlib import Path
from typing import Dict

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from growth.utils.config import (
    ConfigError,
    get_checkpoint_path,
    load_config,
    to_dict,
    validate_config,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, sample_config_file: Path):
        """Test loading a valid configuration file."""
        cfg = load_config(sample_config_file)

        assert isinstance(cfg, DictConfig)
        assert cfg.paths.checkpoint_dir == "/tmp/checkpoints"
        assert cfg.data.batch_size == 4
        assert cfg.encoder.fold == 0

    def test_load_config_with_overrides(self, sample_config_file: Path):
        """Test loading config with CLI overrides."""
        cfg = load_config(
            sample_config_file,
            overrides=["data.batch_size=8", "encoder.fold=2"],
        )

        assert cfg.data.batch_size == 8
        assert cfg.encoder.fold == 2

    def test_load_config_file_not_found(self, temp_dir: Path):
        """Test error on missing config file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(temp_dir / "nonexistent.yaml")

    def test_load_config_invalid_yaml(self, temp_dir: Path):
        """Test error on invalid YAML syntax."""
        bad_config = temp_dir / "bad.yaml"
        bad_config.write_text("invalid: yaml: content: [")

        with pytest.raises(ConfigError, match="Failed to load"):
            load_config(bad_config)

    def test_load_config_resolves_interpolations(self, temp_dir: Path):
        """Test that OmegaConf interpolations are resolved."""
        config_content = {
            "paths": {
                "base": "/data",
                "cache": "${paths.base}/cache",
            },
            "data": {"modalities": ["t1c"]},
            "encoder": {"fold": 0, "feature_size": 48, "feature_dim": 768},
            "train": {"seed": 42},
        }
        config_path = temp_dir / "interp.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        cfg = load_config(config_path, resolve=True)
        assert cfg.paths.cache == "/data/cache"


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_valid_config(self, sample_config_file: Path):
        """Test validation passes for valid config."""
        cfg = load_config(sample_config_file)
        # Should not raise
        validate_config(cfg)

    def test_validate_missing_required_field(self, temp_dir: Path):
        """Test validation fails for missing required field."""
        incomplete_config = {
            "paths": {"checkpoint_dir": "/tmp"},
            # Missing data, encoder, train sections
        }
        config_path = temp_dir / "incomplete.yaml"
        with open(config_path, "w") as f:
            yaml.dump(incomplete_config, f)

        cfg = load_config(config_path)
        with pytest.raises(ConfigError, match="Missing required"):
            validate_config(cfg)

    def test_validate_wrong_type(self, temp_dir: Path):
        """Test validation fails for wrong type."""
        bad_type_config = {
            "paths": {"checkpoint_dir": "/tmp", "data_root": "/tmp"},
            "data": {
                "modalities": ["t1c"],
                "roi_size": "should_be_list",  # Wrong type
                "batch_size": 4,
            },
            "encoder": {"fold": 0, "feature_size": 48, "feature_dim": 768},
            "train": {"seed": 42},
        }
        config_path = temp_dir / "bad_type.yaml"
        with open(config_path, "w") as f:
            yaml.dump(bad_type_config, f)

        cfg = load_config(config_path)
        with pytest.raises(ConfigError, match="type"):
            validate_config(cfg)


class TestGetCheckpointPath:
    """Tests for get_checkpoint_path function."""

    def test_get_checkpoint_path_default_fold(self, sample_config_file: Path, temp_dir: Path):
        """Test checkpoint path construction with default fold."""
        cfg = load_config(sample_config_file)
        # Override to use temp_dir
        cfg.paths.checkpoint_dir = str(temp_dir)

        # Create the expected checkpoint file
        expected_path = temp_dir / "finetuned_model_fold_0.pt"
        expected_path.touch()

        path = get_checkpoint_path(cfg)
        assert path == expected_path

    def test_get_checkpoint_path_custom_fold(self, sample_config_file: Path, temp_dir: Path):
        """Test checkpoint path construction with custom fold."""
        cfg = load_config(sample_config_file)
        cfg.paths.checkpoint_dir = str(temp_dir)

        # Create fold 2 checkpoint
        expected_path = temp_dir / "finetuned_model_fold_2.pt"
        expected_path.touch()

        path = get_checkpoint_path(cfg, fold=2)
        assert path == expected_path

    def test_get_checkpoint_path_not_found(self, sample_config_file: Path, temp_dir: Path):
        """Test error when checkpoint doesn't exist."""
        cfg = load_config(sample_config_file)
        cfg.paths.checkpoint_dir = str(temp_dir)
        # Don't create the file

        with pytest.raises(FileNotFoundError, match="fold_0"):
            get_checkpoint_path(cfg)


class TestToDict:
    """Tests for to_dict function."""

    def test_to_dict_conversion(self, sample_config_file: Path):
        """Test conversion to plain dict."""
        cfg = load_config(sample_config_file)
        result = to_dict(cfg)

        assert isinstance(result, dict)
        assert not isinstance(result, DictConfig)
        assert result["data"]["batch_size"] == 4

    def test_to_dict_nested_conversion(self, sample_config_file: Path):
        """Test nested structures are also converted."""
        cfg = load_config(sample_config_file)
        result = to_dict(cfg)

        # Check nested dicts are plain dicts
        assert isinstance(result["paths"], dict)
        assert isinstance(result["data"], dict)
        assert isinstance(result["logging"]["tensorboard"], dict)

    def test_to_dict_lists_preserved(self, sample_config_file: Path):
        """Test lists are preserved as lists."""
        cfg = load_config(sample_config_file)
        result = to_dict(cfg)

        assert isinstance(result["data"]["modalities"], list)
        assert result["data"]["modalities"] == ["t1c", "t1n", "t2f", "t2w"]
