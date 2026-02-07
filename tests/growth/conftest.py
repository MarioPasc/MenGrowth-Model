# tests/growth/conftest.py
"""Shared fixtures for growth module tests."""

import tempfile
from pathlib import Path
from typing import Dict, Generator

import pytest
import torch
import yaml


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict() -> Dict:
    """Provide a sample configuration dictionary."""
    return {
        "paths": {
            "checkpoint_dir": "/tmp/checkpoints",
            "data_root": "/tmp/data",
            "cache_dir": "/tmp/cache",
            "output_dir": "/tmp/output",
        },
        "data": {
            "modalities": ["t1c", "t1n", "t2f", "t2w"],
            "roi_size": [128, 128, 128],
            "spacing": [1.0, 1.0, 1.0],
            "orientation": "RAS",
            "batch_size": 4,
            "num_workers": 2,
            "val_split": 0.1,
            "test_split": 0.1,
            "persistent_cache": False,
        },
        "encoder": {
            "fold": 0,
            "feature_size": 48,
            "feature_level": "encoder10",
            "feature_dim": 768,
            "freeze": True,
            "use_checkpoint": False,
        },
        "train": {
            "seed": 42,
            "precision": "32-true",
            "accelerator": "cpu",
            "devices": 1,
            "deterministic": False,
            "gradient_clip_val": 1.0,
        },
        "logging": {
            "save_dir": "/tmp/logs",
            "log_every_n_steps": 50,
            "tensorboard": {"enabled": True, "name": "test"},
            "csv": {"enabled": True},
            "checkpointing": {
                "save_top_k": 1,
                "monitor": "val/loss",
                "mode": "min",
                "save_last": True,
            },
        },
    }


@pytest.fixture
def sample_config_file(temp_dir: Path, sample_config_dict: Dict) -> Path:
    """Create a sample config file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def sample_state_dict() -> Dict[str, torch.Tensor]:
    """Provide a sample state dict mimicking BrainSegFounder structure."""
    state_dict = {}

    # swinViT keys (simplified)
    state_dict["swinViT.patch_embed.proj.weight"] = torch.randn(48, 4, 2, 2, 2)
    state_dict["swinViT.patch_embed.proj.bias"] = torch.randn(48)
    state_dict["swinViT.layers1.0.weight"] = torch.randn(48, 48)
    state_dict["swinViT.layers2.0.weight"] = torch.randn(96, 96)
    state_dict["swinViT.layers3.0.weight"] = torch.randn(192, 192)
    state_dict["swinViT.layers4.0.weight"] = torch.randn(384, 384)

    # Encoder keys
    state_dict["encoder1.conv.weight"] = torch.randn(48, 48, 3, 3, 3)
    state_dict["encoder2.conv.weight"] = torch.randn(96, 96, 3, 3, 3)
    state_dict["encoder3.conv.weight"] = torch.randn(192, 192, 3, 3, 3)
    state_dict["encoder4.conv.weight"] = torch.randn(384, 384, 3, 3, 3)
    state_dict["encoder10.conv.weight"] = torch.randn(768, 384, 3, 3, 3)

    # Decoder keys (should be filtered out)
    state_dict["decoder1.conv.weight"] = torch.randn(48, 96, 3, 3, 3)
    state_dict["decoder2.conv.weight"] = torch.randn(96, 192, 3, 3, 3)
    state_dict["decoder3.conv.weight"] = torch.randn(192, 384, 3, 3, 3)
    state_dict["decoder4.conv.weight"] = torch.randn(384, 768, 3, 3, 3)
    state_dict["decoder5.conv.weight"] = torch.randn(768, 768, 3, 3, 3)
    state_dict["out.conv.weight"] = torch.randn(3, 48, 1, 1, 1)

    return state_dict


@pytest.fixture
def sample_checkpoint(temp_dir: Path, sample_state_dict: Dict[str, torch.Tensor]) -> Path:
    """Create a sample checkpoint file."""
    ckpt_path = temp_dir / "checkpoint.pt"
    checkpoint = {
        "state_dict": sample_state_dict,
        "epoch": 100,
        "best_acc": 0.85,
    }
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


@pytest.fixture
def real_checkpoint_path() -> Path:
    """Path to real BrainSegFounder checkpoint (skip if not available)."""
    path = Path(
        "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/"
        "checkpoints/BrainSegFounder_finetuned_BraTS/finetuned_model_fold_0.pt"
    )
    if not path.exists():
        pytest.skip(f"Real checkpoint not available at {path}")
    return path


@pytest.fixture
def real_data_path() -> Path:
    """Path to real BraTS-MEN data (skip if not available)."""
    path = Path("/media/mpascual/PortableSSD/Meningiomas/BraTS/BraTS_Men_Train")
    if not path.exists():
        pytest.skip(f"Real data not available at {path}")
    return path
