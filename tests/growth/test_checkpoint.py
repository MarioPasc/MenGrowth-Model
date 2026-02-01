# tests/growth/test_checkpoint.py
"""Tests for checkpoint utilities."""

from pathlib import Path
from typing import Dict

import pytest
import torch

from growth.utils.checkpoint import (
    DECODER_PREFIXES,
    ENCODER_PREFIXES,
    extract_encoder_weights,
    get_checkpoint_stats,
    load_checkpoint,
    merge_lora_weights,
    print_checkpoint_summary,
)


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_load_checkpoint_with_state_dict_wrapper(self, sample_checkpoint: Path):
        """Test loading checkpoint with standard format."""
        result = load_checkpoint(sample_checkpoint)

        assert "state_dict" in result
        assert "epoch" in result
        assert result["epoch"] == 100
        assert result["best_acc"] == 0.85

    def test_load_raw_state_dict(self, temp_dir: Path):
        """Test loading checkpoint that's just a state dict."""
        raw_path = temp_dir / "raw.pt"
        state_dict = {"layer.weight": torch.randn(10, 10)}
        torch.save(state_dict, raw_path)

        result = load_checkpoint(raw_path)

        assert "state_dict" in result
        assert "layer.weight" in result["state_dict"]
        assert result["epoch"] is None

    def test_load_checkpoint_not_found(self, temp_dir: Path):
        """Test error on missing checkpoint."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_checkpoint(temp_dir / "missing.pt")

    def test_load_checkpoint_map_location(self, sample_checkpoint: Path):
        """Test checkpoint loading with different map_location."""
        result = load_checkpoint(sample_checkpoint, map_location="cpu")
        assert all(
            t.device.type == "cpu" for t in result["state_dict"].values()
        )


class TestExtractEncoderWeights:
    """Tests for extract_encoder_weights function."""

    def test_extract_encoder_weights_default(
        self, sample_state_dict: Dict[str, torch.Tensor]
    ):
        """Test extracting encoder weights with defaults."""
        result = extract_encoder_weights(sample_state_dict)

        # Should include encoder keys
        assert any(k.startswith("swinViT") for k in result)
        assert any(k.startswith("encoder1") for k in result)
        assert any(k.startswith("encoder10") for k in result)

        # Should exclude decoder keys
        assert not any(k.startswith("decoder") for k in result)
        assert not any(k.startswith("out") for k in result)

    def test_extract_encoder_weights_without_encoder10(
        self, sample_state_dict: Dict[str, torch.Tensor]
    ):
        """Test extracting encoder weights without encoder10."""
        result = extract_encoder_weights(sample_state_dict, include_encoder10=False)

        # Should NOT include encoder10
        assert not any(k.startswith("encoder10") for k in result)

        # Should still include other encoder keys
        assert any(k.startswith("swinViT") for k in result)
        assert any(k.startswith("encoder1") for k in result)

    def test_extract_encoder_weights_strict_no_keys(self):
        """Test strict mode raises error when no encoder keys found."""
        empty_state_dict = {"random.key": torch.randn(10)}

        with pytest.raises(ValueError, match="No encoder keys"):
            extract_encoder_weights(empty_state_dict, strict=True)

    def test_extract_encoder_weights_non_strict(self):
        """Test non-strict mode returns empty dict for no encoder keys."""
        empty_state_dict = {"random.key": torch.randn(10)}

        result = extract_encoder_weights(empty_state_dict, strict=False)
        assert len(result) == 0

    def test_encoder_prefixes_coverage(self, sample_state_dict: Dict[str, torch.Tensor]):
        """Test that all expected encoder prefixes are handled."""
        result = extract_encoder_weights(sample_state_dict)

        # All ENCODER_PREFIXES should be extracted
        extracted_prefixes = set(k.split(".")[0] for k in result)
        for prefix in ENCODER_PREFIXES:
            # Check if any key with this prefix was in the original
            if any(k.startswith(prefix) for k in sample_state_dict):
                assert prefix in extracted_prefixes

    def test_decoder_prefixes_excluded(self, sample_state_dict: Dict[str, torch.Tensor]):
        """Test that all decoder prefixes are excluded."""
        result = extract_encoder_weights(sample_state_dict)

        for prefix in DECODER_PREFIXES:
            assert not any(k.startswith(prefix) for k in result)


class TestGetCheckpointStats:
    """Tests for get_checkpoint_stats function."""

    def test_get_checkpoint_stats_basic(
        self, sample_state_dict: Dict[str, torch.Tensor]
    ):
        """Test checkpoint statistics computation."""
        stats = get_checkpoint_stats(sample_state_dict)

        assert "swinViT" in stats
        assert "encoder1" in stats
        assert "decoder1" in stats

        # Check stats structure
        for prefix_stats in stats.values():
            assert "count" in prefix_stats
            assert "params" in prefix_stats
            assert "params_m" in prefix_stats
            assert "shapes" in prefix_stats

    def test_get_checkpoint_stats_param_count(self):
        """Test parameter counting is correct."""
        state_dict = {
            "layer1.weight": torch.randn(100, 100),  # 10000 params
            "layer1.bias": torch.randn(100),  # 100 params
        }

        stats = get_checkpoint_stats(state_dict)

        assert stats["layer1"]["count"] == 2
        assert stats["layer1"]["params"] == 10100
        assert abs(stats["layer1"]["params_m"] - 0.0101) < 0.0001


class TestMergeLoraWeights:
    """Tests for merge_lora_weights function."""

    def test_merge_lora_weights_basic(self):
        """Test basic LoRA weight merging."""
        # Base weight
        base_state_dict = {
            "layer.weight": torch.zeros(10, 10),
        }

        # LoRA weights (rank=2)
        lora_state_dict = {
            "layer.weight.lora_A": torch.ones(2, 10),  # [rank, in_features]
            "layer.weight.lora_B": torch.ones(10, 2),  # [out_features, rank]
        }

        result = merge_lora_weights(base_state_dict, lora_state_dict, alpha=1.0)

        # B @ A = [10, 2] @ [2, 10] = [10, 10] with all elements = 2
        expected = torch.ones(10, 10) * 2
        assert torch.allclose(result["layer.weight"], expected)

    def test_merge_lora_weights_with_alpha(self):
        """Test LoRA merging with scaling factor."""
        base_state_dict = {
            "layer.weight": torch.zeros(10, 10),
        }

        lora_state_dict = {
            "layer.weight.lora_A": torch.ones(2, 10),
            "layer.weight.lora_B": torch.ones(10, 2),
        }

        result = merge_lora_weights(base_state_dict, lora_state_dict, alpha=0.5)

        # alpha * (B @ A) = 0.5 * 2 = 1
        expected = torch.ones(10, 10)
        assert torch.allclose(result["layer.weight"], expected)

    def test_merge_lora_weights_preserves_non_lora(self):
        """Test that non-LoRA weights are preserved."""
        base_state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(5, 5),
        }

        # Only LoRA for layer1
        lora_state_dict = {
            "layer1.weight.lora_A": torch.zeros(2, 10),
            "layer1.weight.lora_B": torch.zeros(10, 2),
        }

        result = merge_lora_weights(base_state_dict, lora_state_dict, alpha=1.0)

        # layer2 should be unchanged
        assert torch.allclose(result["layer2.weight"], base_state_dict["layer2.weight"])

    def test_merge_lora_weights_missing_base_key(self):
        """Test warning when base key not found."""
        base_state_dict = {}  # Empty

        lora_state_dict = {
            "layer.weight.lora_A": torch.ones(2, 10),
            "layer.weight.lora_B": torch.ones(10, 2),
        }

        # Should not raise, just log warning
        result = merge_lora_weights(base_state_dict, lora_state_dict, alpha=1.0)
        assert len(result) == 0


class TestPrintCheckpointSummary:
    """Tests for print_checkpoint_summary function."""

    def test_print_checkpoint_summary(
        self, sample_state_dict: Dict[str, torch.Tensor], capsys
    ):
        """Test checkpoint summary printing."""
        print_checkpoint_summary(sample_state_dict)

        captured = capsys.readouterr()
        assert "Checkpoint Summary" in captured.out
        assert "swinViT" in captured.out
        assert "TOTAL" in captured.out


class TestRealCheckpoint:
    """Tests with real BrainSegFounder checkpoint (skipped if unavailable)."""

    def test_load_real_checkpoint(self, real_checkpoint_path: Path):
        """Test loading real BrainSegFounder checkpoint."""
        result = load_checkpoint(real_checkpoint_path)

        assert "state_dict" in result
        state_dict = result["state_dict"]

        # Check expected prefixes exist
        prefixes = set(k.split(".")[0] for k in state_dict)
        assert "swinViT" in prefixes

    def test_extract_real_encoder_weights(self, real_checkpoint_path: Path):
        """Test extracting encoder weights from real checkpoint."""
        checkpoint = load_checkpoint(real_checkpoint_path)
        state_dict = checkpoint["state_dict"]

        encoder_weights = extract_encoder_weights(state_dict)

        # Should have significant number of keys
        assert len(encoder_weights) > 100

        # No decoder keys
        assert not any(k.startswith("decoder") for k in encoder_weights)

    def test_real_checkpoint_stats(self, real_checkpoint_path: Path):
        """Test statistics from real checkpoint."""
        checkpoint = load_checkpoint(real_checkpoint_path)
        state_dict = checkpoint["state_dict"]

        stats = get_checkpoint_stats(state_dict)

        # swinViT should have most parameters
        assert stats["swinViT"]["params_m"] > 5.0  # > 5M params
