# tests/growth/test_lora_checkpoint_loading.py
"""Tests for PEFT/LoRA checkpoint loading and weight merging."""

import numpy as np
import pytest
import torch

from experiments.stage1_volumetric.segment import (
    _merge_lora_weights,
    _strip_training_checkpoint_prefix,
)

pytestmark = [pytest.mark.phase1, pytest.mark.unit]


class TestPEFTPrefixStripping:
    def test_standard_prefix_stripping(self) -> None:
        """Standard training checkpoint keys are correctly stripped."""
        state_dict = {
            "model.encoder.swinViT.layers1.0.blocks.0.attn.qkv.weight": torch.randn(3, 3),
            "model.decoder.decoder5.layer.0.conv1.conv.weight": torch.randn(3, 3),
        }
        stripped = _strip_training_checkpoint_prefix(state_dict)
        assert "swinViT.layers1.0.blocks.0.attn.qkv.weight" in stripped
        assert "decoder5.layer.0.conv1.conv.weight" in stripped

    def test_peft_prefix_stripping(self) -> None:
        """PEFT-wrapped keys are correctly stripped to SwinUNETR keys."""
        state_dict = {
            "lora_encoder.model.base_model.model.swinViT.layers3.0.blocks.0.attn.qkv.weight": torch.randn(3, 3),
            "lora_encoder.model.base_model.model.encoder10.layer.0.conv1.conv.weight": torch.randn(3, 3),
            "lora_encoder.model.base_model.model.decoder5.layer.0.weight": torch.randn(3, 3),
            "lora_encoder.model.base_model.model.out.conv.conv.weight": torch.randn(3, 3),
        }
        stripped = _strip_training_checkpoint_prefix(state_dict)
        assert "swinViT.layers3.0.blocks.0.attn.qkv.weight" in stripped
        assert "encoder10.layer.0.conv1.conv.weight" in stripped
        assert "decoder5.layer.0.weight" in stripped
        assert "out.conv.conv.weight" in stripped
        assert len(stripped) == 4

    def test_unrecognized_keys_skipped(self) -> None:
        """Keys that don't match any prefix are silently skipped."""
        state_dict = {
            "semantic_heads.vol_head.weight": torch.randn(3, 3),
            "model.encoder.swinViT.patch_embed.weight": torch.randn(3, 3),
        }
        stripped = _strip_training_checkpoint_prefix(state_dict)
        assert "swinViT.patch_embed.weight" in stripped
        assert len(stripped) == 1  # semantic_heads skipped


class TestLoRAWeightMerging:
    def test_basic_merge(self) -> None:
        """LoRA weights are correctly merged: W = W + (alpha/r) * B @ A."""
        alpha, r = 16, 8
        in_feat, out_feat = 10, 10

        W_base = torch.randn(out_feat, in_feat)
        A = torch.randn(r, in_feat)
        B = torch.randn(out_feat, r)

        state_dict = {
            "layer.weight": W_base.clone(),
            "layer.lora_A.default.weight": A.clone(),
            "layer.lora_B.default.weight": B.clone(),
        }

        merged = _merge_lora_weights(state_dict, lora_alpha=alpha, lora_rank=r)

        expected = W_base + (alpha / r) * (B @ A)
        torch.testing.assert_close(merged["layer.weight"], expected)

        # LoRA keys should be removed
        assert "layer.lora_A.default.weight" not in merged
        assert "layer.lora_B.default.weight" not in merged

    def test_multiple_layers(self) -> None:
        """Multiple LoRA adapters merged independently."""
        state_dict = {
            "layer1.weight": torch.randn(8, 8),
            "layer1.lora_A.default.weight": torch.randn(4, 8),
            "layer1.lora_B.default.weight": torch.randn(8, 4),
            "layer2.weight": torch.randn(6, 6),
            "layer2.lora_A.default.weight": torch.randn(4, 6),
            "layer2.lora_B.default.weight": torch.randn(6, 4),
        }

        merged = _merge_lora_weights(state_dict, lora_alpha=16, lora_rank=4)
        assert "layer1.weight" in merged
        assert "layer2.weight" in merged
        assert len(merged) == 2  # Only base weights remain

    def test_no_lora_keys_passthrough(self) -> None:
        """State dict without LoRA keys is returned unchanged."""
        state_dict = {
            "layer.weight": torch.randn(5, 5),
            "layer.bias": torch.randn(5),
        }
        merged = _merge_lora_weights(dict(state_dict))
        assert len(merged) == 2
        torch.testing.assert_close(merged["layer.weight"], state_dict["layer.weight"])

    def test_scaling_factor(self) -> None:
        """Verify that scaling = alpha / rank is applied correctly."""
        W = torch.zeros(4, 4)
        A = torch.ones(2, 4)
        B = torch.ones(4, 2)

        state_dict = {
            "x.weight": W.clone(),
            "x.lora_A.default.weight": A.clone(),
            "x.lora_B.default.weight": B.clone(),
        }

        merged = _merge_lora_weights(state_dict, lora_alpha=8, lora_rank=4)
        # scaling = 8/4 = 2.0, B@A = ones(4,2)@ones(2,4) = 2*ones(4,4)
        # W_merged = 0 + 2.0 * 2*ones = 4*ones
        expected = torch.full((4, 4), 4.0)
        torch.testing.assert_close(merged["x.weight"], expected)
