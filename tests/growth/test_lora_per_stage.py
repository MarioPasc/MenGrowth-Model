# tests/growth/test_lora_per_stage.py
"""Tests for per-stage module type overrides in LoRA adapter."""

import pytest
import torch

from growth.models.encoder.lora_adapter import (
    LoRASwinViT,
    MODULE_TYPE_SUFFIX,
    _find_lora_targets,
    _resolve_stage_for_module,
    count_lora_params,
)
from growth.models.encoder.swin_loader import create_swinunetr

pytestmark = [pytest.mark.phase1, pytest.mark.unit]


class TestResolveStageForModule:
    """Tests for the stage-resolution helper."""

    def test_layers1(self):
        assert _resolve_stage_for_module("swinViT.layers1.0.blocks.0.attn.qkv") == 1

    def test_layers4(self):
        assert _resolve_stage_for_module("layers4.0.blocks.1.mlp.linear2") == 4

    def test_no_match(self):
        assert _resolve_stage_for_module("encoder10.layer.0.conv") is None

    def test_patch_embed(self):
        assert _resolve_stage_for_module("swinViT.patch_embed.proj") is None


class TestFindLoraTargetsPerStage:
    """Tests for per-stage module type overrides in target discovery."""

    @pytest.fixture()
    def model(self):
        return create_swinunetr()

    def test_per_stage_override_restricts_stage1(self, model):
        """Stage 1 gets MLP-only, stages 2-4 get full block."""
        targets = _find_lora_targets(
            model,
            stages=[1, 2, 3, 4],
            module_types=["qkv", "proj", "fc1", "fc2"],
            per_stage_module_types={1: ["fc1", "fc2"]},
        )

        stage1_targets = [t for t in targets if "layers1" in t]
        stage2_targets = [t for t in targets if "layers2" in t]

        # Stage 1: 2 blocks × 2 MLP types = 4 modules
        assert len(stage1_targets) == 4
        assert all(
            t.endswith(".mlp.linear1") or t.endswith(".mlp.linear2")
            for t in stage1_targets
        )
        # No attention modules at stage 1
        assert not any(t.endswith(".attn.qkv") for t in stage1_targets)
        assert not any(t.endswith(".attn.proj") for t in stage1_targets)

        # Stage 2: 2 blocks × 4 types = 8 modules (no override)
        assert len(stage2_targets) == 8

    def test_per_stage_override_qkv_only_at_stage1(self, model):
        """Stage 1 gets QKV-only (R5 config)."""
        targets = _find_lora_targets(
            model,
            stages=[1, 2, 3, 4],
            module_types=["qkv", "proj", "fc1", "fc2"],
            per_stage_module_types={1: ["qkv"]},
        )

        stage1_targets = [t for t in targets if "layers1" in t]

        # Stage 1: 2 blocks × 1 type = 2 modules
        assert len(stage1_targets) == 2
        assert all(t.endswith(".attn.qkv") for t in stage1_targets)

    def test_no_override_matches_default(self, model):
        """per_stage_module_types=None produces identical results to the original."""
        targets_default = _find_lora_targets(
            model,
            stages=[1, 2, 3, 4],
            module_types=["qkv", "proj", "fc1", "fc2"],
        )
        targets_no_override = _find_lora_targets(
            model,
            stages=[1, 2, 3, 4],
            module_types=["qkv", "proj", "fc1", "fc2"],
            per_stage_module_types=None,
        )
        assert targets_default == targets_no_override

    def test_empty_dict_matches_default(self, model):
        """Empty per_stage_module_types dict behaves like None."""
        targets_default = _find_lora_targets(
            model,
            stages=[2, 3, 4],
            module_types=["qkv"],
        )
        targets_empty = _find_lora_targets(
            model,
            stages=[2, 3, 4],
            module_types=["qkv"],
            per_stage_module_types={},
        )
        assert targets_default == targets_empty

    def test_invalid_per_stage_type_raises(self, model):
        """Invalid module type in per-stage override raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            _find_lora_targets(
                model,
                stages=[1, 2],
                module_types=["qkv"],
                per_stage_module_types={1: ["invalid_type"]},
            )


class TestLoRASwinViTPerStage:
    """Integration tests for LoRASwinViT with per-stage overrides."""

    @pytest.fixture()
    def base_encoder(self):
        return create_swinunetr()

    def test_param_count_reduced_with_mlp_only_stage1(self, base_encoder):
        """MLP-only at stage 1 has fewer params than full block at stage 1."""
        lora_full = LoRASwinViT(
            create_swinunetr(),
            rank=8,
            target_stages=[1, 2, 3, 4],
            target_module_types=["qkv", "proj", "fc1", "fc2"],
        )
        lora_mlp_s1 = LoRASwinViT(
            base_encoder,
            rank=8,
            target_stages=[1, 2, 3, 4],
            target_module_types=["qkv", "proj", "fc1", "fc2"],
            per_stage_module_types={1: ["fc1", "fc2"]},
        )

        full_params = lora_full.get_trainable_params()
        mlp_s1_params = lora_mlp_s1.get_trainable_params()

        assert mlp_s1_params < full_params
        # Stage 1 with full block has 4 module types; MLP-only has 2.
        # So we lose exactly 2 modules × 2 blocks at stage 1.

    def test_per_stage_attribute_stored(self, base_encoder):
        """per_stage_module_types is stored correctly on the instance."""
        overrides = {1: ["fc1", "fc2"]}
        lora = LoRASwinViT(
            base_encoder,
            rank=4,
            target_stages=[1, 2],
            target_module_types=["qkv", "proj", "fc1", "fc2"],
            per_stage_module_types=overrides,
        )
        assert lora.per_stage_module_types == {1: ["fc1", "fc2"]}

    def test_save_load_roundtrip_reconstructs_overrides(self, base_encoder, tmp_path):
        """Save + load preserves per-stage module type information."""
        overrides = {1: ["fc1", "fc2"]}
        lora = LoRASwinViT(
            base_encoder,
            rank=4,
            alpha=8,
            target_stages=[1, 2],
            target_module_types=["qkv", "proj", "fc1", "fc2"],
            per_stage_module_types=overrides,
        )

        # Save
        save_path = tmp_path / "adapter"
        lora.save_lora(save_path)

        # Load into a fresh base
        fresh_encoder = create_swinunetr()
        loaded = LoRASwinViT.load_lora(fresh_encoder, save_path, trainable=False)

        # Global types should be the union across all stages
        assert set(loaded.target_module_types) == {"qkv", "proj", "fc1", "fc2"}
        assert loaded.target_stages == [1, 2]
        # Per-stage overrides should be reconstructed: stage 1 only has fc1/fc2
        assert 1 in loaded.per_stage_module_types
        assert set(loaded.per_stage_module_types[1]) == {"fc1", "fc2"}
        # Stage 2 should NOT appear in overrides (it has the full set)
        assert 2 not in loaded.per_stage_module_types

    def test_count_lora_params_per_stage(self, base_encoder):
        """count_lora_params reflects per-stage restrictions."""
        lora = LoRASwinViT(
            base_encoder,
            rank=8,
            target_stages=[1, 2],
            target_module_types=["qkv", "proj", "fc1", "fc2"],
            per_stage_module_types={1: ["fc1", "fc2"]},
        )

        counts = count_lora_params(lora)

        # Stage 1 should have fewer params than stage 2
        assert counts["layers1"] < counts["layers2"]
        assert counts["layers1"] > 0
        assert counts["layers2"] > 0
        assert counts["total"] == counts["layers1"] + counts["layers2"]
