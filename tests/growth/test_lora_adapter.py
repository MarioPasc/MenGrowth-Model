# tests/growth/test_lora_adapter.py
"""Tests for LoRA adapter for SwinViT encoder."""

from pathlib import Path
from typing import Dict

import pytest
import torch

from growth.models.encoder.lora_adapter import (
    LoRASwinViT,
    MODULE_TYPE_SUFFIX,
    SUPPORTED_MODULE_TYPES,
    _find_lora_targets,
    _infer_module_types_from_target_names,
    count_lora_params,
    create_lora_encoder,
)
from growth.models.encoder.swin_loader import create_swinunetr, load_swin_encoder

pytestmark = [pytest.mark.phase1, pytest.mark.unit]


class TestFindLoraTargets:
    """Tests for target module discovery."""

    def test_find_targets_stages_3_4(self):
        """Test finding target modules for stages 3 and 4."""
        model = create_swinunetr()
        targets = _find_lora_targets(model, stages=[3, 4])

        # Should find 4 targets: 2 blocks x 2 stages
        assert len(targets) == 4

        # All should contain qkv
        assert all("qkv" in t for t in targets)

        # Check stage names
        assert any("layers3" in t for t in targets)
        assert any("layers4" in t for t in targets)

    def test_find_targets_single_stage(self):
        """Test finding targets for single stage."""
        model = create_swinunetr()

        targets_3 = _find_lora_targets(model, stages=[3])
        targets_4 = _find_lora_targets(model, stages=[4])

        # Each stage has 2 blocks with qkv
        assert len(targets_3) == 2
        assert len(targets_4) == 2

        # Verify stage specificity
        assert all("layers3" in t for t in targets_3)
        assert all("layers4" in t for t in targets_4)


class TestLoRASwinViT:
    """Tests for LoRASwinViT class."""

    def test_init_with_fresh_model(self):
        """Test initialization with fresh (untrained) model."""
        model = create_swinunetr()
        lora = LoRASwinViT(model, rank=8, alpha=16)

        assert lora.rank == 8
        assert lora.alpha == 16
        assert lora.get_trainable_params() > 0

    def test_init_with_checkpoint(self, real_checkpoint_path: Path):
        """Test initialization with real checkpoint."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora = LoRASwinViT(encoder, rank=8, alpha=16)

        assert lora.rank == 8
        assert lora.get_trainable_params() > 0

    def test_only_lora_trainable(self, real_checkpoint_path: Path):
        """Test that only LoRA parameters are trainable."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora = LoRASwinViT(encoder, rank=8, alpha=16)

        # Get all trainable parameters
        trainable = [n for n, p in lora.model.named_parameters() if p.requires_grad]

        # All trainable params should be LoRA params
        assert all("lora" in n.lower() for n in trainable)

    def test_forward_pass(self, real_checkpoint_path: Path):
        """Test forward pass produces expected output."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora = LoRASwinViT(encoder, rank=8, alpha=16)
        lora.eval()

        x = torch.randn(1, 4, 96, 96, 96)
        with torch.no_grad():
            out = lora(x)

        assert out.shape == (1, 3, 96, 96, 96)
        assert not torch.isnan(out).any()

    def test_get_hidden_states(self, real_checkpoint_path: Path):
        """Test hidden state extraction."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora = LoRASwinViT(encoder, rank=8, alpha=16)
        lora.eval()

        x = torch.randn(1, 4, 96, 96, 96)
        with torch.no_grad():
            hidden = lora.get_hidden_states(x)

        assert len(hidden) == 5
        assert hidden[4].shape == (1, 768, 3, 3, 3)

    def test_param_counts_scale_with_rank(self, real_checkpoint_path: Path):
        """Test parameter counts scale linearly with rank."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)

        lora_r4 = LoRASwinViT(encoder, rank=4)
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora_r8 = LoRASwinViT(encoder, rank=8)

        # r8 should have 2x params of r4
        ratio = lora_r8.get_trainable_params() / lora_r4.get_trainable_params()
        assert abs(ratio - 2.0) < 0.01

    def test_get_lora_params(self, real_checkpoint_path: Path):
        """Test getting LoRA parameters dict."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora = LoRASwinViT(encoder, rank=8)

        lora_params = lora.get_lora_params()

        assert len(lora_params) > 0
        assert all("lora" in k.lower() for k in lora_params.keys())


class TestLoRASaveLoad:
    """Tests for saving and loading LoRA adapters."""

    def test_save_load_roundtrip(self, real_checkpoint_path: Path, tmp_path: Path):
        """Test saving and loading LoRA adapter."""
        # Create and save
        encoder1 = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora1 = LoRASwinViT(encoder1, rank=8, alpha=16)

        save_path = tmp_path / "lora_adapter"
        lora1.save_lora(save_path)

        # Load
        encoder2 = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora2 = LoRASwinViT.load_lora(encoder2, save_path)

        assert lora2.rank == 8
        assert lora2.alpha == 16

    def test_loaded_produces_same_output(
        self, real_checkpoint_path: Path, tmp_path: Path
    ):
        """Test loaded model produces same hidden states as original.

        Note: We test hidden states (encoder features) rather than full forward
        output because load_swin_encoder only loads encoder weights and decoder
        weights are randomly initialized each time.
        """
        # Create and save
        encoder1 = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora1 = LoRASwinViT(encoder1, rank=8, alpha=16)
        lora1.eval()

        save_path = tmp_path / "lora_adapter"
        lora1.save_lora(save_path)

        # Test input
        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            hidden1 = lora1.get_hidden_states(x)

        # Load and test
        encoder2 = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora2 = LoRASwinViT.load_lora(encoder2, save_path)
        lora2.eval()

        with torch.no_grad():
            hidden2 = lora2.get_hidden_states(x)

        # Compare all hidden states (encoder outputs from all stages)
        assert len(hidden1) == len(hidden2)
        for h1, h2 in zip(hidden1, hidden2):
            assert torch.allclose(h1, h2, atol=1e-4)


class TestMergeLoRA:
    """Tests for LoRA weight merging."""

    def test_merge_produces_valid_model(self, real_checkpoint_path: Path):
        """Test merging produces a valid standalone model."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora = LoRASwinViT(encoder, rank=8, alpha=16)
        lora.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            out_lora = lora(x)

        merged = lora.merge_lora()
        merged.eval()

        with torch.no_grad():
            out_merged = merged(x)

        # Merged should produce very similar output
        assert torch.allclose(out_lora, out_merged, atol=1e-4)

    def test_merged_has_no_lora_params(self, real_checkpoint_path: Path):
        """Test merged model has no LoRA parameters."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora = LoRASwinViT(encoder, rank=8)

        merged = lora.merge_lora()

        # Should have no 'lora' in parameter names
        lora_params = [n for n in merged.state_dict().keys() if "lora" in n.lower()]
        assert len(lora_params) == 0


class TestCreateLoraEncoder:
    """Tests for factory function."""

    def test_create_lora_encoder(self, real_checkpoint_path: Path):
        """Test factory function creates working encoder."""
        lora = create_lora_encoder(
            real_checkpoint_path, rank=8, alpha=16, device="cpu"
        )

        assert isinstance(lora, LoRASwinViT)
        assert lora.rank == 8
        assert lora.get_trainable_params() > 0

    def test_create_with_different_ranks(self, real_checkpoint_path: Path):
        """Test creating encoders with different ranks."""
        for rank in [4, 8, 16]:
            lora = create_lora_encoder(
                real_checkpoint_path, rank=rank, device="cpu"
            )
            assert lora.rank == rank


class TestCountLoraParams:
    """Tests for parameter counting."""

    def test_count_lora_params(self, real_checkpoint_path: Path):
        """Test counting LoRA parameters."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora = LoRASwinViT(encoder, rank=8)

        counts = count_lora_params(lora)

        assert "total" in counts
        assert "layers3" in counts
        assert "layers4" in counts
        assert counts["total"] == counts["layers3"] + counts["layers4"]
        assert counts["total"] > 0

    def test_count_scales_with_rank(self, real_checkpoint_path: Path):
        """Test parameter counts scale with rank."""
        counts = {}
        for rank in [4, 8, 16]:
            encoder = load_swin_encoder(real_checkpoint_path, freeze=False)
            lora = LoRASwinViT(encoder, rank=rank)
            counts[rank] = count_lora_params(lora)["total"]

        # Should double with each rank doubling
        assert abs(counts[8] / counts[4] - 2.0) < 0.01
        assert abs(counts[16] / counts[8] - 2.0) < 0.01


# =============================================================================
# Extended module_types: qkv, proj, fc1, fc2 across stages 1-3
# =============================================================================


def _blocks_per_stage() -> int:
    """Return the number of Swin blocks per stage (BRAINSEGFOUNDER_DEPTHS)."""
    from growth.models.encoder.swin_loader import BRAINSEGFOUNDER_DEPTHS

    # All stages share the same depth in BSF-Tiny; assert it for the test harness.
    depths = set(BRAINSEGFOUNDER_DEPTHS)
    assert len(depths) == 1, (
        f"_blocks_per_stage assumes uniform depth across stages; got {BRAINSEGFOUNDER_DEPTHS}"
    )
    return BRAINSEGFOUNDER_DEPTHS[0]


class TestExtendedModuleTypes:
    """Extended LoRA target_module_types covering qkv, proj, fc1, fc2."""

    def test_default_is_qkv_only(self):
        """Backward-compat: default target_module_types is ['qkv']."""
        model = create_swinunetr()
        targets = _find_lora_targets(model, stages=[3, 4])
        # All targets end with ".attn.qkv" (default == qkv-only).
        assert all(name.endswith(".attn.qkv") for name in targets)

    def test_extended_discovers_all_four_suffixes(self):
        """Extended types=[qkv,proj,fc1,fc2] finds exactly len(stages)*blocks*4 targets."""
        model = create_swinunetr()
        stages = [1, 2, 3]
        types = ["qkv", "proj", "fc1", "fc2"]
        targets = _find_lora_targets(model, stages=stages, module_types=types)

        blocks = _blocks_per_stage()
        expected = len(stages) * blocks * len(types)
        assert len(targets) == expected, (
            f"Expected {expected} targets (stages={len(stages)} × blocks={blocks} × "
            f"types={len(types)}), got {len(targets)}"
        )

    def test_extended_combinatorial_presence(self):
        """Every (stage, block, type) triple appears exactly once."""
        model = create_swinunetr()
        stages = [1, 2, 3]
        types = ["qkv", "proj", "fc1", "fc2"]
        targets = _find_lora_targets(model, stages=stages, module_types=types)

        blocks = _blocks_per_stage()
        for stage in stages:
            for block in range(blocks):
                for t in types:
                    suffix = MODULE_TYPE_SUFFIX[t]
                    # Name is 'layersS.0.blocks.B.{attn|mlp}.{type}'
                    expected = f"layers{stage}.0.blocks.{block}{suffix}"
                    matches = [n for n in targets if n == expected]
                    assert len(matches) == 1, (
                        f"Expected exactly 1 target for (stage={stage}, block={block}, "
                        f"type={t}); got {len(matches)}: {matches}"
                    )

    def test_types_subset_qkv_proj(self):
        """Partial subset: types=[qkv,proj] yields 2× qkv-only count at same stages."""
        model = create_swinunetr()
        qkv_only = _find_lora_targets(model, stages=[3, 4], module_types=["qkv"])
        qkv_proj = _find_lora_targets(model, stages=[3, 4], module_types=["qkv", "proj"])
        assert len(qkv_proj) == 2 * len(qkv_only)
        assert all(
            n.endswith(".attn.qkv") or n.endswith(".attn.proj") for n in qkv_proj
        )

    def test_invalid_module_type_raises(self):
        """Unknown module_types raise a clear ValueError."""
        model = create_swinunetr()
        with pytest.raises(ValueError, match="Unsupported LoRA target_module_types"):
            _find_lora_targets(model, stages=[3], module_types=["qkv", "bogus"])

    def test_empty_module_types_raises(self):
        """Empty module_types is rejected early."""
        model = create_swinunetr()
        with pytest.raises(ValueError, match="non-empty"):
            _find_lora_targets(model, stages=[3], module_types=[])

    def test_module_types_deduplicated(self):
        """Duplicate entries in module_types don't produce duplicate targets."""
        model = create_swinunetr()
        with_dup = _find_lora_targets(
            model, stages=[3], module_types=["qkv", "qkv", "proj"]
        )
        without_dup = _find_lora_targets(
            model, stages=[3], module_types=["qkv", "proj"]
        )
        assert len(with_dup) == len(without_dup)


class TestExtendedPeftWiring:
    """PEFT must actually inject lora_A/lora_B for each discovered target.

    Guards against silent mismatches where _find_lora_targets returns a name
    whose suffix doesn't satisfy PEFT's endswith match (e.g. typo in suffix
    table). These tests make the injection visible.
    """

    def test_extended_peft_injects_one_pair_per_target(self):
        """Post-get_peft_model, count of lora_A/lora_B matches expected target count."""
        model = create_swinunetr()
        stages = [1, 2, 3]
        types = ["qkv", "proj", "fc1", "fc2"]
        lora = LoRASwinViT(
            model, rank=4, alpha=8, target_stages=stages, target_module_types=types
        )

        blocks = _blocks_per_stage()
        expected = len(stages) * blocks * len(types)

        lora_a_count = sum(
            1 for n, _ in lora.model.named_modules() if n.endswith(".lora_A.default")
        )
        lora_b_count = sum(
            1 for n, _ in lora.model.named_modules() if n.endswith(".lora_B.default")
        )
        assert lora_a_count == expected, (
            f"Expected {expected} lora_A.default modules wired by PEFT; got {lora_a_count}"
        )
        assert lora_b_count == expected

    def test_extended_param_count_matches_exact_formula(self):
        """Trainable param count == Σ r · (in_features + out_features) across targets."""
        stages = [1, 2, 3]
        types = ["qkv", "proj", "fc1", "fc2"]
        rank = 4

        # Enumerate Linear features on a fresh (non-PEFT) model so names are
        # untouched — PEFT renames wrapped Linears to '<name>.base_layer'.
        reference = create_swinunetr()
        target_modules: Dict[str, torch.nn.Linear] = {}
        for name, mod in reference.named_modules():
            if isinstance(mod, torch.nn.Linear):
                rel = name[len("swinViT."):] if name.startswith("swinViT.") else name
                target_modules[rel] = mod

        lora = LoRASwinViT(
            create_swinunetr(),
            rank=rank,
            alpha=8,
            target_stages=stages,
            target_module_types=types,
        )

        blocks = _blocks_per_stage()
        expected_params = 0
        for stage in stages:
            for block in range(blocks):
                for t in types:
                    suffix = MODULE_TYPE_SUFFIX[t]
                    rel = f"layers{stage}.0.blocks.{block}{suffix}"
                    lin = target_modules.get(rel)
                    assert lin is not None, f"Missing Linear at {rel}"
                    expected_params += rank * (lin.in_features + lin.out_features)

        trainable = lora.get_trainable_params()
        assert trainable == expected_params, (
            f"Expected {expected_params} trainable LoRA params "
            f"(Σ r·(in+out)); got {trainable}"
        )

    def test_extended_param_count_scales_linearly_with_rank(self):
        """Extended-target param count doubles when rank doubles."""
        counts: Dict[int, int] = {}
        for rank in (2, 4, 8):
            model = create_swinunetr()
            lora = LoRASwinViT(
                model,
                rank=rank,
                alpha=2 * rank,
                target_stages=[1, 2, 3],
                target_module_types=["qkv", "proj", "fc1", "fc2"],
            )
            counts[rank] = lora.get_trainable_params()
        assert counts[4] == 2 * counts[2]
        assert counts[8] == 2 * counts[4]


class TestInferModuleTypesFromTargetNames:
    """Reload path must reconstruct target_module_types from saved PEFT config."""

    def test_infer_from_qkv_only(self):
        names = ["layers3.0.blocks.0.attn.qkv", "layers3.0.blocks.1.attn.qkv"]
        assert _infer_module_types_from_target_names(names) == ["qkv"]

    def test_infer_from_extended(self):
        # Use MONAI's actual attribute names (linear1/linear2) — these are what
        # get saved in adapter_config.json; the user-facing keywords (fc1/fc2)
        # are mapped via MODULE_TYPE_SUFFIX.
        names = [
            "layers1.0.blocks.0.attn.qkv",
            "layers1.0.blocks.0.attn.proj",
            "layers2.0.blocks.1.mlp.linear1",
            "layers3.0.blocks.0.mlp.linear2",
        ]
        assert _infer_module_types_from_target_names(names) == ["qkv", "proj", "fc1", "fc2"]

    def test_save_load_preserves_extended_types(
        self, real_checkpoint_path: Path, tmp_path: Path
    ):
        """Round-trip save → load infers the same target_module_types and stages."""
        encoder1 = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora1 = LoRASwinViT(
            encoder1,
            rank=4,
            alpha=8,
            target_stages=[1, 2, 3],
            target_module_types=["qkv", "proj", "fc1", "fc2"],
        )

        save_path = tmp_path / "lora_extended"
        lora1.save_lora(save_path)

        encoder2 = load_swin_encoder(real_checkpoint_path, freeze=False)
        lora2 = LoRASwinViT.load_lora(encoder2, save_path)

        assert lora2.rank == 4
        assert lora2.alpha == 8
        assert lora2.target_stages == [1, 2, 3]
        assert set(lora2.target_module_types) == {"qkv", "proj", "fc1", "fc2"}


class TestSupportedModuleTypesConstant:
    """The public SUPPORTED_MODULE_TYPES tuple covers exactly qkv, proj, fc1, fc2."""

    def test_supported_module_types(self):
        assert set(SUPPORTED_MODULE_TYPES) == {"qkv", "proj", "fc1", "fc2"}
        assert set(MODULE_TYPE_SUFFIX.keys()) == set(SUPPORTED_MODULE_TYPES)
