# tests/growth/test_lora_adapter.py
"""Tests for LoRA adapter for SwinViT encoder."""

from pathlib import Path

import pytest
import torch

from growth.models.encoder.lora_adapter import (
    LoRASwinViT,
    create_lora_encoder,
    count_lora_params,
    _find_lora_targets,
)
from growth.models.encoder.swin_loader import load_swin_encoder, create_swinunetr


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
