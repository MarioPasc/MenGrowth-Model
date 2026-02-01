# tests/growth/test_swin_loader.py
"""Tests for SwinUNETR encoder loading."""

from pathlib import Path

import pytest
import torch

from growth.models.encoder.swin_loader import (
    BRAINSEGFOUNDER_DEPTHS,
    BRAINSEGFOUNDER_FEATURE_SIZE,
    BRAINSEGFOUNDER_IN_CHANNELS,
    BRAINSEGFOUNDER_NUM_HEADS,
    BRAINSEGFOUNDER_OUT_CHANNELS,
    count_parameters,
    create_swinunetr,
    get_encoder_output_dim,
    get_swin_feature_dims,
    load_swin_encoder,
)


class TestCreateSwinunetr:
    """Tests for create_swinunetr function."""

    def test_create_swinunetr_default(self):
        """Test creating SwinUNETR with default parameters."""
        model = create_swinunetr()

        # Check model exists
        assert model is not None

        # Verify architecture constants
        assert BRAINSEGFOUNDER_IN_CHANNELS == 4
        assert BRAINSEGFOUNDER_OUT_CHANNELS == 3
        assert BRAINSEGFOUNDER_FEATURE_SIZE == 48

    def test_create_swinunetr_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = create_swinunetr()
        model.eval()

        # Expected input: [B, 4, 96, 96, 96]
        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            out = model(x)

        # Output should match input spatial dimensions
        assert out.shape == (1, 3, 96, 96, 96)

    def test_create_swinunetr_custom_channels(self):
        """Test creating SwinUNETR with custom channel counts."""
        model = create_swinunetr(in_channels=1, out_channels=5)
        model.eval()

        x = torch.randn(1, 1, 96, 96, 96)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 5, 96, 96, 96)

    def test_create_swinunetr_with_gradient_checkpointing(self):
        """Test creating SwinUNETR with gradient checkpointing enabled."""
        model = create_swinunetr(use_checkpoint=True)

        # Should not raise during creation
        assert model is not None

    def test_create_swinunetr_parameter_count(self):
        """Test that model has expected parameter count."""
        model = create_swinunetr()
        total_params = count_parameters(model)

        # BrainSegFounder SwinUNETR should have ~62M parameters
        assert total_params > 50_000_000
        assert total_params < 80_000_000


class TestGetSwinFeatureDims:
    """Tests for get_swin_feature_dims function."""

    def test_feature_dims_default(self):
        """Test feature dimensions with default feature_size=48."""
        dims = get_swin_feature_dims()

        # MONAI 1.5+ SwinUNETR dimensions
        expected = {
            "patch_embed": (48, 2),
            "layers1": (96, 4),
            "layers2": (192, 8),
            "layers3": (384, 16),
            "layers4": (768, 32),
            "encoder10": (768, 32),
        }

        assert dims == expected

    def test_feature_dims_custom_feature_size(self):
        """Test feature dimensions with custom feature_size."""
        dims = get_swin_feature_dims(feature_size=24)

        # Channels scale with feature_size
        assert dims["patch_embed"][0] == 24
        assert dims["layers2"][0] == 96  # 24 * 4
        assert dims["layers4"][0] == 384  # 24 * 16
        assert dims["encoder10"][0] == 384  # 24 * 16


class TestGetEncoderOutputDim:
    """Tests for get_encoder_output_dim function."""

    def test_encoder10_dim(self):
        """Test encoder10 output dimension."""
        dim = get_encoder_output_dim("encoder10")
        assert dim == 768

    def test_layers4_dim(self):
        """Test layers4 output dimension."""
        dim = get_encoder_output_dim("layers4")
        assert dim == 768

    def test_multi_scale_dim(self):
        """Test multi_scale output dimension."""
        dim = get_encoder_output_dim("multi_scale")
        # layers2(192) + layers3(384) + layers4(768) = 1344
        assert dim == 1344

    def test_unknown_level_raises(self):
        """Test error on unknown feature level."""
        with pytest.raises(ValueError, match="Unknown feature level"):
            get_encoder_output_dim("invalid_level")


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_all_parameters(self):
        """Test counting all parameters."""
        model = torch.nn.Linear(10, 5)  # 10*5 + 5 = 55 params
        count = count_parameters(model)
        assert count == 55

    def test_count_trainable_only(self):
        """Test counting only trainable parameters."""
        model = torch.nn.Linear(10, 5)
        model.weight.requires_grad = False  # Freeze weight (50 params)

        total = count_parameters(model, trainable_only=False)
        trainable = count_parameters(model, trainable_only=True)

        assert total == 55
        assert trainable == 5  # Only bias is trainable


class TestLoadSwinEncoder:
    """Tests for load_swin_encoder function."""

    def test_load_encoder_from_sample_checkpoint(
        self, sample_checkpoint: Path, sample_state_dict
    ):
        """Test loading encoder from sample checkpoint."""
        # Note: This will have missing keys since sample_state_dict is simplified
        model = load_swin_encoder(
            sample_checkpoint,
            freeze=True,
            device="cpu",
            strict_load=False,
        )

        assert model is not None

    def test_load_encoder_freeze(self, sample_checkpoint: Path):
        """Test that freeze=True freezes all parameters."""
        model = load_swin_encoder(
            sample_checkpoint,
            freeze=True,
            device="cpu",
            strict_load=False,
        )

        # All parameters should be frozen
        for param in model.parameters():
            assert not param.requires_grad

    def test_load_encoder_not_found(self, temp_dir: Path):
        """Test error when checkpoint doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_swin_encoder(temp_dir / "missing.pt")


class TestLoadSwinEncoderReal:
    """Tests with real BrainSegFounder checkpoint (skipped if unavailable)."""

    def test_load_real_encoder(self, real_checkpoint_path: Path):
        """Test loading encoder from real checkpoint."""
        model = load_swin_encoder(
            real_checkpoint_path,
            freeze=True,
            device="cpu",
        )

        assert model is not None

        # Check it's frozen
        trainable = count_parameters(model, trainable_only=True)
        assert trainable == 0

    def test_real_encoder_forward_pass(self, real_checkpoint_path: Path):
        """Test forward pass with real encoder."""
        model = load_swin_encoder(
            real_checkpoint_path,
            freeze=True,
            device="cpu",
        )
        model.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            # Test full forward (decoder output)
            out = model(x)
            assert out.shape == (1, 3, 96, 96, 96)

    def test_real_encoder_hidden_states(self, real_checkpoint_path: Path):
        """Test extracting hidden states from real encoder."""
        model = load_swin_encoder(
            real_checkpoint_path,
            freeze=True,
            device="cpu",
        )
        model.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            hidden_states = model.swinViT(x, model.normalize)

        # Check hidden state shapes
        assert len(hidden_states) == 5

        # hidden_states[0]: After patch_embed [B, 48, 48, 48, 48]
        assert hidden_states[0].shape == (1, 48, 48, 48, 48)

        # hidden_states[4]: After layers4 [B, 768, 3, 3, 3]
        assert hidden_states[4].shape == (1, 768, 3, 3, 3)

    def test_real_encoder_encoder10_output(self, real_checkpoint_path: Path):
        """Test encoder10 bottleneck output."""
        model = load_swin_encoder(
            real_checkpoint_path,
            include_encoder10=True,
            freeze=True,
            device="cpu",
        )
        model.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            hidden_states = model.swinViT(x, model.normalize)
            x4 = hidden_states[4]  # [1, 384, 3, 3, 3]
            enc10 = model.encoder10(x4)  # [1, 768, 3, 3, 3]

        assert enc10.shape == (1, 768, 3, 3, 3)
