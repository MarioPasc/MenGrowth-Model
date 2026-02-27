# tests/growth/test_seg_head.py
"""Tests for lightweight segmentation head."""

from pathlib import Path

import pytest
import torch

from growth.models.segmentation.seg_head import (
    ConvBlock,
    UpsampleBlock,
    SegmentationHead,
    LoRASegmentationModel,
)


class TestConvBlock:
    """Tests for ConvBlock."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        block = ConvBlock(64, 32)
        x = torch.randn(1, 64, 12, 12, 12)
        y = block(x)

        assert y.shape == (1, 32, 12, 12, 12)

    def test_different_kernel_sizes(self):
        """Test different kernel sizes."""
        for kernel_size in [1, 3, 5]:
            block = ConvBlock(64, 32, kernel_size=kernel_size)
            x = torch.randn(1, 64, 12, 12, 12)
            y = block(x)
            assert y.shape == (1, 32, 12, 12, 12)

    def test_different_norms(self):
        """Test different normalization types."""
        for norm in ["instance", "batch", None]:
            block = ConvBlock(64, 32, norm=norm)
            x = torch.randn(2, 64, 8, 8, 8)  # batch=2 for batch norm
            y = block(x)
            assert y.shape == (2, 32, 8, 8, 8)

    def test_different_activations(self):
        """Test different activation types."""
        for activation in ["relu", "leaky_relu", "gelu", None]:
            block = ConvBlock(64, 32, activation=activation)
            x = torch.randn(1, 64, 8, 8, 8)
            y = block(x)
            assert y.shape == (1, 32, 8, 8, 8)


class TestUpsampleBlock:
    """Tests for UpsampleBlock."""

    def test_upsample_only(self):
        """Test upsampling without skip connection."""
        block = UpsampleBlock(64, 32, skip_channels=0)
        x = torch.randn(1, 64, 6, 6, 6)
        y = block(x, skip=None)

        assert y.shape == (1, 32, 12, 12, 12)

    def test_with_skip_connection(self):
        """Test upsampling with skip connection."""
        block = UpsampleBlock(64, 32, skip_channels=48)
        x = torch.randn(1, 64, 6, 6, 6)
        skip = torch.randn(1, 48, 12, 12, 12)
        y = block(x, skip)

        assert y.shape == (1, 32, 12, 12, 12)

    def test_different_scale_factors(self):
        """Test different upsampling scale factors."""
        for scale in [2, 4]:
            block = UpsampleBlock(64, 32, skip_channels=0, scale_factor=scale)
            x = torch.randn(1, 64, 6, 6, 6)
            y = block(x, skip=None)
            expected_size = 6 * scale
            assert y.shape == (1, 32, expected_size, expected_size, expected_size)


class TestSegmentationHead:
    """Tests for SegmentationHead."""

    @pytest.fixture
    def mock_hidden_states(self):
        """Create mock hidden states matching SwinViT output."""
        return [
            torch.randn(1, 48, 48, 48, 48),   # stage 0
            torch.randn(1, 96, 24, 24, 24),   # stage 1
            torch.randn(1, 192, 12, 12, 12),  # stage 2
            torch.randn(1, 384, 6, 6, 6),     # stage 3
            torch.randn(1, 768, 3, 3, 3),     # stage 4
        ]

    def test_default_forward(self, mock_hidden_states):
        """Test forward pass with default settings (4 classes for BraTS)."""
        head = SegmentationHead()
        out = head(mock_hidden_states)

        assert out.shape == (1, 4, 96, 96, 96)

    def test_custom_out_channels(self, mock_hidden_states):
        """Test with custom output channels."""
        head = SegmentationHead(out_channels=3)
        out = head(mock_hidden_states)

        assert out.shape == (1, 3, 96, 96, 96)

    def test_deep_supervision(self, mock_hidden_states):
        """Test with deep supervision enabled (4 classes for BraTS)."""
        head = SegmentationHead(use_deep_supervision=True)
        out, ds_outputs = head(mock_hidden_states)

        assert out.shape == (1, 4, 96, 96, 96)
        assert len(ds_outputs) == 3
        assert ds_outputs[0].shape == (1, 4, 6, 6, 6)
        assert ds_outputs[1].shape == (1, 4, 12, 12, 12)
        assert ds_outputs[2].shape == (1, 4, 24, 24, 24)

    def test_batch_size(self, mock_hidden_states):
        """Test with larger batch size (4 classes for BraTS)."""
        # Update mock to batch=4
        mock_hidden_states_batch = [
            torch.randn(4, 48, 48, 48, 48),
            torch.randn(4, 96, 24, 24, 24),
            torch.randn(4, 192, 12, 12, 12),
            torch.randn(4, 384, 6, 6, 6),
            torch.randn(4, 768, 3, 3, 3),
        ]
        head = SegmentationHead()
        out = head(mock_hidden_states_batch)

        assert out.shape == (4, 4, 96, 96, 96)

    def test_param_count(self):
        """Test parameter count is reasonable."""
        head = SegmentationHead()
        param_count = head.get_param_count()

        # Should be in a reasonable range (e.g., 5M-15M)
        assert 1_000_000 < param_count < 20_000_000

    def test_invalid_hidden_states_count(self):
        """Test error on wrong number of hidden states."""
        head = SegmentationHead()
        hidden_states = [torch.randn(1, 48, 48, 48, 48)] * 4  # Only 4, need 5

        with pytest.raises(ValueError, match="Expected 5 hidden states"):
            head(hidden_states)

    def test_output_no_nan(self, mock_hidden_states):
        """Test output contains no NaN values."""
        head = SegmentationHead()
        out = head(mock_hidden_states)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestLoRASegmentationModel:
    """Tests for LoRASegmentationModel."""

    def test_forward_with_real_encoder(self, real_checkpoint_path: Path):
        """Test forward pass with real LoRA encoder (4 classes for BraTS)."""
        from growth.models.encoder.lora_adapter import create_lora_encoder

        lora_encoder = create_lora_encoder(
            real_checkpoint_path, rank=8, alpha=16, device="cpu"
        )
        model = LoRASegmentationModel(lora_encoder)
        model.eval()

        x = torch.randn(1, 4, 96, 96, 96)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 4, 96, 96, 96)

    def test_param_count_breakdown(self, real_checkpoint_path: Path):
        """Test parameter count breakdown."""
        from growth.models.encoder.lora_adapter import create_lora_encoder

        lora_encoder = create_lora_encoder(
            real_checkpoint_path, rank=8, alpha=16, device="cpu"
        )
        model = LoRASegmentationModel(lora_encoder)

        counts = model.get_trainable_param_count()

        assert "encoder_lora" in counts
        assert "decoder" in counts
        assert "total" in counts
        assert counts["total"] == counts["encoder_lora"] + counts["decoder"]

        # LoRA params should be much smaller than decoder
        assert counts["encoder_lora"] < counts["decoder"]

    def test_separate_param_groups(self, real_checkpoint_path: Path):
        """Test that encoder and decoder params can be retrieved separately."""
        from growth.models.encoder.lora_adapter import create_lora_encoder

        lora_encoder = create_lora_encoder(
            real_checkpoint_path, rank=8, alpha=16, device="cpu"
        )
        model = LoRASegmentationModel(lora_encoder)

        encoder_params = list(model.get_encoder_params())
        decoder_params = list(model.get_decoder_params())

        assert len(encoder_params) > 0
        assert len(decoder_params) > 0


# =============================================================================
# BUG-3: AuxiliarySemanticLoss buffer persistence tests
# =============================================================================


class TestAuxiliarySemanticLossBuffers:
    """Tests for BUG-3 fix: .copy_() preserves registered buffers."""

    def test_state_dict_roundtrip_preserves_statistics(self):
        """Save → load should preserve updated buffer values."""
        from growth.models.segmentation.semantic_heads import AuxiliarySemanticLoss

        loss_fn = AuxiliarySemanticLoss()

        # Update statistics
        vol = torch.randn(50, 4) * 10
        loc = torch.randn(50, 3) * 5
        shape = torch.randn(50, 3) * 2
        loss_fn.update_statistics(vol, loc, shape)

        # Save state dict
        state = loss_fn.state_dict()

        # Create fresh instance and load
        loss_fn2 = AuxiliarySemanticLoss()
        loss_fn2.load_state_dict(state)

        # Verify values match
        torch.testing.assert_close(loss_fn2.volume_mean, loss_fn.volume_mean)
        torch.testing.assert_close(loss_fn2.volume_std, loss_fn.volume_std)
        torch.testing.assert_close(loss_fn2.location_mean, loss_fn.location_mean)
        torch.testing.assert_close(loss_fn2.location_std, loss_fn.location_std)
        torch.testing.assert_close(loss_fn2.shape_mean, loss_fn.shape_mean)
        torch.testing.assert_close(loss_fn2.shape_std, loss_fn.shape_std)

    def test_buffers_remain_registered_after_update(self):
        """After update_statistics(), buffers should still be in named_buffers()."""
        from growth.models.segmentation.semantic_heads import AuxiliarySemanticLoss

        loss_fn = AuxiliarySemanticLoss()

        # Get buffer names before
        buffer_names_before = {name for name, _ in loss_fn.named_buffers()}

        # Update
        loss_fn.update_statistics(
            torch.randn(20, 4), torch.randn(20, 3), torch.randn(20, 3),
        )

        # Get buffer names after
        buffer_names_after = {name for name, _ in loss_fn.named_buffers()}

        expected = {
            "volume_mean", "volume_std",
            "location_mean", "location_std",
            "shape_mean", "shape_std",
        }

        assert expected.issubset(buffer_names_before), f"Missing buffers before: {expected - buffer_names_before}"
        assert expected.issubset(buffer_names_after), f"Missing buffers after: {expected - buffer_names_after}"

    def test_buffers_survive_to_device(self):
        """Buffers should move with .to(device) after update."""
        from growth.models.segmentation.semantic_heads import AuxiliarySemanticLoss

        loss_fn = AuxiliarySemanticLoss()
        loss_fn.update_statistics(
            torch.randn(20, 4), torch.randn(20, 3), torch.randn(20, 3),
        )

        # Move to CPU (should work even if already on CPU — tests the mechanism)
        loss_fn = loss_fn.to("cpu")

        # Buffers should still be accessible
        assert loss_fn.volume_mean.device.type == "cpu"
        assert loss_fn.volume_mean.shape == (4,)
