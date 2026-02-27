# tests/growth/test_model_factory.py
"""Tests for BUG-1: Baseline double forward pass fix.

Verifies that BaselineOriginalDecoderModel.forward_with_semantics()
calls swinViT only once (not twice as in the buggy version).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from experiments.lora_ablation.pipeline.model_factory import BaselineOriginalDecoderModel


class _FakeSwinViT(nn.Module):
    """Minimal swinViT stub returning correctly shaped hidden states."""

    def __init__(self):
        super().__init__()
        self.call_count = 0
        # Dummy parameter so .parameters() isn't empty
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, normalize):
        self.call_count += 1
        B = x.shape[0]
        return [
            torch.randn(B, 48, 48, 48, 48),
            torch.randn(B, 96, 24, 24, 24),
            torch.randn(B, 192, 12, 12, 12),
            torch.randn(B, 384, 6, 6, 6),
            torch.randn(B, 768, 3, 3, 3),
        ]


class _FakeDecoderWrapper(nn.Module):
    """Minimal decoder wrapper stub."""

    def __init__(self):
        super().__init__()
        self.out = nn.Conv3d(48, 3, 1)
        # Minimal encoder/decoder stubs
        self.encoder1 = nn.Identity()
        self.encoder2 = nn.Identity()
        self.encoder3 = nn.Identity()
        self.encoder4 = nn.Identity()
        self.encoder10 = nn.Identity()
        self.decoder5 = nn.Identity()
        self.decoder4 = nn.Identity()
        self.decoder3 = nn.Identity()
        self.decoder2 = nn.Identity()
        self.decoder1 = nn.Identity()

    def forward(self, x_in, hidden_states):
        B = x_in.shape[0]
        return torch.randn(B, 3, 128, 128, 128)

    def get_bottleneck_features(self, hidden_states):
        B = hidden_states[4].shape[0]
        return torch.randn(B, 768)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _FakeFullModel(nn.Module):
    """Minimal full SwinUNETR stub for BaselineOriginalDecoderModel."""

    def __init__(self):
        super().__init__()
        self.swinViT = _FakeSwinViT()
        self.normalize = True
        # Fake output layer that _get_out_layer_channels can read
        self.out = nn.Sequential(nn.Conv3d(48, 3, kernel_size=1))
        # Minimal decoder modules
        self.encoder1 = nn.Identity()
        self.encoder2 = nn.Identity()
        self.encoder3 = nn.Identity()
        self.encoder4 = nn.Identity()
        self.encoder10 = nn.Identity()
        self.decoder5 = nn.Identity()
        self.decoder4 = nn.Identity()
        self.decoder3 = nn.Identity()
        self.decoder2 = nn.Identity()
        self.decoder1 = nn.Identity()


class TestBaselineForwardWithSemantics:
    """Tests for BUG-1 fix: single swinViT call."""

    def _make_model(self, use_semantic_heads: bool = False) -> BaselineOriginalDecoderModel:
        """Create a BaselineOriginalDecoderModel with fake internals."""
        full_model = _FakeFullModel()
        model = BaselineOriginalDecoderModel(
            full_model, out_channels=3, use_semantic_heads=use_semantic_heads
        )
        # Replace the decoder with our controllable fake
        fake_decoder = _FakeDecoderWrapper()
        model.model.decoder = fake_decoder
        return model

    def test_forward_with_semantics_calls_swinvit_once(self):
        """SwinViT should be called exactly once in forward_with_semantics."""
        model = self._make_model()
        model.eval()

        x = torch.randn(1, 4, 128, 128, 128)
        swinvit = model.model.encoder.swinViT
        swinvit.call_count = 0

        with torch.no_grad():
            model.forward_with_semantics(x)

        assert swinvit.call_count == 1, (
            f"swinViT called {swinvit.call_count} times, expected 1"
        )

    def test_forward_with_semantics_output_keys_no_heads(self):
        """Without semantic heads: output has logits and features."""
        model = self._make_model(use_semantic_heads=False)
        model.eval()

        x = torch.randn(1, 4, 128, 128, 128)
        with torch.no_grad():
            result = model.forward_with_semantics(x)

        assert "logits" in result
        assert "features" in result
        assert "pred_volume" not in result

    def test_forward_with_semantics_output_keys_with_heads(self):
        """With semantic heads: output has logits, features, and predictions."""
        model = self._make_model(use_semantic_heads=True)
        model.eval()

        x = torch.randn(1, 4, 128, 128, 128)
        with torch.no_grad():
            result = model.forward_with_semantics(x)

        assert "logits" in result
        assert "features" in result
        assert "pred_volume" in result
        assert "pred_location" in result
        assert "pred_shape" in result

    def test_features_shape(self):
        """Features should be [B, 768]."""
        model = self._make_model()
        model.eval()

        x = torch.randn(2, 4, 128, 128, 128)
        with torch.no_grad():
            result = model.forward_with_semantics(x)

        assert result["features"].shape == (2, 768)
