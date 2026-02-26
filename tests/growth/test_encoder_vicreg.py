# tests/growth/test_encoder_vicreg.py
"""Tests for encoder VICReg loss."""

import torch
import pytest

from growth.losses.encoder_vicreg import EncoderVICRegLoss


class TestEncoderVICRegLoss:
    """Tests for EncoderVICRegLoss."""

    def test_output_shape_and_finite(self):
        """Loss should be a finite scalar."""
        loss_fn = EncoderVICRegLoss(lambda_var=5.0, lambda_cov=1.0)
        features = torch.randn(16, 768)
        loss, components = loss_fn(features)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert "vicreg_var_loss" in components
        assert "vicreg_cov_loss" in components
        assert "vicreg_total" in components

    def test_var_loss_zero_for_high_variance(self):
        """Var loss should be ~0 when all dims have std > gamma."""
        loss_fn = EncoderVICRegLoss(lambda_var=5.0, lambda_cov=0.0, gamma=1.0)
        # Features with std >> gamma
        features = torch.randn(64, 768) * 5.0
        loss, components = loss_fn(features)

        assert components["vicreg_var_loss"] < 0.01

    def test_var_loss_high_for_collapsed_features(self):
        """Var loss should be large when features are collapsed."""
        loss_fn = EncoderVICRegLoss(lambda_var=5.0, lambda_cov=0.0, gamma=1.0)
        # All features nearly identical (collapsed)
        features = torch.ones(16, 768) + torch.randn(16, 768) * 0.001
        loss, components = loss_fn(features)

        assert components["vicreg_var_loss"] > 0.5

    def test_cov_loss_zero_for_decorrelated(self):
        """Cov loss should be ~0 for decorrelated features."""
        loss_fn = EncoderVICRegLoss(lambda_var=0.0, lambda_cov=1.0)
        # Identity-like covariance (each dim independent)
        features = torch.randn(256, 32)  # Smaller for speed
        loss, components = loss_fn(features)

        # Random features are approximately decorrelated
        assert components["vicreg_cov_loss"] < 0.1

    def test_cov_loss_high_for_correlated(self):
        """Cov loss should be large for highly correlated features."""
        loss_fn = EncoderVICRegLoss(lambda_var=0.0, lambda_cov=1.0)
        # Create highly correlated features: all dims are copies of first
        base = torch.randn(32, 1)
        features = base.expand(32, 64) + torch.randn(32, 64) * 0.01
        loss, components = loss_fn(features)

        assert components["vicreg_cov_loss"] > 0.1

    def test_total_loss_combines_components(self):
        """Total loss should be weighted sum of var and cov."""
        loss_fn = EncoderVICRegLoss(lambda_var=2.0, lambda_cov=3.0)
        features = torch.randn(16, 128)
        loss, components = loss_fn(features)

        expected = 2.0 * components["vicreg_var_loss"] + 3.0 * components["vicreg_cov_loss"]
        assert abs(loss.item() - expected) < 1e-4

    def test_requires_2d_input(self):
        """Should raise assertion error for non-2D input."""
        loss_fn = EncoderVICRegLoss()
        with pytest.raises(AssertionError):
            loss_fn(torch.randn(4, 768, 4))

    def test_gradient_flows(self):
        """Gradients should flow back through the loss."""
        loss_fn = EncoderVICRegLoss(lambda_var=5.0, lambda_cov=1.0)
        features = torch.randn(8, 768, requires_grad=True)
        loss, _ = loss_fn(features)
        loss.backward()

        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_different_batch_sizes(self):
        """Loss should work with various batch sizes."""
        loss_fn = EncoderVICRegLoss()
        for batch_size in [2, 4, 8, 32]:
            features = torch.randn(batch_size, 768)
            loss, _ = loss_fn(features)
            assert torch.isfinite(loss)
