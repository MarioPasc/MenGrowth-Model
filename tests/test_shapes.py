"""Unit tests for VAE shape verification.

This module tests:
1. Encoder output shapes: mu [B,128], logvar [B,128]
2. Decoder output shape: x_hat matches input [B,4,128,128,128]
3. ELBO loss returns finite positive scalars
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae_dynamics.models import BaselineVAE
from vae_dynamics.losses import compute_elbo


class TestVAEShapes:
    """Test VAE model shapes."""

    @pytest.fixture
    def model(self):
        """Create a BaselineVAE model for testing."""
        return BaselineVAE(
            input_channels=4,
            z_dim=128,
            base_filters=32,
            num_groups=8,
        )

    @pytest.fixture
    def dummy_batch(self):
        """Create a dummy input batch."""
        return torch.randn(2, 4, 128, 128, 128)

    def test_encoder_output_shapes(self, model, dummy_batch):
        """Test that encoder produces correct mu and logvar shapes.

        Expected:
            - mu: [B=2, z_dim=128]
            - logvar: [B=2, z_dim=128]
        """
        mu, logvar = model.encode(dummy_batch)

        assert mu.shape == (2, 128), f"Expected mu shape (2, 128), got {mu.shape}"
        assert logvar.shape == (2, 128), f"Expected logvar shape (2, 128), got {logvar.shape}"

    def test_decoder_output_shape(self, model, dummy_batch):
        """Test that decoder output matches input shape exactly.

        Expected: x_hat shape [B=2, C=4, D=128, H=128, W=128]
        """
        x_hat, mu, logvar = model(dummy_batch)

        assert x_hat.shape == dummy_batch.shape, (
            f"Expected x_hat shape {dummy_batch.shape}, got {x_hat.shape}"
        )
        assert x_hat.shape == (2, 4, 128, 128, 128), (
            f"Expected x_hat shape (2, 4, 128, 128, 128), got {x_hat.shape}"
        )

    def test_forward_returns_all_outputs(self, model, dummy_batch):
        """Test that forward returns (x_hat, mu, logvar)."""
        outputs = model(dummy_batch)

        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
        x_hat, mu, logvar = outputs

        assert isinstance(x_hat, torch.Tensor)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(logvar, torch.Tensor)

    def test_reparameterization(self, model):
        """Test that reparameterization produces correct z shape."""
        mu = torch.randn(2, 128)
        logvar = torch.randn(2, 128)

        z = model.reparameterize(mu, logvar)

        assert z.shape == (2, 128), f"Expected z shape (2, 128), got {z.shape}"


class TestELBOLoss:
    """Test ELBO loss computation."""

    @pytest.fixture
    def dummy_tensors(self):
        """Create dummy tensors for loss computation."""
        x = torch.randn(2, 4, 128, 128, 128)
        x_hat = torch.randn(2, 4, 128, 128, 128)
        mu = torch.randn(2, 128)
        logvar = torch.randn(2, 128)
        return x, x_hat, mu, logvar

    def test_elbo_returns_dict(self, dummy_tensors):
        """Test that compute_elbo returns dict with required keys."""
        x, x_hat, mu, logvar = dummy_tensors

        result = compute_elbo(x, x_hat, mu, logvar, beta=1.0)

        assert isinstance(result, dict), "Expected dict return type"
        assert "loss" in result, "Missing 'loss' key"
        assert "recon" in result, "Missing 'recon' key"
        assert "kl" in result, "Missing 'kl' key"

    def test_elbo_finite_values(self, dummy_tensors):
        """Test that all ELBO components are finite."""
        x, x_hat, mu, logvar = dummy_tensors

        result = compute_elbo(x, x_hat, mu, logvar, beta=1.0)

        assert torch.isfinite(result["loss"]), "Loss is not finite"
        assert torch.isfinite(result["recon"]), "Recon loss is not finite"
        assert torch.isfinite(result["kl"]), "KL loss is not finite"

    def test_elbo_non_negative(self, dummy_tensors):
        """Test that loss components are non-negative."""
        x, x_hat, mu, logvar = dummy_tensors

        result = compute_elbo(x, x_hat, mu, logvar, beta=1.0)

        # Reconstruction loss (MSE sum) is always >= 0
        assert result["recon"] >= 0, f"Recon loss should be >= 0, got {result['recon']}"

        # KL divergence is always >= 0
        assert result["kl"] >= 0, f"KL loss should be >= 0, got {result['kl']}"

        # Total loss is always >= 0 (sum of non-negative terms)
        assert result["loss"] >= 0, f"Total loss should be >= 0, got {result['loss']}"

    def test_elbo_zero_kl_for_standard_normal(self):
        """Test that KL is zero when posterior equals prior."""
        x = torch.randn(2, 4, 128, 128, 128)
        x_hat = x.clone()  # Perfect reconstruction
        mu = torch.zeros(2, 128)  # Mean = 0
        logvar = torch.zeros(2, 128)  # Variance = 1

        result = compute_elbo(x, x_hat, mu, logvar, beta=1.0)

        # KL(N(0,1) || N(0,1)) = 0
        assert torch.abs(result["kl"]) < 1e-5, (
            f"KL should be ~0 for standard normal, got {result['kl']}"
        )

    def test_elbo_beta_scaling(self, dummy_tensors):
        """Test that beta correctly scales KL term."""
        x, x_hat, mu, logvar = dummy_tensors

        result_beta1 = compute_elbo(x, x_hat, mu, logvar, beta=1.0)
        result_beta2 = compute_elbo(x, x_hat, mu, logvar, beta=2.0)

        # Recon should be the same
        assert torch.allclose(result_beta1["recon"], result_beta2["recon"])

        # KL should be the same (raw value)
        assert torch.allclose(result_beta1["kl"], result_beta2["kl"])

        # Total loss should differ by beta * KL
        expected_diff = result_beta1["kl"]  # 2.0 * KL - 1.0 * KL = KL
        actual_diff = result_beta2["loss"] - result_beta1["loss"]
        # Use larger tolerance due to floating point accumulation on large sums
        assert torch.allclose(actual_diff, expected_diff, rtol=1e-2)


class TestVAEIntegration:
    """Integration tests for full VAE pipeline."""

    def test_full_forward_pass(self):
        """Test complete forward pass through VAE."""
        model = BaselineVAE(
            input_channels=4,
            z_dim=128,
            base_filters=32,
            num_groups=8,
        )

        x = torch.randn(2, 4, 128, 128, 128)
        x_hat, mu, logvar = model(x)

        # Verify shapes
        assert x_hat.shape == x.shape
        assert mu.shape == (2, 128)
        assert logvar.shape == (2, 128)

        # Compute loss
        result = compute_elbo(x, x_hat, mu, logvar, beta=1.0)

        # Verify loss is finite and positive
        assert torch.isfinite(result["loss"])
        assert result["loss"] >= 0

    def test_backward_pass(self):
        """Test that gradients flow correctly."""
        model = BaselineVAE(
            input_channels=4,
            z_dim=128,
            base_filters=32,
            num_groups=8,
        )

        x = torch.randn(2, 4, 128, 128, 128)
        x_hat, mu, logvar = model(x)
        result = compute_elbo(x, x_hat, mu, logvar, beta=1.0)

        # Backward pass
        result["loss"].backward()

        # Check that some gradients exist
        has_grads = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        assert has_grads, "No gradients computed during backward pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
