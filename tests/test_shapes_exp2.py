"""Unit tests for Exp2 TCVAE+SBD model shapes.

Tests verify:
1. Model forward returns correct output shapes
2. TCVAE loss returns finite scalar
3. Gradient flow works correctly
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae_dynamics.models import TCVAESBD
from vae_dynamics.losses import compute_tcvae_loss


class TestTCVAESBDShapes:
    """Tests for TCVAE+SBD model shapes."""

    @pytest.fixture
    def model(self):
        """Create TCVAESBD model for testing."""
        return TCVAESBD(
            input_channels=4,
            z_dim=128,
            base_filters=32,
            num_groups=8,
            sbd_grid_size=(8, 8, 8),
            gradient_checkpointing=False,
        )

    @pytest.fixture
    def dummy_batch(self):
        """Create dummy input batch."""
        return torch.randn(2, 4, 128, 128, 128)

    def test_forward_output_count(self, model, dummy_batch):
        """Test that forward returns 4 outputs."""
        outputs = model(dummy_batch)
        assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"

    def test_x_hat_shape(self, model, dummy_batch):
        """Test that x_hat has same shape as input."""
        x_hat, mu, logvar, z = model(dummy_batch)

        assert x_hat.shape == dummy_batch.shape, (
            f"Expected x_hat shape {dummy_batch.shape}, got {x_hat.shape}"
        )

    def test_mu_shape(self, model, dummy_batch):
        """Test that mu has correct shape [B, z_dim]."""
        x_hat, mu, logvar, z = model(dummy_batch)

        assert mu.shape == (2, 128), f"Expected mu shape (2, 128), got {mu.shape}"

    def test_logvar_shape(self, model, dummy_batch):
        """Test that logvar has correct shape [B, z_dim]."""
        x_hat, mu, logvar, z = model(dummy_batch)

        assert logvar.shape == (2, 128), f"Expected logvar shape (2, 128), got {logvar.shape}"

    def test_z_shape(self, model, dummy_batch):
        """Test that z has correct shape [B, z_dim]."""
        x_hat, mu, logvar, z = model(dummy_batch)

        assert z.shape == (2, 128), f"Expected z shape (2, 128), got {z.shape}"

    def test_encode_method(self, model, dummy_batch):
        """Test that encode method returns mu, logvar with correct shapes."""
        mu, logvar = model.encode(dummy_batch)

        assert mu.shape == (2, 128)
        assert logvar.shape == (2, 128)

    def test_decode_method(self, model):
        """Test that decode method returns correct shape."""
        z = torch.randn(2, 128)
        x_hat = model.decode(z)

        assert x_hat.shape == (2, 4, 128, 128, 128)

    def test_reparameterize_method(self, model):
        """Test reparameterization produces correct shape."""
        mu = torch.randn(2, 128)
        logvar = torch.randn(2, 128)

        z = model.reparameterize(mu, logvar)

        assert z.shape == (2, 128)


class TestTCVAESBDLossIntegration:
    """Integration tests for TCVAE+SBD with loss computation."""

    @pytest.fixture
    def model(self):
        """Create TCVAESBD model."""
        return TCVAESBD(
            input_channels=4,
            z_dim=128,
            base_filters=32,
            num_groups=8,
            sbd_grid_size=(8, 8, 8),
        )

    def test_loss_returns_finite(self, model):
        """Test that loss computation returns finite value."""
        x = torch.randn(2, 4, 128, 128, 128)
        x_hat, mu, logvar, z = model(x)

        loss_dict = compute_tcvae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            n_data=100,
            beta_tc=6.0,
        )

        assert torch.isfinite(loss_dict["loss"]), "Loss should be finite"

    def test_backward_pass(self, model):
        """Test that gradients flow correctly."""
        x = torch.randn(2, 4, 128, 128, 128)
        x_hat, mu, logvar, z = model(x)

        loss_dict = compute_tcvae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            n_data=100,
            beta_tc=6.0,
        )

        # Backward pass
        loss_dict["loss"].backward()

        # Check that some gradients exist
        has_grads = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        assert has_grads, "No gradients computed during backward pass"


class TestTCVAESBDGradientCheckpointing:
    """Tests for gradient checkpointing functionality."""

    def test_checkpointing_model_creation(self):
        """Test that model with checkpointing can be created and run."""
        model = TCVAESBD(
            input_channels=4,
            z_dim=128,
            base_filters=32,
            gradient_checkpointing=True,
        )

        # Test forward pass works
        x = torch.randn(2, 4, 128, 128, 128)
        x_hat, mu, logvar, z = model(x)

        # Verify shapes
        assert x_hat.shape == x.shape
        assert mu.shape == (2, 128)
        assert logvar.shape == (2, 128)
        assert z.shape == (2, 128)

    def test_checkpointing_backward_works(self):
        """Test that backward pass works with checkpointing enabled."""
        model = TCVAESBD(
            input_channels=4,
            z_dim=128,
            base_filters=32,
            gradient_checkpointing=True,
        )
        model.train()

        x = torch.randn(2, 4, 128, 128, 128)
        x_hat, mu, logvar, z = model(x)

        # Simple loss
        loss = x_hat.mean() + mu.mean() + logvar.mean()
        loss.backward()

        # Check gradients exist
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grads, "Gradients should flow with checkpointing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
