"""Unit tests for β-TCVAE loss with MWS estimator.

Tests verify:
1. Loss function returns dict with required keys
2. All values are finite
3. Loss computation works with different batch sizes
4. Beta scheduling works correctly
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae_dynamics.losses.tcvae import compute_tcvae_loss, get_beta_tc_schedule


class TestTCVAELoss:
    """Tests for β-TCVAE loss computation."""

    @pytest.fixture
    def synthetic_batch(self):
        """Create synthetic batch for testing.

        M=4 samples, d=128 latent dimensions
        """
        m, d = 4, 128

        # Random but finite tensors
        mu = torch.randn(m, d)
        logvar = torch.randn(m, d).clamp(-10, 2)  # Clamp for stability

        # Sample z via reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Dummy reconstruction tensors
        x = torch.randn(m, 4, 128, 128, 128)
        x_hat = x + 0.1 * torch.randn_like(x)  # Slightly perturbed

        return {
            "x": x,
            "x_hat": x_hat,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def test_loss_returns_required_keys(self, synthetic_batch):
        """Test that compute_tcvae_loss returns dict with required keys."""
        result = compute_tcvae_loss(
            x=synthetic_batch["x"],
            x_hat=synthetic_batch["x_hat"],
            mu=synthetic_batch["mu"],
            logvar=synthetic_batch["logvar"],
            z=synthetic_batch["z"],
            n_data=1000,
            alpha=1.0,
            beta_tc=6.0,
            gamma=1.0,
        )

        required_keys = ["loss", "recon", "mi", "tc", "dwkl", "beta_tc"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_all_values_finite(self, synthetic_batch):
        """Test that all loss components are finite."""
        result = compute_tcvae_loss(
            x=synthetic_batch["x"],
            x_hat=synthetic_batch["x_hat"],
            mu=synthetic_batch["mu"],
            logvar=synthetic_batch["logvar"],
            z=synthetic_batch["z"],
            n_data=1000,
            alpha=1.0,
            beta_tc=6.0,
            gamma=1.0,
        )

        for key, value in result.items():
            assert torch.isfinite(value).all(), f"{key} is not finite: {value}"

    def test_loss_with_small_batch(self):
        """Test loss computation with small batch (M=2)."""
        m, d = 2, 128

        mu = torch.randn(m, d)
        logvar = torch.randn(m, d).clamp(-10, 2)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        x = torch.randn(m, 4, 64, 64, 64)  # Smaller spatial size for speed
        x_hat = x + 0.1 * torch.randn_like(x)

        result = compute_tcvae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            n_data=100,
            beta_tc=6.0,
        )

        assert torch.isfinite(result["loss"]), "Loss should be finite with small batch"

    def test_loss_with_large_n_data(self, synthetic_batch):
        """Test loss computation with large N (dataset size)."""
        result = compute_tcvae_loss(
            x=synthetic_batch["x"],
            x_hat=synthetic_batch["x_hat"],
            mu=synthetic_batch["mu"],
            logvar=synthetic_batch["logvar"],
            z=synthetic_batch["z"],
            n_data=10000,  # Large dataset
            beta_tc=6.0,
        )

        assert torch.isfinite(result["loss"]), "Loss should be finite with large N"

    def test_beta_tc_affects_loss(self, synthetic_batch):
        """Test that beta_tc parameter affects total loss."""
        result_low = compute_tcvae_loss(
            x=synthetic_batch["x"],
            x_hat=synthetic_batch["x_hat"],
            mu=synthetic_batch["mu"],
            logvar=synthetic_batch["logvar"],
            z=synthetic_batch["z"],
            n_data=1000,
            beta_tc=1.0,
        )

        result_high = compute_tcvae_loss(
            x=synthetic_batch["x"],
            x_hat=synthetic_batch["x_hat"],
            mu=synthetic_batch["mu"],
            logvar=synthetic_batch["logvar"],
            z=synthetic_batch["z"],
            n_data=1000,
            beta_tc=10.0,
        )

        # Recon should be the same
        assert torch.allclose(result_low["recon"], result_high["recon"])

        # TC raw value should be the same
        assert torch.allclose(result_low["tc"], result_high["tc"])

        # Total loss should be different (unless TC is exactly 0)
        if result_low["tc"].abs() > 1e-6:
            assert not torch.allclose(result_low["loss"], result_high["loss"])

    def test_fp32_computation(self, synthetic_batch):
        """Test that compute_in_fp32 option works."""
        # Convert to fp16
        x = synthetic_batch["x"].half()
        x_hat = synthetic_batch["x_hat"].half()
        mu = synthetic_batch["mu"].half()
        logvar = synthetic_batch["logvar"].half()
        z = synthetic_batch["z"].half()

        result = compute_tcvae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            n_data=1000,
            beta_tc=6.0,
            compute_in_fp32=True,
        )

        assert torch.isfinite(result["loss"]), "Loss should be finite with fp16 inputs"


class TestBetaTCSchedule:
    """Tests for beta_tc annealing schedule."""

    def test_schedule_starts_at_zero(self):
        """Test that beta_tc is 0 at epoch 0."""
        beta = get_beta_tc_schedule(
            epoch=0,
            beta_tc_target=6.0,
            beta_tc_annealing_epochs=40,
        )
        assert beta == 0.0, f"Expected 0.0 at epoch 0, got {beta}"

    def test_schedule_reaches_target(self):
        """Test that beta_tc reaches target after annealing."""
        beta = get_beta_tc_schedule(
            epoch=40,
            beta_tc_target=6.0,
            beta_tc_annealing_epochs=40,
        )
        assert beta == 6.0, f"Expected 6.0 at epoch 40, got {beta}"

    def test_schedule_stays_at_target(self):
        """Test that beta_tc stays constant after annealing."""
        beta_40 = get_beta_tc_schedule(
            epoch=40,
            beta_tc_target=6.0,
            beta_tc_annealing_epochs=40,
        )
        beta_100 = get_beta_tc_schedule(
            epoch=100,
            beta_tc_target=6.0,
            beta_tc_annealing_epochs=40,
        )
        assert beta_40 == beta_100 == 6.0

    def test_schedule_linear_interpolation(self):
        """Test that schedule interpolates linearly."""
        beta_20 = get_beta_tc_schedule(
            epoch=20,
            beta_tc_target=6.0,
            beta_tc_annealing_epochs=40,
        )
        # At epoch 20 (halfway), should be 3.0
        assert abs(beta_20 - 3.0) < 1e-6, f"Expected 3.0 at epoch 20, got {beta_20}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
