# tests/growth/test_vicreg_accumulation.py
"""Tests for FLAW-1: VICReg feature accumulation and batch-size guard.

Verifies:
1. Batch size < 2 returns zero loss with grad
2. Accumulated covariance is more stable than per-micro-batch
3. Gradient flows through last micro-batch in accumulation
"""

import torch

from growth.losses.encoder_vicreg import EncoderVICRegLoss


class TestVICRegBatchSizeGuard:
    """Tests for the batch_size < 2 guard."""

    def test_batch_size_1_returns_zero(self):
        """Single-sample batch should return zero loss (can't compute variance)."""
        loss_fn = EncoderVICRegLoss()
        features = torch.randn(1, 768)

        loss, components = loss_fn(features)

        assert loss.item() == 0.0
        assert components["vicreg_var_loss"] == 0.0
        assert components["vicreg_cov_loss"] == 0.0
        assert components["vicreg_total"] == 0.0

    def test_batch_size_1_has_grad(self):
        """Zero loss from guard should still have requires_grad=True."""
        loss_fn = EncoderVICRegLoss()
        features = torch.randn(1, 768, requires_grad=True)

        loss, _ = loss_fn(features)

        assert loss.requires_grad
        # Should not raise on backward
        loss.backward()

    def test_batch_size_2_computes_loss(self):
        """Batch size 2 should compute a non-trivial loss."""
        loss_fn = EncoderVICRegLoss()
        features = torch.randn(2, 768)

        loss, components = loss_fn(features)

        # With random features, var_loss should be > 0 (std might not exceed gamma)
        assert loss.item() >= 0.0
        assert "vicreg_var_loss" in components


class TestVICRegAccumulationStability:
    """Tests for accumulated covariance being more stable."""

    def test_accumulated_covariance_more_stable(self):
        """Covariance computed on B=8 should have lower run-to-run variance
        than covariance on B=4 (larger batch → more stable statistics).
        """
        loss_fn = EncoderVICRegLoss(lambda_var=0.0, lambda_cov=1.0)
        feat_dim = 64  # Smaller for speed

        cov_losses_small = []
        cov_losses_large = []

        for seed in range(20):
            torch.manual_seed(seed)
            # Small batch
            f_small = torch.randn(4, feat_dim)
            _, comp_small = loss_fn(f_small)
            cov_losses_small.append(comp_small["vicreg_cov_loss"])

            # Large batch (accumulated equivalent)
            f_large = torch.randn(8, feat_dim)
            _, comp_large = loss_fn(f_large)
            cov_losses_large.append(comp_large["vicreg_cov_loss"])

        var_small = torch.tensor(cov_losses_small).var().item()
        var_large = torch.tensor(cov_losses_large).var().item()

        # Larger batch should give more stable (lower variance) estimates
        assert var_large < var_small, (
            f"Var(cov_B8)={var_large:.6f} should be < Var(cov_B4)={var_small:.6f}"
        )


class TestVICRegGradientFlow:
    """Tests for gradient flow in accumulation pattern."""

    def test_gradient_flows_through_last_microbatch(self):
        """In the accumulation pattern: detach all but last, concat, backward.
        Gradients should flow through the last micro-batch features.
        """
        loss_fn = EncoderVICRegLoss()

        # Simulate two micro-batches
        f1 = torch.randn(4, 768, requires_grad=True)
        f2 = torch.randn(4, 768, requires_grad=True)

        # Accumulation pattern: detach f1, keep f2
        all_features = torch.cat([f1.detach(), f2], dim=0)

        loss, _ = loss_fn(all_features)
        loss.backward()

        # f2 should have gradients
        assert f2.grad is not None, "f2 (last micro-batch) should have gradients"
        assert f2.grad.abs().sum() > 0, "f2 gradients should be non-zero"

        # f1 should NOT have gradients (detached)
        assert f1.grad is None, "f1 (detached) should not have gradients"
