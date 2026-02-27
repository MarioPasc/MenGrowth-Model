# tests/growth/test_semantic_loss_sdp.py
"""Tests for BUG-2: SDP semantic loss double normalization fix.

Verifies that SemanticRegressionLoss computes plain MSE (mean over B×k_p),
not MSE/k_p (which was the double-normalization bug).
"""

import torch
import torch.nn.functional as F

from growth.losses.semantic import SemanticRegressionLoss


class TestSemanticRegressionLossNormalization:
    """Verify the double-normalization fix."""

    def test_loss_magnitude_independent_of_k_p(self):
        """Same element-wise error should give same MSE regardless of k_p.

        With the bug (MSE/k_p), k_p=4 would give 1/4 of the loss of k_p=1.
        After fix, torch.mean() normalizes correctly for any k_p.
        """
        torch.manual_seed(42)

        # Create losses with unit weights so we can compare raw MSE
        loss_fn = SemanticRegressionLoss(lambda_vol=1.0, lambda_loc=1.0, lambda_shape=1.0)
        B = 8

        # k_p=1: single-dim predictions
        pred_1 = {"vol": torch.randn(B, 1), "loc": torch.randn(B, 1), "shape": torch.randn(B, 1)}
        target_1 = {"vol": pred_1["vol"] + 0.5, "loc": pred_1["loc"] + 0.5, "shape": pred_1["shape"] + 0.5}

        # k_p=4: multi-dim predictions with SAME per-element error
        pred_4 = {
            "vol": torch.randn(B, 4),
            "loc": torch.randn(B, 4),
            "shape": torch.randn(B, 4),
        }
        target_4 = {
            "vol": pred_4["vol"] + 0.5,
            "loc": pred_4["loc"] + 0.5,
            "shape": pred_4["shape"] + 0.5,
        }

        _, details_1 = loss_fn(pred_1, target_1)
        _, details_4 = loss_fn(pred_4, target_4)

        # MSE of constant error 0.5 should be 0.25 regardless of k_p
        for name in ["vol", "loc", "shape"]:
            assert abs(details_1[f"mse_{name}"].item() - 0.25) < 0.01, (
                f"mse_{name} for k_p=1: {details_1[f'mse_{name}'].item():.4f}, expected ~0.25"
            )
            assert abs(details_4[f"mse_{name}"].item() - 0.25) < 0.01, (
                f"mse_{name} for k_p=4: {details_4[f'mse_{name}'].item():.4f}, expected ~0.25"
            )

    def test_loss_matches_manual_computation(self):
        """Loss should match F.mse_loss(pred, target) exactly."""
        torch.manual_seed(42)

        loss_fn = SemanticRegressionLoss(lambda_vol=1.0, lambda_loc=1.0, lambda_shape=1.0)

        pred = {
            "vol": torch.randn(8, 4),
            "loc": torch.randn(8, 3),
            "shape": torch.randn(8, 3),
        }
        target = {
            "vol": torch.randn(8, 4),
            "loc": torch.randn(8, 3),
            "shape": torch.randn(8, 3),
        }

        _, details = loss_fn(pred, target)

        for name in ["vol", "loc", "shape"]:
            expected_mse = F.mse_loss(pred[name], target[name])
            actual_mse = details[f"mse_{name}"]
            assert torch.allclose(actual_mse, expected_mse, atol=1e-6), (
                f"mse_{name}: got {actual_mse.item():.6f}, "
                f"expected {expected_mse.item():.6f}"
            )

    def test_lambda_weighting(self):
        """Total loss should be sum of lambda-weighted MSEs."""
        torch.manual_seed(42)

        lambda_vol, lambda_loc, lambda_shape = 20.0, 12.0, 15.0
        loss_fn = SemanticRegressionLoss(
            lambda_vol=lambda_vol, lambda_loc=lambda_loc, lambda_shape=lambda_shape,
        )

        pred = {
            "vol": torch.randn(8, 4),
            "loc": torch.randn(8, 3),
            "shape": torch.randn(8, 3),
        }
        target = {
            "vol": torch.randn(8, 4),
            "loc": torch.randn(8, 3),
            "shape": torch.randn(8, 3),
        }

        total_loss, details = loss_fn(pred, target)

        expected_total = (
            lambda_vol * details["mse_vol"]
            + lambda_loc * details["mse_loc"]
            + lambda_shape * details["mse_shape"]
        )
        assert torch.allclose(total_loss, expected_total, atol=1e-5)
