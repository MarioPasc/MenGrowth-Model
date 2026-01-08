"""Verification test for loss functions after reverting Gemini's scaling fix.

This test ensures:
1. ELBO loss computes correctly without KL/N scaling
2. DIP-VAE loss computes correctly without regularization/N scaling
3. Free bits is effective (floor >> MSE_mean)
4. Loss magnitudes are in expected ranges

Run with: python tests/test_loss_functions_verification.py
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch


def test_elbo_loss():
    """Test ELBO loss without KL scaling."""
    from vae.losses.elbo import compute_elbo

    print("\n" + "=" * 70)
    print("  ELBO LOSS VERIFICATION")
    print("=" * 70)

    # Simulate realistic inputs
    B, C, D, H, W = 2, 4, 32, 32, 32  # Smaller for testing
    z_dim = 128
    N = C * D * H * W  # 131,072 for 32³

    x = torch.randn(B, C, D, H, W)
    x_hat = x + 0.1 * torch.randn_like(x)  # Small reconstruction error
    mu = torch.randn(B, z_dim) * 0.5
    logvar = torch.randn(B, z_dim) - 2  # Small variance

    # Test with mean reduction
    result = compute_elbo(
        x=x,
        x_hat=x_hat,
        mu=mu,
        logvar=logvar,
        beta=1.0,
        reduction="mean",
        kl_free_bits=0.2,
        kl_free_bits_mode="batch_mean",
    )

    print(f"\n  Input shape: {list(x.shape)}")
    print(f"  Voxels per sample (N): {N:,}")
    print(f"  Latent dim (z_dim): {z_dim}")
    print()
    print(f"  Loss components (reduction='mean'):")
    print(f"    Reconstruction (MSE_mean): {result['recon'].item():.6f}")
    print(f"    KL (raw):                  {result['kl_raw'].item():.6f}")
    print(f"    KL (after free bits):      {result['kl'].item():.6f}")
    print(f"    Total loss:                {result['loss'].item():.6f}")

    # Verify free bits floor
    free_bits_floor = z_dim * 0.2  # 25.6
    print(f"\n  Free bits analysis:")
    print(f"    Free bits floor: {free_bits_floor:.1f} nats")
    print(f"    KL after free bits: {result['kl'].item():.2f} nats")
    print(f"    MSE_mean: {result['recon'].item():.4f}")
    print(f"    KL/MSE ratio: {result['kl'].item() / result['recon'].item():.2f}")

    # Assertions
    assert result["recon"].item() < 1.0, "MSE_mean should be < 1 for small noise"
    assert result["kl"].item() >= free_bits_floor * 0.9, "KL should be near free bits floor"
    assert result["loss"].item() > result["recon"].item(), "Total loss should include KL"

    print("\n  ✓ ELBO loss verification PASSED")
    return True


def test_dipvae_loss():
    """Test DIP-VAE loss without regularization scaling."""
    from vae.losses.dipvae import compute_dipvae_loss

    print("\n" + "=" * 70)
    print("  DIP-VAE LOSS VERIFICATION")
    print("=" * 70)

    # Simulate realistic inputs
    B, C, D, H, W = 2, 4, 32, 32, 32
    z_dim = 128
    N = C * D * H * W

    x = torch.randn(B, C, D, H, W)
    x_hat = x + 0.1 * torch.randn_like(x)
    mu = torch.randn(B, z_dim) * 0.5
    logvar = torch.randn(B, z_dim) - 2
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    # Test with mean reduction and typical DIP-VAE hyperparameters
    result = compute_dipvae_loss(
        x=x,
        x_hat=x_hat,
        mu=mu,
        logvar=logvar,
        z=z,
        lambda_od=10.0,
        lambda_d=5.0,
        compute_in_fp32=True,
        reduction="mean",
        kl_free_bits=0.2,
        kl_free_bits_mode="batch_mean",
        beta=1.0,
    )

    print(f"\n  Input shape: {list(x.shape)}")
    print(f"  Voxels per sample (N): {N:,}")
    print(f"  Latent dim (z_dim): {z_dim}")
    print()
    print(f"  Loss components (reduction='mean'):")
    print(f"    Reconstruction (MSE_mean): {result['recon'].item():.6f}")
    print(f"    KL (raw):                  {result['kl_raw'].item():.6f}")
    print(f"    KL (constrained):          {result['kl_constrained'].item():.6f}")
    print(f"    Cov penalty OD:            {result['cov_penalty_od'].item():.6f}")
    print(f"    Cov penalty D:             {result['cov_penalty_d'].item():.6f}")
    print(f"    Total loss:                {result['loss'].item():.6f}")

    print(f"\n  Covariance diagnostics:")
    print(f"    Off-diag Frobenius norm:   {result['cov_offdiag_fro'].item():.4f}")
    print(f"    Diag L2 from 1:            {result['cov_diag_l2'].item():.4f}")

    # Verify regularization is not negligible
    total_reg = result["kl_constrained"].item() + result["cov_penalty_od"].item() + result["cov_penalty_d"].item()
    reg_fraction = total_reg / result["loss"].item()
    print(f"\n  Loss balance:")
    print(f"    Regularization fraction:   {reg_fraction * 100:.2f}%")
    print(f"    (Should be >> 1% if regularization is effective)")

    # Assertions
    assert result["recon"].item() < 1.0, "MSE_mean should be < 1 for small noise"
    assert result["cov_penalty_od"].item() > 0.1, "Cov penalty should be significant (not scaled to negligible)"
    assert reg_fraction > 0.5, "Regularization should dominate with mean reduction"

    print("\n  ✓ DIP-VAE loss verification PASSED")
    return True


def test_loss_not_scaled_by_n():
    """Verify that KL is NOT scaled by 1/N (the reverted change)."""
    from vae.losses.elbo import compute_elbo

    print("\n" + "=" * 70)
    print("  VERIFY: KL IS NOT SCALED BY 1/N")
    print("=" * 70)

    # Test with two different volume sizes
    B, z_dim = 2, 128

    # Same mu/logvar for both
    mu = torch.randn(B, z_dim) * 0.5
    logvar = torch.randn(B, z_dim) - 2

    # Small volume
    x_small = torch.randn(B, 4, 16, 16, 16)
    x_hat_small = x_small + 0.1 * torch.randn_like(x_small)

    # Larger volume
    x_large = torch.randn(B, 4, 32, 32, 32)
    x_hat_large = x_large + 0.1 * torch.randn_like(x_large)

    result_small = compute_elbo(x_small, x_hat_small, mu, logvar, reduction="mean")
    result_large = compute_elbo(x_large, x_hat_large, mu, logvar, reduction="mean")

    print(f"\n  Small volume: 4×16×16×16 = {4*16**3:,} voxels")
    print(f"    KL: {result_small['kl'].item():.4f}")

    print(f"\n  Large volume: 4×32×32×32 = {4*32**3:,} voxels")
    print(f"    KL: {result_large['kl'].item():.4f}")

    # KL should be the same regardless of volume size
    # (if it were scaled by 1/N, larger volumes would have smaller KL)
    kl_ratio = result_small["kl"].item() / result_large["kl"].item()
    print(f"\n  KL ratio (small/large): {kl_ratio:.4f}")
    print(f"  (Should be ~1.0 if KL is NOT scaled by volume size)")

    assert 0.9 < kl_ratio < 1.1, "KL should NOT depend on volume size"

    print("\n  ✓ KL is correctly NOT scaled by 1/N")
    return True


def test_fp16_safe():
    """Verify loss values are FP16-safe with mean reduction.

    Note: With random initialization, covariance penalties can be very large.
    In practice, the model starts with mu near zero (due to weight init),
    and lambda is ramped up slowly via lambda_start_epoch + annealing.
    """
    from vae.losses.dipvae import compute_dipvae_loss

    print("\n" + "=" * 70)
    print("  FP16 SAFETY CHECK")
    print("=" * 70)

    B, C, D, H, W = 2, 4, 32, 32, 32
    z_dim = 128

    x = torch.randn(B, C, D, H, W)
    x_hat = x + torch.randn_like(x)  # Larger error

    # Realistic initialization: mu near zero (like a freshly initialized encoder)
    mu = torch.randn(B, z_dim) * 0.1  # Small mu (encoder output before training)
    logvar = torch.zeros(B, z_dim) - 2  # Moderate variance

    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    # Test with lambda=0 (epoch < lambda_start_epoch)
    result_no_cov = compute_dipvae_loss(
        x=x, x_hat=x_hat, mu=mu, logvar=logvar, z=z,
        lambda_od=0.0, lambda_d=0.0, reduction="mean",
    )

    # Test with full lambda (after annealing)
    result_full_cov = compute_dipvae_loss(
        x=x, x_hat=x_hat, mu=mu, logvar=logvar, z=z,
        lambda_od=10.0, lambda_d=5.0, reduction="mean",
    )

    fp16_max = 65504.0
    print(f"\n  FP16 max representable: {fp16_max}")
    print()
    print(f"  Before lambda warmup (lambda=0):")
    print(f"    Total loss: {result_no_cov['loss'].item():.2f}")
    print(f"    Safe margin: {fp16_max / result_no_cov['loss'].item():.0f}×")
    print()
    print(f"  After lambda warmup (lambda_od=10, lambda_d=5):")
    print(f"    Total loss: {result_full_cov['loss'].item():.2f}")
    print(f"    Cov penalty: {result_full_cov['cov_penalty_od'].item() + result_full_cov['cov_penalty_d'].item():.2f}")
    print(f"    Safe margin: {fp16_max / result_full_cov['loss'].item():.1f}×")

    print(f"\n  Note: Covariance penalties are large because ||Cov_offdiag||_F² ~ z_dim²")
    print(f"  In practice, delayed start + annealing prevents sudden large losses.")

    # With realistic mu, losses should be FP16-safe
    assert result_no_cov["loss"].item() < fp16_max, "Pre-warmup loss should be FP16-safe"
    # After warmup, with realistic mu, should still be safe
    assert result_full_cov["loss"].item() < fp16_max, "Post-warmup loss should be FP16-safe"

    print("\n  ✓ FP16 safety check PASSED")
    return True


if __name__ == "__main__":
    all_passed = True

    try:
        all_passed &= test_elbo_loss()
    except Exception as e:
        print(f"\n  ✗ ELBO test FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_dipvae_loss()
    except Exception as e:
        print(f"\n  ✗ DIP-VAE test FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_loss_not_scaled_by_n()
    except Exception as e:
        print(f"\n  ✗ KL scaling test FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_fp16_safe()
    except Exception as e:
        print(f"\n  ✗ FP16 safety test FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL VERIFICATION TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
    print("=" * 70 + "\n")

    sys.exit(0 if all_passed else 1)
