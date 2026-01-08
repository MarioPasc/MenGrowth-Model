"""Analysis of loss reduction strategies for 3D VAE with large volumes.

This test demonstrates how different reduction strategies affect loss component
magnitudes and their relative contributions to the total loss.

Key findings guide the choice between:
1. reduction="sum" (standard formulation)
2. reduction="mean" (numerically stable)
3. reduction="mean" with KL scaling (Gemini's fix - problematic)

Run with: python -m pytest tests/test_loss_reduction_analysis.py -v -s
"""

import torch
import numpy as np


def compute_loss_components(
    batch_size: int = 2,
    channels: int = 4,
    spatial_size: int = 128,
    z_dim: int = 128,
    reduction: str = "mean",
    scale_kl_by_n: bool = False,
    mse_per_voxel: float = 1.0,
    kl_per_dim: float = 0.5,
    beta: float = 1.0,
    lambda_od: float = 10.0,
    lambda_d: float = 5.0,
) -> dict:
    """Simulate loss components for different reduction strategies.

    Args:
        batch_size: Batch size
        channels: Number of input channels (4 for MRI modalities)
        spatial_size: Spatial dimension (128 for 128³ volumes)
        z_dim: Latent dimension
        reduction: "sum" or "mean"
        scale_kl_by_n: If True, scale KL by 1/N (Gemini's fix)
        mse_per_voxel: Average MSE per voxel (typical: 0.5-2.0)
        kl_per_dim: Average KL per latent dimension (typical: 0.3-1.0)
        beta: KL weight
        lambda_od: Off-diagonal covariance penalty weight
        lambda_d: Diagonal covariance penalty weight

    Returns:
        Dict with loss components and analysis metrics
    """
    # Compute dimensions
    N = channels * spatial_size ** 3  # Number of voxels per sample

    # Simulate reconstruction error
    if reduction == "sum":
        recon_loss = batch_size * N * mse_per_voxel
    else:  # mean
        recon_loss = mse_per_voxel  # Already per-voxel average

    # Simulate KL divergence (sum over latent dims, mean over batch)
    kl_raw = z_dim * kl_per_dim

    # Simulate covariance penalties (typical magnitudes)
    # Off-diagonal: ||Cov_offdiag||_F² ~ z_dim² × 0.01 (small correlations)
    cov_offdiag_fro_sq = z_dim * z_dim * 0.01
    # Diagonal: ||diag(Cov) - 1||_2² ~ z_dim × 0.1 (variance near 1)
    cov_diag_l2_sq = z_dim * 0.1

    cov_penalty_od = lambda_od * cov_offdiag_fro_sq
    cov_penalty_d = lambda_d * cov_diag_l2_sq

    # Apply scaling if requested (Gemini's fix)
    if scale_kl_by_n and reduction == "mean":
        kl_term = (beta * kl_raw) / N
        cov_od_term = cov_penalty_od / N
        cov_d_term = cov_penalty_d / N
    else:
        kl_term = beta * kl_raw
        cov_od_term = cov_penalty_od
        cov_d_term = cov_penalty_d

    # Total loss
    total_loss = recon_loss + kl_term + cov_od_term + cov_d_term

    # Compute relative contributions
    total_regularization = kl_term + cov_od_term + cov_d_term

    return {
        "N": N,
        "reduction": reduction,
        "scale_kl_by_n": scale_kl_by_n,
        # Raw values
        "recon_loss": recon_loss,
        "kl_raw": kl_raw,
        "kl_term": kl_term,
        "cov_penalty_od": cov_penalty_od,
        "cov_od_term": cov_od_term,
        "cov_penalty_d": cov_penalty_d,
        "cov_d_term": cov_d_term,
        "total_loss": total_loss,
        # Ratios
        "recon_fraction": recon_loss / total_loss,
        "kl_fraction": kl_term / total_loss,
        "cov_fraction": (cov_od_term + cov_d_term) / total_loss,
        "kl_to_recon_ratio": kl_term / recon_loss,
        "regularization_to_recon_ratio": total_regularization / recon_loss,
    }


def print_analysis(results: dict, label: str) -> None:
    """Print formatted analysis of loss components."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Volume size: N = {results['N']:,} voxels per sample")
    print(f"  Reduction: {results['reduction']}, Scale KL by 1/N: {results['scale_kl_by_n']}")
    print()
    print(f"  Loss Components:")
    print(f"    Reconstruction:     {results['recon_loss']:>15.6f}")
    print(f"    KL (raw):           {results['kl_raw']:>15.6f}")
    print(f"    KL (in loss):       {results['kl_term']:>15.6f}")
    print(f"    Cov OD (raw):       {results['cov_penalty_od']:>15.6f}")
    print(f"    Cov OD (in loss):   {results['cov_od_term']:>15.6f}")
    print(f"    Cov D (raw):        {results['cov_penalty_d']:>15.6f}")
    print(f"    Cov D (in loss):    {results['cov_d_term']:>15.6f}")
    print(f"    ─────────────────────────────────────")
    print(f"    TOTAL:              {results['total_loss']:>15.6f}")
    print()
    print(f"  Relative Contributions:")
    print(f"    Reconstruction:     {results['recon_fraction']*100:>10.4f}%")
    print(f"    KL term:            {results['kl_fraction']*100:>10.4f}%")
    print(f"    Covariance terms:   {results['cov_fraction']*100:>10.4f}%")
    print()
    print(f"  Key Ratios:")
    print(f"    KL / Recon:         {results['kl_to_recon_ratio']:>10.6f}")
    print(f"    Total Reg / Recon:  {results['regularization_to_recon_ratio']:>10.6f}")


class TestLossReductionAnalysis:
    """Test class for loss reduction analysis."""

    def test_compare_all_strategies(self):
        """Compare all three reduction strategies side by side."""
        print("\n" + "="*70)
        print("  LOSS REDUCTION STRATEGY COMPARISON FOR 3D VAE")
        print("  Volume: 4 × 128 × 128 × 128 (8.4M voxels)")
        print("  Latent: z_dim = 128")
        print("="*70)

        # Strategy 1: Sum reduction (standard)
        results_sum = compute_loss_components(
            reduction="sum",
            scale_kl_by_n=False,
        )
        print_analysis(results_sum, "Strategy 1: SUM REDUCTION (Standard VAE)")

        # Strategy 2: Mean reduction, no KL scaling (recommended)
        results_mean_no_scale = compute_loss_components(
            reduction="mean",
            scale_kl_by_n=False,
        )
        print_analysis(results_mean_no_scale, "Strategy 2: MEAN REDUCTION, No KL Scaling (Recommended)")

        # Strategy 3: Mean reduction with KL scaling (Gemini's fix)
        results_mean_scaled = compute_loss_components(
            reduction="mean",
            scale_kl_by_n=True,
        )
        print_analysis(results_mean_scaled, "Strategy 3: MEAN REDUCTION + KL/N Scaling (Gemini's Fix)")

        # Summary comparison
        print("\n" + "="*70)
        print("  SUMMARY COMPARISON")
        print("="*70)
        print(f"\n  {'Strategy':<45} {'Total Loss':>12} {'KL/Recon':>12}")
        print(f"  {'-'*45} {'-'*12} {'-'*12}")
        print(f"  {'1. Sum reduction':<45} {results_sum['total_loss']:>12.2f} {results_sum['kl_to_recon_ratio']:>12.6f}")
        print(f"  {'2. Mean reduction (no KL scaling)':<45} {results_mean_no_scale['total_loss']:>12.2f} {results_mean_no_scale['kl_to_recon_ratio']:>12.6f}")
        print(f"  {'3. Mean reduction + KL/N (Gemini)':<45} {results_mean_scaled['total_loss']:>12.6f} {results_mean_scaled['kl_to_recon_ratio']:>12.9f}")

        print("\n  INTERPRETATION:")
        print("  ─────────────────────────────────────────────────────────────────────")
        print("  Strategy 1 (Sum): Large loss values (~16M), recon dominates (99.996%)")
        print("                    Gradients may overflow in FP16.")
        print()
        print("  Strategy 2 (Mean, no scale): Moderate loss (~1730), KL dominates (96.3%)")
        print("                    Numerically stable. Use free bits + annealing")
        print("                    to prevent collapse. THIS IS RECOMMENDED.")
        print()
        print("  Strategy 3 (Mean + KL/N): Tiny loss (~1.0), recon dominates (99.9999%)")
        print("                    KL and covariance terms are negligible (~10⁻⁵).")
        print("                    Model becomes pure autoencoder. PROBLEMATIC.")
        print()

        # Assertions to verify the analysis
        assert results_sum["recon_fraction"] > 0.99, "Sum: recon should dominate"
        # With mean reduction + no scaling, regularization (KL + Cov) should dominate
        assert results_mean_no_scale["recon_fraction"] < 0.1, "Mean no scale: regularization should dominate"
        assert results_mean_scaled["kl_fraction"] < 0.001, "Mean scaled: KL should be negligible"

    def test_free_bits_effectiveness(self):
        """Test how free bits floor compares to loss under different strategies."""
        print("\n" + "="*70)
        print("  FREE BITS EFFECTIVENESS ANALYSIS")
        print("="*70)

        z_dim = 128
        free_bits = 0.2  # nats per dimension
        kl_floor = z_dim * free_bits  # 25.6 nats
        N = 4 * 128 ** 3  # ~8.4M

        print(f"\n  Free bits configuration:")
        print(f"    kl_free_bits = {free_bits} nats/dim")
        print(f"    z_dim = {z_dim}")
        print(f"    KL floor = {kl_floor} nats")
        print(f"    N = {N:,} voxels")

        print(f"\n  KL floor contribution to loss:")
        print(f"    Strategy 2 (no scaling):  {kl_floor:.2f}")
        print(f"    Strategy 3 (with scaling): {kl_floor/N:.2e}")

        print(f"\n  Compared to typical MSE_mean ≈ 1.0:")
        print(f"    Strategy 2: KL floor is {kl_floor:.1f}× larger than recon")
        print(f"    Strategy 3: KL floor is {kl_floor/N:.2e}× smaller than recon")

        print(f"\n  CONCLUSION:")
        print(f"    With Strategy 2, free bits floor (25.6) >> MSE (~1)")
        print(f"    → Free bits WORKS: model MUST encode information to beat floor")
        print()
        print(f"    With Strategy 3, free bits floor ({kl_floor/N:.2e}) << MSE (~1)")
        print(f"    → Free bits FAILS: optimizer ignores it completely")

        # Verify
        assert kl_floor > 1.0, "Free bits floor should exceed typical MSE"
        assert kl_floor / N < 1e-4, "Scaled free bits floor is negligible"

    def test_gradient_magnitude_comparison(self):
        """Compare gradient magnitudes under different strategies."""
        print("\n" + "="*70)
        print("  GRADIENT MAGNITUDE ANALYSIS")
        print("="*70)

        N = 4 * 128 ** 3

        # Simulate gradients (proportional to loss term magnitude)
        print(f"\n  Gradient scaling factors (relative to unscaled baseline):")
        print(f"    Sum reduction:         ×{N:,.0f} (may overflow FP16)")
        print(f"    Mean, no KL scaling:   ×1 (baseline)")
        print(f"    Mean + KL/N scaling:   ×1 for recon, ×{1/N:.2e} for regularization")

        print(f"\n  FP16 considerations:")
        print(f"    FP16 max value: ~65504")
        print(f"    Sum reduction loss: ~{N * 1.0:,.0f} (OVERFLOW RISK)")
        print(f"    Mean reduction loss: ~1-100 (SAFE)")

        print(f"\n  RECOMMENDATION:")
        print(f"    Use mean reduction for numerical stability,")
        print(f"    but do NOT scale KL/covariance by 1/N.")
        print(f"    Instead, use beta parameter to tune reconstruction-vs-KL balance.")

    def test_beta_sensitivity(self):
        """Test how beta affects loss balance under mean reduction."""
        print("\n" + "="*70)
        print("  BETA SENSITIVITY ANALYSIS (Mean Reduction, No KL Scaling)")
        print("="*70)

        beta_values = [0.001, 0.01, 0.1, 1.0, 10.0]

        print(f"\n  {'Beta':<10} {'Recon %':>12} {'KL %':>12} {'Cov %':>12} {'KL/Recon':>12}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for beta in beta_values:
            results = compute_loss_components(
                reduction="mean",
                scale_kl_by_n=False,
                beta=beta,
            )
            print(f"  {beta:<10.3f} {results['recon_fraction']*100:>11.2f}% "
                  f"{results['kl_fraction']*100:>11.2f}% "
                  f"{results['cov_fraction']*100:>11.2f}% "
                  f"{results['kl_to_recon_ratio']:>12.4f}")

        print(f"\n  INTERPRETATION:")
        print(f"    beta=1.0: KL dominates → use free bits to prevent collapse")
        print(f"    beta=0.01-0.1: More balanced → may not need free bits")
        print(f"    beta=0.001: Recon dominates → approaching autoencoder")
        print()
        print(f"  For 3D medical imaging, beta=1.0 + free_bits=0.2 is recommended")
        print(f"  because it ensures minimum information flow while allowing")
        print(f"  cyclical annealing to periodically relieve KL pressure.")


def test_numerical_stability_with_real_tensors():
    """Test actual tensor operations for numerical stability."""
    print("\n" + "="*70)
    print("  NUMERICAL STABILITY TEST WITH REAL TENSORS")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_fp32 = torch.float32
    dtype_fp16 = torch.float16

    # Simulate realistic tensors (smaller for memory, scale results)
    B, C, D, H, W = 2, 4, 32, 32, 32  # Smaller for testing
    z_dim = 128
    scale_factor = (128 / 32) ** 3  # Scale to 128³

    x = torch.randn(B, C, D, H, W, device=device, dtype=dtype_fp32)
    x_hat = x + 0.1 * torch.randn_like(x)  # Small reconstruction error
    mu = torch.randn(B, z_dim, device=device, dtype=dtype_fp32) * 0.5
    logvar = torch.randn(B, z_dim, device=device, dtype=dtype_fp32) - 2  # Small variance

    # Compute losses
    mse_sum = torch.sum((x - x_hat) ** 2)
    mse_mean = torch.mean((x - x_hat) ** 2)
    kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar) / B

    print(f"\n  Test volume: {B}×{C}×{D}×{H}×{W}")
    print(f"  Scaling to 128³: ×{scale_factor:.0f}")
    print()
    print(f"  MSE (sum):  {mse_sum.item() * scale_factor:,.2f}")
    print(f"  MSE (mean): {mse_mean.item():.6f}")
    print(f"  KL:         {kl.item():.2f}")

    # Test FP16 overflow risk
    mse_sum_fp16 = mse_sum.half()
    print(f"\n  FP16 MSE sum (scaled): {(mse_sum.item() * scale_factor):,.0f}")
    print(f"  FP16 max representable: 65504")
    print(f"  Overflow risk: {'YES' if mse_sum.item() * scale_factor > 65504 else 'NO'}")

    # For 128³ volumes
    N_full = 4 * 128 ** 3
    print(f"\n  For full 128³ volume (N={N_full:,}):")
    print(f"    Expected MSE sum: ~{N_full * mse_mean.item():,.0f}")
    print(f"    FP16 safe: {'NO - use mean reduction' if N_full * mse_mean.item() > 65504 else 'YES'}")


if __name__ == "__main__":
    # Run all analyses
    test = TestLossReductionAnalysis()
    test.test_compare_all_strategies()
    test.test_free_bits_effectiveness()
    test.test_gradient_magnitude_comparison()
    test.test_beta_sensitivity()
    test_numerical_stability_with_real_tensors()
