"""DIP-VAE loss with moment matching for disentanglement.

This module implements the Disentangled Inferred Prior VAE (DIP-VAE) objective
that regularizes the aggregated posterior distribution to match a factorized prior
through covariance matching.

DIP-VAE-II variant: Matches both on-diagonal and off-diagonal covariance terms.
- λ_od penalty: Encourages off-diagonal covariances → 0 (independence)
- λ_d penalty: Encourages diagonal variances → 1 (unit variance)

**CRITICAL**: Uses correct aggregated posterior covariance estimator:
    Cov_q(z) ≈ Cov_batch(μ) + mean_batch(diag(exp(logvar)))

This combines between-sample variance (from μ) and within-sample variance (from exp(logvar)).

Reference:
    Kumar et al. "Variational Inference of Disentangled Latent Concepts from
    Unlabeled Observations" (ICLR 2018), arXiv:1711.00848
"""

import math
from typing import Dict, Optional

import torch


def compute_dipvae_covariance(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    compute_in_fp32: bool = True,
) -> torch.Tensor:
    """Compute aggregated posterior covariance for DIP-VAE-II.

    Cov_q(z) ≈ Cov_batch(μ) + mean_batch(diag(exp(logvar)))

    This is the correct estimator for the aggregated posterior, combining:
    - Between-sample variance: Cov_batch(μ)
    - Within-sample variance: mean(diag(exp(logvar)))

    Args:
        mu: Posterior means [B, d]
        logvar: Posterior log-variances [B, d]
        compute_in_fp32: If True, compute in float32 for stability

    Returns:
        Covariance matrix [d, d]
    """
    orig_dtype = mu.dtype

    if compute_in_fp32:
        mu = mu.float()
        logvar = logvar.float()

    # 1) Cov_batch(μ): Between-sample covariance
    # Use (batch_size - 1) for unbiased sample covariance (Bessel's correction)
    mu_centered = mu - mu.mean(dim=0, keepdim=True)  # [B, d]
    batch_size = mu.size(0)
    cov_mu = torch.mm(mu_centered.t(), mu_centered) / (batch_size - 1)  # [d, d]

    # 2) Mean encoder variance: Within-sample variance
    mean_encoder_var = torch.exp(logvar).mean(dim=0)  # [d]
    cov_var = torch.diag(mean_encoder_var)  # [d, d]

    # 3) Total aggregated posterior covariance
    cov_q = cov_mu + cov_var  # [d, d]

    # Keep in FP32 if requested (do NOT cast back yet - penalties need FP32 stability)
    return cov_q


def compute_dipvae_covariance_ddp_aware(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    compute_in_fp32: bool = True,
    use_ddp_gather: bool = True,
) -> torch.Tensor:
    """Compute covariance with optional all-gather for DDP.

    When DDP is active, all-gather μ and logvar across ranks to compute
    covariance on the full effective batch (B × world_size).

    Args:
        mu: Posterior means [B, d]
        logvar: Posterior log-variances [B, d]
        compute_in_fp32: If True, compute in float32 for stability
        use_ddp_gather: If True, use all-gather when DDP is active

    Returns:
        Covariance matrix [d, d]
    """
    # Check if DDP is active
    if (use_ddp_gather and
        torch.distributed.is_available() and
        torch.distributed.is_initialized() and
        torch.distributed.get_world_size() > 1):
        # Gather mu and logvar across all ranks
        world_size = torch.distributed.get_world_size()

        # All-gather μ
        mu_list = [torch.zeros_like(mu) for _ in range(world_size)]
        torch.distributed.all_gather(mu_list, mu)
        mu_gathered = torch.cat(mu_list, dim=0)  # [B*world_size, d]

        # All-gather logvar
        logvar_list = [torch.zeros_like(logvar) for _ in range(world_size)]
        torch.distributed.all_gather(logvar_list, logvar)
        logvar_gathered = torch.cat(logvar_list, dim=0)  # [B*world_size, d]

        # Compute covariance on gathered tensors
        return compute_dipvae_covariance(mu_gathered, logvar_gathered, compute_in_fp32)
    else:
        # Single GPU or DDP disabled: use local batch only
        return compute_dipvae_covariance(mu, logvar, compute_in_fp32)


def compute_dipvae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    lambda_od: float = 10.0,
    lambda_d: float = 5.0,
    compute_in_fp32: bool = True,
    reduction: str = "mean",
    kl_free_bits: float = 0.0,
    kl_free_bits_mode: str = "batch_mean",
    use_ddp_gather: bool = True,
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute DIP-VAE-II loss with covariance regularization.

    Loss = recon + beta * KL + λ_od × ||Cov_offdiag||_F² + λ_d × ||diag(Cov) - 1||_2²

    Uses CORRECT DIP-VAE-II covariance estimator:
        Cov_q(z) ≈ Cov_batch(μ) + mean_batch(diag(exp(logvar)))

    Args:
        x: Original input [B, C, D, H, W]
        x_hat: Reconstruction [B, C, D, H, W]
        mu: Posterior means [B, d]
        logvar: Posterior log-variances [B, d] (should be clamped)
        z: Sampled latents [B, d] (not used for covariance, kept for API consistency)
        lambda_od: Weight for off-diagonal covariance penalty (default: 10.0)
        lambda_d: Weight for diagonal covariance penalty (default: 5.0)
        compute_in_fp32: If True, compute covariance in float32 for stability
        reduction: Loss reduction strategy ("mean" or "sum")
                  "mean" recommended for numerical stability with large volumes
        kl_free_bits: Minimum KL threshold per latent dimension (nats)
        kl_free_bits_mode: Free Bits clamping mode ("batch_mean" or "per_sample")
        use_ddp_gather: If True, use all-gather when DDP active (default: True)
        beta: Weight for KL divergence term (default: 1.0)

    Returns:
        Dict with keys:
            - loss: Total loss
            - recon: Reconstruction loss (MSE, scaled by reduction)
            - recon_sum: Reconstruction sum (diagnostic for comparability)
            - kl_raw: Analytic KL before Free Bits
            - kl_constrained: KL after Free Bits clamping
            - cov_penalty_od: Off-diagonal covariance penalty (weighted)
            - cov_penalty_d: Diagonal covariance penalty (weighted)
            - cov_offdiag_fro: Frobenius norm of off-diagonal elements (diagnostic)
            - cov_diag_l2: L2 distance from unit variance (diagnostic)
    """
    # Get dimensions
    batch_size = mu.size(0)
    z_dim = mu.size(1)

    # =========================================================================
    # Reconstruction loss: MSE
    # =========================================================================
    if compute_in_fp32:
        squared_error = (x_hat.float() - x.float()) ** 2
    else:
        squared_error = (x_hat - x) ** 2

    num_voxels = x.shape[2] * x.shape[3] * x.shape[4]  # D * H * W

    if reduction == "mean":
        recon = torch.mean(squared_error)
        recon_sum = recon * (batch_size * x.shape[1] * num_voxels)  # Diagnostic
    elif reduction == "sum":
        recon = torch.sum(squared_error)
        recon_sum = recon  # Already a sum
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    # =========================================================================
    # KL divergence with Free Bits
    # =========================================================================
    # Compute analytic per-dimension KL
    # KL per dimension per sample: 0.5 * (exp(logvar) + mu^2 - 1 - logvar)
    if compute_in_fp32:
        mu_fp32 = mu.float()
        logvar_fp32 = logvar.float()
        kl_per_dim = 0.5 * (torch.exp(logvar_fp32) + mu_fp32 ** 2 - 1.0 - logvar_fp32)
    else:
        kl_per_dim = 0.5 * (torch.exp(logvar) + mu ** 2 - 1.0 - logvar)

    # Clamp to ensure non-negative (numerical stability)
    kl_per_dim = torch.clamp(kl_per_dim, min=0.0)

    # Compute raw KL (for logging)
    kl_raw_per_dim_mean = kl_per_dim.mean(dim=0)  # [d]
    kl_raw = kl_raw_per_dim_mean.sum()  # scalar

    # Apply free bits based on mode
    if kl_free_bits > 0.0:
        if kl_free_bits_mode == "batch_mean":
            # Clamp batch-mean per dimension
            kl_constrained_per_dim = torch.clamp(kl_raw_per_dim_mean, min=kl_free_bits)
            kl_constrained = kl_constrained_per_dim.sum()
        elif kl_free_bits_mode == "per_sample":
            # Clamp each sample per dimension
            kl_constrained_per_sample = torch.clamp(kl_per_dim, min=kl_free_bits)
            kl_constrained = kl_constrained_per_sample.sum(dim=1).mean()
        else:
            raise ValueError(
                f"Invalid kl_free_bits_mode: {kl_free_bits_mode}. "
                "Must be 'per_sample' or 'batch_mean'."
            )
    else:
        kl_constrained = kl_raw

    # NOTE: KL is already a batch mean (mean over batch, sum over dims).
    # Do NOT divide by batch_size again - that would incorrectly scale by 1/B².
    # The `reduction` parameter only affects reconstruction loss scaling.

    # =========================================================================
    # Covariance regularization (DIP-VAE-II)
    # =========================================================================
    # Compute aggregated posterior covariance using CORRECT formula
    cov_q = compute_dipvae_covariance_ddp_aware(
        mu, logvar, compute_in_fp32, use_ddp_gather
    )  # [d, d]

    # Extract diagonal and off-diagonal elements
    diag_cov = torch.diag(cov_q)  # [d]

    # Off-diagonal penalty: ||Cov_offdiag||_F^2
    # Create mask to zero out diagonal elements; the resulting matrix has zeros on diagonal
    # Frobenius norm of this matrix equals sqrt(sum of squared off-diagonal elements)
    mask = torch.ones_like(cov_q) - torch.eye(z_dim, device=cov_q.device, dtype=cov_q.dtype)
    cov_offdiag = cov_q * mask  # [d, d] matrix with zeros on diagonal
    cov_offdiag_fro = torch.norm(cov_offdiag, p="fro")  # Frobenius norm = sqrt(sum(offdiag^2))
    cov_penalty_od = lambda_od * (cov_offdiag_fro ** 2)

    # Diagonal penalty: ||diag(Cov) - 1||_2^2
    cov_diag_l2 = torch.norm(diag_cov - 1.0, p=2)  # L2 distance from unit variance (diagnostic)
    cov_penalty_d = lambda_d * (cov_diag_l2 ** 2)

    # NOTE: Keep all losses in FP32 when compute_in_fp32=True for numerical stability.
    # Lightning/GradScaler handles mixed precision automatically; no need to cast back.

    # =========================================================================
    # Total loss
    # =========================================================================
    total = recon + (beta * kl_constrained) + cov_penalty_od + cov_penalty_d

    # Check for non-finite values
    if not torch.isfinite(total):
        raise RuntimeError(
            f"Non-finite loss detected: total={total.item()}, "
            f"recon={recon.item()}, kl={kl_constrained.item()}, "
            f"cov_od={cov_penalty_od.item()}, cov_d={cov_penalty_d.item()}"
        )

    return {
        "loss": total,
        "recon": recon,
        "recon_sum": recon_sum,  # Diagnostic for comparability
        "kl_raw": kl_raw,
        "kl_constrained": kl_constrained,
        "cov_penalty_od": cov_penalty_od,
        "cov_penalty_d": cov_penalty_d,
        "cov_offdiag_fro": cov_offdiag_fro,  # Diagnostic (unweighted norm)
        "cov_diag_l2": cov_diag_l2,  # Diagnostic (unweighted distance)
    }


def get_lambda_cov_schedule(
    epoch: int,
    lambda_target: float,
    lambda_annealing_epochs: int,
) -> float:
    """Compute current lambda_cov value for covariance penalty annealing.

    Linear annealing schedule:
    - lambda = 0 at epoch 0
    - Linearly increases to lambda_target over lambda_annealing_epochs
    - Stays constant at lambda_target after that

    This warmup prevents optimizer shock when introducing the covariance penalty.

    Args:
        epoch: Current epoch (0-indexed)
        lambda_target: Target lambda value after annealing
        lambda_annealing_epochs: Number of epochs for annealing

    Returns:
        Current lambda value
    """
    if lambda_annealing_epochs <= 0:
        return lambda_target

    if epoch >= lambda_annealing_epochs:
        return lambda_target

    return (epoch / lambda_annealing_epochs) * lambda_target
