"""β-TCVAE loss with Minibatch-Weighted Sampling (MWS) estimator.

This module implements the Total Correlation VAE objective that decomposes
the KL divergence into three interpretable terms:
- Mutual Information (MI): I(x; z)
- Total Correlation (TC): KL(q(z) || prod_j q(z_j))
- Dimension-wise KL (DWKL): sum_j KL(q(z_j) || p(z_j))

The TC term measures statistical dependence among latent dimensions.
Penalizing TC (with beta_tc > 1) encourages factorial/disentangled posteriors.

Reference:
    Chen et al. "Isolating Sources of Disentanglement in VAEs" (NeurIPS 2018)
"""

import math
from typing import Dict

import torch


LOG_2PI = math.log(2 * math.pi)


def gaussian_log_density(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Compute log density of diagonal Gaussian.

    log N(z; mu, diag(exp(logvar))) = sum_k [ -0.5 * (log(2pi) + logvar_k + (z_k - mu_k)^2 / exp(logvar_k)) ]

    Args:
        z: Samples [M, d] or [M, 1, d] for broadcasting.
        mu: Means [M, d] or [1, M, d] for broadcasting.
        logvar: Log-variances [M, d] or [1, M, d] for broadcasting.

    Returns:
        Log density, summed over dimensions. Shape depends on input broadcasting.
    """
    # Element-wise log density: -0.5 * (log(2pi) + logvar + (z-mu)^2/var)
    var = torch.exp(logvar)
    log_density_per_dim = -0.5 * (LOG_2PI + logvar + (z - mu) ** 2 / var)
    # Sum over latent dimensions
    return log_density_per_dim.sum(dim=-1)


def gaussian_log_density_per_dim(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Compute log density of diagonal Gaussian per dimension (no sum).

    Args:
        z: Samples, shape compatible for broadcasting.
        mu: Means.
        logvar: Log-variances.

    Returns:
        Per-dimension log density (not summed over k).
    """
    var = torch.exp(logvar)
    log_density_per_dim = -0.5 * (LOG_2PI + logvar + (z - mu) ** 2 / var)
    return log_density_per_dim


def standard_normal_log_density(z: torch.Tensor) -> torch.Tensor:
    """Compute log density under standard normal prior p(z) = N(0, I).

    log p(z) = sum_k [ -0.5 * (log(2pi) + z_k^2) ]

    Args:
        z: Samples [M, d].

    Returns:
        Log density summed over dimensions [M].
    """
    return -0.5 * (LOG_2PI + z ** 2).sum(dim=-1)


def compute_tcvae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    n_data: int,
    alpha: float = 1.0,
    beta_tc: float = 6.0,
    gamma: float = 1.0,
    compute_in_fp32: bool = True,
    reduction: str = "mean",
    kl_free_bits: float = 0.0,
    kl_free_bits_mode: str = "batch_mean",
) -> Dict[str, torch.Tensor]:
    """Compute β-TCVAE loss with MWS estimator.

    The loss decomposes KL(q(z|x) || p(z)) into:
    - MI: log q(z|x) - log q(z)  (mutual information)
    - TC: log q(z) - log prod_j q(z_j)  (total correlation)
    - DWKL: log prod_j q(z_j) - log p(z)  (dimension-wise KL)

    Total loss: recon + alpha*MI + beta_tc*TC + gamma*DWKL + kl_free_bits_penalty

    Free Bits posterior collapse mitigation (optional):
    Computes analytic KL per dimension and clamps to minimum threshold,
    adding the difference to the loss to prevent latent collapse.

    Args:
        x: Original input [B, C, D, H, W].
        x_hat: Reconstruction [B, C, D, H, W].
        mu: Posterior means [M, d].
        logvar: Posterior log-variances [M, d].
        z: Sampled latents [M, d].
        n_data: Total number of training samples (N).
        alpha: Weight for MI term (default 1.0).
        beta_tc: Weight for TC term (target value, typically > 1).
        gamma: Weight for DWKL term (default 1.0).
        compute_in_fp32: If True, compute TC terms in float32 for stability.
        reduction: Loss reduction strategy ("mean" or "sum").
                  "mean" averages over all elements for numerical stability.
                  "sum" sums over all elements (legacy behavior).
                  Default: "mean".
        kl_free_bits: Minimum KL threshold per latent dimension (nats).
                     Set to 0.0 to disable Free Bits (default).
                     Typical range: 0.05-0.2 nats/dimension.
        kl_free_bits_mode: Free Bits clamping mode ("batch_mean" or "per_sample").
                          "batch_mean": Clamp batch-averaged KL per dimension (recommended for small batches).
                          "per_sample": Clamp each sample's KL per dimension independently.
                          Default: "batch_mean".

    Returns:
        Dict with keys: loss, recon, mi, tc, dwkl, beta_tc, kl_raw, kl_constrained, kl_free_bits_penalty
        When reduction="mean", all terms are normalized by batch size.
        When reduction="sum", all terms are SUMS over batch.
        - kl_raw: Analytic KL before Free Bits clamping (for monitoring).
        - kl_constrained: Analytic KL after Free Bits clamping.
        - kl_free_bits_penalty: Difference added to loss (kl_constrained - kl_raw).
    """
    # Get dimensions
    m = mu.size(0)  # Minibatch size
    d = mu.size(1)  # Latent dimension

    # Store original dtype for casting back
    orig_dtype = x.dtype

    # Reconstruction loss: MSE
    # Compute in fp32 for stability when summing over millions of voxels
    if compute_in_fp32:
        squared_error = (x_hat.float() - x.float()) ** 2
    else:
        squared_error = (x_hat - x) ** 2

    if reduction == "mean":
        # Mean over all elements (batch, channels, spatial dims)
        # Provides numerical stability for FP16 mixed precision
        recon = torch.mean(squared_error)
    elif reduction == "sum":
        # Sum over all elements (backward compatibility)
        recon = torch.sum(squared_error)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    # Cast latent tensors to fp32 for numerical stability if requested
    if compute_in_fp32:
        mu = mu.float()
        logvar = logvar.float()
        z = z.float()

    # =========================================================================
    # Compute pairwise log q(z_i | x_j) matrix
    # =========================================================================
    # z: [M, d], mu: [M, d], logvar: [M, d]
    # We need log q(z_i | x_j) for all i, j pairs

    # Reshape for broadcasting:
    # z_i:      [M, 1, d]
    # mu_j:     [1, M, d]
    # logvar_j: [1, M, d]
    z_expand = z.unsqueeze(1)  # [M, 1, d]
    mu_expand = mu.unsqueeze(0)  # [1, M, d]
    logvar_expand = logvar.unsqueeze(0)  # [1, M, d]

    # log q(z_i | x_j) summed over dimensions: [M, M]
    log_q_zx_mat = gaussian_log_density(z_expand, mu_expand, logvar_expand)

    # log q(z_{i,k} | x_j) per dimension: [M, M, d]
    log_q_zx_dim = gaussian_log_density_per_dim(z_expand, mu_expand, logvar_expand)

    # =========================================================================
    # MWS estimator for log q(z_i)
    # log q(z_i) = logsumexp_j(log q(z_i|x_j)) - log(M)
    # =========================================================================
    log_m = math.log(m)

    # log q(z) for each sample i: [M]
    log_q_z = torch.logsumexp(log_q_zx_mat, dim=1) - log_m

    # log q(z_k) for each sample i and dimension k: [M, d]
    # logsumexp over j (samples in minibatch) for each (i, k)
    log_q_zk = torch.logsumexp(log_q_zx_dim, dim=1) - log_m  # [M, d]

    # log prod_k q(z_k) = sum_k log q(z_k): [M]
    log_prod_q_z = log_q_zk.sum(dim=1)

    # =========================================================================
    # log q(z_i | x_i) - diagonal elements
    # =========================================================================
    # This is log q(z|x) for the matching encoder-sample pairs
    log_q_z_given_x = torch.diag(log_q_zx_mat)  # [M]

    # =========================================================================
    # log p(z) under standard normal prior
    # =========================================================================
    log_p_z = standard_normal_log_density(z)  # [M]

    # =========================================================================
    # Decomposition terms (per sample)
    # =========================================================================
    # MI: log q(z|x) - log q(z)
    mi_per_sample = log_q_z_given_x - log_q_z

    # TC: log q(z) - log prod_k q(z_k)
    tc_per_sample = log_q_z - log_prod_q_z

    # DWKL: log prod_k q(z_k) - log p(z)
    dwkl_per_sample = log_prod_q_z - log_p_z

    # Aggregate as SUMS over batch
    mi_sum = mi_per_sample.sum()
    tc_sum = tc_per_sample.sum()
    dwkl_sum = dwkl_per_sample.sum()

    # Normalize by batch size if using mean reduction
    if reduction == "mean":
        mi = mi_sum / m
        tc = tc_sum / m
        dwkl = dwkl_sum / m
    else:
        mi = mi_sum
        tc = tc_sum
        dwkl = dwkl_sum

    # =========================================================================
    # KL Free Bits (posterior collapse mitigation)
    # =========================================================================
    # Compute analytic per-dimension KL (same as ELBO)
    # KL per dimension per sample: 0.5 * (exp(logvar) + mu^2 - 1 - logvar)
    # Shape: [M, d]
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
            kl_constrained_per_dim = torch.clamp(kl_raw_per_dim_mean, min=kl_free_bits)  # [d]
            kl_constrained = kl_constrained_per_dim.sum()  # scalar
        elif kl_free_bits_mode == "per_sample":
            # Clamp each sample per dimension
            kl_constrained_per_sample = torch.clamp(kl_per_dim, min=kl_free_bits)  # [M, d]
            kl_constrained = kl_constrained_per_sample.sum(dim=1).mean()  # scalar
        else:
            raise ValueError(
                f"Invalid kl_free_bits_mode: {kl_free_bits_mode}. "
                "Must be 'per_sample' or 'batch_mean'."
            )
    else:
        kl_constrained = kl_raw

    # Normalize by batch size if using mean reduction
    if reduction == "mean":
        kl_raw = kl_raw / m
        kl_constrained = kl_constrained / m

    # Compute free bits penalty (difference to add to loss)
    kl_free_bits_penalty = kl_constrained - kl_raw

    # Cast back to original dtype if needed
    if compute_in_fp32:
        mi = mi.to(orig_dtype)
        tc = tc.to(orig_dtype)
        dwkl = dwkl.to(orig_dtype)
        recon = recon.to(orig_dtype)
        kl_raw = kl_raw.to(orig_dtype)
        kl_constrained = kl_constrained.to(orig_dtype)
        kl_free_bits_penalty = kl_free_bits_penalty.to(orig_dtype)

    # =========================================================================
    # Total weighted loss
    # =========================================================================
    total = recon + alpha * mi + beta_tc * tc + gamma * dwkl + kl_free_bits_penalty

    # Check for non-finite values
    if not torch.isfinite(total):
        raise RuntimeError(
            f"Non-finite loss detected: total={total.item()}, "
            f"recon={recon.item()}, mi={mi.item()}, "
            f"tc={tc.item()}, dwkl={dwkl.item()}"
        )

    return {
        "loss": total,
        "recon": recon,
        "mi": mi,
        "tc": tc,
        "dwkl": dwkl,
        "beta_tc": torch.tensor(beta_tc, device=total.device, dtype=total.dtype),
        "kl_raw": kl_raw,
        "kl_constrained": kl_constrained,
        "kl_free_bits_penalty": kl_free_bits_penalty,
    }


def get_beta_tc_schedule(
    epoch: int,
    beta_tc_target: float,
    beta_tc_annealing_epochs: int,
) -> float:
    """Compute current beta_tc value for TC annealing.

    Linear annealing schedule:
    - beta_tc = 0 at epoch 0
    - Linearly increases to beta_tc_target over beta_tc_annealing_epochs
    - Stays constant at beta_tc_target after that

    Args:
        epoch: Current epoch (0-indexed).
        beta_tc_target: Target beta_tc value after annealing.
        beta_tc_annealing_epochs: Number of epochs for annealing.

    Returns:
        Current beta_tc value.
    """
    if beta_tc_annealing_epochs <= 0:
        return beta_tc_target

    if epoch >= beta_tc_annealing_epochs:
        return beta_tc_target

    return (epoch / beta_tc_annealing_epochs) * beta_tc_target
