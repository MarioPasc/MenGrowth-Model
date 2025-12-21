"""Evidence Lower Bound (ELBO) loss for VAE training.

This module implements the negative ELBO loss with exact aggregation rules:
- Reconstruction: MSE with reduction="sum" across batch, channels, and all voxels
- KL: Closed-form KL for diagonal Gaussian vs standard normal, summed

Loss formula:
    total = recon_sum + beta * kl_sum

Where:
    recon_sum = sum_{b,c,d,h,w} (x_hat - x)^2
    kl_per_sample = 0.5 * sum_j (exp(logvar_j) + mu_j^2 - 1 - logvar_j)
    kl_sum = sum_b kl_per_sample
"""

from typing import Dict

import torch


def compute_elbo(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute negative ELBO loss for VAE.

    Args:
        x: Original input tensor [B, C, D, H, W].
        x_hat: Reconstructed tensor [B, C, D, H, W].
        mu: Posterior mean [B, z_dim].
        logvar: Posterior log-variance [B, z_dim].
        beta: Weight for KL term (for beta-VAE and KL annealing).

    Returns:
        Dict with keys:
            - "loss": Total loss (recon_sum + beta * kl_sum)
            - "recon": Reconstruction loss (MSE sum)
            - "kl": KL divergence (sum over batch)
    """
    # Reconstruction loss: MSE with sum reduction
    # Sum over all dimensions: batch, channels, depth, height, width
    recon_sum = torch.sum((x_hat - x) ** 2)

    # KL divergence: closed-form for diagonal Gaussian vs N(0,I)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Per sample: sum over latent dimensions
    # Then sum over batch
    kl_per_sample = 0.5 * torch.sum(
        torch.exp(logvar) + mu ** 2 - 1.0 - logvar,
        dim=1
    )
    kl_sum = torch.sum(kl_per_sample)

    # Total loss with beta weighting on KL
    total = recon_sum + beta * kl_sum

    return {
        "loss": total,
        "recon": recon_sum,
        "kl": kl_sum,
    }


def get_beta_schedule(
    epoch: int,
    kl_beta: float,
    kl_annealing_epochs: int,
) -> float:
    """Compute current beta value for KL annealing.

    Linear annealing schedule:
    - beta = 0 at epoch 0
    - beta linearly increases to kl_beta over kl_annealing_epochs
    - beta stays constant at kl_beta after that

    Args:
        epoch: Current epoch (0-indexed).
        kl_beta: Target beta value after annealing.
        kl_annealing_epochs: Number of epochs for annealing.

    Returns:
        Current beta value.
    """
    if kl_annealing_epochs <= 0:
        return kl_beta

    if epoch >= kl_annealing_epochs:
        return kl_beta

    return (epoch / kl_annealing_epochs) * kl_beta
