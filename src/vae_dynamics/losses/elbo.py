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
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Compute negative ELBO loss for VAE.

    Args:
        x: Original input tensor [B, C, D, H, W].
        x_hat: Reconstructed tensor [B, C, D, H, W].
        mu: Posterior mean [B, z_dim].
        logvar: Posterior log-variance [B, z_dim].
        beta: Weight for KL term (for beta-VAE and KL annealing).
        reduction: Loss reduction strategy ("mean" or "sum").
                  "mean" averages over all elements for numerical stability.
                  "sum" sums over all elements (legacy behavior).
                  Default: "mean".

    Returns:
        Dict with keys:
            - "loss": Total loss (recon + beta * kl)
            - "recon": Reconstruction loss (MSE)
            - "kl": KL divergence (normalized by batch size when reduction="mean")
    """
    # Reconstruction loss: MSE
    squared_error = (x_hat - x) ** 2

    if reduction == "mean":
        # Mean over all elements (batch, channels, spatial dims)
        # Provides numerical stability for FP16 mixed precision
        recon = torch.mean(squared_error)
    elif reduction == "sum":
        # Sum over all elements (backward compatibility)
        recon = torch.sum(squared_error)
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'sum'.")

    # KL divergence: closed-form for diagonal Gaussian vs N(0,I)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Per sample: sum over latent dimensions
    # Then sum over batch
    kl_per_sample = 0.5 * torch.sum(
        torch.exp(logvar) + mu ** 2 - 1.0 - logvar,
        dim=1
    )
    kl_sum = torch.sum(kl_per_sample)

    # Normalize KL by batch size to match reconstruction scale when using mean reduction
    batch_size = x.size(0)
    kl_normalized = kl_sum / batch_size

    # Total loss with beta weighting on KL
    total = recon + beta * kl_normalized

    return {
        "loss": total,
        "recon": recon,
        "kl": kl_normalized,
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
