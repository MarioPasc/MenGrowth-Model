"""Active Units (AU) metric for VAE latent space analysis.

The Active Units metric counts how many latent dimensions have variance
above a threshold across a dataset, indicating which dimensions are actively
being used by the encoder.

Functions:
    compute_active_units: Compute AU count and fraction
"""

import torch
from typing import Dict


def compute_active_units(
    mu: torch.Tensor,
    eps_au: float = 0.01
) -> Dict[str, float]:
    """Compute Active Units metric.

    AU = count(Var(μ_i) > eps_au across dataset)

    The Active Units metric (Burda et al., 2016) measures how many latent
    dimensions have meaningful variation across the dataset. Dimensions with
    low variance have "collapsed" and are not being used by the encoder.

    Reference:
        Burda et al. (2016). "Importance Weighted Autoencoders." ICLR.

    Args:
        mu: Latent means [N, z_dim] collected across dataset
        eps_au: Variance threshold in nats (default: 0.01)

    Returns:
        Dictionary with keys:
        - au_count: Number of active dimensions (int)
        - au_frac: Fraction of z_dim that is active (float in [0, 1])
        - mu_var_per_dim: Variance per dimension [z_dim]
        - mu_var_mean: Mean variance across dimensions

    Example:
        >>> mu = torch.randn(100, 128)  # 100 samples, 128 dims
        >>> results = compute_active_units(mu, eps_au=0.01)
        >>> print(f"Active: {results['au_count']}/{mu.shape[1]}")
    """
    # Compute variance per dimension (population variance, not sample variance)
    var_per_dim = mu.var(dim=0, unbiased=False)  # [z_dim]

    # Count active dimensions
    active_mask = var_per_dim > eps_au
    au_count = active_mask.sum().item()
    z_dim = mu.shape[1]
    au_frac = au_count / z_dim

    return {
        "au_count": float(au_count),
        "au_frac": float(au_frac),
        "mu_var_per_dim": var_per_dim,
        "mu_var_mean": var_per_dim.mean().item(),
    }
