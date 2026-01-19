"""Total Correlation (TC) loss for disentanglement.

Implements the TC penalty from β-TCVAE (Chen et al., NeurIPS 2018):
"Isolating Sources of Disentanglement in Variational Autoencoders"

The TC term encourages a factorial aggregate posterior q(z), meaning
the latent dimensions are statistically independent across the dataset.

Key insight: KL(q(z)||p(z)) = I(x;z) + KL(q(z)||prod_j q(z_j)) + sum_j KL(q(z_j)||p(z_j))
                              ↑          ↑                          ↑
                           Index-MI    Total Correlation         Dimension-KL

β-TCVAE weights the TC term more heavily to encourage disentanglement.

For semi-supervised VAE, TC is applied only to unsupervised (residual) dimensions
while supervised dimensions use regression losses.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.distributed as dist


def compute_tc_loss(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    dataset_size: int,
    estimator: str = "minibatch_weighted",
) -> torch.Tensor:
    """Compute Total Correlation penalty.

    Uses minibatch-weighted sampling (MWS) estimator for tractable TC computation.
    This avoids density estimation by using importance weighting.

    Args:
        z: Sampled latents [B, D]
        mu: Posterior means [B, D]
        logvar: Posterior log-variances [B, D]
        dataset_size: Total number of samples in dataset (for weighting)
        estimator: TC estimator type ("minibatch_weighted" or "stratified")

    Returns:
        TC loss (scalar tensor)
    """
    if estimator == "minibatch_weighted":
        return _compute_tc_mws(z, mu, logvar, dataset_size)
    elif estimator == "stratified":
        return _compute_tc_stratified(z, mu, logvar)
    else:
        raise ValueError(f"Unknown TC estimator: {estimator}")


def _compute_tc_mws(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    dataset_size: int,
) -> torch.Tensor:
    """Minibatch-weighted sampling estimator for TC.

    From Chen et al. (2018), Section 4.2.
    Estimates log q(z) and log prod_j q(z_j) using minibatch samples.

    The TC is:
        TC = E_q(z)[log q(z) - log prod_j q(z_j)]
           ≈ E_z~q [log (1/NM Σ_n q(z|x_n)) - Σ_j log (1/NM Σ_n q(z_j|x_n))]

    where M is batch size, N is dataset size.

    Args:
        z: Sampled latents [B, D]
        mu: Posterior means [B, D]
        logvar: Posterior log-variances [B, D]
        dataset_size: N in the formula

    Returns:
        TC estimate (scalar)
    """
    batch_size, z_dim = z.shape
    device = z.device

    # Compute log q(z_i | x_j) for all pairs (i, j) in batch
    # z: [B, D], mu: [B, D], logvar: [B, D]
    # We need log N(z_i | mu_j, exp(logvar_j)) for each i, j

    # Expand for pairwise computation
    # z_expand: [B, 1, D], mu_expand: [1, B, D], logvar_expand: [1, B, D]
    z_expand = z.unsqueeze(1)  # [B, 1, D]
    mu_expand = mu.unsqueeze(0)  # [1, B, D]
    logvar_expand = logvar.unsqueeze(0)  # [1, B, D]

    # log q(z_i | x_j) = -0.5 * [D*log(2π) + sum(logvar) + sum((z-mu)²/var)]
    # Compute per-dimension log probability
    var = torch.exp(logvar_expand)  # [1, B, D]
    log_qz_given_x = -0.5 * (
        logvar_expand + (z_expand - mu_expand) ** 2 / var
    )  # [B, B, D]

    # For log q(z): sum over dimensions, then logsumexp over samples
    log_qz_joint = log_qz_given_x.sum(dim=2)  # [B, B] - sum over D

    # Importance weighting: approximate expectation over full dataset
    # log q(z_i) ≈ log (1/(NM) Σ_j exp(log q(z_i|x_j)))
    #            = logsumexp_j(log q(z_i|x_j)) - log(NM)
    log_qz = torch.logsumexp(log_qz_joint, dim=1) - torch.log(
        torch.tensor(dataset_size * batch_size, device=device, dtype=z.dtype)
    )  # [B]

    # For log prod_j q(z_j): compute marginal for each dimension separately
    # log q(z_j) = logsumexp over samples, per dimension
    log_qz_marginal = torch.logsumexp(log_qz_given_x, dim=1) - torch.log(
        torch.tensor(dataset_size * batch_size, device=device, dtype=z.dtype)
    )  # [B, D]
    log_qz_product = log_qz_marginal.sum(dim=1)  # [B]

    # TC = E[log q(z) - log prod q(z_j)]
    tc = (log_qz - log_qz_product).mean()

    return tc


def _compute_tc_stratified(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Stratified sampling estimator for TC.

    Alternative to MWS that permutes z across batch to approximate prod q(z_j).

    Args:
        z: Sampled latents [B, D]
        mu: Posterior means [B, D]
        logvar: Posterior log-variances [B, D]

    Returns:
        TC estimate (scalar)
    """
    batch_size, z_dim = z.shape

    # Create permuted z by shuffling each dimension independently
    z_perm = torch.zeros_like(z)
    for d in range(z_dim):
        perm = torch.randperm(batch_size, device=z.device)
        z_perm[:, d] = z[perm, d]

    # log q(z) using kernel density estimation with batch samples
    log_qz = _log_density(z, mu, logvar)

    # log prod q(z_j) using permuted samples
    log_qz_product = _log_density(z_perm, mu, logvar)

    # TC estimate
    tc = (log_qz - log_qz_product).mean()

    return tc


def _log_density(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Estimate log q(z) using minibatch kernel density.

    Args:
        z: Points to evaluate [B, D]
        mu: Gaussian centers [B, D]
        logvar: Log-variances [B, D]

    Returns:
        Log density estimates [B]
    """
    batch_size = z.shape[0]

    # z_expand: [B, 1, D], mu_expand: [1, B, D]
    z_expand = z.unsqueeze(1)
    mu_expand = mu.unsqueeze(0)
    logvar_expand = logvar.unsqueeze(0)

    var = torch.exp(logvar_expand)
    log_qz_given_x = -0.5 * (
        logvar_expand + (z_expand - mu_expand) ** 2 / var
    ).sum(dim=2)  # [B, B]

    # Average over samples (kernel density estimate)
    log_qz = torch.logsumexp(log_qz_given_x, dim=1) - torch.log(
        torch.tensor(batch_size, device=z.device, dtype=z.dtype)
    )

    return log_qz


def compute_tc_loss_on_subset(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    start_idx: int,
    end_idx: int,
    dataset_size: int,
    estimator: str = "minibatch_weighted",
) -> torch.Tensor:
    """Compute TC loss on a subset of latent dimensions.

    For semi-supervised VAE, apply TC only to residual (unsupervised) dimensions.

    Args:
        z: Full sampled latents [B, D_total]
        mu: Full posterior means [B, D_total]
        logvar: Full posterior log-variances [B, D_total]
        start_idx: Start index of subset (inclusive)
        end_idx: End index of subset (exclusive)
        dataset_size: Total dataset size
        estimator: TC estimator type

    Returns:
        TC loss on the specified subset
    """
    z_subset = z[:, start_idx:end_idx]
    mu_subset = mu[:, start_idx:end_idx]
    logvar_subset = logvar[:, start_idx:end_idx]

    return compute_tc_loss(
        z_subset, mu_subset, logvar_subset,
        dataset_size=dataset_size,
        estimator=estimator,
    )


def compute_decomposed_kl(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    dataset_size: int,
) -> Dict[str, torch.Tensor]:
    """Compute full KL decomposition from β-TCVAE.

    KL(q(z|x)||p(z)) = I(x;z) + TC(z) + Σ_j KL(q(z_j)||p(z_j))

    Args:
        z: Sampled latents [B, D]
        mu: Posterior means [B, D]
        logvar: Posterior log-variances [B, D]
        dataset_size: Total dataset size

    Returns:
        Dictionary with:
        - index_mi: Mutual information I(x;z)
        - tc: Total correlation TC(z)
        - dimwise_kl: Dimension-wise KL Σ_j KL(q(z_j)||p(z_j))
        - total_kl: Sum of all terms
    """
    batch_size, z_dim = z.shape
    device = z.device

    # Standard KL per sample: KL(q(z|x)||p(z))
    # = 0.5 * Σ_j [exp(logvar_j) + mu_j² - 1 - logvar_j]
    kl_per_dim = 0.5 * (torch.exp(logvar) + mu ** 2 - 1 - logvar)  # [B, D]
    kl_per_sample = kl_per_dim.sum(dim=1)  # [B]

    # Compute log densities for decomposition
    z_expand = z.unsqueeze(1)
    mu_expand = mu.unsqueeze(0)
    logvar_expand = logvar.unsqueeze(0)

    var = torch.exp(logvar_expand)
    log_qz_given_x = -0.5 * (
        logvar_expand + (z_expand - mu_expand) ** 2 / var
    )  # [B, B, D]

    # log q(z|x) for the matching sample (diagonal)
    log_qz_given_x_diag = log_qz_given_x.diagonal(dim1=0, dim2=1).T  # [B, D]
    log_qz_given_x_total = log_qz_given_x_diag.sum(dim=1)  # [B]

    # log q(z) - joint density
    log_qz_joint = log_qz_given_x.sum(dim=2)  # [B, B]
    log_qz = torch.logsumexp(log_qz_joint, dim=1) - torch.log(
        torch.tensor(dataset_size * batch_size, device=device, dtype=z.dtype)
    )

    # log prod_j q(z_j) - product of marginals
    log_qz_marginal = torch.logsumexp(log_qz_given_x, dim=1) - torch.log(
        torch.tensor(dataset_size * batch_size, device=device, dtype=z.dtype)
    )  # [B, D]
    log_qz_product = log_qz_marginal.sum(dim=1)  # [B]

    # log p(z) = log N(z|0,I)
    log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * z_dim * torch.log(
        torch.tensor(2 * 3.14159265, device=device, dtype=z.dtype)
    )

    # Decomposition:
    # I(x;z) = E[log q(z|x) - log q(z)]
    index_mi = (log_qz_given_x_total - log_qz).mean()

    # TC(z) = E[log q(z) - log prod_j q(z_j)]
    tc = (log_qz - log_qz_product).mean()

    # Dim-wise KL = E[log prod_j q(z_j) - log p(z)]
    dimwise_kl = (log_qz_product - log_pz).mean()

    return {
        "index_mi": index_mi,
        "tc": tc,
        "dimwise_kl": dimwise_kl,
        "total_kl": kl_per_sample.mean(),
    }


def compute_tc_ddp_aware(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    dataset_size: int,
    use_ddp_gather: bool = True,
    estimator: str = "minibatch_weighted",
) -> torch.Tensor:
    """Compute TC loss with DDP-aware gathering.

    Gathers latents across all GPUs for better TC estimation in distributed training.

    Args:
        z: Local sampled latents [B_local, D]
        mu: Local posterior means [B_local, D]
        logvar: Local posterior log-variances [B_local, D]
        dataset_size: Total dataset size
        use_ddp_gather: Whether to gather across GPUs
        estimator: TC estimator type

    Returns:
        TC loss
    """
    if use_ddp_gather and dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()

        # Gather all tensors
        z_gathered = [torch.zeros_like(z) for _ in range(world_size)]
        mu_gathered = [torch.zeros_like(mu) for _ in range(world_size)]
        logvar_gathered = [torch.zeros_like(logvar) for _ in range(world_size)]

        dist.all_gather(z_gathered, z)
        dist.all_gather(mu_gathered, mu)
        dist.all_gather(logvar_gathered, logvar)

        z = torch.cat(z_gathered, dim=0)
        mu = torch.cat(mu_gathered, dim=0)
        logvar = torch.cat(logvar_gathered, dim=0)

    return compute_tc_loss(z, mu, logvar, dataset_size, estimator)
