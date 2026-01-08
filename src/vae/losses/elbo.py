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

from typing import Dict, Optional

import torch


def compute_elbo(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
    kl_free_bits: float = 0.0,
    kl_free_bits_mode: str = "per_sample",
    kl_capacity: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Compute negative ELBO loss for VAE with posterior collapse mitigation.

    Implements three optional posterior collapse mitigation techniques:

    1. Free Bits (Kingma et al., 2016):
       - Applies per-dimension thresholding to prevent KL collapse
       - Two modes available via kl_free_bits_mode:
         * "per_sample": Clamps KL[b,j] for each sample b and dimension j
           floor = z_dim × kl_free_bits (strong constraint, uniform across samples)
         * "batch_mean": Clamps mean_b(KL[b,j]) per dimension j (recommended)
           Weaker constraint, better for small batches and heterogeneous data
       - Set kl_free_bits > 0 to enable (recommended: 0.05-0.2 nats/dim)

    2. Capacity Control (Burgess et al., 2018):
       - Loss = recon + |KL - C| where C increases linearly
       - Allows reconstruction to dominate early, then gradually increase KL pressure
       - Set kl_capacity to current capacity value (scheduled externally)

    3. Beta Annealing (Higgins et al., 2017):
       - Controlled via beta parameter (scheduled externally)
       - Can use linear or cyclical schedules

    References:
        Kingma et al. (2016). "Improved Variational Inference with Inverse
        Autoregressive Flow." NeurIPS 2016. https://arxiv.org/abs/1606.04934

        Burgess et al. (2018). "Understanding disentangling in β-VAE."
        ICLR 2018. https://arxiv.org/abs/1804.03599

        Higgins et al. (2017). "β-VAE: Learning Basic Visual Concepts with a
        Constrained Variational Framework." ICLR 2017.

        Pelsmaeker & Aziz (2020). "Effective Estimation of Deep Generative
        Language Models." EMNLP 2020. https://aclanthology.org/2020.emnlp-main.350.pdf
        (Discussion of balancing constraints in free bits variants)

    Args:
        x: Original input tensor [B, C, D, H, W].
        x_hat: Reconstructed tensor [B, C, D, H, W].
        mu: Posterior mean [B, z_dim].
        logvar: Posterior log-variance [B, z_dim].
        beta: Weight for KL term (for beta-VAE and KL annealing). Default: 1.0.
        reduction: Loss reduction strategy ("mean" or "sum").
                  "mean" averages over all elements for numerical stability.
                  "sum" sums over all elements (legacy behavior).
                  Default: "mean".
        kl_free_bits: Free bits threshold per dimension (in nats). Default: 0.0 (disabled).
                     Recommended values: 0.05-0.2 nats/dim (batch_mean mode).
        kl_free_bits_mode: Clamping strategy for free bits. Default: "per_sample".
                          "per_sample": Clamp each [b,j] element (backward compatible).
                          "batch_mean": Clamp mean over batch per dim (recommended for small batches).
        kl_capacity: Target capacity for capacity control (in nats). None disables.
                    When set, replaces beta weighting with |KL - capacity|.

    Returns:
        Dict with keys:
            - "loss": Total loss (recon + weighted_kl)
            - "recon": Reconstruction loss (MSE)
            - "kl": KL divergence (after free bits, before capacity/beta)
            - "kl_raw": Raw KL (before any modification) - for logging

    Note:
        Capacity control and beta weighting are mutually exclusive:
        - If kl_capacity is not None: loss uses |KL - C|
        - If kl_capacity is None: loss uses beta * KL
        Free bits is applied in both cases.
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

    # === KL divergence computation with free bits ===
    # KL per dimension per sample: 0.5 * (exp(logvar) + mu^2 - 1 - logvar)
    # Shape: [B, z_dim]
    kl_per_dim = 0.5 * (
        torch.exp(logvar) + mu ** 2 - 1.0 - logvar
    )

    # Clamp to ensure non-negative (numerical stability)
    kl_per_dim = torch.clamp(kl_per_dim, min=0.0)

    # Compute raw KL (for logging, same across modes)
    batch_size = x.size(0)
    kl_raw_per_dim_mean = kl_per_dim.mean(dim=0)  # [z_dim]
    kl_raw_normalized = kl_raw_per_dim_mean.sum()  # scalar

    # Apply free bits based on mode
    if kl_free_bits > 0.0:
        if kl_free_bits_mode == "per_sample":
            # Current implementation: clamp each [b,j] element
            kl_constrained = torch.clamp(kl_per_dim, min=kl_free_bits)  # [B, z_dim]
            kl_normalized = kl_constrained.sum(dim=1).mean()  # sum over dims, mean over batch

        elif kl_free_bits_mode == "batch_mean":
            # NEW: clamp batch-mean per dimension
            kl_per_dim_mean = kl_per_dim.mean(dim=0)  # [z_dim]
            kl_constrained = torch.clamp(kl_per_dim_mean, min=kl_free_bits)  # [z_dim]
            kl_normalized = kl_constrained.sum()  # scalar

        else:
            raise ValueError(
                f"Invalid kl_free_bits_mode: {kl_free_bits_mode}. "
                "Must be 'per_sample' or 'batch_mean'."
            )
    else:
        # No free bits: use raw KL
        kl_normalized = kl_raw_normalized

    # === Apply capacity control or beta weighting ===
    if kl_capacity is not None:
        # Capacity control: |KL - C|
        # Use absolute difference to allow KL to be above or below capacity
        kl_weighted = torch.abs(kl_normalized - kl_capacity)
    else:
        # Standard beta-weighted KL
        kl_weighted = beta * kl_normalized

    # NOTE: We intentionally do NOT scale KL by 1/N when using reduction="mean".
    # Although reconstruction is averaged (scaled by 1/N), keeping KL unscaled ensures:
    # 1. Free bits floor remains effective (25.6 nats >> MSE_mean ~1)
    # 2. Beta parameter retains its standard interpretation
    # 3. Regularization pressure prevents posterior collapse
    # The "imbalance" is handled by:
    # - Free bits: enforces minimum information encoding
    # - Cyclical annealing: periodically relieves KL pressure
    # - Beta tuning: explicit control over reconstruction-vs-regularization tradeoff

    # Total loss
    total = recon + kl_weighted

    return {
        "loss": total,
        "recon": recon,
        "kl": kl_normalized,  # KL after free bits
        "kl_raw": kl_raw_normalized,  # KL before free bits (for logging)
    }


def get_beta_schedule(
    epoch: int,
    kl_beta: float,
    kl_annealing_epochs: int,
    kl_annealing_type: str = "linear",
    kl_annealing_cycles: int = 4,
    kl_annealing_ratio: float = 0.5,
) -> float:
    """Compute current beta value for KL annealing.

    Supports two annealing strategies:

    1. Linear annealing (default):
       - beta = 0 at epoch 0
       - beta linearly increases to kl_beta over kl_annealing_epochs
       - beta stays constant at kl_beta after that

    2. Cyclical annealing (Fu et al., 2019):
       - Training divided into kl_annealing_cycles cycles
       - Each cycle has period = kl_annealing_epochs / kl_annealing_cycles
       - Within each cycle:
         * First ratio * period epochs: beta increases linearly 0 -> kl_beta
         * Remaining (1-ratio) * period epochs: beta = kl_beta
       - This periodic relief of KL pressure allows the model to explore the
         latent space and prevents posterior collapse.

    References:
        Fu et al. (2019). "Cyclical Annealing Schedule: A Simple Approach to
        Mitigating KL Vanishing." NAACL-HLT 2019.
        https://arxiv.org/abs/1903.10145

    Args:
        epoch: Current epoch (0-indexed).
        kl_beta: Target beta value after annealing.
        kl_annealing_epochs: Total epochs for annealing schedule.
        kl_annealing_type: "linear" or "cyclical". Default: "linear".
        kl_annealing_cycles: Number of cycles for cyclical annealing. Default: 4.
        kl_annealing_ratio: Fraction of each cycle for annealing (rest at target).
                           Default: 0.5 (anneal for first half, plateau for second half).

    Returns:
        Current beta value in [0, kl_beta].

    Raises:
        ValueError: If kl_annealing_type is not "linear" or "cyclical".

    Example:
        # Linear annealing over 40 epochs
        >>> beta = get_beta_schedule(epoch=20, kl_beta=1.0, kl_annealing_epochs=40)
        >>> assert beta == 0.5

        # Cyclical annealing: 4 cycles over 160 epochs, 50% ratio
        >>> beta = get_beta_schedule(
        ...     epoch=0, kl_beta=1.0, kl_annealing_epochs=160,
        ...     kl_annealing_type="cyclical", kl_annealing_cycles=4, kl_annealing_ratio=0.5
        ... )
        >>> assert beta == 0.0  # Start of cycle 1
        >>> beta = get_beta_schedule(
        ...     epoch=40, kl_beta=1.0, kl_annealing_epochs=160,
        ...     kl_annealing_type="cyclical", kl_annealing_cycles=4, kl_annealing_ratio=0.5
        ... )
        >>> assert beta == 0.0  # Start of cycle 2
    """
    if kl_annealing_epochs <= 0:
        return kl_beta

    if kl_annealing_type == "linear":
        # Original linear behavior
        if epoch >= kl_annealing_epochs:
            return kl_beta
        return (epoch / kl_annealing_epochs) * kl_beta

    elif kl_annealing_type == "cyclical":
        # Cyclical annealing
        # Compute cycle period
        period = kl_annealing_epochs / kl_annealing_cycles

        # Position within current cycle
        cycle_position = epoch % period

        # Anneal phase length
        anneal_phase_length = period * kl_annealing_ratio

        if cycle_position < anneal_phase_length:
            # Annealing phase: linear increase 0 -> kl_beta
            return (cycle_position / anneal_phase_length) * kl_beta
        else:
            # Plateau phase: constant at kl_beta
            return kl_beta

    else:
        raise ValueError(
            f"Invalid kl_annealing_type: {kl_annealing_type}. "
            "Must be 'linear' or 'cyclical'."
        )


def get_capacity_schedule(
    epoch: int,
    kl_target_capacity: float,
    kl_capacity_anneal_epochs: int,
) -> float:
    """Compute current capacity value for capacity control.

    Linearly increases capacity from 0 to kl_target_capacity over
    kl_capacity_anneal_epochs, then stays constant.

    Used in capacity-controlled VAE loss:
        Loss = recon + |KL - C|

    where C is the capacity that gradually increases, allowing the model
    to learn reconstruction first before focusing on latent structure.
    This approach helps prevent posterior collapse by gradually introducing
    KL pressure as training progresses.

    References:
        Burgess et al. (2018). "Understanding disentangling in β-VAE."
        ICLR 2018. https://arxiv.org/abs/1804.03599

    Args:
        epoch: Current epoch (0-indexed).
        kl_target_capacity: Target capacity value (in nats).
        kl_capacity_anneal_epochs: Number of epochs to reach target.

    Returns:
        Current capacity value in [0, kl_target_capacity].

    Example:
        # Capacity increases from 0 to 20 over 100 epochs
        >>> capacity = get_capacity_schedule(epoch=0, kl_target_capacity=20.0,
        ...                                  kl_capacity_anneal_epochs=100)
        >>> assert capacity == 0.0
        >>> capacity = get_capacity_schedule(epoch=50, kl_target_capacity=20.0,
        ...                                  kl_capacity_anneal_epochs=100)
        >>> assert capacity == 10.0
        >>> capacity = get_capacity_schedule(epoch=100, kl_target_capacity=20.0,
        ...                                  kl_capacity_anneal_epochs=100)
        >>> assert capacity == 20.0
    """
    if kl_capacity_anneal_epochs <= 0:
        return kl_target_capacity

    if epoch >= kl_capacity_anneal_epochs:
        return kl_target_capacity

    return (epoch / kl_capacity_anneal_epochs) * kl_target_capacity
