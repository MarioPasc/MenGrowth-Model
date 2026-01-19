"""Cross-partition independence loss for semi-supervised VAE.

Penalizes correlations between supervised latent partitions (z_vol, z_loc, z_shape)
to ensure they encode independent factors of variation. This is critical for
Neural ODE training, where independent latent dimensions simplify dynamics learning.

The loss computes the correlation matrix between partition means and penalizes
off-diagonal elements, encouraging statistical independence.

References:
- DIP-VAE: Kumar et al., "Variational Inference of Disentangled Latent Concepts", ICLR 2018
- Locatello et al., "Challenging Common Assumptions in Unsupervised Learning", ICML 2019
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather operation that preserves gradients for backpropagation.

    Standard torch.distributed.all_gather does NOT propagate gradients backward
    through the network across GPUs. This function uses a workaround to maintain
    gradient flow by replacing the local tensor in the gathered list.

    For PyTorch 1.10+, this uses torch.distributed.nn.all_gather which properly
    handles gradients. For older versions, we use a manual approach.

    Args:
        tensor: Local tensor to gather [B_local, ...]

    Returns:
        Concatenated tensor from all ranks [B_global, ...]
        with gradients preserved for the local portion
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    # Try using PyTorch's built-in gradient-preserving all_gather (1.10+)
    try:
        from torch.distributed.nn import all_gather as all_gather_nn
        gathered = all_gather_nn(tensor)
        return torch.cat(gathered, dim=0)
    except ImportError:
        pass

    # Fallback: Manual gradient preservation
    # 1. Gather tensors (no gradients)
    tensors_gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gathered, tensor.contiguous())

    # 2. Replace local tensor to preserve gradients
    rank = dist.get_rank()
    tensors_gathered[rank] = tensor

    # 3. Concatenate
    return torch.cat(tensors_gathered, dim=0)


def compute_partition_means(
    mu: torch.Tensor,
    partition_indices: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    """Extract mean activation per partition.

    For each partition, computes the mean across dimensions,
    yielding a single scalar per sample that represents
    the "activity" of that partition.

    Args:
        mu: Posterior means [B, z_dim]
        partition_indices: Dict mapping partition name to (start_idx, end_idx)

    Returns:
        Dict mapping partition name to mean activations [B]
    """
    partition_means = {}
    for name, (start, end) in partition_indices.items():
        # Extract partition: [B, partition_dim]
        mu_part = mu[:, start:end]
        # Mean across dimensions: [B]
        partition_means[name] = mu_part.mean(dim=1)

    return partition_means


def compute_correlation_matrix(
    partition_means: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, List[str]]:
    """Compute correlation matrix between partition means.

    Args:
        partition_means: Dict of partition mean activations [B] each
        eps: Small constant for numerical stability

    Returns:
        correlation_matrix: [num_partitions, num_partitions]
        partition_names: List of partition names (row/col order)
    """
    partition_names = list(partition_means.keys())
    num_partitions = len(partition_names)

    # Stack means: [B, num_partitions]
    means_stacked = torch.stack(
        [partition_means[name] for name in partition_names],
        dim=1
    )

    # Center the data
    means_centered = means_stacked - means_stacked.mean(dim=0, keepdim=True)

    # Compute covariance matrix: [num_partitions, num_partitions]
    n_samples = means_centered.shape[0]
    cov_matrix = (means_centered.T @ means_centered) / (n_samples - 1 + eps)

    # Convert to correlation matrix
    std_devs = torch.sqrt(torch.diag(cov_matrix) + eps)
    std_outer = std_devs.unsqueeze(0) * std_devs.unsqueeze(1)
    corr_matrix = cov_matrix / (std_outer + eps)

    return corr_matrix, partition_names


def compute_cross_partition_loss(
    mu: torch.Tensor,
    partition_indices: Dict[str, Tuple[int, int]],
    use_ddp_gather: bool = True,
    compute_in_fp32: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute cross-partition correlation penalty.

    Penalizes correlations between different latent partitions to encourage
    statistical independence. This helps ensure that volume, location, and
    shape encodings remain independent factors, which is critical for
    interpretable Neural ODE dynamics.

    The loss is the Frobenius norm of the off-diagonal correlation blocks:
        L = sum_{i < j} corr(z_i, z_j)^2

    Args:
        mu: Posterior means [B, z_dim]
        partition_indices: Dict mapping partition name to (start_idx, end_idx)
                          e.g., {"z_vol": (0, 16), "z_loc": (16, 28), ...}
                          Only supervised partitions should be included.
        use_ddp_gather: If True, gather mu across GPUs for better correlation
                        estimation in distributed training.
        compute_in_fp32: If True, compute correlations in FP32 for stability.

    Returns:
        Dict with:
            - loss: Total cross-partition penalty (scalar)
            - per_pair: Dict of per-pair absolute correlations for logging
            - corr_matrix: Full correlation matrix for visualization
    """
    # Handle FP16/BF16 input
    original_dtype = mu.dtype
    if compute_in_fp32 and mu.dtype != torch.float32:
        mu = mu.float()

    # Gather across GPUs if in DDP mode
    if use_ddp_gather and dist.is_initialized() and dist.get_world_size() > 1:
        mu = all_gather_with_grad(mu)

    # Compute partition means
    partition_means = compute_partition_means(mu, partition_indices)

    # Compute correlation matrix
    corr_matrix, partition_names = compute_correlation_matrix(partition_means)

    # Extract off-diagonal elements and compute loss
    num_partitions = len(partition_names)
    mask = ~torch.eye(num_partitions, dtype=torch.bool, device=corr_matrix.device)
    off_diag_corrs = corr_matrix[mask]

    # Loss: Frobenius norm of off-diagonal correlations
    # Equivalent to sum of squared correlations
    loss = (off_diag_corrs ** 2).sum()

    # Build per-pair correlations for logging
    per_pair = {}
    for i, name_i in enumerate(partition_names):
        for j, name_j in enumerate(partition_names):
            if i < j:  # Upper triangle only
                key = f"{name_i}_{name_j}"
                per_pair[key] = corr_matrix[i, j].abs()

    return {
        "loss": loss,
        "per_pair": per_pair,
        "corr_matrix": corr_matrix.detach(),
        "partition_names": partition_names,
    }


def compute_cross_partition_covariance_loss(
    mu: torch.Tensor,
    partition_indices: Dict[str, Tuple[int, int]],
    use_ddp_gather: bool = True,
    compute_in_fp32: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute fine-grained cross-partition covariance penalty.

    Unlike compute_cross_partition_loss which uses partition means,
    this function computes the full covariance matrix and penalizes
    all cross-partition dimension pairs. This is more thorough but
    also more computationally expensive.

    Args:
        mu: Posterior means [B, z_dim]
        partition_indices: Dict mapping partition name to (start_idx, end_idx)
        use_ddp_gather: If True, gather mu across GPUs
        compute_in_fp32: If True, compute in FP32 for stability

    Returns:
        Dict with:
            - loss: Total cross-partition covariance penalty
            - off_diag_sum: Sum of absolute cross-partition covariances
    """
    # Handle FP16/BF16 input
    if compute_in_fp32 and mu.dtype != torch.float32:
        mu = mu.float()

    # Gather across GPUs if in DDP mode
    if use_ddp_gather and dist.is_initialized() and dist.get_world_size() > 1:
        mu = all_gather_with_grad(mu)

    # Center the full mu
    mu_centered = mu - mu.mean(dim=0, keepdim=True)

    # Compute full covariance matrix: [z_dim, z_dim]
    n_samples = mu_centered.shape[0]
    cov_full = (mu_centered.T @ mu_centered) / (n_samples - 1)

    # Create mask for cross-partition elements
    z_dim = mu.shape[1]
    cross_mask = torch.zeros(z_dim, z_dim, dtype=torch.bool, device=mu.device)

    partition_list = list(partition_indices.items())
    for i, (name_i, (start_i, end_i)) in enumerate(partition_list):
        for j, (name_j, (start_j, end_j)) in enumerate(partition_list):
            if i != j:  # Cross-partition
                cross_mask[start_i:end_i, start_j:end_j] = True

    # Extract cross-partition covariances
    cross_cov = cov_full[cross_mask]

    # Loss: Frobenius norm of cross-partition covariances
    loss = (cross_cov ** 2).sum()
    off_diag_sum = cross_cov.abs().sum()

    return {
        "loss": loss,
        "off_diag_sum": off_diag_sum,
    }
