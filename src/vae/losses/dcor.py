"""Distance Correlation loss for partition independence.

Computes pairwise dCor between latent partition vectors, using an EMA buffer
to increase effective sample size beyond the per-batch N=8.

Distance Correlation (Székely et al., 2007) measures both linear and non-linear
statistical dependence between random vectors. Unlike Pearson correlation, dCor=0
if and only if the vectors are statistically independent. This makes it strictly
superior to the cross-partition Pearson loss for enforcing independence.

Algorithm (U-centering for unbiased dCov², Székely & Rizzo 2014):
    1. Compute pairwise Euclidean distance matrices A, B
    2. U-center both: Ã[i,j] = A[i,j] - mean_row - mean_col + grand_mean
    3. dCov²(X,Y) = mean(Ã ⊙ B̃)
    4. dVar²(X) = mean(Ã ⊙ Ã)
    5. dCor²(X,Y) = dCov²(X,Y) / sqrt(dVar²(X) · dVar²(Y))
    6. Loss = Σ_{i<j} dCor²(partition_i, partition_j)

References:
    Székely, Rizzo, Bakirov (2007). Measuring and Testing Dependence by
    Correlation of Distances. Annals of Statistics, 35(6), 2769-2794.

    Székely, Rizzo (2014). Partial Distance Correlation with Methods for
    Dissimilarities. Annals of Statistics, 42(6), 2382-2412.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .cross_partition import all_gather_with_grad


class PartitionDCorBuffer:
    """Buffer accumulating partition vectors for stable dCor estimation.

    In DDP with batch_size=2 and 4 GPUs, effective N=8 per step is too small
    for reliable dCor estimation (minimum recommended N≥30). This buffer
    accumulates detached partition vectors across training steps, providing
    statistical context for the loss computation while only backpropagating
    through the current batch.

    The buffer lives on CPU to minimize GPU memory overhead.
    """

    def __init__(self, buffer_size: int = 256):
        """Initialize buffer.

        Args:
            buffer_size: Maximum number of samples to store per partition.
                         256 gives ~32x more samples than per-batch N=8.
        """
        self.buffer_size = buffer_size
        self.buffers: Dict[str, deque] = {}

    def update(self, partition_data: Dict[str, torch.Tensor]) -> None:
        """Add current batch partition data to buffer (detached, on CPU).

        Args:
            partition_data: {partition_name: tensor [B, dim]} (already detached)
        """
        for name, data in partition_data.items():
            if name not in self.buffers:
                self.buffers[name] = deque(maxlen=self.buffer_size)
            # Store individual samples (each [dim]) on CPU
            cpu_data = data.detach().cpu()
            for i in range(cpu_data.shape[0]):
                self.buffers[name].append(cpu_data[i])

    def get_combined(
        self, partition_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Combine current batch with buffer contents.

        The current batch retains gradients; buffer samples are detached context.

        Args:
            partition_data: Current batch {name: [B, dim]} with gradients

        Returns:
            Combined {name: [N_eff, dim]} where N_eff = B + len(buffer)
            Only the first B rows have gradients.
        """
        combined = {}
        for name, current in partition_data.items():
            parts = [current]  # Current batch has gradients
            if name in self.buffers and len(self.buffers[name]) > 0:
                # Buffer samples: stack and move to same device
                buf_tensor = torch.stack(list(self.buffers[name]))
                buf_tensor = buf_tensor.to(
                    device=current.device, dtype=current.dtype
                )
                parts.append(buf_tensor)
            combined[name] = torch.cat(parts, dim=0)
        return combined

    @property
    def size(self) -> int:
        """Current number of samples in buffer (per partition)."""
        if not self.buffers:
            return 0
        return len(next(iter(self.buffers.values())))


def _pairwise_distances(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix.

    Args:
        X: Input tensor [N, D]

    Returns:
        Distance matrix [N, N] where D[i,j] = ||X[i] - X[j]||_2
    """
    # Using the expansion: ||x-y||² = ||x||² + ||y||² - 2<x,y>
    xx = (X * X).sum(dim=1, keepdim=True)  # [N, 1]
    dists_sq = xx + xx.t() - 2.0 * X @ X.t()
    # Clamp for numerical stability before sqrt
    dists_sq = torch.clamp(dists_sq, min=0.0)
    return torch.sqrt(dists_sq + 1e-10)


def _u_center(D: torch.Tensor) -> torch.Tensor:
    """U-center a distance matrix (unbiased estimator).

    U-centering (Székely & Rizzo, 2014) provides an unbiased estimate of
    distance covariance, unlike the original double-centering which is biased
    for finite samples.

    For n×n distance matrix D:
        Ã[i,j] = D[i,j] - (1/(n-2)) * sum_l D[i,l] - (1/(n-2)) * sum_k D[k,j]
                 + (1/((n-1)(n-2))) * sum_{k,l} D[k,l]    for i≠j
        Ã[i,i] = 0

    Args:
        D: Distance matrix [N, N]

    Returns:
        U-centered matrix [N, N]
    """
    n = D.shape[0]
    if n < 3:
        return torch.zeros_like(D)

    # Row and column means (excluding diagonal)
    # For U-centering, we use (n-2) as divisor
    row_sum = D.sum(dim=1, keepdim=True)  # [N, 1]
    col_sum = D.sum(dim=0, keepdim=True)  # [1, N]
    grand_sum = D.sum()

    # U-centering formula
    centered = (
        D
        - row_sum / (n - 2)
        - col_sum / (n - 2)
        + grand_sum / ((n - 1) * (n - 2))
    )

    # Set diagonal to zero (required for U-centering)
    centered = centered - torch.diag(torch.diag(centered))

    return centered


def _dcor_squared(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute dCor²(X, Y) using U-centered distance matrices.

    Args:
        X: First variable [N, D1]
        Y: Second variable [N, D2]

    Returns:
        dCor² scalar (in [0, 1])
    """
    n = X.shape[0]
    if n < 4:
        # Need at least 4 samples for meaningful U-centering
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    # Pairwise distances
    A = _pairwise_distances(X)
    B = _pairwise_distances(Y)

    # U-center
    A_centered = _u_center(A)
    B_centered = _u_center(B)

    # dCov²: mean of element-wise product (excluding diagonal)
    n_pairs = n * (n - 3)  # Number of off-diagonal pairs for U-centering
    if n_pairs <= 0:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    dcov_sq = (A_centered * B_centered).sum() / n_pairs

    # dVar²: self distance covariance
    dvar_x = (A_centered * A_centered).sum() / n_pairs
    dvar_y = (B_centered * B_centered).sum() / n_pairs

    # dCor² = dCov² / sqrt(dVar_X * dVar_Y)
    denom = torch.sqrt(dvar_x * dvar_y + 1e-10)

    if denom < 1e-8:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    dcor_sq = dcov_sq / denom

    # Clamp to [0, 1] (can slightly exceed due to numerics)
    return torch.clamp(dcor_sq, min=0.0, max=1.0)


def compute_dcor_loss(
    mu: torch.Tensor,
    partition_indices: Dict[str, Tuple[int, int]],
    buffer: Optional[PartitionDCorBuffer] = None,
    use_ddp_gather: bool = True,
    compute_in_fp32: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute pairwise distance correlation between supervised partitions.

    This function replaces the broken cross-partition Pearson loss. It operates
    on the full multivariate partition vectors (not dimension means) and detects
    non-linear dependence.

    Args:
        mu: Posterior means [B, z_dim]
        partition_indices: {name: (start, end)} for supervised partitions
        buffer: Optional buffer for increased effective sample size
        use_ddp_gather: Gather mu across GPUs before computing
        compute_in_fp32: Cast to FP32 for numerical stability

    Returns:
        Dict with:
            "loss": Sum of dCor² for all partition pairs (scalar, has gradients)
            "per_pair": {pair_name: dCor² value} for logging
            "n_effective": Effective sample size used
    """
    device = mu.device
    original_dtype = mu.dtype

    # Cast to FP32 if needed
    if compute_in_fp32 and mu.dtype != torch.float32:
        mu = mu.float()

    # DDP gather for larger effective batch
    if use_ddp_gather and dist.is_initialized() and dist.get_world_size() > 1:
        mu = all_gather_with_grad(mu)

    # Extract partition data
    partition_data = {}
    for name, (start, end) in partition_indices.items():
        partition_data[name] = mu[:, start:end]

    # Combine with buffer if available
    if buffer is not None and buffer.size > 0:
        combined = buffer.get_combined(partition_data)
    else:
        combined = partition_data

    # Compute pairwise dCor²
    partition_names = list(partition_indices.keys())
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    per_pair = {}
    n_effective = combined[partition_names[0]].shape[0] if partition_names else 0

    for i in range(len(partition_names)):
        for j in range(i + 1, len(partition_names)):
            name_i = partition_names[i]
            name_j = partition_names[j]

            X = combined[name_i]
            Y = combined[name_j]

            dcor_sq = _dcor_squared(X, Y)
            total_loss = total_loss + dcor_sq

            pair_key = f"{name_i}_{name_j}"
            per_pair[pair_key] = dcor_sq.detach()

    return {
        "loss": total_loss,
        "per_pair": per_pair,
        "n_effective": n_effective,
    }
