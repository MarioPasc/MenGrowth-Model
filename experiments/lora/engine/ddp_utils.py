# experiments/lora/engine/ddp_utils.py
"""DDP utilities for distributed LoRA training.

Encapsulates process group setup/teardown, metric reduction, and a
domain-balanced distributed sampler for dual-domain training.
"""

import logging
import math
import os
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DistributedSampler, Sampler

logger = logging.getLogger(__name__)


# =============================================================================
# Process Group Management
# =============================================================================


def setup_ddp() -> tuple[int, int, int]:
    """Initialize DDP process group from torchrun environment variables.

    Returns:
        Tuple of (rank, local_rank, world_size).

    Raises:
        RuntimeError: If environment variables are missing or GPU count mismatch.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Assertions to prevent world-size bug
    assert local_rank < torch.cuda.device_count(), (
        f"local_rank={local_rank} >= device_count={torch.cuda.device_count()}. "
        f"Check CUDA_VISIBLE_DEVICES."
    )
    assert dist.get_world_size() == world_size, (
        f"dist.get_world_size()={dist.get_world_size()} != env WORLD_SIZE={world_size}"
    )

    # Startup logging on every rank
    logger.info(
        f"DDP initialized: rank={rank}, local_rank={local_rank}, "
        f"world_size={world_size}, device_count={torch.cuda.device_count()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )

    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    """Destroy the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int | None = None) -> bool:
    """Check if current process is rank 0 or non-distributed.

    Args:
        rank: Explicit rank. If None, checks dist.get_rank() or assumes main.

    Returns:
        True if rank 0 or non-distributed.
    """
    if rank is not None:
        return rank == 0
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# =============================================================================
# Metric Utilities
# =============================================================================


def reduce_metric(value: float, world_size: int, device: torch.device | str = "cuda") -> float:
    """All-reduce a scalar metric (average across ranks).

    Args:
        value: Local scalar value.
        world_size: Number of processes.
        device: Device for the tensor.

    Returns:
        Averaged value across all ranks.
    """
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / world_size).item()


def broadcast_scalar(
    value: float, src: int = 0, device: torch.device | str = "cuda"
) -> float:
    """Broadcast a scalar from src rank to all ranks.

    Args:
        value: Scalar value (only meaningful on src rank).
        src: Source rank.
        device: Device for the tensor.

    Returns:
        Broadcasted value on all ranks.
    """
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.broadcast(tensor, src=src)
    return tensor.item()


# =============================================================================
# Domain-Balanced Distributed Sampler
# =============================================================================


class DistributedDomainBalancedSampler(Sampler[int]):
    """Distributed sampler that maintains domain balance across ranks.

    Generates domain-balanced indices (respecting domain_ratio) at the
    global level, then shards by rank. Deterministic via seed + epoch.

    Args:
        dataset: ConcatDataset of [men_dataset, gli_dataset].
        n_men: Number of MEN samples.
        n_gli: Number of GLI samples.
        domain_ratio: Target fraction of MEN samples (0.5 = balanced).
        num_replicas: Number of DDP processes (world_size).
        rank: Current process rank.
        seed: Base random seed.
        drop_last: Whether to drop the tail to make evenly divisible.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        n_men: int,
        n_gli: int,
        domain_ratio: float = 0.5,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.dataset = dataset
        self.n_men = n_men
        self.n_gli = n_gli
        self.domain_ratio = domain_ratio
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Total samples per epoch (same as WeightedRandomSampler logic)
        self.num_samples_global = 2 * max(n_men, n_gli)

        # Per-rank samples
        if drop_last:
            self.num_samples = self.num_samples_global // num_replicas
        else:
            self.num_samples = math.ceil(self.num_samples_global / num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self.seed + self.epoch)

        n_men_samples = int(self.total_size * self.domain_ratio)
        n_gli_samples = self.total_size - n_men_samples

        # Sample MEN indices (with replacement)
        men_indices = rng.choice(self.n_men, size=n_men_samples, replace=True).tolist()
        # Sample GLI indices (offset by n_men in ConcatDataset)
        gli_indices = (
            rng.choice(self.n_gli, size=n_gli_samples, replace=True) + self.n_men
        ).tolist()

        # Interleave and shuffle
        all_indices = men_indices + gli_indices
        rng.shuffle(all_indices)

        # Shard by rank
        indices = all_indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
