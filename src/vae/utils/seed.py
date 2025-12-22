"""Seed utilities for reproducibility.

This module provides functions for setting random seeds across all
relevant libraries to ensure reproducible training.
"""

import pytorch_lightning as pl


def set_seed(seed: int, workers: bool = True) -> None:
    """Set random seed for reproducibility.

    Uses Lightning's seed_everything to set seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - DataLoader workers (if workers=True)

    Args:
        seed: Random seed value.
        workers: If True, also seed DataLoader workers.
    """
    pl.seed_everything(seed, workers=workers)
