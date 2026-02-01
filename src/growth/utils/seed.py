# src/growth/utils/seed.py
"""Reproducibility utilities.

Sets random seeds for Python, NumPy, PyTorch, and CUDA.
Ensures deterministic behavior across runs.
"""

import logging

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


def set_seed(seed: int, workers: bool = True) -> None:
    """Set random seed for reproducibility.

    Uses Lightning's seed_everything to set seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - DataLoader workers (if workers=True)

    Args:
        seed: Random seed value. Use same seed across runs for reproducibility.
        workers: If True, also seed DataLoader workers for deterministic
            data loading order. Recommended for full reproducibility.

    Example:
        >>> from growth.utils.seed import set_seed
        >>> set_seed(42)
        >>> # All random operations are now reproducible
    """
    pl.seed_everything(seed, workers=workers)
    logger.info(f"Set random seed: {seed} (workers={workers})")
