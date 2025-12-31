"""Pure metric computation functions for VAE experiments.

This module provides reusable, testable metric functions extracted from callbacks.
All functions are pure (no side effects) and can be used independently for
analysis, logging, and post-hoc evaluation.

Modules:
    image_quality: PSNR and SSIM for 3D volumes
    latent_statistics: Correlation, covariance, shift sensitivity
    active_units: Active Units (AU) metric
    regression_probes: Ridge regression and segmentation feature extraction

Example:
    >>> from vae.metrics import compute_psnr_3d, compute_active_units
    >>> import torch
    >>> pred = torch.randn(2, 4, 64, 64, 64)
    >>> target = pred + 0.1 * torch.randn_like(pred)
    >>> psnr = compute_psnr_3d(pred, target)
    >>> print(f"PSNR: {psnr.item():.2f} dB")
"""

from .image_quality import compute_psnr_3d, compute_ssim_3d
from .latent_statistics import (
    compute_correlation,
    compute_dipvae_covariance,
    compute_shift_sensitivity,
    compute_cov_batch,
)
from .active_units import compute_active_units
from .regression_probes import extract_segmentation_targets, ridge_probe_cv

__all__ = [
    # Image quality metrics
    "compute_psnr_3d",
    "compute_ssim_3d",
    # Latent statistics
    "compute_correlation",
    "compute_dipvae_covariance",
    "compute_shift_sensitivity",
    "compute_cov_batch",
    # Active units
    "compute_active_units",
    # Regression probes
    "extract_segmentation_targets",
    "ridge_probe_cv",
]
