"""Image quality metrics for 3D medical imaging.

Provides efficient GPU-accelerated implementations of PSNR and SSIM
for 3D volumes, optimized for batch processing with mixed precision.

Functions:
    compute_psnr_3d: Peak Signal-to-Noise Ratio for 3D volumes
    compute_ssim_3d: Structural Similarity Index for 3D volumes
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_psnr_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio for 3D volumes.

    Args:
        pred: Predicted volume [B, C, D, H, W]
        target: Target volume [B, C, D, H, W]
        data_range: Maximum possible pixel value (default: 1.0 for normalized data)
        reduction: 'mean' or 'none'

    Returns:
        PSNR value(s) in dB

    Example:
        >>> pred = torch.randn(2, 1, 64, 64, 64)
        >>> target = pred + 0.1 * torch.randn_like(pred)
        >>> psnr = compute_psnr_3d(pred, target)
        >>> print(f"PSNR: {psnr.item():.2f} dB")
    """
    mse = F.mse_loss(pred, target, reduction=reduction)

    if reduction == "none":
        # Per-sample PSNR
        mse = mse.view(mse.size(0), -1).mean(dim=1)

    psnr = 20 * torch.log10(data_range / torch.sqrt(mse + 1e-8))
    return psnr


def compute_ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute Structural Similarity Index for 3D volumes.

    Uses sliding window approach with Gaussian weighting.
    Simplified implementation for speed (no multi-scale).

    For 3D volumes, SSIM is computed slice-by-slice along the depth dimension
    and averaged. This is faster than full 3D SSIM and suitable for medical
    imaging where axial slices are often the primary view.

    Args:
        pred: Predicted volume [B, C, D, H, W]
        target: Target volume [B, C, D, H, W]
        window_size: Size of Gaussian window (default: 11)
        data_range: Maximum possible pixel value

    Returns:
        Mean SSIM value (scalar)

    Example:
        >>> pred = torch.randn(2, 1, 64, 64, 64)
        >>> target = pred + 0.05 * torch.randn_like(pred)
        >>> ssim = compute_ssim_3d(pred, target)
        >>> print(f"SSIM: {ssim.item():.4f}")
    """
    # Use torchmetrics for efficient GPU implementation
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        # Create metric on same device as input
        ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            kernel_size=window_size,
        ).to(pred.device)

        # Compute SSIM
        # torchmetrics expects [B, C, H, W] so we average over depth slices
        ssim_values = []
        for d in range(pred.size(2)):
            slice_pred = pred[:, :, d, :, :]
            slice_target = target[:, :, d, :, :]
            ssim_val = ssim_metric(slice_pred, slice_target)
            ssim_values.append(ssim_val)

        return torch.stack(ssim_values).mean()

    except ImportError:
        # Fallback: simple correlation-based metric
        # This is a rough approximation, not true SSIM
        # Only used if torchmetrics is not available
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)

        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean

        correlation = (pred_centered * target_centered).sum(dim=1) / (
            torch.sqrt((pred_centered ** 2).sum(dim=1)) *
            torch.sqrt((target_centered ** 2).sum(dim=1)) + 1e-8
        )

        return correlation.mean()
