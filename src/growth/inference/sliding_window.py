# src/growth/inference/sliding_window.py
"""Sliding window inference for full-resolution brain MRI volumes.

Wraps MONAI's sliding_window_inference for segmentation and feature
extraction on volumes larger than the model's training ROI.

Matches BrainSegFounder's inference protocol:
  - Full-resolution volume (no center crop)
  - Overlapping 128^3 patches with Gaussian weighting
  - Stitched output at original resolution
"""

import logging
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

logger = logging.getLogger(__name__)

# Defaults matching BrainSegFounder inference
DEFAULT_SW_ROI_SIZE = (128, 128, 128)
DEFAULT_SW_BATCH_SIZE = 4
DEFAULT_SW_OVERLAP = 0.5


@torch.no_grad()
def sliding_window_segment(
    model: nn.Module,
    images: torch.Tensor,
    roi_size: Tuple[int, int, int] = DEFAULT_SW_ROI_SIZE,
    sw_batch_size: int = DEFAULT_SW_BATCH_SIZE,
    overlap: float = DEFAULT_SW_OVERLAP,
    mode: str = "gaussian",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Run sliding window segmentation on a full-resolution volume.

    Args:
        model: Segmentation model (returns logits [B, C, D, H, W]).
        images: Input volume [1, 4, D, H, W] (batch_size must be 1).
        roi_size: Patch size for sliding window.
        sw_batch_size: Number of patches per forward pass.
        overlap: Fractional overlap between patches (0.0-1.0).
        mode: Blending mode for overlapping regions ("gaussian" or "constant").
        device: Device for inference. If None, uses model's device.

    Returns:
        Segmentation logits [1, C, D, H, W] at full resolution.
    """
    model.eval()

    if device is not None:
        images = images.to(device)

    return sliding_window_inference(
        inputs=images,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode=mode,
    )
