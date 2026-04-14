"""Brain mask derivation from MRI intensities.

The BraTS volumes used in this analysis are skull-stripped and z-score
normalised; voxels outside the brain are exactly zero. ``derive_brain_mask``
exploits this property without introducing a tunable threshold:

    brain = any(image_modality != 0)

A fallback path keeps a configurable threshold ``tau`` for cases where the
input is not perfectly skull-stripped or has been re-normalised in a way
that introduces small non-zero background values.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def derive_brain_mask(
    image: torch.Tensor,
    threshold: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a binary brain mask for one scan.

    Parameters
    ----------
    image : torch.Tensor
        Multi-channel MRI volume of shape ``[C, D, H, W]`` or ``[1, C, D, H, W]``.
        Channel order is irrelevant; the mask is the union of non-background
        voxels across all modalities.
    threshold : float | None
        If ``None`` (default), use the strict ``abs(image) > eps`` rule. This
        is correct for skull-stripped, z-score-normalised BraTS data because
        background is encoded as exact zero.

        If a float is provided, use ``image[c=ref] > threshold`` on the
        first channel as a fallback intensity gate (e.g. ``0.01`` after
        z-normalisation) to handle non-skull-stripped inputs.
    eps : float
        Tolerance for the strict zero comparison.

    Returns
    -------
    torch.Tensor
        Binary mask ``[D, H, W]`` of float dtype with values in ``{0, 1}``.
    """
    if image.dim() == 5:
        if image.shape[0] != 1:
            raise ValueError(
                f"derive_brain_mask expects a single scan, got batch={image.shape[0]}"
            )
        image = image.squeeze(0)
    if image.dim() != 4:
        raise ValueError(
            f"image must be [C, D, H, W] or [1, C, D, H, W], got shape {tuple(image.shape)}"
        )

    if threshold is None:
        # Union of non-zero voxels across modalities.
        mask = (image.abs() > eps).any(dim=0)
    else:
        # Threshold on the first channel only.
        mask = image[0] > threshold

    return mask.float()


def brain_mask_coverage(mask: torch.Tensor) -> float:
    """Fraction of voxels marked as brain (1.0 in the mask).

    Used as a sanity check: realistic skull-stripped brain volumes cover
    roughly 30 to 60 percent of an isotropic 192³ volume.

    Parameters
    ----------
    mask : torch.Tensor
        Binary brain mask of any shape.

    Returns
    -------
    float
        Fraction of brain voxels in [0, 1].
    """
    return float(mask.float().mean().item())
