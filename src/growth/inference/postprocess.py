"""Postprocessing for 3D binary segmentation masks.

Connected-components cleanup to remove "loose pixels" (isolated voxels
predicted spuriously). Preserves multi-blob structures: if a region is
legitimately disconnected (e.g. bilateral peritumoral edema), we keep
every component whose voxel count is ≥ ``min_voxels``. This is NOT
"keep the largest component" — that would silently discard legitimate
secondary foci.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy import ndimage

logger = logging.getLogger(__name__)


# 26-connectivity structuring element (face / edge / corner adjacent).
# Preferred over 6-connectivity here so that thin 1-voxel-wide bridges
# between otherwise-connected tumor regions (sub-voxel segmentation
# artifacts on oblique surfaces) do not artificially split a single
# lesion into multiple pieces.
_STRUCT_26 = ndimage.generate_binary_structure(rank=3, connectivity=3)


def remove_small_components(
    mask: torch.Tensor | np.ndarray,
    min_voxels: int = 64,
    structure: np.ndarray | None = None,
) -> torch.Tensor | np.ndarray:
    """Drop 3D connected components smaller than ``min_voxels``.

    Args:
        mask: 3D binary mask, shape ``[D, H, W]``. Accepts ``torch.Tensor``
            (bool / uint8 / float on any device) or ``np.ndarray``.
        min_voxels: Minimum component size to keep. Components with fewer
            voxels are set to 0. Set to 0 to disable.
        structure: Connectivity structuring element (scipy format).
            Defaults to 26-connectivity. Pass
            ``ndimage.generate_binary_structure(3, 1)`` for 6-connectivity.

    Returns:
        Cleaned binary mask, same type / dtype / device as the input.
        If ``min_voxels == 0`` the input is returned unchanged.
    """
    if min_voxels <= 0:
        return mask

    # Record return type so we can restore it
    is_tensor = isinstance(mask, torch.Tensor)
    if is_tensor:
        orig_device = mask.device
        orig_dtype = mask.dtype
        arr = mask.detach().cpu().numpy()
    else:
        orig_device = None
        orig_dtype = mask.dtype
        arr = mask

    binary = arr.astype(bool, copy=False)
    if not binary.any():
        return mask  # empty input, nothing to do

    struct = structure if structure is not None else _STRUCT_26
    labeled, n_components = ndimage.label(binary, structure=struct)
    if n_components == 0:
        return mask

    # Voxel count per component (component 0 = background, skip).
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    # component 0 is background; always drop it from the "keep" set —
    # it is re-introduced as the 0-valued output.
    keep[1:] = sizes[1:] >= int(min_voxels)

    cleaned = keep[labeled]
    n_dropped = int(n_components - (keep.sum() - (1 if keep[0] else 0)))
    if n_dropped > 0:
        logger.debug(
            "remove_small_components: dropped %d/%d components (<%d voxels)",
            n_dropped,
            n_components,
            min_voxels,
        )

    if is_tensor:
        out = torch.from_numpy(cleaned.astype(np.uint8))
        out = out.to(device=orig_device, dtype=orig_dtype)
        return out
    return cleaned.astype(orig_dtype, copy=False)


def derive_disjoint_regions(
    probs: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    min_component_voxels: int = 64,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Derive disjoint clinical regions from BSF's three hierarchical channels.

    Training targets are BraTS-hierarchical (``TC=1|3``, ``WT=seg>0``,
    ``ET=3``); downstream analyses of meningioma use disjoint per-voxel
    regions that satisfy ``TC_necrotic ⊂ WT_meningioma``,
    ``WT_meningioma ⊥ ED_edema``, ``TC_necrotic ⊥ ED_edema``.

    Args:
        probs: Sigmoid probabilities ``[3, D, H, W]`` in BSF channel order
            (ch0=BraTS-TC, ch1=BraTS-WT, ch2=BraTS-ET). Accepts
            ``torch.Tensor`` or ``np.ndarray``.
        threshold: Binarization threshold for each channel. Default 0.5.
        min_component_voxels: Size threshold for connected-components
            cleanup applied per derived region. 0 disables.

    Returns:
        Dict with four binary masks (same dtype / device / type as
        ``probs[0]``):
            - ``"wt"``: meningioma mass (ch0 ≥ τ) — volume label.
            - ``"et"``: enhancing tumor (ch2 ≥ τ).
            - ``"tc"``: necrotic core alone, ``wt ∧ ¬et`` (label 1 region).
            - ``"ed"``: edema alone, ``(ch1 ≥ τ) ∧ ¬wt`` (label 2 region).
    """
    is_tensor = isinstance(probs, torch.Tensor)
    if is_tensor:
        ch0 = (probs[0] >= threshold).bool()
        ch1 = (probs[1] >= threshold).bool()
        ch2 = (probs[2] >= threshold).bool()
        wt_men = ch0
        et_enh = ch2 & ch0  # enforce ET ⊆ WT_meningioma (clip pretraining noise)
        tc_necr = wt_men & ~et_enh
        ed_ede = ch1 & ~wt_men
    else:
        ch0 = probs[0] >= threshold
        ch1 = probs[1] >= threshold
        ch2 = probs[2] >= threshold
        wt_men = ch0
        et_enh = ch2 & ch0
        tc_necr = wt_men & ~et_enh
        ed_ede = ch1 & ~wt_men

    out: dict[str, torch.Tensor | np.ndarray] = {
        "wt": wt_men,
        "et": et_enh,
        "tc": tc_necr,
        "ed": ed_ede,
    }
    if min_component_voxels > 0:
        for key in ("wt", "tc", "ed", "et"):
            out[key] = remove_small_components(out[key], min_voxels=min_component_voxels)
    return out
