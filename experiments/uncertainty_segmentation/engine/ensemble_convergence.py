"""Ensemble-of-k convergence and threshold-sensitivity analyses.

This module computes two quantities that the per-member sample-mean
convergence curve in ``convergence_analysis.py`` deliberately does NOT
capture:

1. **Ensemble-of-k Dice** ``D_k = Dice(argmax(mean_{m<k}(prob_m)), GT)``
   — the Dice you obtain when you actually ensemble the first ``k``
   members' soft predictions and binarize at the standard threshold
   (0.5). Typically rises with ``k`` and plateaus once ensembling has
   averaged out per-member noise.

2. **Threshold sensitivity** — for each channel, for each member, and
   for the full ensemble, Dice as a function of the binarization
   threshold τ ∈ (0, 1). Identifies the channel-specific optimal τ,
   confirms whether 0.5 is sub-optimal (especially for ET where the
   positive class is small), and quantifies how much the choice of τ
   moves Dice.

Both functions are written to run **streaming** on an in-memory
``probs_list`` (the list of M per-member soft-probability tensors
produced by :class:`EnsemblePredictor` with ``save_per_member=True``),
so the same code path works during evaluation (zero disk I/O) and
post-hoc from saved NIfTI files (loaded via
:func:`experiments.uncertainty_segmentation.engine.save_predictions.load_probs_uint8`).

Channel semantics (BraTS convention): channel 0 = TC, 1 = WT, 2 = ET.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Smoothing matches ``_compute_dice_per_channel`` in evaluate_members.py so
# per-member Dice values are directly comparable across analyses.
DICE_SMOOTH: float = 1e-5

# Fixed channel naming: anything reading these CSVs (plotting, stats)
# assumes this ordering.
CHANNELS: tuple[str, ...] = ("tc", "wt", "et")


def _dice_per_channel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-channel Dice on binary [C, D, H, W] tensors.

    Matches the shape/convention of ``evaluate_members._compute_dice_per_channel``
    but accepts tensors already on CPU/GPU and returns a [C] tensor.
    """
    C = pred.shape[0]
    flat_pred = pred.view(C, -1).float()
    flat_target = target.view(C, -1).float()
    inter = (flat_pred * flat_target).sum(dim=1)
    denom = flat_pred.sum(dim=1) + flat_target.sum(dim=1)
    return (2.0 * inter + DICE_SMOOTH) / (denom + DICE_SMOOTH)


def compute_ensemble_k_dice(
    probs_list: list[torch.Tensor],
    gt_binary: torch.Tensor,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Streaming Dice(ensemble_k) for k = 1..M across all three channels.

    Args:
        probs_list: List of M per-member probability tensors, each of
            shape ``[C, D, H, W]`` with values in [0, 1]. The list is
            assumed to be in member-id order (seed order). Tensors may
            be on CPU or GPU; a single device is inferred from the first.
        gt_binary: Ground-truth tensor ``[C, D, H, W]`` of 0/1 floats
            (output of ``_convert_seg_to_binary``).
        threshold: Binarization threshold for the averaged probabilities.
            Default 0.5 matches the convention used in
            ``ensemble_inference.predict_scan`` (``mean_probs > 0.5``).

    Returns:
        DataFrame with columns ``k, dice_tc, dice_wt, dice_et``. One
        row per k from 1 to M.

    Raises:
        ValueError: If ``probs_list`` is empty or shape-mismatched.
    """
    if not probs_list:
        raise ValueError("probs_list must contain at least one member")

    device = probs_list[0].device
    gt = gt_binary.to(device=device, dtype=torch.float32)
    shape = probs_list[0].shape
    if gt.shape != shape:
        raise ValueError(f"Shape mismatch: probs {shape} vs gt {gt.shape}")

    running_sum = torch.zeros_like(probs_list[0], dtype=torch.float32)
    rows: list[dict] = []
    for idx, probs_m in enumerate(probs_list):
        k = idx + 1
        if probs_m.shape != shape:
            raise ValueError(
                f"Inconsistent per-member shape at member {idx}:"
                f" {probs_m.shape} vs expected {shape}"
            )
        running_sum = running_sum + probs_m.to(device=device, dtype=torch.float32)
        ensemble_k = (running_sum / k > threshold).to(torch.float32)
        dice = _dice_per_channel(ensemble_k, gt).detach().cpu()
        rows.append(
            {
                "k": k,
                "dice_tc": float(dice[0]),
                "dice_wt": float(dice[1]),
                "dice_et": float(dice[2]),
            }
        )
    return pd.DataFrame(rows)


def compute_threshold_sensitivity(
    probs_list: list[torch.Tensor],
    gt_binary: torch.Tensor,
    thresholds: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Dice vs. threshold for every member and for the full ensemble.

    The ensemble probs are the uniform mean of ``probs_list`` (equivalent
    to what :class:`EnsemblePredictor` ultimately stores as
    ``mean_probs``, modulo numerical precision).

    Args:
        probs_list: List of M per-member probability tensors ``[C, D, H, W]``.
        gt_binary: Ground-truth ``[C, D, H, W]`` binary.
        thresholds: Iterable of scalar thresholds ∈ (0, 1).

    Returns:
        Long-form DataFrame with columns
        ``source, threshold, dice_tc, dice_wt, dice_et`` where
        ``source ∈ {"member_0", "member_1", ..., "member_{M-1}", "ensemble"}``.
    """
    if not probs_list:
        raise ValueError("probs_list must contain at least one member")

    device = probs_list[0].device
    gt = gt_binary.to(device=device, dtype=torch.float32)
    thresholds = [float(t) for t in thresholds]

    rows: list[dict] = []

    # Per-member Dice at each threshold.
    for m, probs_m in enumerate(probs_list):
        probs_m = probs_m.to(device=device, dtype=torch.float32)
        for t in thresholds:
            pred = (probs_m > t).to(torch.float32)
            d = _dice_per_channel(pred, gt).detach().cpu()
            rows.append(
                {
                    "source": f"member_{m}",
                    "threshold": t,
                    "dice_tc": float(d[0]),
                    "dice_wt": float(d[1]),
                    "dice_et": float(d[2]),
                }
            )

    # Full-ensemble mean probs (do not rely on a pre-computed mean from
    # the caller — re-derive from probs_list to keep this function
    # self-contained for tests and post-hoc reruns).
    mean_probs = torch.stack(
        [p.to(device=device, dtype=torch.float32) for p in probs_list], dim=0
    ).mean(dim=0)
    for t in thresholds:
        pred = (mean_probs > t).to(torch.float32)
        d = _dice_per_channel(pred, gt).detach().cpu()
        rows.append(
            {
                "source": "ensemble",
                "threshold": t,
                "dice_tc": float(d[0]),
                "dice_wt": float(d[1]),
                "dice_et": float(d[2]),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Disk-loading helpers (post-hoc reanalysis without re-inference)
# ---------------------------------------------------------------------------


def load_per_member_probs(scan_dir: Path) -> list[torch.Tensor]:
    """Load all ``member_*_probs.nii.gz`` from a scan directory.

    Args:
        scan_dir: Directory containing per-member NIfTIs saved via
            ``save_predictions.save_per_member_probs_all``.

    Returns:
        List of tensors ``[C, D, H, W]`` ordered by member id. Empty if
        no files are present.
    """
    # Local import to avoid circular import at module load time.
    from .save_predictions import load_probs_uint8

    paths: list[tuple[int, Path]] = []
    for p in scan_dir.iterdir():
        name = p.name
        if name.startswith("member_") and name.endswith("_probs.nii.gz"):
            try:
                m = int(name[len("member_") : -len("_probs.nii.gz")])
            except ValueError:
                continue
            paths.append((m, p))
    paths.sort(key=lambda kv: kv[0])
    return [load_probs_uint8(p) for _, p in paths]


def compute_for_scan_from_disk(
    scan_dir: Path,
    gt_binary: torch.Tensor,
    thresholds: list[float] | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-hoc helper: run both analyses for one scan directly from disk.

    Useful when re-running with a new threshold grid without re-inferencing.

    Args:
        scan_dir: Directory with saved per-member probs + GT.
        gt_binary: Binary GT tensor ``[C, D, H, W]``.
        thresholds: Threshold grid for the sensitivity analysis.

    Returns:
        (ensemble_k_df, threshold_sensitivity_df). ``ensemble_k_df`` has
        no ``scan_id`` column — the caller is expected to add it before
        concatenating across scans.
    """
    probs_list = load_per_member_probs(scan_dir)
    if not probs_list:
        raise FileNotFoundError(f"No member_*_probs.nii.gz files under {scan_dir}")
    ek = compute_ensemble_k_dice(probs_list, gt_binary)
    ts = compute_threshold_sensitivity(probs_list, gt_binary, thresholds)
    return ek, ts
