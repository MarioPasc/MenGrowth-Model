# experiments/uncertainty_segmentation/engine/uncertainty_metrics.py
"""Uncertainty and calibration metrics for LoRA-Ensemble segmentation.

All functions operate on binary sigmoid probabilities (per-channel), not
categorical softmax distributions. The model outputs 3 overlapping channels
(TC, WT, ET) with independent sigmoid activations.

References:
    Nair et al. (2018). Exploring uncertainty measures in deep networks for
        Multiple Sclerosis lesion detection and segmentation. MICCAI.
    Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Voxel-wise Uncertainty Maps (operate on torch tensors)
# =============================================================================


def compute_binary_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute per-channel binary entropy of sigmoid probabilities.

    For each channel independently: H = -(p log p + (1-p) log(1-p)).
    Maximum entropy is ln(2) ≈ 0.693 at p = 0.5.

    Args:
        probs: Sigmoid probabilities, any shape. Values in [0, 1].
        eps: Small constant for numerical stability.

    Returns:
        Binary entropy, same shape as probs.
    """
    p = probs.clamp(min=eps, max=1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def compute_mutual_information(
    predictive_entropy: torch.Tensor,
    mean_member_entropy: torch.Tensor,
) -> torch.Tensor:
    """Compute mutual information (epistemic uncertainty).

    MI = H[mean_probs] - mean(H[p_m]) = predictive_entropy - mean_member_entropy.
    MI ≥ 0 by Jensen's inequality. Measures how much ensemble members disagree.

    Args:
        predictive_entropy: Binary entropy of the mean probability map.
        mean_member_entropy: Average of per-member binary entropies.

    Returns:
        Mutual information, same shape as inputs. Clamped to ≥ 0.
    """
    return (predictive_entropy - mean_member_entropy).clamp(min=0.0)


# =============================================================================
# Calibration Metrics (operate on numpy arrays)
# =============================================================================


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error for binary sigmoid predictions.

    Computes ECE per channel and averages across channels. For each channel,
    voxels are binned by predicted probability, and ECE = weighted average of
    |accuracy - confidence| per bin.

    Args:
        probs: Predicted probabilities [N_voxels, C] or [C, ...] (will be
            reshaped to [N, C]).
        labels: Binary ground truth labels, same shape as probs (0 or 1).
        n_bins: Number of bins for calibration.

    Returns:
        Scalar ECE value in [0, 1].
    """
    if probs.ndim > 2:
        C = probs.shape[0]
        probs = probs.reshape(C, -1).T  # [N, C]
        labels = labels.reshape(C, -1).T  # [N, C]

    if probs.ndim == 1:
        probs = probs[:, np.newaxis]
        labels = labels[:, np.newaxis]

    n_channels = probs.shape[1]
    ece_per_channel = []

    for c in range(n_channels):
        p = probs[:, c]
        y = labels[:, c]
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_c = 0.0
        total = len(p)

        for i in range(n_bins):
            mask = (p > bin_edges[i]) & (p <= bin_edges[i + 1])
            if i == 0:
                mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            accuracy = y[mask].mean()
            confidence = p[mask].mean()
            ece_c += (n_in_bin / total) * abs(accuracy - confidence)

        ece_per_channel.append(ece_c)

    return float(np.mean(ece_per_channel))


def compute_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Multiclass Brier score for binary sigmoid predictions.

    Per-channel MSE between predicted probabilities and binary labels,
    averaged across channels.

    Args:
        probs: Predicted probabilities, same shape as labels.
        labels: Binary ground truth labels (0 or 1).

    Returns:
        Scalar Brier score in [0, 1].
    """
    if probs.ndim > 2:
        C = probs.shape[0]
        probs = probs.reshape(C, -1).T
        labels = labels.reshape(C, -1).T

    if probs.ndim == 1:
        probs = probs[:, np.newaxis]
        labels = labels[:, np.newaxis]

    brier_per_channel = []
    for c in range(probs.shape[1]):
        brier_c = np.mean((probs[:, c] - labels[:, c]) ** 2)
        brier_per_channel.append(brier_c)

    return float(np.mean(brier_per_channel))


def compute_reliability_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, np.ndarray]:
    """Compute per-bin data for reliability diagrams.

    Aggregates across all channels (treating each channel as an independent
    binary prediction).

    Args:
        probs: Predicted probabilities, any shape (flattened internally).
        labels: Binary ground truth labels, same shape.
        n_bins: Number of bins.

    Returns:
        Dict with keys:
            - bin_edges: [n_bins + 1] bin boundary values
            - bin_accuracy: [n_bins] average accuracy per bin
            - bin_confidence: [n_bins] average confidence per bin
            - bin_count: [n_bins] number of samples per bin
    """
    p = probs.ravel()
    y = labels.ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracy = np.zeros(n_bins)
    bin_confidence = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i == 0:
            mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
        else:
            mask = (p > bin_edges[i]) & (p <= bin_edges[i + 1])
        n_in_bin = mask.sum()
        bin_count[i] = n_in_bin
        if n_in_bin > 0:
            bin_accuracy[i] = y[mask].mean()
            bin_confidence[i] = p[mask].mean()

    return {
        "bin_edges": bin_edges,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_count": bin_count,
    }
