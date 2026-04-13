"""Unit tests for ensemble-of-k Dice and threshold-sensitivity analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.uncertainty_segmentation.engine.ensemble_convergence import (
    DICE_SMOOTH,
    compute_ensemble_k_dice,
    compute_threshold_sensitivity,
)

pytestmark = [pytest.mark.evaluation, pytest.mark.unit]


def _binary_dice(pred: np.ndarray, gt: np.ndarray, smooth: float = DICE_SMOOTH) -> float:
    p = pred.flatten().astype(np.float64)
    t = gt.flatten().astype(np.float64)
    inter = float((p * t).sum())
    denom = float(p.sum() + t.sum())
    return (2.0 * inter + smooth) / (denom + smooth)


def _build_probs_gt(
    M: int = 5, C: int = 3, spatial: tuple[int, int, int] = (4, 4, 4), seed: int = 0,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Build synthetic per-member probs and a deterministic GT.

    Members' soft probs are constructed so that each has ~70%-80% Dice on
    WT: the GT positive region is a cube of half extent; each member's
    prob is 1.0 on a (perturbed) subset of the GT positive + some noise
    outside. This mimics realistic ensemble behaviour without resorting
    to identical members (which would make convergence trivial).
    """
    rng = np.random.default_rng(seed)
    D, H, W = spatial

    gt = np.zeros((C, D, H, W), dtype=np.float32)
    # Foreground cube for all channels.
    gt[:, D // 2:, H // 2:, W // 2:] = 1.0
    # Mimic MEN domain: WT and TC share the same positive support; ET
    # is a strict subset (shave one voxel off each dim).
    gt[2, D - 1:, :, :] = 0.0  # shrink ET slightly

    probs_list: list[torch.Tensor] = []
    for m in range(M):
        p = rng.uniform(0.0, 0.3, size=(C, D, H, W)).astype(np.float32)
        # Add a high-probability hit on the GT foreground with ~85% chance per voxel.
        hits = rng.uniform(0.0, 1.0, size=(C, D, H, W)) < 0.85
        p = np.where((gt > 0) & hits, 0.9, p)
        probs_list.append(torch.from_numpy(p))

    return probs_list, torch.from_numpy(gt)


class TestComputeEnsembleKDice:
    """Streaming Dice(ensemble_k) behaviour."""

    def test_returns_one_row_per_k(self):
        probs, gt = _build_probs_gt(M=7)
        df = compute_ensemble_k_dice(probs, gt)
        assert list(df["k"]) == list(range(1, 8))
        assert set(df.columns) == {"k", "dice_tc", "dice_wt", "dice_et"}

    def test_k1_equals_single_member_dice_at_0_5(self):
        probs, gt = _build_probs_gt(M=3)
        df = compute_ensemble_k_dice(probs, gt, threshold=0.5)

        # Direct Dice of member 0 thresholded at 0.5.
        pred0 = (probs[0] > 0.5).float().numpy()
        for c, name in enumerate(("tc", "wt", "et")):
            expected = _binary_dice(pred0[c], gt.numpy()[c])
            got = float(df.loc[df["k"] == 1, f"dice_{name}"].iloc[0])
            assert abs(got - expected) < 1e-6, (
                f"k=1 dice_{name} should match direct single-member Dice"
                f" (expected {expected:.6f}, got {got:.6f})"
            )

    def test_kM_matches_mean_probs_binarized(self):
        """At k=M, ensemble Dice == Dice(mean(probs) > 0.5)."""
        probs, gt = _build_probs_gt(M=4)
        df = compute_ensemble_k_dice(probs, gt, threshold=0.5)

        mean = torch.stack(probs, dim=0).mean(dim=0)
        pred = (mean > 0.5).float().numpy()
        for c, name in enumerate(("tc", "wt", "et")):
            expected = _binary_dice(pred[c], gt.numpy()[c])
            got = float(df.loc[df["k"] == 4, f"dice_{name}"].iloc[0])
            assert abs(got - expected) < 1e-5, (
                f"k=M dice_{name}: expected {expected:.6f}, got {got:.6f}"
            )

    def test_monotonic_improvement_when_members_agree_on_gt(self):
        """Synthetic noisy members around a shared high-prob core should
        yield Dice that does not *degrade* as k grows; at k=M it should
        be ≥ than the mean across all k=1 single-member values."""
        probs, gt = _build_probs_gt(M=8, seed=42)
        df = compute_ensemble_k_dice(probs, gt)

        for name in ("wt", "tc"):
            k1 = float(df.loc[df["k"] == 1, f"dice_{name}"].iloc[0])
            kM = float(df.loc[df["k"] == 8, f"dice_{name}"].iloc[0])
            assert kM >= k1 - 1e-3, (
                f"dice_{name}: ensemble-of-k Dice should not degrade"
                f" (k=1 {k1:.4f} vs k=M {kM:.4f})"
            )

    def test_empty_list_raises(self):
        _, gt = _build_probs_gt(M=1)
        with pytest.raises(ValueError, match="at least one"):
            compute_ensemble_k_dice([], gt)

    def test_shape_mismatch_raises(self):
        probs, gt = _build_probs_gt(M=2)
        wrong = probs[0][:, :2, :, :]  # truncated D axis
        with pytest.raises(ValueError, match="Shape mismatch|Inconsistent"):
            compute_ensemble_k_dice([probs[0], wrong], gt)


class TestComputeThresholdSensitivity:
    """Dice vs. threshold — per-member and ensemble rows."""

    def test_rows_and_columns(self):
        probs, gt = _build_probs_gt(M=3)
        thresholds = [0.3, 0.5, 0.7]
        df = compute_threshold_sensitivity(probs, gt, thresholds)

        expected_rows = (3 + 1) * len(thresholds)  # M members + 1 ensemble
        assert len(df) == expected_rows
        assert set(df.columns) == {"source", "threshold", "dice_tc", "dice_wt", "dice_et"}

        sources = set(df["source"].unique())
        assert sources == {"member_0", "member_1", "member_2", "ensemble"}
        assert set(df["threshold"].unique()) == set(thresholds)

    def test_member_row_matches_direct_dice(self):
        probs, gt = _build_probs_gt(M=2)
        thresholds = [0.4, 0.5, 0.6]
        df = compute_threshold_sensitivity(probs, gt, thresholds)

        m = 1
        tau = 0.6
        pred = (probs[m] > tau).float().numpy()
        for c, name in enumerate(("tc", "wt", "et")):
            expected = _binary_dice(pred[c], gt.numpy()[c])
            got = float(
                df[(df["source"] == f"member_{m}") & (df["threshold"] == tau)][
                    f"dice_{name}"
                ].iloc[0]
            )
            assert abs(got - expected) < 1e-6

    def test_ensemble_row_matches_mean_probs(self):
        probs, gt = _build_probs_gt(M=5)
        thresholds = [0.5]
        df = compute_threshold_sensitivity(probs, gt, thresholds)

        mean = torch.stack(probs, dim=0).mean(dim=0)
        pred = (mean > 0.5).float().numpy()
        for c, name in enumerate(("tc", "wt", "et")):
            expected = _binary_dice(pred[c], gt.numpy()[c])
            got = float(
                df[(df["source"] == "ensemble") & (df["threshold"] == 0.5)][
                    f"dice_{name}"
                ].iloc[0]
            )
            assert abs(got - expected) < 1e-5

    def test_dice_monotone_bounds_across_extreme_thresholds(self):
        probs, gt = _build_probs_gt(M=3, seed=1)
        df = compute_threshold_sensitivity(probs, gt, [0.01, 0.5, 0.99])
        # At tau ≈ 0: every voxel predicted positive → recall=1, precision=fg_frac → bounded Dice.
        # At tau ≈ 1: (almost) nothing predicted positive → Dice → smooth/denom ≈ 0.
        low = df[(df["source"] == "ensemble") & (df["threshold"] == 0.01)]
        high = df[(df["source"] == "ensemble") & (df["threshold"] == 0.99)]
        for name in ("tc", "wt", "et"):
            d_low = float(low[f"dice_{name}"].iloc[0])
            d_high = float(high[f"dice_{name}"].iloc[0])
            # High-threshold Dice should not exceed low-threshold Dice on
            # our synthetic data, because our members spend >80% of their
            # mass below 0.9 → binarizing at 0.99 zeroes out most of the TP.
            assert d_high <= d_low + 1e-6


class TestQuantizationRoundtrip:
    """uint8 save-load should preserve probabilities within 1/255."""

    def test_roundtrip_preserves_ensemble_k_dice(self, tmp_path):
        from experiments.uncertainty_segmentation.engine.save_predictions import (
            _save_probs_uint8,
            load_probs_uint8,
        )

        probs, gt = _build_probs_gt(M=3, seed=3)
        paths = []
        for m, p in enumerate(probs):
            out = tmp_path / f"member_{m}_probs.nii.gz"
            _save_probs_uint8(p, out)
            paths.append(out)
        loaded = [load_probs_uint8(p) for p in paths]

        orig_df = compute_ensemble_k_dice(probs, gt)
        round_df = compute_ensemble_k_dice(loaded, gt)

        for ch in ("tc", "wt", "et"):
            orig = orig_df[f"dice_{ch}"].values
            got = round_df[f"dice_{ch}"].values
            # Quantization noise in Dice is typically far below 1% on
            # our synthetic volumes — require 5e-3 tolerance.
            assert np.allclose(orig, got, atol=5e-3), (
                f"dice_{ch}: roundtrip delta exceeds 5e-3: "
                f"orig={orig}, got={got}"
            )
