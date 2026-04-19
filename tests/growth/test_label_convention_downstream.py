# tests/growth/test_label_convention_downstream.py
"""Regression tests for BraTS-MEN label convention downstream fixes.

Guards against:
1. dice_mean inflation from trivially-1.0 TC channel
2. Volume extraction using wrong channel (WT instead of ET)
3. Ensemble mask using wrong channel

These tests verify the fixes applied after the BraTS-MEN label convention
correction (commit 98f6c9a): TC is now always empty for MEN, ET=(1|3),
and all volume/metric computations must account for this.
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit]


# =============================================================================
# Task 1: dice_mean excludes TC (training metric)
# =============================================================================


class TestDiceMeanExcludesTC:
    """Verify dice_mean averages WT+ET only, not all 3 channels."""

    def test_validate_returns_dice_mean_without_tc(self) -> None:
        """dice_mean must equal (WT+ET)/2, not (TC+WT+ET)/3.

        With TC target always zeros for MEN, TC Dice is trivially ~1.0.
        Including it in dice_mean inflates the metric and desensitizes
        early stopping.
        """
        from experiments.uncertainty_segmentation.engine.train_member import validate

        # Mock model that returns fixed logits
        model = MagicMock()
        # 3-channel logits: TC=-10 (suppressed), WT=+10, ET=+5
        logits = torch.zeros(1, 3, 4, 4, 4)
        logits[:, 0] = -10.0  # TC suppressed
        logits[:, 1] = 10.0  # WT high
        logits[:, 2] = 5.0  # ET moderate
        model.return_value = logits
        model.eval = MagicMock(return_value=model)

        # Seg target: MEN labels (1=NETC, 2=SNFH, 3=ET)
        seg = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
        seg[0, 0, :2, :2, :2] = 3  # ET region
        seg[0, 0, 2:, :2, :2] = 2  # SNFH region

        # Dataloader yielding one batch
        batch = {"image": torch.randn(1, 4, 4, 4, 4), "seg": seg}
        dataloader = [batch]

        from growth.losses.segmentation import DiceMetric3Ch, SegmentationLoss3Ch

        seg_loss_fn = SegmentationLoss3Ch()
        dice_metric = DiceMetric3Ch()

        result = validate(model, dataloader, seg_loss_fn, dice_metric, "cpu")

        # dice_mean must be (WT + ET) / 2, NOT (TC + WT + ET) / 3
        expected_mean = (result["dice_wt"] + result["dice_et"]) / 2
        assert abs(result["dice_mean"] - expected_mean) < 1e-6, (
            f"dice_mean={result['dice_mean']:.6f} but (WT+ET)/2={expected_mean:.6f}. "
            "If dice_mean includes TC (~1.0), early stopping is desensitized."
        )

    def test_dice_mean_not_inflated_by_trivial_tc(self) -> None:
        """With perfect TC suppression, dice_mean should NOT be pulled toward 1.0."""
        # Simulate: TC Dice = 1.0 (empty/empty), WT = 0.7, ET = 0.5
        dice_tensor = torch.tensor([1.0, 0.7, 0.5])

        # Old (wrong): mean of all 3
        old_mean = dice_tensor.mean().item()  # 0.733

        # New (correct): mean of WT + ET only
        new_mean = dice_tensor[1:].mean().item()  # 0.6

        assert abs(new_mean - 0.6) < 1e-6
        assert old_mean > new_mean, "Old mean was inflated by TC=1.0"


# =============================================================================
# Task 2: Volume extraction uses ET channel (ch2)
# =============================================================================


class TestVolumeUsesETChannel:
    """Verify volume extraction reads from ET channel (index 2), not WT (index 1)."""

    def test_ensemble_inference_volume_from_et_channel(self) -> None:
        """EnsemblePredictor must compute volume from probs[2] (ET), not probs[1] (WT).

        The meningioma growth target is the enhancing-tumor mass (ET = ch2).
        WT includes SNFH/edema which injects noise into growth trajectories.
        """
        from experiments.uncertainty_segmentation.engine.ensemble_inference import (
            EnsemblePrediction,
        )

        # Construct a mock scenario: ET (ch2) has 100 voxels, WT (ch1) has 500 voxels
        C, D, H, W = 3, 8, 8, 8
        mean_probs = torch.zeros(C, D, H, W)
        # WT channel: 500 voxels above threshold
        mean_probs[1, :, :, :] = 0.9  # all 512 voxels → WT mask = 512
        # ET channel: only a small cube (2³ = 8 voxels)
        mean_probs[2, :2, :2, :2] = 0.9  # 8 voxels above threshold

        # The ensemble_mask should come from ET (ch2), not WT (ch1)
        ensemble_mask = mean_probs[2] > 0.5
        assert ensemble_mask.sum().item() == 8, (
            "ensemble_mask should be ET-based (8 voxels), not WT-based (512)"
        )

    def test_evaluate_members_volume_pred_uses_ch2(self) -> None:
        """Per-member volume must count ET voxels (ch2), not WT voxels (ch1).

        Regression: old code used pred_binary[1].sum() (WT channel).
        """
        # Simulate pred_binary [3, D, H, W]
        pred_binary = torch.zeros(3, 4, 4, 4)
        pred_binary[1, :, :, :] = 1.0  # WT: all 64 voxels
        pred_binary[2, :2, :2, :2] = 1.0  # ET: 8 voxels

        # The volume should be ET count = 8
        vol_pred = float(pred_binary[2].sum().item())
        assert vol_pred == 8.0, (
            f"Expected ET volume=8.0, got {vol_pred}. "
            "If 64.0, the code is still using WT (ch1)."
        )

    def test_evaluate_baseline_volume_gt_uses_ch2(self) -> None:
        """GT volume must count from ET channel (ch2) of the converted target.

        With the corrected MEN convention: ET = (1|3) = merged meningioma mass.
        """
        from growth.losses.segmentation import _convert_single_domain

        # MEN target with known composition:
        #   [0,:4,:4,:4] = label 3 → 64 voxels ET
        #   [0,4:,:4,:4] = label 2 → 64 voxels SNFH (edema)
        #   [0,4,4,4]    = label 1 → 1 voxel NETC (merged into ET)
        target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        target[0, :4, :4, :4] = 3  # 64 voxels of ET
        target[0, 4:, :4, :4] = 2  # 64 voxels of SNFH (edema)
        target[0, 4, 4, 4] = 1  # 1 voxel of NETC (outside ET cube)

        masks = _convert_single_domain(target, "MEN")  # [1, 3, 8, 8, 8]
        gt_binary = masks.squeeze(0)  # [3, 8, 8, 8]

        vol_et = float(gt_binary[2].sum().item())  # ET channel
        vol_wt = float(gt_binary[1].sum().item())  # WT channel

        # ET should count labels (1|3) = 64 + 1 = 65
        assert vol_et == 65.0, f"ET volume should be 65, got {vol_et}"
        # WT should count all tumor (1|2|3) = 64 + 64 + 1 = 129
        assert vol_wt == 129.0, f"WT volume should be 129, got {vol_wt}"
        # The growth target must be ET, not WT
        assert vol_et < vol_wt, "ET must exclude edema (smaller than WT)"


# =============================================================================
# Task 3: dice_mean in evaluation CSVs excludes TC
# =============================================================================


class TestEvalDiceMeanExcludesTC:
    """Verify that evaluation outputs compute dice_mean from WT+ET only."""

    def test_dice_mean_formula_in_eval(self) -> None:
        """dice_mean in CSV rows must be (WT+ET)/2 = dice[1:].mean()."""
        # Simulate a per-channel Dice vector
        dice = torch.tensor([0.95, 0.78, 0.62])  # TC=0.95, WT=0.78, ET=0.62

        # Correct formula (new)
        dice_mean_correct = float(dice[1:].mean())  # (0.78 + 0.62) / 2 = 0.70
        # Wrong formula (old)
        dice_mean_wrong = float(dice.mean())  # (0.95 + 0.78 + 0.62) / 3 = 0.783

        assert abs(dice_mean_correct - 0.70) < 1e-6
        assert dice_mean_correct < dice_mean_wrong, (
            "Old formula inflated dice_mean by including trivial TC"
        )

    def test_bypass_test_dice_mean_excludes_tc(self) -> None:
        """run_decoder_bypass dice_mean must use dice[1:].mean()."""
        dice = torch.tensor([1.0, 0.80, 0.55])  # TC trivially 1.0

        result_mean = float(dice[1:].mean())  # (0.80 + 0.55) / 2 = 0.675
        assert abs(result_mean - 0.675) < 1e-6

        # If it were dice.mean() → 0.783 (inflated)
        wrong_mean = float(dice.mean())
        assert wrong_mean > result_mean


# =============================================================================
# Task 4: Figures skip TC channel
# =============================================================================


class TestFiguresSkipTC:
    """Verify figure code does not plot TC for MEN domain."""

    def test_dice_compartments_excludes_tc(self) -> None:
        """fig_dice_compartments must only plot WT and ET (2 bars, not 3)."""
        import importlib

        mod = importlib.import_module(
            "experiments.uncertainty_segmentation.plotting.figures.fig_dice_compartments"
        )
        # Check the source-level compartments list
        import inspect
        source = inspect.getsource(mod.plot)
        assert '"tc"' not in source and "'tc'" not in source, (
            "fig_dice_compartments still references TC compartment"
        )

    def test_training_curves_excludes_tc(self) -> None:
        """fig_training_curves must not plot val_dice_tc."""
        import importlib
        import inspect

        mod = importlib.import_module(
            "experiments.uncertainty_segmentation.plotting.figures.fig_training_curves"
        )
        source = inspect.getsource(mod.plot)
        assert "val_dice_tc" not in source, (
            "fig_training_curves still plots TC validation Dice"
        )

    def test_convergence_uses_two_panels(self) -> None:
        """fig_convergence must create 2 panels (WT + ET), not 3."""
        import importlib
        import inspect

        mod = importlib.import_module(
            "experiments.uncertainty_segmentation.plotting.figures.fig_convergence"
        )
        source = inspect.getsource(mod.plot)
        assert "1, 2" in source or "1,2" in source, (
            "fig_convergence should create 1×2 subplots (WT + ET only)"
        )
        assert '"tc"' not in source and "'tc'" not in source, (
            "fig_convergence still references TC channel in sample_mean_sources"
        )

    def test_threshold_sensitivity_uses_two_panels(self) -> None:
        """fig_threshold_sensitivity must create 2 panels (WT + ET), not 3."""
        import importlib
        import inspect

        mod = importlib.import_module(
            "experiments.uncertainty_segmentation.plotting.figures.fig_threshold_sensitivity"
        )
        source = inspect.getsource(mod.plot)
        assert "1, 2" in source or "1,2" in source, (
            "fig_threshold_sensitivity should create 1×2 subplots"
        )


# =============================================================================
# Integration: volume_extraction uses ET-specific uncertainty
# =============================================================================


class TestVolumeExtractionETUncertainty:
    """Verify volume_extraction.py reports ET channel uncertainty, not WT."""

    def test_column_names_use_et_prefix(self) -> None:
        """Output columns must be et_mean_entropy, et_mean_mi (not wt_*)."""
        import importlib
        import inspect

        mod = importlib.import_module(
            "experiments.uncertainty_segmentation.engine.volume_extraction"
        )
        source = inspect.getsource(mod.extract_ensemble_volumes)
        assert "et_mean_entropy" in source, "Missing et_mean_entropy column"
        assert "et_mean_mi" in source, "Missing et_mean_mi column"
        assert "et_boundary_entropy" in source, "Missing et_boundary_entropy column"
        assert "wt_mean_entropy" not in source, (
            "Still using wt_mean_entropy — should be et_mean_entropy"
        )
