"""Tests for the ASI (Attention Selectivity Index) module.

Coverage:

- **Mask alignment**: downsampling to stage resolution, padding-to-window
  multiple, MONAI-compatible ``window_partition`` (with the shifted-block
  ``torch.roll``), and round-trip recovery of the partition.
- **ASI semantics on synthetic attention**: uniform attention → ASI ≈ 1;
  block-diagonal (tumor↔tumor only) → ASI ≫ 1; tumor-suppressed
  attention → ASI ≪ 1; degenerate windows return ``NaN``.
- **End-to-end on a tiny SwinUNETR**: install :class:`AttentionCapture`
  in callback mode with :class:`ASIScanAccumulator` and verify that all
  expected ``stage_{s}_block_{b}`` keys appear with finite ASI values.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from monai.networks.nets import SwinUNETR

from experiments.uncertainty_segmentation.explainability.engine.asi import (
    ASIScanAccumulator,
    aggregate_asi_across_scans,
    compute_asi_per_window,
    downsample_mask_to_stage,
    partition_mask_windows,
    select_boundary_windows,
)
from experiments.uncertainty_segmentation.explainability.engine.hooks import (
    AttentionCapture,
)

pytestmark = [pytest.mark.experiment]


# ---------------------------------------------------------------------------
# Mask downsampling / window partitioning
# ---------------------------------------------------------------------------


class TestDownsampleMaskToStage:

    @pytest.mark.unit
    def test_output_shape_per_stage(self) -> None:
        mask = torch.zeros(96, 96, 96)
        mask[20:60, 20:60, 20:60] = 1.0
        for s in (1, 2, 3, 4):
            ds = downsample_mask_to_stage(mask, stage=s)
            expected = 96 // (2 ** s)
            assert ds.shape == (expected, expected, expected), (
                f"stage {s}: got {tuple(ds.shape)}, expected ({expected},)*3"
            )

    @pytest.mark.unit
    def test_binary_after_threshold(self) -> None:
        mask = torch.rand(48, 48, 48)
        mask = (mask > 0.5).float()
        ds = downsample_mask_to_stage(mask, stage=2)
        unique = torch.unique(ds)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    @pytest.mark.unit
    def test_invalid_stage_rejected(self) -> None:
        with pytest.raises(ValueError, match="stage"):
            downsample_mask_to_stage(torch.zeros(8, 8, 8), stage=0)
        with pytest.raises(ValueError, match="stage"):
            downsample_mask_to_stage(torch.zeros(8, 8, 8), stage=5)


class TestPartitionMaskWindows:

    @pytest.mark.unit
    def test_round_trip_no_shift(self) -> None:
        """Partitioning then summing recovers the total tumor count (no shift, no padding)."""
        d, h, w, ws = 14, 14, 14, 7
        mask = torch.zeros(d, h, w)
        mask[3:10, 3:10, 3:10] = 1.0
        windows = partition_mask_windows(mask, (ws, ws, ws), (0, 0, 0))
        assert windows.shape == (8, ws ** 3)  # 2^3 windows
        # Sum across windows must equal sum across original mask.
        assert windows.sum().item() == mask.sum().item()

    @pytest.mark.unit
    def test_shift_changes_window_assignment(self) -> None:
        """Shifted-block partitioning must produce a different per-window
        tumor distribution than the non-shifted one."""
        ws = 7
        mask = torch.zeros(14, 14, 14)
        mask[2:5, 2:5, 2:5] = 1.0
        no_shift = partition_mask_windows(mask, (ws,) * 3, (0,) * 3)
        shifted = partition_mask_windows(mask, (ws,) * 3, (ws // 2,) * 3)
        # The total tumor count is preserved regardless of shift.
        assert no_shift.sum().item() == shifted.sum().item() == mask.sum().item()
        # Per-window distribution differs.
        assert not torch.equal(no_shift, shifted)

    @pytest.mark.unit
    def test_padding_to_window_multiple(self) -> None:
        """Mask spatial dims that are NOT a multiple of window_size get padded with zeros."""
        ws = 7
        mask = torch.ones(8, 8, 8)
        windows = partition_mask_windows(mask, (ws,) * 3, (0,) * 3)
        # Padded to 14³ → 2³ = 8 windows.
        assert windows.shape == (8, ws ** 3)
        # Original mask has 8³ = 512 ones; padding adds zeros.
        assert windows.sum().item() == 8 ** 3


# ---------------------------------------------------------------------------
# Boundary window selection
# ---------------------------------------------------------------------------


class TestBoundaryWindows:

    @pytest.mark.unit
    def test_pure_windows_excluded(self) -> None:
        # Window 0: pure tumor; window 1: pure background; window 2: mixed.
        mask_w = torch.tensor([
            [1.0] * 10,
            [0.0] * 10,
            [1.0] * 5 + [0.0] * 5,
        ])
        idx = select_boundary_windows(mask_w, min_tumor=2, min_nontumor=2)
        assert idx.tolist() == [2]

    @pytest.mark.unit
    def test_min_count_threshold_applied(self) -> None:
        # Window has 3 tumor tokens; ``min_tumor=5`` should reject it.
        mask_w = torch.tensor([[1.0] * 3 + [0.0] * 7])
        assert select_boundary_windows(mask_w, min_tumor=5).numel() == 0
        # ``min_tumor=3`` accepts.
        assert select_boundary_windows(mask_w, min_tumor=3).numel() == 1


# ---------------------------------------------------------------------------
# Per-window ASI semantics
# ---------------------------------------------------------------------------


class TestComputeASIPerWindow:

    @pytest.mark.unit
    def test_uniform_attention_gives_unity(self) -> None:
        """Uniform attention rows ⇒ ASI = 1 for any tumor partition."""
        n, h = 16, 4
        attn = torch.full((h, n, n), 1.0 / n)  # already normalised
        mask_w = torch.zeros(n)
        mask_w[:8] = 1.0
        asi = compute_asi_per_window(attn, mask_w)
        torch.testing.assert_close(
            asi, torch.ones(h), atol=1e-6, rtol=1e-6,
        )

    @pytest.mark.unit
    def test_tumor_to_tumor_only_gives_high_asi(self) -> None:
        """Attention concentrated tumor→tumor ⇒ ASI ≫ 1."""
        n, h = 16, 2
        n_t = 8
        # Tumor queries put almost all mass on tumor keys; bg gets a tiny
        # epsilon to avoid div-by-zero.
        eps = 1e-3
        attn = torch.zeros(h, n, n)
        attn[:, :n_t, :n_t] = (1.0 - eps) / n_t
        attn[:, :n_t, n_t:] = eps / (n - n_t)
        # bg-query rows can be uniform.
        attn[:, n_t:, :] = 1.0 / n
        mask_w = torch.zeros(n)
        mask_w[:n_t] = 1.0
        asi = compute_asi_per_window(attn, mask_w)
        # mu_TT/mu_TB ≈ ((1-eps)/n_t) / (eps/(n-n_t))
        expected = ((1.0 - eps) / n_t) / (eps / (n - n_t))
        torch.testing.assert_close(
            asi, torch.full((h,), expected, dtype=torch.float32),
            atol=1e-1, rtol=1e-3,
        )
        assert (asi > 100.0).all()

    @pytest.mark.unit
    def test_tumor_to_bg_only_gives_low_asi(self) -> None:
        """Attention concentrated tumor→bg ⇒ ASI ≪ 1."""
        n, h = 16, 2
        n_t = 8
        eps = 1e-3
        attn = torch.zeros(h, n, n)
        attn[:, :n_t, n_t:] = (1.0 - eps) / (n - n_t)
        attn[:, :n_t, :n_t] = eps / n_t
        attn[:, n_t:, :] = 1.0 / n
        mask_w = torch.zeros(n)
        mask_w[:n_t] = 1.0
        asi = compute_asi_per_window(attn, mask_w)
        assert (asi < 0.01).all()

    @pytest.mark.unit
    def test_degenerate_window_returns_nan(self) -> None:
        n, h = 8, 3
        attn = torch.full((h, n, n), 1.0 / n)
        # Pure tumor.
        mask_t = torch.ones(n)
        out = compute_asi_per_window(attn, mask_t)
        assert torch.isnan(out).all()
        # Pure background.
        mask_b = torch.zeros(n)
        out = compute_asi_per_window(attn, mask_b)
        assert torch.isnan(out).all()

    @pytest.mark.unit
    def test_softmax_rows_normalised_assumption(self) -> None:
        """Sanity check: ASI is well-defined whether or not rows are normalised
        (the metric is a ratio of means)."""
        n, h = 16, 2
        attn = torch.rand(h, n, n)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        mask_w = torch.zeros(n)
        mask_w[:8] = 1.0
        asi = compute_asi_per_window(attn, mask_w)
        assert torch.isfinite(asi).all()


# ---------------------------------------------------------------------------
# End-to-end with the AttentionCapture context manager
# ---------------------------------------------------------------------------


def _build_tiny_swinunetr() -> SwinUNETR:
    return SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )


class TestASIWithCaptureSynthetic:
    """Streaming ASI from a tiny SwinUNETR forward pass."""

    @pytest.mark.unit
    @pytest.mark.slow
    def test_accumulator_collects_all_stages_blocks(self) -> None:
        torch.manual_seed(0)
        model = _build_tiny_swinunetr().eval()
        x = torch.randn(1, 4, 64, 64, 64)
        # GT mask at image resolution; central cube 16-48 ⇒ stage-grid mask
        # is non-trivial at all stages.
        gt = torch.zeros(64, 64, 64)
        gt[16:48, 16:48, 16:48] = 1.0

        acc = ASIScanAccumulator(gt_mask=gt)
        with AttentionCapture(model, mode="callback", process_fn=acc):
            with torch.no_grad():
                _ = model(x)
        result = acc.result()

        expected_keys = {
            f"stage_{s}_block_{b}" for s in (1, 2, 3, 4) for b in (0, 1)
        }
        assert set(result.n_windows_total.keys()) == expected_keys
        # At stage 4 the feature map collapses to 4³ — the window clamps
        # to 4 → 1 single window per block, which may or may not be
        # boundary depending on the downsampled mask.  At stages 1-3 we
        # expect at least one boundary window per block.
        for stage in (1, 2, 3):
            for block in (0, 1):
                key = f"stage_{stage}_block_{block}"
                assert result.n_windows_boundary[key] > 0, (
                    f"{key}: no boundary windows found"
                )
                arr = result.per_block_per_head[key]
                assert arr.ndim == 2
                # Heads per stage: {1:3, 2:6, 3:12, 4:24}.
                expected_heads = {1: 3, 2: 6, 3: 12}[stage]
                assert arr.shape[1] == expected_heads
                # ASI must be positive and finite for non-degenerate windows.
                assert np.all(np.isfinite(arr))
                assert np.all(arr >= 0.0)

    @pytest.mark.unit
    @pytest.mark.slow
    def test_target_stages_filter_propagates(self) -> None:
        torch.manual_seed(0)
        model = _build_tiny_swinunetr().eval()
        x = torch.randn(1, 4, 64, 64, 64)
        gt = torch.zeros(64, 64, 64)
        gt[20:44, 20:44, 20:44] = 1.0

        acc = ASIScanAccumulator(gt_mask=gt, target_stages={3, 4})
        with AttentionCapture(model, mode="callback", process_fn=acc):
            with torch.no_grad():
                _ = model(x)
        result = acc.result()

        observed_stages = {int(k.split("_")[1]) for k in result.n_windows_total}
        assert observed_stages == {3, 4}


class TestAggregateAcrossScans:

    @pytest.mark.unit
    def test_concatenates_arrays(self) -> None:
        from experiments.uncertainty_segmentation.explainability.engine.asi import (
            ASIScanResult,
        )
        r1 = ASIScanResult(
            per_block_per_head={"stage_1_block_0": np.ones((10, 3))},
            n_windows_total={"stage_1_block_0": 10},
            n_windows_boundary={"stage_1_block_0": 10},
        )
        r2 = ASIScanResult(
            per_block_per_head={"stage_1_block_0": np.zeros((5, 3))},
            n_windows_total={"stage_1_block_0": 5},
            n_windows_boundary={"stage_1_block_0": 5},
        )
        out = aggregate_asi_across_scans([r1, r2])
        assert out["stage_1_block_0"].shape == (15, 3)
        assert out["stage_1_block_0"][:10].sum() == 30.0
        assert out["stage_1_block_0"][10:].sum() == 0.0
