"""Tests for the AttentionCapture context manager.

Two tiers:

- **Unit (synthetic)**: instantiate a tiny ``SwinUNETR`` from MONAI and
  verify the hook discovers exactly 8 ``WindowAttention`` modules, that the
  patched forward returns a tensor of the same shape as the unpatched
  forward, and that captured attention tensors have the documented shape.

- **Integration (real checkpoint)**: load the real BSF checkpoint and
  verify the patched decoder output is bit-identical to the unpatched
  output. Marked ``slow + real_data``; skipped automatically if the
  checkpoint is not on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from monai.networks.nets import SwinUNETR

from experiments.uncertainty_segmentation.explainability.engine.hooks import (
    AttentionCapture,
    discover_window_attention_modules,
)

pytestmark = [pytest.mark.experiment]


CHECKPOINT_PATH = Path(
    os.environ.get(
        "BSF_CHECKPOINT",
        "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/checkpoints/"
        "BrainSegFounder_finetuned_BraTS/finetuned_model_fold_0.pt",
    )
)


def _build_tiny_swinunetr() -> SwinUNETR:
    """Mirror the BSF-Tiny architecture without loading checkpoint weights."""
    return SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )


# ---------------------------------------------------------------------------
# Unit tests — synthetic, no checkpoint
# ---------------------------------------------------------------------------


class TestDiscoverWindowAttention:
    """``WindowAttention`` discovery and stage parsing."""

    @pytest.mark.unit
    def test_finds_eight_modules(self) -> None:
        model = _build_tiny_swinunetr()
        modules = discover_window_attention_modules(model)
        assert len(modules) == 8
        # Two blocks per stage, four stages.
        stages = sorted(stage for _, stage, _, _ in modules)
        assert stages == [1, 1, 2, 2, 3, 3, 4, 4]

    @pytest.mark.unit
    def test_block_indices_zero_one(self) -> None:
        model = _build_tiny_swinunetr()
        for _name, stage, block, _module in discover_window_attention_modules(model):
            assert block in (0, 1), f"unexpected block {block} at stage {stage}"

    @pytest.mark.unit
    def test_num_heads_per_stage(self) -> None:
        model = _build_tiny_swinunetr()
        expected = {1: 6, 2: 12, 3: 24, 4: 24}  # heads doubled at each stage from feature_size=48
        # SwinUNETR(num_heads=(3,6,12,24)) means stage1=3, stage2=6, etc.
        # Note: MONAI uses index i=0..3 for layers1..layers4.
        expected = {1: 3, 2: 6, 3: 12, 4: 24}
        for _name, stage, _block, module in discover_window_attention_modules(model):
            assert module.num_heads == expected[stage]


class TestAttentionCaptureContextManager:
    """Patch / unpatch and capture semantics."""

    @pytest.mark.unit
    def test_patches_and_restores_forward(self) -> None:
        model = _build_tiny_swinunetr()
        modules_before = discover_window_attention_modules(model)
        # Compare the underlying function objects, not bound methods (which
        # are recreated on every attribute access).
        original_funcs = {name: m.forward.__func__ for name, _s, _b, m in modules_before}
        with AttentionCapture(model):
            # Forward methods must point to a different function during capture.
            for name, _s, _b, m in modules_before:
                assert m.forward.__func__ is not original_funcs[name]
        # After exit the function reference must be the original again.
        for name, _s, _b, m in modules_before:
            assert m.forward.__func__ is original_funcs[name]

    @pytest.mark.unit
    def test_keys_match_discovered_modules(self) -> None:
        model = _build_tiny_swinunetr()
        with AttentionCapture(model) as cap:
            keys = cap.keys()
        assert sorted(keys) == [
            "stage_1_block_0", "stage_1_block_1",
            "stage_2_block_0", "stage_2_block_1",
            "stage_3_block_0", "stage_3_block_1",
            "stage_4_block_0", "stage_4_block_1",
        ]

    @pytest.mark.unit
    def test_target_stages_filter(self) -> None:
        model = _build_tiny_swinunetr()
        with AttentionCapture(model, target_stages={3, 4}) as cap:
            keys = cap.keys()
        assert sorted(keys) == [
            "stage_3_block_0", "stage_3_block_1",
            "stage_4_block_0", "stage_4_block_1",
        ]

    @pytest.mark.unit
    def test_callback_mode_requires_fn(self) -> None:
        model = _build_tiny_swinunetr()
        with pytest.raises(ValueError, match="process_fn"):
            AttentionCapture(model, mode="callback")

    @pytest.mark.unit
    @pytest.mark.slow
    def test_capture_shapes_synthetic(self) -> None:
        """Run a tiny forward pass and inspect captured attention shapes."""
        torch.manual_seed(0)
        model = _build_tiny_swinunetr().eval()
        x = torch.randn(1, 4, 64, 64, 64)
        with AttentionCapture(model) as cap:
            with torch.no_grad():
                _ = model(x)
            attn_maps = cap.get_attention_maps()
        assert sorted(attn_maps.keys()) == [
            "stage_1_block_0", "stage_1_block_1",
            "stage_2_block_0", "stage_2_block_1",
            "stage_3_block_0", "stage_3_block_1",
            "stage_4_block_0", "stage_4_block_1",
        ]
        for cap_attn in attn_maps.values():
            n = cap_attn.n_tokens
            # attn_weights: [n_windows*B, num_heads, N, N], must be square.
            assert cap_attn.attn_weights.dim() == 4
            assert cap_attn.attn_weights.shape[1] == cap_attn.num_heads
            assert cap_attn.attn_weights.shape[2] == n
            assert cap_attn.attn_weights.shape[3] == n
            # n_tokens cannot exceed the declared window volume.
            wd, wh, ww = cap_attn.declared_window_size
            assert n <= wd * wh * ww
            # Softmax rows must sum to 1 (within float32 tolerance).
            row_sums = cap_attn.attn_weights.sum(dim=-1)
            torch.testing.assert_close(
                row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4
            )

    @pytest.mark.unit
    @pytest.mark.slow
    def test_patched_forward_matches_unpatched_synthetic(self) -> None:
        """Decoder output with hook installed must equal the un-hooked output."""
        torch.manual_seed(0)
        model = _build_tiny_swinunetr().eval()
        x = torch.randn(1, 4, 64, 64, 64)
        with torch.no_grad():
            y_ref = model(x)
        with AttentionCapture(model):
            with torch.no_grad():
                y_hook = model(x)
        torch.testing.assert_close(y_hook, y_ref, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Integration test — real BSF checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.real_data
@pytest.mark.slow
@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"BSF checkpoint not at {CHECKPOINT_PATH}",
)
class TestRealCheckpointHook:
    """Validate the hook against the actual production checkpoint."""

    def _build(self) -> SwinUNETR:
        from growth.models.encoder.swin_loader import load_full_swinunetr

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_full_swinunetr(
            CHECKPOINT_PATH,
            freeze_encoder=True,
            freeze_decoder=True,
            out_channels=3,
            device=device,
        )
        model.eval()
        return model

    def test_discovers_eight_attention_modules(self) -> None:
        model = self._build()
        modules = discover_window_attention_modules(model)
        assert len(modules) == 8

    def test_patched_output_matches_unpatched(self) -> None:
        model = self._build()
        device = next(model.parameters()).device
        torch.manual_seed(0)
        x = torch.randn(1, 4, 128, 128, 128, device=device)
        with torch.no_grad():
            y_ref = model(x)
        with AttentionCapture(model):
            with torch.no_grad():
                y_hook = model(x)
        torch.testing.assert_close(y_hook, y_ref, atol=0.0, rtol=0.0)

    def test_attention_shapes_per_stage(self) -> None:
        model = self._build()
        device = next(model.parameters()).device
        torch.manual_seed(0)
        x = torch.randn(1, 4, 128, 128, 128, device=device)
        expected_heads = {1: 3, 2: 6, 3: 12, 4: 24}
        with AttentionCapture(model) as cap:
            with torch.no_grad():
                _ = model(x)
            attn_maps = cap.get_attention_maps()
        for cap_attn in attn_maps.values():
            stage = cap_attn.stage
            assert cap_attn.num_heads == expected_heads[stage]
            n = cap_attn.n_tokens
            assert cap_attn.attn_weights.shape[1] == cap_attn.num_heads
            assert cap_attn.attn_weights.shape[2] == n
            assert cap_attn.attn_weights.shape[3] == n
            # Softmax rows must normalise.
            row_sums = cap_attn.attn_weights.sum(dim=-1)
            torch.testing.assert_close(
                row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4
            )
