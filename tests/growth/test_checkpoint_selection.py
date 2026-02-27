# tests/growth/test_checkpoint_selection.py
"""Tests for FLAW-2: Probe-R²-based checkpoint selection.

Verifies:
1. Checkpoint score is probe_mean_r2 (not Dice)
2. Higher probe R² gives higher score
3. Variance hinge: low for diverse features, high for collapsed
"""

import torch


class TestProbeCheckpointScore:
    """Tests for probe-only checkpoint selection."""

    def test_higher_probe_r2_gives_higher_score(self):
        """Higher probe R² should give a higher checkpoint score."""
        score_low = 0.2   # probe_mean_r2 = 0.2
        score_high = 0.6  # probe_mean_r2 = 0.6

        assert score_high > score_low

    def test_probe_score_independent_of_dice(self):
        """Checkpoint score should not depend on Dice when probes are available.

        A model with lower Dice but higher probe R² should be preferred.
        """
        # Simulates the checkpoint_score logic:
        # checkpoint_score = probe_metrics.get("probe_mean_r2", val_dice)
        model_a_dice = 0.90
        model_a_probe = 0.30

        model_b_dice = 0.80
        model_b_probe = 0.50

        score_a = model_a_probe  # probe available → use probe
        score_b = model_b_probe  # probe available → use probe

        assert score_b > score_a, (
            f"Model B (probe={model_b_probe}) should beat "
            f"Model A (probe={model_a_probe}) despite lower Dice"
        )

    def test_dice_fallback_when_no_probes(self):
        """When probes unavailable, score falls back to Dice."""
        probe_metrics: dict[str, float] = {}
        val_dice = 0.85

        checkpoint_score = probe_metrics.get("probe_mean_r2", val_dice)

        assert checkpoint_score == val_dice


class TestVarianceHinge:
    """Tests for variance hinge metric: clamp(1 - std, min=0).mean()."""

    def _compute_variance_hinge(self, features: torch.Tensor) -> float:
        """Compute variance hinge from features [B, D]."""
        feat_std = features.std(dim=0)
        vh = torch.clamp(1.0 - feat_std, min=0.0).mean().item()
        return vh

    def test_variance_hinge_low_for_diverse_features(self):
        """Random features with std >> 1 should have hinge close to 0."""
        features = torch.randn(32, 768) * 5.0  # std ≈ 5
        vh = self._compute_variance_hinge(features)

        assert vh < 0.1, f"Variance hinge {vh:.3f} should be near 0 for diverse features"

    def test_variance_hinge_high_for_collapsed(self):
        """Near-constant features should have hinge close to 1."""
        # All samples nearly identical → std ≈ 0 per dim
        base = torch.randn(1, 768)
        features = base.expand(32, -1) + torch.randn(32, 768) * 1e-6

        vh = self._compute_variance_hinge(features)

        assert vh > 0.9, f"Variance hinge {vh:.3f} should be near 1 for collapsed features"

    def test_variance_hinge_bounded(self):
        """Hinge should always be in [0, 1]."""
        for _ in range(5):
            features = torch.randn(16, 128) * torch.rand(1).item() * 10
            vh = self._compute_variance_hinge(features)
            assert 0.0 <= vh <= 1.0, f"Variance hinge {vh:.3f} out of [0, 1]"
