# tests/growth/test_lr_scheduler.py
"""Tests for FLAW-3: ReduceLROnPlateau with warmup schedule.

Verifies:
1. Warmup increases LR from start_factor to 1.0
2. Plateau reduces LR after patience epochs of stagnation
3. Warmup → plateau transition works correctly
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau


def _make_optimizer_and_schedulers(
    base_lr: float = 1e-3,
    warmup_epochs: int = 5,
    start_factor: float = 0.01,
    plateau_factor: float = 0.5,
    plateau_patience: int = 3,
) -> tuple[torch.optim.Optimizer, dict]:
    """Create optimizer + scheduler_info matching create_optimizer() output."""
    param = torch.nn.Parameter(torch.zeros(10))
    optimizer = AdamW([param], lr=base_lr)

    warmup = LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs,
    )
    plateau = ReduceLROnPlateau(
        optimizer, mode='max', factor=plateau_factor, patience=plateau_patience, min_lr=1e-7,
    )

    scheduler_info = {
        'warmup': warmup,
        'plateau': plateau,
        'warmup_epochs': warmup_epochs,
    }
    return optimizer, scheduler_info


class TestWarmupScheduler:
    """Tests for the warmup phase."""

    def test_warmup_increases_lr(self):
        """LR should increase monotonically during warmup."""
        optimizer, si = _make_optimizer_and_schedulers(
            base_lr=1e-3, warmup_epochs=5, start_factor=0.01,
        )

        lrs = [optimizer.param_groups[0]['lr']]
        for epoch in range(5):
            si['warmup'].step()
            lrs.append(optimizer.param_groups[0]['lr'])

        # Each LR should be >= previous
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], (
                f"LR should increase during warmup: {lrs[i - 1]:.6f} -> {lrs[i]:.6f}"
            )

        # Final LR should be close to base_lr
        assert abs(lrs[-1] - 1e-3) < 1e-5, f"Final warmup LR {lrs[-1]:.6f} != base {1e-3}"

    def test_warmup_start_lr(self):
        """Initial LR should be base_lr * start_factor."""
        optimizer, _ = _make_optimizer_and_schedulers(
            base_lr=1e-3, start_factor=0.01,
        )

        initial_lr = optimizer.param_groups[0]['lr']
        assert abs(initial_lr - 1e-3 * 0.01) < 1e-7


class TestPlateauScheduler:
    """Tests for the plateau phase."""

    def test_plateau_reduces_lr_after_patience(self):
        """LR should reduce after patience epochs without improvement."""
        optimizer, si = _make_optimizer_and_schedulers(
            base_lr=1e-3, warmup_epochs=0, start_factor=1.0,
            plateau_patience=3, plateau_factor=0.5,
        )

        # Simulate stagnation: same metric for patience+1 epochs
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(5):
            si['plateau'].step(0.5)  # Same metric each time

        reduced_lr = optimizer.param_groups[0]['lr']
        assert reduced_lr < initial_lr, (
            f"LR should reduce after stagnation: {initial_lr:.6f} -> {reduced_lr:.6f}"
        )

    def test_plateau_does_not_reduce_on_improvement(self):
        """LR should stay stable when metric keeps improving."""
        optimizer, si = _make_optimizer_and_schedulers(
            base_lr=1e-3, warmup_epochs=0, start_factor=1.0,
            plateau_patience=3,
        )

        for i in range(10):
            si['plateau'].step(0.5 + i * 0.01)  # Improving metric

        final_lr = optimizer.param_groups[0]['lr']
        assert abs(final_lr - 1e-3) < 1e-6, (
            f"LR should not reduce during improvement: {final_lr:.6f}"
        )


class TestWarmupToPlateauTransition:
    """Tests for the full warmup → plateau schedule."""

    def test_full_schedule(self):
        """Warmup increases LR, then plateau can reduce it."""
        optimizer, si = _make_optimizer_and_schedulers(
            base_lr=1e-3, warmup_epochs=5, start_factor=0.01,
            plateau_patience=3, plateau_factor=0.5,
        )

        lrs = []

        # Warmup phase
        for epoch in range(5):
            si['warmup'].step()
            lrs.append(optimizer.param_groups[0]['lr'])

        # LR should have reached base_lr
        assert abs(lrs[-1] - 1e-3) < 1e-5

        # Plateau phase with stagnation
        for epoch in range(10):
            si['plateau'].step(0.5)  # Constant metric
            lrs.append(optimizer.param_groups[0]['lr'])

        # LR should have decreased from stagnation
        assert lrs[-1] < lrs[4], (
            f"LR should decrease after stagnation: warmup_end={lrs[4]:.6f}, "
            f"final={lrs[-1]:.6f}"
        )
