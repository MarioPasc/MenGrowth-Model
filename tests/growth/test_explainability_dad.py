"""Tests for the DAD (Domain Attention Divergence) module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from monai.networks.nets import SwinUNETR

from experiments.uncertainty_segmentation.explainability.engine.dad import (
    DADScanAccumulator,
    DADScanResult,
    compute_dad_with_permutation,
    symmetric_kl,
)
from experiments.uncertainty_segmentation.explainability.engine.hooks import (
    AttentionCapture,
)

pytestmark = [pytest.mark.experiment]


# ---------------------------------------------------------------------------
# symmetric_kl
# ---------------------------------------------------------------------------


class TestSymmetricKL:

    @pytest.mark.unit
    def test_zero_for_identical_distributions(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(symmetric_kl(p, p)) < 1e-10

    @pytest.mark.unit
    def test_positive_and_symmetric(self) -> None:
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.4, 0.5])
        d_pq = symmetric_kl(p, q)
        d_qp = symmetric_kl(q, p)
        assert d_pq > 0
        assert abs(d_pq - d_qp) < 1e-10

    @pytest.mark.unit
    def test_handles_zeros(self) -> None:
        """Distributions with exact zeros (from masked attention) should not blow up."""
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([0.0, 0.5, 0.5])
        d = symmetric_kl(p, q)
        assert np.isfinite(d) and d > 0

    @pytest.mark.unit
    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            symmetric_kl(np.ones(3), np.ones(4))


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


class TestPermutationTest:

    @pytest.mark.unit
    def test_identical_cohorts_give_high_pvalue(self) -> None:
        """If both cohorts come from the same distribution, p-value
        should be near 0.5 (no domain effect)."""
        rng = np.random.RandomState(0)
        n_scans, h, n = 8, 2, 16

        def _make() -> DADScanResult:
            v = rng.dirichlet(alpha=np.ones(n), size=h)
            return DADScanResult(
                row_avg={"stage_1_block_0": v.astype(np.float32)},
                n_rows={"stage_1_block_0": 1},
            )

        cohort_a = [_make() for _ in range(n_scans)]
        cohort_b = [_make() for _ in range(n_scans)]
        out = compute_dad_with_permutation(
            cohort_a, cohort_b, n_perm=200, seed=0,
        )
        for stat in out["stage_1_block_0"]:
            assert stat.p_value > 0.05, (
                f"head {stat.head}: identical cohorts produced p={stat.p_value:.3f}"
            )

    @pytest.mark.unit
    def test_disjoint_cohorts_give_low_pvalue(self) -> None:
        """If cohort A puts mass on the first half and B on the second
        half of the support, DAD should be highly significant."""
        n_scans, h, n = 6, 2, 8
        # Cohort A: heavy weight on positions 0-3; cohort B: on 4-7.
        a_template = np.array([0.225, 0.225, 0.225, 0.225, 0.025, 0.025, 0.025, 0.025])
        b_template = np.array([0.025, 0.025, 0.025, 0.025, 0.225, 0.225, 0.225, 0.225])

        rng = np.random.RandomState(1)
        cohort_a = []
        cohort_b = []
        for _ in range(n_scans):
            # Add small noise but keep dominant pattern.
            va = a_template + rng.normal(0, 0.005, size=(h, n))
            vb = b_template + rng.normal(0, 0.005, size=(h, n))
            va = np.clip(va, 1e-6, None)
            vb = np.clip(vb, 1e-6, None)
            va /= va.sum(axis=-1, keepdims=True)
            vb /= vb.sum(axis=-1, keepdims=True)
            cohort_a.append(DADScanResult(
                row_avg={"stage_1_block_0": va.astype(np.float32)},
                n_rows={"stage_1_block_0": 1},
            ))
            cohort_b.append(DADScanResult(
                row_avg={"stage_1_block_0": vb.astype(np.float32)},
                n_rows={"stage_1_block_0": 1},
            ))
        out = compute_dad_with_permutation(
            cohort_a, cohort_b, n_perm=500, seed=42,
        )
        for stat in out["stage_1_block_0"]:
            assert stat.dad_observed > 0.5
            assert stat.p_value < 0.05

    @pytest.mark.unit
    def test_p_value_lower_bound(self) -> None:
        """p-value is at least 1/(n_perm+1) by construction."""
        n_scans, h, n = 4, 1, 4
        cohort_a = [
            DADScanResult(
                row_avg={"stage_1_block_0": np.array([[0.97, 0.01, 0.01, 0.01]])},
                n_rows={"stage_1_block_0": 1},
            ) for _ in range(n_scans)
        ]
        cohort_b = [
            DADScanResult(
                row_avg={"stage_1_block_0": np.array([[0.01, 0.01, 0.01, 0.97]])},
                n_rows={"stage_1_block_0": 1},
            ) for _ in range(n_scans)
        ]
        n_perm = 100
        out = compute_dad_with_permutation(
            cohort_a, cohort_b, n_perm=n_perm, seed=0,
        )
        for stat in out["stage_1_block_0"]:
            assert stat.p_value >= 1.0 / (n_perm + 1)


# ---------------------------------------------------------------------------
# End-to-end DAD with the hook
# ---------------------------------------------------------------------------


def _build_tiny_swinunetr() -> SwinUNETR:
    return SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )


class TestDADWithCapture:

    @pytest.mark.unit
    @pytest.mark.slow
    def test_accumulator_yields_simplex(self) -> None:
        """Streaming row-average must produce probability vectors."""
        torch.manual_seed(0)
        model = _build_tiny_swinunetr().eval()
        x = torch.randn(1, 4, 64, 64, 64)

        acc = DADScanAccumulator()
        with AttentionCapture(model, mode="callback", process_fn=acc):
            with torch.no_grad():
                _ = model(x)
        result = acc.result()

        expected = {f"stage_{s}_block_{b}" for s in (1, 2, 3, 4) for b in (0, 1)}
        assert set(result.row_avg.keys()) == expected
        for key, arr in result.row_avg.items():
            assert arr.ndim == 2  # [H, N]
            row_sums = arr.sum(axis=-1)
            np.testing.assert_allclose(
                row_sums, np.ones_like(row_sums), atol=1e-5,
                err_msg=f"{key}: row sums {row_sums} not 1",
            )
