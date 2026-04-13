"""Tests for epistemic uncertainty diagnostics (Proposal 1: k* threshold)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.uncertainty_segmentation.plotting.epistemic_metrics import (
    K_STAR_MAX,
    compute_bias_diagnostics,
    compute_bias_dominance_threshold,
    compute_rank_summary,
)

pytestmark = [pytest.mark.evaluation, pytest.mark.unit]


def _make_bias_row(
    scan_id: str,
    n_members: int,
    volume_gt: float,
    volume_mean: float,
    volume_std: float,
    logvol_mean: float,
    logvol_std: float,
) -> dict:
    """Construct a single-row bias-diagnostics record with explicit fields."""
    bias = volume_mean - volume_gt
    logvol_gt = float(np.log1p(volume_gt))
    logvol_bias = logvol_mean - logvol_gt
    return {
        "scan_id": scan_id,
        "n_members": n_members,
        "volume_gt": volume_gt,
        "volume_ensemble_mean": volume_mean,
        "volume_ensemble_std": volume_std,
        "bias": bias,
        "abs_bias": abs(bias),
        "bias_to_std_ratio": abs(bias) / max(volume_std, 1e-12),
        "logvol_gt": logvol_gt,
        "logvol_ensemble_mean": logvol_mean,
        "logvol_ensemble_std": logvol_std,
        "logvol_bias": logvol_bias,
        "logvol_abs_bias": abs(logvol_bias),
        "logvol_bias_to_std_ratio": abs(logvol_bias) / max(logvol_std, 1e-12),
    }


class TestKStarFormula:
    """k* = ceil((sigma / |bias|)^2) on a synthetic bias table."""

    def test_k_star_matches_closed_form(self):
        # Construct two scans with known (sigma, bias) and verify k*.
        rows = [
            _make_bias_row(
                scan_id="s_small_bias", n_members=20,
                volume_gt=10.0, volume_mean=10.05, volume_std=0.5,
                logvol_mean=2.42, logvol_std=0.1,  # logvol_gt=log(11)≈2.3979
            ),
            _make_bias_row(
                scan_id="s_big_bias", n_members=20,
                volume_gt=10.0, volume_mean=11.0, volume_std=0.01,
                logvol_mean=2.50, logvol_std=0.005,
            ),
        ]
        bias_df = pd.DataFrame(rows)
        k = compute_bias_dominance_threshold(bias_df)

        # k*_log for s_small_bias ≈ ceil((0.1 / |2.42 - log(11)|)^2).
        logvol_gt = float(np.log1p(10.0))
        expected_small = int(np.ceil((0.1 / abs(2.42 - logvol_gt)) ** 2))
        expected_big = int(np.ceil((0.005 / abs(2.50 - logvol_gt)) ** 2))

        got = dict(zip(k["scan_id"], k["k_star_logvol"]))
        assert int(got["s_small_bias"]) == expected_small
        assert int(got["s_big_bias"]) == expected_big

    def test_k_star_is_non_negative_integer(self):
        rows = [
            _make_bias_row(
                scan_id=f"s{i}", n_members=20,
                volume_gt=10.0, volume_mean=10.0 + 0.1 * i, volume_std=0.3,
                logvol_mean=2.40, logvol_std=0.05,
            )
            for i in range(1, 6)
        ]
        k = compute_bias_dominance_threshold(pd.DataFrame(rows))
        assert (k["k_star_logvol"] >= 0).all()
        assert (k["k_star_raw"] >= 0).all()
        assert pd.api.types.is_integer_dtype(k["k_star_logvol"])


class TestKStarSaturation:
    """|bias| == 0 must saturate k* at K_STAR_MAX and flag the row."""

    def test_zero_log_bias_saturates_and_flags(self):
        rows = [
            _make_bias_row(
                scan_id="s_lucky", n_members=20,
                volume_gt=10.0, volume_mean=12.0, volume_std=0.3,
                # logvol_mean exactly matches logvol_gt → logvol bias == 0.
                logvol_mean=float(np.log1p(10.0)), logvol_std=0.05,
            ),
            _make_bias_row(
                scan_id="s_normal", n_members=20,
                volume_gt=10.0, volume_mean=10.5, volume_std=0.2,
                logvol_mean=2.45, logvol_std=0.03,
            ),
        ]
        k = compute_bias_dominance_threshold(pd.DataFrame(rows))

        row = k.set_index("scan_id").loc["s_lucky"]
        assert bool(row["k_star_saturated"]) is True
        assert int(row["k_star_logvol"]) == K_STAR_MAX

        row2 = k.set_index("scan_id").loc["s_normal"]
        assert bool(row2["k_star_saturated"]) is False


class TestKStarDegenerate:
    """sigma == 0 must flag degenerate_ensemble."""

    def test_zero_sigma_flagged(self):
        rows = [
            _make_bias_row(
                scan_id="s_collapsed", n_members=20,
                volume_gt=10.0, volume_mean=11.0, volume_std=0.0,
                logvol_mean=2.45, logvol_std=0.0,
            ),
            _make_bias_row(
                scan_id="s_healthy", n_members=20,
                volume_gt=10.0, volume_mean=10.5, volume_std=0.2,
                logvol_mean=2.45, logvol_std=0.03,
            ),
        ]
        k = compute_bias_dominance_threshold(pd.DataFrame(rows))
        idx = k.set_index("scan_id")
        assert bool(idx.loc["s_collapsed", "degenerate_ensemble"]) is True
        assert bool(idx.loc["s_healthy", "degenerate_ensemble"]) is False


class TestKStarExceedsM:
    """k_star_exceeds_M flags scans where the ensemble is NOT yet bias-dominated."""

    def test_exceeds_m_when_sigma_large(self):
        # sigma=0.5, |bias|=0.05 → k*=100; if M=20, k*>M.
        rows = [
            _make_bias_row(
                scan_id="s_under", n_members=20,
                volume_gt=10.0, volume_mean=10.5, volume_std=5.0,
                logvol_mean=2.45, logvol_std=0.5,
            ),
            _make_bias_row(
                scan_id="s_over", n_members=20,
                volume_gt=10.0, volume_mean=12.0, volume_std=0.01,
                logvol_mean=2.55, logvol_std=0.001,
            ),
        ]
        k = compute_bias_dominance_threshold(pd.DataFrame(rows))
        idx = k.set_index("scan_id")
        # s_under: sigma=0.5, |bias|≈|2.45-log(11)|≈0.052, k*≈93 > 20
        assert bool(idx.loc["s_under", "k_star_exceeds_M"]) is True
        # s_over: sigma=0.001, |bias|≈0.152, k* very small, well <= 20
        assert bool(idx.loc["s_over", "k_star_exceeds_M"]) is False


class TestRankSummaryWithKStar:
    """compute_rank_summary integrates k* aggregates and respects saturation/degeneracy."""

    def _build(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        bias_rows = [
            _make_bias_row(
                scan_id=f"s{i:02d}", n_members=20,
                volume_gt=10.0, volume_mean=10.5, volume_std=0.2,
                logvol_mean=2.45, logvol_std=0.05,
            )
            for i in range(10)
        ]
        # Inject one degenerate and one saturated row.
        bias_rows.append(
            _make_bias_row(
                scan_id="s_degen", n_members=20,
                volume_gt=10.0, volume_mean=10.0, volume_std=0.0,
                logvol_mean=float(np.log1p(10.0)) + 0.01, logvol_std=0.0,
            )
        )
        bias_rows.append(
            _make_bias_row(
                scan_id="s_sat", n_members=20,
                volume_gt=10.0, volume_mean=12.0, volume_std=0.3,
                logvol_mean=float(np.log1p(10.0)), logvol_std=0.05,
            )
        )
        bias_df = pd.DataFrame(bias_rows)
        k_star_df = compute_bias_dominance_threshold(bias_df)

        # Dummy calibration table (values don't matter for this test).
        calib_df = pd.DataFrame({
            "nominal_level": [0.50, 0.80, 0.90, 0.95],
            "t_multiplier": [1.0, 1.3, 1.7, 2.1],
            "n_scans": [12, 12, 12, 12],
            "n_covered": [6, 9, 10, 11],
            "empirical_coverage": [0.50, 0.75, 0.83, 0.92],
            "coverage_deficit": [0.0, 0.05, 0.07, 0.03],
        })
        return bias_df, calib_df, k_star_df

    def test_summary_contains_k_star_fields(self):
        bias_df, calib_df, k_star_df = self._build()
        summary = compute_rank_summary(bias_df, calib_df, rank=8, k_star_df=k_star_df)

        for key in (
            "median_k_star_logvol",
            "pct_scans_k_star_eq_1",
            "pct_scans_k_star_exceeds_M",
            "pct_scans_k_star_saturated",
            "pct_scans_degenerate_ensemble",
        ):
            assert key in summary, f"{key} missing from rank summary"

    def test_median_excludes_degenerate_and_is_finite(self):
        bias_df, calib_df, k_star_df = self._build()
        summary = compute_rank_summary(bias_df, calib_df, rank=8, k_star_df=k_star_df)

        # s_sat row is saturated (= K_STAR_MAX). Median is robust, so the
        # headline median_k_star_logvol must stay well below K_STAR_MAX.
        assert summary["median_k_star_logvol"] < K_STAR_MAX / 2

        # pct_scans_k_star_saturated must reflect exactly one saturated
        # scan out of the non-degenerate subset (11 scans).
        expected_pct_sat = 1.0 / 11.0
        assert abs(summary["pct_scans_k_star_saturated"] - expected_pct_sat) < 1e-6

        # pct_scans_degenerate_ensemble is over the full 12-scan population.
        assert abs(summary["pct_scans_degenerate_ensemble"] - 1.0 / 12.0) < 1e-6

    def test_summary_without_k_star_backward_compatible(self):
        bias_df, calib_df, _ = self._build()
        # Passing no k_star_df should still work and not emit k_* keys.
        summary = compute_rank_summary(bias_df, calib_df, rank=8)
        assert "median_k_star_logvol" not in summary
        assert "median_logvol_std" in summary  # baseline key still present


class TestComputeBiasDiagnosticsSanity:
    """Basic sanity check that compute_bias_diagnostics produces usable input for k*."""

    def test_bias_diagnostics_roundtrip_feeds_k_star(self):
        n_members = 5
        per_member = pd.DataFrame({
            "scan_id": ["sA"] * n_members + ["sB"] * n_members,
            "member_id": list(range(n_members)) * 2,
            "volume_pred": [9.8, 10.2, 9.9, 10.1, 10.0] + [12.0, 12.5, 11.5, 12.2, 11.8],
        })
        ensemble = pd.DataFrame({
            "scan_id": ["sA", "sB"],
            "volume_gt": [10.0, 10.0],
        })
        bias_df = compute_bias_diagnostics(per_member, ensemble)
        k_star_df = compute_bias_dominance_threshold(bias_df)

        assert set(k_star_df["scan_id"]) == {"sA", "sB"}
        assert "k_star_logvol" in k_star_df.columns
        assert "k_star_exceeds_M" in k_star_df.columns
        # sA: low bias, moderate std → k* likely > 5; sB: large bias, moderate std → k* low.
        idx = k_star_df.set_index("scan_id")
        assert int(idx.loc["sA", "k_star_logvol"]) > int(idx.loc["sB", "k_star_logvol"])
