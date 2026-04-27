"""Tests for compile: compiled_metrics.parquet builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.uncertainty_segmentation.plotting.inter_lora.compile import (
    build_compiled_metrics,
    validate_compiled_metrics,
)
from experiments.uncertainty_segmentation.plotting.inter_lora.io_layer import (
    discover_ranks,
    load_rank_run,
)


@pytest.fixture()
def rank_runs(three_rank_fixture):
    dirs = discover_ranks(three_rank_fixture)
    return [load_rank_run(d) for d in dirs]


class TestBuildCompiledMetrics:
    def test_schema(self, rank_runs):
        df = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        expected_cols = {
            "rank",
            "label",
            "dice_mean",
            "dice_ci_lo",
            "dice_ci_hi",
            "delta_vs_baseline",
            "delta_ci_lo",
            "delta_ci_hi",
            "p_wilcoxon_raw",
            "p_wilcoxon_holm",
            "cohens_d",
            "ece",
            "brier",
            "cov95_deficit",
            "pct_bias_dominated",
            "icc",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_row_count(self, rank_runs):
        df = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        n_ranks = len(rank_runs)
        expected = (n_ranks + 1) * 4  # 3 labels + mean, (n_ranks + baseline)
        assert len(df) == expected

    def test_no_nan_in_core_cols(self, rank_runs):
        df = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        non_baseline = df[(df["rank"] > 0) & (df["label"] != "mean")]
        for col in ["dice_mean", "ece", "brier", "pct_bias_dominated"]:
            assert not non_baseline[col].isna().any(), f"NaN found in {col}"

    def test_baseline_row_present(self, rank_runs):
        df = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        baseline = df[df["rank"] == 0]
        assert len(baseline) == 4
        assert set(baseline["label"].unique()) == {"TC", "WT", "ET", "mean"}

    def test_baseline_delta_is_zero(self, rank_runs):
        df = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        baseline = df[df["rank"] == 0]
        np.testing.assert_allclose(baseline["delta_vs_baseline"].values, 0.0)

    def test_holm_monotonic_within_label(self, rank_runs):
        df = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        non_baseline = df[(df["rank"] > 0) & (df["label"] != "mean")]
        for label in ["TC", "WT", "ET"]:
            subset = non_baseline[non_baseline["label"] == label].sort_values("p_wilcoxon_raw")
            adjusted = subset["p_wilcoxon_holm"].values
            for i in range(1, len(adjusted)):
                assert adjusted[i] >= adjusted[i - 1] - 1e-12, (
                    f"Holm p-values not monotonic for {label}"
                )

    def test_deterministic(self, rank_runs):
        df1 = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        df2 = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestValidateCompiledMetrics:
    def test_valid(self, rank_runs):
        df = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        validate_compiled_metrics(df, n_ranks=len(rank_runs))

    def test_wrong_row_count(self, rank_runs):
        df = build_compiled_metrics(rank_runs, n_boot=50, seed=42)
        with pytest.raises(ValueError, match="Expected"):
            validate_compiled_metrics(df.iloc[:-1], n_ranks=len(rank_runs))
