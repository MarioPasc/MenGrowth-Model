"""Tests for io_layer: rank discovery, loading, and validation."""

from __future__ import annotations

import pytest

from experiments.uncertainty_segmentation.plotting.inter_lora.errors import (
    BaselineMismatchError,
    RankDiscoveryError,
)
from experiments.uncertainty_segmentation.plotting.inter_lora.io_layer import (
    discover_ranks,
    load_rank_run,
    validate_baseline_consistency,
)


class TestDiscoverRanks:
    def test_happy_path(self, three_rank_fixture):
        dirs = discover_ranks(three_rank_fixture)
        assert len(dirs) == 3
        assert all(d.is_dir() for d in dirs)
        names = [d.name for d in dirs]
        assert names[0].startswith("r4_")
        assert names[1].startswith("r8_")
        assert names[2].startswith("r16_")

    def test_sorted_ascending(self, three_rank_fixture):
        dirs = discover_ranks(three_rank_fixture)
        ranks_parsed = [int(d.name.split("_")[0][1:]) for d in dirs]
        assert ranks_parsed == sorted(ranks_parsed)

    def test_too_few_ranks(self, two_rank_fixture):
        with pytest.raises(RankDiscoveryError, match="need >= 3"):
            discover_ranks(two_rank_fixture)

    def test_filter_expected_too_few(self, three_rank_fixture):
        with pytest.raises(RankDiscoveryError):
            discover_ranks(three_rank_fixture, expected=frozenset({4, 16}))

    def test_filter_expected_all(self, three_rank_fixture):
        dirs = discover_ranks(three_rank_fixture, expected=frozenset({4, 8, 16}))
        assert len(dirs) == 3

    def test_empty_dir(self, tmp_path):
        with pytest.raises(RankDiscoveryError):
            discover_ranks(tmp_path)


class TestLoadRankRun:
    def test_load_all_fields(self, three_rank_fixture):
        dirs = discover_ranks(three_rank_fixture)
        run = load_rank_run(dirs[0])
        assert run.rank == 4
        assert len(run.ensemble_dice) > 0
        assert len(run.per_member_dice) > 0
        assert len(run.baseline_dice) > 0
        assert "ece" in run.calibration
        assert "ensemble_vs_baseline" in run.statistical_summary

    def test_rank_parsed_correctly(self, three_rank_fixture):
        dirs = discover_ranks(three_rank_fixture)
        for d in dirs:
            run = load_rank_run(d)
            expected_rank = int(d.name.split("_")[0][1:])
            assert run.rank == expected_rank


class TestBaselineConsistency:
    def test_consistent_baselines(self, three_rank_fixture):
        dirs = discover_ranks(three_rank_fixture)
        runs = [load_rank_run(d) for d in dirs]
        validate_baseline_consistency(runs)

    def test_inconsistent_baselines(self, three_rank_fixture):
        dirs = discover_ranks(three_rank_fixture)
        runs = [load_rank_run(d) for d in dirs]

        bad_run = runs[1]
        modified_baseline = bad_run.baseline_dice.copy()
        modified_baseline["dice_tc"] += 1.0
        bad_run_modified = bad_run.__class__(
            rank=bad_run.rank,
            run_dir=bad_run.run_dir,
            ensemble_dice=bad_run.ensemble_dice,
            per_member_dice=bad_run.per_member_dice,
            baseline_dice=modified_baseline,
            calibration=bad_run.calibration,
            calibration_coverage=bad_run.calibration_coverage,
            bias_diagnostics=bad_run.bias_diagnostics,
            bias_dominance=bad_run.bias_dominance,
            epistemic_taxonomy=bad_run.epistemic_taxonomy,
            statistical_summary=bad_run.statistical_summary,
            paired_differences=bad_run.paired_differences,
            predictions_dir=bad_run.predictions_dir,
        )
        runs[1] = bad_run_modified

        with pytest.raises(BaselineMismatchError):
            validate_baseline_consistency(runs)
