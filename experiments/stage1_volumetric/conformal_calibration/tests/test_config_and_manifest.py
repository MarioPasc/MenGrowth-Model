"""Unit tests for conformal calibration config loading and manifest generation.

Tests are synthetic-only (no real H5 data) and cover:
- Config YAML loading via engine.data.load_config
- iter_task_specs produces correct (base_model, seed) combinations
- build_manifest / write_manifest / read_manifest round-trip
- Manifest task count matches n_seeds × n_active_models
- TaskSpec properties (model_dirname, seed_dirname)
- Aggregator collect_runs handles empty runs/ directory
- Statistics bh_fdr correctness on known p-values
- figures.make_all_figures does not crash on empty DataFrame
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).parent.parent
_SMOKE_CONFIG = _BASE_DIR / "configs" / "local_smoke.yaml"


def _minimal_cfg(
    *,
    n_seeds: int = 2,
    lme_homo: bool = True,
    lme_hetero: bool = True,
    ensemble_bma: bool = False,
    output_dir: str = "/tmp/confcal_test",
) -> dict:
    return {
        "paths": {
            "mengrowth_h5": "/tmp/fake.h5",
            "output_dir": output_dir,
        },
        "time": {"variable": "ordinal"},
        "uncertainty": {
            "signal": "logvol_var",
            "mean_signal": "logvol_mean",
            "floor_variance": 1e-6,
        },
        "patients": {"min_timepoints": 2, "skip_all_zero_volume": True},
        "ensemble": {"n_members": 5},
        "models": {
            "lme_homo": lme_homo,
            "lme_hetero": lme_hetero,
            "ensemble_bma": ensemble_bma,
        },
        "conformal": {
            "alpha": 0.05,
            "layers": ["parametric", "jackknife_plus"],
        },
        "evaluation": {
            "n_seeds": n_seeds,
            "n_restarts": 3,
        },
        "statistics": {
            "bootstrap": {"n_samples": 200, "confidence_level": 0.95, "seed": 42},
            "bh_fdr_q": 0.05,
            "comparison_families": {"calibration_lift": True},
        },
        "reporting": {
            "figures": [],
        },
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_load_smoke_config(self) -> None:
        """local_smoke.yaml must load without error and contain required keys."""
        from experiments.stage1_volumetric.engine.data import load_config

        cfg = load_config(_SMOKE_CONFIG)
        assert "paths" in cfg
        assert "conformal" in cfg
        assert "evaluation" in cfg
        assert "models" in cfg

    def test_smoke_config_n_seeds(self) -> None:
        from experiments.stage1_volumetric.engine.data import load_config

        cfg = load_config(_SMOKE_CONFIG)
        assert cfg["evaluation"]["n_seeds"] == 2

    def test_smoke_config_layers(self) -> None:
        from experiments.stage1_volumetric.engine.data import load_config

        cfg = load_config(_SMOKE_CONFIG)
        layers = cfg["conformal"]["layers"]
        assert "parametric" in layers
        assert "jackknife_plus" in layers

    @pytest.mark.parametrize("key", ["mengrowth_h5", "output_dir"])
    def test_smoke_config_paths_present(self, key: str) -> None:
        from experiments.stage1_volumetric.engine.data import load_config

        cfg = load_config(_SMOKE_CONFIG)
        assert key in cfg["paths"], f"Missing path key: {key}"


# ---------------------------------------------------------------------------
# iter_task_specs
# ---------------------------------------------------------------------------


class TestIterTaskSpecs:
    def test_all_models_two_seeds(self) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.runner import (
            iter_task_specs,
        )

        cfg = _minimal_cfg(n_seeds=2, lme_homo=True, lme_hetero=True, ensemble_bma=True)
        specs = iter_task_specs(cfg)
        assert len(specs) == 6  # 3 models × 2 seeds

    def test_only_homo_three_seeds(self) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.runner import (
            iter_task_specs,
        )

        cfg = _minimal_cfg(n_seeds=3, lme_homo=True, lme_hetero=False, ensemble_bma=False)
        specs = iter_task_specs(cfg)
        assert len(specs) == 3
        assert all(s.base_model == "lme_homo" for s in specs)

    def test_seeds_are_zero_indexed(self) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.runner import (
            iter_task_specs,
        )

        cfg = _minimal_cfg(n_seeds=5, lme_homo=True, lme_hetero=False, ensemble_bma=False)
        specs = iter_task_specs(cfg)
        seeds = [s.seed for s in specs]
        assert seeds == list(range(5))

    def test_no_models_returns_empty(self) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.runner import (
            iter_task_specs,
        )

        cfg = _minimal_cfg(n_seeds=2, lme_homo=False, lme_hetero=False, ensemble_bma=False)
        specs = iter_task_specs(cfg)
        assert specs == []

    @pytest.mark.parametrize("base_model", ["lme_homo", "lme_hetero", "ensemble_bma"])
    def test_model_dirname_matches_key(self, base_model: str) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.runner import TaskSpec

        spec = TaskSpec(base_model=base_model, seed=0)
        assert spec.model_dirname == base_model

    @pytest.mark.parametrize("seed,expected", [(0, "seed_000"), (7, "seed_007"), (42, "seed_042")])
    def test_seed_dirname_zero_padded(self, seed: int, expected: str) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.runner import TaskSpec

        spec = TaskSpec(base_model="lme_homo", seed=seed)
        assert spec.seed_dirname == expected


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------


class TestManifestRoundTrip:
    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        from experiments.stage1_volumetric.conformal_calibration.run import (
            build_manifest,
            read_manifest,
            write_manifest,
        )

        cfg = _minimal_cfg(n_seeds=2, lme_homo=True, lme_hetero=True, ensemble_bma=False)
        tasks = build_manifest(cfg)
        mpath = tmp_path / "manifest.json"
        write_manifest(tasks, mpath)

        loaded = read_manifest(mpath)
        assert len(loaded) == len(tasks)
        for orig, reloaded in zip(tasks, loaded, strict=True):
            assert orig.kind == reloaded.kind
            assert orig.spec == reloaded.spec

    def test_manifest_json_is_valid(self, tmp_path: Path) -> None:
        from experiments.stage1_volumetric.conformal_calibration.run import (
            build_manifest,
            write_manifest,
        )

        cfg = _minimal_cfg(n_seeds=1, lme_homo=True, lme_hetero=False, ensemble_bma=False)
        tasks = build_manifest(cfg)
        mpath = tmp_path / "manifest.json"
        write_manifest(tasks, mpath)

        with open(mpath) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert all("kind" in item for item in data)
        assert all("spec" in item for item in data)

    def test_manifest_task_count(self, tmp_path: Path) -> None:
        """Task count must equal n_active_models × n_seeds."""
        from experiments.stage1_volumetric.conformal_calibration.run import build_manifest

        cfg = _minimal_cfg(n_seeds=4, lme_homo=True, lme_hetero=True, ensemble_bma=True)
        tasks = build_manifest(cfg)
        assert len(tasks) == 12  # 3 × 4

    def test_manifest_spec_fields(self, tmp_path: Path) -> None:
        from experiments.stage1_volumetric.conformal_calibration.run import build_manifest

        cfg = _minimal_cfg(n_seeds=1, lme_homo=True, lme_hetero=False, ensemble_bma=False)
        tasks = build_manifest(cfg)
        assert len(tasks) == 1
        spec = tasks[0].spec
        assert "base_model" in spec
        assert "seed" in spec


# ---------------------------------------------------------------------------
# Aggregator: empty runs directory
# ---------------------------------------------------------------------------


class TestAggregatorEmpty:
    def test_empty_runs_returns_empty_df(self, tmp_path: Path) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.aggregator import (
            collect_runs,
        )

        df = collect_runs(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_missing_runs_dir_returns_empty_df(self, tmp_path: Path) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.aggregator import (
            collect_runs,
        )

        df = collect_runs(tmp_path / "nonexistent")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_collect_runs_with_synthetic_data(self, tmp_path: Path) -> None:
        """Aggregator must parse marginal_metrics.json correctly."""
        from experiments.stage1_volumetric.conformal_calibration.modules.aggregator import (
            collect_runs,
        )

        # Build synthetic run directory structure.
        run_dir = tmp_path / "runs" / "lme_homo" / "seed_000"
        run_dir.mkdir(parents=True)
        marginal = {
            "parametric": {
                "n": 10,
                "r2_log": 0.35,
                "is_95": 1.23,
                "coverage_95": 0.90,
                "mean_width": 0.55,
                "crps": 0.10,
            }
        }
        with open(run_dir / "marginal_metrics.json", "w") as f:
            json.dump(marginal, f)

        df = collect_runs(tmp_path)
        assert len(df) > 0
        assert "base_model" in df.columns
        assert "layer" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns
        assert set(df["base_model"].unique()) == {"lme_homo"}
        assert "parametric" in df["layer"].values


# ---------------------------------------------------------------------------
# Statistics: bh_fdr
# ---------------------------------------------------------------------------


class TestBhFdr:
    def test_all_significant(self) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.statistics import bh_fdr

        p = np.array([0.001, 0.002, 0.003])
        rej, p_adj = bh_fdr(p, q=0.05)
        assert rej.all()
        assert len(p_adj) == 3

    def test_none_significant(self) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.statistics import bh_fdr

        p = np.array([0.5, 0.6, 0.7, 0.8])
        rej, p_adj = bh_fdr(p, q=0.05)
        assert not rej.any()

    def test_nan_treated_as_one(self) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.statistics import bh_fdr

        p = np.array([0.001, float("nan"), 0.002])
        rej, p_adj = bh_fdr(p, q=0.05)
        # NaN index should not be rejected
        assert not rej[1]

    def test_output_shapes_match_input(self) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.statistics import bh_fdr

        p = np.random.default_rng(0).uniform(0, 1, 20)
        rej, p_adj = bh_fdr(p, q=0.05)
        assert rej.shape == p.shape
        assert p_adj.shape == p.shape

    def test_adjusted_p_monotone(self) -> None:
        """BH-adjusted p-values must be monotone non-decreasing when sorted by raw p."""
        from experiments.stage1_volumetric.conformal_calibration.modules.statistics import bh_fdr

        rng = np.random.default_rng(42)
        p = rng.uniform(0, 1, 30)
        _, p_adj = bh_fdr(p, q=0.05)
        order = np.argsort(p)
        padj_sorted = p_adj[order]
        assert np.all(np.diff(padj_sorted) >= -1e-12), "BH-adjusted p must be non-decreasing"


# ---------------------------------------------------------------------------
# Figures: make_all_figures on empty DataFrame must not raise
# ---------------------------------------------------------------------------


class TestFiguresEmptyDf:
    def test_make_all_figures_empty(self, tmp_path: Path) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.figures import (
            make_all_figures,
        )

        cfg = _minimal_cfg(output_dir=str(tmp_path))
        cfg["reporting"]["figures"] = [
            "is_by_model_calibration",
            "coverage_by_model_calibration",
            "tertile_panel",
            "width_vs_sigmav",
        ]
        df = pd.DataFrame(
            columns=["base_model", "layer", "seed", "scope", "tertile", "metric", "value"]
        )
        # Must not raise.
        make_all_figures(df, tmp_path, cfg)

    def test_make_all_figures_unknown_name_does_not_raise(self, tmp_path: Path) -> None:
        from experiments.stage1_volumetric.conformal_calibration.modules.figures import (
            make_all_figures,
        )

        cfg = _minimal_cfg(output_dir=str(tmp_path))
        cfg["reporting"]["figures"] = ["nonexistent_figure_name"]
        df = pd.DataFrame()
        make_all_figures(df, tmp_path, cfg)
