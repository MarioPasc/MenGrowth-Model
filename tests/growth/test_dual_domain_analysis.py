"""Tests for the dual-domain LoRA analysis script.

Validates metric collection, figure generation, summary export,
and HTML report generation using synthetic metrics data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

pytestmark = [pytest.mark.experiment, pytest.mark.phase1, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_results_dir(tmp_path: Path) -> Path:
    """Create a synthetic results directory mimicking the dual-domain layout."""
    conditions = ["baseline", "men_r8", "dual_r8"]

    for cond in conditions:
        cond_dir = tmp_path / "conditions" / cond
        (cond_dir / "dice").mkdir(parents=True)
        (cond_dir / "probes").mkdir(parents=True)
        (cond_dir / "domain_gap").mkdir(parents=True)
        (cond_dir / "features").mkdir(parents=True)

        # Dice summary
        dice = {
            "men": {
                "dice_mean": 0.5 + np.random.rand() * 0.2,
                "dice_TC": 0.4 + np.random.rand() * 0.2,
                "dice_WT": 0.7 + np.random.rand() * 0.1,
                "dice_ET": 0.3 + np.random.rand() * 0.3,
                "dice_std": 0.2,
                "num_samples": 150,
            },
            "gli": {
                "dice_mean": 0.3 + np.random.rand() * 0.3,
                "dice_TC": 0.2 + np.random.rand() * 0.3,
                "dice_WT": 0.5 + np.random.rand() * 0.3,
                "dice_ET": 0.1 + np.random.rand() * 0.4,
                "dice_std": 0.15,
                "num_samples": 259,
            },
        }
        with open(cond_dir / "dice" / "dice_summary.json", "w") as f:
            json.dump(dice, f)

        # Probe results (per domain)
        for domain in ("men", "gli"):
            probes = {
                "r2_volume_linear": np.random.rand() * 0.6,
                "r2_volume_rbf": np.random.rand() * 0.3,
                "r2_location_linear": np.random.rand() * 0.4,
                "r2_location_rbf": np.random.rand() * 0.2,
                "r2_shape_linear": np.random.rand() * 0.2 - 0.1,
                "r2_shape_rbf": np.random.rand() * 0.2 - 0.1,
                "r2_mean_linear": np.random.rand() * 0.3,
                "r2_mean_rbf": np.random.rand() * 0.15,
                "effective_rank": 20 + np.random.rand() * 30,
                "n_dead_dims": 0,
                "n_low_variance_dims": 700,
                "variance_mean": 0.005 + np.random.rand() * 0.06,
                "variance_min": 1e-4,
            }
            with open(cond_dir / "probes" / f"{domain}_probes.json", "w") as f:
                json.dump(probes, f)

        # Cross-domain probes
        xdom = {
            "gli_to_men": {
                "r2_volume_linear": np.random.rand() * 0.3 - 0.5,
                "r2_location_linear": np.random.rand() * 0.3,
                "r2_shape_linear": np.random.rand() * 0.2 - 1.0,
            },
            "men_to_gli": {
                "r2_volume_linear": np.random.rand() * 0.3 - 0.5,
                "r2_location_linear": np.random.rand() * 0.3,
                "r2_shape_linear": np.random.rand() * 0.2 - 1.0,
            },
        }
        with open(cond_dir / "probes" / "cross_domain_probes.json", "w") as f:
            json.dump(xdom, f)

        # Domain gap metrics
        dgap = {
            "mmd_squared": np.random.rand() * 0.12,
            "mmd_pvalue": 0.005,
            "domain_classifier_accuracy": 0.7 + np.random.rand() * 0.15,
            "proxy_a_distance": 1.0,
            "men_effective_rank": 15 + np.random.rand() * 20,
            "gli_effective_rank": 20 + np.random.rand() * 20,
            "combined_effective_rank": 25 + np.random.rand() * 25,
            "men_n_dead_dims": 0,
            "gli_n_dead_dims": 0,
            "men_variance_mean": 0.005 + np.random.rand() * 0.06,
            "gli_variance_mean": 0.005 + np.random.rand() * 0.06,
            "cka_men_vs_gli": np.random.rand() * 0.02,
        }
        with open(cond_dir / "domain_gap" / "domain_gap_metrics.json", "w") as f:
            json.dump(dgap, f)

        # Training log CSV
        n_epochs = 10
        df = pd.DataFrame({
            "epoch": range(1, n_epochs + 1),
            "train_loss": np.random.rand(n_epochs) * 0.5 + 0.3,
            "val_men_dice_mean": np.cumsum(np.random.rand(n_epochs) * 0.05),
            "val_gli_dice_mean": np.cumsum(np.random.rand(n_epochs) * 0.03),
            "val_combined_dice_mean": np.cumsum(np.random.rand(n_epochs) * 0.04),
        })
        df.to_csv(cond_dir / "training_log.csv", index=False)

        # Synthetic features (small: 50 samples, 768 dims)
        for domain in ("men", "gli"):
            feat = torch.randn(50, 768)
            torch.save(feat, cond_dir / "features" / f"features_{domain}_test_encoder10.pt")
            targets = {
                "volume": torch.randn(50, 4),
                "location": torch.randn(50, 3),
                "shape": torch.randn(50, 1),
            }
            torch.save(targets, cond_dir / "features" / f"targets_{domain}_test.pt")

    return tmp_path


@pytest.fixture
def synthetic_config(synthetic_results_dir: Path, tmp_path: Path) -> dict:
    """Create a synthetic config dict pointing to the synthetic results."""
    config = {
        "experiment": {
            "name": "test_dual_domain",
            "seed": 42,
            "output_dir": str(synthetic_results_dir),
        },
        "conditions": [
            {"name": "baseline"},
            {"name": "dual_r8"},
            {"name": "men_r8"},
        ],
    }
    return config


@pytest.fixture
def synthetic_config_path(synthetic_config: dict, tmp_path: Path) -> Path:
    """Write synthetic config to YAML file."""
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(synthetic_config, f)
    return path


# ---------------------------------------------------------------------------
# Tests: Metric Collection
# ---------------------------------------------------------------------------


class TestMetricCollection:
    """Tests for collect_all_metrics."""

    def test_collects_all_conditions(self, synthetic_config: dict) -> None:
        from experiments.lora.analysis.dual_domain_analysis import collect_all_metrics

        metrics = collect_all_metrics(synthetic_config)
        assert set(metrics.keys()) == {"baseline", "men_r8", "dual_r8"}

    def test_dice_metrics_present(self, synthetic_config: dict) -> None:
        from experiments.lora.analysis.dual_domain_analysis import collect_all_metrics

        metrics = collect_all_metrics(synthetic_config)
        for cond in ("baseline", "men_r8", "dual_r8"):
            m = metrics[cond]
            assert "dice_men_dice_mean" in m
            assert "dice_gli_dice_mean" in m
            assert 0 <= m["dice_men_dice_mean"] <= 1
            assert 0 <= m["dice_gli_dice_mean"] <= 1

    def test_probe_metrics_present(self, synthetic_config: dict) -> None:
        from experiments.lora.analysis.dual_domain_analysis import collect_all_metrics

        metrics = collect_all_metrics(synthetic_config)
        for cond in ("baseline", "men_r8", "dual_r8"):
            m = metrics[cond]
            assert "probe_men_r2_volume_linear" in m
            assert "probe_men_r2_location_linear" in m
            assert "probe_men_r2_shape_linear" in m

    def test_domain_gap_metrics_present(self, synthetic_config: dict) -> None:
        from experiments.lora.analysis.dual_domain_analysis import collect_all_metrics

        metrics = collect_all_metrics(synthetic_config)
        for cond in ("baseline", "men_r8", "dual_r8"):
            m = metrics[cond]
            assert "dgap_mmd_squared" in m
            assert m["dgap_mmd_squared"] >= 0
            assert "dgap_combined_effective_rank" in m

    def test_training_log_metrics(self, synthetic_config: dict) -> None:
        from experiments.lora.analysis.dual_domain_analysis import collect_all_metrics

        metrics = collect_all_metrics(synthetic_config)
        for cond in ("baseline", "men_r8", "dual_r8"):
            m = metrics[cond]
            assert "n_epochs" in m
            assert m["n_epochs"] == 10

    def test_handles_missing_files(self, tmp_path: Path) -> None:
        """Metric collection should handle missing files gracefully."""
        from experiments.lora.analysis.dual_domain_analysis import collect_all_metrics

        config = {
            "experiment": {"output_dir": str(tmp_path)},
        }
        # Empty dir — no conditions folders
        (tmp_path / "conditions" / "baseline").mkdir(parents=True)
        metrics = collect_all_metrics(config)
        assert "baseline" in metrics
        # Should have at least the condition key
        assert metrics["baseline"]["condition"] == "baseline"


# ---------------------------------------------------------------------------
# Tests: Summary Generation
# ---------------------------------------------------------------------------


class TestSummaryGeneration:
    """Tests for summary table and JSON export."""

    def test_summary_csv_generated(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import (
            collect_all_metrics,
            generate_summary_table,
        )

        metrics = collect_all_metrics(synthetic_config)
        df = generate_summary_table(metrics, tmp_path)
        csv_path = tmp_path / "summary.csv"

        assert csv_path.exists()
        assert len(df) > 0
        assert "Metric" in df.columns
        assert "Winner" in df.columns

    def test_summary_json_generated(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import (
            collect_all_metrics,
            generate_summary_json,
        )

        metrics = collect_all_metrics(synthetic_config)
        summary = generate_summary_json(metrics, tmp_path)
        json_path = tmp_path / "summary.json"

        assert json_path.exists()
        assert "conditions" in summary
        assert "metrics" in summary
        assert set(summary["conditions"]) == {"baseline", "men_r8", "dual_r8"}

    def test_summary_json_is_valid(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import (
            collect_all_metrics,
            generate_summary_json,
        )

        metrics = collect_all_metrics(synthetic_config)
        generate_summary_json(metrics, tmp_path)

        with open(tmp_path / "summary.json") as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)
        assert "generated_at" in loaded


# ---------------------------------------------------------------------------
# Tests: Figure Generation
# ---------------------------------------------------------------------------


class TestFigureGeneration:
    """Tests for individual figure generation functions."""

    def test_f1_segmentation(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import (
            collect_all_metrics,
            fig_f1_segmentation,
        )
        from experiments.utils.settings import apply_ieee_style

        apply_ieee_style()
        metrics = collect_all_metrics(synthetic_config)
        fig_f1_segmentation(synthetic_config, metrics, tmp_path)

        assert (tmp_path / "F1_segmentation.png").exists()
        assert (tmp_path / "F1_segmentation.pdf").exists()
        assert (tmp_path / "F1_segmentation.png").stat().st_size > 0

    def test_f2_probes(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import (
            collect_all_metrics,
            fig_f2_probes,
        )
        from experiments.utils.settings import apply_ieee_style

        apply_ieee_style()
        metrics = collect_all_metrics(synthetic_config)
        fig_f2_probes(synthetic_config, metrics, tmp_path)

        assert (tmp_path / "F2_probes.png").exists()
        assert (tmp_path / "F2_probes.pdf").exists()

    def test_f3_domain_gap(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import (
            collect_all_metrics,
            fig_f3_domain_gap,
        )
        from experiments.utils.settings import apply_ieee_style

        apply_ieee_style()
        metrics = collect_all_metrics(synthetic_config)
        fig_f3_domain_gap(synthetic_config, metrics, tmp_path)

        assert (tmp_path / "F3_domain_gap.png").exists()

    def test_f4_variance_spectrum(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import fig_f4_variance_spectrum
        from experiments.utils.settings import apply_ieee_style

        apply_ieee_style()
        fig_f4_variance_spectrum(synthetic_config, tmp_path)

        assert (tmp_path / "F4_variance_spectrum.png").exists()

    def test_f6_training_dynamics(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import fig_f6_training_dynamics
        from experiments.utils.settings import apply_ieee_style

        apply_ieee_style()
        fig_f6_training_dynamics(synthetic_config, tmp_path)

        assert (tmp_path / "F6_training_dynamics.png").exists()

    def test_f10_radar(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import (
            collect_all_metrics,
            fig_f10_radar,
        )
        from experiments.utils.settings import apply_ieee_style

        apply_ieee_style()
        metrics = collect_all_metrics(synthetic_config)
        fig_f10_radar(synthetic_config, metrics, tmp_path)

        assert (tmp_path / "F10_radar_summary.png").exists()


# ---------------------------------------------------------------------------
# Tests: HTML Report
# ---------------------------------------------------------------------------


class TestHTMLReport:
    """Tests for the self-contained HTML report."""

    def test_html_report_generated(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import (
            collect_all_metrics,
            generate_html_report,
            generate_summary_table,
        )

        metrics = collect_all_metrics(synthetic_config)
        summary_df = generate_summary_table(metrics, tmp_path)
        # No figures dir — should still produce report
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        generate_html_report(metrics, summary_df, figures_dir, tmp_path)

        report_path = tmp_path / "report.html"
        assert report_path.exists()
        assert report_path.stat().st_size > 0

    def test_html_contains_key_sections(self, synthetic_config: dict, tmp_path: Path) -> None:
        from experiments.lora.analysis.dual_domain_analysis import (
            collect_all_metrics,
            generate_html_report,
            generate_summary_table,
        )

        metrics = collect_all_metrics(synthetic_config)
        summary_df = generate_summary_table(metrics, tmp_path)
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        generate_html_report(metrics, summary_df, figures_dir, tmp_path)

        html = (tmp_path / "report.html").read_text()
        assert "Dual-Domain LoRA" in html
        assert "SDP Readiness" in html
        assert "Conclusions" in html
        assert "Baseline" in html
        assert "Dual LoRA r=8" in html


# ---------------------------------------------------------------------------
# Tests: Settings Integration
# ---------------------------------------------------------------------------


class TestSettingsIntegration:
    """Tests that settings.py has the dual-domain condition entries."""

    def test_condition_colors_present(self) -> None:
        from experiments.utils.settings import CONDITION_COLORS

        assert "dual_r8" in CONDITION_COLORS
        assert "men_r8" in CONDITION_COLORS

    def test_condition_labels_present(self) -> None:
        from experiments.utils.settings import CONDITION_LABELS

        assert "dual_r8" in CONDITION_LABELS
        assert "men_r8" in CONDITION_LABELS

    def test_condition_order_dual(self) -> None:
        from experiments.utils.settings import CONDITION_ORDER_DUAL

        assert CONDITION_ORDER_DUAL == ["baseline", "men_r8", "dual_r8"]

    def test_probe_colors_has_rbf(self) -> None:
        from experiments.utils.settings import PROBE_COLORS

        assert "rbf" in PROBE_COLORS
        assert "linear" in PROBE_COLORS
