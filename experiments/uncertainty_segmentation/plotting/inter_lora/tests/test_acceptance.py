"""End-to-end acceptance test for inter-LoRA report generation (spec §9.5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


class TestEndToEnd:
    """Smoke test: run CLI on 3-rank synthetic fixture and verify outputs."""

    @pytest.fixture(autouse=True)
    def _generate_report(self, three_rank_fixture: Path) -> None:
        from experiments.uncertainty_segmentation.plotting.inter_lora.orchestrate import (
            generate_inter_lora_report,
        )

        self.root = three_rank_fixture
        generate_inter_lora_report(
            root_dir=self.root,
            seed=42,
            bootstrap_n=50,
            dpi=72,
            skip={"qual1"},
        )
        self.out = self.root / "_inter_lora_report"

    def test_output_dir_exists(self):
        assert self.out.is_dir()

    def test_compiled_metrics_exists(self):
        assert (self.out / "data" / "compiled_metrics.csv").exists()

    def test_compiled_metrics_schema(self):
        df = pd.read_csv(self.out / "data" / "compiled_metrics.csv")
        assert "dice_mean" in df.columns
        assert "rank" in df.columns
        assert "label" in df.columns
        non_bl = df[(df["rank"] > 0) & (df["label"] != "mean")]
        assert not non_bl["dice_mean"].isna().any()

    def test_compiled_metrics_consistency(self):
        """Verify compiled_metrics matches statistical_summary (spec §9.5 step 3)."""
        import json

        df = pd.read_csv(self.out / "data" / "compiled_metrics.csv")
        r8_dir = self.root / "r8_M5_s42" / "evaluation"
        with open(r8_dir / "statistical_summary.json") as f:
            ss = json.load(f)

        wt_compiled = df.query("rank == 8 and label == 'WT'")["dice_mean"].item()
        wt_source = ss["ensemble_vs_baseline"]["wt"]["ensemble_mean"]
        assert abs(wt_compiled - wt_source) < 1e-6

    def test_figures_exist(self):
        figs_dir = self.out / "figures"
        for name in [
            "quant1_dice_vs_rank",
            "quant2_calibration_epistemic_vs_rank",
            "qual2_clustered_heatmap",
        ]:
            assert (figs_dir / f"{name}.pdf").exists(), f"Missing {name}.pdf"
            assert (figs_dir / f"{name}.png").exists(), f"Missing {name}.png"

    def test_tables_exist(self):
        tab_dir = self.out / "tables"
        for name in ["tab1_summary_per_rank", "tab2_paired_vs_baseline"]:
            for ext in ["csv", "md", "tex"]:
                assert (tab_dir / f"{name}.{ext}").exists(), f"Missing {name}.{ext}"

    def test_tab1_csv_has_header(self):
        csv_path = self.out / "tables" / "tab1_summary_per_rank.csv"
        with open(csv_path) as f:
            header = f.readline().strip()
        assert "rank" in header.lower() or "Rank" in header

    def test_tab2_tex_uses_booktabs(self):
        tex_path = self.out / "tables" / "tab2_paired_vs_baseline.tex"
        content = tex_path.read_text()
        assert "\\toprule" in content
        assert "\\bottomrule" in content

    def test_selected_slices_persisted(self):
        assert (self.out / "data" / "selected_slices.json").exists()

    def test_report_summary_exists(self):
        assert (self.out / "report_summary.md").exists()

    def test_log_exists(self):
        assert (self.out / "logs" / "plotting.log").exists()
