# tests/growth/test_segment_multi_model.py
"""Unit tests for multi-model segmentation support in segment_based_approach.

These tests use synthetic data only — no GPU, no real checkpoints, no H5 files.
"""

import pytest

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


class TestPerModelResult:
    """Test the PerModelResult dataclass."""

    def test_construction(self):
        from experiments.segment_based_approach.segment import PerModelResult

        pmr = PerModelResult(
            model_name="test_model",
            wt_vol_mm3=1000.0,
            tc_vol_mm3=500.0,
            et_vol_mm3=200.0,
            wt_dice=0.85,
            tc_dice=0.70,
            et_dice=0.60,
            is_empty=False,
        )
        assert pmr.model_name == "test_model"
        assert pmr.wt_vol_mm3 == 1000.0
        assert pmr.wt_dice == 0.85
        assert not pmr.is_empty

    def test_empty_prediction(self):
        from experiments.segment_based_approach.segment import PerModelResult

        pmr = PerModelResult(
            model_name="empty_model",
            wt_vol_mm3=0.0,
            tc_vol_mm3=0.0,
            et_vol_mm3=0.0,
            wt_dice=0.0,
            tc_dice=0.0,
            et_dice=0.0,
            is_empty=True,
        )
        assert pmr.is_empty
        assert pmr.wt_vol_mm3 == 0.0


class TestScanVolumesBackwardCompat:
    """Test backward-compatible property accessors on ScanVolumes."""

    def _make_scan_volumes(self):
        from experiments.segment_based_approach.segment import (
            PerModelResult,
            ScanVolumes,
        )

        pmr = PerModelResult(
            model_name="model_a",
            wt_vol_mm3=1234.0,
            tc_vol_mm3=567.0,
            et_vol_mm3=89.0,
            wt_dice=0.88,
            tc_dice=0.75,
            et_dice=0.55,
            is_empty=False,
        )
        return ScanVolumes(
            scan_id="scan_001",
            patient_id="P001",
            timepoint_idx=0,
            manual_wt_vol_mm3=1200.0,
            manual_tc_vol_mm3=550.0,
            manual_et_vol_mm3=100.0,
            is_empty_manual=False,
            model_results={"model_a": pmr},
        )

    def test_predicted_wt_vol(self):
        sv = self._make_scan_volumes()
        assert sv.predicted_wt_vol_mm3 == 1234.0

    def test_predicted_tc_vol(self):
        sv = self._make_scan_volumes()
        assert sv.predicted_tc_vol_mm3 == 567.0

    def test_predicted_et_vol(self):
        sv = self._make_scan_volumes()
        assert sv.predicted_et_vol_mm3 == 89.0

    def test_wt_dice(self):
        sv = self._make_scan_volumes()
        assert sv.wt_dice == 0.88

    def test_tc_dice(self):
        sv = self._make_scan_volumes()
        assert sv.tc_dice == 0.75

    def test_et_dice(self):
        sv = self._make_scan_volumes()
        assert sv.et_dice == 0.55

    def test_is_empty_predicted(self):
        sv = self._make_scan_volumes()
        assert not sv.is_empty_predicted

    def test_no_models_defaults(self):
        from experiments.segment_based_approach.segment import ScanVolumes

        sv = ScanVolumes(
            scan_id="scan_002",
            patient_id="P002",
            timepoint_idx=0,
            manual_wt_vol_mm3=500.0,
            manual_tc_vol_mm3=200.0,
            manual_et_vol_mm3=50.0,
            is_empty_manual=False,
            model_results={},
        )
        assert sv.predicted_wt_vol_mm3 == 0.0
        assert sv.wt_dice == 0.0
        assert sv.is_empty_predicted is True


class TestParseSegConfig:
    """Test config parsing for both old and new formats."""

    def test_new_format(self):
        from experiments.segment_based_approach.segment import parse_seg_config

        cfg = {
            "segmentation": {
                "use_manual_segmentation": True,
                "models_to_use": [
                    {
                        "model_name": "model_a",
                        "type": "BrainSegFounder",
                        "checkpoints": "/path/to/a.pt",
                        "save_to_h5": True,
                        "enabled": True,
                    },
                    {
                        "model_name": "model_b",
                        "type": "BrainSegFounder",
                        "checkpoints": "/path/to/b.pt",
                        "save_to_h5": False,
                        "enabled": True,
                    },
                    {
                        "model_name": "model_c",
                        "type": "BrainSegFounder",
                        "checkpoints": "/path/to/c.pt",
                        "enabled": False,
                    },
                ],
            }
        }

        models, use_manual = parse_seg_config(cfg)
        assert use_manual is True
        assert len(models) == 2  # model_c is disabled
        assert models[0].model_name == "model_a"
        assert models[0].checkpoint == "/path/to/a.pt"
        assert models[0].save_to_h5 is True
        assert models[1].model_name == "model_b"
        assert models[1].save_to_h5 is False

    def test_old_format_fallback(self):
        from experiments.segment_based_approach.segment import parse_seg_config

        cfg = {
            "paths": {"checkpoint": "/path/to/bsf.pt"},
            "segmentation": {
                "model_name": "brainsegfounder",
                "save_to_h5": True,
            },
        }

        models, use_manual = parse_seg_config(cfg)
        assert use_manual is True
        assert len(models) == 1
        assert models[0].model_name == "brainsegfounder"
        assert models[0].checkpoint == "/path/to/bsf.pt"
        assert models[0].model_type == "BrainSegFounder"

    def test_old_format_no_checkpoint(self):
        from experiments.segment_based_approach.segment import parse_seg_config

        cfg = {"segmentation": {}}

        models, use_manual = parse_seg_config(cfg)
        assert len(models) == 0
        assert use_manual is True

    def test_use_manual_false(self):
        from experiments.segment_based_approach.segment import parse_seg_config

        cfg = {
            "segmentation": {
                "use_manual_segmentation": False,
                "models_to_use": [
                    {
                        "model_name": "model_x",
                        "type": "BrainSegFounder",
                        "checkpoints": "/path/to/x.pt",
                        "enabled": True,
                    }
                ],
            }
        }

        models, use_manual = parse_seg_config(cfg)
        assert use_manual is False
        assert len(models) == 1


class TestStripTrainingCheckpointPrefix:
    """Test the training checkpoint key stripping logic."""

    def test_encoder_keys(self):
        from experiments.segment_based_approach.segment import (
            _strip_training_checkpoint_prefix,
        )

        state_dict = {
            "model.encoder.swinViT.layers1.0.blocks.0.attn.qkv.weight": "w1",
            "model.encoder.encoder10.layer.0.conv1.conv.weight": "w2",
        }
        stripped = _strip_training_checkpoint_prefix(state_dict)
        assert "swinViT.layers1.0.blocks.0.attn.qkv.weight" in stripped
        assert "encoder10.layer.0.conv1.conv.weight" in stripped
        assert len(stripped) == 2

    def test_decoder_keys(self):
        from experiments.segment_based_approach.segment import (
            _strip_training_checkpoint_prefix,
        )

        state_dict = {
            "model.decoder.decoder5.transp_conv.conv.weight": "w1",
            "model.decoder.out.conv.conv.weight": "w2",
        }
        stripped = _strip_training_checkpoint_prefix(state_dict)
        assert "decoder5.transp_conv.conv.weight" in stripped
        assert "out.conv.conv.weight" in stripped

    def test_unrecognized_keys_skipped(self):
        from experiments.segment_based_approach.segment import (
            _strip_training_checkpoint_prefix,
        )

        state_dict = {
            "model.encoder.swinViT.patch_embed.weight": "w1",
            "semantic_heads.vol_head.weight": "w2",
            "optimizer.state.0.exp_avg": "opt",
        }
        stripped = _strip_training_checkpoint_prefix(state_dict)
        assert len(stripped) == 1
        assert "swinViT.patch_embed.weight" in stripped

    def test_all_encoder_decoder_prefixes(self):
        from experiments.segment_based_approach.segment import (
            _strip_training_checkpoint_prefix,
        )

        state_dict = {
            "model.decoder.encoder1.layer.0.weight": "w1",
            "model.decoder.encoder2.layer.0.weight": "w2",
            "model.decoder.encoder3.layer.0.weight": "w3",
            "model.decoder.encoder4.layer.0.weight": "w4",
            "model.decoder.decoder1.layer.0.weight": "d1",
            "model.decoder.decoder2.layer.0.weight": "d2",
            "model.decoder.decoder3.layer.0.weight": "d3",
            "model.decoder.decoder4.layer.0.weight": "d4",
        }
        stripped = _strip_training_checkpoint_prefix(state_dict)
        assert len(stripped) == 8
        assert "encoder1.layer.0.weight" in stripped
        assert "decoder4.layer.0.weight" in stripped


class TestSegmentationReport:
    """Test multi-model segmentation report generation."""

    def test_multi_model_report(self):
        from experiments.segment_based_approach.segment import (
            PerModelResult,
            ScanVolumes,
            generate_segmentation_report,
        )

        volumes = [
            ScanVolumes(
                scan_id=f"scan_{i}",
                patient_id=f"P{i // 2:03d}",
                timepoint_idx=i % 2,
                manual_wt_vol_mm3=float(100 + i * 10),
                manual_tc_vol_mm3=float(50 + i * 5),
                manual_et_vol_mm3=float(10 + i),
                is_empty_manual=False,
                model_results={
                    "model_a": PerModelResult(
                        model_name="model_a",
                        wt_vol_mm3=float(110 + i * 10),
                        tc_vol_mm3=float(55 + i * 5),
                        et_vol_mm3=float(12 + i),
                        wt_dice=0.8 + i * 0.01,
                        tc_dice=0.7 + i * 0.01,
                        et_dice=0.5 + i * 0.01,
                        is_empty=False,
                    ),
                    "model_b": PerModelResult(
                        model_name="model_b",
                        wt_vol_mm3=float(90 + i * 10),
                        tc_vol_mm3=float(45 + i * 5),
                        et_vol_mm3=float(8 + i),
                        wt_dice=0.75 + i * 0.01,
                        tc_dice=0.65 + i * 0.01,
                        et_dice=0.45 + i * 0.01,
                        is_empty=False,
                    ),
                },
            )
            for i in range(6)
        ]

        report = generate_segmentation_report(volumes)

        assert report["n_total_scans"] == 6
        assert "per_model" in report
        assert "model_a" in report["per_model"]
        assert "model_b" in report["per_model"]

        # Check per-region stats exist for both models
        for mn in ["model_a", "model_b"]:
            for region in ["wt", "tc", "et"]:
                stats = report["per_model"][mn]["per_region"][region]
                assert "dice_mean" in stats
                assert "volume_r2" in stats
                assert 0 <= stats["dice_mean"] <= 1

        # Backward compat: top-level per_region from first model
        assert "per_region" in report
        assert report["per_region"] == report["per_model"]["model_a"]["per_region"]

    def test_empty_model_results(self):
        from experiments.segment_based_approach.segment import (
            ScanVolumes,
            generate_segmentation_report,
        )

        volumes = [
            ScanVolumes(
                scan_id="scan_0",
                patient_id="P000",
                timepoint_idx=0,
                manual_wt_vol_mm3=100.0,
                manual_tc_vol_mm3=50.0,
                manual_et_vol_mm3=10.0,
                is_empty_manual=False,
                model_results={},
            )
        ]

        report = generate_segmentation_report(volumes)
        assert report["n_total_scans"] == 1
        assert report["per_model"] == {}
