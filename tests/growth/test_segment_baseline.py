# tests/growth/test_segment_baseline.py
"""Integration tests for the segment-based baseline (Ablation A0).

Requires MenGrowth.h5 and the BrainSegFounder checkpoint.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.phase4, pytest.mark.real_data]


@pytest.fixture
def mengrowth_h5_path():
    from pathlib import Path

    path = Path("/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/MenGrowth.h5")
    if not path.exists():
        pytest.skip(f"MenGrowth.h5 not found at {path}")
    return path


@pytest.fixture
def baseline_config(mengrowth_h5_path, real_checkpoint_path, tmp_path):
    """Build a config dict for the baseline experiment (multi-model format)."""
    return {
        "paths": {
            "mengrowth_h5": str(mengrowth_h5_path),
            "output_dir": str(tmp_path / "results"),
        },
        "segmentation": {
            "sw_roi_size": [128, 128, 128],
            "sw_overlap": 0.5,
            "sw_mode": "gaussian",
            "wt_threshold": 0.5,
            "use_manual_segmentation": True,
            "models_to_use": [
                {
                    "model_name": "brainsegfounder",
                    "type": "BrainSegFounder",
                    "checkpoints": str(real_checkpoint_path),
                    "save_to_h5": False,
                    "enabled": True,
                }
            ],
        },
        "volume": {"transform": "log1p"},
        "gp": {
            "kernel": "matern52",
            "mean_function": "linear",
            "n_restarts": 2,
            "max_iter": 200,
            "lengthscale_bounds": [0.1, 50.0],
            "signal_var_bounds": [0.001, 10.0],
            "noise_var_bounds": [0.000001, 5.0],
        },
        "lme": {"method": "reml"},
        "models": {"scalar_gp": True, "lme": True, "hgp": True},
        "time": {"variable": "ordinal"},
        "experiment": {"seed": 42},
        "patients": {"exclude": ["MenGrowth-0028"], "min_timepoints": 2},
    }


def _build_manual_scan_volumes(h5_path: str) -> list:
    """Helper: build ScanVolumes from manual H5 segmentations (no GPU needed)."""
    import h5py

    from experiments.stage1_volumetric.segment import ScanVolumes

    scan_volumes = []
    with h5py.File(h5_path, "r") as f:
        n_scans = f.attrs["n_scans"]
        spacing = tuple(f.attrs.get("spacing", [1.0, 1.0, 1.0]))
        voxel_vol = float(np.prod(spacing))
        scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
        patient_ids = [s.decode() if isinstance(s, bytes) else s for s in f["patient_ids"][:]]
        timepoint_idx = f["timepoint_idx"][:].astype(int)

        for i in range(n_scans):
            seg = f["segs"][i][0]  # [D, H, W]
            wt_vol = float((seg > 0).sum() * voxel_vol)
            tc_vol = float(((seg == 1) | (seg == 3)).sum() * voxel_vol)
            et_vol = float((seg == 3).sum() * voxel_vol)

            scan_volumes.append(
                ScanVolumes(
                    scan_id=scan_ids[i],
                    patient_id=patient_ids[i],
                    timepoint_idx=int(timepoint_idx[i]),
                    manual_wt_vol_mm3=wt_vol,
                    manual_tc_vol_mm3=tc_vol,
                    manual_et_vol_mm3=et_vol,
                    is_empty_manual=(wt_vol == 0.0),
                    model_results={},
                )
            )
    return scan_volumes


class TestH5VolumeExtraction:
    """Test volume extraction from H5 (manual only, no GPU needed)."""

    def test_manual_volumes_from_h5(self, mengrowth_h5_path):
        """Read manual segmentations and compute WT volumes."""
        import h5py

        with h5py.File(str(mengrowth_h5_path), "r") as f:
            n_scans = f.attrs["n_scans"]
            spacing = tuple(f.attrs.get("spacing", [1.0, 1.0, 1.0]))
            voxel_vol = float(np.prod(spacing))

            volumes = []
            for i in range(n_scans):
                seg = f["segs"][i][0]  # [D, H, W]
                wt = (seg > 0).sum()
                volumes.append(float(wt * voxel_vol))

        assert len(volumes) == n_scans
        assert n_scans > 0
        # At least some scans should have non-zero volume
        assert sum(1 for v in volumes if v > 0) > n_scans * 0.8

    def test_longitudinal_structure(self, mengrowth_h5_path):
        """Verify H5 has longitudinal metadata for trajectory building."""
        import h5py

        with h5py.File(str(mengrowth_h5_path), "r") as f:
            assert "scan_ids" in f
            assert "patient_ids" in f
            assert "timepoint_idx" in f
            assert "longitudinal/patient_offsets" in f
            assert "longitudinal/patient_list" in f

            n_patients = f.attrs["n_patients"]
            n_scans = f.attrs["n_scans"]
            assert n_patients > 0
            assert n_scans > n_patients  # longitudinal -> more scans than patients


class TestTrajectoryBuilding:
    """Test trajectory construction from manual volumes."""

    def test_build_manual_trajectories(self, baseline_config):
        from experiments.stage1_volumetric.segment import SegmentationVolumeExtractor

        scan_volumes = _build_manual_scan_volumes(baseline_config["paths"]["mengrowth_h5"])
        extractor = SegmentationVolumeExtractor(baseline_config)
        trajectories = extractor.build_trajectories(scan_volumes, "manual")

        # Should have patients (minus excluded)
        assert len(trajectories) > 0
        # No excluded patients
        for t in trajectories:
            assert t.patient_id not in baseline_config["patients"]["exclude"]
        # All should have >=2 timepoints
        for t in trajectories:
            assert t.n_timepoints >= 2
        # Observations should be log1p transformed (so > 0 for non-empty)
        for t in trajectories:
            assert t.observations.shape[1] == 1
            assert np.all(np.isfinite(t.observations))


class TestSegmentationComparison:
    """Test segmentation comparison report generation."""

    def test_generate_report(self, baseline_config):
        from experiments.stage1_volumetric.segment import generate_segmentation_report

        scan_volumes = _build_manual_scan_volumes(baseline_config["paths"]["mengrowth_h5"])
        report = generate_segmentation_report(scan_volumes)

        assert "n_total_scans" in report
        assert "per_scan" in report
        # With no model_results, per_model should be empty
        assert "per_model" in report


class TestMultiModelLOPO:
    """End-to-end multi-model LOPO test on manual volumes."""

    def test_lopo_scalar_gp(self, baseline_config):
        from experiments.stage1_volumetric.segment import SegmentationVolumeExtractor
        from growth.evaluation.lopo_evaluator import LOPOEvaluator
        from growth.models.growth.scalar_gp import ScalarGP

        scan_volumes = _build_manual_scan_volumes(baseline_config["paths"]["mengrowth_h5"])
        extractor = SegmentationVolumeExtractor(baseline_config)
        trajectories = extractor.build_trajectories(scan_volumes, "manual")

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(
            ScalarGP,
            trajectories,
            kernel_type="matern52",
            mean_function="linear",
            n_restarts=2,
            max_iter=200,
            seed=42,
        )

        assert len(results.fold_results) > 0
        assert "last_from_rest/r2_log" in results.aggregate_metrics
        assert "last_from_rest/calibration_95" in results.aggregate_metrics
        assert np.isfinite(results.aggregate_metrics["last_from_rest/r2_log"])

    def test_lopo_lme(self, baseline_config):
        from experiments.stage1_volumetric.segment import SegmentationVolumeExtractor
        from growth.evaluation.lopo_evaluator import LOPOEvaluator
        from growth.models.growth.lme_model import LMEGrowthModel

        scan_volumes = _build_manual_scan_volumes(baseline_config["paths"]["mengrowth_h5"])
        extractor = SegmentationVolumeExtractor(baseline_config)
        trajectories = extractor.build_trajectories(scan_volumes, "manual")

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(LMEGrowthModel, trajectories, method="reml")

        assert len(results.fold_results) > 0
        assert "last_from_rest/r2_log" in results.aggregate_metrics
        assert "last_from_rest/calibration_95" in results.aggregate_metrics
        assert np.isfinite(results.aggregate_metrics["last_from_rest/r2_log"])

    def test_lopo_hgp(self, baseline_config):
        from experiments.stage1_volumetric.segment import SegmentationVolumeExtractor
        from growth.evaluation.lopo_evaluator import LOPOEvaluator
        from growth.models.growth.hgp_model import HierarchicalGPModel

        scan_volumes = _build_manual_scan_volumes(baseline_config["paths"]["mengrowth_h5"])
        extractor = SegmentationVolumeExtractor(baseline_config)
        trajectories = extractor.build_trajectories(scan_volumes, "manual")

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(
            HierarchicalGPModel,
            trajectories,
            kernel_type="matern52",
            n_restarts=2,
            max_iter=200,
            seed=42,
        )

        assert len(results.fold_results) > 0
        assert "last_from_rest/r2_log" in results.aggregate_metrics
        assert "last_from_rest/calibration_95" in results.aggregate_metrics
        assert np.isfinite(results.aggregate_metrics["last_from_rest/r2_log"])
