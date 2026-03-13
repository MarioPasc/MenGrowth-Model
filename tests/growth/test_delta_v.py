# tests/growth/test_delta_v.py
"""Tests for delta-V trajectory construction."""

import numpy as np
import pytest

from experiments.segment_based_approach.segment import (
    ScanVolumes,
    SegmentationVolumeExtractor,
)

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


def _make_scan(patient_id: str, tp_idx: int, wt_vol: float) -> ScanVolumes:
    """Create a minimal ScanVolumes for testing."""
    centroid = (0.5, 0.5, 0.5) if wt_vol > 0 else None
    return ScanVolumes(
        scan_id=f"{patient_id}_tp{tp_idx}",
        patient_id=patient_id,
        timepoint_idx=tp_idx,
        manual_wt_vol_mm3=wt_vol,
        manual_tc_vol_mm3=wt_vol * 0.5,
        manual_et_vol_mm3=wt_vol * 0.1,
        is_empty_manual=(wt_vol == 0.0),
        centroid_xyz=centroid,
    )


def _make_cfg(min_tp: int = 2) -> dict:
    """Create a minimal config for SegmentationVolumeExtractor."""
    return {
        "paths": {"mengrowth_h5": "/dev/null", "output_dir": "/tmp/test_deltav"},
        "segmentation": {
            "sw_roi_size": [128, 128, 128],
            "sw_overlap": 0.5,
            "sw_mode": "gaussian",
            "wt_threshold": 0.5,
            "use_manual_segmentation": True,
        },
        "time": {"variable": "ordinal"},
        "patients": {"exclude": [], "min_timepoints": min_tp},
    }


class TestDeltaVTrajectories:
    def test_produces_correct_number_of_deltas(self) -> None:
        """3 timepoints -> 2 delta observations."""
        volumes = [
            _make_scan("P1", 0, 1000.0),
            _make_scan("P1", 1, 1200.0),
            _make_scan("P1", 2, 1500.0),
        ]
        cfg = _make_cfg(min_tp=2)
        extractor = SegmentationVolumeExtractor.__new__(SegmentationVolumeExtractor)
        extractor.cfg = cfg
        extractor.exclude_patients = []
        extractor.h5_path = "/dev/null"

        trajs = extractor.build_delta_trajectories(volumes, "manual")
        assert len(trajs) == 1
        assert trajs[0].n_timepoints == 2

    def test_midpoint_times_correct(self) -> None:
        """Midpoint times should be average of consecutive pairs."""
        volumes = [
            _make_scan("P1", 0, 1000.0),
            _make_scan("P1", 2, 1200.0),
            _make_scan("P1", 6, 1500.0),
        ]
        cfg = _make_cfg()
        extractor = SegmentationVolumeExtractor.__new__(SegmentationVolumeExtractor)
        extractor.cfg = cfg
        extractor.exclude_patients = []
        extractor.h5_path = "/dev/null"

        trajs = extractor.build_delta_trajectories(volumes, "manual")
        expected_times = np.array([1.0, 4.0])  # midpoints of (0,2) and (2,6)
        np.testing.assert_allclose(trajs[0].times, expected_times)

    def test_delta_values_correct(self) -> None:
        """Delta = log1p(V2) - log1p(V1)."""
        v1, v2, v3 = 1000.0, 1200.0, 1500.0
        volumes = [
            _make_scan("P1", 0, v1),
            _make_scan("P1", 1, v2),
            _make_scan("P1", 2, v3),
        ]
        cfg = _make_cfg()
        extractor = SegmentationVolumeExtractor.__new__(SegmentationVolumeExtractor)
        extractor.cfg = cfg
        extractor.exclude_patients = []
        extractor.h5_path = "/dev/null"

        trajs = extractor.build_delta_trajectories(volumes, "manual")
        obs = trajs[0].observations[:, 0]

        expected = np.array([
            np.log1p(v2) - np.log1p(v1),
            np.log1p(v3) - np.log1p(v2),
        ])
        np.testing.assert_allclose(obs, expected)

    def test_requires_min_3_timepoints(self) -> None:
        """Patients with only 2 timepoints should be skipped (need >= 2 deltas)."""
        volumes = [
            _make_scan("P1", 0, 1000.0),
            _make_scan("P1", 1, 1200.0),
        ]
        cfg = _make_cfg(min_tp=2)
        extractor = SegmentationVolumeExtractor.__new__(SegmentationVolumeExtractor)
        extractor.cfg = cfg
        extractor.exclude_patients = []
        extractor.h5_path = "/dev/null"

        trajs = extractor.build_delta_trajectories(volumes, "manual")
        assert len(trajs) == 0  # 2 timepoints -> 1 delta, but min is 3

    def test_covariates_attached(self) -> None:
        """Delta-V trajectories should have centroid covariates."""
        volumes = [
            _make_scan("P1", 0, 1000.0),
            _make_scan("P1", 1, 1200.0),
            _make_scan("P1", 2, 1500.0),
        ]
        cfg = _make_cfg()
        extractor = SegmentationVolumeExtractor.__new__(SegmentationVolumeExtractor)
        extractor.cfg = cfg
        extractor.exclude_patients = []
        extractor.h5_path = "/dev/null"

        trajs = extractor.build_delta_trajectories(volumes, "manual")
        assert trajs[0].covariates is not None
        assert "centroid_x" in trajs[0].covariates
        assert trajs[0].covariates["centroid_x"] == 0.5
