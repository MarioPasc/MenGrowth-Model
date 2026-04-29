# tests/growth/test_uq_propagation.py
"""End-to-end tests for uncertainty-propagated volume prediction.

Markers: phase4, unit
"""

from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


def _create_synthetic_uq_h5(path: str, n_patients: int = 8, seed: int = 42) -> None:
    """Create a synthetic MenGrowth H5 with uncertainty group."""
    rng = np.random.default_rng(seed)

    patient_list = [f"TestPatient-{i:04d}" for i in range(1, n_patients + 1)]
    scans_per_patient = rng.integers(2, 5, size=n_patients)
    total_scans = int(scans_per_patient.sum())

    offsets = np.zeros(n_patients + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(scans_per_patient)

    scan_ids, patient_ids = [], []
    timepoint_idx = np.zeros(total_scans, dtype=np.int32)

    for i, pid in enumerate(patient_list):
        for j in range(scans_per_patient[i]):
            scan_idx = offsets[i] + j
            scan_ids.append(f"{pid}-{j:03d}")
            patient_ids.append(pid)
            timepoint_idx[scan_idx] = j

    # Semantic volumes — 5 columns: [total, NCR, ED, ET, MEN]
    semantic_vol = np.zeros((total_scans, 5), dtype=np.float32)
    for i in range(n_patients):
        base = rng.uniform(5.0, 12.0)
        rate = rng.uniform(-0.1, 0.5)
        for j in range(scans_per_patient[i]):
            idx = offsets[i] + j
            semantic_vol[idx, 3] = base + rate * j + rng.normal(0, 0.05)
            semantic_vol[idx, 1] = semantic_vol[idx, 3] * rng.uniform(0.0, 0.03)
            ncr_raw = np.expm1(max(semantic_vol[idx, 1], 0.0))
            et_raw = np.expm1(max(semantic_vol[idx, 3], 0.0))
            semantic_vol[idx, 4] = np.log1p(ncr_raw + et_raw)

    # Metadata
    age = np.full(total_scans, np.nan, dtype=np.float32)
    sex = np.array(["unknown"] * total_scans, dtype="S10")
    grade = np.full(total_scans, -1, dtype=np.int8)

    # Study dates: 30 days apart per patient
    base_date = date(2020, 1, 1)
    study_dates = []
    for i in range(n_patients):
        for j in range(scans_per_patient[i]):
            d = base_date + timedelta(days=int(j) * 30 + i * 365)
            study_dates.append(d.strftime("%Y-%m-%d").encode())
    study_dates_arr = np.array(study_dates, dtype="S12")

    # Uncertainty group
    logvol_mean = semantic_vol[:, 3].copy()
    logvol_std = rng.uniform(0.001, 0.04, total_scans).astype(np.float32)
    logvol_std[0] = 0.0  # One zero for floor test
    logvol_median = logvol_mean + rng.normal(0, 0.001, total_scans).astype(np.float32)
    logvol_mad_scaled = (logvol_std * 1.1).astype(np.float32)
    logvol_ensemble = logvol_mean + rng.normal(0, 0.002, total_scans).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.attrs["version"] = "2.0"
        f.attrs["n_patients"] = n_patients
        f.attrs["n_scans"] = total_scans
        f.attrs["roi_size"] = [192, 192, 192]
        f.attrs["spacing"] = [1.0, 1.0, 1.0]
        f.attrs["channel_order"] = ["t2f", "t1c", "t1n", "t2w"]
        f.attrs["dataset_type"] = "longitudinal"
        f.attrs["domain"] = "MenGrowth"

        f.create_dataset("scan_ids", data=np.array(scan_ids, dtype="S30"))
        f.create_dataset("patient_ids", data=np.array(patient_ids, dtype="S30"))
        f.create_dataset("timepoint_idx", data=timepoint_idx)

        grp = f.create_group("longitudinal")
        grp.create_dataset("patient_list", data=np.array(patient_list, dtype="S30"))
        grp.create_dataset("patient_offsets", data=offsets)

        sem = f.create_group("semantic")
        sem.create_dataset("volume", data=semantic_vol)
        sem.create_dataset("location", data=np.full((total_scans, 3), 0.5, dtype=np.float32))
        sem.create_dataset("shape", data=np.zeros((total_scans, 3), dtype=np.float32))

        meta = f.create_group("metadata")
        meta.create_dataset("age", data=age)
        meta.create_dataset("sex", data=sex)
        meta.create_dataset("grade", data=grade)
        meta.create_dataset("study_date", data=study_dates_arr)

        uq = f.create_group("uncertainty")
        uq.attrs["n_members"] = 20
        uq.attrs["rank"] = 32
        uq.attrs["seed"] = 42
        uq.create_dataset("logvol_mean", data=logvol_mean)
        uq.create_dataset("logvol_std", data=logvol_std)
        uq.create_dataset("logvol_median", data=logvol_median)
        uq.create_dataset("logvol_mad_scaled", data=logvol_mad_scaled)
        uq.create_dataset("logvol_ensemble", data=logvol_ensemble)


@pytest.fixture
def synthetic_uq_h5(tmp_path: Path) -> Path:
    """Create a temporary synthetic H5 with uncertainty data."""
    h5_path = tmp_path / "test_mengrowth_uq.h5"
    _create_synthetic_uq_h5(str(h5_path))
    return h5_path


class TestUQTrajectoryLoader:
    """Tests for load_uncertainty_trajectories_from_h5."""

    def test_loader_returns_observation_variance(self, synthetic_uq_h5: Path) -> None:
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_uncertainty_trajectories_from_h5,
        )

        trajs = load_uncertainty_trajectories_from_h5(str(synthetic_uq_h5))
        assert len(trajs) > 0
        for t in trajs:
            assert t.observation_variance is not None
            assert len(t.observation_variance) == t.n_timepoints
            assert np.all(np.isfinite(t.observation_variance))
            assert np.all(t.observation_variance >= 0)

    def test_loader_floor_variance_applied(self, synthetic_uq_h5: Path) -> None:
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_uncertainty_trajectories_from_h5,
        )

        floor = 1e-4
        trajs = load_uncertainty_trajectories_from_h5(str(synthetic_uq_h5), floor_variance=floor)
        for t in trajs:
            assert np.all(t.observation_variance >= floor)

    def test_real_time_dates_produce_day_deltas(self, synthetic_uq_h5: Path) -> None:
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_uncertainty_trajectories_from_h5,
        )

        trajs = load_uncertainty_trajectories_from_h5(
            str(synthetic_uq_h5), time_variable="days_from_baseline"
        )
        assert len(trajs) > 0
        for t in trajs:
            assert t.times[0] == 0.0  # baseline = 0
            if t.n_timepoints > 1:
                assert t.times[1] > 0  # subsequent > 0

    def test_estimator_median_mad(self, synthetic_uq_h5: Path) -> None:
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_uncertainty_trajectories_from_h5,
        )

        trajs_mean = load_uncertainty_trajectories_from_h5(
            str(synthetic_uq_h5), estimator="mean_std"
        )
        trajs_mad = load_uncertainty_trajectories_from_h5(
            str(synthetic_uq_h5), estimator="median_mad"
        )

        # Different estimators should produce different variances
        v1 = trajs_mean[0].observation_variance
        v2 = trajs_mad[0].observation_variance
        assert not np.allclose(v1, v2)


class TestUQEndToEnd:
    """End-to-end smoke tests."""

    def test_lme_hetero_lopo_on_synthetic(self, synthetic_uq_h5: Path) -> None:
        """LMEHetero runs through LOPO-CV on synthetic UQ data."""
        from growth.models.growth.lme_hetero import LMEHeteroGrowthModel
        from growth.shared.lopo import LOPOEvaluator
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_uncertainty_trajectories_from_h5,
        )

        trajs = load_uncertainty_trajectories_from_h5(str(synthetic_uq_h5))
        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(LMEHeteroGrowthModel, trajs, n_restarts=1, seed=42)
        assert len(results.fold_results) > 0
        assert "last_from_rest/r2_log" in results.aggregate_metrics
        # UQ metrics should be computed
        assert "last_from_rest/crps" in results.aggregate_metrics
        assert "last_from_rest/coverage_95" in results.aggregate_metrics

    def test_paired_permutation_produces_finite(self, synthetic_uq_h5: Path) -> None:
        """Paired permutation test produces finite p-values."""
        from growth.shared.bootstrap import paired_permutation_test

        rng = np.random.default_rng(42)
        errors_a = rng.normal(0, 1, 20)
        errors_b = rng.normal(0, 1.1, 20)

        result = paired_permutation_test(errors_a, errors_b, n_permutations=500)
        assert np.isfinite(result.p_value)
        assert 0.0 <= result.p_value <= 1.0
