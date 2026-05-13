"""Tests for the variance_key / uncertainty.signal refactor.

Covers:
  * trajectory_loader: legacy ``estimator`` path remains bit-identical when
    no ``variance_key`` is passed.
  * trajectory_loader: ``variance_key`` reads the named H5 dataset
    *as the variance itself* (no squaring), with NaN→0 then floor.
  * Cohort: ``uncertainty.signal`` config field flows through
    ``load_cohort`` to ``empirical_sigma_v_sq_flat`` and ``signal_name``.
  * enrich_h5_uncertainty: idempotent; writes only missing datasets;
    backup is created exactly once on first run.

Markers: phase4, unit
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers — synthetic H5 + a candidate signal dataset.
# ---------------------------------------------------------------------------


def _make_h5(path: Path, *, n_patients: int = 6, seed: int = 7) -> dict:
    """Synthetic v2.0 H5 with /uncertainty group + men_mean_entropy column.

    Returns a dict of arrays used to author the file, for use in assertions.
    """
    rng = np.random.default_rng(seed)
    pids = [f"P-{i:03d}" for i in range(n_patients)]
    scans = rng.integers(2, 5, size=n_patients).astype(np.int64)
    n_scans = int(scans.sum())

    offsets = np.zeros(n_patients + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(scans)

    timepoint_idx = np.zeros(n_scans, dtype=np.int32)
    patient_ids_per_scan: list[str] = []
    for i, pid in enumerate(pids):
        for j in range(int(scans[i])):
            timepoint_idx[offsets[i] + j] = j
            patient_ids_per_scan.append(pid)

    logvol_mean = (
        rng.uniform(5.0, 12.0, size=n_scans) + rng.normal(0, 0.05, size=n_scans)
    ).astype(np.float32)
    logvol_std = rng.uniform(0.01, 0.04, size=n_scans).astype(np.float32)
    logvol_median = (logvol_mean + rng.normal(0, 0.001, size=n_scans)).astype(np.float32)
    logvol_mad_scaled = (logvol_std * 1.1).astype(np.float32)
    logvol_ensemble = (logvol_mean + rng.normal(0, 0.002, size=n_scans)).astype(np.float32)
    vol_mean = np.expm1(logvol_mean).astype(np.float32)
    vol_std = (logvol_std * np.maximum(vol_mean, 1.0)).astype(np.float32)
    # Entropy: bounded in [0, ln2], deliberately *not* a squared std.
    men_entropy = rng.uniform(0.1, 0.6, size=n_scans).astype(np.float32)
    men_boundary_entropy = rng.uniform(0.05, 0.3, size=n_scans).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.attrs["version"] = "2.0"
        f.attrs["n_patients"] = n_patients
        f.attrs["n_scans"] = n_scans
        f.attrs["roi_size"] = [192, 192, 192]
        f.attrs["spacing"] = [1.0, 1.0, 1.0]
        f.attrs["channel_order"] = ["t2f", "t1c", "t1n", "t2w"]

        f.create_dataset("patient_ids", data=np.array(patient_ids_per_scan, dtype="S30"))
        f.create_dataset("timepoint_idx", data=timepoint_idx)
        grp = f.create_group("longitudinal")
        grp.create_dataset("patient_list", data=np.array(pids, dtype="S30"))
        grp.create_dataset("patient_offsets", data=offsets)

        uq = f.create_group("uncertainty")
        uq.create_dataset("logvol_mean", data=logvol_mean)
        uq.create_dataset("logvol_std", data=logvol_std)
        uq.create_dataset("logvol_median", data=logvol_median)
        uq.create_dataset("logvol_mad_scaled", data=logvol_mad_scaled)
        uq.create_dataset("logvol_ensemble", data=logvol_ensemble)
        uq.create_dataset("vol_mean", data=vol_mean)
        uq.create_dataset("vol_std", data=vol_std)
        uq.create_dataset("men_mean_entropy", data=men_entropy)
        uq.create_dataset("men_boundary_entropy", data=men_boundary_entropy)

    return {
        "pids": pids,
        "n_scans": n_scans,
        "logvol_std": logvol_std.astype(np.float64),
        "men_mean_entropy": men_entropy.astype(np.float64),
        "logvol_mad_scaled": logvol_mad_scaled.astype(np.float64),
        "vol_mean": vol_mean.astype(np.float64),
        "vol_std": vol_std.astype(np.float64),
        "men_boundary_entropy": men_boundary_entropy.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# trajectory_loader signature
# ---------------------------------------------------------------------------


class TestVarianceKey:
    """Direct-signal (variance_key) read path."""

    def test_legacy_path_unchanged_when_variance_key_none(self, tmp_path: Path) -> None:
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_uncertainty_trajectories_from_h5,
        )

        h5p = tmp_path / "syn.h5"
        _make_h5(h5p)

        legacy = load_uncertainty_trajectories_from_h5(
            str(h5p), estimator="mean_std", floor_variance=1e-12
        )
        # var_values == logvol_std**2 (after floor=1e-12, which is below any value)
        for traj in legacy:
            np.testing.assert_array_less(0.0, traj.observation_variance)
            # variance should equal std**2 exactly
            assert np.all(np.isfinite(traj.observation_variance))

    def test_variance_key_reads_without_squaring(self, tmp_path: Path) -> None:
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_uncertainty_trajectories_from_h5,
        )

        h5p = tmp_path / "syn.h5"
        truth = _make_h5(h5p)

        trajs = load_uncertainty_trajectories_from_h5(
            str(h5p),
            variance_key="men_mean_entropy",
            floor_variance=1e-12,
        )
        # Concat the per-patient variances back in their (patient, tp) order
        # and check against the source array. The loader sorts by patient_id
        # so we must reproduce that sort.
        with h5py.File(h5p, "r") as f:
            patient_offsets = f["longitudinal"]["patient_offsets"][:]
            patient_list = [
                p.decode() for p in f["longitudinal"]["patient_list"][:]
            ]
        sorted_pids = sorted(patient_list)
        for traj in trajs:
            i = patient_list.index(traj.patient_id)
            start, end = int(patient_offsets[i]), int(patient_offsets[i + 1])
            expected = truth["men_mean_entropy"][start:end]
            np.testing.assert_allclose(traj.observation_variance, expected, atol=1e-12)
        # Sanity: results should not equal logvol_std**2.
        assert len(trajs) == len(sorted_pids)

    def test_variance_key_applies_floor(self, tmp_path: Path) -> None:
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_uncertainty_trajectories_from_h5,
        )

        h5p = tmp_path / "syn.h5"
        _make_h5(h5p)

        floor = 0.3  # higher than most men_entropy values
        trajs = load_uncertainty_trajectories_from_h5(
            str(h5p), variance_key="men_mean_entropy", floor_variance=floor
        )
        for traj in trajs:
            assert np.all(traj.observation_variance >= floor)

    def test_variance_key_unknown_dataset_raises(self, tmp_path: Path) -> None:
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_uncertainty_trajectories_from_h5,
        )

        h5p = tmp_path / "syn.h5"
        _make_h5(h5p)
        with pytest.raises(ValueError, match="missing variance dataset"):
            load_uncertainty_trajectories_from_h5(
                str(h5p), variance_key="does_not_exist"
            )


# ---------------------------------------------------------------------------
# Cohort plumbing
# ---------------------------------------------------------------------------


class TestCohortSignal:
    """`uncertainty.signal` config field flows into Cohort.signal_name."""

    def _base_cfg(self, h5p: Path) -> dict:
        return {
            "paths": {"mengrowth_h5": str(h5p)},
            "time": {"variable": "ordinal", "missing_date_strategy": "mixed"},
            "uncertainty": {
                "estimator": "mean_std",
                "signal": None,
                "floor_variance": 1e-12,
            },
            "patients": {
                "exclude": [],
                "min_timepoints": 2,
                "skip_all_zero_volume": True,
                "max_logvol_std": None,
            },
        }

    def test_signal_none_uses_logvol_std_squared(self, tmp_path: Path) -> None:
        from experiments.stage1_volumetric.main_experiment.modules.cohort import (
            load_cohort,
        )

        h5p = tmp_path / "syn.h5"
        _make_h5(h5p)
        cfg = self._base_cfg(h5p)
        cohort = load_cohort(cfg)
        assert cohort.signal_name == "logvol_std²"
        # Empirical σ²_v should be > 0 and close to the std**2 range.
        assert cohort.empirical_sigma_v_sq_flat.min() > 0.0

    def test_signal_men_entropy_flows_through(self, tmp_path: Path) -> None:
        from experiments.stage1_volumetric.main_experiment.modules.cohort import (
            load_cohort,
        )

        h5p = tmp_path / "syn.h5"
        truth = _make_h5(h5p)
        cfg = self._base_cfg(h5p)
        cfg["uncertainty"]["signal"] = "men_mean_entropy"
        cohort = load_cohort(cfg)
        assert cohort.signal_name == "men_mean_entropy"
        # The mean of the empirical σ²_v should now match the mean of
        # truth["men_mean_entropy"] (within floor noise).
        assert np.isclose(
            cohort.empirical_sigma_v_sq_flat.mean(),
            float(truth["men_mean_entropy"].mean()),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Enrichment script — idempotence + content
# ---------------------------------------------------------------------------


class TestEnrichScript:
    """`enrich_h5_uncertainty.py` end-to-end."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            "-m",
            "experiments.stage1_volumetric.main_experiment.enrich_h5_uncertainty",
            *args,
        ]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[4],
        )

    def test_adds_missing_datasets_and_creates_backup(self, tmp_path: Path) -> None:
        h5p = tmp_path / "syn.h5"
        truth = _make_h5(h5p)

        result = self._run("--h5", str(h5p))
        assert result.returncode == 0, result.stderr

        backups = list(tmp_path.glob("syn.h5.bak.*"))
        assert len(backups) == 1, f"expected 1 .bak.<ts>, got {backups}"

        with h5py.File(h5p, "r") as f:
            uq = f["uncertainty"]
            assert "logvol_var" in uq
            assert "logvol_mad_var" in uq
            assert "vol_cv2" in uq
            assert "composite_logvol_x_boundary_entropy" in uq
            np.testing.assert_allclose(
                uq["logvol_var"][:], truth["logvol_std"] ** 2, rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                uq["logvol_mad_var"][:],
                truth["logvol_mad_scaled"] ** 2,
                rtol=0,
                atol=1e-12,
            )
            cv = truth["vol_std"] / np.maximum(truth["vol_mean"], 1.0)
            np.testing.assert_allclose(uq["vol_cv2"][:], cv**2, rtol=0, atol=1e-10)
            np.testing.assert_allclose(
                uq["composite_logvol_x_boundary_entropy"][:],
                (truth["logvol_std"] ** 2) * (1.0 + truth["men_boundary_entropy"]),
                rtol=0,
                atol=1e-10,
            )

    def test_idempotent_second_run_skips_and_no_new_backup(
        self, tmp_path: Path
    ) -> None:
        h5p = tmp_path / "syn.h5"
        _make_h5(h5p)

        r1 = self._run("--h5", str(h5p))
        assert r1.returncode == 0, r1.stderr
        backups_after_first = sorted(tmp_path.glob("syn.h5.bak.*"))
        assert len(backups_after_first) == 1

        r2 = self._run("--h5", str(h5p))
        assert r2.returncode == 0, r2.stderr
        backups_after_second = sorted(tmp_path.glob("syn.h5.bak.*"))
        # No second backup: nothing to write means nothing to back up.
        assert backups_after_second == backups_after_first
        assert "Nothing to do" in (r2.stdout + r2.stderr)
