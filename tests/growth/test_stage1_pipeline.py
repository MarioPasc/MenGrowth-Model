# tests/growth/test_stage1_pipeline.py
"""Tests for Stage 1 Volumetric Baseline pipeline.

Covers:
- Trajectory loading from H5 (with synthetic H5 fixture)
- Gompertz mean function integration with HGP
- Bootstrap CI computation
- Per-patient error analysis
- Full LOPO-CV pipeline on synthetic data
- HGP linear vs Gompertz ablation

Markers: phase4, unit
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures: synthetic H5 and trajectories
# ---------------------------------------------------------------------------


def _create_synthetic_h5(path: str, n_patients: int = 10, seed: int = 42) -> None:
    """Create a minimal MenGrowth-like H5 file for testing.

    Generates patients with 2-6 timepoints, linear-ish volume growth,
    and partial metadata (only first 3 patients have age/sex).
    """
    rng = np.random.default_rng(seed)

    # Generate patient structure
    patient_list = [f"TestPatient-{i:04d}" for i in range(1, n_patients + 1)]
    scans_per_patient = rng.integers(2, 7, size=n_patients)
    total_scans = int(scans_per_patient.sum())

    # Build CSR offsets
    offsets = np.zeros(n_patients + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(scans_per_patient)

    # Build per-scan arrays
    scan_ids = []
    patient_ids = []
    timepoint_idx = np.zeros(total_scans, dtype=np.int32)

    for i, pid in enumerate(patient_list):
        for j in range(scans_per_patient[i]):
            scan_idx = offsets[i] + j
            scan_ids.append(f"{pid}-{j:03d}")
            patient_ids.append(pid)
            timepoint_idx[scan_idx] = j

    # Semantic features: log-volumes with some growth trend.
    # Column layout (matches src/growth/data/semantic_features.py):
    #   [log(V_total+1), log(V_NCR+1), log(V_ED+1), log(V_ET+1)]
    # Stage 1 reads column 3 (V_ET = enhancing-mass volume) as the growth signal.
    semantic_vol = np.zeros((total_scans, 4), dtype=np.float32)
    for i in range(n_patients):
        base_vol = rng.uniform(5.0, 12.0)  # log(V_ET+1) baseline
        growth_rate = rng.uniform(-0.1, 0.5)

        for j in range(scans_per_patient[i]):
            idx = offsets[i] + j
            # ET volume (column 3) = primary growth signal.
            semantic_vol[idx, 3] = base_vol + growth_rate * j + rng.normal(0, 0.1)
            # NETC and SNFH (columns 1, 2): small fractions, realistic for meningioma.
            semantic_vol[idx, 1] = semantic_vol[idx, 3] * rng.uniform(0.0, 0.05)
            semantic_vol[idx, 2] = semantic_vol[idx, 3] * rng.uniform(0.1, 0.4)
            # V_total (column 0) = sum of the parts (in voxel-count space, not log-space,
            # but the synthetic data only needs to be non-zero and finite — the loader
            # never reads column 0 anymore).
            semantic_vol[idx, 0] = (
                semantic_vol[idx, 1] + semantic_vol[idx, 2] + semantic_vol[idx, 3]
            )

    # Metadata (partial: only first 3 patients have age/sex)
    age = np.full(total_scans, np.nan, dtype=np.float32)
    sex = np.array(["unknown"] * total_scans, dtype="S10")

    metadata_patients = min(3, n_patients)
    ages_known = [50.0, 65.0, 42.0]
    sexes_known = [b"F", b"M", b"M"]
    for i in range(metadata_patients):
        for j in range(scans_per_patient[i]):
            idx = offsets[i] + j
            age[idx] = ages_known[i]
            sex[idx] = sexes_known[i]

    # Write H5
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
        meta.create_dataset("grade", data=np.full(total_scans, -1, dtype=np.int8))


@pytest.fixture
def synthetic_h5(tmp_path: Path) -> Path:
    """Create and return path to a synthetic MenGrowth H5 file."""
    h5_path = tmp_path / "test_mengrowth.h5"
    _create_synthetic_h5(str(h5_path), n_patients=10)
    return h5_path


@pytest.fixture
def synthetic_trajectories():
    """Create synthetic PatientTrajectory objects for unit tests."""
    from growth.shared.growth_models import PatientTrajectory

    rng = np.random.default_rng(42)
    trajectories = []

    for i in range(12):
        n_tp = rng.integers(2, 6)
        base = rng.uniform(5, 12)
        rate = rng.uniform(-0.1, 0.5)
        times = np.arange(n_tp, dtype=np.float64)
        obs = base + rate * times + rng.normal(0, 0.1, size=n_tp)
        trajectories.append(
            PatientTrajectory(
                patient_id=f"Synth-{i:03d}",
                times=times,
                observations=obs,
            )
        )

    return trajectories


# ---------------------------------------------------------------------------
# Tests: Trajectory Loader
# ---------------------------------------------------------------------------


class TestTrajectoryLoader:
    """Tests for load_trajectories_from_h5."""

    def test_load_basic(self, synthetic_h5: Path) -> None:
        """Basic loading produces valid trajectories."""
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_trajectories_from_h5,
        )

        trajs = load_trajectories_from_h5(str(synthetic_h5))
        assert len(trajs) == 10
        for t in trajs:
            assert t.n_timepoints >= 2
            assert t.observations.shape[1] == 1  # D=1
            assert np.all(np.isfinite(t.observations))
            assert np.all(np.isfinite(t.times))

    def test_exclude_patients(self, synthetic_h5: Path) -> None:
        """Excluding patients reduces count."""
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_trajectories_from_h5,
        )

        trajs = load_trajectories_from_h5(
            str(synthetic_h5),
            exclude_patients=["TestPatient-0001", "TestPatient-0002"],
        )
        assert len(trajs) == 8
        pids = {t.patient_id for t in trajs}
        assert "TestPatient-0001" not in pids
        assert "TestPatient-0002" not in pids

    def test_min_timepoints_filter(self, synthetic_h5: Path) -> None:
        """min_timepoints filter works correctly."""
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_trajectories_from_h5,
        )

        trajs_2 = load_trajectories_from_h5(str(synthetic_h5), min_timepoints=2)
        trajs_4 = load_trajectories_from_h5(str(synthetic_h5), min_timepoints=4)
        assert len(trajs_4) <= len(trajs_2)
        for t in trajs_4:
            assert t.n_timepoints >= 4

    def test_ordinal_time(self, synthetic_h5: Path) -> None:
        """Ordinal time uses integer timepoint indices."""
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_trajectories_from_h5,
        )

        trajs = load_trajectories_from_h5(str(synthetic_h5), time_variable="ordinal")
        for t in trajs:
            # Times should be 0, 1, 2, ...
            expected = np.arange(t.n_timepoints, dtype=np.float64)
            np.testing.assert_array_equal(t.times, expected)

    def test_days_fallback_to_ordinal(self, synthetic_h5: Path) -> None:
        """Requesting days_from_baseline falls back to ordinal when unavailable."""
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_trajectories_from_h5,
        )

        trajs = load_trajectories_from_h5(str(synthetic_h5), time_variable="days_from_baseline")
        # Should fall back to ordinal
        for t in trajs:
            expected = np.arange(t.n_timepoints, dtype=np.float64)
            np.testing.assert_array_equal(t.times, expected)

    def test_covariates_partial_metadata(self, synthetic_h5: Path) -> None:
        """Only patients with complete metadata get covariates."""
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_trajectories_from_h5,
        )

        trajs = load_trajectories_from_h5(str(synthetic_h5), covariate_features=["age", "sex"])
        n_with_cov = sum(1 for t in trajs if t.covariates is not None)
        # Only first 3 patients have age/sex
        assert n_with_cov == 3

        for t in trajs:
            if t.covariates is not None:
                assert "age" in t.covariates
                assert "sex" in t.covariates
                assert t.covariates["age"] > 0
                assert t.covariates["sex"] in (0.0, 1.0)

    def test_sorted_by_patient_id(self, synthetic_h5: Path) -> None:
        """Trajectories are sorted by patient_id."""
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_trajectories_from_h5,
        )

        trajs = load_trajectories_from_h5(str(synthetic_h5))
        pids = [t.patient_id for t in trajs]
        assert pids == sorted(pids)

    def test_file_not_found(self) -> None:
        """Missing H5 file raises FileNotFoundError."""
        from growth.stages.stage1_volumetric.trajectory_loader import (
            load_trajectories_from_h5,
        )

        with pytest.raises(FileNotFoundError):
            load_trajectories_from_h5("/nonexistent/path.h5")

    def test_loader_uses_et_volume_column_3(self, tmp_path: Path) -> None:
        """Loader must read semantic/volume[:, 3] (V_ET), not column 0 (V_total).

        Regression guard for the meningioma label fix: BraTS-MEN labels are
        {1=NETC, 2=SNFH (edema), 3=ET}, and the clinically tracked endpoint for
        meningioma growth is the enhancing-tumor volume — column 3 of
        ``semantic/volume`` (per ``src/growth/data/semantic_features.py``). Loading
        column 0 would inject SNFH/edema noise into every trajectory.
        """
        from growth.stages.stage1_volumetric.trajectory_loader import (
            _ET_VOL_IDX,
            load_trajectories_from_h5,
        )

        # Sanity: the constant itself must point at column 3.
        assert _ET_VOL_IDX == 3, (
            f"_ET_VOL_IDX must be 3 (V_ET), got {_ET_VOL_IDX}. "
            "Reading the wrong column silently switches the growth target."
        )

        # Build a tiny H5 with sentinel values per-column so the loader's choice
        # is unambiguous: column 0 = 100 (decoy), column 3 = 7 (the truth).
        h5_path = tmp_path / "et_index_check.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs["version"] = "2.0"
            f.attrs["domain"] = "MenGrowth"
            n_scans = 4
            f.create_dataset(
                "scan_ids", data=np.array([f"S-{i}" for i in range(n_scans)], dtype="S20")
            )
            f.create_dataset("patient_ids", data=np.array(["P-0001"] * n_scans, dtype="S20"))
            f.create_dataset("timepoint_idx", data=np.arange(n_scans, dtype=np.int32))

            grp = f.create_group("longitudinal")
            grp.create_dataset("patient_list", data=np.array(["P-0001"], dtype="S20"))
            grp.create_dataset("patient_offsets", data=np.array([0, n_scans], dtype=np.int32))

            # Column 0 (decoy) = 100, columns 1/2 = 50, column 3 (V_ET) = 7.
            vol = np.zeros((n_scans, 4), dtype=np.float32)
            vol[:, 0] = 100.0
            vol[:, 1] = 50.0
            vol[:, 2] = 50.0
            vol[:, 3] = 7.0
            sem = f.create_group("semantic")
            sem.create_dataset("volume", data=vol)
            sem.create_dataset("location", data=np.full((n_scans, 3), 0.5, dtype=np.float32))
            sem.create_dataset("shape", data=np.zeros((n_scans, 3), dtype=np.float32))

        trajs = load_trajectories_from_h5(str(h5_path))
        assert len(trajs) == 1
        obs = trajs[0].observations.flatten()
        assert np.allclose(obs, 7.0), (
            f"Loader returned {obs.tolist()}; expected all 7.0 (V_ET column). "
            "If you see 100.0 the loader is reading V_total — the bug is back."
        )

    def test_compute_et_volumes_from_segs_counts_merged_meningioma_mass(
        self, tmp_path: Path
    ) -> None:
        """Helper must count NETC ∪ ET (the merged meningioma mass), excluding edema.

        Under the BSF 2-label translation, NETC and ET both belong to the solid
        meningioma mass. The helper counts ``(seg == 1) | (seg == 3)`` so it
        returns the meningioma-mass volume regardless of whether the segs are
        raw BraTS-MEN GT (separate NETC/ET) or post-merge LoRA predictions
        (label 3 already merged).
        """
        from growth.stages.stage1_volumetric.trajectory_loader import (
            compute_et_volumes_from_segs,
        )

        h5_path = tmp_path / "et_seg_count.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs["version"] = "2.0"
            f.attrs["domain"] = "MenGrowth"
            # One patient, two scans. seg shape [N, 1, D, H, W] = [2, 1, 4, 4, 4].
            seg = np.zeros((2, 1, 4, 4, 4), dtype=np.int8)
            # Scan 0: 8 voxels of label 3 (ET) + 16 voxels of label 2 (edema, EXCLUDED).
            seg[0, 0, :2, :2, :2] = 3
            seg[0, 0, 2:, :2, :2] = 2
            # Scan 1: 27 voxels of ET (label 3) + 5 voxels of NETC (label 1) → 32 mass voxels.
            seg[1, 0, :3, :3, :3] = 3
            seg[1, 0, 3, :, :].flat[:5] = 1

            f.create_dataset("segs", data=seg)
            grp = f.create_group("longitudinal")
            grp.create_dataset("patient_list", data=np.array(["P-A"], dtype="S20"))
            grp.create_dataset("patient_offsets", data=np.array([0, 2], dtype=np.int32))

        result = compute_et_volumes_from_segs(str(h5_path))
        assert "P-A" in result
        # Expected log1p of meningioma-mass voxel counts: scan 0 = 8 (ET only),
        # scan 1 = 27 + 5 = 32 (ET + NETC merged). Edema is excluded.
        np.testing.assert_allclose(result["P-A"], [np.log1p(8), np.log1p(32)], rtol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Gompertz Mean Function
# ---------------------------------------------------------------------------


class TestGompertzMeanFunction:
    """Tests for Gompertz fitting and integration with HGP."""

    def test_fit_gompertz_basic(self) -> None:
        """Gompertz fit converges on synthetic data."""
        from growth.stages.stage1_volumetric.gompertz import fit_gompertz

        times = np.linspace(0, 10, 50)
        # True Gompertz: K=10, b=2, c=0.3
        volumes = 10.0 * np.exp(-2.0 * np.exp(-0.3 * times))
        volumes += np.random.default_rng(42).normal(0, 0.1, size=len(times))

        params = fit_gompertz(times, volumes)
        assert params.K > 0
        assert params.b > 0
        assert params.c > 0

    def test_gompertz_mean_function_callable(self) -> None:
        """GompertzMeanFunction is callable and returns correct shape."""
        from growth.stages.stage1_volumetric.gompertz import (
            GompertzMeanFunction,
            GompertzParams,
        )

        mf = GompertzMeanFunction(GompertzParams(K=10.0, b=2.0, c=0.3))
        t = np.array([0.0, 1.0, 5.0, 10.0])
        result = mf(t)

        assert result.shape == (4,)
        assert np.all(np.isfinite(result))
        # Gompertz is monotonically increasing
        assert np.all(np.diff(result) >= 0)

    def test_hgp_gompertz_fit_d1(self, synthetic_trajectories) -> None:
        """HGP with Gompertz mean fits on D=1 data."""
        from growth.models.growth.hgp_model import HierarchicalGPModel

        model = HierarchicalGPModel(
            mean_function="gompertz",
            n_restarts=1,
            max_iter=200,
        )
        result = model.fit(synthetic_trajectories)

        assert np.isfinite(result.log_marginal_likelihood)
        assert result.n_train_patients == len(synthetic_trajectories)
        assert "gompertz_K" in result.hyperparameters
        assert "gompertz_b" in result.hyperparameters
        assert "gompertz_c" in result.hyperparameters

    def test_hgp_gompertz_predict(self, synthetic_trajectories) -> None:
        """HGP with Gompertz produces finite predictions."""
        from growth.models.growth.hgp_model import HierarchicalGPModel

        model = HierarchicalGPModel(
            mean_function="gompertz",
            n_restarts=1,
            max_iter=200,
        )
        model.fit(synthetic_trajectories)

        patient = synthetic_trajectories[0]
        t_pred = np.array([0.0, 1.0, 2.0, 3.0])
        pred = model.predict(patient, t_pred, n_condition=1)

        assert pred.mean.shape == (4, 1)
        assert pred.variance.shape == (4, 1)
        assert np.all(np.isfinite(pred.mean))
        assert np.all(pred.variance >= 0)

    def test_hgp_gompertz_name(self) -> None:
        """HGP Gompertz name includes mean function type."""
        from growth.models.growth.hgp_model import HierarchicalGPModel

        model = HierarchicalGPModel(mean_function="gompertz")
        assert "gompertz" in model.name()

    def test_hgp_gompertz_rejects_multidim(self) -> None:
        """Gompertz mean function is rejected for D > 1."""
        from growth.models.growth.hgp_model import HierarchicalGPModel
        from growth.shared.growth_models import PatientTrajectory

        model = HierarchicalGPModel(mean_function="gompertz", n_restarts=1)
        patients = [
            PatientTrajectory(
                patient_id="p1",
                times=np.array([0, 1, 2]),
                observations=np.random.randn(3, 2),  # D=2
            )
        ]
        with pytest.raises(ValueError, match="D=1"):
            model.fit(patients)


# ---------------------------------------------------------------------------
# Tests: HGP with mean_function parameter
# ---------------------------------------------------------------------------


class TestHGPMeanFunction:
    """Tests for HGP mean_function parameter."""

    def test_linear_mean_default(self) -> None:
        """HGP defaults to linear mean function."""
        from growth.models.growth.hgp_model import HierarchicalGPModel

        model = HierarchicalGPModel()
        assert model.mean_function == "linear"

    def test_invalid_mean_function(self) -> None:
        """Invalid mean_function raises ValueError."""
        from growth.models.growth.hgp_model import HierarchicalGPModel

        with pytest.raises(ValueError, match="Invalid mean_function"):
            HierarchicalGPModel(mean_function="invalid")

    def test_linear_backward_compat(self, synthetic_trajectories) -> None:
        """HGP with explicit linear mean produces same results as before."""
        from growth.models.growth.hgp_model import HierarchicalGPModel

        model = HierarchicalGPModel(
            mean_function="linear",
            n_restarts=1,
            max_iter=200,
            seed=42,
        )
        result = model.fit(synthetic_trajectories)

        assert np.isfinite(result.log_marginal_likelihood)
        assert "linear" in model.name()


# ---------------------------------------------------------------------------
# Tests: Bootstrap CIs
# ---------------------------------------------------------------------------


class TestBootstrapCIs:
    """Tests for bootstrap CI computation on LOPO results."""

    def test_bootstrap_metric_r2(self) -> None:
        """Bootstrap on R^2 produces valid CI."""
        from growth.shared.bootstrap import bootstrap_metric
        from growth.shared.metrics import compute_r2

        rng = np.random.default_rng(42)
        y_true = rng.normal(10, 2, size=30)
        y_pred = y_true + rng.normal(0, 0.5, size=30)

        result = bootstrap_metric(y_true, y_pred, compute_r2, n_bootstrap=500)

        assert result.ci_lower <= result.estimate <= result.ci_upper
        assert result.estimate > 0  # Good predictions
        assert result.n_bootstrap == 500

    def test_bootstrap_on_lopo_results(self, synthetic_trajectories) -> None:
        """End-to-end bootstrap on LOPO-CV results."""
        from experiments.utils.experiment_output import (
            compute_bootstrap_cis_with_distribution,
        )
        from growth.models.growth.scalar_gp import ScalarGP
        from growth.shared.lopo import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, synthetic_trajectories, n_restarts=1, max_iter=200)

        cis, dist = compute_bootstrap_cis_with_distribution(results, n_bootstrap=200, seed=42)

        assert "r2_log" in cis
        assert "mae_log" in cis
        assert "rmse_log" in cis
        assert cis["r2_log"].ci_lower <= cis["r2_log"].ci_upper
        assert len(dist["r2_log"]) == 200  # Full distribution saved

    def test_bootstrap_insufficient_data(self) -> None:
        """Bootstrap handles insufficient data gracefully."""
        from experiments.utils.experiment_output import (
            compute_bootstrap_cis_with_distribution,
        )
        from growth.shared.lopo import LOPOResults

        # Empty results
        results = LOPOResults(model_name="test", fold_results=[])
        cis, dist = compute_bootstrap_cis_with_distribution(results)
        assert len(cis) == 0


# ---------------------------------------------------------------------------
# Tests: Per-Patient Error Analysis
# ---------------------------------------------------------------------------


class TestPerPatientErrors:
    """Tests for per-patient error computation and summary."""

    def test_per_patient_errors_structure(self, synthetic_trajectories) -> None:
        """Per-patient errors have correct structure."""
        from experiments.utils.experiment_output import (
            extract_per_patient_errors,
            summarize_errors,
        )
        from growth.models.growth.lme_model import LMEGrowthModel
        from growth.shared.lopo import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(LMEGrowthModel, synthetic_trajectories)

        errors = extract_per_patient_errors(results)
        assert len(errors) > 0

        for pid, err_dict in errors.items():
            assert "error" in err_dict
            assert "abs_error" in err_dict
            assert "actual" in err_dict
            assert "predicted" in err_dict
            assert "within_95_ci" in err_dict
            assert "ci_width" in err_dict

        summary = summarize_errors(errors)
        assert summary["n_patients"] == len(errors)
        assert summary["abs_error_min"] <= summary["abs_error_mean"]
        assert summary["abs_error_mean"] <= summary["abs_error_max"]


# ---------------------------------------------------------------------------
# Tests: Full LOPO-CV Pipeline
# ---------------------------------------------------------------------------


class TestLOPOPipeline:
    """Tests for the complete LOPO-CV pipeline on synthetic data."""

    def test_scalar_gp_lopo_no_nan(self, synthetic_trajectories) -> None:
        """S1-T1: ScalarGP LOPO-CV completes without NaN."""
        from growth.models.growth.scalar_gp import ScalarGP
        from growth.shared.lopo import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, synthetic_trajectories, n_restarts=1, max_iter=200)

        assert len(results.failed_folds) == 0
        r2 = results.aggregate_metrics.get("last_from_rest/r2_log")
        assert r2 is not None
        assert np.isfinite(r2)

    def test_lme_positive_r2(self, synthetic_trajectories) -> None:
        """S1-T2: LME captures temporal trend (R2 > 0 on synthetic data)."""
        from growth.models.growth.lme_model import LMEGrowthModel
        from growth.shared.lopo import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(LMEGrowthModel, synthetic_trajectories)

        r2 = results.aggregate_metrics.get("last_from_rest/r2_log", -999)
        # On well-behaved synthetic data, LME should do well
        assert r2 > 0, f"LME R2={r2:.4f}, expected > 0"

    def test_hgp_linear_lopo(self, synthetic_trajectories) -> None:
        """HGP with linear mean completes LOPO-CV."""
        from growth.models.growth.hgp_model import HierarchicalGPModel
        from growth.shared.lopo import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(
            HierarchicalGPModel,
            synthetic_trajectories,
            mean_function="linear",
            n_restarts=1,
            max_iter=200,
        )

        assert len(results.failed_folds) == 0
        r2 = results.aggregate_metrics.get("last_from_rest/r2_log")
        assert np.isfinite(r2)

    def test_hgp_gompertz_lopo(self, synthetic_trajectories) -> None:
        """HGP with Gompertz mean completes LOPO-CV."""
        from growth.models.growth.hgp_model import HierarchicalGPModel
        from growth.shared.lopo import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(
            HierarchicalGPModel,
            synthetic_trajectories,
            mean_function="gompertz",
            n_restarts=1,
            max_iter=200,
        )

        assert len(results.failed_folds) == 0
        r2 = results.aggregate_metrics.get("last_from_rest/r2_log")
        assert np.isfinite(r2)

    def test_all_models_finite_predictions(self, synthetic_trajectories) -> None:
        """All three models produce finite predictions under LOPO."""
        from growth.models.growth.hgp_model import HierarchicalGPModel
        from growth.models.growth.lme_model import LMEGrowthModel
        from growth.models.growth.scalar_gp import ScalarGP
        from growth.shared.lopo import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])

        models = {
            "ScalarGP": (ScalarGP, {"n_restarts": 1, "max_iter": 200}),
            "LME": (LMEGrowthModel, {}),
            "HGP": (HierarchicalGPModel, {"n_restarts": 1, "max_iter": 200}),
        }

        for name, (cls, kwargs) in models.items():
            results = evaluator.evaluate(cls, synthetic_trajectories, **kwargs)
            assert len(results.failed_folds) == 0, (
                f"{name} had {len(results.failed_folds)} failed folds"
            )

            for fr in results.fold_results:
                for protocol, preds in fr.predictions.items():
                    for p in preds:
                        assert np.isfinite(p["pred_mean"]), (
                            f"{name}/{fr.patient_id}/{protocol}: NaN prediction"
                        )


# ---------------------------------------------------------------------------
# Tests: Model Comparison (Ablation)
# ---------------------------------------------------------------------------


class TestModelComparison:
    """Tests for model comparison and ranking."""

    def test_hgp_linear_vs_gompertz(self, synthetic_trajectories) -> None:
        """S1-T7: Both HGP variants complete and can be compared."""
        from growth.models.growth.hgp_model import HierarchicalGPModel
        from growth.shared.lopo import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])

        r2_linear = evaluator.evaluate(
            HierarchicalGPModel,
            synthetic_trajectories,
            mean_function="linear",
            n_restarts=1,
            max_iter=200,
        ).aggregate_metrics.get("last_from_rest/r2_log", float("nan"))

        r2_gompertz = evaluator.evaluate(
            HierarchicalGPModel,
            synthetic_trajectories,
            mean_function="gompertz",
            n_restarts=1,
            max_iter=200,
        ).aggregate_metrics.get("last_from_rest/r2_log", float("nan"))

        # Both should be finite (comparison outcome is data-dependent)
        assert np.isfinite(r2_linear), f"HGP linear R2={r2_linear}"
        assert np.isfinite(r2_gompertz), f"HGP Gompertz R2={r2_gompertz}"


# ---------------------------------------------------------------------------
# Tests: Config and Build
# ---------------------------------------------------------------------------


class TestConfigBuild:
    """Tests for config loading and model building."""

    def test_build_model_configs_default(self) -> None:
        """Default config builds 4 models (ScalarGP, LME, HGP, HGP_Gompertz)."""
        from experiments.stage1_volumetric.run_stage1 import _build_model_configs

        cfg = {
            "experiment": {"seed": 42},
            "gp": {
                "kernel": "matern52",
                "mean_function": "linear",
                "n_restarts": 5,
                "max_iter": 1000,
                "lengthscale_bounds": [0.1, 50.0],
                "signal_var_bounds": [0.001, 10.0],
                "noise_var_bounds": [1e-6, 5.0],
            },
            "lme": {"method": "reml"},
            "models": {
                "scalar_gp": True,
                "lme": True,
                "hgp": True,
                "hgp_gompertz": True,
            },
        }

        models = _build_model_configs(cfg)
        assert "ScalarGP" in models
        assert "LME" in models
        assert "HGP" in models
        assert "HGP_Gompertz" in models

    def test_build_model_configs_with_covariates(self) -> None:
        """Config with covariates passes them to model kwargs."""
        from experiments.stage1_volumetric.run_stage1 import _build_model_configs

        cfg = {
            "experiment": {"seed": 42},
            "gp": {
                "kernel": "matern52",
                "mean_function": "linear",
                "n_restarts": 5,
                "max_iter": 1000,
                "lengthscale_bounds": [0.1, 50.0],
                "signal_var_bounds": [0.001, 10.0],
                "noise_var_bounds": [1e-6, 5.0],
            },
            "lme": {"method": "reml"},
            "covariates": {
                "enabled": True,
                "features": ["age", "sex"],
                "missing_strategy": "skip",
            },
            "models": {"scalar_gp": True, "lme": True, "hgp": False, "hgp_gompertz": False},
        }

        models = _build_model_configs(cfg)
        _, gp_kwargs = models["ScalarGP"]
        assert gp_kwargs["use_covariates"] is True
        assert gp_kwargs["covariate_names"] == ["age", "sex"]

    def test_build_model_configs_selective(self) -> None:
        """Disabling models removes them from config."""
        from experiments.stage1_volumetric.run_stage1 import _build_model_configs

        cfg = {
            "experiment": {"seed": 42},
            "gp": {
                "kernel": "matern52",
                "mean_function": "linear",
                "n_restarts": 5,
                "max_iter": 1000,
                "lengthscale_bounds": [0.1, 50.0],
                "signal_var_bounds": [0.001, 10.0],
                "noise_var_bounds": [1e-6, 5.0],
            },
            "lme": {"method": "reml"},
            "models": {"scalar_gp": True, "lme": False, "hgp": False, "hgp_gompertz": False},
        }

        models = _build_model_configs(cfg)
        assert "ScalarGP" in models
        assert "LME" not in models
        assert "HGP" not in models


# ---------------------------------------------------------------------------
# Tests: Result Saving
# ---------------------------------------------------------------------------


class TestResultSaving:
    """Tests for result serialization."""

    def test_save_and_load_results(self, synthetic_trajectories, tmp_path: Path) -> None:
        """Results can be saved and reloaded as JSON."""
        from experiments.utils.experiment_output import (
            save_experiment_metadata,
            save_stage_results,
        )
        from growth.models.growth.lme_model import LMEGrowthModel
        from growth.shared.lopo import LOPOEvaluator

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(LMEGrowthModel, synthetic_trajectories)

        ci_results = save_stage_results(
            output_dir=tmp_path,
            model_name="LME",
            lopo_results=results,
            trajectories=synthetic_trajectories,
            config={"test": True},
            stage_name="test",
            bootstrap_n=100,
        )

        save_experiment_metadata(
            output_dir=tmp_path,
            trajectories=synthetic_trajectories,
            config={"test": True},
            stage_name="test",
            all_model_results={"LME": results},
            all_bootstrap_cis={"LME": ci_results} if ci_results else {},
        )

        # Check files exist
        assert (tmp_path / "LME" / "lopo_results.json").exists()
        assert (tmp_path / "LME" / "raw_predictions.json").exists()
        assert (tmp_path / "LME" / "per_patient_errors.json").exists()
        assert (tmp_path / "LME" / "error_summary.json").exists()
        assert (tmp_path / "LME" / "bootstrap_cis.json").exists()
        assert (tmp_path / "LME" / "bootstrap_samples.json").exists()
        assert (tmp_path / "LME" / "hyperparameters.json").exists()
        assert (tmp_path / "model_comparison.json").exists()
        assert (tmp_path / "run_metadata.json").exists()
        assert (tmp_path / "trajectories_used.json").exists()

        # Verify JSON is valid
        with open(tmp_path / "model_comparison.json") as f:
            comparison = json.load(f)
        assert "models" in comparison
        assert "LME" in comparison["models"]

        # Verify raw predictions have aligned data
        with open(tmp_path / "LME" / "raw_predictions.json") as f:
            raw = json.load(f)
        assert "last_from_rest" in raw
        for pred in raw["last_from_rest"]:
            assert "patient_id" in pred
            assert "actual" in pred
            assert "predicted" in pred
            assert "variance" in pred


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_two_timepoint_patient(self) -> None:
        """Models handle patients with exactly 2 timepoints."""
        from growth.models.growth.scalar_gp import ScalarGP
        from growth.shared.growth_models import PatientTrajectory
        from growth.shared.lopo import LOPOEvaluator

        # 5 patients with exactly 2 timepoints
        patients = []
        rng = np.random.default_rng(42)
        for i in range(5):
            base = rng.uniform(5, 12)
            patients.append(
                PatientTrajectory(
                    patient_id=f"P{i}",
                    times=np.array([0.0, 1.0]),
                    observations=np.array([base, base + rng.uniform(-0.5, 1.0)]),
                )
            )

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(ScalarGP, patients, n_restarts=1, max_iter=200)

        assert len(results.failed_folds) == 0

    def test_constant_volume_patient(self) -> None:
        """Models handle patients with constant volume (zero growth)."""
        from growth.models.growth.lme_model import LMEGrowthModel
        from growth.shared.growth_models import PatientTrajectory
        from growth.shared.lopo import LOPOEvaluator

        patients = []
        rng = np.random.default_rng(42)
        for i in range(6):
            base = rng.uniform(5, 12)
            n_tp = rng.integers(2, 5)
            # Constant volume + tiny noise
            obs = np.full(n_tp, base) + rng.normal(0, 0.01, size=n_tp)
            patients.append(
                PatientTrajectory(
                    patient_id=f"P{i}",
                    times=np.arange(n_tp, dtype=np.float64),
                    observations=obs,
                )
            )

        evaluator = LOPOEvaluator(prediction_protocols=["last_from_rest"])
        results = evaluator.evaluate(LMEGrowthModel, patients)

        assert len(results.failed_folds) == 0
