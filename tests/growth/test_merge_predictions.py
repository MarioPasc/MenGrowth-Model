"""Tests for experiments/uncertainty_segmentation/merge_predictions.py.

All tests use synthetic data (no real checkpoints, H5, or NIfTI files).
Volumes are kept small (≤24³) for fast execution.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from experiments.uncertainty_segmentation.merge_predictions import (
    _SCALAR_COLUMNS,
    append_uncertainty_group,
    build_inversion_transforms,
    discover_dataset_scans,
    discover_prediction_scans,
    invert_and_copy_segmentations,
    invert_single_seg,
)

pytestmark = [pytest.mark.unit]

# Test constants
_TEST_ROI = (16, 16, 16)
_ORIGINAL_SHAPE = (24, 24, 18)
_IDENTITY_AFFINE = np.eye(4, dtype=np.float64)
_N_MEMBERS = 4
_SCAN_IDS = [
    "MenGrowth-0001-000",
    "MenGrowth-0001-001",
    "MenGrowth-0002-000",
    "MenGrowth-0002-001",
]


# =========================================================================
# Fixtures
# =========================================================================


def _save_nii(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Save a numpy array as NIfTI."""
    nib.save(nib.Nifti1Image(data, affine), str(path))


@pytest.fixture()
def synthetic_data_root(tmp_path: Path) -> Path:
    """Create MenGrowth dataset folder structure with small NIfTI files.

    Layout::
        MenGrowth-2025/
            MenGrowth-0001/
                MenGrowth-0001-000/{t2f,t1c,t1n,t2w,seg}.nii.gz
                MenGrowth-0001-001/{t2f,t1c,t1n,t2w,seg}.nii.gz
            MenGrowth-0002/
                MenGrowth-0002-000/{...}
                MenGrowth-0002-001/{...}

    Images have a nonzero interior and zero border so CropForeground
    produces a non-trivial crop.
    """
    root = tmp_path / "MenGrowth-2025"
    rng = np.random.RandomState(42)

    for scan_id in _SCAN_IDS:
        patient_id = scan_id.rsplit("-", 1)[0]
        scan_dir = root / patient_id / scan_id
        scan_dir.mkdir(parents=True)

        # Image with zero border (foreground in [2:-2, 2:-2, 1:-1])
        for mod in ("t2f", "t1c", "t1n", "t2w"):
            img = np.zeros(_ORIGINAL_SHAPE, dtype=np.float32)
            img[2:-2, 2:-2, 1:-1] = rng.rand(20, 20, 16).astype(np.float32)
            _save_nii(img, _IDENTITY_AFFINE, scan_dir / f"{mod}.nii.gz")

        # Seg with labels in the interior
        seg = np.zeros(_ORIGINAL_SHAPE, dtype=np.int8)
        seg[8:16, 8:16, 4:12] = 3  # ET block
        seg[6:8, 8:16, 4:12] = 1  # NETC
        _save_nii(seg, _IDENTITY_AFFINE, scan_dir / "seg.nii.gz")

    return root


@pytest.fixture()
def synthetic_predictions_dir(tmp_path: Path) -> Path:
    """Create predictions directory with ensemble segmentations at test ROI size.

    Each scan gets a ``segmentation.nii.gz`` at ``_TEST_ROI`` shape with
    BraTS labels.
    """
    pred_dir = tmp_path / "predictions"

    for scan_id in _SCAN_IDS:
        scan_pred = pred_dir / scan_id
        scan_pred.mkdir(parents=True)

        seg = np.zeros(_TEST_ROI, dtype=np.int8)
        seg[4:12, 4:12, 4:12] = 3
        seg[2:4, 4:12, 4:12] = 2
        _save_nii(seg, _IDENTITY_AFFINE, scan_pred / "segmentation.nii.gz")

    return pred_dir


@pytest.fixture()
def synthetic_volumes_csv(tmp_path: Path) -> Path:
    """Create a minimal volumes CSV with all required columns."""
    rng = np.random.RandomState(42)
    rows = []
    for scan_id in _SCAN_IDS:
        patient_id = scan_id.rsplit("-", 1)[0]
        tp = int(scan_id.split("-")[-1])
        vol = rng.uniform(1000, 50000)
        row = {
            "scan_id": scan_id,
            "patient_id": patient_id,
            "timepoint_idx": tp,
            "vol_mean": vol,
            "vol_std": vol * 0.1,
            "logvol_mean": math.log(vol + 1),
            "logvol_std": 0.1,
            "vol_median": vol * 0.98,
            "vol_mad": vol * 0.05,
            "logvol_median": math.log(vol * 0.98 + 1),
            "logvol_mad": 0.05,
            "logvol_mad_scaled": 0.05 * 1.4826,
            "vol_ensemble_mask": vol * 0.95,
            "logvol_ensemble_mask": math.log(vol * 0.95 + 1),
            "mean_entropy": rng.uniform(0.01, 0.1),
            "mean_mi": rng.uniform(0.005, 0.05),
            "mean_var": rng.uniform(0.001, 0.01),
            "men_mean_entropy": rng.uniform(0.01, 0.1),
            "men_mean_mi": rng.uniform(0.005, 0.05),
            "men_boundary_entropy": rng.uniform(0.02, 0.15),
            "men_boundary_mi": rng.uniform(0.01, 0.08),
            "inference_time_sec": rng.uniform(30, 90),
        }
        for m in range(_N_MEMBERS):
            member_vol = vol + rng.normal(0, vol * 0.05)
            row[f"vol_m{m}"] = member_vol
            row[f"logvol_m{m}"] = math.log(member_vol + 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "mengrowth_ensemble_volumes.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def synthetic_h5(tmp_path: Path) -> Path:
    """Create a minimal H5 file with scan_ids and semantic/ group."""
    h5_path = tmp_path / "MenGrowth.h5"
    roi = list(_TEST_ROI)

    with h5py.File(h5_path, "w") as f:
        n = len(_SCAN_IDS)
        f.attrs["n_scans"] = n
        f.attrs["n_patients"] = 2
        f.attrs["roi_size"] = roi
        f.attrs["version"] = "2.0"

        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("scan_ids", data=_SCAN_IDS, dtype=dt)
        f.create_dataset("patient_ids", data=[s.rsplit("-", 1)[0] for s in _SCAN_IDS], dtype=dt)
        f.create_dataset(
            "segs",
            data=np.zeros((n, 1, *roi), dtype=np.int8),
        )

        sem = f.create_group("semantic")
        sem.create_dataset("volume", data=np.zeros((n, 4), dtype=np.float32))
        sem.create_dataset("location", data=np.zeros((n, 3), dtype=np.float32))
        sem.create_dataset("shape", data=np.zeros((n, 3), dtype=np.float32))

    return h5_path


# =========================================================================
# Discovery Tests
# =========================================================================


class TestDiscovery:
    """Test scan discovery functions."""

    def test_discover_prediction_scans(self, synthetic_predictions_dir: Path) -> None:
        scans = discover_prediction_scans(synthetic_predictions_dir)
        assert set(scans.keys()) == set(_SCAN_IDS)
        for sid, path in scans.items():
            assert path.is_dir()
            assert path.name == sid

    def test_discover_prediction_scans_skips_non_mengrowth(
        self, synthetic_predictions_dir: Path
    ) -> None:
        (synthetic_predictions_dir / "brats_men_test").mkdir()
        (synthetic_predictions_dir / "some_file.txt").touch()
        scans = discover_prediction_scans(synthetic_predictions_dir)
        assert "brats_men_test" not in scans
        assert len(scans) == len(_SCAN_IDS)

    def test_discover_dataset_scans(self, synthetic_data_root: Path) -> None:
        scans = discover_dataset_scans(synthetic_data_root)
        assert set(scans.keys()) == set(_SCAN_IDS)
        for sid, path in scans.items():
            assert path.is_dir()
            assert (path / "t2f.nii.gz").exists()


# =========================================================================
# Inversion Tests
# =========================================================================


class TestInvertSegToOriginalSpace:
    """Test MONAI inverse transform of ensemble segmentations."""

    def test_inverted_shape_matches_original(
        self,
        synthetic_predictions_dir: Path,
        synthetic_data_root: Path,
    ) -> None:
        scan_id = _SCAN_IDS[0]
        ensemble_seg_path = synthetic_predictions_dir / scan_id / "segmentation.nii.gz"
        patient_id = scan_id.rsplit("-", 1)[0]
        t2f_path = synthetic_data_root / patient_id / scan_id / "t2f.nii.gz"
        forward = build_inversion_transforms(roi_size=_TEST_ROI)
        seg_np, affine = invert_single_seg(ensemble_seg_path, t2f_path, forward)

        assert seg_np.shape == _ORIGINAL_SHAPE
        assert seg_np.ndim == 3

    def test_inverted_labels_are_valid_brats(
        self,
        synthetic_predictions_dir: Path,
        synthetic_data_root: Path,
    ) -> None:
        scan_id = _SCAN_IDS[0]
        ensemble_seg_path = synthetic_predictions_dir / scan_id / "segmentation.nii.gz"
        patient_id = scan_id.rsplit("-", 1)[0]
        t2f_path = synthetic_data_root / patient_id / scan_id / "t2f.nii.gz"
        forward = build_inversion_transforms(roi_size=_TEST_ROI)
        seg_np, _ = invert_single_seg(ensemble_seg_path, t2f_path, forward)

        valid_labels = {0, 1, 2, 3}
        actual_labels = set(np.unique(seg_np).tolist())
        assert actual_labels.issubset(valid_labels), (
            f"Invalid labels: {actual_labels - valid_labels}"
        )

    def test_inverted_seg_has_nonzero_voxels(
        self,
        synthetic_predictions_dir: Path,
        synthetic_data_root: Path,
    ) -> None:
        scan_id = _SCAN_IDS[0]
        ensemble_seg_path = synthetic_predictions_dir / scan_id / "segmentation.nii.gz"
        patient_id = scan_id.rsplit("-", 1)[0]
        t2f_path = synthetic_data_root / patient_id / scan_id / "t2f.nii.gz"
        forward = build_inversion_transforms(roi_size=_TEST_ROI)
        seg_np, _ = invert_single_seg(ensemble_seg_path, t2f_path, forward)

        assert seg_np.sum() > 0, "Inverted seg is all zeros"

    def test_inverted_affine_matches_original(
        self,
        synthetic_predictions_dir: Path,
        synthetic_data_root: Path,
    ) -> None:
        scan_id = _SCAN_IDS[0]
        ensemble_seg_path = synthetic_predictions_dir / scan_id / "segmentation.nii.gz"
        patient_id = scan_id.rsplit("-", 1)[0]
        t2f_path = synthetic_data_root / patient_id / scan_id / "t2f.nii.gz"
        forward = build_inversion_transforms(roi_size=_TEST_ROI)
        _, affine = invert_single_seg(ensemble_seg_path, t2f_path, forward)

        original_affine = nib.load(str(t2f_path)).affine
        np.testing.assert_array_equal(affine, original_affine)


# =========================================================================
# Copy Segmentations Tests
# =========================================================================


class TestCopySegmentations:
    """Test invert_and_copy_segmentations end-to-end."""

    def test_copies_seg_to_dataset(
        self,
        synthetic_predictions_dir: Path,
        synthetic_data_root: Path,
    ) -> None:
        # Record original seg checksums to verify overwrite
        original_sums = {}
        for scan_id in _SCAN_IDS:
            patient_id = scan_id.rsplit("-", 1)[0]
            seg_path = synthetic_data_root / patient_id / scan_id / "seg.nii.gz"
            original_sums[scan_id] = nib.load(str(seg_path)).get_fdata().sum()

        results = invert_and_copy_segmentations(
            synthetic_predictions_dir,
            synthetic_data_root,
            roi_size=_TEST_ROI,
        )

        assert len(results) == len(_SCAN_IDS)
        for scan_id, dest_path in results.items():
            assert dest_path.exists(), f"seg.nii.gz missing for {scan_id}"
            new_seg = nib.load(str(dest_path))
            assert new_seg.shape == _ORIGINAL_SHAPE

    def test_dry_run_does_not_overwrite(
        self,
        synthetic_predictions_dir: Path,
        synthetic_data_root: Path,
    ) -> None:
        scan_id = _SCAN_IDS[0]
        patient_id = scan_id.rsplit("-", 1)[0]
        seg_path = synthetic_data_root / patient_id / scan_id / "seg.nii.gz"
        original_data = nib.load(str(seg_path)).get_fdata().copy()

        invert_and_copy_segmentations(
            synthetic_predictions_dir,
            synthetic_data_root,
            roi_size=_TEST_ROI,
            dry_run=True,
        )

        new_data = nib.load(str(seg_path)).get_fdata()
        np.testing.assert_array_equal(original_data, new_data)

    def test_raises_on_scan_count_mismatch(
        self,
        synthetic_predictions_dir: Path,
        synthetic_data_root: Path,
    ) -> None:
        # Add extra prediction scan
        extra = synthetic_predictions_dir / "MenGrowth-0099-000"
        extra.mkdir(parents=True)
        seg = np.zeros(_TEST_ROI, dtype=np.int8)
        _save_nii(seg, _IDENTITY_AFFINE, extra / "segmentation.nii.gz")

        with pytest.raises(ValueError, match="Scan ID mismatch"):
            invert_and_copy_segmentations(
                synthetic_predictions_dir,
                synthetic_data_root,
                roi_size=_TEST_ROI,
            )

    def test_raises_on_missing_segmentation_nii(
        self,
        synthetic_predictions_dir: Path,
        synthetic_data_root: Path,
    ) -> None:
        scan_id = _SCAN_IDS[0]
        seg_path = synthetic_predictions_dir / scan_id / "segmentation.nii.gz"
        seg_path.unlink()

        with pytest.raises(FileNotFoundError, match="segmentation.nii.gz"):
            invert_and_copy_segmentations(
                synthetic_predictions_dir,
                synthetic_data_root,
                roi_size=_TEST_ROI,
            )


# =========================================================================
# Uncertainty Group Tests
# =========================================================================


class TestAppendUncertaintyGroup:
    """Test H5 uncertainty group creation."""

    def test_all_datasets_present(
        self,
        synthetic_h5: Path,
        synthetic_volumes_csv: Path,
    ) -> None:
        append_uncertainty_group(
            synthetic_h5,
            synthetic_volumes_csv,
            rank=8,
            n_members=_N_MEMBERS,
            seed=42,
        )

        with h5py.File(synthetic_h5, "r") as f:
            assert "uncertainty" in f
            unc = f["uncertainty"]
            for ds_name in _SCALAR_COLUMNS:
                assert ds_name in unc, f"Missing dataset: {ds_name}"
            assert "per_member_volumes" in unc

    def test_shapes_correct(
        self,
        synthetic_h5: Path,
        synthetic_volumes_csv: Path,
    ) -> None:
        n = len(_SCAN_IDS)
        append_uncertainty_group(
            synthetic_h5,
            synthetic_volumes_csv,
            rank=8,
            n_members=_N_MEMBERS,
            seed=42,
        )

        with h5py.File(synthetic_h5, "r") as f:
            unc = f["uncertainty"]
            for ds_name in _SCALAR_COLUMNS:
                assert unc[ds_name].shape == (n,), (
                    f"{ds_name}: expected ({n},), got {unc[ds_name].shape}"
                )
            assert unc["per_member_volumes"].shape == (n, _N_MEMBERS)

    def test_dtypes_float32(
        self,
        synthetic_h5: Path,
        synthetic_volumes_csv: Path,
    ) -> None:
        append_uncertainty_group(
            synthetic_h5,
            synthetic_volumes_csv,
            rank=8,
            n_members=_N_MEMBERS,
            seed=42,
        )

        with h5py.File(synthetic_h5, "r") as f:
            unc = f["uncertainty"]
            for ds_name in list(_SCALAR_COLUMNS) + ["per_member_volumes"]:
                assert unc[ds_name].dtype == np.float32

    def test_attrs_present(
        self,
        synthetic_h5: Path,
        synthetic_volumes_csv: Path,
    ) -> None:
        append_uncertainty_group(
            synthetic_h5,
            synthetic_volumes_csv,
            rank=8,
            n_members=_N_MEMBERS,
            seed=42,
        )

        with h5py.File(synthetic_h5, "r") as f:
            unc = f["uncertainty"]
            assert unc.attrs["n_members"] == _N_MEMBERS
            assert unc.attrs["rank"] == 8
            assert unc.attrs["seed"] == 42
            assert "source_csv" in unc.attrs

            assert f.attrs["uncertainty_rank"] == 8
            assert f.attrs["ensemble_source"] == f"r8_M{_N_MEMBERS}_s42"

    def test_values_finite_nonneg(
        self,
        synthetic_h5: Path,
        synthetic_volumes_csv: Path,
    ) -> None:
        append_uncertainty_group(
            synthetic_h5,
            synthetic_volumes_csv,
            rank=8,
            n_members=_N_MEMBERS,
            seed=42,
        )

        nonneg_datasets = [
            "vol_mean",
            "vol_std",
            "vol_median",
            "vol_mad",
            "mean_entropy",
            "mean_mi",
            "mean_var",
        ]

        with h5py.File(synthetic_h5, "r") as f:
            unc = f["uncertainty"]
            for ds_name in nonneg_datasets:
                data = unc[ds_name][:]
                assert np.all(np.isfinite(data)), f"{ds_name} has non-finite values"
                assert np.all(data >= 0), f"{ds_name} has negative values"

    def test_scan_id_alignment(
        self,
        synthetic_h5: Path,
        synthetic_volumes_csv: Path,
    ) -> None:
        append_uncertainty_group(
            synthetic_h5,
            synthetic_volumes_csv,
            rank=8,
            n_members=_N_MEMBERS,
            seed=42,
        )

        df = pd.read_csv(synthetic_volumes_csv).set_index("scan_id")
        with h5py.File(synthetic_h5, "r") as f:
            scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
            vol_mean = f["uncertainty/vol_mean"][:]

            for i, sid in enumerate(scan_ids):
                expected = df.loc[sid, "vol_mean"]
                np.testing.assert_allclose(
                    vol_mean[i],
                    expected,
                    rtol=1e-5,
                    err_msg=f"vol_mean mismatch for {sid}",
                )

    def test_idempotent_on_rerun(
        self,
        synthetic_h5: Path,
        synthetic_volumes_csv: Path,
    ) -> None:
        for _ in range(2):
            append_uncertainty_group(
                synthetic_h5,
                synthetic_volumes_csv,
                rank=8,
                n_members=_N_MEMBERS,
                seed=42,
            )

        with h5py.File(synthetic_h5, "r") as f:
            assert "uncertainty" in f
            assert len(f["uncertainty/vol_mean"]) == len(_SCAN_IDS)

    def test_raises_on_missing_scan_id(
        self,
        synthetic_h5: Path,
        synthetic_volumes_csv: Path,
    ) -> None:
        # Remove one scan from CSV
        df = pd.read_csv(synthetic_volumes_csv)
        df = df.iloc[1:]  # drop first row
        df.to_csv(synthetic_volumes_csv, index=False)

        with pytest.raises(KeyError, match="scan_ids not in CSV"):
            append_uncertainty_group(
                synthetic_h5,
                synthetic_volumes_csv,
                rank=8,
                n_members=_N_MEMBERS,
                seed=42,
            )

    def test_reindexing_survives_shuffled_csv(
        self,
        synthetic_h5: Path,
        synthetic_volumes_csv: Path,
    ) -> None:
        # Shuffle CSV rows
        df = pd.read_csv(synthetic_volumes_csv)
        df = df.sample(frac=1, random_state=99).reset_index(drop=True)
        df.to_csv(synthetic_volumes_csv, index=False)

        append_uncertainty_group(
            synthetic_h5,
            synthetic_volumes_csv,
            rank=8,
            n_members=_N_MEMBERS,
            seed=42,
        )

        # Verify alignment is still correct
        df_orig = pd.read_csv(synthetic_volumes_csv).set_index("scan_id")
        with h5py.File(synthetic_h5, "r") as f:
            scan_ids = [s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]]
            vol_mean = f["uncertainty/vol_mean"][:]
            for i, sid in enumerate(scan_ids):
                np.testing.assert_allclose(
                    vol_mean[i],
                    df_orig.loc[sid, "vol_mean"],
                    rtol=1e-5,
                )
