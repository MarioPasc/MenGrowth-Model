# tests/growth/test_bratsmendata.py
"""Tests for BraTS dataset (H5 backend)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from growth.data.bratsmendata import (
    BraTSDatasetH5,
    BraTSMENDatasetH5,
    create_dataloaders,
    split_subjects_multi,
)


# =========================================================================
# Shared Utilities
# =========================================================================


def _create_test_h5(path: Path, n_subjects: int = 10, roi: int = 16) -> Path:
    """Create a small test H5 file with synthetic cross-sectional data.

    Args:
        path: Directory to create file in.
        n_subjects: Number of synthetic subjects.
        roi: Spatial size (small for fast tests).

    Returns:
        Path to created H5 file.
    """
    import h5py

    h5_path = path / "test_brats.h5"
    rng = np.random.RandomState(42)

    subject_ids = [f"BraTS-MEN-{i:05d}-000" for i in range(n_subjects)]

    with h5py.File(h5_path, "w") as f:
        # Attributes
        f.attrs["n_subjects"] = n_subjects
        f.attrs["roi_size"] = [roi, roi, roi]
        f.attrs["spacing"] = [1.0, 1.0, 1.0]
        f.attrs["channel_order"] = ["t2f", "t1c", "t1n", "t2w"]
        f.attrs["version"] = "1.0"

        # Images: random float32 volumes (non-zero so normalization works)
        images = rng.rand(n_subjects, 4, roi, roi, roi).astype(np.float32) + 0.1
        f.create_dataset("images", data=images)

        # Segs: random labels {0, 1, 2, 3}
        segs = rng.randint(0, 4, size=(n_subjects, 1, roi, roi, roi)).astype(np.int8)
        f.create_dataset("segs", data=segs)

        # Subject IDs
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("subject_ids", data=subject_ids, dtype=dt)

        # Semantic features
        sem_grp = f.create_group("semantic")
        sem_grp.create_dataset("volume", data=rng.rand(n_subjects, 4).astype(np.float32))
        sem_grp.create_dataset("location", data=rng.rand(n_subjects, 3).astype(np.float32))
        sem_grp.create_dataset("shape", data=rng.rand(n_subjects, 3).astype(np.float32))

        # Metadata
        meta_grp = f.create_group("metadata")
        meta_grp.create_dataset("grade", data=np.ones(n_subjects, dtype=np.int8))
        meta_grp.create_dataset("age", data=rng.uniform(30, 80, n_subjects).astype(np.float32))
        meta_grp.create_dataset(
            "sex", data=["M"] * (n_subjects // 2) + ["F"] * (n_subjects - n_subjects // 2), dtype=dt
        )

        # Splits
        indices = np.arange(n_subjects, dtype=np.int32)
        splits_grp = f.create_group("splits")
        splits_grp.create_dataset("lora_train", data=indices[:7])
        splits_grp.create_dataset("lora_val", data=indices[7:9])
        splits_grp.create_dataset("test", data=indices[9:])

    return h5_path


def _create_test_longitudinal_h5(
    path: Path,
    n_patients: int = 5,
    max_timepoints: int = 3,
    roi: int = 16,
) -> Path:
    """Create a small test H5 file with synthetic longitudinal (GLI) data.

    Args:
        path: Directory to create file in.
        n_patients: Number of patients.
        max_timepoints: Max timepoints per patient.
        roi: Spatial size.

    Returns:
        Path to created H5 file.
    """
    import h5py

    h5_path = path / "test_brats_gli.h5"
    rng = np.random.RandomState(42)

    # Build scan structure: patient i has (i % max_timepoints + 1) timepoints
    scan_ids = []
    patient_ids_per_scan = []
    timepoint_indices = []
    patient_list = []
    patient_offsets = [0]

    scan_idx = 0
    for p in range(n_patients):
        pid = f"BraTS-GLI-{p:05d}"
        patient_list.append(pid)
        n_tp = (p % max_timepoints) + 1
        for tp in range(n_tp):
            scan_ids.append(f"{pid}-{100 + tp}")
            patient_ids_per_scan.append(pid)
            timepoint_indices.append(tp)
            scan_idx += 1
        patient_offsets.append(scan_idx)

    n_scans = len(scan_ids)

    with h5py.File(h5_path, "w") as f:
        # Attributes
        f.attrs["n_scans"] = n_scans
        f.attrs["n_patients"] = n_patients
        f.attrs["roi_size"] = [roi, roi, roi]
        f.attrs["spacing"] = [1.0, 1.0, 1.0]
        f.attrs["channel_order"] = ["t2f", "t1c", "t1n", "t2w"]
        f.attrs["version"] = "2.0"
        f.attrs["dataset_type"] = "longitudinal"
        f.attrs["domain"] = "GLI"

        # Images and segs
        images = rng.rand(n_scans, 4, roi, roi, roi).astype(np.float32) + 0.1
        f.create_dataset("images", data=images)
        segs = rng.randint(0, 5, size=(n_scans, 1, roi, roi, roi)).astype(np.int8)
        f.create_dataset("segs", data=segs)

        # IDs
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("scan_ids", data=scan_ids, dtype=dt)
        f.create_dataset("patient_ids", data=patient_ids_per_scan, dtype=dt)
        f.create_dataset("timepoint_idx", data=np.array(timepoint_indices, dtype=np.int32))

        # Longitudinal group
        long_grp = f.create_group("longitudinal")
        long_grp.create_dataset("patient_offsets", data=np.array(patient_offsets, dtype=np.int32))
        long_grp.create_dataset("patient_list", data=patient_list, dtype=dt)

        # Semantic features
        sem_grp = f.create_group("semantic")
        sem_grp.create_dataset("volume", data=rng.rand(n_scans, 4).astype(np.float32))
        sem_grp.create_dataset("location", data=rng.rand(n_scans, 3).astype(np.float32))
        sem_grp.create_dataset("shape", data=rng.rand(n_scans, 3).astype(np.float32))

        # Patient-level splits (indices into patient_list)
        patient_indices = np.arange(n_patients, dtype=np.int32)
        splits_grp = f.create_group("splits")
        splits_grp.create_dataset("lora_train", data=patient_indices[:3])
        splits_grp.create_dataset("lora_val", data=patient_indices[3:4])
        splits_grp.create_dataset("test", data=patient_indices[4:])

        # Metadata
        meta_grp = f.create_group("metadata")
        meta_grp.create_dataset("grade", data=rng.randint(1, 5, n_scans).astype(np.int8))
        meta_grp.create_dataset("age", data=rng.uniform(30, 80, n_scans).astype(np.float32))
        meta_grp.create_dataset("sex", data=["M"] * n_scans, dtype=dt)

    return h5_path


@pytest.fixture
def h5_fixture(tmp_path: Path) -> Path:
    """Create a small test H5 file (cross-sectional)."""
    return _create_test_h5(tmp_path, n_subjects=10, roi=16)


@pytest.fixture
def longitudinal_h5_fixture(tmp_path: Path) -> Path:
    """Create a small test longitudinal H5 file."""
    return _create_test_longitudinal_h5(tmp_path, n_patients=5, max_timepoints=3, roi=16)


# =========================================================================
# Split Utilities Tests
# =========================================================================


class TestSplitSubjectsMulti:
    """Tests for split_subjects_multi."""

    def test_basic_split(self):
        subjects = [f"BraTS-MEN-{i:05d}-000" for i in range(100)]
        splits = split_subjects_multi(
            subjects, {"train": 60, "val": 20, "test": 20}, seed=42
        )
        assert len(splits["train"]) == 60
        assert len(splits["val"]) == 20
        assert len(splits["test"]) == 20

        # No overlap
        all_split = set(splits["train"] + splits["val"] + splits["test"])
        assert len(all_split) == 100

    def test_reproducibility(self):
        subjects = [f"BraTS-MEN-{i:05d}-000" for i in range(100)]
        s1 = split_subjects_multi(subjects, {"a": 50, "b": 50}, seed=42)
        s2 = split_subjects_multi(subjects, {"a": 50, "b": 50}, seed=42)
        assert s1["a"] == s2["a"]
        assert s1["b"] == s2["b"]

    def test_insufficient_subjects(self):
        subjects = [f"BraTS-MEN-{i:05d}-000" for i in range(10)]
        with pytest.raises(ValueError, match="Requested"):
            split_subjects_multi(subjects, {"a": 8, "b": 5})


# =========================================================================
# Cross-Sectional H5 Tests
# =========================================================================


class TestBraTSDatasetH5:
    """Tests for the HDF5-backed dataset (cross-sectional schema)."""

    def test_init_all_subjects(self, h5_fixture: Path):
        dataset = BraTSDatasetH5(h5_path=h5_fixture, compute_semantic=False)
        assert len(dataset) == 10

    def test_init_with_split(self, h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=h5_fixture, split="lora_train", compute_semantic=False
        )
        assert len(dataset) == 7

    def test_init_invalid_split(self, h5_fixture: Path):
        with pytest.raises(KeyError, match="nonexistent"):
            BraTSDatasetH5(h5_path=h5_fixture, split="nonexistent")

    def test_init_with_indices(self, h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=h5_fixture, indices=np.array([0, 2, 4]), compute_semantic=False
        )
        assert len(dataset) == 3

    def test_getitem_returns_expected_keys(self, h5_fixture: Path):
        dataset = BraTSDatasetH5(h5_path=h5_fixture, compute_semantic=True)
        sample = dataset[0]
        assert "image" in sample
        assert "seg" in sample
        assert "subject_id" in sample
        assert "domain" in sample
        assert "semantic_features" in sample

    def test_getitem_image_shape(self, h5_fixture: Path):
        from growth.data.transforms import get_h5_val_transforms

        dataset = BraTSDatasetH5(
            h5_path=h5_fixture,
            transform=get_h5_val_transforms(roi_size=(16, 16, 16)),
            compute_semantic=False,
        )
        sample = dataset[0]
        assert sample["image"].shape == torch.Size([4, 16, 16, 16])
        assert sample["seg"].shape == torch.Size([1, 16, 16, 16])

    def test_getitem_semantic_features_shapes(self, h5_fixture: Path):
        dataset = BraTSDatasetH5(h5_path=h5_fixture, compute_semantic=True)
        sample = dataset[0]
        features = sample["semantic_features"]
        assert features["volume"].shape == torch.Size([4])
        assert features["location"].shape == torch.Size([3])
        assert features["shape"].shape == torch.Size([3])
        assert features["all"].shape == torch.Size([10])

    def test_subject_ids_property(self, h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=h5_fixture, split="lora_train", compute_semantic=False
        )
        ids = dataset.subject_ids
        assert len(ids) == 7
        assert all(isinstance(s, str) for s in ids)
        assert all(s.startswith("BraTS-MEN-") for s in ids)

    def test_domain_default(self, h5_fixture: Path):
        dataset = BraTSDatasetH5(h5_path=h5_fixture, compute_semantic=False)
        assert dataset.domain == "MEN"
        sample = dataset[0]
        assert sample["domain"] == "MEN"

    def test_is_not_longitudinal(self, h5_fixture: Path):
        dataset = BraTSDatasetH5(h5_path=h5_fixture, compute_semantic=False)
        assert not dataset.is_longitudinal

    def test_load_splits_from_h5(self, h5_fixture: Path):
        splits = BraTSDatasetH5.load_splits_from_h5(h5_fixture)
        assert "lora_train" in splits
        assert "lora_val" in splits
        assert "test" in splits
        assert len(splits["lora_train"]) == 7
        assert len(splits["lora_val"]) == 2

    def test_load_subject_ids_from_h5(self, h5_fixture: Path):
        ids = BraTSDatasetH5.load_subject_ids_from_h5(h5_fixture)
        assert len(ids) == 10
        assert ids[0] == "BraTS-MEN-00000-000"

    def test_no_nans_or_infs(self, h5_fixture: Path):
        from growth.data.transforms import get_h5_val_transforms

        dataset = BraTSDatasetH5(
            h5_path=h5_fixture,
            transform=get_h5_val_transforms(roi_size=(16, 16, 16)),
            compute_semantic=False,
        )
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            assert not torch.isnan(sample["image"]).any(), f"NaN in image {i}"
            assert not torch.isinf(sample["image"]).any(), f"Inf in image {i}"

    def test_backward_compat_alias(self):
        """BraTSMENDatasetH5 should be an alias for BraTSDatasetH5."""
        assert BraTSMENDatasetH5 is BraTSDatasetH5


# =========================================================================
# Longitudinal H5 Tests
# =========================================================================


class TestBraTSDatasetH5Longitudinal:
    """Tests for the HDF5-backed dataset (longitudinal schema)."""

    def test_init_all_scans(self, longitudinal_h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=longitudinal_h5_fixture, compute_semantic=False
        )
        # 5 patients: 1 + 2 + 3 + 1 + 2 = 9 scans
        assert len(dataset) == 9

    def test_is_longitudinal(self, longitudinal_h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=longitudinal_h5_fixture, compute_semantic=False
        )
        assert dataset.is_longitudinal

    def test_domain(self, longitudinal_h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=longitudinal_h5_fixture, compute_semantic=False
        )
        assert dataset.domain == "GLI"

    def test_split_expands_to_scans(self, longitudinal_h5_fixture: Path):
        """Patient-level split indices should expand to scan-level indices."""
        # lora_train = patients [0, 1, 2]
        # Patient 0: 1 tp, Patient 1: 2 tp, Patient 2: 3 tp = 6 scans
        dataset = BraTSDatasetH5(
            h5_path=longitudinal_h5_fixture,
            split="lora_train",
            compute_semantic=False,
        )
        assert len(dataset) == 6

    def test_split_test(self, longitudinal_h5_fixture: Path):
        """Test split (patient 4: 2 timepoints)."""
        dataset = BraTSDatasetH5(
            h5_path=longitudinal_h5_fixture,
            split="test",
            compute_semantic=False,
        )
        assert len(dataset) == 2

    def test_getitem_has_patient_id(self, longitudinal_h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=longitudinal_h5_fixture, compute_semantic=False
        )
        sample = dataset[0]
        assert "patient_id" in sample
        assert sample["patient_id"].startswith("BraTS-GLI-")

    def test_getitem_has_domain(self, longitudinal_h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=longitudinal_h5_fixture, compute_semantic=False
        )
        sample = dataset[0]
        assert sample["domain"] == "GLI"

    def test_subject_ids_are_scan_ids(self, longitudinal_h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=longitudinal_h5_fixture, compute_semantic=False
        )
        ids = dataset.subject_ids
        assert len(ids) == 9
        # Should be scan IDs like "BraTS-GLI-00000-100"
        assert all("-" in sid for sid in ids)

    def test_load_subject_ids_returns_scan_ids(self, longitudinal_h5_fixture: Path):
        ids = BraTSDatasetH5.load_subject_ids_from_h5(longitudinal_h5_fixture)
        assert len(ids) == 9
        assert ids[0] == "BraTS-GLI-00000-100"

    def test_semantic_features(self, longitudinal_h5_fixture: Path):
        dataset = BraTSDatasetH5(
            h5_path=longitudinal_h5_fixture, compute_semantic=True
        )
        sample = dataset[0]
        assert "semantic_features" in sample
        assert sample["semantic_features"]["all"].shape == torch.Size([10])


# =========================================================================
# Dataloader Tests
# =========================================================================


class TestCreateDataloadersH5:
    """Tests for H5-backed dataloader creation."""

    def test_create_dataloaders_h5(self, h5_fixture: Path):
        train_loader, val_loader = create_dataloaders(
            h5_path=h5_fixture,
            batch_size=2,
            num_workers=0,
            compute_semantic=True,
            train_split="lora_train",
            val_split="lora_val",
            roi_size=(16, 16, 16),
        )

        # Check train loader
        train_batch = next(iter(train_loader))
        assert train_batch["image"].shape[0] == 2  # batch size
        assert train_batch["image"].ndim == 5  # [B, 4, D, H, W]
        assert "semantic_features" in train_batch

        # Check val loader
        val_batch = next(iter(val_loader))
        assert val_batch["image"].ndim == 5
