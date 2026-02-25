# tests/growth/test_bratsmendata.py
"""Tests for BraTS-MEN dataset (NIfTI and H5 backends)."""

from pathlib import Path

import pytest
import torch

from growth.data.bratsmendata import (
    BraTSMENDataset,
    create_dataloaders,
    load_splits,
    save_splits,
    split_subjects,
)


class TestBraTSMENDatasetDiscovery:
    """Tests for subject discovery and path loading."""

    def test_get_all_subject_ids(self, real_data_path: Path):
        """Test subject discovery returns expected count."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)

        assert len(subject_ids) > 0
        assert all(s.startswith("BraTS-MEN-") for s in subject_ids)
        assert subject_ids == sorted(subject_ids)  # Should be sorted

    def test_load_subject_paths(self, real_data_path: Path):
        """Test path loading returns all expected keys."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)
        if not subject_ids:
            pytest.skip("No subjects found")

        paths = BraTSMENDataset.load_subject_paths(real_data_path, subject_ids[0])

        expected_keys = {"t1c", "t1n", "t2f", "t2w", "seg"}
        assert set(paths.keys()) == expected_keys
        assert all(p.exists() for p in paths.values())

    def test_load_subject_paths_missing(self, tmp_path: Path):
        """Test error on missing files."""
        # Create directory but no files
        subject_id = "BraTS-MEN-00001-000"
        (tmp_path / subject_id).mkdir()

        with pytest.raises(FileNotFoundError):
            BraTSMENDataset.load_subject_paths(tmp_path, subject_id)


class TestBraTSMENDataset:
    """Tests for dataset initialization and data loading."""

    def test_init_with_real_data(self, real_data_path: Path):
        """Test dataset initialization with real data."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)[:3]
        if len(subject_ids) < 1:
            pytest.skip("No subjects found")

        dataset = BraTSMENDataset(
            data_root=real_data_path,
            subject_ids=subject_ids,
            compute_semantic=False,  # Skip semantic for speed
        )

        assert len(dataset) == len(subject_ids)

    def test_init_auto_discover(self, real_data_path: Path):
        """Test dataset auto-discovers subjects when none provided."""
        dataset = BraTSMENDataset(
            data_root=real_data_path,
            subject_ids=None,
            compute_semantic=False,
        )

        all_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)
        assert len(dataset) == len(all_ids)

    def test_init_missing_subject(self, real_data_path: Path):
        """Test error when subject doesn't exist."""
        with pytest.raises(ValueError, match="Missing"):
            BraTSMENDataset(
                data_root=real_data_path,
                subject_ids=["NONEXISTENT-SUBJECT"],
                compute_semantic=False,
            )

    def test_getitem_returns_expected_keys(self, real_data_path: Path):
        """Test __getitem__ returns expected dictionary keys."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)[:1]
        if not subject_ids:
            pytest.skip("No subjects found")

        dataset = BraTSMENDataset(
            data_root=real_data_path,
            subject_ids=subject_ids,
            compute_semantic=True,
        )

        sample = dataset[0]

        assert "image" in sample
        assert "seg" in sample
        assert "subject_id" in sample
        assert "semantic_features" in sample

    def test_getitem_image_shape(self, real_data_path: Path):
        """Test image has correct shape."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)[:1]
        if not subject_ids:
            pytest.skip("No subjects found")

        dataset = BraTSMENDataset(
            data_root=real_data_path,
            subject_ids=subject_ids,
            compute_semantic=False,
        )

        sample = dataset[0]

        # Default transforms produce 128x128x128 (matching BrainSegFounder)
        assert sample["image"].shape == torch.Size([4, 128, 128, 128])
        assert sample["seg"].shape == torch.Size([1, 128, 128, 128])

    def test_getitem_semantic_features(self, real_data_path: Path):
        """Test semantic features have correct shapes."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)[:1]
        if not subject_ids:
            pytest.skip("No subjects found")

        dataset = BraTSMENDataset(
            data_root=real_data_path,
            subject_ids=subject_ids,
            compute_semantic=True,
        )

        sample = dataset[0]
        features = sample["semantic_features"]

        assert features["volume"].shape == torch.Size([4])
        assert features["location"].shape == torch.Size([3])
        assert features["shape"].shape == torch.Size([3])
        assert features["all"].shape == torch.Size([10])

    def test_semantic_caching(self, real_data_path: Path, tmp_path: Path):
        """Test semantic features are cached."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)[:1]
        if not subject_ids:
            pytest.skip("No subjects found")

        cache_dir = tmp_path / "semantic_cache"

        dataset = BraTSMENDataset(
            data_root=real_data_path,
            subject_ids=subject_ids,
            compute_semantic=True,
            cache_semantic=True,
            cache_dir=cache_dir,
        )

        # First access computes and caches
        _ = dataset[0]

        # Check cache file exists
        cache_file = cache_dir / f"{subject_ids[0]}_semantic.npz"
        assert cache_file.exists()

        # Load again - should use cache
        sample = dataset[0]
        assert sample["semantic_features"]["volume"].shape == torch.Size([4])


class TestSplitSubjects:
    """Tests for subject splitting."""

    def test_basic_split(self):
        """Test basic split functionality."""
        subjects = [f"BraTS-MEN-{i:05d}-000" for i in range(100)]

        splits = split_subjects(subjects, train_size=60, val_size=20, test_size=20)

        assert len(splits["train"]) == 60
        assert len(splits["val"]) == 20
        assert len(splits["test"]) == 20

        # No overlap
        all_split = set(splits["train"] + splits["val"] + splits["test"])
        assert len(all_split) == 100

    def test_split_reproducibility(self):
        """Test split is reproducible with same seed."""
        subjects = [f"BraTS-MEN-{i:05d}-000" for i in range(100)]

        splits1 = split_subjects(subjects, train_size=60, val_size=20, test_size=20, seed=42)
        splits2 = split_subjects(subjects, train_size=60, val_size=20, test_size=20, seed=42)

        assert splits1["train"] == splits2["train"]
        assert splits1["val"] == splits2["val"]
        assert splits1["test"] == splits2["test"]

    def test_split_different_seeds(self):
        """Test different seeds produce different splits."""
        subjects = [f"BraTS-MEN-{i:05d}-000" for i in range(100)]

        splits1 = split_subjects(subjects, train_size=60, val_size=20, test_size=20, seed=42)
        splits2 = split_subjects(subjects, train_size=60, val_size=20, test_size=20, seed=123)

        assert splits1["train"] != splits2["train"]

    def test_split_insufficient_subjects(self):
        """Test error when not enough subjects."""
        subjects = [f"BraTS-MEN-{i:05d}-000" for i in range(50)]

        with pytest.raises(ValueError, match="Requested"):
            split_subjects(subjects, train_size=40, val_size=20, test_size=20)


class TestSaveLoadSplits:
    """Tests for split persistence."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Test saving and loading splits."""
        splits = {
            "train": ["BraTS-MEN-00001-000", "BraTS-MEN-00002-000"],
            "val": ["BraTS-MEN-00003-000"],
            "test": ["BraTS-MEN-00004-000", "BraTS-MEN-00005-000"],
        }

        path = tmp_path / "splits.json"
        save_splits(splits, path)

        loaded = load_splits(path)

        assert loaded == splits

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """Test save creates parent directories if needed."""
        splits = {"train": ["BraTS-MEN-00001-000"]}
        path = tmp_path / "nested" / "dir" / "splits.json"

        save_splits(splits, path)

        assert path.exists()


class TestCreateDataloaders:
    """Tests for dataloader creation."""

    def test_create_dataloaders(self, real_data_path: Path):
        """Test dataloader creation."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)
        if len(subject_ids) < 4:
            pytest.skip("Need at least 4 subjects")

        train_ids = subject_ids[:2]
        val_ids = subject_ids[2:4]

        train_loader, val_loader = create_dataloaders(
            data_root=real_data_path,
            train_ids=train_ids,
            val_ids=val_ids,
            batch_size=1,
            num_workers=0,  # Avoid multiprocessing in tests
            compute_semantic=False,
        )

        # Check loaders work
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        assert train_batch["image"].shape[0] == 1  # Batch size
        assert val_batch["image"].shape[0] == 1


class TestDataLoaderIntegration:
    """Integration tests with real data pipeline."""

    def test_full_pipeline_no_errors(self, real_data_path: Path):
        """Test loading multiple samples without errors."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)[:5]
        if len(subject_ids) < 5:
            pytest.skip("Need at least 5 subjects")

        dataset = BraTSMENDataset(
            data_root=real_data_path,
            subject_ids=subject_ids,
            compute_semantic=True,
        )

        # Load all samples
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["image"].shape == torch.Size([4, 128, 128, 128])
            assert not torch.isnan(sample["image"]).any()
            assert not torch.isinf(sample["image"]).any()

    def test_semantic_features_reasonable_values(self, real_data_path: Path):
        """Test semantic features have reasonable values."""
        subject_ids = BraTSMENDataset.get_all_subject_ids(real_data_path)[:3]
        if not subject_ids:
            pytest.skip("No subjects found")

        dataset = BraTSMENDataset(
            data_root=real_data_path,
            subject_ids=subject_ids,
            compute_semantic=True,
        )

        sample = dataset[0]
        features = sample["semantic_features"]

        # Volume should be positive (log1p(x) >= 0)
        assert (features["volume"] >= 0).all()

        # Location should be in [0, 1]
        assert (features["location"] >= 0).all()
        assert (features["location"] <= 1).all()

        # Shape sphericity/solidity should be in [0, 1]
        assert 0 <= features["shape"][0] <= 1  # sphericity
        assert 0 <= features["shape"][2] <= 1  # solidity


# =========================================================================
# HDF5 Backend Tests
# =========================================================================


def _create_test_h5(path: Path, n_subjects: int = 10, roi: int = 16) -> Path:
    """Create a small test H5 file with synthetic data.

    Args:
        path: Directory to create file in.
        n_subjects: Number of synthetic subjects.
        roi: Spatial size (small for fast tests).

    Returns:
        Path to created H5 file.
    """
    import h5py
    import numpy as np

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
        splits_grp.create_dataset("lora_train", data=indices[:5])
        splits_grp.create_dataset("lora_val", data=indices[5:7])
        splits_grp.create_dataset("sdp_train", data=indices[7:9])
        splits_grp.create_dataset("test", data=indices[9:])

    return h5_path


@pytest.fixture
def h5_fixture(tmp_path: Path) -> Path:
    """Create a small test H5 file."""
    return _create_test_h5(tmp_path, n_subjects=10, roi=16)


class TestBraTSMENDatasetH5:
    """Tests for the HDF5-backed dataset."""

    def test_init_all_subjects(self, h5_fixture: Path):
        """Test initialization loads all subjects when no split given."""
        from growth.data.bratsmendata import BraTSMENDatasetH5

        dataset = BraTSMENDatasetH5(
            h5_path=h5_fixture,
            compute_semantic=False,
        )
        assert len(dataset) == 10

    def test_init_with_split(self, h5_fixture: Path):
        """Test initialization with a named split."""
        from growth.data.bratsmendata import BraTSMENDatasetH5

        dataset = BraTSMENDatasetH5(
            h5_path=h5_fixture,
            split="lora_train",
            compute_semantic=False,
        )
        assert len(dataset) == 5

    def test_init_invalid_split(self, h5_fixture: Path):
        """Test error on invalid split name."""
        from growth.data.bratsmendata import BraTSMENDatasetH5

        with pytest.raises(KeyError, match="nonexistent"):
            BraTSMENDatasetH5(
                h5_path=h5_fixture,
                split="nonexistent",
            )

    def test_init_with_indices(self, h5_fixture: Path):
        """Test initialization with explicit indices."""
        import numpy as np

        from growth.data.bratsmendata import BraTSMENDatasetH5

        dataset = BraTSMENDatasetH5(
            h5_path=h5_fixture,
            indices=np.array([0, 2, 4]),
            compute_semantic=False,
        )
        assert len(dataset) == 3

    def test_getitem_returns_expected_keys(self, h5_fixture: Path):
        """Test __getitem__ returns expected dictionary keys."""
        from growth.data.bratsmendata import BraTSMENDatasetH5

        dataset = BraTSMENDatasetH5(
            h5_path=h5_fixture,
            compute_semantic=True,
        )

        sample = dataset[0]

        assert "image" in sample
        assert "seg" in sample
        assert "subject_id" in sample
        assert "semantic_features" in sample

    def test_getitem_image_shape(self, h5_fixture: Path):
        """Test image has correct shape (no crop since roi=16 matches H5)."""
        from growth.data.bratsmendata import BraTSMENDatasetH5
        from growth.data.transforms import get_h5_val_transforms

        # Use 16³ val transforms (no-op crop since H5 already 16³)
        dataset = BraTSMENDatasetH5(
            h5_path=h5_fixture,
            transform=get_h5_val_transforms(roi_size=(16, 16, 16)),
            compute_semantic=False,
        )

        sample = dataset[0]

        # H5 fixture has roi=16
        assert sample["image"].shape == torch.Size([4, 16, 16, 16])
        assert sample["seg"].shape == torch.Size([1, 16, 16, 16])

    def test_getitem_semantic_features_shapes(self, h5_fixture: Path):
        """Test semantic features have correct shapes."""
        from growth.data.bratsmendata import BraTSMENDatasetH5

        dataset = BraTSMENDatasetH5(
            h5_path=h5_fixture,
            compute_semantic=True,
        )

        sample = dataset[0]
        features = sample["semantic_features"]

        assert features["volume"].shape == torch.Size([4])
        assert features["location"].shape == torch.Size([3])
        assert features["shape"].shape == torch.Size([3])
        assert features["all"].shape == torch.Size([10])

    def test_subject_ids_property(self, h5_fixture: Path):
        """Test subject_ids property returns correct IDs for split."""
        from growth.data.bratsmendata import BraTSMENDatasetH5

        dataset = BraTSMENDatasetH5(
            h5_path=h5_fixture,
            split="lora_train",
            compute_semantic=False,
        )

        ids = dataset.subject_ids
        assert len(ids) == 5
        assert all(isinstance(s, str) for s in ids)
        assert all(s.startswith("BraTS-MEN-") for s in ids)

    def test_load_splits_from_h5(self, h5_fixture: Path):
        """Test static method for loading splits."""
        from growth.data.bratsmendata import BraTSMENDatasetH5

        splits = BraTSMENDatasetH5.load_splits_from_h5(h5_fixture)

        assert "lora_train" in splits
        assert "lora_val" in splits
        assert "sdp_train" in splits
        assert "test" in splits
        assert len(splits["lora_train"]) == 5
        assert len(splits["lora_val"]) == 2

    def test_load_subject_ids_from_h5(self, h5_fixture: Path):
        """Test static method for loading subject IDs."""
        from growth.data.bratsmendata import BraTSMENDatasetH5

        ids = BraTSMENDatasetH5.load_subject_ids_from_h5(h5_fixture)

        assert len(ids) == 10
        assert ids[0] == "BraTS-MEN-00000-000"

    def test_no_nans_or_infs(self, h5_fixture: Path):
        """Test loaded data has no NaN or Inf values."""
        from growth.data.bratsmendata import BraTSMENDatasetH5
        from growth.data.transforms import get_h5_val_transforms

        dataset = BraTSMENDatasetH5(
            h5_path=h5_fixture,
            transform=get_h5_val_transforms(roi_size=(16, 16, 16)),
            compute_semantic=False,
        )

        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            assert not torch.isnan(sample["image"]).any(), f"NaN in image {i}"
            assert not torch.isinf(sample["image"]).any(), f"Inf in image {i}"


class TestCreateDataloadersH5:
    """Tests for H5-backed dataloader creation."""

    def test_create_dataloaders_h5(self, h5_fixture: Path):
        """Test creating dataloaders from H5 file."""

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

    def test_h5_takes_precedence(self, h5_fixture: Path):
        """Test H5 backend is used when both h5_path and data_root are set."""
        train_loader, val_loader = create_dataloaders(
            data_root="/nonexistent/path",  # Would fail for NIfTI
            h5_path=h5_fixture,  # H5 should take precedence
            batch_size=1,
            num_workers=0,
            compute_semantic=False,
            roi_size=(16, 16, 16),
        )

        # If H5 was used, this should work without error
        batch = next(iter(train_loader))
        assert batch["image"].ndim == 5
