# tests/growth/test_bratsmendata.py
"""Tests for BraTS-MEN dataset."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

from growth.data.bratsmendata import (
    BraTSMENDataset,
    create_dataloaders,
    split_subjects,
    save_splits,
    load_splits,
    MODALITY_SUFFIXES,
    SEG_SUFFIX,
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

        # Default transforms produce 96x96x96
        assert sample["image"].shape == torch.Size([4, 96, 96, 96])
        assert sample["seg"].shape == torch.Size([1, 96, 96, 96])

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
            assert sample["image"].shape == torch.Size([4, 96, 96, 96])
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
