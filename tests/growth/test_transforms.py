# tests/growth/test_transforms.py
"""Tests for MONAI transforms."""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from growth.data.transforms import (
    DEFAULT_ORIENTATION,
    DEFAULT_ROI_SIZE,
    DEFAULT_SPACING,
    IMAGE_KEY,
    MODALITY_KEYS,
    SEG_KEY,
    get_augmentation_transforms,
    get_concat_transforms,
    get_finalize_transforms,
    get_inference_transforms,
    get_intensity_transforms,
    get_load_transforms,
    get_reorient_and_spacing_transforms,
    get_sliding_window_transforms,
    get_spatial_transforms,
    get_train_transforms,
    get_val_transforms,
)


class TestConstants:
    """Tests for module constants."""

    def test_modality_keys(self):
        """Test expected modality keys."""
        assert MODALITY_KEYS == ["t1c", "t1n", "t2f", "t2w"]

    def test_seg_key(self):
        """Test segmentation key."""
        assert SEG_KEY == "seg"

    def test_image_key(self):
        """Test image key."""
        assert IMAGE_KEY == "image"

    def test_default_roi_size(self):
        """Test default ROI size."""
        assert DEFAULT_ROI_SIZE == (128, 128, 128)

    def test_default_spacing(self):
        """Test default spacing."""
        assert DEFAULT_SPACING == (1.0, 1.0, 1.0)

    def test_default_orientation(self):
        """Test default orientation."""
        assert DEFAULT_ORIENTATION == "RAS"


class TestGetLoadTransforms:
    """Tests for get_load_transforms function."""

    def test_load_transforms_default(self):
        """Test load transforms with defaults."""
        transforms = get_load_transforms()

        assert len(transforms) == 2  # LoadImaged + EnsureChannelFirstd

    def test_load_transforms_no_seg(self):
        """Test load transforms without segmentation."""
        transforms = get_load_transforms(include_seg=False)

        assert len(transforms) == 2

    def test_load_transforms_custom_keys(self):
        """Test load transforms with custom modality keys."""
        transforms = get_load_transforms(modality_keys=["t1c", "t2w"])

        assert len(transforms) == 2


class TestGetSpatialTransforms:
    """Tests for get_spatial_transforms function."""

    def test_spatial_transforms_default(self):
        """Test spatial transforms with defaults (CropForeground pipeline)."""
        transforms = get_spatial_transforms()

        # Orientationd, Spacingd (modalities), Spacingd (seg),
        # CropForegroundd, SpatialPadd, ResizeWithPadOrCropd
        assert len(transforms) == 6

    def test_spatial_transforms_no_seg(self):
        """Test spatial transforms without segmentation."""
        transforms = get_spatial_transforms(include_seg=False)

        # Orientationd, Spacingd (modalities),
        # CropForegroundd, SpatialPadd, ResizeWithPadOrCropd
        assert len(transforms) == 5

    def test_spatial_transforms_random_crop(self):
        """Test spatial transforms with random crop (training mode)."""
        transforms = get_spatial_transforms(random_crop=True)

        # Orientationd, Spacingd (modalities), Spacingd (seg),
        # CropForegroundd, SpatialPadd, RandSpatialCropd
        assert len(transforms) == 6


class TestGetReorientAndSpacingTransforms:
    """Tests for get_reorient_and_spacing_transforms function."""

    def test_reorient_spacing_default(self):
        """Test reorient+spacing transforms with defaults (with seg)."""
        transforms = get_reorient_and_spacing_transforms()

        # Orientationd, Spacingd (modalities), Spacingd (seg)
        assert len(transforms) == 3

    def test_reorient_spacing_no_seg(self):
        """Test reorient+spacing transforms without segmentation."""
        transforms = get_reorient_and_spacing_transforms(include_seg=False)

        # Orientationd, Spacingd (modalities)
        assert len(transforms) == 2


class TestGetSlidingWindowTransforms:
    """Tests for get_sliding_window_transforms function."""

    def test_sliding_window_transforms(self):
        """Test sliding window transforms pipeline creation."""
        pipeline = get_sliding_window_transforms()

        assert pipeline is not None

    def test_sliding_window_no_seg(self):
        """Test sliding window transforms without segmentation."""
        pipeline = get_sliding_window_transforms(include_seg=False)

        assert pipeline is not None


class TestGetIntensityTransforms:
    """Tests for get_intensity_transforms function."""

    def test_intensity_transforms_default(self):
        """Test intensity transforms with defaults."""
        transforms = get_intensity_transforms()

        assert len(transforms) == 1  # NormalizeIntensityd


class TestGetConcatTransforms:
    """Tests for get_concat_transforms function."""

    def test_concat_transforms_default(self):
        """Test concat transforms with defaults."""
        transforms = get_concat_transforms()

        assert len(transforms) == 1  # ConcatItemsd


class TestGetAugmentationTransforms:
    """Tests for get_augmentation_transforms function."""

    def test_augmentation_transforms_default(self):
        """Test augmentation transforms with defaults."""
        transforms = get_augmentation_transforms()

        # 3 flips + 1 rotate + 2 intensity = 6
        assert len(transforms) == 6

    def test_augmentation_transforms_no_intensity(self):
        """Test augmentation transforms without intensity augmentation."""
        transforms = get_augmentation_transforms(include_intensity=False)

        # 3 flips + 1 rotate = 4
        assert len(transforms) == 4

    def test_augmentation_transforms_minimal(self):
        """Test minimal augmentation transforms."""
        transforms = get_augmentation_transforms(
            include_flip=False,
            include_rotate=False,
            include_intensity=False,
        )

        assert len(transforms) == 0


class TestGetFinalizeTransforms:
    """Tests for get_finalize_transforms function."""

    def test_finalize_transforms_default(self):
        """Test finalize transforms with defaults."""
        transforms = get_finalize_transforms()

        assert len(transforms) == 1  # EnsureTyped


class TestGetTrainTransforms:
    """Tests for get_train_transforms function."""

    def test_train_transforms_default(self):
        """Test train transforms with defaults (augmentation enabled)."""
        pipeline = get_train_transforms()

        assert pipeline is not None

    def test_train_transforms_no_augment(self):
        """Test train transforms without augmentation."""
        pipeline = get_train_transforms(augment=False)

        assert pipeline is not None


class TestGetValTransforms:
    """Tests for get_val_transforms function."""

    def test_val_transforms(self):
        """Test validation transforms."""
        pipeline = get_val_transforms()

        assert pipeline is not None


class TestGetInferenceTransforms:
    """Tests for get_inference_transforms function."""

    def test_inference_transforms(self):
        """Test inference transforms (no segmentation)."""
        pipeline = get_inference_transforms()

        assert pipeline is not None


@pytest.fixture
def synthetic_nifti_data(tmp_path: Path):
    """Create synthetic NIfTI files for testing."""
    # Create random 3D volumes (small for testing)
    shape = (64, 64, 64)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([2.0, 2.0, 2.0])  # 2mm spacing

    data_dict = {}

    # Create modality images
    for modality in MODALITY_KEYS:
        img_data = np.random.randn(*shape).astype(np.float32)
        img = nib.Nifti1Image(img_data, affine)
        path = tmp_path / f"{modality}.nii.gz"
        nib.save(img, path)
        data_dict[modality] = str(path)

    # Create segmentation (integer labels)
    seg_data = np.random.randint(0, 4, shape).astype(np.uint8)
    seg_img = nib.Nifti1Image(seg_data, affine)
    seg_path = tmp_path / "seg.nii.gz"
    nib.save(seg_img, seg_path)
    data_dict[SEG_KEY] = str(seg_path)

    return data_dict


class TestTransformPipelines:
    """Integration tests for complete transform pipelines."""

    def test_train_pipeline_output_shape(self, synthetic_nifti_data):
        """Test that training pipeline produces correct output shapes."""
        pipeline = get_train_transforms(augment=False)  # Deterministic for testing

        result = pipeline(synthetic_nifti_data)

        # Image should be [4, 128, 128, 128] (4 modalities concatenated)
        assert IMAGE_KEY in result
        assert result[IMAGE_KEY].shape == (4, 128, 128, 128)

        # Segmentation should be [1, 128, 128, 128]
        assert SEG_KEY in result
        assert result[SEG_KEY].shape == (1, 128, 128, 128)

    def test_val_pipeline_output_shape(self, synthetic_nifti_data):
        """Test that validation pipeline produces correct output shapes."""
        pipeline = get_val_transforms()

        result = pipeline(synthetic_nifti_data)

        assert result[IMAGE_KEY].shape == (4, 128, 128, 128)
        assert result[SEG_KEY].shape == (1, 128, 128, 128)

    def test_inference_pipeline_output_shape(self, synthetic_nifti_data):
        """Test that inference pipeline produces correct output shapes."""
        # Remove seg from data for inference
        inference_data = {k: v for k, v in synthetic_nifti_data.items() if k != SEG_KEY}

        pipeline = get_inference_transforms()

        result = pipeline(inference_data)

        assert result[IMAGE_KEY].shape == (4, 128, 128, 128)
        assert SEG_KEY not in result

    def test_train_pipeline_dtype(self, synthetic_nifti_data):
        """Test that outputs are float32 tensors."""
        pipeline = get_train_transforms(augment=False)

        result = pipeline(synthetic_nifti_data)

        assert isinstance(result[IMAGE_KEY], torch.Tensor)
        assert result[IMAGE_KEY].dtype == torch.float32
        assert isinstance(result[SEG_KEY], torch.Tensor)

    def test_custom_roi_size(self, synthetic_nifti_data):
        """Test pipeline with custom ROI size."""
        pipeline = get_train_transforms(
            roi_size=(64, 64, 64),
            augment=False,
        )

        result = pipeline(synthetic_nifti_data)

        assert result[IMAGE_KEY].shape == (4, 64, 64, 64)

    def test_custom_modalities(self, synthetic_nifti_data):
        """Test pipeline with subset of modalities."""
        # Only use t1c and t2w
        subset_data = {
            "t1c": synthetic_nifti_data["t1c"],
            "t2w": synthetic_nifti_data["t2w"],
            SEG_KEY: synthetic_nifti_data[SEG_KEY],
        }

        pipeline = get_train_transforms(
            modality_keys=["t1c", "t2w"],
            augment=False,
        )

        result = pipeline(subset_data)

        # Should have 2 channels
        assert result[IMAGE_KEY].shape == (2, 128, 128, 128)


class TestTransformPipelinesReal:
    """Tests with real BraTS-MEN data (skipped if unavailable)."""

    def test_real_data_train_pipeline(self, real_data_path: Path):
        """Test training pipeline with real data."""
        # Find a subject
        subjects = list(real_data_path.iterdir())
        if not subjects:
            pytest.skip("No subjects found in data directory")

        subject = subjects[0]

        # Build data dict
        data = {}
        for modality in MODALITY_KEYS:
            files = list(subject.glob(f"*-{modality}.nii.gz"))
            if not files:
                pytest.skip(f"Missing {modality} file for subject {subject.name}")
            data[modality] = str(files[0])

        seg_files = list(subject.glob("*-seg.nii.gz"))
        if seg_files:
            data[SEG_KEY] = str(seg_files[0])
        else:
            pytest.skip(f"Missing seg file for subject {subject.name}")

        pipeline = get_train_transforms(augment=False)

        result = pipeline(data)

        # Check output shapes
        assert result[IMAGE_KEY].shape == (4, 128, 128, 128)
        assert result[SEG_KEY].shape == (1, 128, 128, 128)

        # Check data is not all zeros
        assert result[IMAGE_KEY].abs().sum() > 0

    def test_real_data_val_pipeline(self, real_data_path: Path):
        """Test validation pipeline with real data."""
        subjects = list(real_data_path.iterdir())
        if not subjects:
            pytest.skip("No subjects found in data directory")

        subject = subjects[0]

        data = {}
        for modality in MODALITY_KEYS:
            files = list(subject.glob(f"*-{modality}.nii.gz"))
            if not files:
                pytest.skip(f"Missing {modality} file")
            data[modality] = str(files[0])

        seg_files = list(subject.glob("*-seg.nii.gz"))
        if seg_files:
            data[SEG_KEY] = str(seg_files[0])
        else:
            pytest.skip("Missing seg file")

        pipeline = get_val_transforms()

        result = pipeline(data)

        assert result[IMAGE_KEY].shape == (4, 128, 128, 128)
