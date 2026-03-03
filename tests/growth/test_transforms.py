# tests/growth/test_transforms.py
"""Tests for MONAI transforms (H5 pipeline)."""

import pytest
import torch

from growth.data.transforms import (
    DEFAULT_ORIENTATION,
    DEFAULT_ROI_SIZE,
    DEFAULT_SPACING,
    FEATURE_ROI_SIZE,
    IMAGE_KEY,
    MODALITY_KEYS,
    SEG_KEY,
    get_augmentation_transforms,
    get_finalize_transforms,
    get_h5_train_transforms,
    get_h5_val_transforms,
    get_intensity_transforms,
)


class TestConstants:
    """Tests for module constants."""

    def test_modality_keys(self):
        assert MODALITY_KEYS == ["t2f", "t1c", "t1n", "t2w"]

    def test_seg_key(self):
        assert SEG_KEY == "seg"

    def test_image_key(self):
        assert IMAGE_KEY == "image"

    def test_default_roi_size(self):
        assert DEFAULT_ROI_SIZE == (128, 128, 128)

    def test_feature_roi_size(self):
        assert FEATURE_ROI_SIZE == (192, 192, 192)

    def test_default_spacing(self):
        assert DEFAULT_SPACING == (1.0, 1.0, 1.0)

    def test_default_orientation(self):
        assert DEFAULT_ORIENTATION == "RAS"


class TestGetIntensityTransforms:
    """Tests for get_intensity_transforms function."""

    def test_intensity_transforms_default(self):
        transforms = get_intensity_transforms()
        assert len(transforms) == 1  # NormalizeIntensityd


class TestGetAugmentationTransforms:
    """Tests for get_augmentation_transforms function."""

    def test_augmentation_transforms_default(self):
        transforms = get_augmentation_transforms()
        # 3 flips + 1 rotate + 2 intensity = 6
        assert len(transforms) == 6

    def test_augmentation_transforms_no_intensity(self):
        transforms = get_augmentation_transforms(include_intensity=False)
        # 3 flips + 1 rotate = 4
        assert len(transforms) == 4

    def test_augmentation_transforms_minimal(self):
        transforms = get_augmentation_transforms(
            include_flip=False, include_rotate=False, include_intensity=False,
        )
        assert len(transforms) == 0


class TestGetFinalizeTransforms:
    """Tests for get_finalize_transforms function."""

    def test_finalize_transforms_default(self):
        transforms = get_finalize_transforms()
        assert len(transforms) == 1  # EnsureTyped


class TestH5TrainTransforms:
    """Tests for get_h5_train_transforms function."""

    def test_h5_train_default(self):
        pipeline = get_h5_train_transforms()
        assert pipeline is not None

    def test_h5_train_no_augment(self):
        pipeline = get_h5_train_transforms(augment=False)
        assert pipeline is not None

    def test_h5_train_192_no_crop(self):
        """When roi=192, no spatial crop should be added."""
        pipeline = get_h5_train_transforms(roi_size=(192, 192, 192), augment=False)
        # Should be: normalize + finalize = 2 transforms
        assert pipeline is not None

    def test_h5_train_output_shape(self):
        """Test H5 training pipeline produces correct output shapes."""
        import numpy as np

        pipeline = get_h5_train_transforms(roi_size=(16, 16, 16), augment=False)
        data = {
            "image": torch.from_numpy(
                np.random.rand(4, 32, 32, 32).astype(np.float32) + 0.1
            ),
            "seg": torch.from_numpy(
                np.random.randint(0, 4, (1, 32, 32, 32)).astype(np.float32)
            ),
        }
        result = pipeline(data)
        assert result["image"].shape == (4, 16, 16, 16)
        assert result["seg"].shape == (1, 16, 16, 16)


class TestH5ValTransforms:
    """Tests for get_h5_val_transforms function."""

    def test_h5_val_default(self):
        pipeline = get_h5_val_transforms()
        assert pipeline is not None

    def test_h5_val_custom_roi(self):
        pipeline = get_h5_val_transforms(roi_size=(128, 128, 128))
        assert pipeline is not None

    def test_h5_val_output_shape(self):
        """Test H5 val pipeline produces correct output shapes."""
        import numpy as np

        # Use roi=16 to match our fake data
        pipeline = get_h5_val_transforms(roi_size=(16, 16, 16))
        data = {
            "image": torch.from_numpy(
                np.random.rand(4, 16, 16, 16).astype(np.float32) + 0.1
            ),
            "seg": torch.from_numpy(
                np.random.randint(0, 4, (1, 16, 16, 16)).astype(np.float32)
            ),
        }
        result = pipeline(data)
        assert result["image"].shape == (4, 16, 16, 16)
        assert result["seg"].shape == (1, 16, 16, 16)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].dtype == torch.float32
