# tests/growth/test_semantic_features.py
"""Tests for semantic feature extraction from segmentation masks."""

import numpy as np
import pytest

from growth.data.semantic_features import (
    LABEL_NCR,
    LABEL_ED,
    LABEL_ET,
    compute_volumes,
    compute_log_volumes,
    compute_centroid,
    compute_bounding_box,
    compute_surface_area,
    compute_sphericity,
    compute_solidity,
    compute_aspect_ratios,
    compute_composition_features,
    compute_shape_features,
    compute_shape_array,
    extract_semantic_features,
)


class TestComputeVolumes:
    """Tests for volume computation."""

    def test_empty_mask(self):
        """Empty mask should return zero volumes."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        volumes = compute_volumes(mask)

        assert volumes["total"] == 0.0
        assert volumes["ncr"] == 0.0
        assert volumes["ed"] == 0.0
        assert volumes["et"] == 0.0

    def test_single_label_ncr(self):
        """Test volume with only NCR label."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        # 10x10x10 cube = 1000 voxels
        mask[40:50, 40:50, 40:50] = LABEL_NCR

        volumes = compute_volumes(mask, spacing=(1.0, 1.0, 1.0))

        assert volumes["ncr"] == 1000.0
        assert volumes["total"] == 1000.0
        assert volumes["ed"] == 0.0
        assert volumes["et"] == 0.0

    def test_multiple_labels(self):
        """Test volume with multiple labels."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[0:10, 0:10, 0:10] = LABEL_NCR  # 1000 voxels
        mask[20:30, 20:30, 20:30] = LABEL_ED  # 1000 voxels
        mask[40:45, 40:45, 40:45] = LABEL_ET  # 125 voxels

        volumes = compute_volumes(mask)

        assert volumes["ncr"] == 1000.0
        assert volumes["ed"] == 1000.0
        assert volumes["et"] == 125.0
        assert volumes["total"] == 2125.0

    def test_non_unit_spacing(self):
        """Test volume with non-unit spacing."""
        mask = np.zeros((10, 10, 10), dtype=np.int32)
        mask[0:5, 0:5, 0:5] = LABEL_NCR  # 125 voxels

        # Each voxel is 2mm x 2mm x 2mm = 8 mm^3
        volumes = compute_volumes(mask, spacing=(2.0, 2.0, 2.0))

        assert volumes["ncr"] == 125 * 8.0
        assert volumes["total"] == 125 * 8.0


class TestComputeLogVolumes:
    """Tests for log-transformed volume computation."""

    def test_output_shape(self):
        """Output should be [4] array."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        log_vols = compute_log_volumes(mask)

        assert log_vols.shape == (4,)
        assert log_vols.dtype == np.float32

    def test_empty_mask_log1p(self):
        """Empty mask should return log(1) = 0 for all."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        log_vols = compute_log_volumes(mask)

        np.testing.assert_array_almost_equal(log_vols, [0.0, 0.0, 0.0, 0.0])

    def test_log_transform_correctness(self):
        """Verify log transform is applied correctly."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[0:10, 0:10, 0:10] = LABEL_NCR  # 1000 voxels

        log_vols = compute_log_volumes(mask)

        # log1p(1000) ≈ 6.908
        expected_log = np.log1p(1000.0)
        assert abs(log_vols[0] - expected_log) < 0.001  # total
        assert abs(log_vols[1] - expected_log) < 0.001  # ncr


class TestComputeCentroid:
    """Tests for centroid computation."""

    def test_centered_tumor(self):
        """Centered tumor should have centroid at (0.5, 0.5, 0.5)."""
        mask = np.zeros((100, 100, 100), dtype=np.int32)
        # 20x20x20 cube centered at (50, 50, 50)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        centroid = compute_centroid(mask, normalize=True)

        # Centroid should be at center: (49.5/100, 49.5/100, 49.5/100) ≈ (0.5, 0.5, 0.5)
        np.testing.assert_array_almost_equal(centroid, [0.495, 0.495, 0.495], decimal=2)

    def test_corner_tumor(self):
        """Tumor in corner should have low centroid values."""
        mask = np.zeros((100, 100, 100), dtype=np.int32)
        mask[0:10, 0:10, 0:10] = LABEL_NCR

        centroid = compute_centroid(mask, normalize=True)

        # Centroid at (4.5/100, 4.5/100, 4.5/100) ≈ (0.045, 0.045, 0.045)
        assert all(c < 0.1 for c in centroid)

    def test_empty_mask_returns_center(self):
        """Empty mask should return center (0.5, 0.5, 0.5)."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        centroid = compute_centroid(mask, normalize=True)

        np.testing.assert_array_equal(centroid, [0.5, 0.5, 0.5])

    def test_unnormalized_centroid(self):
        """Test centroid in physical coordinates."""
        mask = np.zeros((100, 100, 100), dtype=np.int32)
        mask[0:10, 0:10, 0:10] = LABEL_NCR

        centroid = compute_centroid(mask, spacing=(1.0, 1.0, 1.0), normalize=False)

        # Should be at (4.5, 4.5, 4.5) mm
        np.testing.assert_array_almost_equal(centroid, [4.5, 4.5, 4.5], decimal=1)


class TestComputeBoundingBox:
    """Tests for bounding box computation."""

    def test_known_box(self):
        """Test bounding box for known tumor extent."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[10:20, 30:50, 40:60] = LABEL_NCR

        min_coords, max_coords = compute_bounding_box(mask)

        np.testing.assert_array_equal(min_coords, [10, 30, 40])
        np.testing.assert_array_equal(max_coords, [19, 49, 59])

    def test_empty_mask(self):
        """Empty mask returns default box."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        min_coords, max_coords = compute_bounding_box(mask)

        np.testing.assert_array_equal(min_coords, [0, 0, 0])
        np.testing.assert_array_equal(max_coords, [1, 1, 1])


class TestComputeSurfaceArea:
    """Tests for surface area computation."""

    def test_empty_mask(self):
        """Empty mask should have zero surface area."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        sa = compute_surface_area(mask)

        assert sa == 0.0

    def test_single_voxel(self):
        """Single voxel should have non-zero surface area."""
        mask = np.zeros((10, 10, 10), dtype=np.int32)
        mask[5, 5, 5] = LABEL_NCR

        sa = compute_surface_area(mask)

        # Single voxel is all surface
        assert sa > 0


class TestComputeSphericity:
    """Tests for sphericity computation."""

    def test_empty_mask(self):
        """Empty mask should have zero sphericity."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        sphericity = compute_sphericity(mask)

        assert sphericity == 0.0

    def test_sphericity_bounded(self):
        """Sphericity should be in [0, 1]."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        sphericity = compute_sphericity(mask)

        assert 0.0 <= sphericity <= 1.0

    def test_cube_vs_very_elongated(self):
        """Cube should have higher sphericity than very elongated shape."""
        # Cube: 20x20x20 = 8000 voxels
        mask_cube = np.zeros((96, 96, 96), dtype=np.int32)
        mask_cube[40:60, 40:60, 40:60] = LABEL_NCR

        # Very elongated rod: 80x5x5 = 2000 voxels (much higher surface/volume)
        mask_elong = np.zeros((96, 96, 96), dtype=np.int32)
        mask_elong[8:88, 45:50, 45:50] = LABEL_NCR

        sphericity_cube = compute_sphericity(mask_cube)
        sphericity_elong = compute_sphericity(mask_elong)

        # Cube is more spherical than a thin rod
        assert sphericity_cube > sphericity_elong


class TestComputeSolidity:
    """Tests for solidity computation."""

    def test_empty_mask(self):
        """Empty mask should have zero solidity."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        solidity = compute_solidity(mask)

        assert solidity == 0.0

    def test_convex_shape(self):
        """Convex shape (cube) should have solidity close to 1."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        solidity = compute_solidity(mask)

        # Cube is convex, so solidity should be ~1
        assert solidity > 0.9

    def test_solidity_bounded(self):
        """Solidity should be in [0, 1]."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        solidity = compute_solidity(mask)

        assert 0.0 <= solidity <= 1.0


class TestComputeAspectRatios:
    """Tests for aspect ratio computation."""

    def test_cube_aspect_ratios(self):
        """Cube should have aspect ratios close to 1."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR  # 20x20x20

        aspects = compute_aspect_ratios(mask)

        assert abs(aspects["aspect_dh"] - 1.0) < 0.1
        assert abs(aspects["aspect_dw"] - 1.0) < 0.1
        assert abs(aspects["aspect_hw"] - 1.0) < 0.1

    def test_elongated_aspect_ratios(self):
        """Elongated shape should have non-unit aspect ratios."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:80, 40:50, 40:50] = LABEL_NCR  # 40x10x10

        aspects = compute_aspect_ratios(mask)

        # Depth (40) vs height (10): aspect_dh = 4
        assert aspects["aspect_dh"] > 3.0


class TestComputeShapeFeatures:
    """Tests for combined shape feature computation."""

    def test_output_keys(self):
        """Should return all expected keys."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        features = compute_shape_features(mask)

        expected_keys = {
            "sphericity",
            "surface_area_log",
            "solidity",
            "aspect_dh",
            "aspect_dw",
            "aspect_hw",
        }
        assert set(features.keys()) == expected_keys


class TestComputeShapeArray:
    """Tests for shape feature array computation."""

    def test_output_shape(self):
        """Output should be [6] array."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        shape_arr = compute_shape_array(mask)

        assert shape_arr.shape == (3,)
        assert shape_arr.dtype == np.float32

    def test_consistency_with_components(self):
        """Array values should match individual function outputs."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        shape_arr = compute_shape_array(mask)
        composition = compute_composition_features(mask)

        assert abs(shape_arr[0] - compute_sphericity(mask)) < 1e-6
        assert abs(shape_arr[1] - composition["enhancement_ratio"]) < 1e-6
        assert abs(shape_arr[2] - composition["infiltration_index"]) < 1e-6


class TestCompositionFeatures:
    """Tests for tumor composition features."""

    def test_enhancement_ratio_range(self):
        """Enhancement ratio should be in [0, 1]."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:50, 40:50, 40:50] = LABEL_NCR
        mask[50:55, 40:50, 40:50] = LABEL_ED
        mask[55:60, 40:50, 40:50] = LABEL_ET

        comp = compute_composition_features(mask)

        assert 0.0 <= comp["enhancement_ratio"] <= 1.0

    def test_infiltration_index_nonnegative(self):
        """Infiltration index should be >= 0."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:50, 40:50, 40:50] = LABEL_NCR
        mask[50:60, 40:60, 40:60] = LABEL_ED

        comp = compute_composition_features(mask)

        assert comp["infiltration_index"] >= 0.0

    def test_empty_mask(self):
        """Empty mask should return zeros."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        comp = compute_composition_features(mask)

        assert comp["enhancement_ratio"] == 0.0
        assert comp["infiltration_index"] == 0.0

    def test_only_et_gives_ratio_one(self):
        """Tumor with only ET should have enhancement_ratio close to 1."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_ET

        comp = compute_composition_features(mask)

        assert comp["enhancement_ratio"] > 0.99

    def test_high_edema_gives_high_infiltration(self):
        """Tumor with much more edema than solid should have high infiltration."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        # Small solid tumor
        mask[45:50, 45:50, 45:50] = LABEL_NCR  # 125 voxels
        # Large edema
        mask[10:90, 10:90, 10:90] = LABEL_ED  # Will overwrite non-NCR

        # Actually set up properly: NCR first, then ED in surrounding region
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[45:50, 45:50, 45:50] = LABEL_NCR  # 125 voxels
        mask[30:70, 30:70, 30:70] = LABEL_ED  # 64000 voxels
        mask[45:50, 45:50, 45:50] = LABEL_NCR  # Re-set NCR (overwritten by ED)

        comp = compute_composition_features(mask)

        # infiltration_index = V_ED / (V_NCR + V_ET + eps) >> 1
        assert comp["infiltration_index"] > 1.0

    def test_no_edema_gives_zero_infiltration(self):
        """No edema should give infiltration_index ~0."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR
        mask[60:70, 40:60, 40:60] = LABEL_ET

        comp = compute_composition_features(mask)

        assert comp["infiltration_index"] < 0.01


class TestExtractSemanticFeatures:
    """Tests for combined semantic feature extraction."""

    def test_output_keys(self):
        """Should return all expected keys."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        features = extract_semantic_features(mask)

        assert "volume" in features
        assert "location" in features
        assert "shape" in features
        assert "all" in features

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        features = extract_semantic_features(mask)

        assert features["volume"].shape == (4,)
        assert features["location"].shape == (3,)
        assert features["shape"].shape == (3,)
        assert features["all"].shape == (10,)

    def test_all_is_concatenation(self):
        """'all' should be concatenation of volume, location, shape."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)
        mask[40:60, 40:60, 40:60] = LABEL_NCR

        features = extract_semantic_features(mask)

        expected_all = np.concatenate(
            [features["volume"], features["location"], features["shape"]]
        )
        np.testing.assert_array_almost_equal(features["all"], expected_all)

    def test_empty_mask(self):
        """Empty mask should return valid (but zero/default) features."""
        mask = np.zeros((96, 96, 96), dtype=np.int32)

        features = extract_semantic_features(mask)

        # Volumes should be zero (log1p(0) = 0)
        np.testing.assert_array_almost_equal(features["volume"], [0, 0, 0, 0])

        # Location should be center
        np.testing.assert_array_almost_equal(features["location"], [0.5, 0.5, 0.5])

        # Shape features should handle empty case
        assert features["shape"].shape == (3,)


class TestCentroidNormalization:
    """FLAW-5: Centroid normalization depends on volume dimensions.

    Centroids are normalized by dividing by image dimensions. If computed
    from native-resolution masks (variable per subject), the same physical
    tumor position gives different centroids. Using 192³ volumes fixes this.
    """

    def test_centroid_consistent_for_standardized_volumes(self):
        """Two 192³ masks with same tumor position → same centroid."""
        # Both masks are 192³ (standardized frame)
        mask_a = np.zeros((192, 192, 192), dtype=np.int32)
        mask_a[80:100, 80:100, 80:100] = LABEL_NCR

        mask_b = np.zeros((192, 192, 192), dtype=np.int32)
        mask_b[80:100, 80:100, 80:100] = LABEL_NCR

        centroid_a = compute_centroid(mask_a, normalize=True)
        centroid_b = compute_centroid(mask_b, normalize=True)

        np.testing.assert_array_almost_equal(centroid_a, centroid_b)

    def test_centroid_differs_for_different_volume_sizes(self):
        """Same voxel position in different-sized volumes → different centroid.

        This documents the old bug: centroids computed from native-resolution
        masks are not comparable across subjects with different image sizes.
        """
        # 100³ volume: tumor at voxel [50, 50, 50]
        mask_small = np.zeros((100, 100, 100), dtype=np.int32)
        mask_small[45:55, 45:55, 45:55] = LABEL_NCR

        # 200³ volume: tumor at same voxel position
        mask_large = np.zeros((200, 200, 200), dtype=np.int32)
        mask_large[45:55, 45:55, 45:55] = LABEL_NCR

        centroid_small = compute_centroid(mask_small, normalize=True)
        centroid_large = compute_centroid(mask_large, normalize=True)

        # Centroids should DIFFER because normalization divides by image dims
        # Small: ~0.495, Large: ~0.2475
        assert not np.allclose(centroid_small, centroid_large, atol=0.01), (
            f"Centroids should differ: small={centroid_small}, large={centroid_large}"
        )
