# tests/growth/test_feature_extractor.py
"""Tests for feature extraction from SwinUNETR encoder."""

from pathlib import Path

import pytest
import torch

from growth.models.encoder.feature_extractor import (
    LEVEL_DIMS,
    FeatureExtractor,
    create_feature_extractor,
    get_feature_dim,
)
from growth.models.encoder.swin_loader import create_swinunetr, load_swin_encoder


class TestLevelDims:
    """Tests for LEVEL_DIMS constants."""

    def test_level_dims_values(self):
        """Test expected dimension values."""
        assert LEVEL_DIMS["encoder10"] == 768
        assert LEVEL_DIMS["layers4"] == 768
        assert LEVEL_DIMS["multi_scale"] == 1344  # 192 + 384 + 768


class TestGetFeatureDim:
    """Tests for get_feature_dim function."""

    def test_get_feature_dim_encoder10(self):
        """Test getting encoder10 dimension."""
        assert get_feature_dim("encoder10") == 768

    def test_get_feature_dim_layers4(self):
        """Test getting layers4 dimension."""
        assert get_feature_dim("layers4") == 768

    def test_get_feature_dim_multi_scale(self):
        """Test getting multi_scale dimension."""
        assert get_feature_dim("multi_scale") == 1344

    def test_get_feature_dim_unknown(self):
        """Test error on unknown level."""
        with pytest.raises(ValueError, match="Unknown feature level"):
            get_feature_dim("invalid")


class TestFeatureExtractorInit:
    """Tests for FeatureExtractor initialization."""

    @pytest.fixture
    def encoder(self):
        """Create a SwinUNETR encoder for testing."""
        return create_swinunetr()

    def test_init_encoder10(self, encoder):
        """Test initialization with encoder10 level."""
        extractor = FeatureExtractor(encoder, level="encoder10")

        assert extractor.level == "encoder10"
        assert extractor.output_dim == 768
        assert extractor.feature_dim == 768  # Alias

    def test_init_layers4(self, encoder):
        """Test initialization with layers4 level."""
        extractor = FeatureExtractor(encoder, level="layers4")

        assert extractor.level == "layers4"
        assert extractor.output_dim == 768

    def test_init_multi_scale(self, encoder):
        """Test initialization with multi_scale level."""
        extractor = FeatureExtractor(encoder, level="multi_scale")

        assert extractor.level == "multi_scale"
        assert extractor.output_dim == 1344

    def test_init_with_normalize(self, encoder):
        """Test initialization with LayerNorm."""
        extractor = FeatureExtractor(encoder, level="layers4", normalize=True)

        assert extractor.layer_norm is not None
        assert extractor.layer_norm.normalized_shape == (768,)

    def test_init_without_normalize(self, encoder):
        """Test initialization without LayerNorm."""
        extractor = FeatureExtractor(encoder, level="layers4", normalize=False)

        assert extractor.layer_norm is None

    def test_init_invalid_level(self, encoder):
        """Test error on invalid level."""
        with pytest.raises(ValueError, match="Unknown feature level"):
            FeatureExtractor(encoder, level="invalid")


class TestFeatureExtractorForward:
    """Tests for FeatureExtractor forward pass."""

    @pytest.fixture
    def encoder(self):
        """Create a SwinUNETR encoder for testing."""
        model = create_swinunetr()
        model.eval()
        return model

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 4, 96, 96, 96)

    def test_forward_encoder10(self, encoder, sample_input):
        """Test forward pass with encoder10 level."""
        extractor = FeatureExtractor(encoder, level="encoder10")
        extractor.eval()

        with torch.no_grad():
            features = extractor(sample_input)

        assert features.shape == (2, 768)

    def test_forward_layers4(self, encoder, sample_input):
        """Test forward pass with layers4 level."""
        extractor = FeatureExtractor(encoder, level="layers4")
        extractor.eval()

        with torch.no_grad():
            features = extractor(sample_input)

        assert features.shape == (2, 768)

    def test_forward_multi_scale(self, encoder, sample_input):
        """Test forward pass with multi_scale level."""
        extractor = FeatureExtractor(encoder, level="multi_scale")
        extractor.eval()

        with torch.no_grad():
            features = extractor(sample_input)

        assert features.shape == (2, 1344)

    def test_forward_with_normalize(self, encoder, sample_input):
        """Test forward pass with LayerNorm applied."""
        extractor = FeatureExtractor(encoder, level="layers4", normalize=True)
        extractor.eval()

        with torch.no_grad():
            features = extractor(sample_input)

        assert features.shape == (2, 768)

        # LayerNorm should normalize along last dimension
        # Mean should be close to 0, std close to 1 (per sample)
        means = features.mean(dim=-1)
        stds = features.std(dim=-1)

        assert torch.allclose(means, torch.zeros_like(means), atol=0.1)
        assert torch.allclose(stds, torch.ones_like(stds), atol=0.1)

    def test_forward_batch_independence(self, encoder):
        """Test that batch samples are processed independently."""
        extractor = FeatureExtractor(encoder, level="layers4")
        extractor.eval()

        # Create two different inputs
        x1 = torch.randn(1, 4, 96, 96, 96)
        x2 = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            # Process separately
            f1 = extractor(x1)
            f2 = extractor(x2)

            # Process together
            x_batch = torch.cat([x1, x2], dim=0)
            f_batch = extractor(x_batch)

        # Results should match
        assert torch.allclose(f1, f_batch[0:1], atol=1e-5)
        assert torch.allclose(f2, f_batch[1:2], atol=1e-5)


class TestCreateFeatureExtractor:
    """Tests for create_feature_extractor factory function."""

    def test_create_feature_extractor_default(self):
        """Test factory function with defaults."""
        encoder = create_swinunetr()
        extractor = create_feature_extractor(encoder)

        assert isinstance(extractor, FeatureExtractor)
        assert extractor.level == "encoder10"
        assert extractor.output_dim == 768

    def test_create_feature_extractor_custom(self):
        """Test factory function with custom parameters."""
        encoder = create_swinunetr()
        extractor = create_feature_extractor(
            encoder, level="layers4", normalize=True
        )

        assert extractor.level == "layers4"
        assert extractor.layer_norm is not None


class TestFeatureExtractorReal:
    """Tests with real BrainSegFounder checkpoint (skipped if unavailable)."""

    def test_real_encoder_encoder10(self, real_checkpoint_path: Path):
        """Test encoder10 extraction with real encoder."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=True)
        extractor = FeatureExtractor(encoder, level="encoder10")
        extractor.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            features = extractor(x)

        assert features.shape == (1, 768)

    def test_real_encoder_layers4(self, real_checkpoint_path: Path):
        """Test layers4 extraction with real encoder."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=True)
        extractor = FeatureExtractor(encoder, level="layers4")
        extractor.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            features = extractor(x)

        assert features.shape == (1, 768)

    def test_real_encoder_multi_scale(self, real_checkpoint_path: Path):
        """Test multi_scale extraction with real encoder."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=True)
        extractor = FeatureExtractor(encoder, level="multi_scale")
        extractor.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            features = extractor(x)

        assert features.shape == (1, 1344)

    def test_real_encoder_deterministic(self, real_checkpoint_path: Path):
        """Test that frozen encoder produces deterministic outputs."""
        encoder = load_swin_encoder(real_checkpoint_path, freeze=True)
        extractor = FeatureExtractor(encoder, level="layers4")
        extractor.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            f1 = extractor(x)
            f2 = extractor(x)

        assert torch.allclose(f1, f2)
