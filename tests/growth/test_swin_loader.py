# tests/growth/test_swin_loader.py
"""Tests for SwinUNETR encoder loading."""

from pathlib import Path

import pytest
import torch

from growth.models.encoder.swin_loader import (
    BRAINSEGFOUNDER_DEPTHS,
    BRAINSEGFOUNDER_FEATURE_SIZE,
    BRAINSEGFOUNDER_IN_CHANNELS,
    BRAINSEGFOUNDER_NUM_HEADS,
    BRAINSEGFOUNDER_OUT_CHANNELS,
    count_parameters,
    create_swinunetr,
    get_encoder_output_dim,
    get_swin_feature_dims,
    load_swin_encoder,
)


class TestCreateSwinunetr:
    """Tests for create_swinunetr function."""

    def test_create_swinunetr_default(self):
        """Test creating SwinUNETR with default parameters."""
        model = create_swinunetr()

        # Check model exists
        assert model is not None

        # Verify architecture constants
        assert BRAINSEGFOUNDER_IN_CHANNELS == 4
        assert BRAINSEGFOUNDER_OUT_CHANNELS == 3
        assert BRAINSEGFOUNDER_FEATURE_SIZE == 48

    def test_create_swinunetr_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = create_swinunetr()
        model.eval()

        # Expected input: [B, 4, 96, 96, 96]
        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            out = model(x)

        # Output should match input spatial dimensions
        assert out.shape == (1, 3, 96, 96, 96)

    def test_create_swinunetr_custom_channels(self):
        """Test creating SwinUNETR with custom channel counts."""
        model = create_swinunetr(in_channels=1, out_channels=5)
        model.eval()

        x = torch.randn(1, 1, 96, 96, 96)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 5, 96, 96, 96)

    def test_create_swinunetr_with_gradient_checkpointing(self):
        """Test creating SwinUNETR with gradient checkpointing enabled."""
        model = create_swinunetr(use_checkpoint=True)

        # Should not raise during creation
        assert model is not None

    def test_create_swinunetr_parameter_count(self):
        """Test that model has expected parameter count."""
        model = create_swinunetr()
        total_params = count_parameters(model)

        # BrainSegFounder SwinUNETR should have ~62M parameters
        assert total_params > 50_000_000
        assert total_params < 80_000_000


class TestGetSwinFeatureDims:
    """Tests for get_swin_feature_dims function."""

    def test_feature_dims_default(self):
        """Test feature dimensions with default feature_size=48."""
        dims = get_swin_feature_dims()

        # MONAI 1.5+ SwinUNETR dimensions
        expected = {
            "patch_embed": (48, 2),
            "layers1": (96, 4),
            "layers2": (192, 8),
            "layers3": (384, 16),
            "layers4": (768, 32),
            "encoder10": (768, 32),
        }

        assert dims == expected

    def test_feature_dims_custom_feature_size(self):
        """Test feature dimensions with custom feature_size."""
        dims = get_swin_feature_dims(feature_size=24)

        # Channels scale with feature_size
        assert dims["patch_embed"][0] == 24
        assert dims["layers2"][0] == 96  # 24 * 4
        assert dims["layers4"][0] == 384  # 24 * 16
        assert dims["encoder10"][0] == 384  # 24 * 16


class TestGetEncoderOutputDim:
    """Tests for get_encoder_output_dim function."""

    def test_encoder10_dim(self):
        """Test encoder10 output dimension."""
        dim = get_encoder_output_dim("encoder10")
        assert dim == 768

    def test_layers4_dim(self):
        """Test layers4 output dimension."""
        dim = get_encoder_output_dim("layers4")
        assert dim == 768

    def test_multi_scale_dim(self):
        """Test multi_scale output dimension."""
        dim = get_encoder_output_dim("multi_scale")
        # layers2(192) + layers3(384) + layers4(768) = 1344
        assert dim == 1344

    def test_unknown_level_raises(self):
        """Test error on unknown feature level."""
        with pytest.raises(ValueError, match="Unknown feature level"):
            get_encoder_output_dim("invalid_level")


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_all_parameters(self):
        """Test counting all parameters."""
        model = torch.nn.Linear(10, 5)  # 10*5 + 5 = 55 params
        count = count_parameters(model)
        assert count == 55

    def test_count_trainable_only(self):
        """Test counting only trainable parameters."""
        model = torch.nn.Linear(10, 5)
        model.weight.requires_grad = False  # Freeze weight (50 params)

        total = count_parameters(model, trainable_only=False)
        trainable = count_parameters(model, trainable_only=True)

        assert total == 55
        assert trainable == 5  # Only bias is trainable


class TestLoadSwinEncoder:
    """Tests for load_swin_encoder function."""

    def test_load_encoder_from_sample_checkpoint(
        self, sample_checkpoint: Path, sample_state_dict
    ):
        """Test loading encoder from sample checkpoint."""
        # Note: This will have missing keys since sample_state_dict is simplified
        model = load_swin_encoder(
            sample_checkpoint,
            freeze=True,
            device="cpu",
            strict_load=False,
        )

        assert model is not None

    def test_load_encoder_freeze(self, sample_checkpoint: Path):
        """Test that freeze=True freezes all parameters."""
        model = load_swin_encoder(
            sample_checkpoint,
            freeze=True,
            device="cpu",
            strict_load=False,
        )

        # All parameters should be frozen
        for param in model.parameters():
            assert not param.requires_grad

    def test_load_encoder_not_found(self, temp_dir: Path):
        """Test error when checkpoint doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_swin_encoder(temp_dir / "missing.pt")


class TestLoadSwinEncoderReal:
    """Tests with real BrainSegFounder checkpoint (skipped if unavailable)."""

    def test_load_real_encoder(self, real_checkpoint_path: Path):
        """Test loading encoder from real checkpoint."""
        model = load_swin_encoder(
            real_checkpoint_path,
            freeze=True,
            device="cpu",
        )

        assert model is not None

        # Check it's frozen
        trainable = count_parameters(model, trainable_only=True)
        assert trainable == 0

    def test_real_encoder_forward_pass(self, real_checkpoint_path: Path):
        """Test forward pass with real encoder."""
        model = load_swin_encoder(
            real_checkpoint_path,
            freeze=True,
            device="cpu",
        )
        model.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            # Test full forward (decoder output)
            out = model(x)
            assert out.shape == (1, 3, 96, 96, 96)

    def test_real_encoder_hidden_states(self, real_checkpoint_path: Path):
        """Test extracting hidden states from real encoder."""
        model = load_swin_encoder(
            real_checkpoint_path,
            freeze=True,
            device="cpu",
        )
        model.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            hidden_states = model.swinViT(x, model.normalize)

        # Check hidden state shapes
        assert len(hidden_states) == 5

        # hidden_states[0]: After patch_embed [B, 48, 48, 48, 48]
        assert hidden_states[0].shape == (1, 48, 48, 48, 48)

        # hidden_states[4]: After layers4 [B, 768, 3, 3, 3]
        assert hidden_states[4].shape == (1, 768, 3, 3, 3)

    def test_real_encoder_encoder10_output(self, real_checkpoint_path: Path):
        """Test encoder10 bottleneck output."""
        model = load_swin_encoder(
            real_checkpoint_path,
            include_encoder10=True,
            freeze=True,
            device="cpu",
        )
        model.eval()

        x = torch.randn(1, 4, 96, 96, 96)

        with torch.no_grad():
            hidden_states = model.swinViT(x, model.normalize)
            x4 = hidden_states[4]  # [1, 768, 3, 3, 3]
            enc10 = model.encoder10(x4)  # [1, 768, 3, 3, 3]

        assert enc10.shape == (1, 768, 3, 3, 3)


class TestWeightLoadingVerification:
    """Critical tests to verify all encoder weights are loaded correctly.

    These tests ensure no weights are silently dropped during loading.
    """

    def test_all_encoder_keys_present(self, real_checkpoint_path: Path):
        """Verify all encoder keys from checkpoint exist in model."""
        from growth.utils.checkpoint import load_checkpoint, extract_encoder_weights, ENCODER_PREFIXES

        # Load checkpoint
        checkpoint = load_checkpoint(real_checkpoint_path)
        state_dict = checkpoint["state_dict"]
        encoder_weights = extract_encoder_weights(state_dict, include_encoder10=True)

        # Create fresh model
        model = create_swinunetr()
        model_state = model.state_dict()

        # Get model's encoder keys only
        model_encoder_keys = {
            k for k in model_state.keys()
            if k.split(".")[0] in ENCODER_PREFIXES
        }

        ckpt_keys = set(encoder_weights.keys())

        # All checkpoint keys should exist in model
        extra_in_ckpt = ckpt_keys - model_encoder_keys
        assert len(extra_in_ckpt) == 0, (
            f"Checkpoint has {len(extra_in_ckpt)} keys not in model: "
            f"{sorted(extra_in_ckpt)[:5]}..."
        )

        # All model encoder keys should exist in checkpoint
        missing_in_ckpt = model_encoder_keys - ckpt_keys
        assert len(missing_in_ckpt) == 0, (
            f"Model has {len(missing_in_ckpt)} encoder keys not in checkpoint: "
            f"{sorted(missing_in_ckpt)[:5]}..."
        )

    def test_all_encoder_shapes_match(self, real_checkpoint_path: Path):
        """Verify all encoder weight shapes match between checkpoint and model."""
        from growth.utils.checkpoint import load_checkpoint, extract_encoder_weights

        # Load checkpoint
        checkpoint = load_checkpoint(real_checkpoint_path)
        state_dict = checkpoint["state_dict"]
        encoder_weights = extract_encoder_weights(state_dict, include_encoder10=True)

        # Create fresh model
        model = create_swinunetr()
        model_state = model.state_dict()

        # Check each weight shape matches
        shape_mismatches = []
        for key in encoder_weights.keys():
            if key in model_state:
                ckpt_shape = tuple(encoder_weights[key].shape)
                model_shape = tuple(model_state[key].shape)
                if ckpt_shape != model_shape:
                    shape_mismatches.append((key, ckpt_shape, model_shape))

        assert len(shape_mismatches) == 0, (
            f"Shape mismatches found:\n" +
            "\n".join(f"  {k}: ckpt={cs} vs model={ms}"
                      for k, cs, ms in shape_mismatches[:10])
        )

    def test_encoder_weight_count(self, real_checkpoint_path: Path):
        """Verify expected number of encoder weights are loaded."""
        from growth.utils.checkpoint import load_checkpoint, extract_encoder_weights

        checkpoint = load_checkpoint(real_checkpoint_path)
        state_dict = checkpoint["state_dict"]
        encoder_weights = extract_encoder_weights(state_dict, include_encoder10=True)

        # BrainSegFounder encoder should have 137 weight tensors
        assert len(encoder_weights) == 137, (
            f"Expected 137 encoder weights, got {len(encoder_weights)}"
        )

    def test_load_state_dict_no_unexpected_keys(self, real_checkpoint_path: Path):
        """Verify load_state_dict reports no unexpected keys."""
        from growth.utils.checkpoint import load_checkpoint, extract_encoder_weights

        checkpoint = load_checkpoint(real_checkpoint_path)
        state_dict = checkpoint["state_dict"]
        encoder_weights = extract_encoder_weights(state_dict, include_encoder10=True)

        model = create_swinunetr()
        result = model.load_state_dict(encoder_weights, strict=False)

        # Should have no unexpected keys (all ckpt keys exist in model)
        assert len(result.unexpected_keys) == 0, (
            f"Unexpected keys: {result.unexpected_keys[:5]}..."
        )

        # Missing keys should only be decoder keys
        non_decoder_missing = [
            k for k in result.missing_keys
            if not k.startswith(("decoder", "out"))
        ]
        assert len(non_decoder_missing) == 0, (
            f"Missing non-decoder keys: {non_decoder_missing}"
        )


class TestRealDataForwardPass:
    """Tests with real BraTS-MEN data to verify end-to-end functionality."""

    def test_forward_pass_real_data(
        self, real_checkpoint_path: Path, real_data_path: Path
    ):
        """Test complete forward pass with real MRI data."""
        from growth.models.encoder.feature_extractor import FeatureExtractor
        from growth.data.transforms import get_val_transforms

        # Load encoder and create feature extractor
        encoder = load_swin_encoder(real_checkpoint_path, freeze=True)
        extractor = FeatureExtractor(encoder, level="encoder10")
        extractor.eval()

        # Load real subject
        subjects = list(real_data_path.iterdir())
        subject = subjects[0]

        data = {}
        for modality in ["t1c", "t1n", "t2f", "t2w"]:
            files = list(subject.glob(f"*-{modality}.nii.gz"))
            if not files:
                pytest.skip(f"Missing {modality} for {subject.name}")
            data[modality] = str(files[0])

        seg_files = list(subject.glob("*-seg.nii.gz"))
        if seg_files:
            data["seg"] = str(seg_files[0])

        # Apply transforms
        transforms = get_val_transforms()
        result = transforms(data)
        image = result["image"].unsqueeze(0)  # [1, 4, 128, 128, 128]

        # Forward pass
        with torch.no_grad():
            features = extractor(image)

        # Verify output
        assert features.shape == (1, 768), f"Expected (1, 768), got {features.shape}"
        assert not torch.isnan(features).any(), "NaN values in output"
        assert not torch.isinf(features).any(), "Inf values in output"
        assert features.abs().sum() > 0, "Output is all zeros"

    def test_forward_pass_output_statistics(
        self, real_checkpoint_path: Path, real_data_path: Path
    ):
        """Verify output statistics are reasonable for real data."""
        from growth.models.encoder.feature_extractor import FeatureExtractor
        from growth.data.transforms import get_val_transforms

        encoder = load_swin_encoder(real_checkpoint_path, freeze=True)
        extractor = FeatureExtractor(encoder, level="encoder10")
        extractor.eval()

        # Load multiple subjects for statistics
        subjects = list(real_data_path.iterdir())[:3]
        transforms = get_val_transforms()

        all_features = []
        for subject in subjects:
            data = {}
            try:
                for modality in ["t1c", "t1n", "t2f", "t2w"]:
                    files = list(subject.glob(f"*-{modality}.nii.gz"))
                    data[modality] = str(files[0])
                seg_files = list(subject.glob("*-seg.nii.gz"))
                if seg_files:
                    data["seg"] = str(seg_files[0])
            except (IndexError, StopIteration):
                continue

            result = transforms(data)
            image = result["image"].unsqueeze(0)

            with torch.no_grad():
                features = extractor(image)
            all_features.append(features)

        if len(all_features) < 2:
            pytest.skip("Not enough valid subjects for statistics")

        features = torch.cat(all_features, dim=0)

        # Features should have reasonable statistics
        mean = features.mean().item()
        std = features.std().item()

        # Mean should be roughly in [-5, 5] range
        assert -10 < mean < 10, f"Mean {mean} seems unreasonable"
        # Std should be positive and not too extreme
        assert 0.1 < std < 10, f"Std {std} seems unreasonable"

        # Different subjects should produce different features
        if len(all_features) >= 2:
            diff = (all_features[0] - all_features[1]).abs().mean()
            assert diff > 0.01, "Different subjects produced nearly identical features"

    def test_forward_pass_deterministic(
        self, real_checkpoint_path: Path, real_data_path: Path
    ):
        """Verify forward pass is deterministic for frozen encoder."""
        from growth.models.encoder.feature_extractor import FeatureExtractor
        from growth.data.transforms import get_val_transforms

        encoder = load_swin_encoder(real_checkpoint_path, freeze=True)
        extractor = FeatureExtractor(encoder, level="encoder10")
        extractor.eval()

        subjects = list(real_data_path.iterdir())
        subject = subjects[0]

        data = {}
        for modality in ["t1c", "t1n", "t2f", "t2w"]:
            files = list(subject.glob(f"*-{modality}.nii.gz"))
            if not files:
                pytest.skip(f"Missing {modality}")
            data[modality] = str(files[0])
        seg_files = list(subject.glob("*-seg.nii.gz"))
        if seg_files:
            data["seg"] = str(seg_files[0])

        transforms = get_val_transforms()
        result = transforms(data)
        image = result["image"].unsqueeze(0)

        # Run twice
        with torch.no_grad():
            features1 = extractor(image.clone())
            features2 = extractor(image.clone())

        # Should be exactly identical
        assert torch.allclose(features1, features2, atol=1e-6), (
            "Forward pass not deterministic for frozen encoder"
        )
