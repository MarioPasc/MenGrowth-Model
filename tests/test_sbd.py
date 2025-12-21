"""Unit tests for Spatial Broadcast Decoder (SBD).

Tests verify:
1. Decoder input shape after broadcast and concat
2. Coordinate buffer existence and shape
3. Coordinate values are within [-1, 1]
4. Coordinate axis ordering (D, H, W)
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae_dynamics.models.components import SpatialBroadcastDecoder


class TestSpatialBroadcastDecoder:
    """Tests for SpatialBroadcastDecoder component."""

    @pytest.fixture
    def sbd(self):
        """Create SBD instance with default settings."""
        return SpatialBroadcastDecoder(
            z_dim=128,
            output_channels=4,
            base_filters=32,
            grid_size=(8, 8, 8),
            num_groups=8,
        )

    @pytest.fixture
    def dummy_z(self):
        """Create dummy latent vector."""
        return torch.randn(2, 128)

    def test_decoder_input_shape(self, sbd, dummy_z):
        """Test that broadcast_and_concat produces correct shape.

        Expected: [B=2, z_dim+3=131, D=8, H=8, W=8]
        """
        decoder_input = sbd.broadcast_and_concat(dummy_z)

        assert decoder_input.shape == (2, 131, 8, 8, 8), (
            f"Expected shape (2, 131, 8, 8, 8), got {decoder_input.shape}"
        )

    def test_coords_buffer_exists(self, sbd):
        """Test that coords buffer exists with correct shape."""
        assert hasattr(sbd, 'coords'), "SBD should have 'coords' buffer"

        assert sbd.coords.shape == (1, 3, 8, 8, 8), (
            f"Expected coords shape (1, 3, 8, 8, 8), got {sbd.coords.shape}"
        )

    def test_coords_values_in_range(self, sbd):
        """Test that coordinate values are within [-1, 1]."""
        coords = sbd.coords

        assert coords.min() >= -1.0, f"Min coord {coords.min()} < -1"
        assert coords.max() <= 1.0, f"Max coord {coords.max()} > 1"

    def test_coords_depth_axis_ordering(self, sbd):
        """Test that first coordinate channel (depth) varies only along depth axis.

        coords[:, 0, :, :, :] should be the depth coordinate grid.
        For indexing='ij', it should vary along the first spatial axis (depth).
        """
        coords = sbd.coords  # [1, 3, D, H, W]
        depth_coords = coords[0, 0]  # [D, H, W]

        # Depth should vary along dim 0 (depth)
        # For each (h, w), values should be the same across all d positions
        # Actually, depth should vary along the depth axis, so:
        # depth_coords[:, 0, 0] should be different values
        # depth_coords[0, :, :] should all be the same value

        # Check that depth varies along depth axis
        depth_values = depth_coords[:, 0, 0]  # Take one (h,w) position
        assert len(torch.unique(depth_values)) > 1, (
            "Depth coordinate should vary along depth axis"
        )

        # Check that for fixed depth, values are constant
        for d in range(depth_coords.shape[0]):
            unique_at_d = torch.unique(depth_coords[d, :, :])
            assert len(unique_at_d) == 1, (
                f"At depth {d}, values should be constant but got {len(unique_at_d)} unique values"
            )

    def test_coords_height_axis_ordering(self, sbd):
        """Test that second coordinate channel (height) varies only along height axis."""
        coords = sbd.coords  # [1, 3, D, H, W]
        height_coords = coords[0, 1]  # [D, H, W]

        # Height should vary along dim 1 (height)
        height_values = height_coords[0, :, 0]  # Take one (d,w) position
        assert len(torch.unique(height_values)) > 1, (
            "Height coordinate should vary along height axis"
        )

        # Check that for fixed height, values are constant (along d and w)
        for h in range(height_coords.shape[1]):
            unique_at_h = torch.unique(height_coords[:, h, :])
            assert len(unique_at_h) == 1, (
                f"At height {h}, values should be constant but got {len(unique_at_h)} unique values"
            )

    def test_coords_width_axis_ordering(self, sbd):
        """Test that third coordinate channel (width) varies only along width axis."""
        coords = sbd.coords  # [1, 3, D, H, W]
        width_coords = coords[0, 2]  # [D, H, W]

        # Width should vary along dim 2 (width)
        width_values = width_coords[0, 0, :]  # Take one (d,h) position
        assert len(torch.unique(width_values)) > 1, (
            "Width coordinate should vary along width axis"
        )

        # Check that for fixed width, values are constant (along d and h)
        for w in range(width_coords.shape[2]):
            unique_at_w = torch.unique(width_coords[:, :, w])
            assert len(unique_at_w) == 1, (
                f"At width {w}, values should be constant but got {len(unique_at_w)} unique values"
            )

    def test_full_decoder_output_shape(self, sbd, dummy_z):
        """Test full decoder forward pass produces correct output shape."""
        x_hat = sbd(dummy_z)

        assert x_hat.shape == (2, 4, 128, 128, 128), (
            f"Expected output shape (2, 4, 128, 128, 128), got {x_hat.shape}"
        )

    def test_coords_is_buffer_not_parameter(self, sbd):
        """Test that coords is registered as buffer, not parameter."""
        # Check it's in buffers
        buffer_names = [name for name, _ in sbd.named_buffers()]
        assert 'coords' in buffer_names, "coords should be a buffer"

        # Check it's not in parameters
        param_names = [name for name, _ in sbd.named_parameters()]
        assert 'coords' not in param_names, "coords should not be a parameter"

    def test_coords_not_requires_grad(self, sbd):
        """Test that coords does not require gradient."""
        assert not sbd.coords.requires_grad, "coords should not require grad"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
