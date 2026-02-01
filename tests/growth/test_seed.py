# tests/growth/test_seed.py
"""Tests for seed utilities."""

import random

import numpy as np
import pytest
import torch

from growth.utils.seed import set_seed


class TestSetSeed:
    """Tests for set_seed function."""

    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible random numbers."""
        set_seed(42)
        rand1_torch = torch.rand(10).tolist()
        rand1_numpy = np.random.rand(10).tolist()
        rand1_python = [random.random() for _ in range(10)]

        set_seed(42)
        rand2_torch = torch.rand(10).tolist()
        rand2_numpy = np.random.rand(10).tolist()
        rand2_python = [random.random() for _ in range(10)]

        assert rand1_torch == rand2_torch
        assert rand1_numpy == rand2_numpy
        assert rand1_python == rand2_python

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        rand1 = torch.rand(10)

        set_seed(123)
        rand2 = torch.rand(10)

        assert not torch.allclose(rand1, rand2)

    def test_set_seed_with_workers_true(self):
        """Test set_seed with workers=True (default)."""
        # Should not raise
        set_seed(42, workers=True)
        result = torch.rand(1).item()
        assert isinstance(result, float)

    def test_set_seed_with_workers_false(self):
        """Test set_seed with workers=False."""
        # Should not raise
        set_seed(42, workers=False)
        result = torch.rand(1).item()
        assert isinstance(result, float)

    def test_set_seed_negative(self):
        """Test set_seed rejects negative seed."""
        # PyTorch Lightning rejects negative seeds (numpy constraint)
        with pytest.raises(ValueError, match="not in bounds"):
            set_seed(-1)

    def test_set_seed_zero(self):
        """Test set_seed with zero seed."""
        set_seed(0)
        rand1 = torch.rand(10)

        set_seed(0)
        rand2 = torch.rand(10)

        assert torch.allclose(rand1, rand2)
