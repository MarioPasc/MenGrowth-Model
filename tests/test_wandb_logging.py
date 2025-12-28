"""Dry-run tests for wandb logging without full training.

This module provides unit tests for the wandb logging infrastructure,
including logger initialization, metric computation, and callback functionality.
"""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf
import tempfile


def test_wandb_logger_initialization():
    """Test WandbLogger can be initialized in offline mode."""
    # Create minimal config
    cfg = OmegaConf.create({
        "logging": {
            "logger": {
                "type": "wandb",
                "wandb": {
                    "project": "test-project",
                    "offline": True,
                }
            }
        }
    })

    # Test logger creation
    from engine.train import create_logger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = create_logger(cfg, Path(tmpdir), "test_exp")

        assert logger is not None
        assert hasattr(logger, 'experiment')


def test_csv_logger_fallback():
    """Test fallback to CSVLogger when wandb not requested."""
    cfg = OmegaConf.create({
        "logging": {
            "logger": {
                "type": "csv",
            }
        }
    })

    from engine.train import create_logger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = create_logger(cfg, Path(tmpdir), "test_exp")

        assert logger is not None
        # CSVLogger should be created
        assert logger.__class__.__name__ == "CSVLogger"


def test_image_metrics_psnr():
    """Test PSNR computation on dummy 3D volumes."""
    from vae.utils.image_metrics import compute_psnr_3d

    # Create dummy volumes
    pred = torch.randn(2, 1, 64, 64, 64)
    target = pred + 0.1 * torch.randn_like(pred)

    psnr = compute_psnr_3d(pred, target)

    assert psnr.numel() == 1  # Scalar output
    assert psnr > 0  # PSNR should be positive
    assert torch.isfinite(psnr)  # Should not be NaN or Inf


def test_image_metrics_ssim():
    """Test SSIM computation on dummy 3D volumes."""
    from vae.utils.image_metrics import compute_ssim_3d

    # Create dummy volumes
    pred = torch.randn(2, 1, 64, 64, 64)
    target = pred + 0.05 * torch.randn_like(pred)

    ssim = compute_ssim_3d(pred, target)

    assert ssim.numel() == 1  # Scalar output
    assert 0 <= ssim <= 1  # SSIM should be in [0, 1]
    assert torch.isfinite(ssim)


def test_system_metrics_callback():
    """Test SystemMetricsCallback instantiation."""
    from vae.training.callbacks.system_callbacks import SystemMetricsCallback

    callback = SystemMetricsCallback()

    assert callback is not None
    assert hasattr(callback, 'on_train_epoch_start')
    assert hasattr(callback, 'on_train_batch_end')
    assert hasattr(callback, 'on_train_epoch_end')


def test_wandb_dashboard_callback():
    """Test WandbDashboardCallback instantiation."""
    from vae.training.callbacks.wandb_callbacks import WandbDashboardCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        callback = WandbDashboardCallback(
            run_dir=Path(tmpdir),
            every_n_epochs=10,
        )

        assert callback is not None
        assert callback.every_n_epochs == 10
        assert callback.run_dir == Path(tmpdir)


def test_wandb_latent_viz_callback():
    """Test WandbLatentVizCallback instantiation."""
    from vae.training.callbacks.wandb_callbacks import WandbLatentVizCallback

    callback = WandbLatentVizCallback(
        every_n_epochs=20,
        n_samples=100,
    )

    assert callback is not None
    assert callback.every_n_epochs == 20
    assert callback.n_samples == 100


def test_reconstruction_callback_with_wandb_flag():
    """Test ReconstructionCallback accepts log_to_wandb parameter."""
    from vae.training.callbacks.callbacks import ReconstructionCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        callback = ReconstructionCallback(
            run_dir=tmpdir,
            recon_every_n_epochs=5,
            num_recon_samples=2,
            log_to_wandb=True,
        )

        assert callback is not None
        assert callback.log_to_wandb is True


def test_per_modality_mse_computation():
    """Test per-modality MSE computation logic."""
    # Simulate 4-channel MRI data
    x_hat = torch.randn(2, 4, 64, 64, 64)
    x = x_hat + 0.1 * torch.randn_like(x_hat)

    modality_names = ["t1c", "t1n", "t2f", "t2w"]
    modality_mses = {}

    for i, mod_name in enumerate(modality_names):
        mod_recon = torch.nn.functional.mse_loss(
            x_hat[:, i:i+1],
            x[:, i:i+1],
            reduction="mean"
        )
        modality_mses[mod_name] = mod_recon

    # Verify all modalities computed
    assert len(modality_mses) == 4
    for mod_name in modality_names:
        assert mod_name in modality_mses
        assert torch.isfinite(modality_mses[mod_name])
        assert modality_mses[mod_name] >= 0  # MSE should be non-negative


def test_config_backward_compatibility():
    """Test that old configs still work with new structure."""
    # Old config structure (legacy)
    old_cfg = OmegaConf.create({
        "logging": {
            "save_dir": "experiments/runs",
            "recon_every_n_epochs": 5,
            "num_recon_samples": 2,
        }
    })

    # Should be able to access with .get() and defaults
    logger_type = old_cfg.logging.get("logger", {}).get("type", "csv")
    assert logger_type == "csv"  # Should default to csv

    # New config structure
    new_cfg = OmegaConf.create({
        "logging": {
            "save_dir": "experiments/runs",
            "logger": {
                "type": "wandb",
                "wandb": {
                    "project": "test",
                    "offline": True,
                }
            },
            "visual": {
                "recon_every_n_epochs": 5,
                "num_recon_samples": 2,
            }
        }
    })

    logger_type = new_cfg.logging.get("logger", {}).get("type", "csv")
    assert logger_type == "wandb"


if __name__ == "__main__":
    # Run tests individually for debugging
    print("Running wandb logging tests...")

    print("✓ Testing WandbLogger initialization...")
    test_wandb_logger_initialization()

    print("✓ Testing CSV logger fallback...")
    test_csv_logger_fallback()

    print("✓ Testing PSNR computation...")
    test_image_metrics_psnr()

    print("✓ Testing SSIM computation...")
    test_image_metrics_ssim()

    print("✓ Testing SystemMetricsCallback...")
    test_system_metrics_callback()

    print("✓ Testing WandbDashboardCallback...")
    test_wandb_dashboard_callback()

    print("✓ Testing WandbLatentVizCallback...")
    test_wandb_latent_viz_callback()

    print("✓ Testing ReconstructionCallback with wandb flag...")
    test_reconstruction_callback_with_wandb_flag()

    print("✓ Testing per-modality MSE computation...")
    test_per_modality_mse_computation()

    print("✓ Testing config backward compatibility...")
    test_config_backward_compatibility()

    print("\n✅ All tests passed!")
