"""Utility functions module."""

from .seed import set_seed
from .logging import setup_logging
from .io import (
    save_config,
    create_run_dir,
    save_split_csvs,
    update_runs_index,
    get_hardware_info,
    generate_run_id,
    get_experiment_dir,
)
from .curriculum import (
    print_curriculum_schedule,
    print_latent_partitioning,
    print_model_summary,
    log_curriculum_schedule,
)
# Re-export from vae.metrics for backward compatibility
from vae.metrics import compute_psnr_3d, compute_ssim_2d_slices, compute_ssim_3d

__all__ = [
    "set_seed",
    "setup_logging",
    "save_config",
    "create_run_dir",
    "save_split_csvs",
    "update_runs_index",
    "get_hardware_info",
    "generate_run_id",
    "get_experiment_dir",
    "compute_psnr_3d",
    "compute_ssim_2d_slices",
    "compute_ssim_3d",  # Deprecated alias
    # Curriculum visualization
    "print_curriculum_schedule",
    "print_latent_partitioning",
    "print_model_summary",
    "log_curriculum_schedule",
]
