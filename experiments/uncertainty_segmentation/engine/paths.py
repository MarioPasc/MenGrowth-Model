# experiments/uncertainty_segmentation/engine/paths.py
"""Centralised path derivation for the uncertainty_segmentation module.

Each experiment run is uniquely identified by (rank, n_members, base_seed).
The run directory is derived automatically from these parameters.

Example:
    results/uncertainty_segmentation/r8_M5_s42/
"""

from pathlib import Path

from omegaconf import DictConfig


def get_run_dir(config: DictConfig, override: Path | str | None = None) -> Path:
    """Derive experiment run directory from config parameters.

    Format: {output_dir}/r{rank}_M{n_members}_s{base_seed}/

    Args:
        config: Full experiment configuration.
        override: If provided, use this directly (from SLURM --run-dir).
            Overrides the derived path.

    Returns:
        Path to the run directory.
    """
    if override is not None:
        return Path(override)
    base = Path(config.experiment.output_dir)
    run_name = (
        f"r{config.lora.rank}"
        f"_M{config.ensemble.n_members}"
        f"_s{config.ensemble.base_seed}"
    )
    return base / run_name
