# experiments/stage1_volumetric/run_stage1_uq.py
"""DEPRECATED: Use run_all.py instead.

This shim preserves backward compatibility for existing scripts and
documentation that reference run_stage1_uq.py.

New usage::

    # Full pipeline (sequential, with resume):
    python -m experiments.stage1_volumetric.run_all \\
        --config experiments/stage1_volumetric/configs/config_uq.yaml

    # Single model (for SLURM):
    python -m experiments.stage1_volumetric.run_single_model \\
        --model LME --config experiments/stage1_volumetric/configs/config_uq.yaml

    # Post-hoc analysis only:
    python -m experiments.stage1_volumetric.run_analysis \\
        --config experiments/stage1_volumetric/configs/config_uq.yaml
"""

import warnings

warnings.warn(
    "run_stage1_uq.py is deprecated. Use run_all.py, run_single_model.py, "
    "or run_analysis.py instead.",
    DeprecationWarning,
    stacklevel=1,
)

from experiments.stage1_volumetric.run_all import main  # noqa: E402, F401

if __name__ == "__main__":
    main()
