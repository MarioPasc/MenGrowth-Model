#!/usr/bin/env bash
# experiments/uncertainty_segmentation/slurm/setup.sh
# Pre-flight validation for LoRA-Ensemble SLURM jobs.
#
# Usage:
#   bash experiments/uncertainty_segmentation/slurm/setup.sh [config_path] [override_path]
#
# If config_path is the picasso override, the base config is auto-detected
# and merged. You can also pass both explicitly:
#   bash setup.sh base_config.yaml picasso_override.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_CONFIG="${MODULE_DIR}/config.yaml"

CONFIG_PATH="${1:-${BASE_CONFIG}}"
OVERRIDE_PATH="${2:-}"
CONDA_ENV="${CONDA_ENV_NAME:-growth}"

echo "=========================================="
echo "LORA-ENSEMBLE PRE-FLIGHT CHECKS"
echo "=========================================="
echo "Config:   ${CONFIG_PATH}"
if [ -n "${OVERRIDE_PATH}" ]; then
    echo "Override: ${OVERRIDE_PATH}"
fi
echo ""

# Activate conda for pre-flight
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV}" 2>/dev/null || source activate "${CONDA_ENV}"
fi

FAIL=0

# 1. Config exists
if [ -f "${CONFIG_PATH}" ]; then
    echo "  [OK]   Config file exists"
else
    echo "  [FAIL] Config not found: ${CONFIG_PATH}"
    FAIL=1
fi

# 2. No placeholder paths
if grep -q "/PLACEHOLDER/" "${CONFIG_PATH}" 2>/dev/null; then
    echo "  [FAIL] Config contains /PLACEHOLDER/ paths"
    FAIL=1
else
    echo "  [OK]   No placeholder paths"
fi

# 3. Python import check
python -c "
from experiments.uncertainty_segmentation.engine.train_member import train_single_member
from experiments.uncertainty_segmentation.engine.ensemble_inference import EnsemblePredictor
from experiments.uncertainty_segmentation.engine.volume_extraction import extract_ensemble_volumes
print('  [OK]   Python imports')
" 2>/dev/null || { echo "  [FAIL] Python imports failed"; FAIL=1; }

# 4. Verify critical paths from MERGED config
python -c "
from omegaconf import OmegaConf
from pathlib import Path
import os

# Load base config
base_path = '${BASE_CONFIG}'
config_path = '${CONFIG_PATH}'
override_path = '${OVERRIDE_PATH}'

# Determine merge strategy
if config_path != base_path and os.path.exists(base_path):
    # Passed config is not the base — treat as override on top of base
    cfg = OmegaConf.merge(OmegaConf.load(base_path), OmegaConf.load(config_path))
elif override_path and os.path.exists(override_path):
    # Explicit base + override
    cfg = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.load(override_path))
else:
    # Single standalone config (must be complete)
    cfg = OmegaConf.load(config_path)

# Check paths
ckpt_dir = Path(cfg.paths.checkpoint_dir)
ckpt_file = ckpt_dir / cfg.paths.checkpoint_filename
h5_file = Path(cfg.paths.men_h5_file)

if ckpt_file.exists():
    print(f'  [OK]   Checkpoint: {ckpt_file}')
else:
    print(f'  [FAIL] Checkpoint not found: {ckpt_file}')
    exit(1)

if h5_file.exists():
    print(f'  [OK]   H5 file: {h5_file}')
else:
    print(f'  [FAIL] H5 file not found: {h5_file}')
    exit(1)

n = cfg.ensemble.n_members
r = cfg.lora.rank
s = cfg.ensemble.base_seed
print(f'  [OK]   Ensemble: M={n}, r={r}, seed={s}')
print(f'  [OK]   Run dir: {cfg.experiment.output_dir}/r{r}_M{n}_s{s}/')
" || FAIL=1

# 5. GPU check (if available)
python -c "
import torch
if torch.cuda.is_available():
    print(f'  [OK]   GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  [WARN] No GPU detected (expected on login node)')
" 2>/dev/null || true

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "All pre-flight checks PASSED"
    exit 0
else
    echo "Pre-flight checks FAILED"
    exit 1
fi
