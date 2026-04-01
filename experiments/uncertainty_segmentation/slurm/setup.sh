#!/usr/bin/env bash
# experiments/uncertainty_segmentation/slurm/setup.sh
# Pre-flight validation for LoRA-Ensemble SLURM jobs.
#
# Usage: bash experiments/uncertainty_segmentation/slurm/setup.sh [config_path]

set -euo pipefail

CONFIG_PATH="${1:-experiments/uncertainty_segmentation/config.yaml}"
CONDA_ENV="${CONDA_ENV_NAME:-growth}"

echo "=========================================="
echo "LORA-ENSEMBLE PRE-FLIGHT CHECKS"
echo "=========================================="
echo "Config: ${CONFIG_PATH}"
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

# 4. Verify critical paths from config
python -c "
import yaml
from pathlib import Path

with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)

ckpt_dir = Path(cfg['paths']['checkpoint_dir'])
ckpt_file = ckpt_dir / cfg['paths']['checkpoint_filename']
h5_file = Path(cfg['paths']['men_h5_file'])

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

n = cfg['ensemble']['n_members']
print(f'  [OK]   Ensemble: {n} members')
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
