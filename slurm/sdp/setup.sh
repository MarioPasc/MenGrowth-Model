#!/usr/bin/env bash
# slurm/sdp/setup.sh
# =============================================================================
# SDP MODULE — SETUP SCRIPT (run on login node with internet)
#
# Pre-flight checks before submitting the SLURM job:
#   1. Git pull latest code
#   2. Verify conda environment + h5py
#   3. Verify H5 file, checkpoint, and LoRA adapter exist
#   4. Import checks
#
# Usage:
#   bash slurm/sdp/setup.sh
# =============================================================================

set -euo pipefail

# ---- Configuration ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_DIR}/experiments/sdp/config/picasso/sdp_default.yaml}"

echo "=========================================="
echo "SDP MODULE — SETUP"
echo "=========================================="
echo "Date:      $(date)"
echo "Hostname:  $(hostname)"
echo "Repo:      ${REPO_DIR}"
echo "Config:    ${CONFIG_FILE}"
echo ""

# ---- Step 1: Activate conda environment ----
echo "[1/5] Activating conda environment: ${CONDA_ENV_NAME}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo ""

# ---- Parse paths from YAML config ----
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Use Python to parse YAML (available in conda env, no extra deps)
read_yaml_key() {
    python3 -c "
import yaml, sys
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
keys = '$1'.split('.')
val = cfg
for k in keys:
    val = val[k]
print(val)
"
}

H5_FILE="${H5_FILE:-$(read_yaml_key paths.h5_file)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(read_yaml_key paths.checkpoint_dir)}"
LORA_ADAPTER="${LORA_ADAPTER:-$(read_yaml_key paths.lora_checkpoint)}"

echo "Paths (from config):"
echo "  H5 file:      ${H5_FILE}"
echo "  Checkpoint:   ${CHECKPOINT_DIR}"
echo "  LoRA adapter: ${LORA_ADAPTER}"
echo ""

# ---- Step 2: Git pull ----
echo "[2/5] Git pull..."
cd "${REPO_DIR}"
git pull --ff-only || echo "  WARNING: git pull failed (offline or conflicts)"
echo ""

# ---- Step 3: Verify h5py ----
echo "[3/5] Checking h5py..."

# Ensure h5py is installed
python -c "import h5py; print(f'  h5py: {h5py.__version__}')" || {
    echo "  Installing h5py..."
    pip install h5py
}
echo ""

# ---- Step 4: Verify data files ----
echo "[4/5] Verifying data files..."

check_file() {
    local path="$1"
    local desc="$2"
    if [ -e "$path" ]; then
        echo "  OK: ${desc}"
    else
        echo "  MISSING: ${desc} → ${path}"
        return 1
    fi
}

ok=true
check_file "${H5_FILE}" "H5 dataset" || ok=false
check_file "${CHECKPOINT_DIR}" "BrainSegFounder checkpoint dir" || ok=false
check_file "${LORA_ADAPTER}" "LoRA adapter" || ok=false

if [ "$ok" = false ]; then
    echo ""
    echo "ERROR: Missing required files. Fix paths before submitting."
    exit 1
fi
echo ""

# ---- Step 5: Import checks ----
echo "[5/5] Import checks..."
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
import monai
print(f'  monai: {monai.__version__}')
import h5py
print(f'  h5py: {h5py.__version__}')
from growth.data.bratsmendata import BraTSMENDatasetH5
print('  BraTSMENDatasetH5: OK')
from experiments.lora_ablation.extract_features import extract_features_for_split
print('  extract_features_for_split: OK')
"

echo ""
echo "=========================================="
echo "SETUP COMPLETE — Ready to submit:"
echo "  sbatch slurm/sdp/encode_volumes.sh"
echo "=========================================="
