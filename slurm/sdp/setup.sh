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

# Picasso paths (override via env vars if different)
H5_FILE="${H5_FILE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/meningiomas/brats_men_train.h5}"
CHECKPOINT="${CHECKPOINT:-/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/BrainSegFounder_finetuned_BraTS/finetuned_model_fold_0.pt}"
LORA_ADAPTER="${LORA_ADAPTER:-/mnt/home/users/tic_163_uma/mpascual/execs/growth/results/lora_ablation_semantic_heads/conditions/lora_r8/adapter}"

echo "=========================================="
echo "SDP MODULE — SETUP"
echo "=========================================="
echo "Date:      $(date)"
echo "Hostname:  $(hostname)"
echo "Repo:      ${REPO_DIR}"
echo ""

# ---- Step 1: Git pull ----
echo "[1/4] Git pull..."
cd "${REPO_DIR}"
git pull --ff-only || echo "  WARNING: git pull failed (offline or conflicts)"
echo ""

# ---- Step 2: Conda environment ----
echo "[2/4] Activating conda environment: ${CONDA_ENV_NAME}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# Ensure h5py is installed
python -c "import h5py; print(f'  h5py: {h5py.__version__}')" || {
    echo "  Installing h5py..."
    pip install h5py
}
echo ""

# ---- Step 3: Verify data files ----
echo "[3/4] Verifying data files..."

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
check_file "${CHECKPOINT}" "BrainSegFounder checkpoint" || ok=false
check_file "${LORA_ADAPTER}" "LoRA adapter" || ok=false

if [ "$ok" = false ]; then
    echo ""
    echo "ERROR: Missing required files. Fix paths before submitting."
    exit 1
fi
echo ""

# ---- Step 4: Import checks ----
echo "[4/4] Import checks..."
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
