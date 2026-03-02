#!/usr/bin/env bash
#SBATCH -J domain_gap
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# DOMAIN GAP ANALYSIS — GPU PIPELINE
#
# Extracts encoder10 features and computes Dice for both BraTS-GLI and
# BraTS-MEN using frozen BrainSegFounder, then computes domain shift metrics.
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "DOMAIN GAP ANALYSIS — GPU PIPELINE"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "Config:      ${CONFIG_PATH:-NOT SET}"
echo ""

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done

if [ "$module_loaded" -eq 0 ]; then
    echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "=========================================="
echo "ENVIRONMENT"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

# ========================================================================
# PRE-FLIGHT
# ========================================================================
cd "${REPO_SRC}"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[FAIL] Config not found: ${CONFIG_PATH}"
    exit 1
fi

python -c "
from experiments.domain_gap.run_domain_gap import main as _
print('Imports OK')
"

echo ""

# ========================================================================
# RUN PIPELINE
# ========================================================================
echo "=========================================="
echo "RUNNING DOMAIN GAP PIPELINE"
echo "=========================================="
echo ""

python -m experiments.domain_gap.run_domain_gap --config "${CONFIG_PATH}"

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "DOMAIN GAP GPU PIPELINE COMPLETED"
echo "=========================================="
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Finished: $(date)"
