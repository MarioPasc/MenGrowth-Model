#!/usr/bin/env bash
#SBATCH -J lora_ens_train
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# LORA-ENSEMBLE TRAINING WORKER
#
# Each SLURM array element trains one LoRA ensemble member.
# MEMBER_ID = SLURM_ARRAY_TASK_ID
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH
# =============================================================================

set -euo pipefail

MEMBER_ID="${SLURM_ARRAY_TASK_ID}"

START_TIME=$(date +%s)
echo "=========================================="
echo "LORA-ENSEMBLE TRAINING WORKER"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "Array ID:    ${SLURM_ARRAY_TASK_ID:-?}"
echo "Member ID:   ${MEMBER_ID}"
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
from experiments.uncertainty_segmentation.engine.train_member import train_single_member
print('Imports OK')
"
echo ""

# ========================================================================
# GPU MONITORING
# ========================================================================
GPU_LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${GPU_LOG_DIR}"

echo "GPU Status (pre-training):"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader
echo ""

# Background GPU logger (sample every 60s)
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv -l 60 > "${GPU_LOG_DIR}/gpu_train_${MEMBER_ID}_${SLURM_JOB_ID}.csv" 2>/dev/null &
GPU_MONITOR_PID=$!
echo "[gpu-monitor] Started (PID=${GPU_MONITOR_PID}, interval=60s)"

# ========================================================================
# TRAIN MEMBER
# ========================================================================
echo "=========================================="
echo "TRAINING MEMBER ${MEMBER_ID}"
echo "=========================================="

python -m experiments.uncertainty_segmentation.run_train \
    --config "${CONFIG_PATH}" \
    --member-id "${MEMBER_ID}" \
    --run-dir "${RUN_DIR}"

echo "  [OK] Training complete for member ${MEMBER_ID}"

# Stop GPU monitor
if [ -n "${GPU_MONITOR_PID:-}" ] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    echo "[gpu-monitor] Stopped"
fi

echo "GPU Status (post-training):"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
echo ""

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "WORKER COMPLETED: member_${MEMBER_ID}"
echo "=========================================="
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Finished: $(date)"
