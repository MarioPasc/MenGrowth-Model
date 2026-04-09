#!/usr/bin/env bash
#SBATCH -J tsi_explain
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# TSI EXPLAINABILITY WORKER — PICASSO
#
# Runs TSI analysis at full 192³ resolution on all test scans.
# 150 scans × 2 conditions × ~3s/scan + 3 ranks × ~3s/scan = ~20 min.
#
# Expected env vars (exported by launch_tsi.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH, TSI_CONFIG_PATH, RANKS, OUTPUT_DIR
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "TSI EXPLAINABILITY WORKER"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "Config:      ${CONFIG_PATH:-NOT SET}"
echo "TSI Config:  ${TSI_CONFIG_PATH:-NOT SET}"
echo "Ranks:       ${RANKS:-NOT SET}"
echo "Output:      ${OUTPUT_DIR:-NOT SET}"
echo ""

# Environment setup
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module; assuming conda in PATH."

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_SRC}"

# ========================================================================
# GPU MONITORING
# ========================================================================
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "GPU Status (pre-analysis):"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader
echo ""

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv -l 30 > "${LOG_DIR}/gpu_tsi_${SLURM_JOB_ID}.csv" 2>/dev/null &
GPU_MONITOR_PID=$!
echo "[gpu-monitor] Started (PID=${GPU_MONITOR_PID}, interval=30s)"

# ========================================================================
# VALIDATION (1 scan, verify hidden states)
# ========================================================================
echo "Running validation..."
FIRST_RANK=$(echo "${RANKS}" | tr ',' ' ' | awk '{print $1}')
python -m experiments.uncertainty_segmentation.explainability.run_tsi \
    --config "${CONFIG_PATH}" \
    --tsi-config "${TSI_CONFIG_PATH}" \
    --device cuda:0 \
    --rank ${FIRST_RANK} \
    --validate-only

echo "  [OK] Validation passed"
echo ""

# ========================================================================
# FULL ANALYSIS (all ranks)
# ========================================================================
echo "Running full TSI analysis (ranks: ${RANKS})..."
python -m experiments.uncertainty_segmentation.explainability.run_tsi \
    --config "${CONFIG_PATH}" \
    --tsi-config "${TSI_CONFIG_PATH}" \
    --device cuda:0 \
    --rank ${RANKS//,/ }

echo "  [OK] TSI analysis complete"

# ========================================================================
# CLEANUP
# ========================================================================
if [ -n "${GPU_MONITOR_PID:-}" ] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    echo "[gpu-monitor] Stopped"
fi

echo "GPU Status (post-analysis):"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
echo ""

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Finished: $(date)"
