#!/usr/bin/env bash
#SBATCH -J explain
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# EXPLAINABILITY WORKER — PICASSO  (refactored: brain-masked TSI + ASI + DAD)
#
# Runs the full spec §10 pipeline at 192³ resolution.  Budget per condition:
#   - TSI + ASI: ~3 s/scan × 150 scans = ~8 min per condition
#   - 5 adapted members per rank => extra 5 × 8 min ≈ 40 min per rank
#   - DAD: 30 + 30 scans × 2 conditions ≈ 6 min + 1000-permutation test (~3 min)
# Total expected ≈ 60-90 min for one rank with 5 members.
#
# Expected env vars (exported by launch_tsi.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH, ANALYSIS_CONFIG_PATH, RANKS, OUTPUT_DIR
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "EXPLAINABILITY WORKER (TSI + ASI + DAD)"
echo "=========================================="
echo "Started:           $(date)"
echo "Hostname:          $(hostname)"
echo "Parent config:     ${CONFIG_PATH:-NOT SET}"
echo "Analysis config:   ${ANALYSIS_CONFIG_PATH:-NOT SET}"
echo "Ranks:             ${RANKS:-NOT SET}"
echo "Output:            ${OUTPUT_DIR:-NOT SET}"
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
    --format=csv -l 30 > "${LOG_DIR}/gpu_${SLURM_JOB_ID}.csv" 2>/dev/null &
GPU_MONITOR_PID=$!
echo "[gpu-monitor] Started (PID=${GPU_MONITOR_PID}, interval=30s)"

# ========================================================================
# RUN PIPELINE
# ========================================================================
echo "Running brain-masked TSI + ASI + DAD pipeline (ranks: ${RANKS})..."

# RANKS may be comma-separated; convert to space-separated for argparse nargs="+".
RANK_ARGS=$(echo "${RANKS}" | tr ',' ' ')

python -m experiments.uncertainty_segmentation.explainability.run_analysis \
    --config "${CONFIG_PATH}" \
    --analysis-config "${ANALYSIS_CONFIG_PATH}" \
    --device cuda:0 \
    --output "${OUTPUT_DIR}" \
    --ranks ${RANK_ARGS}

echo "  [OK] Analysis complete"

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
