#!/usr/bin/env bash
#SBATCH -J dd_explain
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# DOMAIN DIVERGENCE WORKER — PICASSO
#
# Runs the full domain divergence pipeline:
#   extract (GPU) → metrics (CPU) → CKA cross-stage (CPU) → CKA drift (GPU)
#   → decoder_patch (GPU, optional) → correlate (CPU) → figures (CPU)
#
# Time budget (A100, 192^3, N=150/domain):
#   - extract: 300 scans × 3s ≈ 15 min
#   - metrics: 5 stages × (1000 perms + 1000 bootstrap) ≈ 20 min
#   - cka_drift: 2 configs × 150 scans × 3s ≈ 15 min
#   - decoder_patch: 20 pairs × 5 stages × 3s ≈ 5 min
#   - figures: < 1 min
#   Total: ~60 min (fits in 4h wall-time with margin)
#
# Resources:
#   - 64 GB RAM for sklearn parallelism on feature arrays (150 × 768 = ~500 KB)
#   - 8 CPUs for sklearn cross-validation jobs
#   - 1 GPU for forward passes
#
# Expected env vars (exported by launch_domain_divergence.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH, ANALYSIS_CONFIG_PATH, OUTPUT_DIR
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "DOMAIN DIVERGENCE WORKER"
echo "=========================================="
echo "Started:           $(date)"
echo "Hostname:          $(hostname)"
echo "Parent config:     ${CONFIG_PATH:-NOT SET}"
echo "Analysis config:   ${ANALYSIS_CONFIG_PATH:-NOT SET}"
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
echo "Running domain divergence pipeline (all phases)..."

python -m experiments.uncertainty_segmentation.explainability.run_domain_divergence \
    --config "${CONFIG_PATH}" \
    --analysis-config "${ANALYSIS_CONFIG_PATH}" \
    --device cuda:0 \
    --output "${OUTPUT_DIR}" \
    --phase all

echo "  [OK] Domain divergence analysis complete"

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
