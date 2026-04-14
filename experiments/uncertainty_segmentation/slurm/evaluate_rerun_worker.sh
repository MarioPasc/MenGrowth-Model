#!/usr/bin/env bash
#SBATCH -J lora_ens_eval_rerun
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# LORA-ENSEMBLE EVAL-ONLY RERUN WORKER
#
# Runs ONLY Step 2 of run_evaluate (per-subject ensemble evaluation). Steps 1
# (per-member) and 3 (baseline) are loaded from cached CSVs via --skip flags.
# Produces the streaming outputs that the original run missed:
#   evaluation/convergence_ensemble_dice_{wt,tc,et}.csv
#   evaluation/threshold_sensitivity.csv
#   predictions/brats_men_test/<scan>/member_{0..M-1}_probs.nii.gz
#
# Expected env vars (exported by relaunch_eval_streaming.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH, RUN_DIR
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "LORA-ENSEMBLE EVAL-ONLY RERUN WORKER"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "RUN_DIR:     ${RUN_DIR:-NOT SET}"
echo "Config:      ${CONFIG_PATH:-NOT SET}"
echo ""

# Environment setup (same pattern as evaluate_worker.sh)
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

# GPU monitoring
GPU_LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${GPU_LOG_DIR}"

echo "GPU Status (pre-eval):"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader
echo ""

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv -l 60 > "${GPU_LOG_DIR}/gpu_eval_rerun_${SLURM_JOB_ID}.csv" 2>/dev/null &
GPU_MONITOR_PID=$!
echo "[gpu-monitor] Started (PID=${GPU_MONITOR_PID}, interval=60s)"
echo ""

echo "Running eval-only re-run (skip per-member + baseline)..."
python -m experiments.uncertainty_segmentation.run_evaluate \
    --config "${CONFIG_PATH}" \
    --run-dir "${RUN_DIR}" \
    --skip-per-member \
    --skip-baseline

echo "  [OK] Evaluation rerun complete"

if [ -n "${GPU_MONITOR_PID:-}" ] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    echo "[gpu-monitor] Stopped"
fi

echo ""
echo "Verifying new outputs:"
for f in convergence_ensemble_dice_wt.csv convergence_ensemble_dice_tc.csv \
         convergence_ensemble_dice_et.csv threshold_sensitivity.csv; do
    p="${RUN_DIR}/evaluation/${f}"
    if [ -f "$p" ]; then
        echo "  [OK]      ${f} ($(wc -l < "$p") lines)"
    else
        echo "  [MISSING] ${f}"
    fi
done

PROBS_COUNT=$(find "${RUN_DIR}/predictions/brats_men_test" \
    -name "member_*_probs.nii.gz" 2>/dev/null | wc -l)
echo "  Per-member soft-prob files: ${PROBS_COUNT} (expect 150 scans x M members)"

echo ""
echo "GPU Status (post-eval):"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
echo ""

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Duration: $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "Finished: $(date)"
