#!/usr/bin/env bash
#SBATCH -J lora_ens_infer
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# LORA-ENSEMBLE INFERENCE WORKER
#
# Runs after evaluation job completes:
#   Ensemble inference on MenGrowth longitudinal cohort → volume CSV
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH, RUN_DIR
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "LORA-ENSEMBLE INFERENCE WORKER"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "RUN_DIR:     ${RUN_DIR:-NOT SET}"
echo "Config:      ${CONFIG_PATH:-NOT SET}"
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

echo "Running ensemble inference on MenGrowth..."
python -m experiments.uncertainty_segmentation.run_inference \
    --config "${CONFIG_PATH}" \
    --dataset mengrowth \
    --run-dir "${RUN_DIR}"

echo "  [OK] Inference complete"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Finished: $(date)"
