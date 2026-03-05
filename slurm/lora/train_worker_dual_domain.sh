#!/usr/bin/env bash
#SBATCH -J dd_lora_train
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# DUAL-DOMAIN LORA — PER-CONDITION TRAINING WORKER (1 GPU)
#
# Each SLURM array element trains one condition on a single GPU, then
# extracts features, runs domain analysis, evaluates probes, computes
# test Dice, evaluates feature quality, and generates summary tables.
#
# Expected env vars (exported by launch_dual_domain.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH
# =============================================================================

set -euo pipefail

# ========================================================================
# CONDITION MAPPING
# ========================================================================
CONDITIONS=(
    baseline
    dual_r8
    men_r8
)

COND="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

START_TIME=$(date +%s)
echo "=========================================="
echo "DUAL-DOMAIN LORA TRAINING WORKER"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "Array ID:    ${SLURM_ARRAY_TASK_ID:-?}"
echo "Condition:   ${COND}"
echo "Config:      ${CONFIG_PATH:-NOT SET}"
echo ""

# Update SLURM job name to include condition
scontrol update JobId="${SLURM_JOB_ID}" JobName="dd_${COND}" 2>/dev/null || true

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

# Quick import check
python -c "
from experiments.lora.engine.train_condition import train_condition
print('Imports OK')
"

echo ""

# ========================================================================
# MEMORY OPTIMIZATION
# ========================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========================================================================
# PIPELINE (per condition)
# ========================================================================
echo "=========================================="
echo "PIPELINE: ${COND}"
echo "=========================================="

step_start() { STEP_T0=$(date +%s); }
step_done() {
    local elapsed=$(($(date +%s) - STEP_T0))
    echo "  [OK] Done in ${elapsed}s"
}

# Step 1: Train (single GPU)
echo "[1/7] Training ${COND}..."
step_start
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    train --condition "${COND}"
step_done

# Step 2: Extract features
echo "[2/7] Extracting features for ${COND}..."
step_start
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    extract --condition "${COND}"
step_done

# Step 3: Domain gap
echo "[3/7] Domain gap analysis for ${COND}..."
step_start
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    domain-gap --condition "${COND}" || echo "  [WARN] Domain gap failed (non-fatal)"
step_done

# Step 4: Evaluate probes
echo "[4/7] Evaluating probes for ${COND}..."
step_start
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    probes --condition "${COND}"
step_done

# Step 5: Test Dice
echo "[5/7] Computing test Dice for ${COND}..."
step_start
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    dice --condition "${COND}"
step_done

# Step 6: Feature quality (CPU, reads cached features)
echo "[6/7] Feature quality evaluation for ${COND}..."
step_start
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    feature-quality --condition "${COND}" || echo "  [WARN] Feature quality failed (non-fatal)"
step_done

# Step 7: Generate summary tables (CPU, aggregates all JSONs)
echo "[7/7] Generating summary tables for ${COND}..."
step_start
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    generate-tables || echo "  [WARN] Table generation failed (non-fatal)"
step_done

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "WORKER COMPLETED: ${COND}"
echo "=========================================="
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Finished: $(date)"
