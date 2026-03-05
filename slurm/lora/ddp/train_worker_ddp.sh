#!/usr/bin/env bash
#SBATCH -J ddp_lora_train
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:2

# =============================================================================
# LORA DDP — PER-CONDITION TRAINING WORKER (2 GPUs)
#
# Each SLURM array element trains one condition using torchrun (2-GPU DDP),
# then extracts features, runs domain analysis, evaluates probes, computes
# test Dice, evaluates feature quality, and generates summary tables.
# All compute-heavy work happens here; plotting/reports are done locally.
#
# Expected env vars (exported by launch_ddp.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH
# =============================================================================

set -euo pipefail

NGPUS=2

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
echo "LORA DDP TRAINING WORKER"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "Array ID:    ${SLURM_ARRAY_TASK_ID:-?}"
echo "Condition:   ${COND}"
echo "Config:      ${CONFIG_PATH:-NOT SET}"
echo "GPUs req'd:  ${NGPUS}"
echo "CUDA_VIS:    ${CUDA_VISIBLE_DEVICES:-unset}"
echo ""

# Update SLURM job name to include condition
scontrol update JobId="${SLURM_JOB_ID}" JobName="ddp_${COND}" 2>/dev/null || true

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
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print(f'  GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
echo ""

# ========================================================================
# GPU COUNT ASSERTION (critical: prevents world-size bug)
# ========================================================================
echo "=========================================="
echo "GPU VERIFICATION"
echo "=========================================="
nvidia-smi -L
ACTUAL_GPUS=$(nvidia-smi -L | wc -l)
if [ "${ACTUAL_GPUS}" -ne "${NGPUS}" ]; then
    echo ""
    echo "[FATAL] Expected ${NGPUS} GPUs but found ${ACTUAL_GPUS}. Aborting."
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    exit 1
fi
echo ""
echo "[OK] GPU count: ${ACTUAL_GPUS}/${NGPUS}"
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
from experiments.lora.run import main as _
from experiments.lora.engine.ddp_utils import setup_ddp
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

# Master port offset prevents conflicts between array elements
MASTER_PORT=$((29500 + ${SLURM_ARRAY_TASK_ID:-0}))

step_start() { STEP_T0=$(date +%s); }
step_done() {
    local elapsed=$(($(date +%s) - STEP_T0))
    echo "  [OK] Done in ${elapsed}s"
}

# Step 1: Train (DDP, 2 GPUs)
echo "[1/7] Training ${COND} (DDP, ${NGPUS} GPUs, port=${MASTER_PORT})..."
step_start
torchrun \
    --nproc_per_node=${NGPUS} \
    --master_port=${MASTER_PORT} \
    -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    train-ddp --condition "${COND}"
step_done

# Step 2: Extract features (single GPU)
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
