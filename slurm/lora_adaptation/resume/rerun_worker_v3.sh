#!/usr/bin/env bash
#SBATCH -J v3_rerun
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# LORA V3 — RE-RUN WORKER (post-training)
#
# Runs Steps 2-5 from existing checkpoints (skips training).
# Fixes applied: TAP extractor model-agnostic interface + OOM fallback.
#
# Expected env vars (exported by launch_rerun_v3.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH
# =============================================================================

set -euo pipefail

CONDITIONS=(
    baseline_frozen
    baseline
    lora_r4_full
    lora_r8_full
    lora_r16_full
    lora_r32_full
    lora_r64_full
)

COND="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

START_TIME=$(date +%s)
echo "=========================================="
echo "LORA V3 RE-RUN WORKER"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "Array ID:    ${SLURM_ARRAY_TASK_ID:-?}"
echo "Condition:   ${COND}"
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
from experiments.lora_ablation.run_ablation import main as _
print('Imports OK')
"

echo ""

# ========================================================================
# PIPELINE — SKIP TRAINING, RUN STEPS 2-5
# ========================================================================
echo "=========================================="
echo "RE-RUN PIPELINE: ${COND}"
echo "=========================================="
echo "(Training skipped — using existing checkpoints)"
echo ""

# Step 2: Extract features
echo "[1/4] Extracting features..."
python -m experiments.lora_ablation.run_ablation \
    --config "${CONFIG_PATH}" \
    extract --condition "${COND}"
echo "  [OK] Features extracted"

# Step 3: Domain features
echo "[2/4] Extracting domain features..."
python -m experiments.lora_ablation.run_ablation \
    --config "${CONFIG_PATH}" \
    domain --condition "${COND}" || echo "  [WARN] Domain features failed (non-fatal)"

# Step 4: Evaluate probes
echo "[3/4] Evaluating probes..."
python -m experiments.lora_ablation.run_ablation \
    --config "${CONFIG_PATH}" \
    probes --condition "${COND}"
echo "  [OK] Probes evaluated"

# Step 5: Test Dice (MEN only — GLI can be added later)
echo "[4/4] Computing test Dice..."
python -m experiments.lora_ablation.run_ablation \
    --config "${CONFIG_PATH}" \
    test-dice --condition "${COND}" --dataset men
echo "  [OK] Test Dice computed"

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
