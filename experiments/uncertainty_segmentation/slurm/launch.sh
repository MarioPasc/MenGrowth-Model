#!/usr/bin/env bash
# =============================================================================
# LORA-ENSEMBLE — PICASSO LAUNCHER
#
# Submits a 3-step dependency chain:
#   STEP 1: Array job (M GPUs, parallel) — train M LoRA members
#   STEP 2: Evaluation job (1 GPU) — per-member + ensemble + baseline Dice
#   STEP 3: Inference job (1 GPU) — ensemble inference on MenGrowth
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model
#   bash experiments/uncertainty_segmentation/slurm/launch.sh
#   # or with explicit picasso override:
#   bash experiments/uncertainty_segmentation/slurm/launch.sh config/picasso/config_picasso.yaml
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "LORA-ENSEMBLE — PICASSO LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export REPO_SRC="${REPO_SRC:-${REPO_ROOT}}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"

# Base config is ALWAYS the full config.yaml
BASE_CONFIG="${MODULE_DIR}/config.yaml"
PICASSO_OVERRIDE="${MODULE_DIR}/config/picasso/config_picasso.yaml"

# Accept optional argument: either the base config or the picasso override
# In both cases, we always merge base + picasso override if override exists
export CONFIG_PATH="${1:-${BASE_CONFIG}}"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Base config: ${BASE_CONFIG}"
echo "  Override:    ${PICASSO_OVERRIDE}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo ""

# Activate conda for pre-flight
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
fi

# ========================================================================
# RESOLVE MERGED CONFIG → derive RUN_DIR
# ========================================================================
echo "Resolving merged configuration..."

MERGED_OUTPUT=""
MERGE_RC=0
MERGED_OUTPUT=$(python3 -c "
from omegaconf import OmegaConf
import os, sys, pathlib

# Always start from the complete base config
base_path = '${BASE_CONFIG}'
override_path = '${PICASSO_OVERRIDE}'

if not os.path.exists(base_path):
    print(f'FAIL Base config not found: {base_path}')
    sys.exit(1)

cfg = OmegaConf.load(base_path)
if os.path.exists(override_path):
    override = OmegaConf.load(override_path)
    cfg = OmegaConf.merge(cfg, override)
    print('[config] Merged base + picasso override')
else:
    print('[config] Using base config only (no picasso override found)')

r = cfg.lora.rank
M = cfg.ensemble.n_members
s = cfg.ensemble.base_seed
out = cfg.experiment.output_dir
run_dir = f'{out}/r{r}_M{M}_s{s}'

# Create run dir and save resolved config
pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, f'{run_dir}/config_snapshot.yaml', resolve=True)

print(f'RESULT {M} {run_dir}')
" 2>&1) || MERGE_RC=$?

echo "${MERGED_OUTPUT}" | grep -v "^RESULT" || true

if [ "${MERGE_RC}" -ne 0 ]; then
    echo "[FAIL] Config resolution failed (exit code ${MERGE_RC}):"
    echo "${MERGED_OUTPUT}"
    exit 1
fi

RESULT_LINE=$(echo "${MERGED_OUTPUT}" | grep "^RESULT" || true)

if [ -z "${RESULT_LINE}" ]; then
    echo "[FAIL] Config resolution produced no RESULT. Output was:"
    echo "${MERGED_OUTPUT}"
    exit 1
fi

read -r _ N_MEMBERS RUN_DIR <<< "${RESULT_LINE}"

export RUN_DIR
# Export the snapshot config for workers (complete, resolved)
export CONFIG_PATH="${RUN_DIR}/config_snapshot.yaml"

SLURM_LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${SLURM_LOG_DIR}"

echo ""
echo "Resolved:"
echo "  Run dir:  ${RUN_DIR}"
echo "  Config:   ${CONFIG_PATH}"
echo "  Members:  ${N_MEMBERS}"
echo "  Logs:     ${SLURM_LOG_DIR}"
echo ""

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo "Pre-flight checks..."
cd "${REPO_SRC}"
bash "${SCRIPT_DIR}/setup.sh" "${CONFIG_PATH}"
echo ""

# ========================================================================
# STEP 1: Training array job
# ========================================================================
ARRAY_MAX=$((N_MEMBERS - 1))
echo "Submitting STEP 1: training array job (${N_MEMBERS} members)..."

TRAIN_JOB_RAW=$(sbatch --parsable \
    --array="0-${ARRAY_MAX}" \
    --job-name="lora_ens_train" \
    --output="${SLURM_LOG_DIR}/train_%a_%j.out" \
    --error="${SLURM_LOG_DIR}/train_%a_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",RUN_DIR="${RUN_DIR}" \
    "${SCRIPT_DIR}/train_worker.sh")

TRAIN_JOB_ID="${TRAIN_JOB_RAW%%[_;]*}"
echo "  Train job ID: ${TRAIN_JOB_ID} (array 0-${ARRAY_MAX})"
echo ""

# ========================================================================
# STEP 2: Evaluation job (dependent on training)
# ========================================================================
echo "Submitting STEP 2: evaluation job (dependent on ${TRAIN_JOB_ID})..."

EVAL_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB_ID}" \
    --job-name="lora_ens_eval" \
    --output="${SLURM_LOG_DIR}/evaluate_%j.out" \
    --error="${SLURM_LOG_DIR}/evaluate_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",RUN_DIR="${RUN_DIR}" \
    "${SCRIPT_DIR}/evaluate_worker.sh")

echo "  Eval job ID: ${EVAL_JOB_ID}"
echo ""

# ========================================================================
# STEP 3: Inference job (dependent on evaluation)
# ========================================================================
echo "Submitting STEP 3: inference job (dependent on ${EVAL_JOB_ID})..."

INFER_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${EVAL_JOB_ID}" \
    --job-name="lora_ens_infer" \
    --output="${SLURM_LOG_DIR}/inference_%j.out" \
    --error="${SLURM_LOG_DIR}/inference_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",RUN_DIR="${RUN_DIR}" \
    "${SCRIPT_DIR}/inference_worker.sh")

echo "  Infer job ID: ${INFER_JOB_ID}"
echo ""

# ========================================================================
# MONITORING
# ========================================================================
echo "=========================================="
echo "ALL JOBS SUBMITTED"
echo "=========================================="
echo ""
echo "Dependency chain:"
echo "  ${TRAIN_JOB_ID} (train) → ${EVAL_JOB_ID} (evaluate) → ${INFER_JOB_ID} (inference)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -j ${TRAIN_JOB_ID}         # training array"
echo "  squeue -j ${EVAL_JOB_ID}           # evaluation"
echo "  squeue -j ${INFER_JOB_ID}          # inference"
echo ""
echo "Logs: ${SLURM_LOG_DIR}/"
echo ""
echo "Cancel all:"
echo "  scancel ${TRAIN_JOB_ID} ${EVAL_JOB_ID} ${INFER_JOB_ID}"
echo ""
echo "Estimated timeline:"
echo "  Training:   ~6-12h per member (parallel)"
echo "  Evaluation: ~2-4h (sequential, per-member + ensemble + baseline)"
echo "  Inference:  ~2-4h (sequential, MenGrowth cohort)"
echo "  Total:      ~16-20h"
