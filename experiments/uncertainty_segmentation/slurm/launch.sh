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
#   bash experiments/uncertainty_segmentation/slurm/launch.sh              # r=8 (default)
#   bash experiments/uncertainty_segmentation/slurm/launch.sh --rank 4     # r=4 ablation
#   bash experiments/uncertainty_segmentation/slurm/launch.sh --rank 16    # r=16 ablation
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

# Parse arguments
RANK_OVERRIDE=""
ENCODER_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank)
            RANK_OVERRIDE="$2"
            shift 2
            ;;
        --encoder-only)
            ENCODER_ONLY=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Resolve rank override config (optional third merge layer)
RANK_OVERRIDE_PATH=""
if [ -n "${RANK_OVERRIDE}" ]; then
    RANK_OVERRIDE_PATH="${MODULE_DIR}/config/picasso/config_r${RANK_OVERRIDE}.yaml"
    if [ ! -f "${RANK_OVERRIDE_PATH}" ]; then
        echo "[FAIL] Rank override config not found: ${RANK_OVERRIDE_PATH}"
        echo "       Available: config_r2.yaml, config_r4.yaml, config_r16.yaml, config_r32.yaml (r=8 is default)"
        exit 1
    fi
fi

# Resolve encoder-only config (optional fourth merge layer)
ENCODER_ONLY_PATH=""
ENCODER_ONLY_PICASSO_PATH=""
if [ "${ENCODER_ONLY}" -eq 1 ]; then
    ENCODER_ONLY_PATH="${MODULE_DIR}/config/encoder_only.yaml"
    ENCODER_ONLY_PICASSO_PATH="${MODULE_DIR}/config/picasso/config_encoder_only_picasso.yaml"
    if [ ! -f "${ENCODER_ONLY_PATH}" ]; then
        echo "[FAIL] Encoder-only config not found: ${ENCODER_ONLY_PATH}"
        exit 1
    fi
    echo "  Mode: ENCODER-ONLY (frozen decoder + trainable output head)"
fi

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Base config: ${BASE_CONFIG}"
echo "  Override:    ${PICASSO_OVERRIDE}"
if [ -n "${RANK_OVERRIDE_PATH}" ]; then
    echo "  Rank ablation: ${RANK_OVERRIDE_PATH} (r=${RANK_OVERRIDE})"
fi
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
rank_override_path = '${RANK_OVERRIDE_PATH}'
encoder_only_path = '${ENCODER_ONLY_PATH}'
encoder_only_picasso_path = '${ENCODER_ONLY_PICASSO_PATH}'

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

# Apply rank ablation override (optional third layer)
if rank_override_path and os.path.exists(rank_override_path):
    rank_override = OmegaConf.load(rank_override_path)
    cfg = OmegaConf.merge(cfg, rank_override)
    print(f'[config] Applied rank override: r={cfg.lora.rank}, alpha={cfg.lora.alpha}')

# Apply encoder-only override (optional fourth layer)
if encoder_only_path and os.path.exists(encoder_only_path):
    eo = OmegaConf.load(encoder_only_path)
    cfg = OmegaConf.merge(cfg, eo)
    print(f'[config] Applied encoder-only: stages={list(cfg.lora.target_stages)}, freeze_decoder={cfg.training.freeze_decoder}')
if encoder_only_picasso_path and os.path.exists(encoder_only_picasso_path):
    eop = OmegaConf.load(encoder_only_picasso_path)
    cfg = OmegaConf.merge(cfg, eop)
    print('[config] Applied encoder-only picasso path override')

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
# Helper: extract numeric job ID from sbatch output.
# Picasso wraps sbatch in a Lua script, so output may be:
#   "Submitted batch job 12345"
#   "Submitted batch job 12345 on cluster picasso"
# We extract the FIRST sequence of digits after "job".
# ========================================================================
extract_job_id() {
    local sbatch_output="$1"
    local job_id
    job_id=$(echo "$sbatch_output" | grep -oP 'job\s+\K[0-9]+' | head -1)
    if [ -z "$job_id" ]; then
        # Fallback: extract any number from the output
        job_id=$(echo "$sbatch_output" | grep -oP '[0-9]+' | head -1)
    fi
    echo "$job_id"
}

# ========================================================================
# STEP 1: Training array job
# ========================================================================
ARRAY_MAX=$((N_MEMBERS - 1))
echo "Submitting STEP 1: training array job (${N_MEMBERS} members)..."

TRAIN_OUTPUT=$(sbatch \
    --array="0-${ARRAY_MAX}" \
    --job-name="lora_ens_train" \
    --output="${SLURM_LOG_DIR}/train_%a_%j.out" \
    --error="${SLURM_LOG_DIR}/train_%a_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",RUN_DIR="${RUN_DIR}" \
    "${SCRIPT_DIR}/train_worker.sh" 2>&1)

TRAIN_JOB_ID=$(extract_job_id "${TRAIN_OUTPUT}")
echo "  sbatch output: ${TRAIN_OUTPUT}"
echo "  Train job ID: ${TRAIN_JOB_ID} (array 0-${ARRAY_MAX})"

if [ -z "${TRAIN_JOB_ID}" ]; then
    echo "[FAIL] Could not extract train job ID"
    exit 1
fi
echo ""

# ========================================================================
# STEP 2: Evaluation job (dependent on training)
# ========================================================================
echo "Submitting STEP 2: evaluation job (dependent on ${TRAIN_JOB_ID})..."

EVAL_OUTPUT=$(sbatch \
    --dependency="afterok:${TRAIN_JOB_ID}" \
    --job-name="lora_ens_eval" \
    --output="${SLURM_LOG_DIR}/evaluate_%j.out" \
    --error="${SLURM_LOG_DIR}/evaluate_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",RUN_DIR="${RUN_DIR}" \
    "${SCRIPT_DIR}/evaluate_worker.sh" 2>&1)

EVAL_JOB_ID=$(extract_job_id "${EVAL_OUTPUT}")
echo "  sbatch output: ${EVAL_OUTPUT}"
echo "  Eval job ID: ${EVAL_JOB_ID}"

if [ -z "${EVAL_JOB_ID}" ]; then
    echo "[FAIL] Could not extract eval job ID"
    exit 1
fi
echo ""

# ========================================================================
# STEP 3: Inference job (dependent on evaluation)
# ========================================================================
echo "Submitting STEP 3: inference job (dependent on ${EVAL_JOB_ID})..."

INFER_OUTPUT=$(sbatch \
    --dependency="afterok:${EVAL_JOB_ID}" \
    --job-name="lora_ens_infer" \
    --output="${SLURM_LOG_DIR}/inference_%j.out" \
    --error="${SLURM_LOG_DIR}/inference_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",RUN_DIR="${RUN_DIR}" \
    "${SCRIPT_DIR}/inference_worker.sh" 2>&1)

INFER_JOB_ID=$(extract_job_id "${INFER_OUTPUT}")
echo "  sbatch output: ${INFER_OUTPUT}"
echo "  Infer job ID: ${INFER_JOB_ID}"

if [ -z "${INFER_JOB_ID}" ]; then
    echo "[FAIL] Could not extract inference job ID"
    exit 1
fi
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
echo "Estimated timeline (based on r8_M10_s42 actuals, scaled for M=20):"
echo "  Training:   ~2-9h per member (parallel, median ~3.5h)"
echo "  Evaluation: ~3h (20 members × 150 test scans, sequential)"
echo "  Inference:  ~2h (179 MenGrowth scans × 20 members, ~16s/scan/10members)"
echo "  Total:      ~7-14h wall clock (dominated by training)"
