#!/usr/bin/env bash
# =============================================================================
# LORA-ENSEMBLE — EVAL-ONLY RELAUNCH (streaming D_k + threshold sensitivity)
#
# Re-submits ONLY Step 2 (per-subject ensemble evaluation) of run_evaluate
# for an existing run dir, reusing cached per-member and baseline CSVs. Used
# to back-fill the streaming Dice(ensemble_k) and threshold-sensitivity
# analyses on runs that finished before save_per_member_probs_all and the
# streaming code path were wired in.
#
# What it does:
#   1. Patches ${RUN_DIR}/config_snapshot.yaml with the streaming flags
#      (idempotent: keys are overwritten with the production values).
#   2. Submits a one-off sbatch job that runs:
#        run_evaluate --skip-per-member --skip-baseline
#      so Step 1 (per-member, ~85 min) and Step 3 (baseline) are loaded from
#      cache and only Step 2 (~90 min on A100) actually executes.
#
# Disk cost: ~20-30 GB extra (per-member soft probs, uint8 NIfTI) per run.
#
# Usage (Picasso login node):
#   bash experiments/uncertainty_segmentation/slurm/relaunch_eval_streaming.sh           # r=8
#   bash experiments/uncertainty_segmentation/slurm/relaunch_eval_streaming.sh --rank 4
#   bash experiments/uncertainty_segmentation/slurm/relaunch_eval_streaming.sh --run-dir /path/to/r8_M20_s42
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="${REPO_SRC:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"

# Default run-dir parent for the encoder-only experiment on Picasso.
DEFAULT_RUN_PARENT="/mnt/home/users/tic_163_uma/mpascual/execs/growth/uncertainty_segmentation_encoder_only"
N_MEMBERS_DEFAULT=20
BASE_SEED_DEFAULT=42

RANK=8
RUN_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank)     RANK="$2"; shift 2 ;;
        --run-dir)  RUN_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *)
            echo "[FAIL] Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "${RUN_DIR}" ]; then
    RUN_DIR="${DEFAULT_RUN_PARENT}/r${RANK}_M${N_MEMBERS_DEFAULT}_s${BASE_SEED_DEFAULT}"
fi

CONFIG_PATH="${RUN_DIR}/config_snapshot.yaml"
SLURM_LOG_DIR="${RUN_DIR}/logs"

echo "=========================================="
echo "EVAL-ONLY RELAUNCH (streaming analyses)"
echo "=========================================="
echo "Time:        $(date)"
echo "Repo:        ${REPO_SRC}"
echo "Run dir:     ${RUN_DIR}"
echo "Config:      ${CONFIG_PATH}"
echo "Conda env:   ${CONDA_ENV_NAME}"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if [ ! -d "${RUN_DIR}" ]; then
    echo "[FAIL] Run dir does not exist: ${RUN_DIR}"
    exit 1
fi
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[FAIL] Config snapshot missing: ${CONFIG_PATH}"
    exit 1
fi

# Sanity: cached per-member + baseline CSVs must exist (we --skip them).
for f in evaluation/per_member_test_dice.csv evaluation/baseline_test_dice.csv; do
    if [ ! -f "${RUN_DIR}/${f}" ]; then
        echo "[FAIL] Missing cached CSV (needed by --skip-* flags): ${RUN_DIR}/${f}"
        echo "       Re-run the full evaluation chain instead."
        exit 1
    fi
done

# Adapter checkpoints must exist (otherwise inference cannot reload members).
ADAPTER_COUNT=$(find "${RUN_DIR}/adapters" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "${ADAPTER_COUNT}" -eq 0 ]; then
    echo "[FAIL] No adapter checkpoints under ${RUN_DIR}/adapters"
    exit 1
fi
echo "[OK] Found ${ADAPTER_COUNT} adapter checkpoints"

# Activate conda for the patch step (we use python3 + yaml on the login node).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}" || true
fi

# ---------------------------------------------------------------------------
# Patch config_snapshot.yaml with the streaming flags
# ---------------------------------------------------------------------------
echo "Patching ${CONFIG_PATH} with streaming flags..."

CONFIG_PATH="${CONFIG_PATH}" python3 - <<'PY'
import os, sys
import yaml

p = os.environ['CONFIG_PATH']
with open(p) as f:
    cfg = yaml.safe_load(f) or {}

cfg.setdefault('inference', {})['save_per_member_probs_all'] = True
cfg.setdefault('evaluation', {}).update({
    'compute_ensemble_k_dice': True,
    'compute_threshold_sensitivity': True,
    'threshold_grid': [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    ],
})

with open(p, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'  Patched: {p}')
PY

echo "Verification:"
grep -E "save_per_member_probs_all|compute_ensemble_k_dice|compute_threshold_sensitivity" "${CONFIG_PATH}" \
    | sed 's/^/  /'
echo ""

# ---------------------------------------------------------------------------
# Disk-space sanity (~30 GB needed for soft probs)
# ---------------------------------------------------------------------------
echo "Disk usage of run-dir parent:"
df -h "$(dirname "${RUN_DIR}")" | sed 's/^/  /'
echo ""

mkdir -p "${SLURM_LOG_DIR}"

# ---------------------------------------------------------------------------
# Submit eval-only re-run
# ---------------------------------------------------------------------------
echo "Submitting eval-only re-run..."

SUBMIT_OUTPUT=$(sbatch \
    --job-name=lora_ens_eval_rerun \
    --time=0-04:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=16G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${SLURM_LOG_DIR}/evaluate_rerun_%j.out" \
    --error="${SLURM_LOG_DIR}/evaluate_rerun_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",RUN_DIR="${RUN_DIR}" \
    --wrap='
set -euo pipefail
START_TIME=$(date +%s)
echo "=========================================="
echo "EVAL-ONLY RERUN WORKER"
echo "=========================================="
echo "Started:  $(date)"
echo "Hostname: $(hostname)"
echo "RUN_DIR:  ${RUN_DIR}"
echo "Config:   ${CONFIG_PATH}"
echo ""

# Conda
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
echo "  Per-member soft-prob files: ${PROBS_COUNT} (expect 150 scans x 20 members = 3000)"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Duration: $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "Finished: $(date)"
' 2>&1)

JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oP 'job\s+\K[0-9]+' | head -1)
if [ -z "${JOB_ID}" ]; then
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oP '[0-9]+' | head -1)
fi

echo "  sbatch output: ${SUBMIT_OUTPUT}"
echo "  Job ID:        ${JOB_ID}"
echo ""
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${SLURM_LOG_DIR}/evaluate_rerun_${JOB_ID}.err"
echo ""
echo "Cancel:"
echo "  scancel ${JOB_ID}"
echo ""
echo "Expected wall-clock: ~90 min on A100 (Step 2 only; Steps 1,3 are loaded from cache)."
