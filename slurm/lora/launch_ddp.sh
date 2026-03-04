#!/usr/bin/env bash
# =============================================================================
# LORA DDP SWEEP — PICASSO LAUNCHER
#
# Submits a 2-element array job (2 A100 GPUs each, 24h walltime) plus a
# dependent CPU-only analysis job.
#
# Conditions (SLURM_ARRAY_TASK_ID):
#   0: baseline        (frozen encoder + trainable decoder)
#   1: dual_r8         (LoRA r8 + VICReg + dual-domain MEN+GLI)
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model
#   bash slurm/lora/launch_ddp.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "LORA DDP SWEEP — PICASSO LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model"
export CONDA_ENV_NAME="growth"
export CONFIG_PATH="${REPO_SRC}/experiments/lora/config/picasso/ddp_dual_domain.yaml"

echo "Activating conda environment: ${CONDA_ENV_NAME}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo ""

# Extract output dir from config
OUTPUT_DIR=$(python3 -c "
import yaml
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment']['output_dir'])
")
export OUTPUT_DIR

SLURM_LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${SLURM_LOG_DIR}"

echo "Configuration:"
echo "  Repo:       ${REPO_SRC}"
echo "  Config:     ${CONFIG_PATH}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Logs:       ${SLURM_LOG_DIR}"
echo "  Conda env:  ${CONDA_ENV_NAME}"
echo ""

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo "Pre-flight checks:"

# Config exists
if [ -f "${CONFIG_PATH}" ]; then
    echo "  [OK]   Config file"
else
    echo "  [FAIL] Config not found: ${CONFIG_PATH}"
    exit 1
fi

# No placeholder paths
if grep -q "/PLACEHOLDER/" "${CONFIG_PATH}"; then
    echo "  [FAIL] Config contains /PLACEHOLDER/ paths"
    exit 1
fi
echo "  [OK]   No placeholder paths"

# Verify critical paths from config
python3 -c "
import yaml
from pathlib import Path

with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)

checkpoint = Path(cfg['paths']['checkpoint'])
men_h5 = Path(cfg['paths']['men_h5_file'])
gli_h5 = Path(cfg['paths']['gli_h5_file'])

assert checkpoint.exists(), f'Checkpoint not found: {checkpoint}'
print(f'  [OK]   Checkpoint: {checkpoint}')

assert men_h5.exists(), f'MEN H5 not found: {men_h5}'
print(f'  [OK]   MEN H5: {men_h5}')

assert gli_h5.exists(), f'GLI H5 not found: {gli_h5}'
print(f'  [OK]   GLI H5: {gli_h5}')

n_cond = len(cfg['conditions'])
print(f'  [OK]   {n_cond} conditions defined')
"

if [ $? -ne 0 ]; then
    echo "  [FAIL] Pre-flight checks failed"
    exit 1
fi

# Quick import check
python3 -c "
from experiments.lora.engine.ddp_utils import setup_ddp, DistributedDomainBalancedSampler
print('  [OK]   DDP imports')
"

echo ""

# ========================================================================
# CONDITION NAMES (for per-element job naming)
# ========================================================================
CONDITION_NAMES=("baseline" "dual_r8")
N_CONDITIONS=${#CONDITION_NAMES[@]}
LAST_IDX=$((N_CONDITIONS - 1))

# ========================================================================
# SUBMIT ARRAY JOB (2 conditions, 2 GPUs each)
# ========================================================================
echo "Submitting DDP training array job (${N_CONDITIONS} conditions, 2 GPUs each)..."
echo "  Conditions: ${CONDITION_NAMES[*]}"

ARRAY_JOB_RAW=$(sbatch --parsable \
    --array=0-${LAST_IDX}%${N_CONDITIONS} \
    --job-name="ddp_lora" \
    --output="${SLURM_LOG_DIR}/train_%a_%j.out" \
    --error="${SLURM_LOG_DIR}/train_%a_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    "${SCRIPT_DIR}/train_worker_ddp.sh")

# --parsable on array jobs may return "JOBID;cluster" — extract base ID
ARRAY_JOB_ID="${ARRAY_JOB_RAW%%[_;]*}"

echo "  Array job ID: ${ARRAY_JOB_ID} (${N_CONDITIONS} elements, 0-${LAST_IDX})"
for i in $(seq 0 ${LAST_IDX}); do
    echo "    Element ${i} -> ${CONDITION_NAMES[$i]} (job ${ARRAY_JOB_ID}_${i})"
done
echo ""

# Note: per-element job names are set by train_worker_ddp.sh at runtime
# via scontrol (SLURM doesn't support %a in --job-name at submission time)
echo ""

# ========================================================================
# SUBMIT ANALYSIS JOB (dependent on array completion)
# ========================================================================
echo "Submitting analysis job (dependent on array ${ARRAY_JOB_ID})..."

# Use base array job ID — SLURM waits for ALL elements to complete
echo "  Dependency: afterok:${ARRAY_JOB_ID}"

ANALYSIS_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --job-name="ddp_analysis" \
    --constraint=cpu \
    --time=02:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --output="${SLURM_LOG_DIR}/analysis_%j.out" \
    --error="${SLURM_LOG_DIR}/analysis_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    "${SCRIPT_DIR}/analysis_worker_v3.sh")

echo "  Analysis job ID: ${ANALYSIS_JOB_ID}"
echo ""

# ========================================================================
# MONITORING COMMANDS
# ========================================================================
echo "=========================================="
echo "ALL JOBS SUBMITTED"
echo "=========================================="
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -j ${ARRAY_JOB_ID}         # training array (2x DDP)"
echo "  squeue -j ${ANALYSIS_JOB_ID}       # analysis (dependent)"
echo ""
echo "Per-condition logs:"
for i in $(seq 0 ${LAST_IDX}); do
    echo "  tail -f ${SLURM_LOG_DIR}/train_${i}_*.out   # ${CONDITION_NAMES[$i]}"
done
echo ""
echo "Cancel all:"
echo "  scancel ${ARRAY_JOB_ID} ${ANALYSIS_JOB_ID}"
echo ""
echo "Estimated timeline:"
echo "  Training:  ~3-6h per condition (2x GPU DDP)"
echo "  Analysis:  ~30 min after all training completes"
echo "  Total:     ~6.5h (both conditions in parallel)"
