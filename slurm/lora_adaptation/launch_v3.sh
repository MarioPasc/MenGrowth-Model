#!/usr/bin/env bash
# =============================================================================
# LORA V3 RANK SWEEP — PICASSO LAUNCHER
#
# Submits a 7-element array job (1 A100 GPU each, 12h walltime) plus a
# dependent CPU-only analysis job that regenerates figures/tables/reports.
#
# Conditions (SLURM_ARRAY_TASK_ID):
#   0: baseline_frozen    3: lora_r8_full     6: lora_r64_full
#   1: baseline           4: lora_r16_full
#   2: lora_r4_full       5: lora_r32_full
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model
#   bash slurm/lora_adaptation/launch_v3.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "LORA V3 RANK SWEEP — PICASSO LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model"
export CONDA_ENV_NAME="growth"
export CONFIG_PATH="${REPO_SRC}/experiments/lora_ablation/config/picasso/v3_rank_sweep.yaml"

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
h5_file = Path(cfg['paths']['h5_file'])

assert checkpoint.exists(), f'Checkpoint not found: {checkpoint}'
print(f'  [OK]   Checkpoint: {checkpoint}')

assert h5_file.exists(), f'H5 file not found: {h5_file}'
print(f'  [OK]   H5 file: {h5_file}')

n_cond = len(cfg['conditions'])
print(f'  [OK]   {n_cond} conditions defined')
"

if [ $? -ne 0 ]; then
    echo "  [FAIL] Pre-flight checks failed"
    exit 1
fi

# Quick conda/import check
if command -v conda >/dev/null 2>&1; then
    echo "  [OK]   Conda available"
else
    echo "  [WARN] Conda not in PATH (will be loaded by workers)"
fi

echo ""

# ========================================================================
# STEP 1: Generate data splits (fast, on login node)
# ========================================================================
echo "Generating data splits..."
cd "${REPO_SRC}"
python3 -m experiments.lora_ablation.run_ablation \
    --config "${CONFIG_PATH}" splits
echo "  [OK]   Data splits generated"
echo ""

# ========================================================================
# STEP 2: Submit array job (7 conditions, 1 GPU each)
# ========================================================================
echo "Submitting training array job..."

ARRAY_JOB_ID=$(sbatch --parsable \
    --array=0-6 \
    --job-name="v3_rank" \
    --output="${SLURM_LOG_DIR}/train_%a_%j.out" \
    --error="${SLURM_LOG_DIR}/train_%a_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    "${SCRIPT_DIR}/train_worker_v3.sh")

echo "  Array job ID: ${ARRAY_JOB_ID} (7 elements, 0-6)"
echo ""

# ========================================================================
# STEP 3: Submit analysis job (dependent on array completion)
# ========================================================================
echo "Submitting analysis job (dependent on array ${ARRAY_JOB_ID})..."

ANALYSIS_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --job-name="v3_analysis" \
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
echo "  squeue -j ${ARRAY_JOB_ID}         # training array"
echo "  squeue -j ${ANALYSIS_JOB_ID}       # analysis (dependent)"
echo ""
echo "Per-condition logs:"
echo "  tail -f ${SLURM_LOG_DIR}/train_<ARRAY_ID>_<JOB_ID>.out"
echo ""
echo "Cancel all:"
echo "  scancel ${ARRAY_JOB_ID} ${ANALYSIS_JOB_ID}"
echo ""
echo "Estimated timeline:"
echo "  Training:  ~6-12h per condition (12h limit)"
echo "  Analysis:  ~30 min after all training completes"
echo "  Total:     ~12.5h (all 7 train in parallel)"
