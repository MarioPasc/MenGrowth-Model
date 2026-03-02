#!/usr/bin/env bash
# =============================================================================
# DOMAIN GAP ANALYSIS — LAUNCHER
#
# Submits two SLURM jobs:
#   1. GPU job: feature extraction + Dice + domain metrics
#   2. CPU job (dependent): figure + LaTeX table generation
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model
#   bash slurm/domain_gap/launch.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "DOMAIN GAP ANALYSIS — LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model"
export CONDA_ENV_NAME="growth"
export CONFIG_PATH="${REPO_SRC}/experiments/domain_gap/config/picasso.yaml"

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

if [ -f "${CONFIG_PATH}" ]; then
    echo "  [OK]   Config file"
else
    echo "  [FAIL] Config not found: ${CONFIG_PATH}"
    exit 1
fi

python3 -c "
import yaml
from pathlib import Path

with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)

checkpoint = Path(cfg['paths']['checkpoint'])
h5_file = Path(cfg['paths']['h5_file'])
glioma_root = Path(cfg['paths']['glioma_root'])

assert checkpoint.exists(), f'Checkpoint not found: {checkpoint}'
print(f'  [OK]   Checkpoint: {checkpoint}')

assert h5_file.exists(), f'H5 file not found: {h5_file}'
print(f'  [OK]   H5 file: {h5_file}')

assert glioma_root.exists(), f'GLI data not found: {glioma_root}'
gli_count = len([d for d in glioma_root.iterdir() if d.is_dir() and d.name.startswith('BraTS-GLI-')])
print(f'  [OK]   GLI data: {gli_count} subjects in {glioma_root}')
"

if [ $? -ne 0 ]; then
    echo "  [FAIL] Pre-flight checks failed"
    exit 1
fi

echo ""

# ========================================================================
# STEP 1: Submit GPU job
# ========================================================================
echo "Submitting GPU job (feature extraction + Dice + metrics)..."

GPU_JOB_RAW=$(sbatch --parsable \
    --job-name="domain_gap" \
    --output="${SLURM_LOG_DIR}/gpu_%j.out" \
    --error="${SLURM_LOG_DIR}/gpu_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    "${SCRIPT_DIR}/run_domain_gap.sh")

GPU_JOB_ID="${GPU_JOB_RAW%%[_;]*}"

echo "  GPU job ID: ${GPU_JOB_ID}"
echo ""

# ========================================================================
# STEP 2: Submit CPU job (dependent on GPU completion)
# ========================================================================
echo "Submitting plots job (dependent on GPU job ${GPU_JOB_ID})..."

PLOTS_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${GPU_JOB_ID}" \
    --job-name="domain_gap_plots" \
    --output="${SLURM_LOG_DIR}/plots_%j.out" \
    --error="${SLURM_LOG_DIR}/plots_%j.err" \
    --export=ALL,OUTPUT_DIR="${OUTPUT_DIR}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    "${SCRIPT_DIR}/run_plots.sh")

echo "  Plots job ID: ${PLOTS_JOB_ID}"
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
echo "  squeue -j ${GPU_JOB_ID}          # GPU pipeline"
echo "  squeue -j ${PLOTS_JOB_ID}        # plots (dependent)"
echo ""
echo "Logs:"
echo "  tail -f ${SLURM_LOG_DIR}/gpu_${GPU_JOB_ID}.out"
echo "  tail -f ${SLURM_LOG_DIR}/plots_${PLOTS_JOB_ID}.out"
echo ""
echo "Cancel all:"
echo "  scancel ${GPU_JOB_ID} ${PLOTS_JOB_ID}"
echo ""
echo "Estimated timeline:"
echo "  GPU pipeline:  ~2-3h (feature extraction + Dice for 200 subjects)"
echo "  Plots/table:   ~5 min (CPU-only, dependent on GPU job)"
echo ""
