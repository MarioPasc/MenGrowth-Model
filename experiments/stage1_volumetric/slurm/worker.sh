#!/bin/bash
# experiments/stage1_volumetric/slurm/worker.sh
# -----------------------------------------------
# Per-model SLURM worker for Stage 1 UQ growth prediction.
# Receives MODEL_NAME and CONFIG_PATH via sbatch --export.
#
# Usage (submitted by launcher.sh, not called directly):
#   sbatch --export=ALL,MODEL_NAME=LME,CONFIG_PATH=configs/picasso.yaml worker.sh

#SBATCH --job-name=growth_uq
#SBATCH --output=growth_uq_%x_%j.out
#SBATCH --error=growth_uq_%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00

set -euo pipefail

# --- Environment ---
echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
echo "Model: ${MODEL_NAME}"
echo "Config: ${CONFIG_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

# --- Run ---
python -m experiments.stage1_volumetric.run_single_model \
    --model "${MODEL_NAME}" \
    --config "${CONFIG_PATH}" \
    --force

echo "=== Completed ${MODEL_NAME} at $(date) ==="
