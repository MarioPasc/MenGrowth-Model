#!/bin/bash
# experiments/stage1_volumetric/slurm/analysis_worker.sh
# -------------------------------------------------------
# Post-hoc analysis worker. Runs after all model jobs complete.
# Receives CONFIG_PATH via sbatch --export.

#SBATCH --job-name=growth_uq_analysis
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00

set -euo pipefail

echo "=== Analysis job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
echo "Config: ${CONFIG_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

python -m experiments.stage1_volumetric.run_analysis \
    --config "${CONFIG_PATH}"

echo "=== Analysis completed at $(date) ==="
