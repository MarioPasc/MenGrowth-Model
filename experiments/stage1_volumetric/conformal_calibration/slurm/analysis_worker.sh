#!/bin/bash
# -----------------------------------------------------------
# Aggregator + statistics + figures worker.
# Runs after the array job completes (dependency=afterany).
# -----------------------------------------------------------

#SBATCH --job-name=confcal_analysis
#SBATCH --output=confcal_analysis_%j.out
#SBATCH --error=confcal_analysis_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00

set -euo pipefail

echo "=== Analysis job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="
echo "Config: ${CONFIG_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

python -m experiments.stage1_volumetric.conformal_calibration.run \
    --config "${CONFIG_PATH}" \
    --analyze

echo "=== Analysis complete at $(date) ==="
