#!/bin/bash
# Aggregator + statistics + figures (run after the array job completes).

#SBATCH --job-name=mainexp_analysis
#SBATCH --output=mainexp_analysis_%j.out
#SBATCH --error=mainexp_analysis_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00

set -euo pipefail

echo "=== Analysis job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="
echo "Config: ${CONFIG_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-mengrowth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

python -m experiments.stage1_volumetric.main_experiment.run \
    --config "${CONFIG_PATH}" \
    --analyze

echo "=== Analysis complete at $(date) ==="
