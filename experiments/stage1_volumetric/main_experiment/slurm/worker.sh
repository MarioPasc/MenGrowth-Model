#!/bin/bash
# -----------------------------------------------------------
# Per-task SLURM worker for main_experiment.
# Receives TASK_INDEX and CONFIG_PATH via sbatch --export.
# Resolves the manifest task at TASK_INDEX and runs it.
# -----------------------------------------------------------

#SBATCH --job-name=mainexp
#SBATCH --output=mainexp_%x_%A_%a.out
#SBATCH --error=mainexp_%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00

set -euo pipefail

# Resolve task index: SLURM_ARRAY_TASK_ID for array jobs, TASK_INDEX otherwise.
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-}}"
if [[ -z "${TASK_INDEX}" ]]; then
    echo "ERROR: neither SLURM_ARRAY_TASK_ID nor TASK_INDEX is set" >&2
    exit 1
fi

echo "=== Job ${SLURM_JOB_ID:-local} task=${TASK_INDEX} on $(hostname) at $(date) ==="
echo "Config: ${CONFIG_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-mengrowth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

python -m experiments.stage1_volumetric.main_experiment.run \
    --config "${CONFIG_PATH}" \
    --task-index "${TASK_INDEX}"

echo "=== Completed task ${TASK_INDEX} at $(date) ==="
