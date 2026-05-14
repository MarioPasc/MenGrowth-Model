#!/bin/bash
# -----------------------------------------------------------
# Per-task SLURM worker for conformal_calibration experiment.
# Receives TASK_INDEX and CONFIG_PATH via sbatch --export.
# -----------------------------------------------------------

#SBATCH --job-name=confcal
#SBATCH --output=confcal_%x_%A_%a.out
#SBATCH --error=confcal_%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00

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
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

python -m experiments.stage1_volumetric.conformal_calibration.run \
    --config "${CONFIG_PATH}" \
    --task-index "${TASK_INDEX}"

echo "=== Completed task ${TASK_INDEX} at $(date) ==="
