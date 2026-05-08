#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 2 per-task SLURM worker for the candidate-uncertainty-signal
# diagnostic. Receives TASK_INDEX + CONFIG_PATH via sbatch --export.
# Resolves the manifest task and runs LOPO LME-hetero with the appropriate
# σ²_v vector (candidate × scaling, or a control).
# CPU-only.
# ---------------------------------------------------------------------------

#SBATCH --job-name=uq_diag_stage2
#SBATCH --output=uq_diag_stage2_%A_%a.out
#SBATCH --error=uq_diag_stage2_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00

set -euo pipefail

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-}}"
if [[ -z "${TASK_INDEX}" ]]; then
    echo "ERROR: neither SLURM_ARRAY_TASK_ID nor TASK_INDEX is set" >&2
    exit 1
fi

echo "=== Stage 2 task=${TASK_INDEX} on $(hostname) at $(date) ==="
echo "Config: ${CONFIG_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.run \
    --config "${CONFIG_PATH}" \
    --task-index "${TASK_INDEX}"

echo "=== Completed task ${TASK_INDEX} at $(date) ==="
