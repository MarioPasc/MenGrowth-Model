#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 0 (H5 repair) per-scan SLURM worker.
# Reads per-member soft-prob NIfTIs and re-aggregates corrected entropy /
# MI scalars for ONE scan_idx. Writes a per-task CSV to avoid race
# conditions during array execution; repair_patch.sh concatenates them.
# CPU-only — no GPU required (NIfTI read + numpy aggregation).
# ---------------------------------------------------------------------------

#SBATCH --job-name=uq_diag_repair
#SBATCH --output=uq_diag_repair_%A_%a.out
#SBATCH --error=uq_diag_repair_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00

set -euo pipefail

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-}}"
if [[ -z "${TASK_INDEX}" ]]; then
    echo "ERROR: neither SLURM_ARRAY_TASK_ID nor TASK_INDEX is set" >&2
    exit 1
fi

echo "=== Stage 0 repair: scan_idx=${TASK_INDEX} on $(hostname) at $(date) ==="
echo "Config:           ${CONFIG_PATH}"
echo "Per-task CSV dir: ${TASK_CSV_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

mkdir -p "${TASK_CSV_DIR}"
TASK_CSV="${TASK_CSV_DIR}/task_$(printf '%05d' "${TASK_INDEX}").csv"

python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.recompute_h5_uncertainty \
    --config "${CONFIG_PATH}" \
    --scan-indices "${TASK_INDEX}" \
    --out "${TASK_CSV}"

echo "Wrote ${TASK_CSV}"
echo "=== Done scan_idx=${TASK_INDEX} at $(date) ==="
