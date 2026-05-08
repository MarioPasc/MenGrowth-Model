#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the LoRA re-inference for the v5 MenGrowth H5: an array of shards
# (one GPU each) followed by a dependent merge + H5 patch.
#
# Usage:
#   bash reinfer_launcher.sh <LORA_CONFIG> <LORA_RUN_DIR> <MENGROWTH_H5> <OUTPUT_DIR> [N_SHARDS] [--dry-run]
#
# Defaults: N_SHARDS=8 (179 scans / 8 ≈ 22 per shard, ~12 min wall on A100).
# Assumes conda env "growth" and repo at /mnt/home/users/.../MenGrowth-Model.
# ---------------------------------------------------------------------------

set -euo pipefail

DRY_RUN=false
ARGS=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=true ;;
        *) ARGS+=("$a") ;;
    esac
done

if [[ "${#ARGS[@]}" -lt 4 ]]; then
    echo "Usage: $0 <LORA_CONFIG> <LORA_RUN_DIR> <MENGROWTH_H5> <OUTPUT_DIR> [N_SHARDS] [--dry-run]" >&2
    exit 1
fi

export LORA_CONFIG="${ARGS[0]}"
export LORA_RUN_DIR="${ARGS[1]}"
export MENGROWTH_H5="${ARGS[2]}"
export OUTPUT_DIR="${ARGS[3]}"
export N_SHARDS="${ARGS[4]:-8}"
export CONDA_ENV="${CONDA_ENV:-growth}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
LOGS_DIR="${LOGS_DIR:-${REPO_DIR}/logs/test_candidate_uncertainty}"

LAST_SHARD=$((N_SHARDS - 1))

echo "==========================================="
echo "LoRA re-inference launcher"
echo "==========================================="
echo "LoRA config:   ${LORA_CONFIG}"
echo "LoRA run dir:  ${LORA_RUN_DIR}"
echo "MenGrowth H5:  ${MENGROWTH_H5}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "N shards:      ${N_SHARDS} (array 0-${LAST_SHARD})"
echo "Repo:          ${REPO_DIR}"
echo "Logs:          ${LOGS_DIR}"
echo "Conda env:     ${CONDA_ENV}"
echo

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}" "${OUTPUT_DIR}/shards"
fi

ARRAY_CMD="sbatch --parsable \
    --array=0-${LAST_SHARD} \
    --constraint=dgx \
    --gres=gpu:1 \
    --output=${LOGS_DIR}/uq_reinfer_%A_%a.out \
    --error=${LOGS_DIR}/uq_reinfer_%A_%a.err \
    --export=ALL,LORA_CONFIG=${LORA_CONFIG},LORA_RUN_DIR=${LORA_RUN_DIR},MENGROWTH_H5=${MENGROWTH_H5},OUTPUT_DIR=${OUTPUT_DIR},N_SHARDS=${N_SHARDS},CONDA_ENV=${CONDA_ENV},REPO_DIR=${REPO_DIR} \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/reinfer_worker.sh"

echo "[1/2] Re-inference array job:"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${ARRAY_CMD}"
    ARRAY_JOB_ID="DRYRUN"
else
    ARRAY_JOB_ID=$(eval "${ARRAY_CMD}")
    echo "  -> array job ${ARRAY_JOB_ID}"
fi

MERGE_CMD="sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --constraint=cpu \
    --output=${LOGS_DIR}/uq_reinfer_merge_%j.out \
    --error=${LOGS_DIR}/uq_reinfer_merge_%j.err \
    --export=ALL,OUTPUT_DIR=${OUTPUT_DIR},MENGROWTH_H5=${MENGROWTH_H5},CONDA_ENV=${CONDA_ENV},REPO_DIR=${REPO_DIR} \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/reinfer_merge.sh"

echo "[2/2] Merge + H5 patch (depends on array):"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${MERGE_CMD}"
else
    MERGE_JOB_ID=$(eval "${MERGE_CMD}")
    echo "  -> merge job ${MERGE_JOB_ID}"
fi

echo
echo "Done."
[[ "${DRY_RUN}" == "false" ]] && \
    echo "After merge completes, run Phase B with: bash slurm/launcher.sh configs/picasso.yaml --depend-on ${MERGE_JOB_ID}"
