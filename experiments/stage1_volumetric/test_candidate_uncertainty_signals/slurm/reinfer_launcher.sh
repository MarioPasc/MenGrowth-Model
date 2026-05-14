#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the World-A LoRA re-inference for the MenGrowth H5: an array of
# shards (one GPU each) followed by a dependent merge + verification step.
#
# This reproduces the original r32_M20_s42 inference (the "World A" the H5
# volumes/segs come from) and saves per-member voxel masks for every scan —
# the inputs for the segmentation-variance-map thesis figures. The fix lives
# in reinfer_h5_uncertainty.py: it now applies get_h5_val_transforms (z-score
# intensity normalisation) before the ensemble, which the broken version
# skipped — that omission made BSF over-segment and diverge from the H5.
#
# Usage:
#   bash reinfer_launcher.sh <LORA_CONFIG> <LORA_RUN_DIR> <MENGROWTH_H5> <OUTPUT_DIR> [N_SHARDS] [--dry-run]
#
#   LORA_CONFIG   merged r32 run config — use:
#                 experiments/stage1_volumetric/test_candidate_uncertainty_signals/configs/r32_inference.yaml
#   LORA_RUN_DIR  dir containing adapters/member_*/  e.g.
#                 /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/LoRA_finetuned_BSF/r32_M20_s42
#   MENGROWTH_H5  /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/h5_growth_datasets/MenGrowth.h5
#   OUTPUT_DIR    where per_scan/<id>/member_*_mask.nii.gz + shards/ land
#
# Env: PATCH_H5=true  also patches the H5 /uncertainty/ entropy-MI scalars in
#      the merge step (default false — verification only; modifying the
#      experiment H5 stays an opt-in, deliberate step).
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
export PATCH_H5="${PATCH_H5:-false}"
LOGS_DIR="${LOGS_DIR:-${REPO_DIR}/logs/reinfer_worldA}"

LAST_SHARD=$((N_SHARDS - 1))

echo "==========================================="
echo "World-A LoRA re-inference launcher"
echo "==========================================="
echo "LoRA config:   ${LORA_CONFIG}"
echo "LoRA run dir:  ${LORA_RUN_DIR}"
echo "MenGrowth H5:  ${MENGROWTH_H5}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "N shards:      ${N_SHARDS} (array 0-${LAST_SHARD})"
echo "Patch H5:      ${PATCH_H5}"
echo "Repo:          ${REPO_DIR}"
echo "Logs:          ${LOGS_DIR}"
echo "Conda env:     ${CONDA_ENV}"
echo

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}" "${OUTPUT_DIR}/shards"
fi

# --constraint=dgx passed explicitly: the Picasso lua plugin overrides the
# worker's #SBATCH --constraint directive otherwise.
ARRAY_CMD="sbatch \
    --job-name=reinfer_worldA_array \
    --array=0-${LAST_SHARD} \
    --constraint=dgx \
    --gres=gpu:1 \
    --output=${LOGS_DIR}/reinfer_worldA_%A_%a.out \
    --error=${LOGS_DIR}/reinfer_worldA_%A_%a.err \
    --export=ALL,LORA_CONFIG=${LORA_CONFIG},LORA_RUN_DIR=${LORA_RUN_DIR},MENGROWTH_H5=${MENGROWTH_H5},OUTPUT_DIR=${OUTPUT_DIR},N_SHARDS=${N_SHARDS},CONDA_ENV=${CONDA_ENV},REPO_DIR=${REPO_DIR} \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/reinfer_worker.sh"

echo "[1/2] Re-inference array job:"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${ARRAY_CMD}"
    ARRAY_JOB_ID="DRYRUN"
else
    OUT=$(eval "${ARRAY_CMD}" 2>&1)
    echo "  ${OUT}"
    ARRAY_JOB_ID=$(echo "$OUT" | grep -oP 'Submitted batch job \K[0-9]+' | head -1)
    if [[ -z "${ARRAY_JOB_ID}" ]]; then
        echo "ERROR: could not parse array job ID from sbatch output" >&2
        exit 1
    fi
    echo "  -> array job ${ARRAY_JOB_ID}"
fi

MERGE_CMD="sbatch \
    --job-name=reinfer_worldA_merge \
    --constraint=cpu \
    --time=0-00:30:00 \
    --cpus-per-task=2 \
    --mem=8G \
    --dependency=afterany:${ARRAY_JOB_ID} \
    --output=${LOGS_DIR}/reinfer_worldA_merge_%j.out \
    --error=${LOGS_DIR}/reinfer_worldA_merge_%j.err \
    --export=ALL,OUTPUT_DIR=${OUTPUT_DIR},MENGROWTH_H5=${MENGROWTH_H5},CONDA_ENV=${CONDA_ENV},REPO_DIR=${REPO_DIR},PATCH_H5=${PATCH_H5} \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/reinfer_merge.sh"

echo "[2/2] Merge + verification (depends on array):"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${MERGE_CMD}"
    MERGE_JOB_ID="DRYRUN"
else
    OUT=$(eval "${MERGE_CMD}" 2>&1)
    echo "  ${OUT}"
    MERGE_JOB_ID=$(echo "$OUT" | grep -oP 'Submitted batch job \K[0-9]+' | head -1)
fi

echo
echo "Done. Outputs once the merge job finishes:"
echo "  ${OUTPUT_DIR}/per_scan/<scan_id>/member_*_mask.nii.gz   (variance-map inputs)"
echo "  ${OUTPUT_DIR}/per_scan/<scan_id>/ensemble_mask.nii.gz"
echo "  ${OUTPUT_DIR}/recomputed_uncertainty.csv"
echo "Check the merge log for 'VERIFY PASSED' before trusting the masks."
