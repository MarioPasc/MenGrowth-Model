#!/bin/bash
# ---------------------------------------------------------------------------
# Re-inference SLURM worker: loads each of the M trained LoRA members in
# turn (one forward pass per member), computes the corrected entropy/MI
# scalars per scan, and writes a sharded CSV to ${OUTPUT_DIR}/shards/.
# After the array completes, reinfer_merge.sh concatenates the shards
# and patches the H5.
#
# Receives env vars from reinfer_launcher.sh:
#   LORA_CONFIG   path to the run config YAML
#   LORA_RUN_DIR  directory containing adapters/member_*/
#   MENGROWTH_H5  path to the v5 H5
#   OUTPUT_DIR    where shards + per-scan masks land
#   N_SHARDS      total shards (= --array size)
#   REPO_DIR, CONDA_ENV
# ---------------------------------------------------------------------------

#SBATCH --job-name=uq_diag_reinfer
#SBATCH --output=uq_diag_reinfer_%A_%a.out
#SBATCH --error=uq_diag_reinfer_%A_%a.err
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

set -euo pipefail

SHARD_K="${SLURM_ARRAY_TASK_ID:-${SHARD_K:-0}}"

echo "=== Re-inference shard=${SHARD_K}/${N_SHARDS} on $(hostname) at $(date) ==="
echo "LoRA config:   ${LORA_CONFIG}"
echo "LoRA run dir:  ${LORA_RUN_DIR}"
echo "MenGrowth H5:  ${MENGROWTH_H5}"
echo "Output dir:    ${OUTPUT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.reinfer_h5_uncertainty \
    --lora-config "${LORA_CONFIG}" \
    --lora-run-dir "${LORA_RUN_DIR}" \
    --h5 "${MENGROWTH_H5}" \
    --output-dir "${OUTPUT_DIR}" \
    --shard "${SHARD_K}/${N_SHARDS}" \
    --save-ensemble-mask \
    --save-member-masks

echo "=== Done shard=${SHARD_K}/${N_SHARDS} at $(date) ==="
