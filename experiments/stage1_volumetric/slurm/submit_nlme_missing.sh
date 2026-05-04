#!/bin/bash
# Submit the two missing NLME models + analysis job on Picasso.
# Usage: bash experiments/stage1_volumetric/slurm/submit_nlme_missing.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

# Configs
MODEL_CONFIG="experiments/stage1_volumetric/configs/picasso_nlme_missing.yaml"
FULL_CONFIG="experiments/stage1_volumetric/configs/picasso.yaml"
WORKER="${SCRIPT_DIR}/worker.sh"
ANALYSIS_WORKER="${SCRIPT_DIR}/analysis_worker.sh"

# SLURM settings
PARTITION="gputhin"
CONSTRAINT="cpu"
TIME="0-01:00:00"
CPUS=4
MEM="16G"
CONDA_ENV="growth"
SLURM_REPO="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model"
LOGS="/mnt/home/users/tic_163_uma/mpascual/execs/growth/logs/stage1_uq"

echo "========================================="
echo "Submit missing NLME models + analysis"
echo "========================================="

mkdir -p "${LOGS}"

MODELS=("NLME_Logistic" "NLME_Gompertz")
JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    OUTPUT=$(sbatch \
        --job-name="growth_uq_${MODEL}" \
        --partition="${PARTITION}" \
        --constraint="${CONSTRAINT}" \
        --time="${TIME}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --output="${LOGS}/growth_uq_${MODEL}_%j.out" \
        --error="${LOGS}/growth_uq_${MODEL}_%j.err" \
        --export=ALL,MODEL_NAME="${MODEL}",CONFIG_PATH="${MODEL_CONFIG}",CONDA_ENV="${CONDA_ENV}",REPO_DIR="${SLURM_REPO}" \
        "${WORKER}" 2>&1)

    echo "${OUTPUT}"
    JOB_ID=$(echo "${OUTPUT}" | grep -oP 'job\s+\K[0-9]+' | head -1)
    if [[ -z "${JOB_ID}" ]]; then
        JOB_ID=$(echo "${OUTPUT}" | grep -oP '[0-9]+' | head -1)
    fi

    if [[ -n "${JOB_ID}" ]]; then
        echo "  -> ${MODEL} = job ${JOB_ID}"
        JOB_IDS+=("${JOB_ID}")
    else
        echo "  [WARN] Could not extract job ID for ${MODEL}"
    fi
done

# Analysis job uses the FULL config (all 11 models) so comparisons work
if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    ALL_DEP=$(IFS=:; echo "${JOB_IDS[*]}")
    echo ""
    echo "--- Analysis job (depends on: ${ALL_DEP}) ---"

    ANALYSIS_OUTPUT=$(sbatch \
        --job-name="growth_uq_analysis" \
        --partition="${PARTITION}" \
        --constraint="${CONSTRAINT}" \
        --time="0-00:30:00" \
        --cpus-per-task=2 \
        --mem="8G" \
        --output="${LOGS}/growth_uq_analysis_%j.out" \
        --error="${LOGS}/growth_uq_analysis_%j.err" \
        --dependency="afterany:${ALL_DEP}" \
        --export=ALL,CONFIG_PATH="${FULL_CONFIG}",CONDA_ENV="${CONDA_ENV}",REPO_DIR="${SLURM_REPO}" \
        "${ANALYSIS_WORKER}" 2>&1)

    echo "${ANALYSIS_OUTPUT}"
    ANALYSIS_ID=$(echo "${ANALYSIS_OUTPUT}" | grep -oP 'job\s+\K[0-9]+' | head -1)
    if [[ -n "${ANALYSIS_ID}" ]]; then
        echo "  -> analysis = job ${ANALYSIS_ID}"
    fi
fi

echo ""
echo "Done. 2 model jobs + 1 analysis job submitted."
