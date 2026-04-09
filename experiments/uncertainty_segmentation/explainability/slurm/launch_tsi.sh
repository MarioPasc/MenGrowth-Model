#!/usr/bin/env bash
# =============================================================================
# TSI EXPLAINABILITY — PICASSO LAUNCHER
#
# Submits a single-GPU job for TSI analysis at full 192³ resolution.
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model
#   bash experiments/uncertainty_segmentation/explainability/slurm/launch_tsi.sh
#   bash experiments/uncertainty_segmentation/explainability/slurm/launch_tsi.sh --rank 4
#   bash experiments/uncertainty_segmentation/explainability/slurm/launch_tsi.sh --rank 4,8,16
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="$(cd "${MODULE_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"

echo "=========================================="
echo "TSI EXPLAINABILITY — PICASSO LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export REPO_SRC="${REPO_SRC:-${REPO_ROOT}}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"

# Config files
BASE_CONFIG="${EXPERIMENT_DIR}/config.yaml"
PICASSO_OVERRIDE="${EXPERIMENT_DIR}/config/picasso/config_picasso.yaml"
TSI_BASE_CONFIG="${MODULE_DIR}/config.yaml"
TSI_PICASSO_OVERRIDE="${MODULE_DIR}/config/picasso/config_tsi_picasso.yaml"

# Parse arguments
RANKS="4,8,16"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank)
            RANKS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Configuration:"
echo "  Repo:            ${REPO_SRC}"
echo "  Base config:     ${BASE_CONFIG}"
echo "  Picasso config:  ${PICASSO_OVERRIDE}"
echo "  TSI config:      ${TSI_BASE_CONFIG}"
echo "  TSI Picasso:     ${TSI_PICASSO_OVERRIDE}"
echo "  Ranks:           ${RANKS}"
echo "  Conda env:       ${CONDA_ENV_NAME}"
echo ""

# Activate conda for pre-flight
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
fi

# ========================================================================
# RESOLVE MERGED CONFIGS
# ========================================================================
echo "Resolving merged configuration..."

MERGED_OUTPUT=""
MERGE_RC=0
MERGED_OUTPUT=$(python3 -c "
from omegaconf import OmegaConf
import os, sys, pathlib

# Parent config: base + picasso override
base_path = '${BASE_CONFIG}'
picasso_path = '${PICASSO_OVERRIDE}'
tsi_base_path = '${TSI_BASE_CONFIG}'
tsi_picasso_path = '${TSI_PICASSO_OVERRIDE}'

if not os.path.exists(base_path):
    print(f'FAIL Base config not found: {base_path}')
    sys.exit(1)

# Merge parent config
parent_cfg = OmegaConf.load(base_path)
if os.path.exists(picasso_path):
    parent_cfg = OmegaConf.merge(parent_cfg, OmegaConf.load(picasso_path))
    print('[config] Parent: base + picasso override')

# Merge TSI config
tsi_cfg = OmegaConf.load(tsi_base_path)
if os.path.exists(tsi_picasso_path):
    tsi_cfg = OmegaConf.merge(tsi_cfg, OmegaConf.load(tsi_picasso_path))
    print('[config] TSI: base + picasso override')

output_dir = tsi_cfg.paths.output_dir
roi = list(tsi_cfg.analysis.roi_size)
n_scans = tsi_cfg.analysis.n_scans

# Save resolved configs
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
OmegaConf.save(parent_cfg, f'{output_dir}/parent_config_snapshot.yaml', resolve=True)
OmegaConf.save(tsi_cfg, f'{output_dir}/tsi_config_snapshot.yaml', resolve=True)

print(f'[config] ROI: {roi}, N_scans: {n_scans}')
print(f'RESULT {output_dir}')
" 2>&1) || MERGE_RC=$?

echo "${MERGED_OUTPUT}" | grep -v "^RESULT" || true

if [ "${MERGE_RC}" -ne 0 ]; then
    echo "[FAIL] Config resolution failed (exit code ${MERGE_RC}):"
    echo "${MERGED_OUTPUT}"
    exit 1
fi

RESULT_LINE=$(echo "${MERGED_OUTPUT}" | grep "^RESULT" || true)
if [ -z "${RESULT_LINE}" ]; then
    echo "[FAIL] Config resolution produced no RESULT. Output was:"
    echo "${MERGED_OUTPUT}"
    exit 1
fi

read -r _ OUTPUT_DIR <<< "${RESULT_LINE}"

export OUTPUT_DIR
export CONFIG_PATH="${OUTPUT_DIR}/parent_config_snapshot.yaml"
export TSI_CONFIG_PATH="${OUTPUT_DIR}/tsi_config_snapshot.yaml"
export RANKS

SLURM_LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${SLURM_LOG_DIR}"

echo ""
echo "Resolved:"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Config:  ${CONFIG_PATH}"
echo "  TSI cfg: ${TSI_CONFIG_PATH}"
echo "  Ranks:   ${RANKS}"
echo "  Logs:    ${SLURM_LOG_DIR}"
echo ""

# ========================================================================
# SUBMIT
# ========================================================================
echo "Submitting TSI analysis job..."

JOB_OUTPUT=$(sbatch \
    --job-name="tsi_explain" \
    --output="${SLURM_LOG_DIR}/tsi_%j.out" \
    --error="${SLURM_LOG_DIR}/tsi_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",TSI_CONFIG_PATH="${TSI_CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",OUTPUT_DIR="${OUTPUT_DIR}",RANKS="${RANKS}" \
    "${SCRIPT_DIR}/run_tsi_worker.sh" 2>&1)

JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'job\s+\K[0-9]+' | head -1)
if [ -z "$JOB_ID" ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '[0-9]+' | head -1)
fi

echo "  sbatch output: ${JOB_OUTPUT}"
echo "  Job ID: ${JOB_ID}"

if [ -z "${JOB_ID}" ]; then
    echo "[FAIL] Could not extract job ID"
    exit 1
fi

echo ""
echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo ""
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${SLURM_LOG_DIR}/tsi_${JOB_ID}.out"
echo ""
echo "Cancel:"
echo "  scancel ${JOB_ID}"
echo ""
echo "Estimated time: ~20-30 min (150 scans × 4 conditions × ~3s/scan)"
