#!/usr/bin/env bash
# =============================================================================
# DOMAIN DIVERGENCE ANALYSIS — PICASSO LAUNCHER
#
# Submits a single-GPU job for the full domain divergence pipeline:
#   extract → metrics → CKA cross-stage → CKA drift → correlate → figures
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model
#   bash experiments/uncertainty_segmentation/explainability/slurm/launch_domain_divergence.sh
#   bash experiments/uncertainty_segmentation/explainability/slurm/launch_domain_divergence.sh --encoder-only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="$(cd "${MODULE_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"

echo "=========================================="
echo "DOMAIN DIVERGENCE — PICASSO LAUNCHER"
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
DD_BASE_CONFIG="${MODULE_DIR}/config/domain_divergence.yaml"
DD_PICASSO_OVERRIDE="${MODULE_DIR}/config/picasso/config_domain_divergence_picasso.yaml"

# Parse arguments
ENCODER_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder-only)
            ENCODER_ONLY=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Resolve encoder-only overrides
ENCODER_ONLY_PATH=""
ENCODER_ONLY_PICASSO_PATH=""
if [ "${ENCODER_ONLY}" -eq 1 ]; then
    ENCODER_ONLY_PATH="${EXPERIMENT_DIR}/config/encoder_only.yaml"
    ENCODER_ONLY_PICASSO_PATH="${EXPERIMENT_DIR}/config/picasso/config_encoder_only_picasso.yaml"
    if [ ! -f "${ENCODER_ONLY_PATH}" ]; then
        echo "[FAIL] Encoder-only config not found: ${ENCODER_ONLY_PATH}"
        exit 1
    fi
fi

echo "Configuration:"
echo "  Repo:                ${REPO_SRC}"
echo "  Parent base:         ${BASE_CONFIG}"
echo "  Parent picasso:      ${PICASSO_OVERRIDE}"
echo "  DD base:             ${DD_BASE_CONFIG}"
echo "  DD picasso:          ${DD_PICASSO_OVERRIDE}"
echo "  Conda env:           ${CONDA_ENV_NAME}"
if [ "${ENCODER_ONLY}" -eq 1 ]; then
    echo "  Mode:                ENCODER-ONLY (frozen decoder)"
    echo "  encoder_only:        ${ENCODER_ONLY_PATH}"
    echo "  encoder_only_p:      ${ENCODER_ONLY_PICASSO_PATH}"
fi
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

base_path = '${BASE_CONFIG}'
picasso_path = '${PICASSO_OVERRIDE}'
encoder_only_path = '${ENCODER_ONLY_PATH}'
encoder_only_picasso_path = '${ENCODER_ONLY_PICASSO_PATH}'
dd_base_path = '${DD_BASE_CONFIG}'
dd_picasso_path = '${DD_PICASSO_OVERRIDE}'

if not os.path.exists(base_path):
    print(f'FAIL Base config not found: {base_path}')
    sys.exit(1)

# Merge parent config
parent_cfg = OmegaConf.load(base_path)
if os.path.exists(picasso_path):
    parent_cfg = OmegaConf.merge(parent_cfg, OmegaConf.load(picasso_path))
    print('[config] Parent: base + picasso override')

# Apply encoder-only overrides
if encoder_only_path and os.path.exists(encoder_only_path):
    parent_cfg = OmegaConf.merge(parent_cfg, OmegaConf.load(encoder_only_path))
    print(f'[config] Applied encoder-only: target_stages={list(parent_cfg.lora.target_stages)}')
if encoder_only_picasso_path and os.path.exists(encoder_only_picasso_path):
    parent_cfg = OmegaConf.merge(parent_cfg, OmegaConf.load(encoder_only_picasso_path))
    print('[config] Applied encoder-only picasso path override')

# Merge domain divergence config
dd_cfg = OmegaConf.load(dd_base_path)
if os.path.exists(dd_picasso_path):
    dd_cfg = OmegaConf.merge(dd_cfg, OmegaConf.load(dd_picasso_path))
    print('[config] DD: base + picasso override')

output_dir = dd_cfg.paths.output_dir
n_scans = dd_cfg.domain_divergence.n_scans_per_domain
roi = list(dd_cfg.domain_divergence.roi_size)

# Save resolved configs
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
OmegaConf.save(parent_cfg, f'{output_dir}/parent_config_snapshot.yaml', resolve=True)
OmegaConf.save(dd_cfg, f'{output_dir}/analysis_config_snapshot.yaml', resolve=True)

print(f'[config] ROI: {roi}, N_scans_per_domain: {n_scans}')
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
export ANALYSIS_CONFIG_PATH="${OUTPUT_DIR}/analysis_config_snapshot.yaml"

SLURM_LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${SLURM_LOG_DIR}"

echo ""
echo "Resolved:"
echo "  Output:        ${OUTPUT_DIR}"
echo "  Parent cfg:    ${CONFIG_PATH}"
echo "  Analysis cfg:  ${ANALYSIS_CONFIG_PATH}"
echo "  Logs:          ${SLURM_LOG_DIR}"
echo ""

# ========================================================================
# SUBMIT
# ========================================================================
echo "Submitting domain divergence job..."

JOB_OUTPUT=$(sbatch \
    --job-name="dd_explain" \
    --output="${SLURM_LOG_DIR}/dd_%j.out" \
    --error="${SLURM_LOG_DIR}/dd_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",ANALYSIS_CONFIG_PATH="${ANALYSIS_CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",OUTPUT_DIR="${OUTPUT_DIR}" \
    "${SCRIPT_DIR}/run_domain_divergence_worker.sh" 2>&1)

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
echo "  tail -f ${SLURM_LOG_DIR}/dd_${JOB_ID}.out"
echo ""
echo "Cancel:"
echo "  scancel ${JOB_ID}"
echo ""
echo "Estimated time: ~60 min (extract 2×150 scans + metrics + CKA drift)"
