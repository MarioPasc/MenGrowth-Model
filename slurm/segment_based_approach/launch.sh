#!/usr/bin/env bash
# slurm/segment_based_approach/launch.sh
# =============================================================================
# ABLATION A0: SEGMENT-BASED BASELINE — LAUNCHER (run on login node)
#
# Pre-flight validation + SLURM job submission. Checks:
#   1. Conda environment and key Python packages
#   2. Critical paths (H5 data, checkpoint)
#   3. Output directory writability
#
# Usage:
#   bash slurm/segment_based_approach/launch.sh
#   bash slurm/segment_based_approach/launch.sh --config path/to/config.yaml
#   bash slurm/segment_based_approach/launch.sh --force
# =============================================================================

set -euo pipefail

# ---- Defaults ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-experiments/segment_based_approach/config.yaml}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIG_PATH="$2"; shift 2 ;;
        --force)   FORCE_RECOMPUTE=1; shift ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve config to absolute path
if [[ ! "${CONFIG_PATH}" = /* ]]; then
    CONFIG_PATH="${REPO_DIR}/${CONFIG_PATH}"
fi

echo "=========================================="
echo "ABLATION A0: SEGMENT-BASED BASELINE"
echo "=========================================="
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Repo:     ${REPO_DIR}"
echo "Config:   ${CONFIG_PATH}"
echo ""

# ---- Helper ----
ERRORS=0
check_path() {
    local path="$1"
    local desc="$2"
    local kind="${3:-file}"
    if [ "${kind}" = "dir" ]; then
        if [ -d "${path}" ]; then
            echo "  OK   ${desc}"
        else
            echo "  FAIL ${desc} -> ${path}"
            ERRORS=$((ERRORS + 1))
        fi
    else
        if [ -f "${path}" ]; then
            echo "  OK   ${desc}"
        else
            echo "  FAIL ${desc} -> ${path}"
            ERRORS=$((ERRORS + 1))
        fi
    fi
}

# ---- 1. Conda environment ----
echo "[1/4] Conda environment"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "  OK   python: $(which python) ($(python --version 2>&1))"
echo ""

# ---- 2. Git pull ----
echo "[2/4] Git pull..."
cd "${REPO_DIR}"
git pull --ff-only || echo "  WARNING: git pull failed (offline or conflicts)"
echo ""

# ---- 3. Critical paths from config ----
echo "[3/4] Critical paths"

check_path "${CONFIG_PATH}" "Config YAML"

# Parse paths from config
eval "$(python -c "
import yaml
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
p = cfg['paths']
print(f'CFG_CHECKPOINT=\"{p.get(\"checkpoint\", \"\")}\"')
print(f'CFG_H5_FILE=\"{p.get(\"mengrowth_h5\", \"\")}\"')
print(f'CFG_OUTPUT_DIR=\"{p.get(\"output_dir\", \"\")}\"')
")"

[ -n "${CFG_CHECKPOINT}" ] && check_path "${CFG_CHECKPOINT}" "BSF checkpoint"
[ -n "${CFG_H5_FILE}" ]    && check_path "${CFG_H5_FILE}" "MenGrowth H5"

echo ""

# ---- 4. Output directory ----
echo "[4/4] Output directory"

if [ -n "${CFG_OUTPUT_DIR}" ]; then
    # Resolve relative to repo
    if [[ ! "${CFG_OUTPUT_DIR}" = /* ]]; then
        CFG_OUTPUT_DIR="${REPO_DIR}/${CFG_OUTPUT_DIR}"
    fi
    mkdir -p "${CFG_OUTPUT_DIR}" 2>/dev/null && echo "  OK   ${CFG_OUTPUT_DIR}" \
        || { echo "  FAIL Cannot create ${CFG_OUTPUT_DIR}"; ERRORS=$((ERRORS + 1)); }
else
    echo "  FAIL output_dir not set in config"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ---- Abort on errors ----
if [ "${ERRORS}" -gt 0 ]; then
    echo "=========================================="
    echo "VALIDATION FAILED (${ERRORS} error(s))"
    echo "=========================================="
    echo "Fix the issues above before submitting."
    exit 1
fi

echo "=========================================="
echo "VALIDATION PASSED"
echo "=========================================="
echo ""

# ---- Create log directory ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${CFG_OUTPUT_DIR}/logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "Log directory: ${LOG_DIR}"
echo ""

# ---- Copy config snapshot ----
cp "${CONFIG_PATH}" "${LOG_DIR}/config_snapshot.yaml"

# ---- Submit SLURM job ----
SLURM_SCRIPT="${REPO_DIR}/slurm/segment_based_approach/run_baseline.sh"

JOB_ID=$(sbatch \
    --output="${LOG_DIR}/baseline_%j.out" \
    --error="${LOG_DIR}/baseline_%j.err" \
    --export="ALL,CONFIG_PATH=${CONFIG_PATH},REPO_DIR=${REPO_DIR},CONDA_ENV_NAME=${CONDA_ENV_NAME},FORCE_RECOMPUTE=${FORCE_RECOMPUTE},LOG_DIR=${LOG_DIR}" \
    "${SLURM_SCRIPT}" \
    | grep -oP '\d+')

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "  Job ID:    ${JOB_ID}"
echo "  Log dir:   ${LOG_DIR}"
echo "  Stdout:    ${LOG_DIR}/baseline_${JOB_ID}.out"
echo "  Stderr:    ${LOG_DIR}/baseline_${JOB_ID}.err"
echo ""
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${LOG_DIR}/baseline_${JOB_ID}.out"
