#!/usr/bin/env bash
# slurm/sdp/launch.sh
# =============================================================================
# SDP MODULE — LAUNCHER (run on login node)
#
# Pre-flight validation + SLURM job submission. Checks:
#   1. Conda environment and key Python packages
#   2. All critical paths from config (H5, checkpoints, splits config, etc.)
#   3. Output directory writability
#
# Creates a timestamped log directory under the configured output_dir and
# routes all SLURM .out/.err files there (not the repo root).
#
# Usage:
#   bash slurm/sdp/launch.sh
#   bash slurm/sdp/launch.sh --config path/to/config.yaml
#   FORCE_EXTRACT=1 bash slurm/sdp/launch.sh
#   SKIP_EVAL=1 bash slurm/sdp/launch.sh
# =============================================================================

set -euo pipefail

# ---- Defaults ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-experiments/sdp/config/picasso/sdp_default.yaml}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIG_PATH="$2"; shift 2 ;;
        --force)   FORCE_EXTRACT=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve config to absolute path
if [[ ! "${CONFIG_PATH}" = /* ]]; then
    CONFIG_PATH="${REPO_DIR}/${CONFIG_PATH}"
fi

echo "=========================================="
echo "SDP MODULE — PRE-FLIGHT VALIDATION"
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
    local kind="${3:-file}"  # file | dir
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
echo "[1/5] Conda environment"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "  OK   python: $(which python) ($(python --version 2>&1))"
echo ""

echo "[2/5] Git pull..."
cd "${REPO_DIR}"
git pull --ff-only || echo "  WARNING: git pull failed (offline or conflicts)"
echo ""

# ---- 2. Python packages ----
echo "[3/5] Python packages"

python -c "
import sys
ok = True

checks = [
    ('torch',              'import torch; v=torch.__version__'),
    ('monai',              'import monai; v=monai.__version__'),
    ('h5py',               'import h5py; v=h5py.__version__'),
    ('pytorch_lightning',  'import pytorch_lightning; v=pytorch_lightning.__version__'),
    ('omegaconf',          'import omegaconf; v=omegaconf.__version__'),
    ('peft',               'import peft; v=peft.__version__'),
]

for name, code in checks:
    try:
        loc = {}
        exec(code, {}, loc)
        print(f'  OK   {name}: {loc[\"v\"]}')
    except ImportError:
        print(f'  FAIL {name}: not installed')
        ok = False

# Project imports
try:
    sys.path.insert(0, '${REPO_DIR}/src')
    from growth.data.bratsmendata import BraTSMENDatasetH5
    from experiments.sdp.train_sdp import main as _train
    print('  OK   project imports (BraTSMENDatasetH5, train_sdp)')
except ImportError as e:
    print(f'  FAIL project imports: {e}')
    ok = False

if not ok:
    sys.exit(1)
" || { ERRORS=$((ERRORS + 1)); }
echo ""

# ---- 3. Critical paths from config ----
echo "[4/5] Critical paths"

check_path "${CONFIG_PATH}" "Config YAML"

# Parse paths from config using Python + OmegaConf (handles interpolation)
eval "$(python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG_PATH}')
p = cfg.paths
print(f'CFG_H5_FILE=\"{p.get(\"h5_file\", \"\")}\"')
print(f'CFG_CHECKPOINT_DIR=\"{p.get(\"checkpoint_dir\", \"\")}\"')
print(f'CFG_LORA_CHECKPOINT=\"{p.get(\"lora_checkpoint\", \"\")}\"')
print(f'CFG_DATA_ROOT=\"{p.get(\"data_root\", \"\")}\"')
print(f'CFG_OUTPUT_DIR=\"{p.get(\"output_dir\", \"\")}\"')
print(f'CFG_FEATURES_DIR=\"{OmegaConf.to_container(cfg, resolve=True)[\"paths\"][\"features_dir\"]}\"')
d = cfg.data
print(f'CFG_SPLITS_CONFIG=\"{d.get(\"splits_config\", \"\")}\"')
")"

[ -n "${CFG_H5_FILE}" ]           && check_path "${CFG_H5_FILE}" "H5 dataset"
[ -n "${CFG_CHECKPOINT_DIR}" ]    && check_path "${CFG_CHECKPOINT_DIR}" "BrainSegFounder checkpoint" dir
[ -n "${CFG_LORA_CHECKPOINT}" ]   && check_path "${CFG_LORA_CHECKPOINT}" "LoRA adapter" dir
[ -n "${CFG_DATA_ROOT}" ]         && check_path "${CFG_DATA_ROOT}" "Data root" dir

# Splits config — resolve relative to repo
if [ -n "${CFG_SPLITS_CONFIG}" ]; then
    if [[ ! "${CFG_SPLITS_CONFIG}" = /* ]]; then
        check_path "${REPO_DIR}/${CFG_SPLITS_CONFIG}" "Splits config (${CFG_SPLITS_CONFIG})"
    else
        check_path "${CFG_SPLITS_CONFIG}" "Splits config"
    fi
fi

echo ""

# ---- 4. Output directory ----
echo "[5/5] Output directory"

if [ -n "${CFG_OUTPUT_DIR}" ]; then
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
SLURM_SCRIPT="${REPO_DIR}/slurm/sdp/run_sdp_pipeline.sh"

JOB_ID=$(sbatch \
    --output="${LOG_DIR}/sdp_pipeline_%j.out" \
    --error="${LOG_DIR}/sdp_pipeline_%j.err" \
    --export="ALL,CONFIG_PATH=${CONFIG_PATH},REPO_DIR=${REPO_DIR},CONDA_ENV_NAME=${CONDA_ENV_NAME},FORCE_EXTRACT=${FORCE_EXTRACT},SKIP_EVAL=${SKIP_EVAL},LOG_DIR=${LOG_DIR}" \
    "${SLURM_SCRIPT}" \
    | grep -oP '\d+')

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "  Job ID:    ${JOB_ID}"
echo "  Log dir:   ${LOG_DIR}"
echo "  Stdout:    ${LOG_DIR}/sdp_pipeline_${JOB_ID}.out"
echo "  Stderr:    ${LOG_DIR}/sdp_pipeline_${JOB_ID}.err"
echo ""
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${LOG_DIR}/sdp_pipeline_${JOB_ID}.out"
