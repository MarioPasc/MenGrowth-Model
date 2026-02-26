#!/usr/bin/env bash
#SBATCH -J sdp_pipeline
#SBATCH --time=0-10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# SDP MODULE — COMPUTE JOB (submitted by launch.sh)
#
# 1. Feature encoding  — skipped if features already exist (override: FORCE_EXTRACT=1)
# 2. SDP training      — trains the projection network
# 3. Evaluation        — probes, DCI, variance, Jacobian, figures, tables
#
# This script is meant to be submitted via launch.sh, which handles
# pre-flight validation and log directory creation.
#
# Direct usage (for debugging):
#   CONFIG_PATH=... LOG_DIR=... bash slurm/sdp/run_sdp_pipeline.sh
# =============================================================================

set -euo pipefail

# ---- Configuration (set by launch.sh via --export) ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-experiments/sdp/config/picasso/sdp_default.yaml}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
LOG_DIR="${LOG_DIR:-$(pwd)}"

START_TIME=$(date +%s)

log_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

log_elapsed() {
    local start=$1
    local end=$(date +%s)
    local elapsed=$((end - start))
    echo "  Elapsed: $((elapsed / 60))m $((elapsed % 60))s"
}

log_header "SDP MODULE — FULL PIPELINE"
echo "Started:       $(date)"
echo "Hostname:      $(hostname)"
echo "SLURM Job:     ${SLURM_JOB_ID:-local}"
echo "Config:        ${CONFIG_PATH}"
echo "Log dir:       ${LOG_DIR}"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Force extract: ${FORCE_EXTRACT}"
echo "Skip eval:     ${SKIP_EVAL}"

# ---- Environment Setup ----
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail "$m" 2>&1 | grep -qi "$m"; then
        module load "$m" && module_loaded=1 && break
    fi
done

if [ "$module_loaded" -eq 0 ]; then
    echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

log_header "ENVIRONMENT"
echo "[python] $(which python)"
python -c "
import torch
print(f'[torch]  {torch.__version__}  CUDA={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[gpu]    {torch.cuda.get_device_name(0)}')
    print(f'[vram]   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import h5py
print(f'[h5py]   {h5py.__version__}')
"

# =============================================================================
# STEP 1: FEATURE EXTRACTION (conditional)
# =============================================================================

FEATURES_EXIST=$(python -c "
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load('${CONFIG_PATH}')
features_dir = Path(OmegaConf.to_container(cfg, resolve=True)['paths']['features_dir'])
expected_splits = list(cfg.data.train_splits) + [cfg.data.val_split]
test_split = cfg.data.get('test_split', 'test')
if test_split:
    expected_splits.append(test_split)

missing = [s for s in expected_splits if not (features_dir / f'{s}.h5').exists()]
if not missing:
    print('yes')
else:
    print('no')
    import sys
    print(f'  Missing: {missing}', file=sys.stderr)
")

if [ "${FORCE_EXTRACT}" = "1" ] || [ "${FEATURES_EXIST}" != "yes" ]; then
    log_header "STEP 1/2: FEATURE EXTRACTION"

    if [ "${FORCE_EXTRACT}" = "1" ]; then
        echo "[info] FORCE_EXTRACT=1 — re-extracting all features"
    else
        echo "[info] Features not found — extracting now"
    fi

    STEP_START=$(date +%s)

    python -m experiments.sdp.extract_all_features \
        --config "${CONFIG_PATH}" \
        --device cuda

    # Verify output
    python -c "
from pathlib import Path
from omegaconf import OmegaConf
import h5py

cfg = OmegaConf.load('${CONFIG_PATH}')
features_dir = Path(OmegaConf.to_container(cfg, resolve=True)['paths']['features_dir'])

h5_files = sorted(features_dir.glob('*.h5'))
if not h5_files:
    raise RuntimeError(f'No feature H5 files in {features_dir}')

print(f'  Output: {features_dir}')
for f in h5_files:
    with h5py.File(f, 'r') as h:
        n = len(h['subject_ids'])
        feat_shape = h['features/encoder10'].shape if 'features/encoder10' in h else '?'
    print(f'    {f.name}: {n} subjects, encoder10={feat_shape}, {f.stat().st_size / (1024**2):.1f} MB')
"

    log_elapsed "$STEP_START"
else
    log_header "STEP 1/2: FEATURE EXTRACTION — SKIPPED (cached)"

    python -c "
from pathlib import Path
from omegaconf import OmegaConf
import h5py

cfg = OmegaConf.load('${CONFIG_PATH}')
features_dir = Path(OmegaConf.to_container(cfg, resolve=True)['paths']['features_dir'])
for f in sorted(features_dir.glob('*.h5')):
    with h5py.File(f, 'r') as h:
        n = len(h['subject_ids'])
    print(f'  [cached] {f.name}: {n} subjects')
"
fi

# =============================================================================
# STEP 2: SDP TRAINING + EVALUATION
# =============================================================================
log_header "STEP 2/2: SDP TRAINING"

STEP_START=$(date +%s)

EVAL_FLAG=""
if [ "${SKIP_EVAL}" = "1" ]; then
    EVAL_FLAG="--skip-eval"
    echo "[info] Post-training eval skipped (SKIP_EVAL=1)"
fi

# Run training; capture run directory from the last "All outputs in:" log line
python -m experiments.sdp.train_sdp \
    --config "${CONFIG_PATH}" \
    ${EVAL_FLAG} \
    2>&1 | tee "${LOG_DIR}/train_sdp.log"

# Extract run directory
RUN_DIR=$(grep -oP '(?<=All outputs in: ).*' "${LOG_DIR}/train_sdp.log" | tail -1 || true)

# Fallback: find most recent run with a checkpoint
if [ -z "${RUN_DIR}" ]; then
    RUN_DIR=$(python -c "
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load('${CONFIG_PATH}')
output_dir = Path(cfg.paths.get('output_dir', 'outputs/sdp'))
runs = sorted(output_dir.glob('*/checkpoints/phase2_sdp.pt'), key=lambda p: p.stat().st_mtime)
print(runs[-1].parent.parent if runs else '')
" || true)
fi

log_elapsed "$STEP_START"

if [ -z "${RUN_DIR}" ]; then
    echo "[error] Could not determine run directory. Check ${LOG_DIR}/train_sdp.log"
    exit 1
fi

echo "  Run directory: ${RUN_DIR}"

# Symlink latest run from log dir for easy access
ln -sfn "${RUN_DIR}" "${LOG_DIR}/run" 2>/dev/null || true

# =============================================================================
# SUMMARY
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

log_header "PIPELINE COMPLETE"
echo "Run directory: ${RUN_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Total elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Finished:      $(date)"

# Print quality report
QUALITY_REPORT="${RUN_DIR}/evaluation/quality_report.json"
if [ -f "${QUALITY_REPORT}" ]; then
    echo ""
    echo "--- Quality Report ---"
    python -c "
import json
with open('${QUALITY_REPORT}') as f:
    report = json.load(f)
for key, value in report.items():
    print(f'  {key}: {value:.4f}')
"
fi

echo ""
echo "Next steps:"
echo "  1. Review: cat ${RUN_DIR}/evaluation/quality_report.json"
echo "  2. If BLOCKING thresholds pass -> proceed to Phase 3"
