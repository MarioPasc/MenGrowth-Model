#!/usr/bin/env bash
#SBATCH -J sdp_pipeline
#SBATCH --time=0-10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=sdp_pipeline_%j.out
#SBATCH --error=sdp_pipeline_%j.err

# =============================================================================
# SDP MODULE — FULL PIPELINE (SLURM job, no internet)
#
# 1. Feature encoding: Extracts encoder10 features (768-dim) from all BraTS-MEN
#    subjects using the LoRA-adapted encoder. SKIPPED if features already exist.
# 2. SDP training: Trains the Supervised Disentangled Projection network.
# 3. Evaluation: Runs full post-training evaluation (probes, DCI, variance,
#    Jacobian XAI, cross-probing, tables, and publication figures).
#
# Usage:
#   sbatch slurm/sdp/encode_volumes.sh
#   # Custom config:
#   CONFIG_PATH=path/to/config.yaml sbatch slurm/sdp/encode_volumes.sh
#   # Force re-extraction even if features exist:
#   FORCE_EXTRACT=1 sbatch slurm/sdp/encode_volumes.sh
#   # Skip evaluation for fast iteration:
#   SKIP_EVAL=1 sbatch slurm/sdp/encode_volumes.sh
# =============================================================================

set -euo pipefail

# ---- Configuration ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-experiments/sdp/config/picasso/sdp_default.yaml}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

START_TIME=$(date +%s)
echo "=========================================="
echo "SDP MODULE — FULL PIPELINE"
echo "=========================================="
echo "Started:       $(date)"
echo "Hostname:      $(hostname)"
echo "SLURM Job:     ${SLURM_JOB_ID:-local}"
echo "Config:        ${CONFIG_PATH}"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Force extract: ${FORCE_EXTRACT}"
echo "Skip eval:     ${SKIP_EVAL}"
echo ""

# ---- Environment Setup ----
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
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

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python)"
python -c "
import torch
print(f'[torch]  {torch.__version__}  CUDA={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[gpu]    {torch.cuda.get_device_name(0)}')
    print(f'[vram]   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import h5py
print(f'[h5py]   {h5py.__version__}')
"
echo ""

# ---- Pre-flight Checks ----
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

python -c "
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load('${CONFIG_PATH}')

paths_to_check = {
    'H5 file': cfg.get('paths', {}).get('h5_file'),
    'LoRA adapter': cfg.get('paths', {}).get('lora_checkpoint'),
}

ok = True
for desc, path in paths_to_check.items():
    if path and Path(str(path)).exists():
        print(f'  OK: {desc}')
    elif path:
        print(f'  MISSING: {desc} -> {path}')
        ok = False
    else:
        print(f'  SKIP: {desc} (not configured)')

if not ok:
    raise RuntimeError('Pre-flight check failed')
print('  All pre-flight checks passed!')
"
echo ""

# =============================================================================
# STEP 1: FEATURE EXTRACTION (conditional)
# =============================================================================

# Check if features already exist
FEATURES_EXIST=$(python -c "
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load('${CONFIG_PATH}')
features_dir = Path(cfg.paths.features_dir)
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
    print(f'  Missing splits: {missing}', file=sys.stderr)
")

if [ "${FORCE_EXTRACT}" = "1" ] || [ "${FEATURES_EXIST}" != "yes" ]; then
    echo "=========================================="
    echo "STEP 1: FEATURE EXTRACTION"
    echo "=========================================="

    if [ "${FORCE_EXTRACT}" = "1" ]; then
        echo "[info] FORCE_EXTRACT=1 — re-extracting all features"
    else
        echo "[info] Features not found — extracting now"
    fi
    echo ""

    EXTRACT_START=$(date +%s)

    python -m experiments.sdp.extract_all_features \
        --config "${CONFIG_PATH}" \
        --device cuda

    EXTRACT_END=$(date +%s)
    EXTRACT_ELAPSED=$((EXTRACT_END - EXTRACT_START))

    # Verify extraction output
    echo ""
    echo "--- Feature extraction verification ---"
    python -c "
from pathlib import Path
from omegaconf import OmegaConf
import h5py

cfg = OmegaConf.load('${CONFIG_PATH}')
features_dir = Path(cfg.paths.features_dir)

if not features_dir.exists():
    raise RuntimeError(f'Features dir not found: {features_dir}')

h5_files = sorted(features_dir.glob('*.h5'))
print(f'  Features directory: {features_dir}')
print(f'  H5 files found: {len(h5_files)}')

for f in h5_files:
    with h5py.File(f, 'r') as h:
        n = len(h['subject_ids'])
        feat_shape = h['features/encoder10'].shape if 'features/encoder10' in h else '?'
    size_mb = f.stat().st_size / (1024**2)
    print(f'    {f.name}: {n} subjects, encoder10={feat_shape}, {size_mb:.1f} MB')

if not h5_files:
    raise RuntimeError('No output H5 files found!')
"

    echo "  Feature extraction: $((EXTRACT_ELAPSED / 60))m $((EXTRACT_ELAPSED % 60))s"
    echo ""
else
    echo "=========================================="
    echo "STEP 1: FEATURE EXTRACTION — SKIPPED"
    echo "=========================================="
    echo "[info] All expected feature splits already exist."

    python -c "
from pathlib import Path
from omegaconf import OmegaConf
import h5py

cfg = OmegaConf.load('${CONFIG_PATH}')
features_dir = Path(cfg.paths.features_dir)
h5_files = sorted(features_dir.glob('*.h5'))
for f in h5_files:
    with h5py.File(f, 'r') as h:
        n = len(h['subject_ids'])
        feat_shape = h['features/encoder10'].shape if 'features/encoder10' in h else '?'
    size_mb = f.stat().st_size / (1024**2)
    print(f'  [cached] {f.name}: {n} subjects, encoder10={feat_shape}, {size_mb:.1f} MB')
"
    echo ""
fi

# =============================================================================
# STEP 2: SDP TRAINING
# =============================================================================
echo "=========================================="
echo "STEP 2: SDP TRAINING"
echo "=========================================="

TRAIN_START=$(date +%s)

EVAL_FLAG=""
if [ "${SKIP_EVAL}" = "1" ]; then
    EVAL_FLAG="--skip-eval"
    echo "[info] Post-training eval will be skipped (SKIP_EVAL=1)"
fi

RUN_DIR=$(python -m experiments.sdp.train_sdp \
    --config "${CONFIG_PATH}" \
    ${EVAL_FLAG} \
    2>&1 | tee /dev/stderr | grep -oP '(?<=All outputs in: ).*' | tail -1)

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END - TRAIN_START))

echo ""
echo "  SDP training: $((TRAIN_ELAPSED / 60))m $((TRAIN_ELAPSED % 60))s"

# If train_sdp didn't print the run dir, find the most recent one
if [ -z "${RUN_DIR:-}" ]; then
    RUN_DIR=$(python -c "
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load('${CONFIG_PATH}')
output_dir = Path(cfg.paths.get('output_dir', 'outputs/sdp'))
runs = sorted(output_dir.glob('*/checkpoints/phase2_sdp.pt'), key=lambda p: p.stat().st_mtime)
if runs:
    print(runs[-1].parent.parent)
else:
    print('')
")
fi

if [ -z "${RUN_DIR}" ]; then
    echo "[error] Could not determine run directory. SDP training may have failed."
    exit 1
fi

echo "  Run directory: ${RUN_DIR}"
echo ""

# =============================================================================
# STEP 3: EVALUATION (unless --skip-eval was already handled by train_sdp)
# =============================================================================
# train_sdp already runs evaluate_sdp + visualize_sdp unless --skip-eval is set.
# This step runs standalone evaluation only if SKIP_EVAL was set during training
# but the user still wants evaluation (controlled by a separate flag).

# =============================================================================
# SUMMARY
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=========================================="
echo "SDP PIPELINE COMPLETE"
echo "=========================================="
echo "Run directory: ${RUN_DIR}"
echo "Total elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Finished:      $(date)"
echo ""

# Print quality report if available
QUALITY_REPORT="${RUN_DIR}/evaluation/quality_report.json"
if [ -f "${QUALITY_REPORT}" ]; then
    echo "--- Quality Report ---"
    python -c "
import json
with open('${QUALITY_REPORT}') as f:
    report = json.load(f)
for key, value in report.items():
    print(f'  {key}: {value:.4f}')
"
    echo ""
fi

echo "Next steps:"
echo "  1. Review quality report: cat ${RUN_DIR}/evaluation/quality_report.json"
echo "  2. Check BLOCKING thresholds in training logs"
echo "  3. If thresholds pass → proceed to Phase 3 (Encoding + ComBat)"
