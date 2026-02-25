#!/usr/bin/env bash
#SBATCH -J sdp_encode
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=sdp_encode_%j.out
#SBATCH --error=sdp_encode_%j.err

# =============================================================================
# SDP MODULE — FEATURE ENCODING (SLURM job, no internet)
#
# Extracts encoder10 features (768-dim) from all BraTS-MEN subjects using
# the LoRA-adapted encoder. Reads from H5 file for fast I/O.
#
# Output: per-split HDF5 files with features + semantic targets in
#   ${OUTPUT_DIR}/features/{split_name}.h5
#
# Usage:
#   sbatch slurm/sdp/encode_volumes.sh
#   # or with custom config:
#   CONFIG_PATH=path/to/config.yaml sbatch slurm/sdp/encode_volumes.sh
# =============================================================================

set -euo pipefail

# ---- Configuration ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-experiments/sdp/config/picasso/sdp_default.yaml}"

START_TIME=$(date +%s)
echo "=========================================="
echo "SDP MODULE — FEATURE ENCODING"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "Config:      ${CONFIG_PATH}"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-not set}"
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
import yaml
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
        print(f'  MISSING: {desc} → {path}')
        ok = False
    else:
        print(f'  SKIP: {desc} (not configured)')

if not ok:
    raise RuntimeError('Pre-flight check failed')
print('  All pre-flight checks passed!')
"
echo ""

# ---- Run Feature Extraction ----
echo "=========================================="
echo "RUNNING FEATURE EXTRACTION"
echo "=========================================="

python -m experiments.sdp.extract_all_features \
    --config "${CONFIG_PATH}" \
    --device cuda

# ---- Post-flight Checks ----
echo ""
echo "=========================================="
echo "POST-FLIGHT CHECKS"
echo "=========================================="

python -c "
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load('${CONFIG_PATH}')
features_dir = Path(cfg.paths.features_dir)

if not features_dir.exists():
    raise RuntimeError(f'Features dir not found: {features_dir}')

h5_files = sorted(features_dir.glob('*.h5'))
print(f'  Features directory: {features_dir}')
print(f'  H5 files found: {len(h5_files)}')

for f in h5_files:
    import h5py
    with h5py.File(f, 'r') as h:
        n = len(h['subject_ids'])
        feat_shape = h['features/encoder10'].shape if 'features/encoder10' in h else '?'
    size_mb = f.stat().st_size / (1024**2)
    print(f'    {f.name}: {n} subjects, encoder10={feat_shape}, {size_mb:.1f} MB')

if not h5_files:
    raise RuntimeError('No output H5 files found!')
"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "FEATURE ENCODING COMPLETE"
echo "=========================================="
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Finished: $(date)"
