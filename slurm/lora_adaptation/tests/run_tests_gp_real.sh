#!/usr/bin/env bash
#SBATCH -J gp_real_tests
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# GP PROBE REAL-DATA INTEGRATION TESTS
#
# Runs pytest tests/growth/test_gp_probes_real_data.py on a GPU node.
# Pipeline per test class:
#   1. baseline_frozen: extract 768-dim features → GP probes (linear + RBF)
#   2. lora_r8_full: train 3 epochs → extract features → GP probes
#   3. Full evaluate_probes pipeline → verify JSON outputs
#
# Usage:
#   sbatch slurm/lora_adaptation/run_tests_gp_real.sh
#
# Or from login node:
#   bash slurm/lora_adaptation/run_tests_gp_real.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV_NAME:-growth}"

echo "=========================================="
echo "GP Probe Real-Data Integration Tests"
echo "=========================================="
echo "Repo:     ${REPO_ROOT}"
echo "Conda:    ${CONDA_ENV}"
echo "Host:     $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Date:     $(date)"
echo ""

# --- Environment setup ---
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
    conda activate "${CONDA_ENV}" 2>/dev/null || source activate "${CONDA_ENV}"
else
    source activate "${CONDA_ENV}"
fi

echo "[python] $(which python)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# --- Pre-flight: check data availability ---
echo "Pre-flight checks:"
python -c "
from pathlib import Path
h5 = Path('${REPO_ROOT}/../../../fscratch/datasets/meningiomas/brats_men/BraTS_MEN.h5')
# Try Picasso default paths
import os
h5_env = os.environ.get('MENGROWTH_H5_PATH', '')
ckpt_env = os.environ.get('MENGROWTH_CKPT_PATH', '')

picasso_h5 = '/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/meningiomas/brats_men/BraTS_MEN.h5'
picasso_ckpt = '/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/BrainSegFounder_finetuned_BraTS/finetuned_model_fold_0.pt'

h5_found = any(Path(p).exists() for p in [h5_env, picasso_h5] if p)
ckpt_found = any(Path(p).exists() for p in [ckpt_env, picasso_ckpt] if p)

print(f'  H5 data:    {\"OK\" if h5_found else \"NOT FOUND\"} ')
print(f'  Checkpoint: {\"OK\" if ckpt_found else \"NOT FOUND\"} ')

if not h5_found or not ckpt_found:
    print('  WARNING: Tests will be SKIPPED if data is not found.')
"
echo ""

# --- Run unit tests first (quick sanity check) ---
echo "=== Step 1/2: Synthetic GP tests (quick) ==="
python -m pytest tests/growth/test_gp_probes.py -v --tb=short
echo ""

# --- Run real-data integration tests ---
echo "=== Step 2/2: Real-data GP integration tests ==="
python -m pytest tests/growth/test_gp_probes_real_data.py -v -s --tb=long 2>&1

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "ALL GP REAL-DATA TESTS PASSED"
elif [ $EXIT_CODE -eq 5 ]; then
    echo "TESTS SKIPPED (data not available)"
    EXIT_CODE=0  # Not a failure — just skipped
else
    echo "SOME TESTS FAILED (exit code: $EXIT_CODE)"
fi
echo "=========================================="
echo "Finished: $(date)"

exit $EXIT_CODE
