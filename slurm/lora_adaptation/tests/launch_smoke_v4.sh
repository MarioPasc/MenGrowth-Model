#!/usr/bin/env bash
#SBATCH -J gp_smoke_gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# GP PROBE GPU SMOKE TEST
#
# Runs a minimal end-to-end pipeline on Picasso:
#   1. Train baseline_frozen + lora_r8 for 3 epochs on tiny subset
#   2. Extract features
#   3. Run GP probes
#   4. Verify full pipeline works
#
# Usage:
#   sbatch slurm/lora_adaptation/launch_smoke_v4.sh
# =============================================================================

set -euo pipefail

# Resolve repo root: prefer REPO_SRC, then SLURM_SUBMIT_DIR (set by sbatch),
# then fall back to dirname (only works when running directly, not via sbatch).
REPO_ROOT="${REPO_SRC:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
CONDA_ENV="${CONDA_ENV_NAME:-growth}"

# --- Results directory ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/smoke_tests/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
LOGFILE="${RESULTS_DIR}/smoke_test_gpu.log"

# Tee all output to both stdout and the log file
exec > >(tee -a "${LOGFILE}") 2>&1

echo "========================================"
echo "GP Probe GPU Smoke Test"
echo "Repo: ${REPO_ROOT}"
echo "Conda: ${CONDA_ENV}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Results: ${RESULTS_DIR}"
echo "Date: $(date)"
echo "========================================"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# --- Step 1: Run CPU smoke test first ---
echo ""
echo "=== Step 1: CPU smoke test ==="
bash "${REPO_ROOT}/slurm/lora_adaptation/smoke_test_v4.sh"

# --- Step 2: GPU-accelerated end-to-end ---
echo ""
echo "=== Step 2: GPU pipeline smoke test ==="
echo "This step requires the full BraTS-MEN dataset and model weights."
echo "Skipping if data is not available."

CONFIG_PATH="${REPO_ROOT}/experiments/lora_ablation/config/picasso/v3_rank_sweep.yaml"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Config not found: ${CONFIG_PATH}"
    echo "Skipping GPU smoke test."
    exit 0
fi

# Check if H5 data is available
H5_PATH=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG_PATH}')); print(c.get('paths',{}).get('h5_file',''))" 2>/dev/null || echo "")

if [ -z "${H5_PATH}" ] || [ ! -f "${H5_PATH}" ]; then
    echo "H5 data file not found. Skipping GPU pipeline test."
    echo "To run the full smoke test, ensure the H5 file is available."
    exit 0
fi

echo "H5 data found: ${H5_PATH}"

# --- Generate splits if they don't exist ---
echo ""
echo "=== Step 2a: Ensure data splits exist ==="
python -m experiments.lora_ablation.pipeline.data_splits \
    --config "${CONFIG_PATH}" 2>&1 | tail -20

echo ""
echo "Running baseline_frozen condition (3 epochs)..."

# Run a quick training + extraction + probe evaluation for one condition
python -m experiments.lora_ablation.pipeline.train_condition \
    --config "${CONFIG_PATH}" \
    --condition baseline_frozen \
    --max-epochs 3 \
    --device cuda 2>&1 | tee "${RESULTS_DIR}/train_condition.log" | tail -20

echo ""
echo "Running feature extraction..."
python -m experiments.lora_ablation.pipeline.extract_features \
    --config "${CONFIG_PATH}" \
    --condition baseline_frozen \
    --device cuda 2>&1 | tee "${RESULTS_DIR}/extract_features.log" | tail -10

echo ""
echo "Running GP probe evaluation..."
python -m experiments.lora_ablation.pipeline.evaluate_probes \
    --config "${CONFIG_PATH}" \
    --condition baseline_frozen \
    --device cuda 2>&1 | tee "${RESULTS_DIR}/evaluate_probes.log" | tail -30

echo ""
echo "========================================"
echo "GPU Smoke Test PASSED"
echo "Results saved to: ${RESULTS_DIR}"
echo "========================================"
