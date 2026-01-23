#!/usr/bin/env bash
#SBATCH -J log_semivae
#SBATCH --time=2-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:4
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# =============================================================================
# SEMI-SUPERVISED VAE TRAINING (Exp4)
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"

# ========================================================================
# CONFIGURATION
# ========================================================================
EXPERIMENT_NAME="semivae"
CONDA_ENV_NAME="vae-dynamics"

REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model"
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/meningiomas/brats_men"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/${EXPERIMENT_NAME}"
CONFIG_FILE="${REPO_SRC}/src/vae/config/${EXPERIMENT_NAME}_run5.yaml"

# Number of GPUs (should match --gres=gpu:N)
NUM_GPUS=4

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[env] No conda module loaded; assuming conda already in PATH."
fi

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

# Verify environment
echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"
python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version())"

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ========================================================================
# PRE-TRAINING SETUP
# ========================================================================

# Create results directory
if [ ! -d "${RESULTS_DST}" ]; then
    echo "Creating results directory: ${RESULTS_DST}"
    mkdir -p "${RESULTS_DST}"
fi

# Create cache directory for semantic normalizer
CACHE_DIR="${RESULTS_DST}/cache_semivae"
mkdir -p "${CACHE_DIR}"

# Copy and modify configuration file
CONFIG_BASENAME=$(basename "${CONFIG_FILE}")
MODIFIED_CONFIG="${RESULTS_DST}/${CONFIG_BASENAME}"

echo "Copying config file to: ${MODIFIED_CONFIG}"
cp "${CONFIG_FILE}" "${MODIFIED_CONFIG}"

echo "Modifying configuration file..."
sed -i "s|  root_dir: .*|  root_dir: \"${DATA_SRC}\"|" "${MODIFIED_CONFIG}"
sed -i "s|  save_dir: .*|  save_dir: \"${RESULTS_DST}\"|" "${MODIFIED_CONFIG}"
sed -i "s|  persistent_cache_subdir: .*|  persistent_cache_subdir: \"${CACHE_DIR}\"|" "${MODIFIED_CONFIG}"

# Update GPU configuration based on available devices
sed -i "s|  devices: .*|  devices: ${NUM_GPUS}|" "${MODIFIED_CONFIG}"

# ========================================================================
# PRE-COMPUTE SEMANTIC NORMALIZER (if not exists)
# ========================================================================
NORMALIZER_FILE="${CACHE_DIR}/semantic_normalizer.json"

if [ ! -f "${NORMALIZER_FILE}" ]; then
    echo ""
    echo "=========================================="
    echo "PRE-COMPUTING SEMANTIC NORMALIZER"
    echo "=========================================="
    cd "${REPO_SRC}"

    python -m vae.utils.precompute_semantic_normalizer \
        --data-root "${DATA_SRC}" \
        --output-dir "${CACHE_DIR}" \
        --val-split 0.1 \
        --seed 42

    echo "Normalizer saved to: ${NORMALIZER_FILE}"
else
    echo "Semantic normalizer already exists: ${NORMALIZER_FILE}"
fi

# ========================================================================
# TRAINING EXECUTION
# ========================================================================
echo ""
echo "=========================================="
echo "STARTING SEMIVAE TRAINING"
echo "=========================================="
echo "Config: ${MODIFIED_CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo "Data: ${DATA_SRC}"
echo "Results: ${RESULTS_DST}"

cd "${REPO_SRC}"
echo "Working directory: $(pwd)"

# Run training
# The vae-train command handles DDP internally via PyTorch Lightning
vae-train --config "${MODIFIED_CONFIG}"

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "Job finished at: $(date)"
echo "Total execution time: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Results saved to: ${RESULTS_DST}"
echo "✅ SemiVAE training completed successfully."
