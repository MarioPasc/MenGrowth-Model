#!/usr/bin/env bash
#SBATCH -J ablation_train
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=ablation_train_%j.out
#SBATCH --error=ablation_train_%j.err

# =============================================================================
# LORA ABLATION — TRAINING WORKER
#
# Runs a single LoRA/DoRA ablation experiment on one A100 80GB GPU.
# Executes the full pipeline: train all conditions → extract features →
# evaluate probes → compute Dice → generate visualizations.
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "LORA ABLATION TRAINING WORKER"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "Config:      ${CONFIG_PATH:-NOT SET}"
echo ""

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

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Devices:', torch.cuda.device_count()); print('bf16:', torch.cuda.is_bf16_supported())"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Verify config exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[FAIL] Config not found: ${CONFIG_PATH}"
    exit 1
fi
echo "[OK]   Config: ${CONFIG_PATH}"

# Verify data and checkpoint paths from config
python -c "
import yaml
from pathlib import Path

with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)

checkpoint = Path(cfg['paths']['checkpoint'])
data_root = Path(cfg['paths']['data_root'])
glioma_root = Path(cfg['paths'].get('glioma_root', ''))

print(f'Checkpoint: {checkpoint} (exists={checkpoint.exists()})')
print(f'Data root:  {data_root} (exists={data_root.exists()})')
if glioma_root:
    print(f'GLI root:   {glioma_root} (exists={glioma_root.exists()})')

assert checkpoint.exists(), f'Checkpoint not found: {checkpoint}'
assert data_root.exists(), f'Data root not found: {data_root}'

# Log key training params
t = cfg['training']
print(f'batch_size={t[\"batch_size\"]}, max_epochs={t[\"max_epochs\"]}, '
      f'use_amp={t.get(\"use_amp\", False)}, num_workers={t[\"num_workers\"]}')
print('Pre-flight checks PASSED')
"

if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED — aborting."
    exit 1
fi

# Quick import check
python -c "
from growth.data.bratsmendata import create_dataloaders
from growth.losses.segmentation import SegmentationLoss3Ch, DiceMetric3Ch
from growth.losses.encoder_vicreg import EncoderVICRegLoss
from experiments.lora_ablation.model_factory import create_ablation_model
from experiments.lora_ablation.run_ablation import main as run_ablation_main
from experiments.lora_ablation.evaluate_feature_quality import evaluate_feature_quality_single
print('All imports OK')
"

if [ $? -ne 0 ]; then
    echo "IMPORT CHECK FAILED — aborting."
    exit 1
fi

echo ""

# ========================================================================
# RUN ABLATION
# ========================================================================
echo "=========================================="
echo "RUNNING ABLATION PIPELINE"
echo "=========================================="
echo "Config: $(basename "${CONFIG_PATH}")"
echo ""

python -m experiments.lora_ablation.run_ablation \
    --config "${CONFIG_PATH}" \
    run-all --domain-features

ABLATION_EXIT=$?

# ========================================================================
# POST-FLIGHT VERIFICATION
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

# Extract output_dir from config
OUTPUT_DIR=$(python -c "
import yaml
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment']['output_dir'])
")

EXPECTED_FILES=(
    "${OUTPUT_DIR}/comprehensive_results.csv"
    "${OUTPUT_DIR}/test_dice_summary.csv"
    "${OUTPUT_DIR}/feature_quality_comparison.csv"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   $f (${SIZE} bytes)"
    else
        echo "[MISS] $f"
        MISSING=$((MISSING + 1))
    fi
done

# Check condition directories
CONDITION_COUNT=$(ls -d "${OUTPUT_DIR}/conditions"/*/ 2>/dev/null | wc -l)
echo "Condition directories: ${CONDITION_COUNT}"

if [ "$MISSING" -gt 0 ]; then
    echo "WARNING: ${MISSING} expected files missing."
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "TRAINING WORKER COMPLETED"
echo "=========================================="
echo "Config:     $(basename "${CONFIG_PATH}")"
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Exit code:  ${ABLATION_EXIT}"

if [ "$ABLATION_EXIT" -eq 0 ]; then
    echo "Ablation pipeline completed successfully."
else
    echo "Ablation pipeline FAILED with exit code ${ABLATION_EXIT}."
    exit "${ABLATION_EXIT}"
fi
