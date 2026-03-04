#!/usr/bin/env bash
#SBATCH -J smoke_dual_r8
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# DUAL-DOMAIN SMOKE TEST (loginexa / V100-DGXS-32GB)
#
# End-to-end pipeline test for the unified LoRA module:
#   1. Train dual_r8 for 2 epochs (MEN + GLI mixed-batch)
#   2. Extract per-domain features (MEN + GLI)
#   3. Evaluate per-domain Dice
#   4. Evaluate GP probes (per-domain + cross-domain)
#   5. Evaluate domain gap (MMD², CKA, PAD)
#   6. Generate tables
#   7. Validate outputs
#
# Expected runtime: ~30 min on V100-32GB
#
# Usage:
#   sbatch slurm/lora_adaptation/tests/smoke_loginexa.sh
#   # Or directly on login node with GPU:
#   bash slurm/lora_adaptation/tests/smoke_loginexa.sh
# =============================================================================

set -euo pipefail

# Resolve repo root
REPO_ROOT="${REPO_SRC:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}}"
CONDA_ENV="${CONDA_ENV_NAME:-growth}"
CONFIG_PATH="${REPO_ROOT}/experiments/lora/config/picasso/smoke_dual_r8.yaml"

# --- Results directory ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/smoke_dual_r8_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
LOGFILE="${RESULTS_DIR}/smoke_test.log"

# Tee all output to both stdout and the log file
exec > >(tee -a "${LOGFILE}") 2>&1

echo "========================================"
echo "DUAL-DOMAIN SMOKE TEST"
echo "========================================"
echo "Repo:    ${REPO_ROOT}"
echo "Conda:   ${CONDA_ENV}"
echo "Config:  ${CONFIG_PATH}"
echo "Results: ${RESULTS_DIR}"
echo "Date:    $(date)"
echo ""

# ========================================================================
# ENVIRONMENT
# ========================================================================
# --- Activate conda env ---
# On loginexa there is no 'module' command and conda is not in PATH.
# Use the env's bin/ directory directly.
ENV_DIR="${CONDA_ENV_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/${CONDA_ENV}}"
echo "Activating environment from: ${ENV_DIR}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "ERROR: Python not found at ${ENV_DIR}/bin/python" >&2
    echo "Set CONDA_ENV_DIR to the correct path and retry." >&2
    exit 1
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export CONDA_PREFIX="${ENV_DIR}"

echo "Python:   $(which python)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo ""

# --- GPU info ---
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo ""
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-all}"
echo "torch.cuda.device_count(): $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo "=== Pre-flight checks ==="

# Config
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[FAIL] Config not found: ${CONFIG_PATH}"
    exit 1
fi
echo "[OK] Config file exists"

# H5 data files
MEN_H5=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG_PATH}')); print(c['paths']['men_h5_file'])")
GLI_H5=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG_PATH}')); print(c['paths']['gli_h5_file'])")

if [ ! -f "${MEN_H5}" ]; then
    echo "[FAIL] MEN H5 not found: ${MEN_H5}"
    exit 1
fi
echo "[OK] MEN H5: ${MEN_H5}"

if [ ! -f "${GLI_H5}" ]; then
    echo "[FAIL] GLI H5 not found: ${GLI_H5}"
    exit 1
fi
echo "[OK] GLI H5: ${GLI_H5}"

# Import check
python -c "
from experiments.lora.run import main as _
print('[OK] Imports verified')
"
echo ""

# ========================================================================
# PIPELINE
# ========================================================================
COND="dual_r8"
START_TIME=$(date +%s)

# Step 1: Train
echo "=== [1/6] Training ${COND} (2 epochs) ==="
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    train --condition "${COND}" 2>&1 | tail -30
echo "[OK] Training complete"
echo ""

# Step 2: Extract features
echo "=== [2/6] Extracting per-domain features ==="
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    extract --condition "${COND}" 2>&1 | tail -20
echo "[OK] Features extracted"
echo ""

# Step 3: Dice evaluation
echo "=== [3/6] Evaluating per-domain Dice ==="
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    dice --condition "${COND}" 2>&1 | tail -20
echo "[OK] Dice evaluation complete"
echo ""

# Step 4: GP probes (per-domain + cross-domain)
echo "=== [4/6] Evaluating GP probes ==="
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    probes --condition "${COND}" 2>&1 | tail -30
echo "[OK] Probes evaluated"
echo ""

# Step 5: Domain gap
echo "=== [5/6] Evaluating domain gap ==="
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    domain-gap --condition "${COND}" 2>&1 | tail -20
echo "[OK] Domain gap evaluated"
echo ""

# Step 6: Tables
echo "=== [6/6] Generating tables ==="
python -m experiments.lora.run \
    --config "${CONFIG_PATH}" \
    generate-tables 2>&1 | tail -10
echo "[OK] Tables generated"
echo ""

# ========================================================================
# VALIDATION
# ========================================================================
echo "=== Output validation ==="

OUTPUT_DIR=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG_PATH}')); print(c['experiment']['output_dir'])")
COND_DIR="${OUTPUT_DIR}/conditions/${COND}"

PASS=true

# Training log
if [ -f "${COND_DIR}/training_log.csv" ]; then
    ROWS=$(wc -l < "${COND_DIR}/training_log.csv")
    if [ "$ROWS" -ge 3 ]; then  # header + 2 epoch rows
        echo "[OK] training_log.csv: ${ROWS} lines"
    else
        echo "[FAIL] training_log.csv: only ${ROWS} lines (expected >= 3)"
        PASS=false
    fi
else
    echo "[FAIL] training_log.csv not found"
    PASS=false
fi

# Feature files
for domain in men gli; do
    for split in test probe_train; do
        feat="${COND_DIR}/features/${domain}_${split}_features.pt"
        if [ -f "${feat}" ]; then
            echo "[OK] ${domain}_${split}_features.pt exists"
        else
            echo "[FAIL] ${feat} not found"
            PASS=false
        fi
    done
done

# Dice results
for domain in men gli; do
    dice="${COND_DIR}/test_dice_${domain}.json"
    if [ -f "${dice}" ]; then
        echo "[OK] test_dice_${domain}.json exists"
    else
        echo "[FAIL] ${dice} not found"
        PASS=false
    fi
done

# Probe results
if ls "${COND_DIR}"/metrics_*.json 1>/dev/null 2>&1; then
    echo "[OK] Probe metrics JSON(s) exist"
else
    echo "[FAIL] No probe metrics JSON found"
    PASS=false
fi

# Domain gap
if [ -f "${COND_DIR}/domain_gap.json" ]; then
    echo "[OK] domain_gap.json exists"
else
    echo "[WARN] domain_gap.json not found (may be non-fatal)"
fi

# Loss is finite
python -c "
import pandas as pd, sys
df = pd.read_csv('${COND_DIR}/training_log.csv')
loss = df['train_loss'].iloc[-1]
if pd.isna(loss) or loss > 1e6:
    print(f'[FAIL] Final train_loss = {loss} (not finite)')
    sys.exit(1)
print(f'[OK] Final train_loss = {loss:.4f}')
" || PASS=false

# ========================================================================
# SUMMARY
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
if [ "$PASS" = true ]; then
    echo "SMOKE TEST PASSED"
else
    echo "SMOKE TEST FAILED (see above)"
fi
echo "========================================"
echo "Duration: $(($ELAPSED / 60))m $(($ELAPSED % 60))s"
echo "Results:  ${RESULTS_DIR}"
echo "Finished: $(date)"

if [ "$PASS" = false ]; then
    exit 1
fi
