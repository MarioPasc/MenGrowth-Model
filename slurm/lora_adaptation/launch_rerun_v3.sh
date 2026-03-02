#!/usr/bin/env bash
# =============================================================================
# LORA V3 RANK SWEEP — RE-RUN LAUNCHER (POST-TRAINING)
#
# Re-runs Steps 2-5 (extract, domain, probes, dice) from existing checkpoints
# after bug fixes (TAP extractor + OOM handling), then runs analysis.
#
# Conditions (SLURM_ARRAY_TASK_ID):
#   0: baseline_frozen    3: lora_r8_full     6: lora_r64_full
#   1: baseline           4: lora_r16_full
#   2: lora_r4_full       5: lora_r32_full
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model
#   bash slurm/lora_adaptation/launch_rerun_v3.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "LORA V3 RANK SWEEP — RE-RUN LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model"
export CONDA_ENV_NAME="growth"
export CONFIG_PATH="${REPO_SRC}/experiments/lora_ablation/config/picasso/v3_rank_sweep.yaml"

echo "Activating conda environment: ${CONDA_ENV_NAME}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo ""

OUTPUT_DIR=$(python3 -c "
import yaml
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment']['output_dir'])
")
export OUTPUT_DIR

SLURM_LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${SLURM_LOG_DIR}"

echo "Configuration:"
echo "  Repo:       ${REPO_SRC}"
echo "  Config:     ${CONFIG_PATH}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Logs:       ${SLURM_LOG_DIR}"
echo "  Conda env:  ${CONDA_ENV_NAME}"
echo ""

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo "Pre-flight checks:"

if [ -f "${CONFIG_PATH}" ]; then
    echo "  [OK]   Config file"
else
    echo "  [FAIL] Config not found: ${CONFIG_PATH}"
    exit 1
fi

# Verify checkpoints exist for all trained conditions
python3 -c "
import yaml
from pathlib import Path

with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)

output_dir = Path(cfg['experiment']['output_dir'])
checkpoint = Path(cfg['paths']['checkpoint'])
h5_file = Path(cfg['paths']['h5_file'])

assert checkpoint.exists(), f'Checkpoint not found: {checkpoint}'
print(f'  [OK]   Checkpoint: {checkpoint}')

assert h5_file.exists(), f'H5 file not found: {h5_file}'
print(f'  [OK]   H5 file: {h5_file}')

for cond in cfg['conditions']:
    name = cond['name']
    cond_dir = output_dir / 'conditions' / name
    if cond.get('skip_training', False):
        print(f'  [OK]   {name} (skip_training, no checkpoint needed)')
        continue
    best_model = cond_dir / 'best_model.pt'
    assert best_model.exists(), f'Missing checkpoint: {best_model}'
    print(f'  [OK]   {name}: best_model.pt exists')

print(f'  [OK]   All {len(cfg[\"conditions\"])} conditions ready for re-run')
"

if [ $? -ne 0 ]; then
    echo "  [FAIL] Pre-flight checks failed"
    exit 1
fi

echo ""

# ========================================================================
# STEP 1: Submit array job (7 conditions, 1 GPU each, 4h walltime)
# ========================================================================
echo "Submitting re-run array job (Steps 2-5 only, skip training)..."

ARRAY_JOB_RAW=$(sbatch --parsable \
    --array=0-6 \
    --job-name="v3_rerun" \
    --output="${SLURM_LOG_DIR}/rerun_%a_%j.out" \
    --error="${SLURM_LOG_DIR}/rerun_%a_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    "${SCRIPT_DIR}/rerun_worker_v3.sh")

ARRAY_JOB_ID="${ARRAY_JOB_RAW%%[_;]*}"

echo "  Array job ID: ${ARRAY_JOB_ID} (7 elements, 0-6)"
echo ""

# ========================================================================
# STEP 2: Submit analysis job (dependent on array completion)
# ========================================================================
echo "Submitting analysis job (dependent on array ${ARRAY_JOB_ID})..."

ANALYSIS_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --job-name="v3_analysis" \
    --constraint=cpu \
    --output="${SLURM_LOG_DIR}/analysis_rerun_%j.out" \
    --error="${SLURM_LOG_DIR}/analysis_rerun_%j.err" \
    --export=ALL,CONFIG_PATH="${CONFIG_PATH}",REPO_SRC="${REPO_SRC}",CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    "${SCRIPT_DIR}/analysis_worker_v3.sh")

echo "  Analysis job ID: ${ANALYSIS_JOB_ID}"
echo ""

# ========================================================================
# MONITORING COMMANDS
# ========================================================================
echo "=========================================="
echo "ALL JOBS SUBMITTED"
echo "=========================================="
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -j ${ARRAY_JOB_ID}         # re-run array"
echo "  squeue -j ${ANALYSIS_JOB_ID}       # analysis (dependent)"
echo ""
echo "Per-condition logs:"
echo "  tail -f ${SLURM_LOG_DIR}/rerun_<ARRAY_ID>_<JOB_ID>.out"
echo ""
echo "Cancel all:"
echo "  scancel ${ARRAY_JOB_ID} ${ANALYSIS_JOB_ID}"
echo ""
echo "Estimated timeline:"
echo "  Re-run:    ~1-2h per condition (4h limit)"
echo "  Analysis:  ~30 min after all re-runs complete"
echo "  Total:     ~2.5h (all 7 conditions in parallel)"
