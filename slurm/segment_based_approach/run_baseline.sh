#!/usr/bin/env bash
#SBATCH -J a0_baseline
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# ABLATION A0: SEGMENT-BASED BASELINE — COMPUTE JOB (submitted by launch.sh)
#
# 1. Sliding-window segmentation on all MenGrowth scans (GPU)
# 2. Volume extraction + Dice computation
# 3. LOPO-CV with ScalarGP on manual and predicted volumes (CPU)
# 4. Figure generation (CPU)
#
# Expected runtime: ~2-3h (95 scans × ~1-2 min each, then ~5 min LOPO)
#
# Direct usage (for debugging):
#   CONFIG_PATH=... LOG_DIR=... bash slurm/segment_based_approach/run_baseline.sh
# =============================================================================

set -euo pipefail

# ---- Configuration (set by launch.sh via --export) ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-growth}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-experiments/segment_based_approach/config.yaml}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"
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

log_header "ABLATION A0: SEGMENT-BASED BASELINE"
echo "Started:         $(date)"
echo "Hostname:        $(hostname)"
echo "SLURM Job:       ${SLURM_JOB_ID:-local}"
echo "Config:          ${CONFIG_PATH}"
echo "Log dir:         ${LOG_DIR}"
echo "GPU:             ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Force recompute: ${FORCE_RECOMPUTE}"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log_header "ENVIRONMENT"
echo "[python] $(which python)"
python -c "
import torch
print(f'[torch]  {torch.__version__}  CUDA={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[gpu]    {torch.cuda.get_device_name(0)}')
    print(f'[vram]   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import GPy
print(f'[GPy]    {GPy.__version__}')
import h5py
print(f'[h5py]   {h5py.__version__}')
"

# =============================================================================
# RUN FULL BASELINE PIPELINE
# =============================================================================
log_header "RUNNING BASELINE PIPELINE"
STEP_START=$(date +%s)

FORCE_FLAG=""
if [ "${FORCE_RECOMPUTE}" = "1" ]; then
    FORCE_FLAG="--force-recompute"
    echo "[info] FORCE_RECOMPUTE=1 — re-running segmentation"
fi

python -m experiments.stage1_volumetric.run_baseline \
    --config "${CONFIG_PATH}" \
    ${FORCE_FLAG} \
    2>&1 | tee "${LOG_DIR}/run_baseline.log"

log_elapsed "$STEP_START"

# =============================================================================
# SUMMARY
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

log_header "PIPELINE COMPLETE"
echo "Log directory:  ${LOG_DIR}"
echo "Total elapsed:  $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Finished:       $(date)"

# Print results summary
eval "$(python -c "
import yaml
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
print(f'OUTPUT_DIR=\"{cfg[\"paths\"][\"output_dir\"]}\"')
")"

if [[ ! "${OUTPUT_DIR}" = /* ]]; then
    OUTPUT_DIR="${REPO_DIR}/${OUTPUT_DIR}"
fi

# Print all LOPO results (multi-model)
for f in ${OUTPUT_DIR}/lopo_results_*.json; do
    if [ -f "${f}" ]; then
        BASENAME=$(basename "${f}")
        echo ""
        echo "--- ${BASENAME} ---"
        python -c "
import json
with open('${f}') as fp:
    data = json.load(fp)
print(f'  Model: {data.get(\"model_name\", \"?\")}')
print(f'  Folds: {data.get(\"n_folds\", \"?\")} ({data.get(\"n_failed\", 0)} failed)')
if 'aggregate_metrics' in data:
    for k, v in sorted(data['aggregate_metrics'].items()):
        print(f'  {k}: {v:.4f}')
"
    fi
done

# Print volume/segmentation summaries
for f in volume_summary.json segmentation_comparison.json model_comparison.json; do
    RESULT_FILE="${OUTPUT_DIR}/${f}"
    if [ -f "${RESULT_FILE}" ]; then
        echo ""
        echo "--- ${f} ---"
        python -c "
import json
with open('${RESULT_FILE}') as fp:
    data = json.load(fp)
# Print top-level scalar values
for k, v in data.items():
    if isinstance(v, (int, float, str, bool)):
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
# Print per_region summary if present
if 'per_region' in data:
    for region, stats in data['per_region'].items():
        dice_mean = stats.get('dice_mean', 0)
        dice_std = stats.get('dice_std', 0)
        vol_r = stats.get('volume_pearson_r', 0)
        print(f'  {region.upper()}: Dice={dice_mean:.3f}+/-{dice_std:.3f}, Vol r={vol_r:.3f}')
"
    fi
done

echo ""
echo "Figures: ${OUTPUT_DIR}/figures/"
echo "Done."
