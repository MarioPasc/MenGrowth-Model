#!/usr/bin/env bash
#SBATCH -J v3_analysis
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=cpu

# =============================================================================
# LORA V3 — ANALYSIS WORKER (CPU-only)
#
# Runs after all array elements complete. Aggregates feature quality,
# generates figures, tables, and reports.
#
# Expected env vars (exported by launch_v3.sh):
#   REPO_SRC, CONDA_ENV_NAME, CONFIG_PATH
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "LORA V3 ANALYSIS WORKER"
echo "=========================================="
echo "Started:  $(date)"
echo "Hostname: $(hostname)"
echo "Config:   ${CONFIG_PATH:-NOT SET}"
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

cd "${REPO_SRC}"

# ========================================================================
# ANALYSIS PIPELINE
# ========================================================================

# Step 1: Feature quality evaluation (all conditions)
echo "[1/2] Feature quality evaluation..."
python -m experiments.lora_ablation.run_ablation \
    --config "${CONFIG_PATH}" \
    feature-quality
echo "  [OK] Feature quality complete"

# Step 2: Regenerate analysis (figures, tables, reports)
echo "[2/2] Regenerating analysis..."
python -m experiments.lora_ablation.analysis.regenerate_analysis \
    --config "${CONFIG_PATH}"
echo "  [OK] Analysis regenerated"

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "ANALYSIS COMPLETE"
echo "=========================================="
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Finished: $(date)"

# Print output summary
OUTPUT_DIR=$(python3 -c "
import yaml
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment']['output_dir'])
")

echo ""
echo "Output:"
echo "  ${OUTPUT_DIR}/figures/      (8 PDF + PNG figures)"
echo "  ${OUTPUT_DIR}/tables/       (LaTeX tables)"
echo "  ${OUTPUT_DIR}/results/      (CSV summaries)"
echo "  ${OUTPUT_DIR}/reports/      (analysis + recommendation)"
