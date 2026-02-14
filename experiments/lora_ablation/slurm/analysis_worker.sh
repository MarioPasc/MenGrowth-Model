#!/usr/bin/env bash
#SBATCH -J ablation_report
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=ablation_report_%j.out
#SBATCH --error=ablation_report_%j.err

# =============================================================================
# LORA ABLATION — ANALYSIS WORKER
#
# Generates cross-experiment HTML report after all 4 training workers complete.
# Runs with --dependency=afterok on all training jobs.
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONDA_ENV_NAME, RESULTS_DIR
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "LORA ABLATION — ANALYSIS WORKER"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "Results dir: ${RESULTS_DIR:-NOT SET}"
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

echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
echo ""

cd "${REPO_SRC}"

# ========================================================================
# PRE-FLIGHT: Verify training outputs exist
# ========================================================================
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

EXPERIMENT_DIRS=(
    "${RESULTS_DIR}/growth/lora_ablation_semantic_heads"
    "${RESULTS_DIR}/growth/lora_ablation_no_semantic_heads"
    "${RESULTS_DIR}/growth/dora_ablation_semantic_heads"
    "${RESULTS_DIR}/growth/dora_ablation_no_semantic_heads"
)

ALL_OK=true
for d in "${EXPERIMENT_DIRS[@]}"; do
    if [ -d "$d" ]; then
        CSV_COUNT=$(ls "$d"/comprehensive_results.csv 2>/dev/null | wc -l)
        echo "[OK]   $(basename "$d") (results csv: ${CSV_COUNT})"
    else
        echo "[MISS] $d"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" != "true" ]; then
    echo ""
    echo "WARNING: Some experiment directories missing. Report may be incomplete."
fi

echo ""

# ========================================================================
# GENERATE REPORT
# ========================================================================
echo "=========================================="
echo "GENERATING CROSS-EXPERIMENT REPORT"
echo "=========================================="

REPORT_DIR="${RESULTS_DIR}/growth/report"
mkdir -p "${REPORT_DIR}"

python -m experiments.lora_ablation.generate_report \
    --results-dir "${RESULTS_DIR}/growth" \
    --output-dir "${REPORT_DIR}" \
    --mode both \
    --compare-semantic

REPORT_EXIT=$?

# ========================================================================
# POST-FLIGHT VERIFICATION
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

if [ -d "${REPORT_DIR}" ]; then
    FILE_COUNT=$(find "${REPORT_DIR}" -type f | wc -l)
    echo "[OK]   Report directory: ${REPORT_DIR} (${FILE_COUNT} files)"

    # List generated files
    find "${REPORT_DIR}" -type f -name "*.html" -o -name "*.png" -o -name "*.csv" | sort | while read -r f; do
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "       $(basename "$f") (${SIZE} bytes)"
    done
else
    echo "[MISS] Report directory not created"
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "ANALYSIS WORKER COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Exit code:  ${REPORT_EXIT}"
echo "Report:     ${REPORT_DIR}"

if [ "$REPORT_EXIT" -eq 0 ]; then
    echo "Report generation completed successfully."
else
    echo "Report generation FAILED with exit code ${REPORT_EXIT}."
    exit "${REPORT_EXIT}"
fi
