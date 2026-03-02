#!/usr/bin/env bash
#SBATCH -J domain_gap_plots
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=cpu

# =============================================================================
# DOMAIN GAP — FIGURE & TABLE GENERATION (CPU-only)
#
# Runs after the GPU pipeline completes. Reads saved features/dice/metrics
# and generates publication-quality figures and LaTeX table.
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONDA_ENV_NAME, OUTPUT_DIR
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================="
echo "DOMAIN GAP — PLOTS & TABLE"
echo "=========================================="
echo "Started:     $(date)"
echo "Hostname:    $(hostname)"
echo "SLURM Job:   ${SLURM_JOB_ID:-local}"
echo "Output Dir:  ${OUTPUT_DIR:-NOT SET}"
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

echo ""

# ========================================================================
# CHECK PREREQUISITE DATA EXISTS
# ========================================================================
if [ ! -f "${OUTPUT_DIR}/metrics/domain_metrics.json" ]; then
    echo "[FAIL] GPU pipeline data not found — GPU job likely failed."
    echo "  Expected: ${OUTPUT_DIR}/metrics/domain_metrics.json"
    exit 1
fi
echo "[OK] GPU pipeline data found"
echo ""

# ========================================================================
# GENERATE FIGURES
# ========================================================================
echo "[1/2] Generating figures..."
python -m experiments.domain_gap.plot_domain_gap --output-dir "${OUTPUT_DIR}"
echo "  [OK] Figures saved"

# ========================================================================
# GENERATE LATEX TABLE
# ========================================================================
echo "[2/2] Generating LaTeX table..."
python -m experiments.domain_gap.generate_latex_table --output-dir "${OUTPUT_DIR}"
echo "  [OK] Table saved"

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "PLOTS & TABLE COMPLETED"
echo "=========================================="
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Finished: $(date)"
