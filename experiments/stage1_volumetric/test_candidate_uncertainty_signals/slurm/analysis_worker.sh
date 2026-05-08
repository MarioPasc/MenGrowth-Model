#!/bin/bash
# ---------------------------------------------------------------------------
# Post-array analysis worker for the candidate-uncertainty-signal diagnostic.
# Runs Stage 1 extract + correlations, Stage 2 aggregation + bootstrap, and
# the two thesis figures. Depends on the Stage 2 array (afterany).
# ---------------------------------------------------------------------------

#SBATCH --job-name=uq_diag_analysis
#SBATCH --output=uq_diag_analysis_%j.out
#SBATCH --error=uq_diag_analysis_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00

set -euo pipefail

echo "=== Analysis on $(hostname) at $(date) ==="
echo "Config: ${CONFIG_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

echo "[1/4] Extract candidate signals from (post-repair) H5"
python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.extract_candidates \
    --config "${CONFIG_PATH}"

echo "[2/4] Stage 1 correlations (information-content diagnostic)"
python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.analyses.stage1_correlations \
    --config "${CONFIG_PATH}"

echo "[3/4] Stage 2 aggregation + paired BCa bootstrap + BH-FDR"
python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.analyses.aggregate_stage2 \
    --config "${CONFIG_PATH}"

echo "[4/4] Figures"
python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.analyses.plot_stage1_correlation \
    --config "${CONFIG_PATH}"
python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.analyses.plot_stage2_is_per_candidate \
    --config "${CONFIG_PATH}"

echo "=== Analysis complete at $(date) ==="
