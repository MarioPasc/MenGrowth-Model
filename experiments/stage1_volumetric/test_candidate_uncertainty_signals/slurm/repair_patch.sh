#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 0 (H5 repair) post-array patch task.
# Concatenates the per-task CSVs from the repair array, then runs
# patch_h5_uncertainty.py to overwrite the broken /uncertainty/ datasets
# in the v5 H5 (creates a timestamped backup first).
# ---------------------------------------------------------------------------

#SBATCH --job-name=uq_diag_patch
#SBATCH --output=uq_diag_patch_%j.out
#SBATCH --error=uq_diag_patch_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00

set -euo pipefail

echo "=== Stage 0 patch on $(hostname) at $(date) ==="
echo "Config:           ${CONFIG_PATH}"
echo "Per-task CSV dir: ${TASK_CSV_DIR}"
echo "Recomputed CSV:   ${RECOMPUTED_CSV}"
echo "MenGrowth H5:     ${MENGROWTH_H5}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${RECOMPUTED_CSV}")"
python - <<PY
import glob, sys
import pandas as pd
files = sorted(glob.glob("${TASK_CSV_DIR}/task_*.csv"))
if not files:
    sys.exit("No per-task CSVs found under ${TASK_CSV_DIR}")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["scan_idx_in_h5"], keep="last")
df = df.sort_values("scan_idx_in_h5").reset_index(drop=True)
df.to_csv("${RECOMPUTED_CSV}", index=False)
print(f"Concatenated {len(files)} per-task CSVs -> {len(df)} rows")
PY

python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.patch_h5_uncertainty \
    --csv "${RECOMPUTED_CSV}" \
    --h5 "${MENGROWTH_H5}"

echo "=== Patch complete at $(date) ==="
