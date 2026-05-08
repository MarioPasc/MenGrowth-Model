#!/bin/bash
# ---------------------------------------------------------------------------
# Post-array merge + H5 patch for the LoRA re-inference.
#   1. Concatenate ${OUTPUT_DIR}/shards/shard_*.csv → recomputed_uncertainty.csv
#   2. Run patch_h5_uncertainty.py to overwrite the broken /uncertainty/
#      datasets in the v5 H5 (timestamped backup created automatically).
# ---------------------------------------------------------------------------

#SBATCH --job-name=uq_diag_reinfer_merge
#SBATCH --output=uq_diag_reinfer_merge_%j.out
#SBATCH --error=uq_diag_reinfer_merge_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00

set -euo pipefail

echo "=== Merge + patch on $(hostname) at $(date) ==="
echo "Output dir:    ${OUTPUT_DIR}"
echo "MenGrowth H5:  ${MENGROWTH_H5}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

RECOMPUTED_CSV="${OUTPUT_DIR}/recomputed_uncertainty.csv"

python - <<PY
import glob, sys
import pandas as pd
files = sorted(glob.glob("${OUTPUT_DIR}/shards/shard_*.csv"))
if not files: sys.exit(f"No shard CSVs found under ${OUTPUT_DIR}/shards/")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["scan_idx_in_h5"], keep="last")
df = df.sort_values("scan_idx_in_h5").reset_index(drop=True)
df.to_csv("${RECOMPUTED_CSV}", index=False)
print(f"Concatenated {len(files)} shards -> {len(df)} rows ({df.columns.size} cols)")
PY

python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.patch_h5_uncertainty \
    --csv "${RECOMPUTED_CSV}" \
    --h5 "${MENGROWTH_H5}"

echo "=== Done at $(date) ==="
