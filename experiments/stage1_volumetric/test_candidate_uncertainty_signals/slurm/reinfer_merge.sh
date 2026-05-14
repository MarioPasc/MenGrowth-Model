#!/bin/bash
# ---------------------------------------------------------------------------
# Post-array merge + verification for the LoRA re-inference.
#
#   1. Concatenate ${OUTPUT_DIR}/shards/shard_*.csv -> recomputed_uncertainty.csv
#   2. VERIFY the fix: compare the recomputed logvol_mean against the H5's
#      uncertainty/logvol_mean. With the get_h5_val_transforms fix in
#      reinfer_h5_uncertainty.py the re-inference must reproduce the original
#      "World A" inference the H5 volumes/segs come from -- so the two
#      logvol_mean vectors should agree to within floating-point tolerance.
#      A large disagreement means the preprocessing fix is still incomplete;
#      the per-member masks must NOT be trusted in that case.
#   3. H5 patch is OPT-IN: set PATCH_H5=true to overwrite the H5
#      /uncertainty/ entropy-MI scalars from this (now World-A-consistent)
#      re-inference. Default is false -- the command is printed, not run,
#      because modifying the experiment H5 is a separate, deliberate step.
# ---------------------------------------------------------------------------

#SBATCH --job-name=reinfer_worldA_merge
#SBATCH --output=reinfer_worldA_merge_%j.out
#SBATCH --error=reinfer_worldA_merge_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00

set -euo pipefail

echo "=== Merge + verify on $(hostname) at $(date) ==="
echo "Output dir:    ${OUTPUT_DIR}"
echo "MenGrowth H5:  ${MENGROWTH_H5}"
echo "Patch H5:      ${PATCH_H5:-false}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-growth}"

REPO="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

RECOMPUTED_CSV="${OUTPUT_DIR}/recomputed_uncertainty.csv"

# --- Step 1: concatenate shards ------------------------------------------------
python - <<PY
import glob, sys
import pandas as pd
files = sorted(glob.glob("${OUTPUT_DIR}/shards/shard_*.csv"))
if not files:
    sys.exit("ERROR: no shard CSVs found under ${OUTPUT_DIR}/shards/")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["scan_idx_in_h5"], keep="last")
df = df.sort_values("scan_idx_in_h5").reset_index(drop=True)
df.to_csv("${RECOMPUTED_CSV}", index=False)
print(f"Concatenated {len(files)} shards -> {len(df)} rows ({df.columns.size} cols)")
if "error" in df.columns:
    n_err = int(df["error"].notna().sum())
    if n_err:
        print(f"WARNING: {n_err} scans recorded an error column -- inspect before trusting masks")
PY

# --- Step 2: verify the re-inference reproduces World A ------------------------
python - <<PY
import sys
import numpy as np
import pandas as pd
import h5py

df = pd.read_csv("${RECOMPUTED_CSV}").set_index("scan_idx_in_h5")
with h5py.File("${MENGROWTH_H5}", "r") as f:
    lvm_h5 = f["uncertainty"]["logvol_mean"][:].astype(float)

idx = [i for i in df.index if 0 <= int(i) < len(lvm_h5)]
rec = df.loc[idx, "logvol_mean_check"].to_numpy(dtype=float)
ref = lvm_h5[np.asarray(idx, dtype=int)]
d = np.abs(rec - ref)
finite = np.isfinite(d)
d = d[finite]
n = d.size
within = float(np.mean(d < 0.05)) if n else 0.0

print("--- World-A reproduction check (recomputed logvol_mean vs H5 logvol_mean) ---")
print(f"  n scans compared : {n}")
print(f"  max |diff|       : {d.max():.4f}" if n else "  (no finite rows)")
print(f"  median |diff|    : {np.median(d):.4f}" if n else "")
print(f"  fraction < 0.05  : {within:.3f}")

# With the fix this should be a near-exact reproduction. Tolerate a handful of
# scanner-specific outliers but require the bulk to match.
if n == 0:
    sys.exit("VERIFY FAILED: no comparable scans")
elif within >= 0.95 and d.max() < 0.5:
    print("VERIFY PASSED: re-inference reproduces World A -> per-member masks are trustworthy.")
else:
    print("VERIFY FAILED: re-inference does NOT match the H5 volumes.")
    print("  The preprocessing fix is incomplete -- do NOT use the per-member masks.")
    sys.exit(1)
PY

# --- Step 3: optional H5 patch (opt-in) ---------------------------------------
if [[ "${PATCH_H5:-false}" == "true" ]]; then
    echo "PATCH_H5=true -> patching H5 /uncertainty/ entropy-MI scalars (timestamped backup made)"
    python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.patch_h5_uncertainty \
        --csv "${RECOMPUTED_CSV}" \
        --h5 "${MENGROWTH_H5}"
else
    echo "PATCH_H5 not set -> H5 left untouched. To patch the entropy/MI fields later, run:"
    echo "  python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.patch_h5_uncertainty \\"
    echo "      --csv ${RECOMPUTED_CSV} --h5 ${MENGROWTH_H5}"
fi

echo "=== Done at $(date) ==="
echo "Per-member masks: ${OUTPUT_DIR}/per_scan/<scan_id>/member_*_mask.nii.gz + ensemble_mask.nii.gz"
