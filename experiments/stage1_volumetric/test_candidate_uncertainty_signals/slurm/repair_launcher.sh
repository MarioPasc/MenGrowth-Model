#!/bin/bash
# ---------------------------------------------------------------------------
# Submit Stage 0 (H5 /uncertainty/ repair) — one array task per scan +
# a dependent patch task that merges the per-task CSVs and writes the H5.
# CPU-only.
#
# Usage:
#   bash repair_launcher.sh [CONFIG_PATH] [--dry-run]
# ---------------------------------------------------------------------------

set -euo pipefail

DRY_RUN=false
CONFIG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) CONFIG="$arg" ;;
    esac
done

CONFIG="${CONFIG:-experiments/stage1_volumetric/test_candidate_uncertainty_signals/configs/picasso.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

_BOOTSTRAP_ENV() {
    local env_name
    env_name=$(grep -E "^[[:space:]]*conda_env:" "${CONFIG}" 2>/dev/null \
        | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")
    env_name="${env_name:-growth}"
    if [[ -z "$(command -v conda 2>/dev/null)" ]]; then
        echo "WARNING: conda not on PATH; using current Python ($(command -v python))" >&2
        return
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${env_name}" 2>/dev/null && echo "Activated conda env: ${env_name}" \
        || echo "WARNING: failed to activate '${env_name}'; staying in current env" >&2
}
_BOOTSTRAP_ENV

PYTHON="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
if ! "$PYTHON" -c "import yaml" 2>/dev/null; then
    echo "ERROR: ${PYTHON} cannot import yaml. Activate the project env." >&2
    exit 1
fi

read_yaml() {
    "$PYTHON" -c "
import yaml
with open('${CONFIG}') as f: cfg = yaml.safe_load(f)
keys='$1'.split('.'); v=cfg
for k in keys:
    v = v.get(k, '$2') if isinstance(v, dict) else '$2'
print(v)
"
}

PARTITION=$(read_yaml slurm.repair_partition gputhin)
CONSTRAINT=$(read_yaml slurm.repair_constraint cpu)
TIME_LIMIT=$(read_yaml slurm.repair_time "0-01:00:00")
THROTTLE=$(read_yaml slurm.repair_array_throttle 32)
CPUS=$(read_yaml slurm.cpus_per_task 4)
MEM=$(read_yaml slurm.mem "16G")
CONDA_ENV=$(read_yaml slurm.conda_env growth)
SLURM_REPO_DIR=$(read_yaml slurm.repo_dir "/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model")
LOGS_DIR=$(read_yaml slurm.logs_dir "${SLURM_REPO_DIR}/logs/test_candidate_uncertainty")
MENGROWTH_H5=$(read_yaml paths.mengrowth_h5 "")
OUTPUT_DIR=$(read_yaml paths.output_dir "")
RECOMPUTED_CSV=$(read_yaml stage0.recomputed_csv "${OUTPUT_DIR}/recomputed_uncertainty.csv")
TASK_CSV_DIR="${OUTPUT_DIR}/recomputed_per_task"

# Determine n_scans by reading the H5 attr.
N_SCANS=$("$PYTHON" -c "import h5py; f=h5py.File('${MENGROWTH_H5}','r'); print(int(f.attrs.get('n_scans', f['images'].shape[0]))); f.close()")
LAST_INDEX=$((N_SCANS - 1))

echo "==========================================="
echo "Test-candidates Stage 0 (H5 repair) launcher"
echo "==========================================="
echo "Config:          ${CONFIG}"
echo "MenGrowth H5:    ${MENGROWTH_H5}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Recomputed CSV:  ${RECOMPUTED_CSV}"
echo "Per-task dir:    ${TASK_CSV_DIR}"
echo "Partition:       ${PARTITION}    Constraint: ${CONSTRAINT}    Time: ${TIME_LIMIT}"
echo "CPUs/Mem:        ${CPUS} / ${MEM}    Conda env: ${CONDA_ENV}"
echo "N scans:         ${N_SCANS} (array 0-${LAST_INDEX}%${THROTTLE})"
echo

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}" "${TASK_CSV_DIR}" "$(dirname "${RECOMPUTED_CSV}")"
fi

ARRAY_CMD="sbatch --parsable \
    --job-name=uq_diag_repair_array \
    --partition=${PARTITION} \
    --constraint=${CONSTRAINT} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --array=0-${LAST_INDEX}%${THROTTLE} \
    --output=${LOGS_DIR}/uq_repair_%A_%a.out \
    --error=${LOGS_DIR}/uq_repair_%A_%a.err \
    --export=ALL,CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR},TASK_CSV_DIR=${TASK_CSV_DIR} \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/repair_worker.sh"

echo "[1/2] Repair array job:"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${ARRAY_CMD}"
    REPAIR_ARRAY_ID="DRYRUN"
else
    OUT=$(eval "${ARRAY_CMD}" 2>&1)
    echo "  ${OUT}"
    REPAIR_ARRAY_ID=$(echo "$OUT" | grep -oP '[0-9]+' | head -1)
    echo "  -> repair array job ${REPAIR_ARRAY_ID}"
fi

PATCH_CMD="sbatch --parsable \
    --job-name=uq_diag_patch \
    --partition=${PARTITION} \
    --constraint=${CONSTRAINT} \
    --time=0-00:30:00 \
    --cpus-per-task=2 \
    --mem=8G \
    --dependency=afterok:${REPAIR_ARRAY_ID} \
    --output=${LOGS_DIR}/uq_patch_%j.out \
    --error=${LOGS_DIR}/uq_patch_%j.err \
    --export=ALL,CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR},TASK_CSV_DIR=${TASK_CSV_DIR},RECOMPUTED_CSV=${RECOMPUTED_CSV},MENGROWTH_H5=${MENGROWTH_H5} \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/repair_patch.sh"

echo "[2/2] Patch job (depends on array):"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${PATCH_CMD}"
    PATCH_JOB_ID="DRYRUN"
else
    OUT=$(eval "${PATCH_CMD}" 2>&1)
    echo "  ${OUT}"
    PATCH_JOB_ID=$(echo "$OUT" | grep -oP '[0-9]+' | head -1)
    echo "  -> patch job ${PATCH_JOB_ID}"
fi

echo
echo "Done. Pass --depend-on ${PATCH_JOB_ID} to launcher.sh so Stage 2 waits for the H5 patch."
