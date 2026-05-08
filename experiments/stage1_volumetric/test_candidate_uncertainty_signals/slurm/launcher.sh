#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the candidate-uncertainty-signal Stage 2 sweep:
#   * SLURM array (one task per (candidate, scaling) cell or control)
#   * dependent analysis job (Stage 1 + Stage 2 aggregation + figures)
#
# Usage:
#   bash launcher.sh [CONFIG_PATH] [--dry-run] [--depend-on JOB_ID]
#
# Pass --depend-on <REPAIR_PATCH_JOB_ID> (returned by repair_launcher.sh) to
# block the Stage 2 array until the H5 has been repaired.
# ---------------------------------------------------------------------------

set -euo pipefail

DRY_RUN=false
DEPEND_ON=""
CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --depend-on) DEPEND_ON="$2"; shift 2 ;;
        *) CONFIG="$1"; shift ;;
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

PARTITION=$(read_yaml slurm.partition gputhin)
CONSTRAINT=$(read_yaml slurm.constraint cpu)
TIME_LIMIT=$(read_yaml slurm.time "0-01:00:00")
CPUS=$(read_yaml slurm.cpus_per_task 4)
MEM=$(read_yaml slurm.mem "16G")
CONDA_ENV=$(read_yaml slurm.conda_env growth)
SLURM_REPO_DIR=$(read_yaml slurm.repo_dir "/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model")
LOGS_DIR=$(read_yaml slurm.logs_dir "${SLURM_REPO_DIR}/logs/test_candidate_uncertainty")
THROTTLE=$(read_yaml slurm.array_throttle 8)

PRE_REPO_DIR="${SLURM_REPO_DIR:-$(pwd)}"
export PYTHONPATH="${PRE_REPO_DIR}/src:${PRE_REPO_DIR}:${PYTHONPATH:-}"

echo "[1/3] Building Stage 2 task manifest..."
"$PYTHON" -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.run \
    --config "${CONFIG}" \
    --write-manifest

OUTPUT_DIR=$(read_yaml paths.output_dir "")
MANIFEST="${OUTPUT_DIR}/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST" >&2
    exit 1
fi
N_TASKS=$("$PYTHON" -c "import json; print(len(json.load(open('${MANIFEST}'))))")
LAST_INDEX=$((N_TASKS - 1))

DEP_FLAG=""
if [[ -n "${DEPEND_ON}" ]]; then
    DEP_FLAG="--dependency=afterok:${DEPEND_ON}"
fi

echo "==========================================="
echo "Test-candidates Stage 2 launcher"
echo "==========================================="
echo "Config:      ${CONFIG}"
echo "Partition:   ${PARTITION} / ${CONSTRAINT}    Time: ${TIME_LIMIT}"
echo "CPUs/Mem:    ${CPUS} / ${MEM}    Conda env: ${CONDA_ENV}"
echo "Repo dir:    ${SLURM_REPO_DIR}"
echo "Logs dir:    ${LOGS_DIR}"
echo "N tasks:     ${N_TASKS} (array 0-${LAST_INDEX}%${THROTTLE})"
[[ -n "${DEPEND_ON}" ]] && echo "Depends on:  ${DEPEND_ON}"
echo

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

ARRAY_CMD="sbatch \
    --job-name=uq_diag_stage2_array \
    --partition=${PARTITION} \
    --constraint=${CONSTRAINT} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --array=0-${LAST_INDEX}%${THROTTLE} \
    ${DEP_FLAG} \
    --output=${LOGS_DIR}/uq_stage2_%A_%a.out \
    --error=${LOGS_DIR}/uq_stage2_%A_%a.err \
    --export=ALL,CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/worker.sh"

echo "[2/3] Stage 2 array job:"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${ARRAY_CMD}"
    ARRAY_JOB_ID="DRYRUN"
else
    OUT=$(eval "${ARRAY_CMD}" 2>&1)
    echo "  ${OUT}"
    ARRAY_JOB_ID=$(echo "$OUT" | grep -oP 'Submitted batch job \K[0-9]+' | head -1)
    echo "  -> array job ${ARRAY_JOB_ID}"
fi

ANALYSIS_CMD="sbatch \
    --job-name=uq_diag_stage2_analysis \
    --partition=${PARTITION} \
    --constraint=${CONSTRAINT} \
    --time=0-02:00:00 \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --dependency=afterany:${ARRAY_JOB_ID} \
    --output=${LOGS_DIR}/uq_analysis_%j.out \
    --error=${LOGS_DIR}/uq_analysis_%j.err \
    --export=ALL,CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/analysis_worker.sh"

echo "[3/3] Analysis job (depends on array):"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${ANALYSIS_CMD}"
else
    OUT=$(eval "${ANALYSIS_CMD}" 2>&1)
    echo "  ${OUT}"
fi

echo
echo "Done."
