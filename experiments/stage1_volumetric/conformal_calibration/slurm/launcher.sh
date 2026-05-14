#!/bin/bash
# -------------------------------------------------------------------
# Submit the conformal calibration experiment as a SLURM array job
# plus a dependent analysis job.
#
# Usage:
#   bash launcher.sh [CONFIG_PATH] [--dry-run]
#
# NOTE (Picasso lua-plugin gotcha): every sbatch call in this script
# passes --constraint=cpu explicitly on the command line.  The Picasso
# lua plugin overrides #SBATCH directives and defaults to cpu; without
# the explicit flag, GPU jobs remain stuck PD and dependents fail.
# -------------------------------------------------------------------

set -euo pipefail

DRY_RUN=false
CONFIG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) CONFIG="$arg" ;;
    esac
done

CONFIG="${CONFIG:-experiments/stage1_volumetric/conformal_calibration/configs/picasso.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

# Activate the experiment's conda env (read from YAML via grep) so the
# inline read_yaml calls and the manifest build below see all project
# deps (PyYAML, GPy, h5py, ...).
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
    if conda activate "${env_name}" 2>/dev/null; then
        echo "Activated conda env: ${env_name}"
    else
        echo "WARNING: failed to activate '${env_name}'; staying in current env" >&2
    fi
}
_BOOTSTRAP_ENV

PYTHON="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

if ! "$PYTHON" -c "import yaml" 2>/dev/null; then
    echo "ERROR: ${PYTHON} cannot import yaml." >&2
    echo "  Set 'slurm.conda_env' in the config to an env that has PyYAML." >&2
    exit 1
fi

read_yaml() {
    "$PYTHON" -c "
import yaml, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
keys = '$1'.split('.')
v = cfg
for k in keys:
    v = v.get(k, '$2') if isinstance(v, dict) else '$2'
print(v)
"
}

PARTITION=$(read_yaml slurm.partition gputhin)
TIME_LIMIT=$(read_yaml slurm.time "0-02:00:00")
CPUS=$(read_yaml slurm.cpus_per_task 4)
MEM=$(read_yaml slurm.mem "16G")
CONDA_ENV=$(read_yaml slurm.conda_env growth)
CONSTRAINT=$(read_yaml slurm.constraint cpu)
SLURM_REPO_DIR=$(read_yaml slurm.repo_dir "/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model")
LOGS_DIR=$(read_yaml slurm.logs_dir "${SLURM_REPO_DIR}/logs/conformal_calibration")
THROTTLE=$(read_yaml slurm.array_throttle 20)

# Pre-build the manifest so we know how many array tasks to launch.
SLURM_REPO_DIR_PRE=$(read_yaml slurm.repo_dir "")
PRE_REPO_DIR="${SLURM_REPO_DIR_PRE:-$(pwd)}"
export PYTHONPATH="${PRE_REPO_DIR}/src:${PRE_REPO_DIR}:${PYTHONPATH:-}"

echo "[1/3] Building task manifest..."
"$PYTHON" -m experiments.stage1_volumetric.conformal_calibration.run \
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

echo "==========================================="
echo "Conformal Calibration — SLURM Launcher"
echo "==========================================="
echo "Config:      ${CONFIG}"
echo "Partition:   ${PARTITION}"
echo "Constraint:  ${CONSTRAINT}"
echo "Time:        ${TIME_LIMIT}"
echo "CPUs:        ${CPUS}"
echo "Memory:      ${MEM}"
echo "Conda env:   ${CONDA_ENV}"
echo "Repo dir:    ${SLURM_REPO_DIR}"
echo "Logs dir:    ${LOGS_DIR}"
echo "N tasks:     ${N_TASKS} (array 0-${LAST_INDEX}%${THROTTLE})"
echo

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

ARRAY_CMD="sbatch \
    --job-name=confcal_array \
    --partition=${PARTITION} \
    --constraint=${CONSTRAINT} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --array=0-${LAST_INDEX}%${THROTTLE} \
    --output=${LOGS_DIR}/confcal_%A_%a.out \
    --error=${LOGS_DIR}/confcal_%A_%a.err \
    --export=ALL,CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
    experiments/stage1_volumetric/conformal_calibration/slurm/worker.sh"

echo "[2/3] Array job:"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${ARRAY_CMD}"
    ARRAY_JOB_ID="DRYRUN"
else
    OUT=$(eval "${ARRAY_CMD}" 2>&1)
    echo "  ${OUT}"
    ARRAY_JOB_ID=$(echo "$OUT" | grep -oP '[0-9]+' | head -1)
    echo "  -> array job ${ARRAY_JOB_ID}"
fi

ANALYSIS_CMD="sbatch \
    --job-name=confcal_analysis \
    --partition=${PARTITION} \
    --constraint=${CONSTRAINT} \
    --time=0-01:00:00 \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --dependency=afterany:${ARRAY_JOB_ID} \
    --output=${LOGS_DIR}/confcal_analysis_%j.out \
    --error=${LOGS_DIR}/confcal_analysis_%j.err \
    --export=ALL,CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
    experiments/stage1_volumetric/conformal_calibration/slurm/analysis_worker.sh"

echo "[3/3] Analysis job (depends on array):"
if ${DRY_RUN}; then
    echo "[DRY-RUN] ${ANALYSIS_CMD}"
else
    OUT=$(eval "${ANALYSIS_CMD}" 2>&1)
    echo "  ${OUT}"
fi

echo
echo "Done."
