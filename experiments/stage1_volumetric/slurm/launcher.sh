#!/bin/bash
# experiments/stage1_volumetric/slurm/launcher.sh
# -------------------------------------------------
# Submit one SLURM job per enabled model, then an analysis job
# that runs after all models complete.
#
# Usage:
#   bash experiments/stage1_volumetric/slurm/launcher.sh [CONFIG_PATH]
#   bash experiments/stage1_volumetric/slurm/launcher.sh --dry-run [CONFIG_PATH]
#   bash experiments/stage1_volumetric/slurm/launcher.sh experiments/stage1_volumetric/configs/picasso.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${1:-experiments/stage1_volumetric/configs/picasso.yaml}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    CONFIG="${2:-experiments/stage1_volumetric/configs/picasso.yaml}"
fi

cd "${REPO_DIR}"

# --- Activate conda for inline Python calls ---
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate growth 2>/dev/null || conda activate mengrowth 2>/dev/null || true

PYTHON="$(command -v python)"

# --- Read SLURM settings from config ---
read_yaml() {
    "$PYTHON" -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
slurm = cfg.get('slurm', {})
print(slurm.get('$1', '$2'))
"
}

PARTITION=$(read_yaml partition gputhin)
TIME_LIMIT=$(read_yaml time "0-01:00:00")
CPUS=$(read_yaml cpus_per_task 4)
MEM=$(read_yaml mem "16G")
CONDA_ENV=$(read_yaml conda_env mengrowth)
CONSTRAINT=$(read_yaml constraint "cpu")
SLURM_REPO_DIR=$(read_yaml repo_dir "${REPO_DIR}")
LOGS_DIR=$(read_yaml logs_dir "${REPO_DIR}/logs")

# --- Get model list from config ---
MODELS=$("$PYTHON" -c "
import sys
sys.path.insert(0, '.')
from experiments.stage1_volumetric.engine.data import load_config
from experiments.stage1_volumetric.engine.model_registry import build_model_configs
cfg = load_config('${CONFIG}')
for name in build_model_configs(cfg):
    print(name)
")

echo "========================================="
echo "Stage 1 UQ Growth Prediction — SLURM Launcher"
echo "========================================="
echo "Config:     ${CONFIG}"
echo "Partition:  ${PARTITION}"
echo "Constraint: ${CONSTRAINT}"
echo "Time:       ${TIME_LIMIT}"
echo "CPUs:       ${CPUS}"
echo "Memory:     ${MEM}"
echo "Conda env:  ${CONDA_ENV}"
echo "Repo dir:   ${SLURM_REPO_DIR}"
echo "Logs dir:   ${LOGS_DIR}"
echo ""
echo "Models to submit:"
echo "${MODELS}" | while read -r m; do echo "  - ${m}"; done
echo ""

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

# ---------------------------------------------------------------------------
# Submit model jobs, collecting job IDs for dependency chaining.
# Picasso wraps sbatch in a Lua script, so --parsable may not work.
# We extract the job ID from "Submitted batch job NNNNN" output.
# ---------------------------------------------------------------------------
JOB_IDS=()
N_SUBMITTED=0

for MODEL in ${MODELS}; do
    SBATCH_CMD="sbatch \
        --job-name=growth_uq_${MODEL} \
        --partition=${PARTITION} \
        --constraint=${CONSTRAINT} \
        --time=${TIME_LIMIT} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM} \
        --output=${LOGS_DIR}/growth_uq_${MODEL}_%j.out \
        --error=${LOGS_DIR}/growth_uq_${MODEL}_%j.err \
        --export=ALL,MODEL_NAME=${MODEL},CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
        experiments/stage1_volumetric/slurm/worker.sh"

    if ${DRY_RUN}; then
        echo "[DRY-RUN] ${SBATCH_CMD}"
        JOB_IDS+=("99999")
        N_SUBMITTED=$((N_SUBMITTED + 1))
    else
        SBATCH_OUTPUT=$(eval "${SBATCH_CMD}" 2>&1)
        echo "  ${SBATCH_OUTPUT}"

        # Extract numeric job ID from Picasso's lua-wrapped output
        JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP 'job\s+\K[0-9]+' | head -1)
        if [[ -z "$JOB_ID" ]]; then
            JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '[0-9]+' | head -1)
        fi

        if [[ -n "$JOB_ID" ]]; then
            echo "  -> ${MODEL} = job ${JOB_ID}"
            JOB_IDS+=("$JOB_ID")
            N_SUBMITTED=$((N_SUBMITTED + 1))
        else
            echo "  [WARN] Could not extract job ID for ${MODEL}"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Submit analysis job dependent on all model jobs (afterany = run even if
# some models fail, so partial results still get analysed).
# ---------------------------------------------------------------------------
if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    ALL_DEP=$(IFS=:; echo "${JOB_IDS[*]}")

    ANALYSIS_CMD="sbatch \
        --job-name=growth_uq_analysis \
        --partition=${PARTITION} \
        --constraint=${CONSTRAINT} \
        --time=0-00:30:00 \
        --cpus-per-task=2 \
        --mem=8G \
        --output=${LOGS_DIR}/growth_uq_analysis_%j.out \
        --error=${LOGS_DIR}/growth_uq_analysis_%j.err \
        --dependency=afterany:${ALL_DEP} \
        --export=ALL,CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
        experiments/stage1_volumetric/slurm/analysis_worker.sh"

    echo ""
    echo "--- Analysis job (depends on ${N_SUBMITTED} model jobs: ${ALL_DEP}) ---"

    if ${DRY_RUN}; then
        echo "[DRY-RUN] ${ANALYSIS_CMD}"
    else
        ANALYSIS_OUTPUT=$(eval "${ANALYSIS_CMD}" 2>&1)
        echo "  ${ANALYSIS_OUTPUT}"
        ANALYSIS_ID=$(echo "$ANALYSIS_OUTPUT" | grep -oP 'job\s+\K[0-9]+' | head -1)
        if [[ -n "$ANALYSIS_ID" ]]; then
            echo "  -> analysis = job ${ANALYSIS_ID}"
        fi
    fi
fi

echo ""
echo "Done. ${N_SUBMITTED} model jobs submitted."
