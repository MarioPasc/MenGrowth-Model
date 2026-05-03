#!/bin/bash
# experiments/stage1_volumetric/slurm/launcher.sh
# -------------------------------------------------
# Submit one SLURM job per enabled model, then an analysis job
# that runs after all models complete.
#
# Usage:
#   bash experiments/stage1_volumetric/slurm/launcher.sh [CONFIG_PATH]
#   bash experiments/stage1_volumetric/slurm/launcher.sh --dry-run
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
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
# Try the env from config first, fall back to growth
LAUNCHER_ENV="${CONDA_ENV_OVERRIDE:-growth}"
conda activate "${LAUNCHER_ENV}" 2>/dev/null || conda activate mengrowth 2>/dev/null || true

# --- Read SLURM settings from config ---
read_yaml() {
    python -c "
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
SLURM_REPO_DIR=$(read_yaml repo_dir "${REPO_DIR}")
LOGS_DIR=$(read_yaml logs_dir "${REPO_DIR}/logs")

# --- Get model list from config ---
MODELS=$(python -c "
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
echo "Config:    ${CONFIG}"
echo "Partition: ${PARTITION}"
echo "Time:      ${TIME_LIMIT}"
echo "CPUs:      ${CPUS}"
echo "Memory:    ${MEM}"
echo "Conda env: ${CONDA_ENV}"
echo "Logs dir:  ${LOGS_DIR}"
echo ""
echo "Models to submit:"
echo "${MODELS}" | while read -r m; do echo "  - ${m}"; done
echo ""

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

JOB_IDS=""
N_SUBMITTED=0

for MODEL in ${MODELS}; do
    SBATCH_CMD="sbatch --parsable \
        --job-name=growth_uq_${MODEL} \
        --partition=${PARTITION} \
        --time=${TIME_LIMIT} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM} \
        --output=${LOGS_DIR}/growth_uq_${MODEL}_%j.out \
        --error=${LOGS_DIR}/growth_uq_${MODEL}_%j.err \
        --export=ALL,MODEL_NAME=${MODEL},CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
        experiments/stage1_volumetric/slurm/worker.sh"

    if ${DRY_RUN}; then
        echo "[DRY-RUN] ${SBATCH_CMD}"
    else
        JOB_ID=$(eval "${SBATCH_CMD}")
        echo "Submitted ${MODEL} as job ${JOB_ID}"
        JOB_IDS="${JOB_IDS}:${JOB_ID}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
done

# --- Submit analysis job dependent on all model jobs ---
if ! ${DRY_RUN} && [[ -n "${JOB_IDS}" ]]; then
    ANALYSIS_JOB=$(sbatch --parsable \
        --job-name=growth_uq_analysis \
        --partition=${PARTITION} \
        --time=0-00:30:00 \
        --cpus-per-task=2 \
        --mem=8G \
        --output="${LOGS_DIR}/growth_uq_analysis_%j.out" \
        --error="${LOGS_DIR}/growth_uq_analysis_%j.err" \
        --dependency="afterok${JOB_IDS}" \
        --export=ALL,CONFIG_PATH=${CONFIG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
        --wrap="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${CONDA_ENV} && cd ${SLURM_REPO_DIR} && python -m experiments.stage1_volumetric.run_analysis --config ${CONFIG}")
    echo ""
    echo "Submitted analysis job ${ANALYSIS_JOB} (depends on all ${N_SUBMITTED} model jobs)"
fi

echo ""
echo "Done. ${N_SUBMITTED} model jobs + 1 analysis job submitted."
