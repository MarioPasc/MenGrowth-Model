#!/usr/bin/env bash
# =============================================================================
# WORKER: BraTS23_2 (brainles/brats23_meningioma_blackbean)
#
# Quirks:
#   - MLCube interface, BUT mlcube.py:24 calls os.makedirs('./inputs', ...)
#     relative to PWD=/mlcube_project. The squashfs path is owned by root with
#     mode 755 → host UID cannot write → PermissionError. --writable-tmpfs is
#     not reliable in Singularity 3.7.2 over root-owned squashfs subdirs.
#     → Stage /mlcube_project to writable host dir and bind it back.
#   - infer command requires --parameters_file=/mlcube_io1/params.yaml.
#     Provide a dummy empty params.yaml.
# =============================================================================
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_common.sh"

bm_header
bm_setup_env
bm_check_inputs
bm_setup_workdir

CONTAINER_PWD="/mlcube_project"
STAGE_DIR="${MODEL_DIR}/work/mlcube_project_writable"
PARAMS_DIR="${MODEL_DIR}/work/params"

bm_stage_writable "${CONTAINER_PWD}" "${STAGE_DIR}"

# Provide a minimal params.yaml so mlcube.py:infer is happy.
mkdir -p "${PARAMS_DIR}"
cat > "${PARAMS_DIR}/params.yaml" <<'YAML'
# Minimal MLCube params placeholder for blackbean (BraTS23_2).
# The container writes its own internal config; an empty mapping is enough
# to satisfy the --parameters_file argument required by mlcube.py:infer.
{}
YAML

echo "=========================================="
echo "RUNNING INFERENCE: ${MODEL_ID} (blackbean, MLCube + params)"
echo "  Bind: ${STAGE_DIR}   → ${CONTAINER_PWD} (rw, staged)"
echo "  Bind: ${WORK_INPUT}  → /mlcube_io0 (ro)"
echo "  Bind: ${WORK_OUTPUT} → /mlcube_io2 (rw)"
echo "  Bind: ${PARAMS_DIR}  → /mlcube_io1 (ro, dummy params)"
echo "=========================================="

INFER_START=$(date +%s)
set +e
singularity run \
    --nv \
    --cleanenv \
    --no-home \
    --writable-tmpfs \
    --pwd "${CONTAINER_PWD}" \
    --bind "${STAGE_DIR}:${CONTAINER_PWD}:rw" \
    --bind "${WORK_INPUT}:/mlcube_io0:ro" \
    --bind "${WORK_OUTPUT}:/mlcube_io2:rw" \
    --bind "${PARAMS_DIR}:/mlcube_io1:ro" \
    "${SIF_PATH}" \
    infer --data_path=/mlcube_io0 --output_path=/mlcube_io2 --parameters_file=/mlcube_io1/params.yaml
INFER_EXIT=$?
set -e
INFER_END=$(date +%s)
INFER_ELAPSED=$((INFER_END - INFER_START))

echo ""
echo "Inference exit: ${INFER_EXIT}  duration: $((INFER_ELAPSED/60))m $((INFER_ELAPSED%60))s"

if [ "${INFER_EXIT}" -ne 0 ]; then
    bm_failure_diag "${INFER_EXIT}"
    bm_save_metadata "FAILED" "${INFER_EXIT}" "${INFER_ELAPSED}" "$(( $(date +%s) - START_TIME ))"
    exit "${INFER_EXIT}"
fi

bm_collect_predictions
TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
bm_save_metadata "SUCCESS" 0 "${INFER_ELAPSED}" "${TOTAL_ELAPSED}"

rm -rf "${MODEL_DIR}/work"
echo ""
echo "COMPLETE: ${MODEL_ID}  preds=${N_PREDS}  total=$((TOTAL_ELAPSED/60))m"
echo "Next: python3 ${BENCHMARK_DIR}/evaluate_benchmark.py --output-dir ${OUTPUT_DIR} --models ${MODEL_ID}"
