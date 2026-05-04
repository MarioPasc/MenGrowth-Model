#!/usr/bin/env bash
# =============================================================================
# WORKER: BraTS23_1 (brainles/brats23_meningioma_nvauto)
#
# Quirks:
#   - MLCube interface: infer --data_path=/mlcube_io0 --output_path=/mlcube_io2
#   - PWD: /mlcube_project (read-only in the SIF; --writable-tmpfs is unreliable
#     in Singularity 3.7.2 over root-owned squashfs subdirectories).
#     → Stage /mlcube_project to a writable host dir and bind it back.
# =============================================================================
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
set -euo pipefail

# Resolve common helpers via BENCHMARK_DIR (exported by launcher).
# Cannot rely on ${BASH_SOURCE[0]} — SLURM copies this script into
# /var/spool/slurmd/jobXXX/slurm_script so any relative path breaks.
: "${BENCHMARK_DIR:?BENCHMARK_DIR must be exported by benchmark_launcher.sh}"
COMMON_SH="${BENCHMARK_DIR}/slurm/_common.sh"
if [ ! -f "${COMMON_SH}" ]; then
    echo "ERROR: cannot find ${COMMON_SH}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${COMMON_SH}"

bm_header
bm_setup_env
bm_check_inputs
bm_setup_workdir

CONTAINER_PWD="/mlcube_project"
STAGE_DIR="${MODEL_DIR}/work/mlcube_project_writable"
bm_stage_writable "${CONTAINER_PWD}" "${STAGE_DIR}"

echo "=========================================="
echo "RUNNING INFERENCE: ${MODEL_ID} (nvauto, MLCube)"
echo "  Bind: ${STAGE_DIR}   → ${CONTAINER_PWD} (rw, staged)"
echo "  Bind: ${WORK_INPUT}  → /mlcube_io0      (ro)"
echo "  Bind: ${WORK_OUTPUT} → /mlcube_io2      (rw)"
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
    "${SIF_PATH}" \
    infer --data_path=/mlcube_io0 --output_path=/mlcube_io2
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
