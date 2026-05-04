#!/usr/bin/env bash
# =============================================================================
# WORKER: BraTS25_2 (brainles/brats25_men_mmdp)
#
# Quirks:
#   - PWD: /workspace  (entrypoint is /workspace/main.py — NOT inference.py)
#   - The earlier auto-detection probe in benchmark_worker.sh searched for
#     inference.py and produced a corrupted CONTAINER_PWD. Pin /workspace.
#   - Reads /input, writes /output. --writable-tmpfs is sufficient.
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

CONTAINER_PWD="/workspace"

echo "=========================================="
echo "RUNNING INFERENCE: ${MODEL_ID} (mmdp)"
echo "  Bind: ${WORK_INPUT}  → /input  (ro)"
echo "  Bind: ${WORK_OUTPUT} → /output (rw)"
echo "  PWD:  ${CONTAINER_PWD}"
echo "=========================================="

INFER_START=$(date +%s)
set +e
singularity run \
    --nv \
    --cleanenv \
    --no-home \
    --writable-tmpfs \
    --pwd "${CONTAINER_PWD}" \
    --bind "${WORK_INPUT}:/input:ro" \
    --bind "${WORK_OUTPUT}:/output:rw" \
    "${SIF_PATH}"
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
