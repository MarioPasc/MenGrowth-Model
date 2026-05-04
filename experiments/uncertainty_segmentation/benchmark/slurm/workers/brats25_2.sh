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
    "${EXTRA_BINDS[@]}" \
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
