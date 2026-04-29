#!/usr/bin/env bash
# =============================================================================
# LOCAL DOCKER TEST — verify container interfaces before Picasso submission
#
# Extracts a small subset from H5 and runs each container locally with Docker.
# Requires: docker, conda env with h5py+nibabel.
# GPU optional: auto-detects nvidia-container-toolkit; falls back to CPU.
#
# Usage:
#   bash experiments/uncertainty_segmentation/benchmark/test_containers_local.sh --models BraTS23_2
#   bash experiments/uncertainty_segmentation/benchmark/test_containers_local.sh --models BraTS25_1 --limit 1
#   bash experiments/uncertainty_segmentation/benchmark/test_containers_local.sh --cleanup
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HOME}/.conda/envs/growth/bin/python"

# Local paths
H5_FILE="/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/source/BraTS_MEN.h5"
TEST_DIR="/tmp/benchmark_test"
LIMIT=2

# Model registry: ID|DOCKER_IMAGE|YEAR|INTERFACE|PARAMS_FILE (mirrors launcher)
MODELS=(
    "BraTS25_1|brainles/brats25_men_qing:latest|2025|docker_only|no"
    "BraTS25_2|brainles/brats25_men_mmdp:latest|2025|docker_only|no"
    "BraTS23_1|brainles/brats23_meningioma_nvauto:latest|2023|mlcube|yes"
    "BraTS23_2|brainles/brats23_meningioma_blackbean:latest|2023|mlcube|yes"
    "BraTS23_3|brainles/brats23_meningioma_cnmc_pmi2023:latest|2023|mlcube|no"
)

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
SELECTED_MODELS=""
CLEANUP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) shift; SELECTED_MODELS="$*"; break ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --h5) H5_FILE="$2"; shift 2 ;;
        --cleanup)
            echo "Cleaning up test data and Docker images..."
            rm -rf "${TEST_DIR}"
            for entry in "${MODELS[@]}"; do
                IFS='|' read -r _ docker_image _ _ _ <<< "${entry}"
                docker rmi "${docker_image}" 2>/dev/null && echo "  Removed ${docker_image}" || true
            done
            docker image prune -f 2>/dev/null || true
            echo "Done."
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "LOCAL DOCKER CONTAINER TEST"
echo "=========================================="
echo "  H5:       ${H5_FILE}"
echo "  Test dir: ${TEST_DIR}"
echo "  Limit:    ${LIMIT} patients"

# Auto-detect GPU support
GPU_FLAG=""
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 true 2>/dev/null; then
    GPU_FLAG="--gpus all"
    echo "  GPU:      available"
else
    echo "  GPU:      not available (CPU mode)"
fi
echo ""

# ========================================================================
# STEP 1: EXTRACT TEST SUBSET
# ========================================================================
EXTRACTION_DIR="${TEST_DIR}/extraction"
MANIFEST="${EXTRACTION_DIR}/manifest.json"

if [ -f "${MANIFEST}" ]; then
    N_EXISTING=$(${PYTHON} -c "import json; print(json.load(open('${MANIFEST}'))['n_patients'])")
    echo "[OK] Already extracted ${N_EXISTING} patients (delete ${MANIFEST} to redo)"
else
    echo "[RUN] Extracting ${LIMIT} patients..."
    ${PYTHON} "${SCRIPT_DIR}/extract_h5_to_nifti.py" \
        --h5 "${H5_FILE}" \
        --output "${EXTRACTION_DIR}" \
        --limit "${LIMIT}"
fi

NIFTI_DIR="${EXTRACTION_DIR}/nifti"
echo "[OK] NIfTIs: $(find "${NIFTI_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l) patients"

SAMPLE=$(find "${NIFTI_DIR}" -mindepth 1 -maxdepth 1 -type d | head -1)
echo "[OK] Sample: $(basename "${SAMPLE}")"
ls "${SAMPLE}"
echo ""

# ========================================================================
# STEP 2: TEST EACH CONTAINER
# ========================================================================
RESULTS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id docker_image year interface params_file <<< "${entry}"

    if [ -n "${SELECTED_MODELS}" ]; then
        echo "${SELECTED_MODELS}" | grep -qw "${model_id}" || continue
    fi

    echo "=========================================="
    echo "TESTING: ${model_id} (${interface}, params=${params_file})"
    echo "=========================================="

    # Check if image exists, pull if not
    if docker image inspect "${docker_image}" >/dev/null 2>&1; then
        echo "[OK]   Image already pulled"
    else
        echo "[PULL] ${docker_image}"
        docker pull "${docker_image}" 2>&1 | tail -3
    fi

    MODEL_OUTPUT="${TEST_DIR}/models/${model_id}/output"
    mkdir -p "${MODEL_OUTPUT}"

    echo "[RUN]  Running inference..."
    set +e

    if [ "${interface}" = "docker_only" ]; then
        docker run --rm ${GPU_FLAG} \
            -v "${NIFTI_DIR}:/input:ro" \
            -v "${MODEL_OUTPUT}:/output:rw" \
            "${docker_image}" \
            2>&1 | tail -20

    elif [ "${interface}" = "mlcube" ]; then
        PARAMS_VOLUME=""
        PARAMS_ARG=""
        if [ "${params_file}" = "yes" ]; then
            PARAMS_DIR="${TEST_DIR}/models/${model_id}/params"
            mkdir -p "${PARAMS_DIR}"
            echo "{}" > "${PARAMS_DIR}/params.yaml"
            PARAMS_VOLUME="-v ${PARAMS_DIR}:/mlcube_io1:ro"
            PARAMS_ARG="--parameters_file=/mlcube_io1/params.yaml"
        fi

        docker run --rm ${GPU_FLAG} \
            -v "${NIFTI_DIR}:/mlcube_io0:ro" \
            -v "${MODEL_OUTPUT}:/mlcube_io2:rw" \
            ${PARAMS_VOLUME} \
            "${docker_image}" \
            infer --data_path=/mlcube_io0 --output_path=/mlcube_io2 ${PARAMS_ARG} \
            2>&1 | tail -20
    fi

    INFER_RC=$?
    set -e

    N_PREDS=$(find "${MODEL_OUTPUT}" -name "*.nii.gz" 2>/dev/null | wc -l)

    if [ "${INFER_RC}" -eq 0 ] && [ "${N_PREDS}" -gt 0 ]; then
        echo "[PASS] ${model_id}: exit=${INFER_RC}, ${N_PREDS} predictions"
        RESULTS+=("${model_id}: PASS (${N_PREDS} predictions)")
    elif [ "${INFER_RC}" -eq 0 ]; then
        echo "[WARN] ${model_id}: exit=0 but no .nii.gz found"
        find "${MODEL_OUTPUT}" -type f | head -10
        RESULTS+=("${model_id}: WARN (exit=0, no nii.gz)")
    else
        echo "[FAIL] ${model_id}: exit=${INFER_RC}"
        RESULTS+=("${model_id}: FAIL (exit=${INFER_RC})")
    fi
    echo ""
done

# ========================================================================
# SUMMARY
# ========================================================================
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
for r in "${RESULTS[@]}"; do
    echo "  ${r}"
done
echo ""
echo "Test data: ${TEST_DIR}"
echo "Cleanup:   bash $0 --cleanup"
