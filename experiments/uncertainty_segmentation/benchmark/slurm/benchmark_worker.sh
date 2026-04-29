#!/usr/bin/env bash
# =============================================================================
# BENCHMARK WORKER — Picasso compute node
#
# Runs a single BraTS meningioma segmentation model (Singularity container)
# on the full 150-patient test split. Handles both BraTS25 (Docker-only
# interface: /input, /output) and BraTS23 (MLCube interface: /mlcube_io0,
# /mlcube_io2).
#
# Expected environment variables (exported by benchmark_launcher.sh):
#   MODEL_ID        — e.g. BraTS25_1, BraTS23_2
#   SIF_PATH        — absolute path to .sif file
#   INTERFACE       — "docker_only" or "mlcube"
#   YEAR            — 2025 or 2023
#   PARAMS_FILE     — "yes" if mlcube container requires --parameters_file
#   OUTPUT_DIR      — root benchmark output directory
#   EXTRACTION_DIR  — path to extracted NIfTIs
#   CONDA_ENV_NAME  — conda environment name
#   REPO_ROOT       — repository root path
#   BENCHMARK_DIR   — benchmark module path
#
# Reference:
#   Existing pattern: MenGrowth/slurm/segmentation/meningioma_seg_worker.sh
# =============================================================================
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00

set -euo pipefail

START_TIME=$(date +%s)

echo "=============================================="
echo "BENCHMARK WORKER: ${MODEL_ID}"
echo "=============================================="
echo "  Job ID:     ${SLURM_JOB_ID:-local}"
echo "  Node:       $(hostname)"
echo "  Model:      ${MODEL_ID}"
echo "  SIF:        ${SIF_PATH}"
echo "  Interface:  ${INTERFACE}"
echo "  Year:       ${YEAR}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Extraction: ${EXTRACTION_DIR}"
echo ""

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
echo "=========================================="
echo "ENVIRONMENT"
echo "=========================================="

# Load singularity module
module load singularity 2>/dev/null || true
echo "[singularity] $(singularity --version 2>/dev/null || echo 'NOT FOUND')"

# Load conda
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail "$m" 2>&1 | grep -qi "${m}"; then
        module load "$m" && module_loaded=1 && break
    fi
done

if [ "$module_loaded" -eq 0 ]; then
    echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "[python] $(which python 2>/dev/null || echo 'NOT FOUND')"
python -c "import sys; print('Python', sys.version.split()[0])" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "[WARN] nvidia-smi not available"
echo ""

# ========================================================================
# VERIFY INPUTS
# ========================================================================
echo "=========================================="
echo "INPUT VERIFICATION"
echo "=========================================="

if [ ! -f "${SIF_PATH}" ]; then
    echo "ERROR: SIF file not found: ${SIF_PATH}"
    exit 1
fi
echo "[OK] SIF exists: $(du -h "${SIF_PATH}" | cut -f1)"

NIFTI_DIR="${EXTRACTION_DIR}/nifti"
if [ ! -d "${NIFTI_DIR}" ]; then
    echo "ERROR: Extracted NIfTIs not found: ${NIFTI_DIR}"
    exit 1
fi
N_PATIENTS=$(find "${NIFTI_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "[OK] Extracted NIfTIs: ${N_PATIENTS} patients in ${NIFTI_DIR}"

# Verify GPU is accessible via Singularity
set +e
singularity exec --nv "${SIF_PATH}" nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
GPU_CHECK=$?
set -e
if [ "${GPU_CHECK}" -ne 0 ]; then
    echo "[WARN] GPU not accessible inside container — inference may fail or be slow"
else
    echo "[OK] GPU accessible inside container"
fi
echo ""

# ========================================================================
# SETUP WORK DIRECTORY
# ========================================================================
echo "=========================================="
echo "WORK DIRECTORY SETUP"
echo "=========================================="

MODEL_DIR="${OUTPUT_DIR}/models/${MODEL_ID}"
WORK_OUTPUT="${MODEL_DIR}/work/output"
PRED_DIR="${MODEL_DIR}/predictions"

mkdir -p "${WORK_OUTPUT}" "${PRED_DIR}"

# Bind extraction/nifti directly as the container input — no symlinks.
# Singularity cannot follow symlinks whose targets are outside bind mounts.
echo "[OK] Input (bind-mounted):  ${NIFTI_DIR} (${N_PATIENTS} patients)"
echo "[OK] Output (work dir):     ${WORK_OUTPUT}"
echo ""

# ========================================================================
# RUN INFERENCE
# ========================================================================
echo "=========================================="
echo "RUNNING INFERENCE: ${MODEL_ID}"
echo "=========================================="

INFER_START=$(date +%s)

set +e
if [ "${INTERFACE}" = "docker_only" ]; then
    # BraTS25: simple /input → /output interface
    echo "Interface: Docker-only (BraTS25)"
    echo "  Bind: ${NIFTI_DIR} → /input (ro)"
    echo "  Bind: ${WORK_OUTPUT} → /output (rw)"
    echo ""

    singularity run \
        --nv \
        --cleanenv \
        --no-home \
        --writable-tmpfs \
        --bind "${NIFTI_DIR}:/input:ro" \
        --bind "${WORK_OUTPUT}:/output:rw" \
        "${SIF_PATH}"

elif [ "${INTERFACE}" = "mlcube" ]; then
    # BraTS23: MLCube /mlcube_io0 → /mlcube_io2 interface
    echo "Interface: MLCube (BraTS23)"
    echo "  Bind: ${NIFTI_DIR} → /mlcube_io0 (ro)"
    echo "  Bind: ${WORK_OUTPUT} → /mlcube_io2 (rw)"

    # Some MLCube containers require --parameters_file (dummy, unused).
    # Determined per-model from BraTS Orchestrator meningioma.yml config.
    PARAMS_BIND=""
    PARAMS_ARG=""
    if [ "${PARAMS_FILE:-no}" = "yes" ]; then
        PARAMS_DIR="${MODEL_DIR}/work/params"
        mkdir -p "${PARAMS_DIR}"
        echo "{}" > "${PARAMS_DIR}/params.yaml"
        PARAMS_BIND="--bind ${PARAMS_DIR}:/mlcube_io1:ro"
        PARAMS_ARG="--parameters_file=/mlcube_io1/params.yaml"
        echo "  Bind: ${PARAMS_DIR} → /mlcube_io1 (ro)  [dummy params]"
    fi
    echo ""

    singularity run \
        --nv \
        --cleanenv \
        --no-home \
        --writable-tmpfs \
        --bind "${NIFTI_DIR}:/mlcube_io0:ro" \
        --bind "${WORK_OUTPUT}:/mlcube_io2:rw" \
        ${PARAMS_BIND} \
        "${SIF_PATH}" \
        infer --data_path=/mlcube_io0 --output_path=/mlcube_io2 ${PARAMS_ARG}
else
    echo "ERROR: Unknown interface type: ${INTERFACE}"
    exit 1
fi

INFER_EXIT=$?
set -e

INFER_END=$(date +%s)
INFER_ELAPSED=$((INFER_END - INFER_START))

echo ""
echo "Inference exit code: ${INFER_EXIT}"
echo "Inference duration:  $(($INFER_ELAPSED / 60))m $(($INFER_ELAPSED % 60))s"

if [ "${INFER_EXIT}" -ne 0 ]; then
    echo ""
    echo "ERROR: Inference failed with exit code ${INFER_EXIT}"
    echo ""
    echo "Debugging tips:"
    echo "  1. Inspect work dir: ${MODEL_DIR}/work/"
    echo "  2. Interactive shell: singularity shell --nv ${SIF_PATH}"
    echo "  3. Runscript: singularity inspect --runscript ${SIF_PATH}"
    echo "  4. For BraTS23 requires_root, try: --fakeroot"
    echo ""
    echo "Work directory preserved for debugging."

    # Save partial metadata even on failure
    python3 -c "
import json
meta = {
    'model_id': '${MODEL_ID}',
    'sif_path': '${SIF_PATH}',
    'interface': '${INTERFACE}',
    'year': int('${YEAR}'),
    'exit_code': ${INFER_EXIT},
    'inference_seconds': ${INFER_ELAPSED},
    'status': 'FAILED',
    'n_patients_input': ${N_PATIENTS},
}
with open('${MODEL_DIR}/run_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
" 2>/dev/null || true

    exit "${INFER_EXIT}"
fi

# ========================================================================
# COLLECT PREDICTIONS
# ========================================================================
echo ""
echo "=========================================="
echo "COLLECTING PREDICTIONS"
echo "=========================================="

# Move predictions from work/output to predictions/
# Containers write either flat files or nested dirs — handle both
N_PREDS=0

for nii_file in $(find "${WORK_OUTPUT}" -name "*.nii.gz" -type f 2>/dev/null); do
    rel_path="${nii_file#${WORK_OUTPUT}/}"
    dest="${PRED_DIR}/${rel_path}"
    mkdir -p "$(dirname "${dest}")"
    cp "${nii_file}" "${dest}"
    N_PREDS=$((N_PREDS + 1))
done

echo "[OK] Collected ${N_PREDS} prediction files → ${PRED_DIR}"

if [ "${N_PREDS}" -eq 0 ]; then
    echo "[WARN] No predictions found! Listing work/output:"
    find "${WORK_OUTPUT}" -type f | head -20
fi

# ========================================================================
# SAVE METADATA
# ========================================================================
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

python3 -c "
import json
meta = {
    'model_id': '${MODEL_ID}',
    'sif_path': '${SIF_PATH}',
    'interface': '${INTERFACE}',
    'year': int('${YEAR}'),
    'exit_code': 0,
    'inference_seconds': ${INFER_ELAPSED},
    'total_seconds': ${TOTAL_ELAPSED},
    'status': 'SUCCESS',
    'n_patients_input': ${N_PATIENTS},
    'n_predictions': ${N_PREDS},
    'node': '$(hostname)',
    'job_id': '${SLURM_JOB_ID:-local}',
}
with open('${MODEL_DIR}/run_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
" 2>/dev/null || true

# ========================================================================
# CLEANUP
# ========================================================================
echo ""
echo "=========================================="
echo "CLEANUP"
echo "=========================================="

# Remove work/output (container output already collected to predictions/)
rm -rf "${MODEL_DIR}/work"
echo "[OK] Removed work directory"

echo ""
echo "=========================================="
echo "COMPLETE: ${MODEL_ID}"
echo "=========================================="
echo "  Predictions: ${PRED_DIR} (${N_PREDS} files)"
echo "  Metadata:    ${MODEL_DIR}/run_metadata.json"
echo "  Duration:    $(($TOTAL_ELAPSED / 60))m $(($TOTAL_ELAPSED % 60))s"
echo ""
echo "Next step: python3 ${BENCHMARK_DIR}/evaluate_benchmark.py --output-dir ${OUTPUT_DIR} --models ${MODEL_ID}"
